#!/usr/bin/env python
"""Count how often the adapted autoreject detector flags fragments over whole recordings.

A check that reads no labels. For each animal and each of its recordings (one LongRecordingOrganizer, one
animalday), the cutoffs are fit on the recording's pool of up to ``--max-fit-fragments`` fragments
(``adr.build_fit_epochs``). ``--fit-seed`` seeds the subsample of a longer recording and nothing else: the
cutoff search keeps the detector's default split seed, as ``extract_dev_features.py`` and
``score_detectors.py`` do, so the cutoffs come from the procedure being evaluated. Every full-length fragment
of the recording is then read again, a chunk of a few hundred at a time so memory stays bounded, scored with
the stored parameters, and flagged where its score exceeds the channel's cutoff, which is what
``adr.apply_fit`` does. Fragments are read the way ``build_fit_epochs`` reads them, and rescoring the pool
fragments must reproduce the fit's own scores, or the run stops. Flags are counted per configuration and
channel, split by light phase (``isday``, 6 to 18 h by the fragment's clock time, as WAR rows compute it) and
by calendar date.

Outputs:

- ``--out``: one row per (dataset, animal, animalday, config, channel) and group, with ``n_fragments``,
  ``n_flagged`` and ``rate``. Rows with ``by == "isday"`` carry ``isday`` and rows with ``by == "date"``
  carry ``date``; the other column is empty.
- ``<out stem>_thresholds.csv`` beside ``--out``: the fitted cutoff per (animalday, config, channel) and the
  number of fragments it was fit on.
- ``<out>.provenance.json`` beside ``--out``: ``score_detectors.provenance`` of the run. It holds the cohort
  manifest, the config files each dataset loads and this script by sha256, the git commit, the parsed
  arguments, the configuration names, and the selection: per dataset the active animals and the quality
  filter's drops with their reasons, then the requested ids, or for a draw the labeled animals left out, the
  candidates and the drawn ids.
- ``--evidence-out`` (optional parquet): one row per (animalday, fragment, channel) with each configuration's
  score in a column named after it. A fragment is flagged where its score exceeds the cutoff in the
  thresholds file. The CSVs hold floats exactly; pandas reads them back exactly with
  ``float_precision="round_trip"``.

Animals come from ``--animal``, or from ``--draw N --seed S``, which picks N animals per dataset reproducibly
from the animals the pipeline analyzes and leaves out the labeled animals listed in
``config/labeling/cohort_manifest.json``. The animals the pipeline analyzes are the active ones (entries not
marked ``exclude``) that its quality filter, ``workflow/rules/war_quality_filter.smk``, keeps: it drops an
animal whose genotype is Unknown while ``samples.quality_filter.exclude_unknown_genotypes`` is on, and one
listed in ``bad_folder_animalday`` while ``exclude_bad_animaldays`` is on. The selection is printed with each
animal's genotype, and the provenance records it. Naming an animal with ``--animal`` that the quality filter
drops logs a warning. The labeled animals' windows belong to the evaluation, so naming one with ``--animal``
is refused unless ``--allow-labeled`` is given. Any failure stops the run before anything is written, and so
does a run with nothing to measure (the draw is empty in every dataset, or no selected animal has a
recording), so empty outputs never replace earlier ones. To run animals as separate jobs, pass the printed
draw to ``--animal`` one animal at a time.

``--temp-dir`` sets the temporary directory after each dataset is prepared, which sets the dataset config's
``temp_directory``, and before its recordings load, so a run can reuse the intermediates of a chosen
directory.

Run from the repository root, since dataset paths resolve relative to the working directory. Whole recordings
are heavy, so run on the cluster::

    uv run python scripts/labeling/rejection_rates.py --dataset arx_parv sox5_bin --draw 3 --seed 0 \\
        --out results/labeling/rejection_rates.csv --evidence-out results/labeling/rejection_scores.parquet \\
        --workers 16
"""

import argparse
import contextlib
import dataclasses
import json
import logging
import random
import re
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cohort_bundle as C  # noqa: E402  (the cohort load seam)
import render_context as R  # noqa: E402  (FRAG_S, the labeling window length)
from score_detectors import provenance  # noqa: E402  (one provenance format across the labeling outputs)

sys.path.insert(0, str(REPO / "src"))
from neurodent import constants  # noqa: E402
from neurodent.core.utils import is_day, resolve_channels  # noqa: E402
from neurodent.core.utils.paths import set_temp_directory  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.workflow.utils import enumerate_cohort, load_dataset_config  # noqa: E402

log = logging.getLogger("rejection_rates")

MANIFEST = REPO / "config/labeling/cohort_manifest.json"
# The files load_dataset_config merges, at its default paths, which resolve against the working directory.
BASE_CONFIG = Path("config/config.yaml")
LOCAL_CONFIG = Path("config/config.local.yaml")
DATASETS_DIR = Path("config/datasets")
CHUNK_FRAGMENTS = 300
RATE_COLUMNS = [
    "dataset", "animal", "genotype", "animalday", "config", "channel", "by", "isday", "date",
    "n_fragments", "n_flagged", "rate",
]
THRESHOLD_COLUMNS = [
    "dataset", "animal", "genotype", "animalday", "config", "channel", "threshold", "n_fit_fragments",
]
# A spectral grid name, as adr.grid_config_names spells it: {representation}/{config}@nf{n}_na{m}.
_SPECTRAL_NAME = re.compile(r"(?P<rep>[^/@]+)/(?P<base>[^@]+)@nf(?P<nf>\d+)_na(?P<na>\d+)")


# Animals


def labeled_animals(manifest=MANIFEST) -> set:
    """Return the ids of the animals rendered for labeling.

    Ids are compared as strings across datasets, so an animal in another dataset that shares a labeled
    animal's id counts as labeled.

    Args:
        manifest: Cohort manifest JSON with a ``rendered_animals`` list.

    Returns:
        A set of animal ids.
    """
    return {str(a) for a in json.loads(Path(manifest).read_text())["rendered_animals"]}


def genotype_of(animal_id) -> str:
    """Return an animal's genotype from the installed ``ANIMAL_METADATA``, as the cohort loader reads it."""
    return constants.ANIMAL_METADATA.get(animal_id, {}).get("genotype", "Unknown")


def quality_filter_reasons(animal_ids, genotypes, config, samples_config) -> dict:
    """Return why the pipeline's quality filter drops each animal it drops.

    Mirrors ``workflow/rules/war_quality_filter.smk``. With ``samples.quality_filter.exclude_bad_animaldays``
    on, an animal whose id is listed in ``bad_folder_animalday`` is dropped. With ``exclude_unknown_genotypes``
    on, an animal whose genotype is exactly ``"Unknown"`` is dropped. No later pipeline step analyzes a dropped
    animal.

    Args:
        animal_ids: Animal ids as the config spells them.
        genotypes: ``{animal id: genotype}`` covering ``animal_ids``, as the loader resolves them
            (:func:`genotype_of` after ``C._prepare``); the quality filter reads the same value from the WAR.
        config: The dataset's merged config.
        samples_config: The dataset's samples config.

    Returns:
        ``{animal id: reason}`` for the dropped animals, in the order of ``animal_ids``.
    """
    settings = config["samples"]["quality_filter"]
    bad = samples_config.get("bad_folder_animalday") or []
    reasons = {}
    for a in animal_ids:
        if settings["exclude_bad_animaldays"] and a in bad:
            reasons[a] = "listed in bad_folder_animalday, and samples.quality_filter.exclude_bad_animaldays is on"
        elif settings["exclude_unknown_genotypes"] and genotypes[a] == "Unknown":
            reasons[a] = "genotype is Unknown, and samples.quality_filter.exclude_unknown_genotypes is on"
    return reasons


def prepare_cohort(dataset):
    """Prepare a dataset and return its active animals, their genotypes and the quality filter's drops.

    Runs ``C._prepare``, which installs the dataset's ``ANIMAL_METADATA`` and genotype map, so each genotype
    is the one the loader resolves.

    Args:
        dataset: Dataset name, as in ``config/datasets/<name>.yaml``.

    Returns:
        ``(cohort, genotypes, dropped)``: the active animal ids (entries not marked ``exclude``) in config
        order, ``{animal id: genotype}``, and :func:`quality_filter_reasons` for the cohort.
    """
    config, samples_config = C._prepare(dataset)
    cohort = enumerate_cohort(samples_config)
    genotypes = {a: genotype_of(a) for a in cohort}
    return cohort, genotypes, quality_filter_reasons(cohort, genotypes, config, samples_config)


def draw_candidates(cohort, labeled) -> list:
    """Return the animals a draw picks from: ``cohort`` without the labeled ids, sorted by id.

    Args:
        cohort: The dataset's animal ids to draw from: the active animals the quality filter keeps.
        labeled: Ids to leave out, compared as strings.

    Returns:
        A list of ids.
    """
    return sorted((a for a in cohort if str(a) not in labeled), key=str)


def draw_animals(cohort, labeled, n, seed) -> list:
    """Pick ``n`` animals of one dataset, leaving out labeled ones, reproducibly.

    Candidates (:func:`draw_candidates`) are sorted by id before the seeded draw, so the draw does not depend
    on the order of the config entries.

    Args:
        cohort: The dataset's animal ids to draw from: the active animals the quality filter keeps.
        labeled: Ids to leave out, compared as strings.
        n: Number of animals to draw.
        seed: Seed of the draw.

    Returns:
        The drawn ids sorted by id; every candidate when ``n`` is not smaller than their number.
    """
    candidates = draw_candidates(cohort, labeled)
    if n >= len(candidates):
        return candidates
    return sorted(random.Random(seed).sample(candidates, n), key=str)


def resolve_animal(dataset, cohort, requested):
    """Return the config's id for an animal named on the command line, matched as a string.

    Args:
        dataset: Dataset name, for the error message.
        cohort: The dataset's active animal ids.
        requested: The id as given on the command line.

    Returns:
        The id in ``cohort`` that equals ``requested`` as a string.

    Raises:
        ValueError: If ``cohort`` has no such id.
    """
    by_str = {str(a): a for a in cohort}
    if str(requested) not in by_str:
        raise ValueError(f"{dataset} has no active animal {requested!r}")
    return by_str[str(requested)]


def _drop_records(dropped) -> list:
    """Return the quality filter's drops as ``[{"animal": id, "reason": reason}]``, which keeps each id's type."""
    return [{"animal": a, "reason": reason} for a, reason in dropped.items()]


def select_animals(datasets, *, animals=None, draw=None, seed=0, labeled=frozenset(), allow_labeled=False):
    """Return the animals to measure, from explicit ids or from a draw, and a record of how they were chosen.

    Each dataset is prepared with :func:`prepare_cohort`, so its globals are installed. A draw picks from the
    active animals the pipeline's quality filter keeps, leaving out ``labeled``; a dataset named twice is
    drawn once. Explicit ids are taken as given, and each one the quality filter drops is logged as a warning.

    Args:
        datasets: Dataset names.
        animals: Explicit ids; needs exactly one dataset.
        draw: Number of animals to draw per dataset when ``animals`` is empty.
        seed: Seed of the draw.
        labeled: Ids of the labeled animals, compared as strings; a draw leaves them out.
        allow_labeled: Accept labeled ids in ``animals``.

    Returns:
        ``(plan, record)``. ``plan`` is ``[(dataset, [(animal id, genotype)])]`` in the order of
        ``datasets``; explicit ids keep the order given, without repeats, and drawn ids are sorted by id.
        ``record`` maps each dataset to what the provenance keeps of the choice: ``active`` (the active
        animals in config order) and ``quality_filter_dropped`` (:func:`quality_filter_reasons` as a list of
        ``{"animal", "reason"}``), then ``requested`` and ``labeled`` (the requested ids that are labeled)
        for explicit ids, or ``labeled_left_out``, ``candidates`` and ``drawn`` for a draw.

    Raises:
        ValueError: On ids with several datasets, an id that is not an active animal, a labeled id without
            ``allow_labeled``, or neither ids nor a positive ``draw``.
    """
    if animals:
        if len(datasets) != 1:
            raise ValueError("--animal needs exactly one --dataset")
        ds = datasets[0]
        cohort, genotypes, dropped = prepare_cohort(ds)
        ids = list(dict.fromkeys(resolve_animal(ds, cohort, a) for a in animals))
        hit = [a for a in ids if str(a) in labeled]
        if hit and not allow_labeled:
            raise ValueError(f"labeled animal(s) {[str(a) for a in hit]}: their windows belong to the evaluation; "
                             "pass --allow-labeled to measure them anyway")
        for a in ids:
            if a in dropped:
                log.warning(f"{ds}: the pipeline's quality filter drops animal {a!r} ({dropped[a]}); measuring it "
                            "as requested")
        record = {ds: {"active": cohort, "quality_filter_dropped": _drop_records(dropped), "requested": ids,
                       "labeled": hit}}
        return [(ds, [(a, genotypes[a]) for a in ids])], record
    if draw is None or draw < 1:
        raise ValueError("give explicit animals or a positive number to draw")
    plan, record = [], {}
    for ds in dict.fromkeys(datasets):
        cohort, genotypes, dropped = prepare_cohort(ds)
        kept = [a for a in cohort if a not in dropped]
        candidates = draw_candidates(kept, labeled)
        ids = draw_animals(kept, labeled, draw, seed)
        log.info(f"{ds}: {len(cohort)} active animals, {len(dropped)} dropped by the pipeline's quality filter, "
                 f"{len(kept) - len(candidates)} of the rest labeled; drew {len(ids)} of {len(candidates)}")
        if len(ids) < draw:
            log.warning(f"{ds}: {len(ids)} animals to draw from, fewer than the {draw} requested; taking all")
        plan.append((ds, [(a, genotypes[a]) for a in ids]))
        record[ds] = {"active": cohort, "quality_filter_dropped": _drop_records(dropped),
                      "labeled_left_out": [a for a in kept if str(a) in labeled], "candidates": candidates,
                      "drawn": ids}
    return plan, record


def plan_lines(plan, *, draw=None, seed=0) -> list:
    """Return the printed selection: a heading, then one line per animal with its genotype.

    Ids are quoted for a shell where needed, so a line's id can be passed to ``--animal`` as printed.

    Args:
        plan: ``[(dataset, [(animal id, genotype)])]`` from :func:`select_animals`.
        draw: Animals drawn per dataset, or None for explicit ids.
        seed: Seed of the draw.

    Returns:
        A list of lines.
    """
    if draw is None:
        lines = ["animals:"]
    else:
        lines = [f"draw of {draw} per dataset, seed {seed}, from the active animals the pipeline's quality filter "
                 "keeps, labeled animals left out:"]
    for ds, animals in plan:
        if not animals:
            lines.append(f"  {ds}: none")
        lines += [f"  {ds}: {shlex.quote(str(a))} (genotype {g})" for a, g in animals]
    return lines


# Configurations and fitting


def resolve_config_names(names=None) -> list:
    """Return the requested configuration names in order without repeats, checked against the grid.

    Args:
        names: Names from ``adr.grid_config_names()``, or None for the whole grid.

    Returns:
        A list of names.

    Raises:
        ValueError: If a name is not in the grid.
    """
    grid = adr.grid_config_names()
    if names is None:
        return list(grid)
    unknown = [n for n in names if n not in grid]
    if unknown:
        raise ValueError(f"unknown configuration(s) {unknown}; choose from adr.grid_config_names()")
    return list(dict.fromkeys(names))


def parse_config_name(name):
    """Split a grid configuration name into its base configuration and spectral settings.

    Args:
        name: A name from ``adr.grid_config_names()``.

    Returns:
        ``(base, spectral)``: ``base`` names an entry of ``adr.CONFIG_NAMES``; ``spectral`` holds
        ``representation``, ``n_frequencies`` and ``n_averages`` for a spectral name, or is None for a
        peak-to-peak one.
    """
    m = _SPECTRAL_NAME.fullmatch(name)
    if m is None:
        return name, None
    return m["base"], {"representation": m["rep"], "n_frequencies": int(m["nf"]), "n_averages": int(m["na"])}


def fit_requested(X, fs, names, *, parallel=False) -> dict:
    """Fit the named grid configurations on one pool.

    Names that share spectral settings are fit by one ``adr.fit_configs`` call, which computes their spectra
    once, and each fit is renamed to its grid name. ``adr.fit_grid`` does the same per setting, so a
    configuration gets the same cutoffs, baselines and scores either way. The cutoff search keeps the
    detector's default split seed, as in ``extract_dev_features.py`` and ``score_detectors.py``, so nothing
    here can fit with other splits than the detector being evaluated.

    Args:
        X: ``(F, C, T)`` fragments in volts.
        fs: Sampling rate in Hz.
        names: Names from ``adr.grid_config_names()``.
        parallel: Fan spectral work out over dask.

    Returns:
        ``{name: DetectorFit}`` in the order of ``names``.
    """
    names = resolve_config_names(names)
    X = np.asarray(X, dtype=float)  # converted once rather than in every fit_configs call
    groups = {}
    for name in names:
        base, spectral = parse_config_name(name)
        key = None if spectral is None else tuple(sorted(spectral.items()))
        groups.setdefault(key, []).append((name, base))
    fits = {}
    for key, members in groups.items():
        settings = {} if key is None else dict(key)
        got = adr.fit_configs(X, fs, configs=[base for _, base in members], parallel=parallel, **settings)
        for name, base in members:
            fits[name] = dataclasses.replace(got[base], config=name)
    return {name: fits[name] for name in names}


# Scanning a recording


def fragment_chunks(lro, *, fragment_len_s, chunk_fragments=CHUNK_FRAGMENTS, apply_notch_filter=True):
    """Open a recording for reading its full-length fragments a chunk at a time.

    Fragments come from ``LongRecordingAnalyzer`` with the per-fragment notch, converted from uV to volts,
    exactly as ``adr.build_fit_epochs`` reads them. A fragment shorter than the others, the ragged final one,
    is skipped.

    Args:
        lro: A LongRecordingOrganizer.
        fragment_len_s: Fragment length in seconds.
        chunk_fragments: Fragments per chunk.
        apply_notch_filter: Notch each fragment at the line frequency.

    Returns:
        ``(channels, fs, chunks)``: canonical channel names, the integer sampling rate, and a generator of
        ``(indices (n,), X (n, C, T) volts)`` in recording order.
    """
    # Lazy import, as in build_fit_epochs: neurodent.analysis imports neurodent.results.
    from neurodent.analysis.long_recording_analyzer import LongRecordingAnalyzer

    if chunk_fragments < 1:
        raise ValueError(f"chunk_fragments must be at least 1, got {chunk_fragments}")
    lan = LongRecordingAnalyzer(lro, fragment_len_s=fragment_len_s, apply_notch_filter=apply_notch_filter)
    win = int(round(fragment_len_s * lan.f_s))

    def chunks():
        idx, frags = [], []
        for i in range(lan.n_fragments):
            f = lan.get_fragment_np(i)  # (n_samples, C) in uV
            if f.shape[0] != win:
                continue
            idx.append(i)
            frags.append(f.T * 1e-6)
            if len(frags) == chunk_fragments:
                yield np.asarray(idx, dtype=int), np.stack(frags, axis=0)
                idx, frags = [], []
        if frags:
            yield np.asarray(idx, dtype=int), np.stack(frags, axis=0)

    return resolve_channels(list(lan.channel_names)), int(lan.f_s), chunks()


def score_chunks(chunks, fits, *, parallel=False):
    """Score fragments chunk by chunk with each fit's stored parameters.

    Args:
        chunks: Iterable of ``(indices (n,), X (n, C, T) volts)``.
        fits: ``{config: DetectorFit}``.
        parallel: Fan spectral work out over dask.

    Returns:
        ``(indices (N,), {config: (N, C) scores})`` over all chunks in order. ``scores > fit.thresholds`` are
        the flags ``adr.apply_fit`` gives.

    Raises:
        ValueError: If there are no fragments.
    """
    idx_parts = []
    parts = {name: [] for name in fits}
    for idx, X in chunks:
        idx_parts.append(np.asarray(idx, dtype=int))
        X = np.asarray(X, dtype=float)  # converted once rather than in every score_fragments call
        for name, fit in fits.items():
            parts[name].append(adr.score_fragments(X, fit, parallel=parallel))
    if not idx_parts:
        raise ValueError("no full-length fragments to score")
    return np.concatenate(idx_parts), {name: np.concatenate(p, axis=0) for name, p in parts.items()}


def check_pool_scores(indices, scores, fits, pool_indices):
    """Raise unless the scan's scores on the pool fragments equal each fit's own scores.

    The scan reads the recording a second time, so equal scores show it read the samples the cutoffs were
    fit on.

    Args:
        indices: ``(N,)`` sorted fragment indices of the scan.
        scores: ``{config: (N, C)}`` scan scores.
        fits: ``{config: DetectorFit}``.
        pool_indices: ``(n_pool,)`` fragment indices of the pool, in the order of each fit's scores.

    Raises:
        RuntimeError: If a pool fragment is missing from the scan or its scores differ.
    """
    indices = np.asarray(indices, dtype=int)
    pool_indices = np.asarray(pool_indices, dtype=int)
    pos = np.searchsorted(indices, pool_indices)
    found = pos < indices.size
    found[found] = indices[pos[found]] == pool_indices[found]
    if not found.all():
        raise RuntimeError(f"{int((~found).sum())} fit-pool fragment(s) missing from the scan, first "
                           f"{int(pool_indices[~found][0])}")
    for name, fit in fits.items():
        if not np.allclose(scores[name][pos], fit.scores, rtol=1e-9, atol=0.0):
            raise RuntimeError(f"{name}: rescored pool fragments differ from the scores the cutoffs were fit on")


def fragment_times(lro, indices, fragment_len_s) -> list:
    """Return each fragment's start as clock time, through the recording's file timestamps.

    Uses ``lro.get_datetime_fragment``, as WAR rows do, so gaps between files are accounted for.

    Args:
        lro: A LongRecordingOrganizer.
        indices: Fragment indices.
        fragment_len_s: Fragment length in seconds.

    Returns:
        One datetime per index, in the order of ``indices``.
    """
    return [lro.get_datetime_fragment(fragment_len_s, int(i)) for i in indices]


@dataclasses.dataclass
class RecordingScan:
    """One recording's fits and the scores of all its full-length fragments.

    Attributes:
        channels: Canonical channel names.
        fits: ``{config: DetectorFit}``.
        n_fit_fragments: Fragments in the pool the cutoffs were fit on.
        indices: ``(N,)`` fragment indices.
        scores: ``{config: (N, C)}`` scores.
        timestamps: ``N`` fragment start times.
    """

    channels: list
    fits: dict
    n_fit_fragments: int
    indices: np.ndarray
    scores: dict
    timestamps: list

    def flags(self) -> dict:
        """``{config: (N, C) bool}``, True meaning flagged."""
        return {name: self.scores[name] > fit.thresholds for name, fit in self.fits.items()}


def scan_recording(lro, configs, *, fragment_len_s, max_fit_fragments, fit_seed=0,
                   chunk_fragments=CHUNK_FRAGMENTS, parallel=False) -> RecordingScan:
    """Fit the configurations on a recording's pool, then score every full-length fragment.

    Args:
        lro: A LongRecordingOrganizer.
        configs: Names from ``adr.grid_config_names()``.
        fragment_len_s: Fragment length in seconds.
        max_fit_fragments: Pool cap; None uses the whole recording.
        fit_seed: Seed of the pool subsample, passed to ``adr.build_fit_epochs`` and nothing else; the cutoff
            search keeps the detector's default split seed (:func:`fit_requested`).
        chunk_fragments: Fragments scored per chunk.
        parallel: Fan spectral work out over dask.

    Returns:
        A :class:`RecordingScan`.

    Raises:
        RuntimeError: If the scan reads other channels or samples than the pool.
    """
    pool = adr.build_fit_epochs(lro, None, fragment_len_s=fragment_len_s, max_fit_fragments=max_fit_fragments,
                                seed=fit_seed)
    fits = fit_requested(pool.X, pool.fs, configs, parallel=parallel)
    pool_indices = np.rint(pool.grid_times / fragment_len_s).astype(int)
    pool_channels, pool_fs, n_pool = list(pool.channels), pool.fs, int(pool.X.shape[0])
    del pool  # release the pool's samples before the scan

    channels, fs, chunks = fragment_chunks(lro, fragment_len_s=fragment_len_s, chunk_fragments=chunk_fragments)
    if channels != pool_channels or fs != pool_fs:
        raise RuntimeError(f"scan reads channels {channels} at {fs} Hz, pool has {pool_channels} at {pool_fs} Hz")
    indices, scores = score_chunks(chunks, fits, parallel=parallel)
    check_pool_scores(indices, scores, fits, pool_indices)
    return RecordingScan(channels, fits, n_pool, indices, scores, fragment_times(lro, indices, fragment_len_s))


# Aggregation and outputs


def rate_rows(flags, channels, timestamps, keys) -> list:
    """Count flags per configuration and channel, split by light phase and by calendar date.

    Args:
        flags: ``{config: (N, C) bool}``, True meaning flagged.
        channels: ``C`` channel names.
        timestamps: ``N`` fragment start times (datetime).
        keys: Columns copied into every row (dataset, animal, genotype, animalday).

    Returns:
        Rows with ``by == "isday"`` (``isday`` set, ``date`` None) and rows with ``by == "date"`` (``date``
        set as ISO text, ``isday`` None); a group without fragments has no row.
    """
    isday = np.array([is_day(t) for t in timestamps], dtype=bool)
    dates = np.array([t.date().isoformat() for t in timestamps], dtype=object)
    groups = [("isday", v, isday == v) for v in (True, False)]
    groups += [("date", d, dates == d) for d in sorted(set(dates))]
    rows = []
    for name, m in flags.items():
        m = np.asarray(m, dtype=bool)
        if m.shape != (len(timestamps), len(channels)):
            raise ValueError(f"{name}: flags are {m.shape}, expected {(len(timestamps), len(channels))}")
        for by, value, sel in groups:
            n = int(sel.sum())
            if n == 0:
                continue
            n_flagged = m[sel].sum(axis=0)
            for c, ch in enumerate(channels):
                rows.append({**keys, "config": name, "channel": ch, "by": by,
                             "isday": bool(value) if by == "isday" else None,
                             "date": value if by == "date" else None,
                             "n_fragments": n, "n_flagged": int(n_flagged[c]), "rate": float(n_flagged[c]) / n})
    return rows


def threshold_rows(fits, channels, keys, n_fit_fragments) -> list:
    """Return one row per configuration and channel with the fitted cutoff."""
    return [{**keys, "config": name, "channel": ch, "threshold": float(fit.thresholds[c]),
             "n_fit_fragments": int(n_fit_fragments)}
            for name, fit in fits.items() for c, ch in enumerate(channels)]


def evidence_frame(scan, keys, fragment_len_s) -> pd.DataFrame:
    """Return one row per (fragment, channel) of a scan with each configuration's score as a column."""
    n, n_ch = len(scan.indices), len(scan.channels)
    df = pd.DataFrame({
        "dataset": str(keys["dataset"]),
        "animal": str(keys["animal"]),
        "animalday": str(keys["animalday"]),
        "fragment": np.repeat(scan.indices, n_ch),
        "t_start_s": np.repeat(scan.indices * float(fragment_len_s), n_ch),
        "timestamp": np.repeat(pd.to_datetime(scan.timestamps).to_numpy(), n_ch),
        "isday": np.repeat(np.array([is_day(t) for t in scan.timestamps], dtype=bool), n_ch),
        "channel": np.tile(np.asarray(scan.channels, dtype=object), n),
    })
    for name in scan.fits:
        df[name] = scan.scores[name].reshape(-1)  # (N, C) row-major: fragment-major, channel-minor
    return df


class EvidenceWriter:
    """Append per-fragment score tables to one parquet file that appears only when the run succeeds.

    Used as a context manager. Tables go to a temporary file beside the target, renamed onto it on a clean
    exit and removed on an error. Without a table nothing is written, and the target is left as it was.

    Args:
        path: The parquet file to write, or None to make every call do nothing.

    Attributes:
        written: True once the file has been published at ``path``.
    """

    def __init__(self, path):
        self.path = None if path is None else Path(path)
        self._tmp = None if path is None else self.path.with_name(self.path.name + ".tmp")
        self._writer = None
        self.written = False

    def __enter__(self):
        return self

    def write(self, df):
        """Append one table; its columns must match the first table's."""
        if self.path is None:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self._tmp, table.schema)
        else:
            table = table.cast(self._writer.schema)
        self._writer.write_table(table)

    def __exit__(self, exc_type, exc, tb):
        if self.path is None:
            return False
        if self._writer is not None:
            self._writer.close()
        if exc_type is None and self._writer is not None:
            self._tmp.replace(self.path)
            self.written = True
        elif self._tmp.exists():
            self._tmp.unlink()
        return False


def _write_through_temp(path, write):
    """Call ``write`` on a temporary file beside ``path``, then rename it onto ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def write_csv(df, path):
    """Write a CSV through a temporary file, so a partial file never appears at ``path``."""
    _write_through_temp(path, lambda tmp: df.to_csv(tmp, index=False))


def write_text(text, path):
    """Write text through a temporary file, so a partial file never appears at ``path``."""
    _write_through_temp(path, lambda tmp: tmp.write_text(text))


def thresholds_path(out) -> Path:
    """Return the thresholds CSV path beside ``out``."""
    out = Path(out)
    return out.with_name(f"{out.stem}_thresholds.csv")


def provenance_path(out) -> Path:
    """Return ``<out>.provenance.json``, the provenance file beside ``out``."""
    return Path(str(out) + ".provenance.json")


def cohort_inputs(manifest, datasets) -> dict:
    """Return the files that decide which animals a run selects and how they load, keyed for ``provenance``.

    These are the cohort manifest and, for each dataset, the files ``load_dataset_config`` merges: the base
    config, the local override when there is one, and the dataset's config, plus the samples file of a
    dataset that keeps its samples in a separate file. Paths are resolved against the working directory, where
    the loader reads them.

    Args:
        manifest: Cohort manifest JSON.
        datasets: Dataset names.

    Returns:
        ``{name: path}``.

    Raises:
        FileNotFoundError: If a dataset has no config.
    """
    inputs = {"manifest": Path(manifest).resolve(), "base_config": BASE_CONFIG.resolve()}
    if LOCAL_CONFIG.exists():
        inputs["local_config"] = LOCAL_CONFIG.resolve()
    for ds in datasets:
        inputs[f"dataset_config_{ds}"] = (DATASETS_DIR / f"{ds}.yaml").resolve()
        config = load_dataset_config(ds)
        # resolve_samples_config reads a samples file only when the samples are not inline.
        samples_file = config.get("samples", {}).get("samples_file") if config.get("samples_data") is None else None
        if samples_file:
            inputs[f"samples_file_{ds}"] = Path(samples_file).resolve()
    return inputs


def cli_params(args) -> dict:
    """Return parsed arguments as ``provenance`` parameters, with paths written as strings.

    Args:
        args: An ``argparse.Namespace``.

    Returns:
        ``{argument: value}``.
    """
    def plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    return {name: plain(value) for name, value in vars(args).items()}


@contextlib.contextmanager
def dask_workers(n_workers):
    """Run the block on a local dask cluster of ``n_workers`` processes, or without one when it is 0.

    The detector's spectral work runs on whatever dask client is active.

    Args:
        n_workers: Worker processes; 0 or fewer starts no cluster.

    Yields:
        The dask ``Client``, or None without a cluster.
    """
    if n_workers <= 0:
        yield None
        return
    from dask.distributed import Client, LocalCluster

    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        try:
            client.wait_for_workers(n_workers, timeout=180)
        except Exception as e:  # proceed with the workers that registered
            log.warning(f"{len(client.scheduler_info()['workers'])} of {n_workers} dask workers registered "
                        f"within 180 s ({type(e).__name__}); proceeding")
        yield client


def main(argv=None):
    """Parse command-line arguments, measure the selected animals and write the outputs.

    Args:
        argv: Arguments to parse; by default ``sys.argv[1:]``.

    Returns:
        Process exit code, 0 on success.

    Raises:
        SystemExit: On bad arguments, a refused animal, an empty draw, or no recording to measure.
        RuntimeError: If an animal's recordings and animaldays do not line up, or a recording fails to scan.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True, help="dataset name(s), as in config/datasets/<name>.yaml")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--animal", nargs="+", help="animal id(s) of the one --dataset")
    who.add_argument("--draw", type=int, metavar="N",
                     help="draw N animals per dataset from the active animals the pipeline's quality filter keeps, "
                          "leaving out the labeled animals")
    ap.add_argument("--seed", type=int, default=0, help="seed of the --draw (default 0)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="configuration names from adr.grid_config_names() (default: the whole grid)")
    ap.add_argument("--fragment-len-s", type=float, default=R.FRAG_S, help=f"fragment length (default {R.FRAG_S})")
    ap.add_argument("--max-fit-fragments", type=int, default=adr.MAX_FIT_FRAGMENTS,
                    help=f"pool cap per recording (default {adr.MAX_FIT_FRAGMENTS}); 0 fits on the whole recording")
    ap.add_argument("--fit-seed", type=int, default=0,
                    help="seed of the fit-pool subsample of a recording longer than --max-fit-fragments (default 0); "
                         "the cutoff search keeps the detector's default split seed")
    ap.add_argument("--chunk-fragments", type=int, default=CHUNK_FRAGMENTS,
                    help=f"fragments scored per chunk (default {CHUNK_FRAGMENTS})")
    ap.add_argument("--out", type=Path, required=True,
                    help="rates CSV; the thresholds CSV and <out>.provenance.json are written beside it")
    ap.add_argument("--evidence-out", type=Path, default=None, help="optional parquet of per-fragment scores")
    ap.add_argument("--allow-labeled", action="store_true", help="accept labeled animals in --animal")
    ap.add_argument("--manifest", type=Path, default=MANIFEST,
                    help="cohort manifest listing the labeled animals (default: config/labeling/cohort_manifest.json)")
    ap.add_argument("--workers", type=int, default=0,
                    help="dask worker processes for the spectral work (default 0: none, run in this process)")
    ap.add_argument("--temp-dir", type=Path, default=None,
                    help="temporary directory to load intermediates from, instead of each dataset config's "
                         "temp_directory")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    C._require_repo_root()
    if args.draw is not None and args.draw < 1:
        ap.error("--draw needs a positive N")
    if args.chunk_fragments < 1:
        ap.error("--chunk-fragments needs a positive count")
    try:
        configs = resolve_config_names(args.configs)
        plan, selection = select_animals(args.dataset, animals=args.animal, draw=args.draw, seed=args.seed,
                                         labeled=labeled_animals(args.manifest), allow_labeled=args.allow_labeled)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))

    # Flushed, so the selection heads a redirected log instead of appearing when the run ends.
    for line in plan_lines(plan, draw=args.draw, seed=args.seed):
        print(line, flush=True)
    n_animals = sum(len(animals) for _, animals in plan)
    if n_animals == 0:
        ap.error("the draw is empty in every dataset, so there is nothing to measure; nothing was written")
    # Serialized before the long run, so a record that cannot be written stops the run before it starts.
    inputs = {**cohort_inputs(args.manifest, args.dataset), "script_rejection_rates": Path(__file__).resolve()}
    prov = json.dumps(provenance(inputs, script="rejection_rates", **cli_params(args), config_names=configs,
                                 selection=selection), indent=2)

    cap = args.max_fit_fragments if args.max_fit_fragments > 0 else None
    parallel = args.workers > 0
    rates, thresholds = [], []
    n_recordings = 0
    with dask_workers(args.workers), EvidenceWriter(args.evidence_out) as evidence:
        for ds, animals in plan:
            if not animals:
                continue
            config, samples_config = C._prepare(ds)
            if args.temp_dir is not None:
                set_temp_directory(args.temp_dir)  # after _prepare, which set the config's directory
            for animal_id, genotype in animals:
                ao = C.load_cohort_animal(samples_config, config, animal_id)
                if len(ao.animaldays) != len(ao.long_recordings):
                    raise RuntimeError(f"{animal_id}: {len(ao.long_recordings)} recordings but "
                                       f"{len(ao.animaldays)} animaldays")
                for i, lro in enumerate(ao.long_recordings):
                    keys = {"dataset": ds, "animal": animal_id, "genotype": genotype, "animalday": ao.animaldays[i]}
                    t0 = time.time()
                    try:
                        scan = scan_recording(lro, configs, fragment_len_s=args.fragment_len_s,
                                              max_fit_fragments=cap, fit_seed=args.fit_seed,
                                              chunk_fragments=args.chunk_fragments, parallel=parallel)
                    except Exception as e:
                        raise RuntimeError(f"{ds} {animal_id} {ao.animaldays[i]}: {type(e).__name__}: {e}") from e
                    rates += rate_rows(scan.flags(), scan.channels, scan.timestamps, keys)
                    thresholds += threshold_rows(scan.fits, scan.channels, keys, scan.n_fit_fragments)
                    evidence.write(evidence_frame(scan, keys, args.fragment_len_s))
                    n_recordings += 1
                    log.info(f"{animal_id} {ao.animaldays[i]}: fit on {scan.n_fit_fragments} fragments, scored "
                             f"{len(scan.indices)} fragments x {len(scan.channels)} channels x {len(configs)} "
                             f"configurations in {time.time() - t0:.0f} s")
        if n_recordings == 0:
            raise SystemExit(f"none of the {n_animals} selected animal(s) has a recording to measure; nothing was "
                             "written")
        write_csv(pd.DataFrame(rates, columns=RATE_COLUMNS), args.out)
        write_csv(pd.DataFrame(thresholds, columns=THRESHOLD_COLUMNS), thresholds_path(args.out))
        write_text(prov, provenance_path(args.out))

    print(f"wrote {len(rates)} rate rows over {n_recordings} recordings of {n_animals} animals to {args.out}")
    print(f"wrote {len(thresholds)} cutoffs to {thresholds_path(args.out)}")
    print(f"wrote the provenance to {provenance_path(args.out)}")
    if evidence.written:
        print(f"wrote per-fragment scores to {args.evidence_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
