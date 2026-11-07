#!/usr/bin/env python
"""Extract per-cell detector scores for one split partition, the input to select_detectors.py and score_test.py.

Each labeled animal is loaded once. The WAR (rms, psdband, psdtotal) and LOF scores are computed, the full
autoreject grid (38 configurations, ``adr.grid_config_names()``) is fit per recording, and everything is
aligned to the consensus labels by start time within ``FRAG_S/2``, as ``scoring.score_mask`` does. The
output has one row per labeled (recording, window, channel) with:

- metadata: recording (the label key), animalday, animal, dataset, cv_group (animals recorded together
  share one), genotype, timestamp and isday of the window, the consensus y, n_raters, any_event, each
  rater's category (``cat_<rater>``), and the config list columns below;
- raw filter and LOF scores: rms, logrmsz_abs, beta_wmax, lof. As the pipeline's filters take them,
  logrmsz_abs is the absolute z score of log RMS over the animal's whole WAR (every recording at once), in
  float64 as the pipeline reads the WAR back from parquet, and beta_wmax the window's largest beta
  proportion over the channels where it is defined;
- one 0/1 reject column per autoreject configuration.

``config_list`` is 1 when the cell's channel is on the dataset config's ``bad_channels`` list for its
recording, and ``config_list_defined`` says whether a list applies to the recording at all
(:func:`match_config_lists`). Keys that apply to nothing and list entries that are not among the recording's
channels are listed in the provenance. Beside the parquet, ``<out stem>.autoreject_scores.parquet`` holds,
for every cell and configuration, the fragment score and the channel's cutoff from the fit.

``select_detectors.py`` then builds the pipeline's detector combinations from these cells and chooses the
autoreject configuration offline, on identical cells for every detector.

The labels must lie in one partition of ``config/labeling/split.json`` (``--which dev|test``). The test
parquet is read only by ``score_test.py``. The cell count depends on the labels, since the consensus drops
cells every rater abstained on, so it is printed for dev only; for test only the numbers of animals,
recordings and configurations are printed.

Recordings are matched to label keys through ``config/labeling/recording_map.csv`` (see
``score_detectors.RecordingMap``). Before anything loads, labeled recordings the map leaves unmapped are
refused unless ``--allow-unmapped``, which leaves them out and skips every animal with no mapped labeled
recording; the provenance lists both. An animal whose load, features or grid fit fails (an unreadable file,
or the detector refusing NaN samples) is logged with its traceback and the run goes on; if any failed,
``<out>.errors.json`` lists them and the run exits 1 without writing the parquet. A mapped labeled recording
that yields no cells is refused.

Each finished animal is written to ``<out>.parts/`` with the parameters, input and code hashes and
``src_tree`` it was computed under, and the shards are kept. With ``--resume``, an animal whose shard
matches the run is read back instead of recomputed. The outputs are always assembled from the shards in the
cohort order, so a resumed run writes the same bytes as a fresh one.

The input files, the code files and the state of ``src`` are hashed when the run starts, after each animal
and once more before the outputs are written. If any of them changed, the run stops without saving that
animal or writing outputs, so the shards and the provenance describe the one state the run was keyed under.

When extraction starts, an earlier run's parquet, scores file, provenance and errors file at ``--out`` are
renamed to ``<name>.stale``, so a run that then fails, is refused or is killed leaves none of them to be
taken for its own. The parquet, the scores file and ``<out>.provenance.json`` are written only when every
animal succeeded.

Run from the repository root. Pass the sample-hash check's current-code ``--temp-dir`` so MNE-loaded
recordings read the intermediates that check validated. The grid is the slow phase; give the job cores and
wall time, for example::

    uv run python scripts/labeling/extract_dev_features.py --which dev \\
        --out results/labeling/dev_features.parquet \\
        JD=scripts/labeling/labels/labels_cohort4strains_JD_dev.csv \\
        JL=scripts/labeling/labels/labels_cohort4strains_JL_dev.csv \\
        Yong=scripts/labeling/labels/labels_cohort4strains_Yong_dev.csv
"""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import traceback
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

from dask.distributed import Client, LocalCluster  # noqa: E402
import build_cohort_bundle as C  # noqa: E402
import render_context as R  # noqa: E402
import score_detectors as SDET  # noqa: E402  (git state and code hashes, looked up at call time)
from score_detectors import (  # noqa: E402  (one consensus, split guard, recording map and feature set)
    CONFIG_RECORDING_MAP,
    CONFIG_SPLIT,
    FEATURES,
    RecordingMap,
    available_cpus,
    build_consensus,
    check_split,
    config_files,
    parse_raters,
    provenance,
    require_repo_cwd,
)
from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.core.utils import resolve_channel  # noqa: E402
from neurodent.core.utils.paths import set_temp_directory  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402
from neurodent.results.scoring import RATER_CATEGORY_PREFIX  # noqa: E402

log = logging.getLogger("extract_dev_features")

# Per-cell columns before the per-rater category columns (cat_<rater>) and the autoreject configuration
# columns (adr.grid_config_names()).
CELL_COLS = ["recording", "window", "channel", "t_start_s", "y",
             "rms", "logrmsz_abs", "beta_wmax", "lof",
             "animal", "animalday", "dataset", "cv_group", "genotype", "timestamp", "isday",
             "n_raters", "any_event", "config_list", "config_list_defined"]
# Rows of <out stem>.autoreject_scores.parquet: one per cell and configuration.
SCORE_COLS = ["recording", "window", "channel", "config", "score", "cutoff"]
# The date format of an animalday ("{animal} {genotype} {date}") and of a dict-form bad_channels key.
DATE_FORMAT = "%b-%d-%Y"


def beta_prop_grid(df_stats):
    """Return the (W, C) beta power proportion, psdband['beta'] / psdtotal, which ``high_beta`` thresholds.

    It is divided in float64, as the pipeline divides the values it reads back from parquet.
    """
    band = pd.DataFrame(df_stats["psdband"].tolist())
    beta = np.array(band["beta"].tolist(), dtype=np.float64)
    total = np.array(df_stats["psdtotal"].tolist(), dtype=np.float64)
    return beta / total


def war_filter_scores(war):
    """Return the ``(W, C)`` rms, log-RMS z score and beta proportion over an animal's whole WAR.

    These are the values the pipeline's fragment filters threshold. The loader's rms is float32; the pipeline
    reads the WAR back from parquet as float64 and takes the log and the z score there. The conversion is
    exact, so casting first gives its z score to the last bit. The z score runs over every recording at once,
    as the pipeline filters the whole WAR.

    Returns:
        ``(rms, logrmsz, beta)``; logrmsz is signed.
    """
    rms = extract_linear_array(war.result["rms"]).astype(np.float64)
    return rms, zscore(np.log(rms), axis=0, nan_policy="omit"), beta_prop_grid(war.result)


def lof_by_abbrev(war, animalday):
    """Return ``{channel abbreviation: LOF score}`` for one animalday.

    ``lof_scores_dict[animalday]`` is ``{"lof_scores": [...], "channel_names": [raw names]}``, two parallel
    lists keyed by raw names, so the lists are zipped and the raw names mapped to the WAR's canonical
    abbreviations, which the consensus ``channel`` column uses. Returns {} when LOF is absent.
    """
    data = war.lof_scores_dict.get(animalday)
    if not data or not data.get("channel_names"):
        return {}
    abbrevs = set(war.channel_abbrevs)
    raw2ab = dict(zip(war.channel_names or [], war.channel_abbrevs or []))
    out = {}
    for raw, score in zip(data["channel_names"], data["lof_scores"]):
        ab = raw if raw in abbrevs else raw2ab.get(raw)  # names already stored as abbreviations are kept
        if ab is not None:
            out[ab] = float(score)
    return out


def align_cells(rms, logrmsz, beta_wmax, lof_by_ch, ch_names, cons_rec, grid_times, frag_s=R.FRAG_S,
                row_meta=None, cell_cols=()):
    """Return one dict per labeled cell with the scores each filter thresholds.

    Each cell is matched to the WAR row whose start time is within ``frag_s/2`` (as ``score_mask`` does).
    Transforms that need more than the cell (log-RMS z over the animal's whole WAR, high_beta's maximum over
    channels) are computed by the caller.

    Args:
        rms: ``(W, C)`` RMS.
        logrmsz: ``(W, C)`` z score of log RMS over the animal's whole WAR, as ``logrms_range`` takes it.
        beta_wmax: ``(W,)`` maximum beta proportion over channels, ignoring undefined ones.
        lof_by_ch: ``{channel: LOF score}``.
        ch_names: Channel abbreviations in column order.
        cons_rec: The recording's consensus rows.
        grid_times: ``(W,)`` row start times in seconds.
        frag_s: Fragment length in seconds.
        row_meta: ``{column: (W,) array}`` of per-row values to copy into each cell (timestamp, isday).
        cell_cols: Consensus columns to copy into each cell (n_raters, any_event, rater categories).

    Returns:
        A list of dicts; unmatched cells are left out.
    """
    ch_idx = {c: i for i, c in enumerate(ch_names)}
    gt = np.asarray(grid_times, float)
    tol = frag_s / 2.0
    out = []
    for _, r in cons_rec.iterrows():
        ch = r["channel"]
        if ch not in ch_idx:
            continue
        t = float(r["t_start_s"])
        d = np.abs(gt - t)
        row = int(d.argmin())
        if d[row] > tol:
            continue
        ci = ch_idx[ch]
        cell = {"recording": r["recording"], "window": int(r["window"]), "channel": ch,
                "t_start_s": t, "y": int(r["y_true"]),
                "rms": float(rms[row, ci]),  # high_rms (above tau), low_rms (below tau)
                "logrmsz_abs": float(abs(logrmsz[row, ci])),  # logrms_range (above tau)
                "beta_wmax": float(beta_wmax[row]),  # high_beta: the window is rejected if any channel is above
                "lof": float(lof_by_ch.get(ch, np.nan))}  # lof (above tau), one score per channel
        for name, values in (row_meta or {}).items():
            cell[name] = values[row]
        for name in cell_cols:
            cell[name] = r[name]
        out.append(cell)
    return out


def pool_rows(ar_ch_names, ar_grid_times, cons_rec, frag_s=R.FRAG_S):
    """Return ``{(window, channel): (pool row, channel index)}`` for the labeled cells found in the fit pool.

    Each cell's start time is matched to the fit-pool grid (``ar_grid_times``) within ``frag_s/2`` and its
    channel to ``ar_ch_names``. Labeled fragments are always in the pool (``adr.fit_pool_indices``), so a
    match is expected; an unmatched cell is left out.
    """
    ch_idx = {c: i for i, c in enumerate(ar_ch_names)}
    gt = np.asarray(ar_grid_times, float)
    tol = frag_s / 2.0
    out = {}
    for _, r in cons_rec.iterrows():
        ch = r["channel"]
        if ch not in ch_idx:
            continue
        t = float(r["t_start_s"])
        d = np.abs(gt - t)
        row = int(d.argmin())
        if d[row] > tol:
            continue
        out[(int(r["window"]), ch)] = (row, ch_idx[ch])
    return out


def ar_cell_bits(grids, ar_ch_names, ar_grid_times, cons_rec, frag_s=R.FRAG_S):
    """Return ``{(window, channel): {config: 0 or 1}}`` for the labeled cells.

    Cells are matched as in :func:`pool_rows`; an unmatched cell gets no bits (NaN downstream). 1 means
    reject, the same polarity as the label y.
    """
    return {cell: {name: int(m[row, ci]) for name, m in grids.items()}
            for cell, (row, ci) in pool_rows(ar_ch_names, ar_grid_times, cons_rec, frag_s).items()}


def ar_cell_evidence(fits, ar_ch_names, ar_grid_times, cons_rec, frag_s=R.FRAG_S):
    """Return ``{(window, channel): {config: (score, cutoff)}}`` for the labeled cells.

    The score is the cell's fragment score in the fit and the cutoff its channel's learned threshold, both
    read from each :class:`~neurodent.results.autoreject_detector.DetectorFit`, not recomputed. Cells are
    matched as in :func:`pool_rows`.
    """
    return {cell: {name: (float(f.scores[row, ci]), float(f.thresholds[ci])) for name, f in fits.items()}
            for cell, (row, ci) in pool_rows(ar_ch_names, ar_grid_times, cons_rec, frag_s).items()}


def date_named(text):
    """Return the date the last word of an animalday or a bad_channels key names (``Jan-08-2022``), or None."""
    words = str(text).split()
    try:
        return datetime.strptime(words[-1], DATE_FORMAT).date() if words else None
    except ValueError:
        return None


def match_config_lists(entry, recording_dates):
    """Return the dataset config's bad_channels list for each labeled recording of one animal.

    A list-form entry, ``{"_all": [...]}`` in the expanded samples config, applies to every recording. A
    dict-form key such as ``"M3 MHet Jan-08-2022"`` has the animalday shape, but its id and genotype predate
    the current ones, so only its date is read: it applies to the recording whose date is the same. A
    recording's date is the one its animalday names, which the loader stamps from the start of the
    recording's first file (``get_date_string``). sox5_bin configures session end times, which the loader
    turns into starts with the loaded durations, so a session that runs past midnight is dated by the day it
    started. A key that names no date, or whose date matches no recording or more than one, or any key while
    a recording has no dated animalday, applies to nothing.

    Args:
        entry: The animal's ``samples_config["bad_channels"]`` value, or None.
        recording_dates: ``{recording: date or None}`` of the animal's labeled recordings.

    Returns:
        ``(lists, matches, unmatched)``: ``{recording: [channel name, ...]}`` for every recording a list
        applies to, the ``{"key", "recording"}`` pairs applied, and ``{"key", "reason"}`` for each key that
        applies to nothing.
    """
    lists, matches, unmatched = {}, [], []
    if not entry:
        return lists, matches, unmatched

    def add(recording, channels):
        merged = lists.setdefault(recording, [])
        merged.extend(c for c in (channels or []) if c not in merged)

    if "_all" in entry:
        for recording in recording_dates:
            add(recording, entry["_all"])
            matches.append({"key": "_all", "recording": recording})
    undated = sorted(r for r, d in recording_dates.items() if d is None)
    for key, channels in entry.items():
        if key == "_all":
            continue
        date = date_named(key)
        hits = sorted(r for r, d in recording_dates.items() if d is not None and d == date)
        if date is None:
            unmatched.append({"key": key, "reason": f"names no date in the form {DATE_FORMAT}"})
        elif undated:
            unmatched.append({"key": key, "reason": f"labeled recordings with no dated animalday: {undated}"})
        elif not hits:
            unmatched.append({"key": key, "reason": f"no labeled recording on {date.strftime(DATE_FORMAT)}"})
        elif len(hits) > 1:
            unmatched.append({"key": key, "reason": f"{len(hits)} labeled recordings on "
                                                    f"{date.strftime(DATE_FORMAT)}: {hits}"})
        else:
            add(hits[0], channels)
            matches.append({"key": key, "recording": hits[0]})
    return lists, matches, unmatched


def resolve_list(names, channels, resolve=None):
    """Return the canonical channels of ``names`` that are among ``channels``, and the names that are not.

    Each name is resolved as the pipeline's channel filters do (``resolve_channel``), so an abbreviation or a
    configured raw name both work.

    Args:
        names: Channel names from a bad_channels list.
        channels: The recording's channel abbreviations.
        resolve: Name resolver; defaults to ``resolve_channel``.

    Returns:
        ``(found, unknown)``: a set of abbreviations, and ``{"channel", "reason"}`` per name left out, with the
        name's ``abbrev`` when it resolves, which ``select_detectors.listed_absent`` reads.
    """
    resolve = resolve or resolve_channel
    found, unknown = set(), []
    for name in names:
        try:
            abbrev = resolve(name)
        except ValueError:
            unknown.append({"channel": name, "reason": "not in the dataset's channel map"})
            continue
        if abbrev in channels:
            found.add(abbrev)
        else:
            unknown.append({"channel": name, "abbrev": abbrev,
                            "reason": f"{abbrev} is not among the recording's channels"})
    return found, unknown


def load_and_measure(samples_config, config, animal_id, *, lof_chunk_duration_s, chunk_duration_s):
    """Load one animal and compute its LOF scores and its WAR (FEATURES over notched FRAG_S windows).

    Returns:
        ``(ao, war)``.
    """
    ao = C.load_cohort_animal(samples_config, config, animal_id)
    az = AnimalAnalyzer(ao)
    az.compute_bad_channels(lof_threshold=1.0, lof_chunk_duration_s=lof_chunk_duration_s)  # scores only
    war = az.compute_windowed_analysis(FEATURES, window_s=R.FRAG_S, apply_notch_filter=True,
                                       multiprocess_mode="dask", chunk_duration_s=chunk_duration_s)
    return ao, war


def animal_cells(ao, war, *, animal_id, dataset, cv_group, cons_animal, rmap, cat_cols, ar_cols, config_entry,
                 max_fit_fragments, seed, resolve=None):
    """Return one animal's labeled cells, its autoreject evidence and its config list report.

    Each labeled recording is the WAR slice with its animalday; its cells take the filter and LOF scores
    from the WAR and the autoreject bits, scores and cutoffs from the grid fit on that recording.

    Args:
        ao: The loaded AnimalOrganizer.
        war: Its WindowAnalysisResult, with LOF scores.
        animal_id: Animal id.
        dataset: Dataset name.
        cv_group: The animal's cross-validation group.
        cons_animal: The animal's consensus rows.
        rmap: A :class:`RecordingMap`.
        cat_cols: Rater category columns to copy.
        ar_cols: Grid configuration names, in column order.
        config_entry: The animal's ``samples_config["bad_channels"]`` value, or None.
        max_fit_fragments: Fit-pool cap per recording; None uses the whole recording.
        seed: Seed of the fit-pool subsample.
        resolve: Channel name resolver for the bad_channels lists; defaults to ``resolve_channel``.

    Returns:
        ``(cells, scores, report)``: the cell DataFrame (``CELL_COLS``, ``cat_cols``, ``ar_cols``), the
        evidence DataFrame (``SCORE_COLS``, one row per cell and configuration, NaN where the cell is not in
        the pool), and a dict of lists: ``config_list_matches`` (each key and the recording it applies to),
        ``config_list_unmatched`` (keys that apply to nothing, with the reason), ``config_list_uncovered``
        (recordings of an animal with a dict-form entry that no key applies to) and
        ``config_list_unknown_channels`` (list entries that are not among the recording's channels).

    Raises:
        ValueError: If two loaded recordings map to one label key, or the detector refuses its input.
    """
    ch_names = list(war.channel_abbrevs)
    animaldays = war.result["animalday"].to_numpy()
    rms_all, logrmsz_all, beta_all = war_filter_scores(war)
    labeled = set(cons_animal["recording"])

    recordings = []  # (position, animalday, label key) of each labeled recording that loaded
    for i, animalday in enumerate(ao.animaldays):
        recording = rmap.label(animal_id, i, animalday)
        if recording is None:
            log.warning(f"{animal_id} position {i} ({animalday!r}) is not in the recording map; skipped")
            continue
        if recording not in labeled:
            continue
        if any(recording == r for _, _, r in recordings):
            raise ValueError(f"{animal_id}: two loaded recordings map to {recording}")
        recordings.append((i, animalday, recording))

    animalday_of = {rec: ad for _, ad, rec in recordings}
    lists, matches, unmatched = match_config_lists(
        config_entry, {rec: date_named(ad) for _, ad, rec in recordings})
    tag = {"animal": animal_id, "dataset": dataset}
    applied, unknown = {}, []
    for rec, names in lists.items():
        applied[rec], left_out = resolve_list(names, ch_names, resolve)
        unknown += [{**tag, "recording": rec, "animalday": animalday_of[rec], **u} for u in left_out]
    report = {
        "config_list_matches": [{**tag, **m, "animalday": animalday_of[m["recording"]]} for m in matches],
        "config_list_unmatched": [{**tag, **u} for u in unmatched],
        "config_list_uncovered": [{**tag, "recording": rec, "animalday": ad} for _, ad, rec in recordings
                                  if config_entry and rec not in lists],
        "config_list_unknown_channels": unknown,
    }

    rows, score_rows = [], []
    for i, animalday, recording in recordings:
        cons_rec = cons_animal[cons_animal["recording"] == recording]
        rowsel = animaldays == animalday
        n = int(rowsel.sum())
        if n == 0:
            log.warning(f"{recording}: no WAR fragments for animalday {animalday!r}; skipped")
            continue
        grid_times = np.arange(n) * R.FRAG_S
        lof_by_ch = lof_by_abbrev(war, animalday)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a window with no defined proportion stays NaN
            # high_beta rejects a window when any channel is over, so a 0/0 proportion must not hide the others.
            beta_wmax = np.nanmax(beta_all[rowsel], axis=1)
        rec_rows = war.result.loc[rowsel]
        row_meta = {"timestamp": pd.to_datetime(rec_rows["timestamp"]).to_numpy(),
                    "isday": rec_rows["isday"].astype(bool).to_numpy(),
                    "genotype": rec_rows["genotype"].astype(str).to_numpy()}
        cells = align_cells(rms_all[rowsel], logrmsz_all[rowsel], beta_wmax, lof_by_ch, ch_names, cons_rec,
                            grid_times, row_meta=row_meta, cell_cols=["n_raters", "any_event", *cat_cols])

        # The autoreject grid runs on the same recording and the same consensus cells.
        Xar, ar_ch, fs_ar, ar_gt = adr.build_fit_epochs(
            ao.long_recordings[i], cons_rec["window"].unique(), fragment_len_s=R.FRAG_S,
            max_fit_fragments=max_fit_fragments, seed=seed)
        log.info(f"{recording}: autoreject grid, {Xar.shape[0]} fragments x {Xar.shape[1]} channels x "
                 f"{len(ar_cols)} configurations")
        fits = adr.fit_grid(Xar, fs_ar)  # the fits compute_masks_grid flags from, with scores and cutoffs
        log.info(f"{recording}: autoreject grid done")
        bits = ar_cell_bits({name: fit.mask for name, fit in fits.items()}, ar_ch, ar_gt, cons_rec)
        evidence = ar_cell_evidence(fits, ar_ch, ar_gt, cons_rec)
        on_list = applied.get(recording)
        for row in cells:
            cell = (row["window"], row["channel"])
            row.update(animal=animal_id, animalday=animalday, dataset=dataset, cv_group=cv_group,
                       config_list=int(on_list is not None and row["channel"] in on_list),
                       config_list_defined=on_list is not None)
            row.update(bits.get(cell, {}))
            ev = evidence.get(cell, {})
            score_rows += [(recording, row["window"], row["channel"], name, *ev.get(name, (np.nan, np.nan)))
                           for name in ar_cols]
        rows += cells
    return (pd.DataFrame(rows, columns=CELL_COLS + cat_cols + ar_cols),
            pd.DataFrame(score_rows, columns=SCORE_COLS), report)


# Per-animal shards in <out>.parts/


def _sha256_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shard_paths(parts_dir, dataset, animal_id):
    """Return the JSON, cells and scores paths of one animal's shard in ``parts_dir``."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{dataset}__{animal_id}")
    stem = f"{safe}-{hashlib.sha256(f'{dataset}/{animal_id}'.encode()).hexdigest()[:10]}"
    return {"json": parts_dir / f"{stem}.json", "cells": parts_dir / f"{stem}.cells.parquet",
            "scores": parts_dir / f"{stem}.autoreject_scores.parquet"}


def _replace_write(path, write):
    """Write ``path`` through ``<path>.tmp`` and a rename, so a partial file never has the final name."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_shard(paths, key, cells, scores, report):
    """Write one animal's cells, scores and report under ``key``; the JSON goes last and records the files' sha256."""
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    _replace_write(paths["cells"], cells.to_parquet)
    _replace_write(paths["scores"], scores.to_parquet)
    doc = {"key": key, "report": report,
           "sha256": {"cells": _sha256_bytes(paths["cells"]), "scores": _sha256_bytes(paths["scores"])}}
    _replace_write(paths["json"], lambda p: p.write_text(json.dumps(doc, indent=2, default=str)))


def shard_matches(paths, key):
    """Return whether a complete shard computed under ``key`` exists at ``paths``."""
    try:
        doc = json.loads(paths["json"].read_text())
        return (doc["key"] == key and _sha256_bytes(paths["cells"]) == doc["sha256"]["cells"]
                and _sha256_bytes(paths["scores"]) == doc["sha256"]["scores"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def read_shard(paths):
    """Return ``(cells, scores, report)`` from one animal's shard."""
    doc = json.loads(paths["json"].read_text())
    return pd.read_parquet(paths["cells"]), pd.read_parquet(paths["scores"]), doc["report"]


# The state of the files a run reads, and the outputs it writes


def run_inputs(manifests, keymap, split, recording_map, datasets):
    """Return ``{name: path}`` of the files a run reads besides its code.

    These are the rater CSVs, the keymap, the split, the recording map (None with positional recordings, and
    then left out) and the configs of ``datasets`` (:func:`config_files`).
    """
    inputs = {f"labels_{r}": p for r, p in sorted(manifests.items())}
    inputs.update(keymap=keymap, split=split, recording_map=recording_map)
    inputs.update(config_files(datasets))
    return {name: Path(p) for name, p in inputs.items() if p is not None}


def file_state(inputs):
    """Return the sha256 of each input and code file, and the git state of ``src``.

    Args:
        inputs: ``{name: path}`` from :func:`run_inputs`.

    Returns:
        ``{"inputs": {name: sha256, None if missing}, "code": {path: sha256}, "src_tree", "src_dirty"}``.
    """
    return {"inputs": {name: _sha256_bytes(p) if p.exists() else None for name, p in inputs.items()},
            "code": SDET.code_hashes(), **SDET.src_state()}


def state_changes(before, after):
    """Return what differs between two :func:`file_state` values.

    Each difference is named ``"inputs <name>"``, ``"code <path>"``, ``"src_tree"`` or ``"src_dirty"``.
    """
    out = []
    for part in ("inputs", "code"):
        a, b = before[part], after[part]
        out += [f"{part} {name}" for name in sorted(set(a) | set(b)) if a.get(name) != b.get(name)]
    return out + [k for k in ("src_tree", "src_dirty") if before[k] != after[k]]


def output_paths(out):
    """Return the paths a run writes for ``--out``: features, autoreject scores, provenance and errors file."""
    out = Path(out)
    return {"features": out, "autoreject_scores": out.with_name(out.stem + ".autoreject_scores.parquet"),
            "provenance": Path(f"{out}.provenance.json"), "errors": Path(f"{out}.errors.json")}


def retire_outputs(paths):
    """Rename each existing file in ``paths`` to ``<name>.stale`` and return the ones renamed.

    A run does this before extracting, so a run that fails, is refused or is killed leaves no earlier
    outputs under the names its readers open.
    """
    retired = []
    for p in paths:
        if p.exists():
            os.replace(p, p.with_name(p.name + ".stale"))
            retired.append(p)
    return retired


@contextlib.contextmanager
def dask_cluster(n_workers):
    """Run a dask LocalCluster of ``n_workers`` single-thread processes (dashboard off) with a client."""
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        try:
            client.wait_for_workers(n_workers, timeout=180)
        except Exception as e:
            log.warning(f"workers: {type(e).__name__}; proceeding with {len(client.scheduler_info()['workers'])}")
        yield client


def _animal_of(recording):
    return recording.rsplit("__", 1)[0]


def main(argv=None):
    """Extract the features of one split partition.

    Returns:
        0 when the outputs were written, 1 when an animal failed (``<out>.errors.json`` lists them).

    Raises:
        SystemExit: On inputs the run refuses, before or after extraction, and when an input file, a code file
            or the state of ``src`` changes during the run.
    """
    ap = argparse.ArgumentParser(
        description="Export per-cell filter, LOF and autoreject scores for one split partition.")
    ap.add_argument("raters", nargs="+", metavar="rater=csv", help="rater=csv pairs, one or more")
    ap.add_argument("--keymap", type=Path, default=REPO / "config/labeling/keymap.csv",
                    help="slot to channel unblinding key (default: config/labeling/keymap.csv)")
    ap.add_argument("--split", type=Path, default=CONFIG_SPLIT)
    ap.add_argument("--which", choices=["dev", "test"], required=True)
    ap.add_argument("--recording-map", type=Path, default=CONFIG_RECORDING_MAP,
                    help="labeled recording to current recording map from compare_window_hashes.py")
    ap.add_argument("--positional-recordings", action="store_true",
                    help="match recordings by position, only for a bundle rendered by this code")
    ap.add_argument("--allow-unmapped", action="store_true",
                    help="leave out labelled recordings the recording map leaves unmapped, and every animal "
                         "with no mapped labelled recording (both listed in the provenance), instead of refusing")
    ap.add_argument("--resume", action="store_true",
                    help="read back each animal whose shard in <out>.parts/ was computed under this run's "
                         "parameters, inputs and code instead of recomputing it")
    ap.add_argument("--datasets", nargs="+", default=list(C.REAL_STRAINS),
                    help="datasets whose animals to scan (default: the labeled strains)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rule", default="majority", choices=["majority", "unanimous", "any"])
    ap.add_argument("--lof-chunk-duration-s", type=float, default=60)
    ap.add_argument("--chunk-duration-s", type=float, default=3600)
    ap.add_argument("--ar-max-fit-fragments", type=int, default=adr.MAX_FIT_FRAGMENTS,
                    help="autoreject fit-pool cap per recording (labeled fragments always included; 0 uses the "
                         "whole recording)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the fit-pool subsample and cohort iteration; equal seeds and caps give "
                         "score_detectors.py identical fragments")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--temp-dir", type=Path, default=None,
                    help="load intermediates from this directory instead of the config's temp_directory (use the "
                         "sample-hash check's current-code --temp-dir)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    require_repo_cwd()

    manifests = parse_raters(ap, args.raters)

    # The files the run reads, hashed before any is read and compared after each animal and before the
    # outputs are written.
    def current_inputs():
        return run_inputs(manifests, args.keymap, args.split,
                          None if args.positional_recordings else args.recording_map, args.datasets)

    start = file_state(current_inputs())

    def refuse_changes(consequence, now=None):
        changed = state_changes(start, file_state(current_inputs()) if now is None else now)
        if changed:
            raise SystemExit(f"{', '.join(changed)} changed during the run; refusing. {consequence} Rerun once the "
                             "files are settled; with --resume, shards computed under that state are reused.")

    cons = build_consensus(manifests, args.keymap, args.rule)
    labeled_recs = sorted(cons["recording"].unique())
    labeled_animals = {_animal_of(r) for r in labeled_recs}
    check_split(labeled_animals, args.split, args.which)
    print(f"split ok: {len(labeled_animals)} labelled animals within split['{args.which}']")
    rmap = RecordingMap(None if args.positional_recordings else args.recording_map,
                        positional=args.positional_recordings)
    cat_cols = sorted(c for c in cons.columns if c.startswith(RATER_CATEGORY_PREFIX))
    ar_cols = adr.grid_config_names()  # every configuration, always, on identical cells

    # Unmapped labeled recordings, found before anything loads.
    unmapped = rmap.unmapped(labeled_recs)
    unmapped_recs = {u["recording"] for u in unmapped}
    if unmapped and not args.allow_unmapped:
        raise SystemExit(f"{len(unmapped)} labelled recording(s) the recording map leaves unmapped: "
                         f"{sorted(unmapped_recs)}. Refusing before loading anything; pass --allow-unmapped to "
                         "leave them out, as the pre-registration does.")
    dropped_animals = sorted(a for a in labeled_animals
                             if all(r in unmapped_recs for r in labeled_recs if _animal_of(r) == a))
    animals_wanted = labeled_animals - set(dropped_animals)
    if dropped_animals:
        log.warning(f"no mapped labelled recording, so not extracted: {dropped_animals}")
    if not animals_wanted:
        raise SystemExit("no labelled animal has a mapped recording; nothing to extract")

    # The cohort walk fixes the order the outputs are assembled in.
    plan = [(ds, a) for ds, _, _, a in C.iter_cohort_animals(args.datasets, limit_per_dataset=None, seed=args.seed)
            if a in animals_wanted]
    repeated = sorted(a for a, k in Counter(a for _, a in plan).items() if k > 1)
    if repeated:
        raise SystemExit(f"animal ids found in more than one of {args.datasets}: {repeated}; refusing")
    absent = sorted(animals_wanted - {a for _, a in plan})
    if absent:
        raise SystemExit(f"labelled animals in none of the datasets {args.datasets}: {absent}; refusing")

    # A shard is reused only when it was computed with these parameters, inputs and code.
    run_key = {
        "params": {"which": args.which, "rule": args.rule, "raters": sorted(manifests),
                   "positional_recordings": args.positional_recordings,
                   "ar_max_fit_fragments": args.ar_max_fit_fragments, "seed": args.seed,
                   "lof_chunk_duration_s": args.lof_chunk_duration_s, "chunk_duration_s": args.chunk_duration_s,
                   "temp_dir": None if args.temp_dir is None else str(args.temp_dir), "configs": ar_cols,
                   "features": FEATURES, "fragment_s": R.FRAG_S},
        "inputs": {name: start["inputs"].get(name)
                   for name in [*(f"labels_{r}" for r in sorted(manifests)), "keymap", "recording_map"]},
        "code": start["code"],
        "src_tree": start["src_tree"],
        "src_dirty": start["src_dirty"],
    }
    dataset_keys = {}  # dataset: the run key plus that dataset's config files

    def shard_key(ds):
        if ds not in dataset_keys:
            configs = {name: start["inputs"].get(name) for name in config_files([ds])}
            dataset_keys[ds] = json.loads(json.dumps({**run_key, "dataset": ds, "configs": configs}))
        return dataset_keys[ds]

    parts_dir = Path(str(args.out) + ".parts")
    outputs = output_paths(args.out)
    errors_path = outputs["errors"]
    retired = retire_outputs(outputs.values())
    if retired:
        log.warning(f"renamed an earlier run's outputs to <name>.stale: {[str(p) for p in retired]}")
    n_workers = args.workers or available_cpus()
    max_fit = args.ar_max_fit_fragments or None
    errors, resumed, done = [], [], set()
    with contextlib.ExitStack() as stack:
        cluster_up = False
        for ds, samples_config, config, animal_id in C.iter_cohort_animals(args.datasets, limit_per_dataset=None,
                                                                           seed=args.seed):
            if animal_id not in animals_wanted:
                continue
            paths, key = shard_paths(parts_dir, ds, animal_id), shard_key(ds)
            if args.resume and shard_matches(paths, key):
                resumed.append(animal_id)
                done.add((ds, animal_id))
                log.info(f"{animal_id}: resumed from {paths['json']}")
                continue
            if not cluster_up:
                stack.enter_context(dask_cluster(n_workers))
                cluster_up = True
            if args.temp_dir is not None:
                set_temp_directory(args.temp_dir)  # after _prepare, which set the config's directory
            try:
                ao, war = load_and_measure(samples_config, config, animal_id,
                                           lof_chunk_duration_s=args.lof_chunk_duration_s,
                                           chunk_duration_s=args.chunk_duration_s)
                cells, scores, report = animal_cells(
                    ao, war, animal_id=animal_id, dataset=ds,
                    cv_group=samples_config.get("_animal_groups", {}).get(animal_id, animal_id),
                    cons_animal=cons[cons["recording"].map(_animal_of) == animal_id], rmap=rmap,
                    cat_cols=cat_cols, ar_cols=ar_cols,
                    config_entry=samples_config.get("bad_channels", {}).get(animal_id),
                    max_fit_fragments=max_fit, seed=args.seed)
                # SystemExit passes the handler below, so the run stops before keying this animal's shard.
                refuse_changes(f"{animal_id} was not saved; the animals finished before it are in {parts_dir}.")
                write_shard(paths, key, cells, scores, report)
            except Exception as e:
                log.exception(f"{animal_id}: failed; continuing with the other animals")
                errors.append({"animal": animal_id, "dataset": ds, "error_type": type(e).__name__,
                               "message": str(e), "traceback": traceback.format_exc()})
                continue
            done.add((ds, animal_id))
            log.info(f"{animal_id}: done")

    if errors:
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors_path.write_text(json.dumps(errors, indent=2))
        print(f"{len(errors)} of {len(plan)} animal(s) failed: {[e['animal'] for e in errors]}. No parquet written; "
              f"see {errors_path}. Finished animals are kept in {parts_dir}, so a rerun with --resume redoes only "
              "the rest.", file=sys.stderr)
        return 1

    # Assemble from the shards in the plan order, so a resumed run writes the same bytes as a fresh one. Only
    # shards this run computed or matched are read.
    if set(plan) - done:
        raise SystemExit(f"the cohort walk did not reach {sorted(a for _, a in set(plan) - done)}; refusing")
    shards = [read_shard(shard_paths(parts_dir, ds, a)) for ds, a in plan]
    df, scores = (pd.concat([s[k] for s in shards if len(s[k])] or [shards[0][k]], ignore_index=True)
                  for k in (0, 1))
    reports = [s[2] for s in shards]
    empty = sorted(set(labeled_recs) - unmapped_recs - set(df["recording"]))
    if empty:
        raise SystemExit(f"mapped labelled recordings with no extracted cells: {empty}. Refusing to write a "
                         "parquet that drops them; check the recording map and the loaded recordings.")

    prov = provenance(current_inputs(), script="extract_dev_features", which=args.which, rule=args.rule,
                      raters=sorted(manifests), positional_recordings=args.positional_recordings,
                      ar_max_fit_fragments=args.ar_max_fit_fragments, seed=args.seed,
                      configs=ar_cols, datasets=list(args.datasets), dropped_recordings=sorted(unmapped_recs),
                      temp_dir=None if args.temp_dir is None else str(args.temp_dir),
                      lof_chunk_duration_s=args.lof_chunk_duration_s, chunk_duration_s=args.chunk_duration_s,
                      allow_unmapped=args.allow_unmapped, resume=args.resume)
    # provenance() hashes the files once more; it is written only if they are still as every shard was keyed.
    refuse_changes(f"No outputs were written; the shards are in {parts_dir}.",
                   now={"inputs": {name: v["sha256"] for name, v in prov["inputs"].items()}, "code": prov["code"],
                        "src_tree": prov["src_tree"], "src_dirty": prov["src_dirty"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores_path = outputs["autoreject_scores"]
    tmp_out, tmp_scores = (p.with_name(p.name + ".tmp") for p in (args.out, scores_path))
    df.to_parquet(tmp_out)
    scores.to_parquet(tmp_scores)
    prov.update(
        unmapped_recordings=unmapped,
        dropped_animals=dropped_animals,
        resumed_animals=resumed,
        **{name: [item for r in reports for item in r.get(name, [])]
           for name in ("config_list_matches", "config_list_unmatched", "config_list_uncovered",
                        "config_list_unknown_channels")},
        outputs={"features": {"path": str(args.out), "sha256": _sha256_bytes(tmp_out)},
                 "autoreject_scores": {"path": str(scores_path), "sha256": _sha256_bytes(tmp_scores)}},
    )
    _replace_write(outputs["provenance"], lambda p: p.write_text(json.dumps(prov, indent=2, default=str)))
    os.replace(tmp_scores, scores_path)
    os.replace(tmp_out, args.out)  # --out appears only on full success
    n_animals, n_recs = df["animal"].nunique(), df["recording"].nunique()
    counts = f"{len(df)} cells over " if args.which == "dev" else ""  # test cell counts depend on label values
    print(f"wrote {counts}{n_animals} animals, {n_recs} recordings, {len(ar_cols)} autoreject configurations to "
          f"{args.out}, with the scores in {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
