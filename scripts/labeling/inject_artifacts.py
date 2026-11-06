#!/usr/bin/env python
"""Add synthetic artifacts to a fit pool and count what the adapted autoreject detector flags.

A check that reads no labels. Six kinds of artifact are added to seeded (fragment, channel) cells. Amplitudes
are multiples of the channel's reference RMS, the median over the clean pool's fragments of each fragment's
RMS about its mean (default multiples 2, 5, 10 and 20):

- ``clip``: a span held at the fragment's mean plus amplitude times the reference RMS.
- ``flatline``: a span held at the fragment's mean; amplitude is ignored.
- ``step``: a DC step of amplitude times the reference RMS that lasts to the end of the fragment.
- ``broadband``: a span of white noise whose RMS is amplitude times the reference RMS.
- ``line60``: a span of a 60 Hz sinusoid whose RMS is amplitude times the reference RMS.
- ``drift``: a 0.5 Hz sinusoid whose peak is amplitude times the reference RMS, over the whole fragment.

A span covers a seeded 10 to 50 percent of the fragment at a seeded position; a step's span ends at the
fragment's end, so the step starts 10 to 50 percent of the fragment before it.

Artifacts are added to the pool the detector fits on. With real data that pool comes after the per-fragment
60 Hz notch (``adr.build_fit_epochs`` notches each fragment), so ``line60`` stands for 60 Hz energy the notch
did not remove, not for line noise as recorded.

The experiment fits the configurations on the clean pool once. Then, for each kind and amplitude, it adds the
artifacts to a copy of the pool and fits again, so each kind and amplitude moves the cutoffs on its own. One
seeded set of cells per pool, a ``--fraction`` of every channel's fragments (default 5 percent) drawn from
``--seed`` alone, takes every kind and amplitude. The kinds are therefore measured on the same cells, and a run
split by kind gives the same numbers as one run with every kind. The amplitudes of one kind also reuse its
spans, noise and phases, so the amplitude is the only thing that changes between them. Every fit keeps the
detector's default split seed for its cutoff search, as ``extract_dev_features.py`` and
``score_detectors.py`` do. One row per configuration, kind and amplitude reports:

- ``rate_injected``: flag rate on the injected cells, in the fit on the injected pool.
- ``rate_injected_clean``: flag rate on the same cells, in the clean fit.
- ``rate_other``: flag rate on all other cells, in the fit on the injected pool.
- ``rate_other_clean``: flag rate on all other cells, in the clean fit.

With real data, each recording of the animal gives one pool, built by ``adr.build_fit_epochs`` as
``rejection_rates.py`` builds it; ``--fit-seed`` seeds the subsample of a recording longer than
``--max-fit-fragments`` and nothing else. ``--temp-dir`` sets the temporary directory after the dataset is
prepared, which sets the dataset config's ``temp_directory``, and before the recordings load, so a run can
reuse the intermediates of a chosen directory. The animal is checked as ``rejection_rates.py --animal``
checks it: a labeled animal is refused unless ``--allow-labeled`` is given, and one the pipeline's quality
filter drops logs a warning.

``<out>.provenance.json`` beside ``--out`` is ``score_detectors.provenance`` of the run: the files read by
sha256 (with real data the cohort manifest and the config files the dataset loads, and always the two
scripts), the git commit, the parsed arguments, the configuration names and, with real data, the selection
as ``rejection_rates.py`` records it. A run that measures no pool stops before writing anything, so an empty
output never replaces an earlier one.

Run from the repository root, on the cluster::

    uv run python scripts/labeling/inject_artifacts.py --dataset <dataset> --animal <id> --seed 0 \\
        --out results/labeling/inject_<id>.csv --workers 16

On a synthetic pool, locally::

    uv run python scripts/labeling/inject_artifacts.py --synthetic --out /tmp/inject_synthetic.csv
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cohort_bundle as C  # noqa: E402  (the cohort load seam)
import rejection_rates as RR  # noqa: E402  (fitting, animal selection, the labeled set and output helpers)
import render_context as R  # noqa: E402  (FRAG_S, the labeling window length)
from score_detectors import provenance  # noqa: E402  (one provenance format across the labeling outputs)

sys.path.insert(0, str(REPO / "src"))
from neurodent.core.utils.paths import set_temp_directory  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402

log = logging.getLogger("inject_artifacts")

KINDS = ("clip", "flatline", "step", "broadband", "line60", "drift")
AMPLITUDES = (2.0, 5.0, 10.0, 20.0)
FRACTION = 0.05
SPAN_FRACTION = (0.1, 0.5)
LINE_HZ = 60.0
DRIFT_HZ = 0.5
ROW_COLUMNS = [
    "dataset", "animal", "genotype", "animalday", "config", "kind", "amplitude", "n_injected", "n_other",
    "rate_injected", "rate_injected_clean", "rate_other", "rate_other_clean",
]


# Artifacts


def reference_rms(X) -> np.ndarray:
    """Return each channel's median over fragments of the fragment's RMS about its mean.

    Args:
        X: ``(F, C, T)`` fragments.

    Returns:
        ``(C,)`` reference RMS.
    """
    return np.median(np.asarray(X, dtype=float).std(axis=2), axis=0)


def draw_spans(kind, n_cells, n_samples, rng) -> np.ndarray:
    """Draw one ``[start, stop)`` sample span per cell.

    A span covers 10 to 50 percent of the fragment at a uniform position. A step's span ends at the
    fragment's end. A drift covers the whole fragment and draws nothing.

    Args:
        kind: One of ``KINDS``.
        n_cells: Number of spans.
        n_samples: Samples per fragment.
        rng: A numpy Generator.

    Returns:
        ``(n_cells, 2)`` int spans.
    """
    if kind == "drift":
        return np.tile(np.array([0, n_samples], dtype=int), (n_cells, 1))
    lo = max(1, int(np.ceil(SPAN_FRACTION[0] * n_samples)))
    hi = max(lo, int(np.floor(SPAN_FRACTION[1] * n_samples)))
    lengths = rng.integers(lo, hi + 1, size=n_cells)
    starts = n_samples - lengths if kind == "step" else rng.integers(0, n_samples - lengths + 1)
    return np.stack([starts, starts + lengths], axis=1).astype(int)


def _scaled_to_rms(w, level):
    """Return ``w`` scaled so its RMS is ``level``; zeros when ``w`` is all zero."""
    rms = np.sqrt(np.mean(w ** 2))
    return w * (level / rms) if rms > 0 else np.zeros_like(w)


def inject(X, fs, cells, kind, amplitude, rng, *, ref_rms=None, spans=None) -> np.ndarray:
    """Return a copy of ``X`` with one artifact added at each (fragment, channel) cell.

    Args:
        X: ``(F, C, T)`` fragments in volts.
        fs: Sampling rate in Hz.
        cells: ``(n, 2)`` distinct ``(fragment, channel)`` indices.
        kind: One of ``KINDS`` (see the module docstring).
        amplitude: Multiple of the channel's reference RMS; ignored by ``flatline``.
        rng: A numpy Generator. It draws the spans when ``spans`` is None, then the noise or phases.
        ref_rms: ``(C,)`` reference RMS per channel; by default :func:`reference_rms` of ``X``.
        spans: ``(n, 2)`` ``[start, stop)`` per cell; by default :func:`draw_spans`. A step's span must end at
            the fragment's end and a drift's must cover the whole fragment.

    Returns:
        ``(F, C, T)`` float64 copy of ``X`` with the artifacts added.

    Raises:
        ValueError: On an unknown kind, cells out of range or repeated, spans of the wrong shape or out of
            range, or ``line60`` at a sampling rate of 120 Hz or less.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"expected (fragments, channels, samples), got shape {X.shape}")
    F, C, T = X.shape
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {list(KINDS)}")
    if kind == "line60" and fs <= 2 * LINE_HZ:
        raise ValueError(f"a {LINE_HZ:g} Hz artifact needs a sampling rate above {2 * LINE_HZ:g} Hz, got {fs:g}")
    cells = np.asarray(cells, dtype=int).reshape(-1, 2)
    n = cells.shape[0]
    if n and (cells.min() < 0 or (cells[:, 0] >= F).any() or (cells[:, 1] >= C).any()):
        raise ValueError(f"cells must index fragments below {F} and channels below {C}")
    if np.unique(cells, axis=0).shape[0] != n:
        raise ValueError("cells must be distinct; a repeated cell would get two artifacts")
    ref = reference_rms(X) if ref_rms is None else np.asarray(ref_rms, dtype=float)
    if ref.shape != (C,):
        raise ValueError(f"ref_rms must have one value per channel ({C}), got shape {ref.shape}")
    spans = draw_spans(kind, n, T, rng) if spans is None else np.asarray(spans, dtype=int).reshape(-1, 2)
    if spans.shape[0] != n or (n and ((spans[:, 0] < 0).any() or (spans[:, 1] > T).any()
                                      or (spans[:, 1] <= spans[:, 0]).any())):
        raise ValueError(f"spans must be one [start, stop) per cell with 0 <= start < stop <= {T}")
    if kind == "step" and (spans[:, 1] != T).any():
        raise ValueError("a step lasts to the end of the fragment, so its spans must stop at the last sample")
    if kind == "drift" and ((spans[:, 0] != 0) | (spans[:, 1] != T)).any():
        raise ValueError("a drift covers the whole fragment, so its spans must be [0, n_samples)")

    Y = X.copy()
    t = np.arange(T) / float(fs)
    phases = rng.uniform(0.0, 2 * np.pi, size=n) if kind in ("line60", "drift") else None
    for j in range(n):
        f, c = cells[j]
        s, e = spans[j]
        level = float(amplitude) * ref[c]
        if kind == "clip":
            Y[f, c, s:e] = X[f, c].mean() + level
        elif kind == "flatline":
            Y[f, c, s:e] = X[f, c].mean()
        elif kind == "step":
            Y[f, c, s:e] += level
        elif kind == "broadband":
            Y[f, c, s:e] += _scaled_to_rms(rng.standard_normal(e - s), level)
        elif kind == "line60":
            Y[f, c, s:e] += _scaled_to_rms(np.sin(2 * np.pi * LINE_HZ * t[s:e] + phases[j]), level)
        else:  # drift
            Y[f, c, s:e] += level * np.sin(2 * np.pi * DRIFT_HZ * t[s:e] + phases[j])
    return Y


def synthetic_pool(n_fragments=200, n_channels=4, fs=250.0, fragment_len_s=5.0, seed=0) -> np.ndarray:
    """Return a synthetic clean pool: red noise with a different gain per channel.

    Each channel is a first-order autoregressive process (coefficient 0.95) scaled to 50 uV RMS times a
    seeded gain between 0.5 and 2, cut into fragments.

    Args:
        n_fragments: Fragments in the pool.
        n_channels: Channels in the pool.
        fs: Sampling rate in Hz.
        fragment_len_s: Fragment length in seconds.
        seed: Seed of the noise and the gains.

    Returns:
        ``(n_fragments, n_channels, T)`` volts, with ``T = round(fs * fragment_len_s)``.
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    T = int(round(fs * fragment_len_s))
    x = lfilter([1.0], [1.0, -0.95], rng.standard_normal((n_channels, n_fragments * T)), axis=1)
    x *= 50e-6 / x.std(axis=1, keepdims=True) * rng.uniform(0.5, 2.0, size=(n_channels, 1))
    return np.ascontiguousarray(x.reshape(n_channels, n_fragments, T).transpose(1, 0, 2))


# The experiment


def canonical_kinds(kinds) -> list:
    """Return the requested kinds in the order of ``KINDS``, without repeats.

    Args:
        kinds: Kind names, each one of ``KINDS``.

    Returns:
        A list of kinds.

    Raises:
        ValueError: On an unknown kind.
    """
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown kind(s) {unknown}; choose from {list(KINDS)}")
    return [k for k in KINDS if k in set(kinds)]


def draw_cells(n_fragments, n_channels, fraction, rng) -> np.ndarray:
    """Draw the cells of one pool: ``fraction`` of every channel's fragments, one set for every kind.

    Each channel takes its own seeded draw of fragments, without repeats. Nothing about the kinds enters the
    draw, so the cells do not depend on which kinds are requested.

    Args:
        n_fragments: Fragments in the pool.
        n_channels: Channels in the pool.
        fraction: Share of each channel's fragments; at least one fragment, and at least one left clean.
        rng: A numpy Generator.

    Returns:
        ``(k * n_channels, 2)`` ``(fragment, channel)`` cells, channel by channel with fragments sorted, where
        ``k = max(1, round(fraction * n_fragments))``.

    Raises:
        ValueError: If ``k`` leaves no fragment of a channel clean.
    """
    k = max(1, int(round(fraction * n_fragments)))
    if k >= n_fragments:
        raise ValueError(f"{k} of {n_fragments} fragments per channel leaves none clean; lower the fraction or use "
                         "a larger pool")
    return np.concatenate([np.stack([np.sort(rng.permutation(n_fragments)[:k]), np.full(k, c)], axis=1)
                           for c in range(n_channels)], axis=0)


def _rate(mask, sel):
    """Flag rate of ``mask`` over the cells in ``sel``, NaN when ``sel`` is empty."""
    n = int(sel.sum())
    return float(mask[sel].sum()) / n if n else float("nan")


def injection_rows(clean_masks, injected_masks, injected, keys) -> list:
    """Compare flags on the injected and the other cells between the clean and the injected fit.

    Args:
        clean_masks: ``{config: (F, C) bool}`` flags of the fit on the clean pool.
        injected_masks: ``{config: (F, C) bool}`` flags of the fit on the injected pool.
        injected: ``(F, C)`` bool, True on the injected cells.
        keys: Columns copied into every row.

    Returns:
        One row per configuration of ``clean_masks``.
    """
    injected = np.asarray(injected, dtype=bool)
    other = ~injected
    rows = []
    for name, clean in clean_masks.items():
        after = injected_masks[name]
        rows.append({**keys, "config": name, "n_injected": int(injected.sum()), "n_other": int(other.sum()),
                     "rate_injected": _rate(after, injected), "rate_injected_clean": _rate(clean, injected),
                     "rate_other": _rate(after, other), "rate_other_clean": _rate(clean, other)})
    return rows


def run_experiment(X, fs, configs, *, kinds=KINDS, amplitudes=AMPLITUDES, fraction=FRACTION, seed=0,
                   parallel=False, keys=None) -> list:
    """Fit on the clean pool, then on one injected copy per kind and amplitude, and compare the flags.

    One set of cells (:func:`draw_cells`, from ``seed`` alone) takes every kind and amplitude, and each kind's
    spans, noise and phases come from ``seed`` and the kind alone, so a kind's rows do not depend on which
    other kinds or amplitudes are requested. Every fit keeps the detector's default split seed
    (``RR.fit_requested``).

    Args:
        X: ``(F, C, T)`` clean pool in volts.
        fs: Sampling rate in Hz.
        configs: Names from ``adr.grid_config_names()``.
        kinds: Kinds of artifact.
        amplitudes: Multiples of the reference RMS; ``flatline`` runs once, with amplitude NaN in its rows.
        fraction: Share of each channel's fragments injected, the same cells for every kind and amplitude.
        seed: Seed of the cells, spans, noise and phases.
        parallel: Fan spectral work out over dask.
        keys: Columns copied into every row.

    Returns:
        Rows as described in the module docstring.
    """
    X = np.asarray(X, dtype=float)
    F, C, _ = X.shape
    kinds = canonical_kinds(kinds)
    configs = RR.resolve_config_names(configs)
    ref = reference_rms(X)
    if (ref == 0).any():
        log.warning(f"channel(s) {np.flatnonzero(ref == 0).tolist()} are flat, so their artifacts other than "
                    "flatline have zero amplitude")
    clean = RR.fit_requested(X, fs, configs, parallel=parallel)
    clean_masks = {name: fit.mask for name, fit in clean.items()}
    cells = draw_cells(F, C, fraction, np.random.default_rng([seed, 2]))  # [seed, 0] would equal seed
    injected = np.zeros((F, C), dtype=bool)
    injected[cells[:, 0], cells[:, 1]] = True
    rows = []
    for kind in kinds:
        for amplitude in ([float("nan")] if kind == "flatline" else [float(a) for a in amplitudes]):
            t0 = time.time()
            # The same Generator seed for every amplitude of a kind reproduces its spans, noise and phases.
            rng = np.random.default_rng([seed, 1, KINDS.index(kind)])
            Xi = inject(X, fs, cells, kind, 0.0 if kind == "flatline" else amplitude, rng, ref_rms=ref)
            fits = RR.fit_requested(Xi, fs, configs, parallel=parallel)
            rows += injection_rows(clean_masks, {name: fit.mask for name, fit in fits.items()}, injected,
                                   {**(keys or {}), "kind": kind, "amplitude": amplitude})
            log.info(f"{kind} at {amplitude:g}: {int(injected.sum())} cells, {len(configs)} configurations, "
                     f"{time.time() - t0:.1f} s")
    return rows


def main(argv=None):
    """Parse command-line arguments, run the experiment on each pool and write the rows.

    Args:
        argv: Arguments to parse; by default ``sys.argv[1:]``.

    Returns:
        Process exit code, 0 on success.

    Raises:
        SystemExit: On bad or conflicting arguments, a refused animal, or no pool to measure.
        RuntimeError: If an animal's recordings and animaldays do not line up.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="use a synthetic pool; reads no recordings")
    source.add_argument("--dataset", help="dataset name, as in config/datasets/<name>.yaml")
    ap.add_argument("--animal", help="animal id within --dataset")
    ap.add_argument("--recording", type=int, nargs="+", default=None,
                    help="indices of the animal's recordings to use (default: all)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the cells, spans, noise and phases, and of the synthetic pool (default 0)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="configuration names from adr.grid_config_names() (default: the whole grid)")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), help=f"kinds of artifact (default: {' '.join(KINDS)})")
    ap.add_argument("--amplitudes", nargs="+", type=float, default=list(AMPLITUDES),
                    help="multiples of the channel's reference RMS (default: 2 5 10 20)")
    ap.add_argument("--fraction", type=float, default=FRACTION,
                    help=f"share of each channel's fragments injected, the same cells for every kind and amplitude "
                         f"(default {FRACTION})")
    ap.add_argument("--fragment-len-s", type=float, default=R.FRAG_S, help=f"fragment length (default {R.FRAG_S})")
    ap.add_argument("--max-fit-fragments", type=int, default=None,
                    help=f"pool cap per recording (default {adr.MAX_FIT_FRAGMENTS}); 0 uses the whole recording")
    ap.add_argument("--fit-seed", type=int, default=None,
                    help="seed of the fit-pool subsample of a recording longer than --max-fit-fragments (default 0); "
                         "the cutoff search keeps the detector's default split seed")
    ap.add_argument("--n-fragments", type=int, default=None, help="synthetic pool: fragments (default 200)")
    ap.add_argument("--n-channels", type=int, default=None, help="synthetic pool: channels (default 4)")
    ap.add_argument("--fs", type=float, default=None, help="synthetic pool: sampling rate in Hz (default 250)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output CSV; <out>.provenance.json is written beside it")
    ap.add_argument("--allow-labeled", action="store_true", help="accept a labeled animal")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="cohort manifest listing the labeled animals (default: config/labeling/cohort_manifest.json)")
    ap.add_argument("--workers", type=int, default=0,
                    help="dask worker processes for the spectral work (default 0: none, run in this process)")
    ap.add_argument("--temp-dir", type=Path, default=None,
                    help="temporary directory to load intermediates from, instead of the dataset config's "
                         "temp_directory")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not 0 < args.fraction < 1:
        ap.error("--fraction must be between 0 and 1")
    real_only = {"--animal": args.animal, "--recording": args.recording, "--max-fit-fragments": args.max_fit_fragments,
                 "--fit-seed": args.fit_seed, "--temp-dir": args.temp_dir, "--manifest": args.manifest,
                 "--allow-labeled": args.allow_labeled or None}
    synthetic_only = {"--n-fragments": args.n_fragments, "--n-channels": args.n_channels, "--fs": args.fs}
    given = [k for k, v in (real_only if args.synthetic else synthetic_only).items() if v is not None]
    if given:
        ap.error(f"{' '.join(given)} {'do' if len(given) > 1 else 'does'} not apply with "
                 f"{'--synthetic' if args.synthetic else '--dataset'}")
    # Defaults are filled after the check, so an explicit value can be told from a default.
    args.max_fit_fragments = adr.MAX_FIT_FRAGMENTS if args.max_fit_fragments is None else args.max_fit_fragments
    args.fit_seed = 0 if args.fit_seed is None else args.fit_seed
    args.manifest = RR.MANIFEST if args.manifest is None else args.manifest
    args.n_fragments = 200 if args.n_fragments is None else args.n_fragments
    args.n_channels = 4 if args.n_channels is None else args.n_channels
    args.fs = 250.0 if args.fs is None else args.fs
    try:
        configs = RR.resolve_config_names(args.configs)
        kinds = canonical_kinds(args.kinds)
    except ValueError as e:
        ap.error(str(e))
    common = dict(kinds=kinds, amplitudes=args.amplitudes, fraction=args.fraction, seed=args.seed,
                  parallel=args.workers > 0)
    inputs = {"script_inject_artifacts": Path(__file__).resolve(),
              "script_rejection_rates": Path(RR.__file__).resolve()}
    selection = None
    if not args.synthetic:
        if not args.animal:
            ap.error("--dataset needs --animal")
        C._require_repo_root()
        try:
            [(ds, [(animal_id, genotype)])], selection = RR.select_animals(
                [args.dataset], animals=[args.animal], labeled=RR.labeled_animals(args.manifest),
                allow_labeled=args.allow_labeled)
            inputs.update(RR.cohort_inputs(args.manifest, [ds]))
        except (ValueError, FileNotFoundError) as e:
            ap.error(str(e))
    # Serialized before the long run, so a record that cannot be written stops the run before it starts.
    prov = json.dumps(provenance(inputs, script="inject_artifacts", **RR.cli_params(args), config_names=configs,
                                 selection=selection), indent=2)

    rows = []
    with RR.dask_workers(args.workers):
        if args.synthetic:
            X = synthetic_pool(args.n_fragments, args.n_channels, args.fs, args.fragment_len_s, seed=args.seed)
            keys = {"dataset": "synthetic", "animal": "synthetic", "genotype": "", "animalday": "synthetic"}
            rows += run_experiment(X, args.fs, configs, keys=keys, **common)
        else:
            config, samples_config = C._prepare(ds)
            if args.temp_dir is not None:
                set_temp_directory(args.temp_dir)  # after _prepare, which set the config's directory
            ao = C.load_cohort_animal(samples_config, config, animal_id)
            if len(ao.animaldays) != len(ao.long_recordings):
                raise RuntimeError(f"{animal_id}: {len(ao.long_recordings)} recordings but "
                                   f"{len(ao.animaldays)} animaldays")
            chosen = (list(range(len(ao.long_recordings))) if args.recording is None
                      else list(dict.fromkeys(args.recording)))
            missing = [i for i in chosen if not 0 <= i < len(ao.long_recordings)]
            if missing:
                ap.error(f"{animal_id} has {len(ao.long_recordings)} recordings; no index {missing}")
            cap = args.max_fit_fragments if args.max_fit_fragments > 0 else None
            for i in chosen:
                pool = adr.build_fit_epochs(ao.long_recordings[i], None, fragment_len_s=args.fragment_len_s,
                                            max_fit_fragments=cap, seed=args.fit_seed)
                keys = {"dataset": ds, "animal": animal_id, "genotype": genotype, "animalday": ao.animaldays[i]}
                log.info(f"{animal_id} {ao.animaldays[i]}: pool of {pool.X.shape[0]} fragments x "
                         f"{pool.X.shape[1]} channels")
                rows += run_experiment(pool.X, pool.fs, configs, keys=keys, **common)
    if not rows:
        raise SystemExit("no pool was measured, since there is no recording to inject into; nothing was written")
    RR.write_csv(pd.DataFrame(rows, columns=ROW_COLUMNS), args.out)
    RR.write_text(prov, RR.provenance_path(args.out))
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote the provenance to {RR.provenance_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
