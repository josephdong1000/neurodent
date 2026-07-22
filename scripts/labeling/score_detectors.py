#!/usr/bin/env python
"""Score artifact detectors against the human labels from a cohort bundle.

Feeds the rater CSV(s) through the SAME path as ``score_bundle.py`` (``ingest`` -> ``unblind`` ->
``consensus``) to build ground truth, then re-derives each detector's per-``(window, channel)`` decision
on the *same recordings the rater saw* and reports precision/recall/F1 at ``(recording, window, channel)``
granularity.

**No drift, by construction.** Recordings are loaded via the shared
``build_cohort_bundle.iter_cohort_animals`` / ``load_cohort_animal`` seam — the identical
``load_animal_recordings`` + ``channel_subset`` the bundle used. Features come from
``AnimalAnalyzer.compute_windowed_analysis(window_s=FRAG_S, apply_notch_filter=True)`` — the exact
WAR-generation path (``workflow/scripts/generate_wars.py``), which internally builds the same
``LongRecordingAnalyzer(fragment_len_s=FRAG_S, apply_notch_filter=True)`` the labeling renderer used. So
the detector is scored on the identical signal the rater judged, *and* on the identical features the
production pipeline's detectors run on. ``window_s`` is pinned to ``render_context.FRAG_S`` so the two can
never silently diverge.

Alignment: a labelled ``window`` is a direct ``FRAG_S``-second fragment index (``t_start_s = window *
FRAG_S``); ``recording = "{animal}__{i}"`` is the i-th LRO, whose WAR rows are the ``animalday ==
ao.animaldays[i]`` slice. ``score_keep_mask`` matches each labelled cell to the detector row whose start
time (``grid_times = arange(n) * FRAG_S``) is within ``FRAG_S/2`` — so ``n_uncovered`` should be ~0.

**DOGFOOD caveat:** with a single rater there is no interrater kappa ceiling. This is a smoke read of how
the current thresholds track one human's labels, NOT calibration. Multi-rater calibration is step 4.

Heavy load (feature extraction over full recordings) -> run on the cluster::

    uv run python scripts/labeling/score_detectors.py \
        --all --keymap results/labeling/mixed/_unblind/keymap.csv \
        JD=labels_cohort4strains_JD.csv
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cohort_bundle as C  # noqa: E402  (shared cohort load seam: no drift vs the bundle)
import render_context as R  # noqa: E402  (FRAG_S — single source of the window size)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.analysis.long_recording_analyzer import LongRecordingAnalyzer  # noqa: E402
from neurodent.core.utils import resolve_channels  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.scoring import consensus, ingest, score_keep_mask, score_mask, unblind  # noqa: E402

log = logging.getLogger("score_detectors")

# Fragment (temporal) detectors previewed here: detector -> (WAR keep-mask accessor, default params
# from config/config.yaml fragment_filter_config). Each accessor returns a (W, C) True=KEEP mask over
# the whole WAR; we slice it per recording below. LOF is spatial and opt-in (needs compute_bad_channels).
FRAGMENT_DETECTORS = {
    "logrms_range": ("get_filter_logrms_range", {"z_range": 3}),
    "high_rms": ("get_filter_high_rms", {"max_rms": 500}),
    "low_rms": ("get_filter_low_rms", {"min_rms": 50}),
    "high_beta": ("get_filter_high_beta", {"max_beta_prop": 0.4}),
}
# Enough features for the four fragment filters (rms drives logrms/high/low_rms; psdband+psdtotal drive
# high_beta). Deliberately NOT "all" — no coherence/correlation matrices needed, so extraction stays cheap.
FEATURES = ["rms", "psdband", "psdtotal"]


def _prf(tn, fp, fn, tp):
    """Precision/recall/F1 from a pooled confusion, zero-safe."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score_animal(war, ao, animal_id, consensus_df, detectors=FRAGMENT_DETECTORS):
    """Score each detector on one animal's WAR against the consensus, per recording.

    Returns a list of per-(recording, detector) dicts. ``war.channel_abbrevs`` are the canonical channel
    names the consensus is keyed on (``score_keep_mask`` matches by name, so column order is irrelevant).
    Recording ``"{animal_id}__{i}"`` is the ``animalday == ao.animaldays[i]`` slice — the SAME naming the
    bundle used — scored with ``grid_times`` = the per-slice fragment start seconds, so labelled window
    ``w`` lines up with detector row ``w``.
    """
    ch_names = list(war.channel_abbrevs)
    animaldays = war.result["animalday"].to_numpy()
    recs = set(consensus_df["recording"])
    masks = {name: getattr(war, method)(**params) for name, (method, params) in detectors.items()}

    rows = []
    for i, animalday in enumerate(ao.animaldays):
        recording = f"{animal_id}__{i}"
        if recording not in recs:
            continue
        rowsel = animaldays == animalday
        n = int(rowsel.sum())
        if n == 0:
            log.warning(f"{recording}: no WAR fragments for animalday {animalday!r}; skipped")
            continue
        grid_times = np.arange(n) * R.FRAG_S
        for name, keep in masks.items():
            res = score_keep_mask(keep[rowsel], ch_names, consensus_df, recording,
                                  grid_times=grid_times, frag_s=R.FRAG_S, strict=False)
            rows.append({"recording": recording, "detector": name, **res})
    return rows


def autoreject_grids(lro, needed_windows=None, *, max_fit_fragments=5000, seed=0, parallel=True):
    """Run the 6 adapted-autoreject arms on ONE recording (LRO).

    Builds 5 s epochs from the LRO's own fragments (the SAME ``LongRecordingAnalyzer(fragment_len_s=FRAG_S,
    apply_notch_filter=True)`` the labeling used → identical signal), then fits+applies the leak-free CV per
    :mod:`neurodent.results.autoreject_detector`. Returns ``({config: (n_pool, C) REJECT grid}, ch_names,
    grid_times)`` aligned by ``grid_times`` so :func:`score_mask` picks the labelled fragments.

    Fit pool = the labelled fragments (always included) + a capped random subsample of the rest
    (``max_fit_fragments``, default 5000) — the CV threshold needs only a representative, mostly-clean pool,
    so this bounds memory/compute on 24 h recordings (~17k fragments). Pass ``max_fit_fragments=None`` to fit
    on the whole recording.
    """
    lan = LongRecordingAnalyzer(lro, fragment_len_s=R.FRAG_S, apply_notch_filter=True)
    idxs = adr.fit_pool_indices(lan.n_fragments, needed_windows, max_fit_fragments, seed)
    # (n_pool, C, T) volts; get_fragment_np is (n_samples, C) µV. The last fragment is usually ragged
    # (recording length not an exact multiple of FRAG_S); skip any non-full fragment so the stack and the
    # LPSD bin count stay uniform. Labelled windows are never edge fragments (the labeler reserves flanks).
    win = int(round(R.FRAG_S * lan.f_s))
    frags, keep = [], []
    for i in idxs:
        f = lan.get_fragment_np(int(i))                     # (n_samples, C) µV
        if f.shape[0] != win:
            continue
        frags.append(f.T * 1e-6)
        keep.append(int(i))
    if not frags:
        raise ValueError(f"no full-length fragments (win={win} samples) to fit on")
    X = np.stack(frags, axis=0)
    idxs = np.array(keep, dtype=int)
    ch_names = resolve_channels(list(lan.channel_names))
    grids = adr.compute_masks(X, int(lan.f_s), parallel=parallel)   # {config: (n_pool, C) True=REJECT}
    grid_times = idxs.astype(float) * R.FRAG_S
    return grids, ch_names, grid_times


def score_animal_autoreject(ao, animal_id, consensus_df, *, max_fit_fragments=5000, seed=0):
    """Score the 6 autoreject arms per recording (per LRO) against the consensus via ``score_mask``.

    Autoreject is naturally PER-LRO (fit on that recording's fragments), so unlike the WAR filters it does
    not slice a whole-animal mask; each LRO yields its own reject grids. The grids are ``True = REJECT`` so
    they go through :func:`score_mask` directly (not ``score_keep_mask``).
    """
    recs = set(consensus_df["recording"])
    rows = []
    for i, lro in enumerate(ao.long_recordings):
        recording = f"{animal_id}__{i}"
        if recording not in recs:
            continue
        needed = consensus_df.loc[consensus_df["recording"] == recording, "window"].unique()
        try:
            grids, ch_names, grid_times = autoreject_grids(
                lro, needed, max_fit_fragments=max_fit_fragments, seed=seed)
        except Exception as e:
            log.error(f"{recording}: autoreject FAILED -- {type(e).__name__}: {e}")
            continue
        for name, grid in grids.items():
            res = score_mask(grid, ch_names, consensus_df, recording,
                             grid_times=grid_times, frag_s=R.FRAG_S, strict=False)
            rows.append({"recording": recording, "detector": name, **res})
    return rows


def build_consensus(manifests, keymap_path, rule):
    """rater CSVs + keymap -> consensus ground truth (the score_bundle.py round-trip)."""
    long_df = ingest(manifests)
    keymap = pd.read_csv(keymap_path)
    restored = unblind(long_df, keymap)
    return consensus(restored, rule=rule)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raters", nargs="+", metavar="rater=csv", help="one or more rater_id=exported_csv")
    ap.add_argument("--keymap", type=Path, required=True, help="path to _unblind/keymap.csv")
    ap.add_argument("--dataset", nargs="+", default=None,
                    help="dataset name(s) whose animals to score (config/datasets/<name>.yaml)")
    ap.add_argument("--all", action="store_true",
                    help=f"scan all real strains ({' '.join(C.REAL_STRAINS)})")
    ap.add_argument("--rule", default="majority", choices=["majority", "unanimous", "any"])
    ap.add_argument("--lof", action="store_true",
                    help="also score the LOF bad-channel detector (needs compute_bad_channels; slower)")
    ap.add_argument("--lof-threshold", type=float, default=2.5)
    ap.add_argument("--autoreject", action="store_true",
                    help="also score the 6 adapted-autoreject arms (amp/v2/v3 x +/-CMR); builds raw epochs "
                         "per recording + LPSD (slower)")
    ap.add_argument("--ar-fit-max-fragments", type=int, default=5000,
                    help="cap on the autoreject fit pool per recording (labelled fragments always included; "
                         "0/negative -> fit on the whole recording)")
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="match the bundle's --limit-per-dataset so the same animals are loaded")
    ap.add_argument("--seed", type=int, default=0, help="match the bundle's --seed for --limit selection")
    ap.add_argument("--workers", type=int, default=None,
                    help="dask process workers for feature extraction (default: all cores allocated to "
                         "the job). Feature extraction over a full recording is the cost -- give this job "
                         "many cores.")
    ap.add_argument("--chunk-duration-s", type=float, default=3600,
                    help="seconds of data buffered per chunk during dask extraction (matches the pipeline; "
                         "raise on a high-memory node for throughput)")
    ap.add_argument("--out", type=Path, default=None, help="write per-(recording, detector) rows to CSV")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    C._require_repo_root()

    datasets = list(C.REAL_STRAINS) if args.all else args.dataset
    if not datasets:
        ap.error("provide --dataset <name ...> or --all")

    manifests = {}
    for spec in args.raters:
        if "=" not in spec:
            ap.error(f"expected rater=csv, got {spec!r}")
        rater, path = spec.split("=", 1)
        p = Path(path).expanduser()
        if not p.exists():
            ap.error(f"{rater}: no such file {p}")
        manifests[rater] = p
    if not args.keymap.exists():
        ap.error(f"keymap not found: {args.keymap}")

    cons = build_consensus(manifests, args.keymap, args.rule)
    animals_wanted = {r.rsplit("__", 1)[0] for r in cons["recording"].unique()}
    print(f"consensus: {len(cons)} labelled cells over {cons['recording'].nunique()} recordings "
          f"({len(animals_wanted)} animals); raters={list(manifests)}")

    detectors = dict(FRAGMENT_DETECTORS)

    # Feature extraction over full recordings is the cost (a big recording is O(10k) fragments). Serial is
    # hopelessly slow (~1.6 s/fragment -> hours per animal); the pipeline parallelizes with dask, so we do
    # the same. Dashboard is off: we care that the analysis RUNS, not about scheduler introspection.
    n_workers = args.workers or len(os.sched_getaffinity(0))
    log.info(f"starting dask LocalCluster with {n_workers} process workers (dashboard off)")

    all_rows = []
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        # Workers are subprocesses that each import spikeinterface/neurodent (~15 s), so they register
        # gradually. Block for the full pool before any compute, else the first animal's dask.compute
        # fans out across only however few had registered at that instant.
        try:
            client.wait_for_workers(n_workers, timeout=180)
        except Exception as e:
            log.warning(f"only {len(client.scheduler_info()['workers'])}/{n_workers} workers up after "
                        f"180 s ({type(e).__name__}); proceeding with what registered")
        log.info(f"dask cluster up: {len(client.scheduler_info()['workers'])} workers")
        for ds, samples_config, config, animal_id in C.iter_cohort_animals(
            datasets, limit_per_dataset=args.limit_per_dataset, seed=args.seed
        ):
            if animal_id not in animals_wanted:
                continue
            try:
                ao = C.load_cohort_animal(samples_config, config, animal_id)
                az = AnimalAnalyzer(ao)
                if args.lof:
                    az.compute_bad_channels(lof_threshold=args.lof_threshold, lof_chunk_duration_s=60)
                # window_s pinned to the labeling FRAG_S; notch on -> identical signal to the rendered
                # windows; dask parallelizes the per-fragment feature extraction across the cluster.
                war = az.compute_windowed_analysis(
                    FEATURES, window_s=R.FRAG_S, apply_notch_filter=True,
                    multiprocess_mode="dask", chunk_duration_s=args.chunk_duration_s)
                if args.lof:
                    lof_dict = war.get_bad_channels_by_lof_threshold(args.lof_threshold)
                    detectors = {**FRAGMENT_DETECTORS,
                                 "lof": ("get_filter_reject_channels_by_recording_session",
                                         {"bad_channels_dict": lof_dict})}
                all_rows += score_animal(war, ao, animal_id, cons, detectors)
                if args.autoreject:
                    cap = args.ar_fit_max_fragments if args.ar_fit_max_fragments > 0 else None
                    all_rows += score_animal_autoreject(ao, animal_id, cons,
                                                        max_fit_fragments=cap, seed=args.seed)
                log.info(f"{animal_id}: scored {len(ao.animaldays)} recording(s)")
            except Exception as e:
                log.error(f"{animal_id}: FAILED -- {type(e).__name__}: {e}")

    if not all_rows:
        print("no recordings scored (no overlap between the cohort and the consensus recordings).")
        return 1

    df = pd.DataFrame(all_rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote per-recording rows -> {args.out}")

    # Pooled per-detector metrics from summed confusions (a cell counts once, regardless of recording).
    order = list(FRAGMENT_DETECTORS) + (["lof"] if args.lof else []) + \
        (adr.CONFIG_NAMES if args.autoreject else [])
    present = [n for n in order if (df["detector"] == n).any()]
    print(f"\n{'detector':<22}{'n':>7}{'uncov':>7}{'prec':>8}{'recall':>8}{'f1':>8}   confusion[tn fp / fn tp]")
    print("-" * 84)
    for name in present:
        sub = df[df["detector"] == name]
        conf = np.zeros((2, 2), dtype=int)
        for c in sub["confusion"].dropna():
            conf += np.array(c, dtype=int)
        tn, fp, fn, tp = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
        p, r, f1 = _prf(tn, fp, fn, tp)
        n = int(sub["n"].fillna(0).sum())
        uncov = int(sub["n_uncovered"].fillna(0).sum())
        print(f"{name:<22}{n:>7}{uncov:>7}{p:>8.3f}{r:>8.3f}{f1:>8.3f}   [{tn} {fp} / {fn} {tp}]")
    print("\nSingle-rater dogfood: no interrater kappa ceiling yet — smoke read, not calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
