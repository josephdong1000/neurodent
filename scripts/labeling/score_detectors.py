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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cohort_bundle as C  # noqa: E402  (shared cohort load seam: no drift vs the bundle)
import render_context as R  # noqa: E402  (FRAG_S — single source of the window size)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.results.scoring import consensus, ingest, unblind, score_keep_mask  # noqa: E402

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
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="match the bundle's --limit-per-dataset so the same animals are loaded")
    ap.add_argument("--seed", type=int, default=0, help="match the bundle's --seed for --limit selection")
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

    all_rows = []
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
            # window_s pinned to the labeling FRAG_S; notch on -> identical signal to the rendered windows.
            war = az.compute_windowed_analysis(FEATURES, window_s=R.FRAG_S,
                                               apply_notch_filter=True, multiprocess_mode="serial")
            if args.lof:
                lof_dict = war.get_bad_channels_by_lof_threshold(args.lof_threshold)
                detectors = {**FRAGMENT_DETECTORS,
                             "lof": ("get_filter_reject_channels_by_recording_session",
                                     {"bad_channels_dict": lof_dict})}
            all_rows += score_animal(war, ao, animal_id, cons, detectors)
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
    print(f"\n{'detector':<16}{'n':>7}{'uncov':>7}{'prec':>8}{'recall':>8}{'f1':>8}   confusion[tn fp / fn tp]")
    print("-" * 78)
    for name in detectors:
        sub = df[df["detector"] == name]
        conf = np.zeros((2, 2), dtype=int)
        for c in sub["confusion"].dropna():
            conf += np.array(c, dtype=int)
        tn, fp, fn, tp = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
        p, r, f1 = _prf(tn, fp, fn, tp)
        n = int(sub["n"].fillna(0).sum())
        uncov = int(sub["n_uncovered"].fillna(0).sum())
        print(f"{name:<16}{n:>7}{uncov:>7}{p:>8.3f}{r:>8.3f}{f1:>8.3f}   [{tn} {fp} / {fn} {tp}]")
    print("\nSingle-rater dogfood: no interrater kappa ceiling yet — smoke read, not calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
