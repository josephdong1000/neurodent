#!/usr/bin/env python
"""Extract per-cell detector feature scores for the DEV animals — the leak-free input to select_detectors.py.

Loads each animal ONCE, builds the WAR (rms, psdband, psdtotal) + LOF scores, aligns to the consensus labels
the SAME way `scoring.score_mask` does (time-match within FRAG_S/2), and writes a tidy per-(recording, window,
channel) table with the raw scores the filter/LOF sweeps need: `rms`, `beta_prop`, `lof`, plus the label `y`.
`select_detectors.py` then sweeps thresholds + ranks offline (no reload).

GATED by `results/labeling/split.json`: refuses to run if the labels contain any animal outside the chosen
partition (`--which dev|test`) — structural protection against scoring the sealed test by accident.

    sbatch --cpus-per-task=32 --mem=100G --time=6:00:00 --wrap="uv run python scripts/labeling/extract_dev_features.py \\
        --which dev --out results/labeling/dev_features_JD.parquet 'JD=labels_cohort4strains_JD(1)_dev.csv'"
        # --keymap defaults to config/labeling/keymap.csv (committed); --out under results/ is disposable
"""
import argparse
import json
import logging
import os
import sys
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
from score_detectors import build_consensus, FEATURES  # noqa: E402 (reuse the exact consensus + feature set)
from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("extract_dev_features")


def beta_prop_grid(df_stats):
    """(W, C) beta-power proportion = psdband['beta'] / psdtotal — the raw score `high_beta` thresholds."""
    band = pd.DataFrame(df_stats["psdband"].tolist())
    beta = np.array(band["beta"].tolist())
    total = np.array(df_stats["psdtotal"].tolist())
    return beta / total


def align_cells(rms, logrmsz, beta_wmax, lof_by_ch, ch_names, cons_rec, grid_times, frag_s=R.FRAG_S):
    """Per labelled cell → the per-detector score it is thresholded on, matched to the WAR row by start-time
    (mirrors score_mask). Filter transforms that need the FULL grid (logrms z over all windows; high_beta's
    per-window max over all channels) are computed by the caller, so the export is directly sweep-ready."""
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
        out.append({"recording": r["recording"], "window": int(r["window"]), "channel": ch,
                    "t_start_s": t, "y": int(r["y_true"]),
                    "rms": float(rms[row, ci]),                    # high_rms (>τ) / low_rms (<τ)
                    "logrmsz_abs": float(abs(logrmsz[row, ci])),   # logrms_range (>τ)
                    "beta_wmax": float(beta_wmax[row]),            # high_beta: window rejected if any ch >τ
                    "lof": float(lof_by_ch.get(ch, np.nan))})      # lof (>τ), per-channel score
    return out


def main():
    ap = argparse.ArgumentParser(description="Export per-cell filter/LOF scores for the split partition.")
    ap.add_argument("raters", nargs="+", metavar="rater=csv")
    ap.add_argument("--keymap", type=Path, default=REPO / "config/labeling/keymap.csv",
                    help="slot->channel unblinding key (default: committed config/labeling/keymap.csv)")
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--which", choices=["dev", "test"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rule", default="majority", choices=["majority", "unanimous", "any"])
    ap.add_argument("--lof-chunk-duration-s", type=float, default=60)
    ap.add_argument("--chunk-duration-s", type=float, default=3600)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    manifests = {}
    for spec in args.raters:
        if "=" not in spec:
            ap.error(f"expected rater=csv, got {spec!r}")
        rater, path = spec.split("=", 1)
        manifests[rater] = Path(path).expanduser()
    cons = build_consensus(manifests, args.keymap, args.rule)
    animals_wanted = {r.rsplit("__", 1)[0] for r in cons["recording"].unique()}

    # ---- the split GATE: labels must live entirely within the chosen partition ----
    split = json.loads(args.split.read_text())
    allowed = set(split[args.which])
    leaked = animals_wanted - allowed
    if leaked:
        raise SystemExit(f"GATE FAILED: labels include animals outside split['{args.which}']: {sorted(leaked)}. "
                         "Refusing — this would mix the partitions. Pass the matching _dev/_test CSV.")
    print(f"gate OK: {len(animals_wanted)} labelled animals ⊆ split['{args.which}'] ({len(allowed)})")

    n_workers = args.workers or len(os.sched_getaffinity(0))
    rows = []
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        try:
            client.wait_for_workers(n_workers, timeout=180)
        except Exception as e:
            log.warning(f"workers: {type(e).__name__}; proceeding with {len(client.scheduler_info()['workers'])}")
        for ds, samples_config, config, animal_id in C.iter_cohort_animals(C.REAL_STRAINS, limit_per_dataset=None, seed=0):
            if animal_id not in animals_wanted:
                continue
            try:
                ao = C.load_cohort_animal(samples_config, config, animal_id)
                az = AnimalAnalyzer(ao)
                az.compute_bad_channels(lof_threshold=1.0, lof_chunk_duration_s=args.lof_chunk_duration_s)  # scores, not the cut
                war = az.compute_windowed_analysis(FEATURES, window_s=R.FRAG_S, apply_notch_filter=True,
                                                   multiprocess_mode="dask", chunk_duration_s=args.chunk_duration_s)
                ch_names = list(war.channel_abbrevs)
                animaldays = war.result["animalday"].to_numpy()
                rms_all = extract_linear_array(war.result["rms"])
                beta_all = beta_prop_grid(war.result)
                for i, animalday in enumerate(ao.animaldays):
                    recording = f"{animal_id}__{i}"
                    cons_rec = cons[cons["recording"] == recording]
                    if cons_rec.empty:
                        continue
                    rowsel = animaldays == animalday
                    n = int(rowsel.sum())
                    if n == 0:
                        log.warning(f"{recording}: no WAR fragments for animalday {animalday!r}; skipped")
                        continue
                    grid_times = np.arange(n) * R.FRAG_S
                    lof_by_ch = war.lof_scores_dict.get(animalday, {})
                    rms_rec = rms_all[rowsel]
                    logrmsz = zscore(np.log(rms_rec), axis=0, nan_policy="omit")   # over ALL windows (matches the filter)
                    beta_wmax = beta_all[rowsel].max(axis=1)                        # over ALL channels (matches the filter)
                    rows += align_cells(rms_rec, logrmsz, beta_wmax, lof_by_ch, ch_names, cons_rec, grid_times)
                log.info(f"{animal_id}: extracted")
            except Exception as e:
                log.error(f"{animal_id}: FAILED -- {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("no cells extracted (no overlap between cohort and consensus).")
    df = pd.DataFrame(rows)
    df["animal"] = df["recording"].str.rsplit("__", n=1).str[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} cells over {df['animal'].nunique()} animals, "
          f"{df['recording'].nunique()} recordings, {int(df['y'].sum())} artifact ({100*df['y'].mean():.1f}%) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
