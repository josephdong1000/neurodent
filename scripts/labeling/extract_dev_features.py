#!/usr/bin/env python
"""Extract per-cell detector scores for the dev animals, the leak-free input to select_detectors.py.

Loads each animal once, builds the WAR (rms, psdband, psdtotal) + LOF scores, runs the full 74-config adapted
autoreject grid per recording, and aligns everything to the consensus labels the same way `scoring.score_mask`
does (time-match within FRAG_S/2). Writes a tidy per-(recording, window, channel) table: the raw filter/LOF
scores (`rms`, `logrmsz_abs`, `beta_wmax`, `lof`), one 0/1 reject column per autoreject config, and the label
`y`. `select_detectors.py` then sweeps thresholds and ranks every detector offline (no reload), so all
detectors are always scored on the identical cells, apples-to-apples. There is one code path: every run scores
every detector.

Gated by `config/labeling/split.json`: refuses to run if the labels contain any animal outside the chosen
partition (`--which dev|test`), a structural protection against scoring the sealed test by accident.

Atomic, no resume: every run regenerates from scratch and writes `--out` once at the very end (temp then
rename), only after every wanted animal has succeeded. A crash or timeout, or any animal/recording raising,
leaves no output file (never a half-baked parquet to silently consume), so a rerun simply starts clean. The
autoreject grid is the slow phase; give the job cores and a generous wall time:

    sbatch --cpus-per-task=64 --mem=64G --time=24:00:00 --wrap="uv run python \\
        scripts/labeling/extract_dev_features.py --which dev \\
        --out results/labeling/dev_features_JD.parquet 'JD=scripts/labeling/labels/labels_cohort4strains_JD_dev.csv'"

--keymap defaults to the committed key; --out under results/ is disposable (regenerate any time).
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
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("extract_dev_features")

# One row per labelled (recording, window, channel) cell = align_cells keys + animal; the full autoreject grid
# appends one 0/1 reject column per config (adr.grid_config_names()) at write time.
CELL_COLS = ["recording", "window", "channel", "t_start_s", "y",
             "rms", "logrmsz_abs", "beta_wmax", "lof", "animal"]


def beta_prop_grid(df_stats):
    """(W, C) beta-power proportion = psdband['beta'] / psdtotal, the raw score `high_beta` thresholds."""
    band = pd.DataFrame(df_stats["psdband"].tolist())
    beta = np.array(band["beta"].tolist())
    total = np.array(df_stats["psdtotal"].tolist())
    return beta / total


def lof_by_abbrev(war, animalday):
    """{canonical-abbrev channel -> LOF score} for one animalday. ``lof_scores_dict[animalday]`` is
    ``{"lof_scores": [...], "channel_names": [raw...]}`` (parallel lists keyed by raw names), not a
    channel->score map, so we zip the two lists and remap the raw names to the WAR's canonical abbrevs,
    which is what the consensus `channel` column uses. Returns {} if LOF is absent for this animalday."""
    data = war.lof_scores_dict.get(animalday)
    if not data or not data.get("channel_names"):
        return {}
    abbrevs = set(war.channel_abbrevs)
    raw2ab = dict(zip(war.channel_names or [], war.channel_abbrevs or []))
    out = {}
    for raw, score in zip(data["channel_names"], data["lof_scores"]):
        ab = raw if raw in abbrevs else raw2ab.get(raw)   # tolerate names already stored as abbrevs
        if ab is not None:
            out[ab] = float(score)
    return out


def align_cells(rms, logrmsz, beta_wmax, lof_by_ch, ch_names, cons_rec, grid_times, frag_s=R.FRAG_S):
    """Per labelled cell → the per-detector score it is thresholded on, matched to the WAR row by start-time
    (mirrors score_mask). Filter transforms that need the full grid (logrms z over all windows; high_beta's
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


def ar_cell_bits(grids, ar_ch_names, ar_grid_times, cons_rec, frag_s=R.FRAG_S):
    """{(window, channel) -> {config: 0/1 reject}} for the labelled cells, matching each cell's start time to
    the autoreject fit-pool grid (``ar_grid_times``) within frag_s/2 and its channel to ``ar_ch_names``. The
    labelled fragments are always in the fit pool (:func:`fit_pool_indices`), so a match is expected; an
    unmatched cell (edge/ragged fragment) simply gets no autoreject bits (NaN downstream). A config's mask is
    True = reject, i.e. the detector's positive/"bad" call, the same polarity as the filter scores' y."""
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
        ci = ch_idx[ch]
        out[(int(r["window"]), ch)] = {name: int(m[row, ci]) for name, m in grids.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description="Export per-cell filter/LOF + autoreject scores for the split partition.")
    ap.add_argument("raters", nargs="+", metavar="rater=csv", help="rater=csv pairs (>= 1)")
    ap.add_argument("--keymap", type=Path, default=REPO / "config/labeling/keymap.csv",
                    help="slot->channel unblinding key (default: committed config/labeling/keymap.csv)")
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--which", choices=["dev", "test"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rule", default="majority", choices=["majority", "unanimous", "any"])
    ap.add_argument("--lof-chunk-duration-s", type=float, default=60)
    ap.add_argument("--chunk-duration-s", type=float, default=3600)
    ap.add_argument("--ar-max-fit-fragments", type=int, default=2000,
                    help="cap on the autoreject fit pool per recording (labelled fragments always included; "
                         "0 -> fit on the whole recording). Smaller = faster; scored cells are unaffected.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the autoreject fit-pool subsample + cohort iteration; must match "
                         "score_detectors' --seed for byte-identical epochs/masks (both default 0)")
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

    # ---- the split gate: labels must live entirely within the chosen partition ----
    split = json.loads(args.split.read_text())
    allowed = set(split[args.which])
    leaked = animals_wanted - allowed
    if leaked:
        raise SystemExit(f"gate failed: labels include animals outside split['{args.which}']: {sorted(leaked)}. "
                         "Refusing, this would mix the partitions. Pass the matching _dev/_test CSV.")
    print(f"gate ok: {len(animals_wanted)} labelled animals within split['{args.which}'] ({len(allowed)})")

    ar_cols = adr.grid_config_names()            # the full 74-config grid, always scored, apples-to-apples
    n_workers = args.workers or len(os.sched_getaffinity(0))
    rows, done = [], set()                       # accumulate every cell in memory (~3k rows), one atomic write

    # No per-animal try/except, no resume: any failure propagates and the run dies before writing --out, so a
    # crash or timeout leaves no partial parquet. A rerun regenerates from scratch.
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        try:
            client.wait_for_workers(n_workers, timeout=180)
        except Exception as e:
            log.warning(f"workers: {type(e).__name__}; proceeding with {len(client.scheduler_info()['workers'])}")
        for ds, samples_config, config, animal_id in C.iter_cohort_animals(C.REAL_STRAINS, limit_per_dataset=None,
                                                                           seed=args.seed):
            if animal_id not in animals_wanted:
                continue
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
                lof_by_ch = lof_by_abbrev(war, animalday)   # {abbrev -> LOF}; parses the list-pair structure
                rms_rec = rms_all[rowsel]
                logrmsz = zscore(np.log(rms_rec), axis=0, nan_policy="omit")   # over all windows (matches the filter)
                beta_wmax = beta_all[rowsel].max(axis=1)                        # over all channels (matches the filter)
                cells = align_cells(rms_rec, logrmsz, beta_wmax, lof_by_ch, ch_names, cons_rec, grid_times)

                # autoreject always runs, on the same recording and same leak-free consensus cells, apples-to-apples.
                Xar, ar_ch, fs_ar, ar_gt = adr.build_fit_epochs(
                    ao.long_recordings[i], cons_rec["window"].unique(), fragment_len_s=R.FRAG_S,
                    max_fit_fragments=(args.ar_max_fit_fragments or None), seed=args.seed)
                log.info(f"{recording}: autoreject grid, {Xar.shape[0]} fragments x {Xar.shape[1]} ch x "
                         f"{len(ar_cols)} configs (slow phase, no per-config log)")
                grids = adr.compute_masks_grid(Xar, fs_ar)          # {config: (n_pool, C) True = reject}
                log.info(f"{recording}: autoreject grid done")
                bits = ar_cell_bits(grids, ar_ch, ar_gt, cons_rec)
                for row in cells:
                    row["animal"] = animal_id                       # recording is animal_id__i -> constant here
                    row.update(bits.get((row["window"], row["channel"]), {}))   # add the 74 autoreject config bits
                rows += cells
            done.add(animal_id)
            log.info(f"{animal_id}: done ({len(done)}/{len(animals_wanted)} animals)")

    # ---- all-or-nothing: write --out only if every wanted animal completed (else no partial parquet) ----
    missing = animals_wanted - done
    if missing:
        raise SystemExit(f"{len(missing)} wanted animal(s) never completed: {sorted(missing)}. Refusing to write "
                         "a partial parquet; fix the cause and rerun (regenerates from scratch).")
    df = pd.DataFrame(rows, columns=CELL_COLS + ar_cols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    df.to_parquet(tmp)
    tmp.rename(args.out)                                    # atomic: --out appears only on full success
    print(f"wrote {len(df)} cells over {df['animal'].nunique()} animals, {df['recording'].nunique()} recordings, "
          f"{int(df['y'].sum())} artifact ({100 * df['y'].mean():.1f}%), {len(ar_cols)} autoreject configs "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
