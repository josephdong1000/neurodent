#!/usr/bin/env python
"""Extract per-cell detector scores for one split partition, the input to select_detectors.py and score_test.py.

Each labeled animal is loaded once. The WAR (rms, psdband, psdtotal) and LOF scores are computed, the full
autoreject grid (38 configurations, ``adr.grid_config_names()``) is fit per recording, and everything is
aligned to the consensus labels by start time within ``FRAG_S/2``, as ``scoring.score_mask`` does. The
output has one row per labeled (recording, window, channel) with:

- metadata: recording (the label key), animalday, animal, dataset, cv_group (animals recorded together
  share one), genotype, timestamp and isday of the window, the consensus y, n_raters, any_event, and each
  rater's category (``cat_<rater>``);
- raw filter and LOF scores: rms, logrmsz_abs, beta_wmax, lof;
- one 0/1 reject column per autoreject configuration.

``select_detectors.py`` then tunes thresholds and ranks every detector offline on identical cells.

The labels must lie in one partition of ``config/labeling/split.json`` (``--which dev|test``). The test
parquet is read only by ``score_test.py``. Nothing derived from label values is printed.

Recordings are matched to label keys through ``config/labeling/recording_map.csv`` (see
``score_detectors.RecordingMap``). Every run starts from scratch and writes ``--out`` once at the end, only
after every wanted animal succeeded, so a crash leaves no partial parquet. A ``<out>.provenance.json`` is
written beside it. The grid is the slow phase; give the job cores and wall time, for example::

    uv run python scripts/labeling/extract_dev_features.py --which dev \\
        --out results/labeling/dev_features.parquet \\
        JD=scripts/labeling/labels/labels_cohort4strains_JD_dev.csv \\
        JL=scripts/labeling/labels/labels_cohort4strains_JL_dev.csv \\
        Yong=scripts/labeling/labels/labels_cohort4strains_Yong_dev.csv
"""

import argparse
import json
import logging
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
from score_detectors import (  # noqa: E402  (one consensus, split guard, recording map and feature set)
    CONFIG_RECORDING_MAP,
    CONFIG_SPLIT,
    FEATURES,
    RecordingMap,
    available_cpus,
    build_consensus,
    check_split,
    parse_raters,
    provenance,
)
from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402
from neurodent.results.scoring import RATER_CATEGORY_PREFIX  # noqa: E402

log = logging.getLogger("extract_dev_features")

# Per-cell columns before the autoreject configuration columns (adr.grid_config_names()) and the
# per-rater category columns (cat_<rater>).
CELL_COLS = ["recording", "window", "channel", "t_start_s", "y",
             "rms", "logrmsz_abs", "beta_wmax", "lof",
             "animal", "animalday", "dataset", "cv_group", "genotype", "timestamp", "isday",
             "n_raters", "any_event"]


def beta_prop_grid(df_stats):
    """Return the (W, C) beta power proportion, psdband['beta'] / psdtotal, which ``high_beta`` thresholds."""
    band = pd.DataFrame(df_stats["psdband"].tolist())
    beta = np.array(band["beta"].tolist())
    total = np.array(df_stats["psdtotal"].tolist())
    return beta / total


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
    Transforms that need the whole grid (log-RMS z over all windows, high_beta's maximum over channels) are
    computed by the caller.

    Args:
        rms: ``(W, C)`` RMS.
        logrmsz: ``(W, C)`` z score of log RMS over the recording.
        beta_wmax: ``(W,)`` maximum beta proportion over channels.
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


def ar_cell_bits(grids, ar_ch_names, ar_grid_times, cons_rec, frag_s=R.FRAG_S):
    """Return ``{(window, channel): {config: 0 or 1}}`` for the labeled cells.

    Each cell's start time is matched to the fit-pool grid (``ar_grid_times``) within ``frag_s/2`` and its
    channel to ``ar_ch_names``. Labeled fragments are always in the pool (``adr.fit_pool_indices``), so a
    match is expected; an unmatched cell gets no bits (NaN downstream). 1 means reject, the same polarity as
    the label y.
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
        ci = ch_idx[ch]
        out[(int(r["window"]), ch)] = {name: int(m[row, ci]) for name, m in grids.items()}
    return out


def main(argv=None):
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
                    help="drop labelled recordings the recording map leaves unmapped (listed in the provenance) "
                         "instead of refusing")
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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    manifests = parse_raters(ap, args.raters)
    cons = build_consensus(manifests, args.keymap, args.rule)
    animals_wanted = {r.rsplit("__", 1)[0] for r in cons["recording"].unique()}
    check_split(animals_wanted, args.split, args.which)
    print(f"split ok: {len(animals_wanted)} labelled animals within split['{args.which}']")
    rmap = RecordingMap(None if args.positional_recordings else args.recording_map,
                        positional=args.positional_recordings)
    cat_cols = sorted(c for c in cons.columns if c.startswith(RATER_CATEGORY_PREFIX))

    ar_cols = adr.grid_config_names()  # every configuration, always, on identical cells
    n_workers = args.workers or available_cpus()
    rows, done = [], set()  # every cell in memory (a few thousand rows), one atomic write

    # No per-animal try/except and no resume: a failure ends the run before --out is written, so a crash
    # or timeout leaves no partial parquet. A rerun starts from scratch.
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        try:
            client.wait_for_workers(n_workers, timeout=180)
        except Exception as e:
            log.warning(f"workers: {type(e).__name__}; proceeding with {len(client.scheduler_info()['workers'])}")
        for ds, samples_config, config, animal_id in C.iter_cohort_animals(args.datasets, limit_per_dataset=None,
                                                                           seed=args.seed):
            if animal_id not in animals_wanted:
                continue
            cv_group = samples_config.get("_animal_groups", {}).get(animal_id, animal_id)
            ao = C.load_cohort_animal(samples_config, config, animal_id)
            az = AnimalAnalyzer(ao)
            az.compute_bad_channels(lof_threshold=1.0, lof_chunk_duration_s=args.lof_chunk_duration_s)  # scores only
            war = az.compute_windowed_analysis(FEATURES, window_s=R.FRAG_S, apply_notch_filter=True,
                                               multiprocess_mode="dask", chunk_duration_s=args.chunk_duration_s)
            ch_names = list(war.channel_abbrevs)
            animaldays = war.result["animalday"].to_numpy()
            rms_all = extract_linear_array(war.result["rms"])
            beta_all = beta_prop_grid(war.result)
            for i, animalday in enumerate(ao.animaldays):
                recording = rmap.label(animal_id, i, animalday)
                if recording is None:
                    log.warning(f"{animal_id} position {i} ({animalday!r}) is not in the recording map; skipped")
                    continue
                cons_rec = cons[cons["recording"] == recording]
                if cons_rec.empty:
                    continue
                rowsel = animaldays == animalday
                n = int(rowsel.sum())
                if n == 0:
                    log.warning(f"{recording}: no WAR fragments for animalday {animalday!r}; skipped")
                    continue
                grid_times = np.arange(n) * R.FRAG_S
                lof_by_ch = lof_by_abbrev(war, animalday)
                rms_rec = rms_all[rowsel]
                logrmsz = zscore(np.log(rms_rec), axis=0, nan_policy="omit")  # over all windows, as the filter
                beta_wmax = beta_all[rowsel].max(axis=1)  # over all channels, as the filter
                rec_rows = war.result.loc[rowsel]
                row_meta = {"timestamp": pd.to_datetime(rec_rows["timestamp"]).to_numpy(),
                            "isday": rec_rows["isday"].astype(bool).to_numpy(),
                            "genotype": rec_rows["genotype"].astype(str).to_numpy()}
                cells = align_cells(rms_rec, logrmsz, beta_wmax, lof_by_ch, ch_names, cons_rec, grid_times,
                                    row_meta=row_meta, cell_cols=["n_raters", "any_event", *cat_cols])

                # The autoreject grid runs on the same recording and the same consensus cells.
                Xar, ar_ch, fs_ar, ar_gt = adr.build_fit_epochs(
                    ao.long_recordings[i], cons_rec["window"].unique(), fragment_len_s=R.FRAG_S,
                    max_fit_fragments=(args.ar_max_fit_fragments or None), seed=args.seed)
                log.info(f"{recording}: autoreject grid, {Xar.shape[0]} fragments x {Xar.shape[1]} channels x "
                         f"{len(ar_cols)} configurations")
                grids = adr.compute_masks_grid(Xar, fs_ar)  # {config: (n_pool, C) bool, True meaning reject}
                log.info(f"{recording}: autoreject grid done")
                bits = ar_cell_bits(grids, ar_ch, ar_gt, cons_rec)
                for row in cells:
                    row.update(animal=animal_id, animalday=animalday, dataset=ds, cv_group=cv_group)
                    row.update(bits.get((row["window"], row["channel"]), {}))
                rows += cells
            done.add(animal_id)
            log.info(f"{animal_id}: done ({len(done)}/{len(animals_wanted)} animals)")

    # All or nothing: write --out only if every wanted animal completed.
    missing = animals_wanted - done
    if missing:
        raise SystemExit(f"{len(missing)} wanted animal(s) never completed: {sorted(missing)}. Refusing to write "
                         "a partial parquet; fix the cause and rerun.")
    df = pd.DataFrame(rows, columns=CELL_COLS + cat_cols + ar_cols)
    unmatched = sorted(set(cons["recording"]) - set(df["recording"]))
    if unmatched and not args.allow_unmapped:
        raise SystemExit(f"labelled recordings with no extracted cells: {unmatched}. Refusing to write a parquet "
                         "that drops labelled recordings; check the recording map, or pass --allow-unmapped.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    df.to_parquet(tmp)
    inputs = {f"labels_{r}": p for r, p in manifests.items()}
    inputs.update(keymap=args.keymap, split=args.split,
                  recording_map=None if args.positional_recordings else args.recording_map)
    prov = provenance(inputs, script="extract_dev_features", which=args.which, rule=args.rule,
                      raters=sorted(manifests), positional_recordings=args.positional_recordings,
                      ar_max_fit_fragments=args.ar_max_fit_fragments, seed=args.seed,
                      configs=ar_cols, datasets=list(args.datasets), dropped_recordings=unmatched)
    Path(str(args.out) + ".provenance.json").write_text(json.dumps(prov, indent=2))
    tmp.rename(args.out)  # --out appears only on full success
    print(f"wrote {len(df)} cells over {df['animal'].nunique()} animals, {df['recording'].nunique()} recordings, "
          f"{len(ar_cols)} autoreject configurations to {args.out}")


if __name__ == "__main__":
    main()
