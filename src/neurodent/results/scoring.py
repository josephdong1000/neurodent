"""Score artifact detectors against human ground-truth labels, per (recording, window, channel).

Consumes the rater CSVs produced by the labeling tooling in ``scripts/labeling/``.

Binary target: 1 = reject/artifact, 0 = keep (clean or real-event). Real events (seizures,
epileptiform bursts) are kept but tagged (``any_event``), so that over-rejection of real activity is
measurable rather than indistinguishable from correct artifact rejection.

Unrecognised label tokens, and labelled cells the detector grid does not cover, raise rather than
being dropped: either would shrink the truth set and inflate the score.

Finer-grained than :func:`neurodent.results.war_lof.evaluate_lof_threshold_binary`, which scores at
(animalday, channel) and returns raw label vectors instead of metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

# The manifest schema. Owned here, imported by the rendering tooling, so emitter and reader cannot drift.
LABEL_COL_PREFIX = "label_"
CATEGORIES = ["clean", "bad", "event", "unsure"]
DEFAULT_CATEGORY = "clean"

# label token -> (binary y: 1=reject/artifact, 0=keep, nan=drop from consensus; category)
LABEL_MAP = {
    "": (np.nan, "unlabeled"),
    "g": (0, "clean"), "good": (0, "clean"), "clean": (0, "clean"), "keep": (0, "clean"),
    "b": (1, "artifact"), "bad": (1, "artifact"), "artifact": (1, "artifact"),
    "e": (0, "event"), "event": (0, "event"), "seizure": (0, "event"),  # real event -> keep, but tagged
    "u": (np.nan, "unsure"), "unsure": (np.nan, "unsure"), "uncertain": (np.nan, "unsure"),
    "?": (np.nan, "unsure"),
}


def ingest(rater_manifests, label_map=LABEL_MAP):
    """``{rater_id: filled_manifest_csv}`` -> tidy long DataFrame.

    The manifest is one row per window with a ``label_<channel>`` column per channel. Melts those
    into one row per (recording, window, channel, rater).

    Raises:
        ValueError: on any unrecognised label token, which would otherwise drop that cell from
            consensus and silently shrink the truth set.
    """
    rows, unknown = [], {}
    for rater, path in rater_manifests.items():
        df = pd.read_csv(path)
        label_cols = [c for c in df.columns if c.startswith(LABEL_COL_PREFIX)]
        if not label_cols:
            raise ValueError(f"{path}: no {LABEL_COL_PREFIX}<channel> columns found")
        has_t = "t_start_s" in df.columns
        for _, r in df.iterrows():
            t_start = float(r["t_start_s"]) if has_t and not pd.isna(r["t_start_s"]) else np.nan
            for lc in label_cols:
                tok = "" if pd.isna(r[lc]) else str(r[lc]).strip().lower()
                if tok not in label_map:
                    unknown.setdefault(tok, []).append(rater)
                    continue
                y, cat = label_map[tok]
                rows.append({"recording": r["recording"], "window": int(r["window"]),
                             "t_start_s": t_start, "channel": lc[len(LABEL_COL_PREFIX):],
                             "rater": rater, "y": y, "category": cat})
    if unknown:
        raise ValueError(
            "unrecognised label tokens: "
            + "; ".join(f"{t!r} x{len(rs)} (raters: {sorted(set(rs))})" for t, rs in unknown.items())
            + f"\nExpected one of {sorted(k for k in label_map if k)} (or blank for unlabelled)."
        )
    return pd.DataFrame(rows)


def consensus(long_df, tie="reject"):
    """Majority vote per (recording, window, channel) over raters who gave that cell a verdict.

    All-`unsure` cells drop out rather than being guessed at. Ties default to `reject`, since keeping
    a suspect window costs more than dropping a clean one.
    """
    out = []
    for key, g in long_df.dropna(subset=["y"]).groupby(["recording", "window", "channel"]):
        frac = g["y"].mean()
        yv = (1 if tie == "reject" else 0) if frac == 0.5 else int(frac > 0.5)
        t_start = float(g["t_start_s"].iloc[0]) if "t_start_s" in g else np.nan
        out.append({"recording": key[0], "window": int(key[1]), "channel": key[2],
                    "t_start_s": t_start, "y_true": yv, "n_raters": len(g),
                    "any_event": bool((g["category"] == "event").any())})
    return pd.DataFrame(out)


def interrater(long_df):
    """Cohen (exactly 2 raters) or Fleiss (>=3) kappa over cells every rater scored.

    The ceiling on any detector score worth reporting: a detector matching consensus as closely as the
    raters match each other is at the human noise floor.
    """
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

    wide = long_df.pivot_table(index=["recording", "window", "channel"], columns="rater",
                               values="y", aggfunc="first").dropna(axis=0)
    raters = list(wide.columns)
    if len(raters) < 2 or len(wide) == 0:
        return {"metric": None, "kappa": np.nan, "n_cells": len(wide), "n_raters": len(raters)}
    if len(raters) == 2:
        k = cohen_kappa_score(wide.iloc[:, 0].astype(int), wide.iloc[:, 1].astype(int))
        return {"metric": "cohen", "kappa": float(k), "n_cells": len(wide), "n_raters": 2}
    table, _ = aggregate_raters(wide.to_numpy().astype(int))
    return {"metric": "fleiss", "kappa": float(fleiss_kappa(table)),
            "n_cells": len(wide), "n_raters": len(raters)}


def score_mask(reject_grid, ch_names, consensus_df, recording, grid_times=None, frag_s=5.0,
               strict=True):
    """Score a detector's REJECT grid against consensus ground truth.

    Args:
        reject_grid: ``(n_windows, n_channels)`` bool, True = REJECT (matching ``y_true``: 1 =
            artifact). For a ``FILTER_REGISTRY`` mask, which is the other way round, use
            :func:`score_keep_mask`.
        ch_names: the detector's channels, positionally matching ``reject_grid``'s columns.
        consensus_df: output of :func:`consensus`.
        recording: which recording's cells to score.
        grid_times: absolute start time (s) of each detector row, length ``n_windows``. **Strongly
            preferred** over the default integer-index match: a labelled cell is matched to the
            detector row whose start time is within ``frag_s/2`` of the cell's ``t_start_s`` (no match
            -> uncovered), so scoring is robust to a differing crop start or fragment numbering rather
            than trusting rater-window ``w`` == detector-row ``w``. When ``None``, falls back to the
            integer window index.
        frag_s: fragment length (s); sets the time-match tolerance (``frag_s/2``).
        strict: raise when the grid does not cover a labelled cell. Skipping those would shrink the
            denominator and return a confident score over whatever happened to line up. Pass False to
            score the overlap and report the rest in ``n_uncovered``.
    """
    cd = consensus_df[consensus_df["recording"] == recording]
    ch_idx = {c: i for i, c in enumerate(ch_names)}
    gt = None if grid_times is None else np.asarray(grid_times, dtype=float)
    tol = frag_s / 2.0
    yt, yp, uncovered = [], [], []
    for _, r in cd.iterrows():
        ch = r["channel"]
        if ch not in ch_idx:
            uncovered.append(f"channel {ch!r} not in the detector's channels")
            continue
        if gt is not None:
            t = float(r["t_start_s"])
            d = np.abs(gt - t)
            row = int(d.argmin())
            if d[row] > tol:
                uncovered.append(f"cell at t={t:g}s has no detector fragment within {tol:g}s")
                continue
        else:
            row = int(r["window"])
            if row >= reject_grid.shape[0]:
                uncovered.append(f"window {row} beyond the detector grid ({reject_grid.shape[0]} rows)")
                continue
        yt.append(int(r["y_true"]))
        yp.append(int(bool(reject_grid[row, ch_idx[ch]])))

    if uncovered and strict:
        u = sorted(set(uncovered))
        raise ValueError(
            f"{recording}: {len(uncovered)} labelled cells are not covered by the detector grid, e.g. "
            + "; ".join(u[:3])
            + ".\nScoring only the rest would silently shrink the truth set. Run the detector over the "
              "same windows/channels that were labelled, or pass strict=False to score the overlap."
        )
    if not yt:
        return {"n": 0, "n_uncovered": len(uncovered)}

    yt, yp = np.array(yt), np.array(yp)
    p, rc, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    return {"n": len(yt), "n_uncovered": len(uncovered),
            "precision": float(p), "recall": float(rc), "f1": float(f1),
            "cohen_kappa": (float(cohen_kappa_score(yt, yp)) if len(set(yt.tolist())) > 1 else np.nan),
            "confusion": confusion_matrix(yt, yp, labels=[0, 1]).tolist()}


def score_keep_mask(keep_mask, ch_names, consensus_df, recording, grid_times=None, frag_s=5.0,
                    strict=True):
    """Score a KEEP mask (the ``FILTER_REGISTRY`` convention) against ground truth.

    Registry filters return ``True = KEEP``; :func:`score_mask` wants ``True = REJECT``. The two are
    indistinguishable at runtime, so the polarity is carried by the function name.
    """
    return score_mask(~np.asarray(keep_mask, dtype=bool), ch_names, consensus_df, recording,
                      grid_times=grid_times, frag_s=frag_s, strict=strict)
