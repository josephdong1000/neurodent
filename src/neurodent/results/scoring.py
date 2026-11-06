"""Score artifact detectors against human labels, per (recording, window, channel).

Reads the rater CSVs produced by the labeling tooling in ``scripts/labeling/``.

Binary target: 1 means reject (artifact), 0 means keep (clean or a real event). Real events such as
seizures and epileptiform bursts are kept but tagged (``any_event``), so that rejecting real activity can be
measured rather than counted as correct rejection.

Unrecognized label tokens, and labeled cells a detector grid does not cover, raise instead of being
dropped, since either would shrink the truth set and inflate the score.

This scores at (recording, window, channel), finer than
:func:`neurodent.results.war_lof.evaluate_lof_threshold_binary`, which works at (animalday, channel) and
returns raw label vectors instead of metrics.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

# The manifest schema. Defined here and imported by the rendering tooling, so writer and reader agree.
LABEL_COL_PREFIX = "label_"
CATEGORIES = ["clean", "bad", "event", "unsure"]
DEFAULT_CATEGORY = "clean"
# Prefix of the per-rater category columns that consensus() adds.
RATER_CATEGORY_PREFIX = "cat_"

# Label token: (binary y, with 1 reject, 0 keep, NaN dropped from consensus; category).
LABEL_MAP = {
    "": (np.nan, "unlabeled"),
    "g": (0, "clean"), "good": (0, "clean"), "clean": (0, "clean"), "keep": (0, "clean"),
    "b": (1, "artifact"), "bad": (1, "artifact"), "artifact": (1, "artifact"),
    "e": (0, "event"), "event": (0, "event"), "seizure": (0, "event"),  # a real event is kept, but tagged
    "u": (np.nan, "unsure"), "unsure": (np.nan, "unsure"), "uncertain": (np.nan, "unsure"),
    "?": (np.nan, "unsure"),
}


def ingest(rater_manifests, label_map=LABEL_MAP):
    """Read filled rater manifests into one long table.

    A manifest has one row per window and a ``label_<channel>`` column per channel. These are melted into
    one row per (recording, window, channel, rater).

    Args:
        rater_manifests: ``{rater_id: path to the filled manifest CSV}``.
        label_map: Token to ``(y, category)`` mapping.

    Returns:
        DataFrame with columns recording, window, t_start_s, channel, rater, y, category.

    Raises:
        ValueError: On any unrecognized label token, which would otherwise drop that cell from the
            consensus and shrink the truth set.
    """
    y_of = {t: v[0] for t, v in label_map.items()}
    cat_of = {t: v[1] for t, v in label_map.items()}
    parts, unknown = [], {}
    for rater, path in rater_manifests.items():
        df = pd.read_csv(path)
        label_cols = [c for c in df.columns if c.startswith(LABEL_COL_PREFIX)]
        if not label_cols:
            raise ValueError(f"{path}: no {LABEL_COL_PREFIX}<channel> columns found")
        long = df.melt(id_vars=["recording", "window", "t_start_s"], value_vars=label_cols,
                       var_name="channel", value_name="tok")
        long["channel"] = long["channel"].str[len(LABEL_COL_PREFIX):]
        long["tok"] = long["tok"].where(long["tok"].notna(), "").astype(str).str.strip().str.lower()
        long["rater"] = rater
        for t in long.loc[~long["tok"].isin(y_of), "tok"].unique():  # collected, not dropped
            unknown.setdefault(t, set()).add(rater)
        parts.append(long)
    if unknown:
        raise ValueError(
            "unrecognised label tokens: "
            + "; ".join(f"{t!r} (raters: {sorted(rs)})" for t, rs in unknown.items())
            + f"\nExpected one of {sorted(k for k in label_map if k)} (or blank for unlabelled)."
        )
    out = pd.concat(parts, ignore_index=True)
    out["window"] = out["window"].astype(int)
    out["y"] = out["tok"].map(y_of)
    out["category"] = out["tok"].map(cat_of)
    return out[["recording", "window", "t_start_s", "channel", "rater", "y", "category"]]


def unblind(long_df, keymap):
    """Replace blinded channel slots with the true channel names.

    A blinded bundle shows each window's channels shuffled, as neutral slots (``Ch A``, ``Ch B``, ...), so
    anatomy cannot bias the rater (see ``blind_channels`` in ``scripts/labeling/render_context.py``). The
    rater CSVs, and so :func:`ingest`'s output, are keyed by slot. Blinding is per window, so the true
    channel is restored per (recording, window, slot) from the keymap, which lets consensus and scoring key
    on real channels. Call it between :func:`ingest` and :func:`consensus`; skip it for unblinded bundles.

    Args:
        long_df: Output of :func:`ingest`, whose ``channel`` column holds the slot.
        keymap: Key with columns recording, window, slot and channel (the true name); the ``keymap.csv``
            the cohort bundler writes outside the rater bundle.

    Returns:
        ``long_df`` with ``channel`` set to the true channel and the slot kept in a ``slot`` column.

    Raises:
        ValueError: If a labeled (recording, window, slot) has no keymap entry, which would drop it from
            the truth set.
    """
    km = keymap[["recording", "window", "slot", "channel"]].rename(columns={"channel": "true_channel"})
    km["window"] = km["window"].astype(int)
    merged = long_df.merge(
        km, how="left",
        left_on=["recording", "window", "channel"], right_on=["recording", "window", "slot"],
    )
    missing = merged["true_channel"].isna()
    # A labeled cell without a keymap entry is a real mismatch and raises. An unlabeled one is a padding
    # slot that a recording with fewer channels lacks in a mixed bundle; consensus drops it anyway.
    labelled_missing = missing & merged["y"].notna()
    if labelled_missing.any():
        bad = list(merged.loc[labelled_missing, ["recording", "window", "channel"]].drop_duplicates()
                   .itertuples(index=False))
        raise ValueError(
            "unblind: a labelled cell has no keymap entry: "
            + "; ".join(f"{r.recording}/w{r.window}/{r.channel}" for r in bad[:5])
            + ". The keymap must cover every (recording, window, slot) that carries a label."
        )
    merged = merged[~missing].copy()
    merged["channel"] = merged["true_channel"]
    return merged.drop(columns="true_channel")


def consensus(long_df, rule="majority"):
    """Combine raters into one ground-truth label per (recording, window, channel).

    A rater votes on a cell only when their label is not unsure or blank (``y`` not NaN); cells where
    every rater abstained are dropped. Rules:

    - ``"majority"``: at least half of the voting raters call the cell bad (a tie counts as reject).
    - ``"unanimous"``: every voting rater calls it bad.
    - ``"any"``: at least one voting rater calls it bad.

    Args:
        long_df: Output of :func:`unblind` (or :func:`ingest` for an unblinded bundle).
        rule: One of the rules above.

    Returns:
        One row per voted cell with recording, window, channel, t_start_s, n_raters (voting raters),
        any_event, y_true, and one ``cat_<rater>`` column per rater holding that rater's category
        (clean, artifact, event, unsure or unlabeled), abstentions included.

    Raises:
        ValueError: On an unknown rule, or if a rater labeled the same cell twice.
    """
    keys = ["recording", "window", "channel"]
    d = long_df.dropna(subset=["y"]).copy()
    d["is_event"] = d["category"] == "event"
    res = (d.groupby(keys, sort=False)
             .agg(frac=("y", "mean"), n_raters=("y", "size"),
                  any_event=("is_event", "any"), t_start_s=("t_start_s", "first"))
             .reset_index())
    if rule == "majority":
        yv = res["frac"] >= 0.5  # a tie counts as reject
    elif rule == "unanimous":
        yv = res["frac"] == 1.0
    elif rule == "any":
        yv = res["frac"] > 0.0
    else:
        raise ValueError(f"rule must be 'majority', 'unanimous', or 'any'; got {rule!r}")
    res["y_true"] = yv.astype(int)

    if long_df.duplicated(keys + ["rater"]).any():
        raise ValueError("a rater labeled the same (recording, window, channel) more than once")
    cats = long_df.set_index(keys + ["rater"])["category"].unstack("rater")
    cats.columns = [f"{RATER_CATEGORY_PREFIX}{r}" for r in cats.columns]
    res = res.merge(cats.reset_index(), on=keys, how="left")
    return res.drop(columns="frac")


def interrater(long_df):
    """Return Cohen's kappa (2 raters) or Fleiss' kappa (3 or more) over cells every rater voted on.

    This is the ceiling for any detector score: a detector that matches the consensus as closely as the
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


def pairwise_agreement(long_df, by=None):
    """Return bad-against-keep agreement for every pair of raters, optionally per group.

    For each pair, only cells both raters voted on count (unsure and blank are dropped).

    Args:
        long_df: Output of :func:`unblind` or :func:`ingest`, optionally with a grouping column.
        by: Name of a column of ``long_df`` to group by (for example a dataset column), or None.

    Returns:
        DataFrame with one row per (group, rater pair): n_cells, agreement (fraction of equal votes),
        cohen_kappa (NaN when either rater used one class only), and the two disagreement counts
        ``a_keep_b_bad`` and ``a_bad_b_keep``.
    """
    index = ["recording", "window", "channel"] + ([by] if by else [])
    wide = long_df.pivot_table(index=index, columns="rater", values="y", aggfunc="first")
    raters = sorted(wide.columns)
    groups = [(None, wide)] if not by else list(wide.groupby(level=by))
    rows = []
    for g, sub in groups:
        for a, b in itertools.combinations(raters, 2):
            pair = sub[[a, b]].dropna()
            ya, yb = pair[a].astype(int).to_numpy(), pair[b].astype(int).to_numpy()
            two_classes = len(set(ya.tolist())) > 1 and len(set(yb.tolist())) > 1
            rows.append({
                **({by: g} if by else {}),
                "rater_a": a, "rater_b": b, "n_cells": int(len(pair)),
                "agreement": float((ya == yb).mean()) if len(pair) else np.nan,
                "cohen_kappa": float(cohen_kappa_score(ya, yb)) if two_classes else np.nan,
                "a_keep_b_bad": int(((ya == 0) & (yb == 1)).sum()),
                "a_bad_b_keep": int(((ya == 1) & (yb == 0)).sum()),
            })
    return pd.DataFrame(rows)


def score_mask(reject_grid, ch_names, consensus_df, recording, grid_times=None, frag_s=5.0,
               strict=True):
    """Score a detector's reject grid against the consensus.

    Args:
        reject_grid: ``(n_windows, n_channels)`` bool, True meaning reject (matching ``y_true`` of 1). For a
            ``FILTER_REGISTRY`` mask, where True means keep, use :func:`score_keep_mask`.
        ch_names: The detector's channels, in the order of ``reject_grid``'s columns.
        consensus_df: Output of :func:`consensus`.
        recording: Which recording's cells to score.
        grid_times: Start time in seconds of each detector row, ``n_windows`` long. Preferred over the
            default index match: a labeled cell matches the row whose start is within ``frag_s/2`` of the
            cell's ``t_start_s`` (no match leaves it uncovered), so scoring survives a different crop start
            or fragment numbering. With None, the integer window index is used.
        frag_s: Fragment length in seconds; sets the time tolerance (``frag_s/2``).
        strict: Raise when the grid does not cover a labeled cell. Skipping those would shrink the
            denominator. Pass False to score the overlap and report the rest in ``n_uncovered``.

    Returns:
        Dict with n, n_uncovered and, when cells were scored, precision, recall, f1, cohen_kappa and
        confusion.

    Raises:
        ValueError: With ``strict``, when labeled cells are not covered.
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
    """Score a keep mask (the ``FILTER_REGISTRY`` convention) against the consensus.

    Registry filters return True for keep, while :func:`score_mask` takes True for reject. The two look the
    same at runtime, so the polarity is carried by the function name. Arguments are as in
    :func:`score_mask`.
    """
    return score_mask(~np.asarray(keep_mask, dtype=bool), ch_names, consensus_df, recording,
                      grid_times=grid_times, frag_s=frag_s, strict=strict)
