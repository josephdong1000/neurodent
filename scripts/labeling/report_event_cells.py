#!/usr/bin/env python
"""Report how often each combination and detector rejects cells, grouped by what the raters called them (dev).

Epileptiform events can look like artifacts in a 5 s window, and the consensus counts an event vote as keep.
This report shows each rejection rate per rater-category pattern: the sorted categories of all raters on the
cell (for example ``artifact+artifact+event``), plus a coarse class:

- ``all_bad``: every voting rater said bad;
- ``bad_and_event``: at least one bad and at least one event vote;
- ``bad_and_clean``: bad and clean votes, no event;
- ``event_no_bad``: at least one event vote and no bad;
- ``all_clean``: only clean votes;
- ``no_votes``: every rater unsure or blank (normally absent, since consensus drops such cells).

Rows come first for the combinations of ``select_detectors.py`` with the selection's autoreject configuration
k and settings: today (F or M), replace (F or A(k)), add (F or M or A(k)), lof_branch (F or L),
autoreject_alone, filters_alone and config_lists_alone (the last only on cells whose recording has a config
list). F, L and M read the recording channels and config lists of the dev features' provenance the selection
recorded, so ``--features`` must be the parquet the selection was made on; the run refuses another. Then the
single detectors of the diagnostic ranking: filters with their deployable tau, autoreject configurations with
their masks. The patterns hold every rater's category, so they do not depend on a consensus rule. The test
split's version of this report is produced only inside ``score_test.py``, for the combinations.

    uv run python scripts/labeling/report_event_cells.py --features results/labeling/dev_features.parquet \\
        --selection results/labeling/dev_selection.json --out results/labeling/dev_event_cells.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
import select_detectors as SD  # noqa: E402
from neurodent.results.scoring import RATER_CATEGORY_PREFIX  # noqa: E402


def coarse_class(categories):
    """Return the coarse class of one cell from its raters' categories (see the module docstring)."""
    votes = [c for c in categories if c in ("artifact", "clean", "event")]
    if not votes:
        return "no_votes"
    bad, event, clean = "artifact" in votes, "event" in votes, "clean" in votes
    if bad and event:
        return "bad_and_event"
    if bad and clean:
        return "bad_and_clean"
    if bad:
        return "all_bad"
    if event:
        return "event_no_bad"
    return "all_clean"


def category_columns(df):
    """Return the ``cat_<rater>`` columns of ``df``, sorted.

    Raises:
        SystemExit: If there is none.
    """
    cat_cols = sorted(c for c in df.columns if c.startswith(RATER_CATEGORY_PREFIX))
    if not cat_cols:
        raise SystemExit("the features hold no rater category columns (cat_<rater>)")
    return cat_cols


def patterns(df):
    """Return ``(pattern, coarse class)`` series for the cells of ``df`` from its ``cat_<rater>`` columns."""
    cat_cols = category_columns(df)
    cats = df[cat_cols].fillna("unlabeled").astype(str).to_numpy()
    pattern = pd.Series(["+".join(sorted(row)) for row in cats], index=df.index)
    coarse = pd.Series([coarse_class(row) for row in cats], index=df.index)
    return pattern, coarse


def detector_predictions(df, name, entry):
    """Return 0/1 rejections of one detector, with a filter's deployable tau or an autoreject mask."""
    if entry["kind"] == "filter":
        col, direction = SD.DETECTORS[name]
        tau = entry["tau"] if direction == "hi" else -entry["tau"]
        return SD.predict(SD.score_of(df, col, direction), tau)
    return SD.mask_pred(df, name)


def combination_predictions(df, selection):
    """Return the combinations' 0/1 rejections and scored cells with the selection's k and settings.

    Args:
        df: Cells with the member columns.
        selection: The dev selection JSON from ``select_detectors.py``; F, L and M read the recording records of
            the features provenance it recorded.

    Returns:
        ``({combination: 0/1 per cell}, {combination: bool per cell})``.

    Raises:
        SystemExit: If the recorded provenance lacks the recording records or they do not describe ``df``.
    """
    SD.require_member_columns(df)
    prov = (selection.get("features") or {}).get("provenance") or {}
    SD.require_recording_records(prov, "selection's dev")
    members, _ = SD.member_masks(df, selection["settings"], prov)
    preds = SD.combination_preds(members, SD.mask_pred(df, selection["pick"]["autoreject_config"]))
    return preds, SD.combination_cells(df)


def event_table(df, preds, cells=None):
    """Return rejection counts per detector, coarse class and pattern.

    Args:
        df: Cells with ``cat_<rater>`` columns.
        preds: ``{detector or combination: 0/1 predictions aligned to df}``.
        cells: Optional ``{name: bool per cell}``; a name listed here is counted only on those cells.

    Returns:
        DataFrame with detector, coarse_class, pattern, n_cells, n_rejected, rejection_rate.
    """
    pattern, coarse = patterns(df)
    cells = cells or {}
    rows = []
    for name, yhat in preds.items():
        frame = pd.DataFrame({"coarse_class": coarse, "pattern": pattern, "rejected": np.asarray(yhat, int)})
        if name in cells:
            frame = frame[np.asarray(cells[name], bool)]
        for (cc, pat), sub in frame.groupby(["coarse_class", "pattern"], sort=True):
            rows.append({"detector": name, "coarse_class": cc, "pattern": pat, "n_cells": int(len(sub)),
                         "n_rejected": int(sub["rejected"].sum()),
                         "rejection_rate": float(sub["rejected"].mean())})
    return pd.DataFrame(rows, columns=["detector", "coarse_class", "pattern", "n_cells", "n_rejected",
                                       "rejection_rate"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rejection rate per rater-category pattern, dev split.")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True, help="dev selection JSON from select_detectors.py")
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--detectors", nargs="+", default=None,
                    help="single detectors to add after the combinations (default: every detector, in the "
                         "diagnostic order)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from score_detectors import check_split

    selection = json.loads(args.selection.read_text())
    if not args.features.exists():
        raise SystemExit(f"{args.features} not found; pass the parquet the selection was made on")
    actual, selected_on = SD._sha256(args.features), (selection.get("features") or {}).get("sha256")
    if actual != selected_on:
        raise SystemExit(f"{args.features} is not the parquet the selection was made on (sha256 {actual}, "
                         f"selection {selected_on!r}); refusing")
    df = pd.read_parquet(args.features)
    check_split(set(df["animal"]), args.split, "dev")
    board = selection["diagnostic"]["leaderboard"]
    names = args.detectors or selection["diagnostic"]["order"]
    unknown = [n for n in names if n not in board]
    if unknown:
        raise SystemExit(f"detectors not in the selection: {unknown}")
    preds, cells = combination_predictions(df, selection)
    preds.update({n: detector_predictions(df, n, board[n]) for n in names})
    table = event_table(df, preds, cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} rows for {len(SD.COMBINATIONS)} combinations (k = "
          f"{selection['pick']['autoreject_config']}) and {len(names)} detectors to {args.out}")


if __name__ == "__main__":
    main()
