#!/usr/bin/env python
"""Report how often each detector rejects cells, grouped by what the raters called them (dev split).

Epileptiform events can look like artifacts in a 5 s window, and the consensus counts an event vote as keep.
This report shows each detector's rejection rate per rater-category pattern: the sorted categories of all
raters on the cell (for example ``artifact+artifact+event``), plus a coarse class:

- ``all_bad``: every voting rater said bad;
- ``bad_and_event``: at least one bad and at least one event vote;
- ``bad_and_clean``: bad and clean votes, no event;
- ``event_no_bad``: at least one event vote and no bad;
- ``all_clean``: only clean votes;
- ``no_votes``: every rater unsure or blank (normally absent, since consensus drops such cells).

Filters use the deployable tau from the dev selection; autoreject configurations use their masks. The test
split's version of this report is produced only inside ``score_test.py``, for the frozen pick.

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


def patterns(df):
    """Return ``(pattern, coarse class)`` series for the cells of ``df`` from its ``cat_<rater>`` columns."""
    cat_cols = sorted(c for c in df.columns if c.startswith(RATER_CATEGORY_PREFIX))
    if not cat_cols:
        raise SystemExit("the features hold no rater category columns (cat_<rater>)")
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


def event_table(df, preds):
    """Return rejection counts per detector, coarse class and pattern.

    Args:
        df: Cells with ``cat_<rater>`` columns.
        preds: ``{detector: 0/1 predictions aligned to df}``.

    Returns:
        DataFrame with detector, coarse_class, pattern, n_cells, n_rejected, rejection_rate.
    """
    pattern, coarse = patterns(df)
    rows = []
    for name, yhat in preds.items():
        yhat = np.asarray(yhat, int)
        frame = pd.DataFrame({"coarse_class": coarse, "pattern": pattern, "rejected": yhat})
        for (cc, pat), sub in frame.groupby(["coarse_class", "pattern"], sort=True):
            rows.append({"detector": name, "coarse_class": cc, "pattern": pat, "n_cells": int(len(sub)),
                         "n_rejected": int(sub["rejected"].sum()),
                         "rejection_rate": float(sub["rejected"].mean())})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rejection rate per rater-category pattern, dev split.")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True, help="dev selection JSON from select_detectors.py")
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--detectors", nargs="+", default=None, help="default: every detector, in selection order")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from score_detectors import check_split

    df = pd.read_parquet(args.features)
    check_split(set(df["animal"]), args.split, "dev")
    selection = json.loads(args.selection.read_text())
    names = args.detectors or selection["order"]
    unknown = [n for n in names if n not in selection["leaderboard"]]
    if unknown:
        raise SystemExit(f"detectors not in the selection: {unknown}")
    preds = {n: detector_predictions(df, n, selection["leaderboard"][n]) for n in names}
    table = event_table(df, preds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} rows for {len(names)} detectors to {args.out}")


if __name__ == "__main__":
    main()
