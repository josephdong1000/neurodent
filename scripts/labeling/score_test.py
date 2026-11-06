#!/usr/bin/env python
"""Score the frozen dev pick on the test split, once.

This is the only script that reads test labels' agreement with a detector. It needs:

- the test parquet from ``extract_dev_features.py --which test`` and its provenance file;
- the dev selection JSON from ``select_detectors.py``, whose winner is applied unchanged (a filter with its
  deployable tau, or an autoreject mask).

It refuses to run when the output already exists, when the parquet holds an animal outside
``split.json``'s test list, or when the test features were produced differently from the dev features
(rule, raters, fit-pool cap, seed, configurations, recording matching, or the hashed code). The output
records the sha256 of the selection it applied.

Reported for the pick: pooled precision, recall, F1 and Cohen's kappa against the consensus, with a 95
percent interval from a bootstrap over test groups (``cv_group``); tp, fp, fn, tn, flag rate and F1 per
dataset, per group and per animal; rejection rates per rater-category pattern; and the raters' pairwise
agreement on the test cells.

    uv run python scripts/labeling/score_test.py --features results/labeling/test_features.parquet \\
        --pick results/labeling/dev_selection.json --out results/labeling/test_score.json
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
import report_event_cells as EV  # noqa: E402
import select_detectors as SD  # noqa: E402

# Provenance parameters that must agree between the dev and the test features.
MATCHED_PARAMS = ["rule", "raters", "ar_max_fit_fragments", "seed", "configs", "positional_recordings"]
# Code whose content must agree between the two extractions.
MATCHED_CODE = ["src/neurodent/results/autoreject_detector.py", "src/neurodent/results/scoring.py",
                "scripts/labeling/extract_dev_features.py", "scripts/labeling/build_cohort_bundle.py"]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance_mismatches(dev_prov, test_prov):
    """Return a list of differences between the dev and test feature provenance that matter for scoring."""
    out = []
    dp, tp = dev_prov.get("params", {}), test_prov.get("params", {})
    for k in MATCHED_PARAMS:
        if dp.get(k) != tp.get(k):
            out.append(f"param {k}: dev {dp.get(k)!r}, test {tp.get(k)!r}")
    if dp.get("which") != "dev" or tp.get("which") != "test":
        out.append(f"partitions: dev features say {dp.get('which')!r}, test features say {tp.get('which')!r}")
    dc, tc = dev_prov.get("code", {}), test_prov.get("code", {})
    for f in MATCHED_CODE:
        if dc.get(f) != tc.get(f):
            out.append(f"code {f} differs between the two extractions")
    return out


def apply_pick(df, winner):
    """Return the pick's 0/1 rejections on ``df``, applying the frozen tau or mask unchanged."""
    name = winner["detector"]
    if winner["kind"] == "filter":
        col, direction = SD.DETECTORS[name]
        tau = winner["tau"] if direction == "hi" else -winner["tau"]
        return SD.predict(SD.score_of(df, col, direction), tau)
    return SD.mask_pred(df, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the frozen dev pick on the test split, once.")
    ap.add_argument("--features", type=Path, required=True, help="test parquet from extract_dev_features.py")
    ap.add_argument("--pick", type=Path, required=True, help="dev selection JSON from select_detectors.py")
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)

    from score_detectors import check_split, provenance

    if args.out.exists():
        raise SystemExit(f"{args.out} exists: the test split has been scored. It is scored once; refusing.")
    if not args.pick.exists():
        raise SystemExit(f"no frozen pick at {args.pick}; run select_detectors.py on the dev features first")
    selection = json.loads(args.pick.read_text())
    winner = selection.get("winner")
    dev_prov = (selection.get("features") or {}).get("provenance")
    if not winner or not dev_prov:
        raise SystemExit(f"{args.pick} lacks a winner or the dev features' provenance; refusing")
    prov_path = Path(str(args.features) + ".provenance.json")
    if not prov_path.exists():
        raise SystemExit(f"missing {prov_path}; test features without provenance cannot be scored")
    test_prov = json.loads(prov_path.read_text())
    problems = provenance_mismatches(dev_prov, test_prov)
    if problems:
        raise SystemExit("dev and test features were produced differently:\n  " + "\n  ".join(problems))

    df = pd.read_parquet(args.features)
    check_split(set(df["animal"]), args.split, "test")
    SD.classify_columns(df)

    # Claim the output before any label is compared, so a second run cannot start.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as f:
        f.write("")

    y = df["y"].to_numpy()
    groups = df[SD.GROUP_COL].astype(str).to_numpy()
    yhat = apply_pick(df, winner)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    kappa = float(cohen_kappa_score(y, yhat)) if len(set(y.tolist())) > 1 else np.nan
    lo, hi = SD.cluster_boot_ci(y, yhat, groups, n=args.boot)
    preds = {winner["detector"]: yhat}
    result = {
        "pick": {"path": str(args.pick), "sha256": _sha256(args.pick), "winner": winner},
        "pooled": {"precision": float(p), "recall": float(r), "f1": float(f1), "ci95": [lo, hi],
                   "cohen_kappa": kappa, **SD.counts(y, yhat)},
        "per_dataset": SD.breakdown(df, yhat, "dataset"),
        "per_group": SD.breakdown(df, yhat, SD.GROUP_COL),
        "per_animal": SD.breakdown(df, yhat, "animal"),
        "event_cells": EV.event_table(df, preds).to_dict("records"),
        "rater_agreement": SD.rater_agreement(df),
        "features": {"path": str(args.features), "sha256": _sha256(args.features), "provenance": test_prov},
        "provenance": provenance({"features": args.features, "pick": args.pick, "split": args.split},
                                 script="score_test", boot=args.boot),
    }
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=float))
    os.replace(tmp, args.out)
    print(f"test score of {winner['detector']} ({winner['kind']}): precision {p:.3f}, recall {r:.3f}, "
          f"F1 {f1:.3f} [{lo:.3f}, {hi:.3f}], kappa {kappa:.3f}, {len(y)} cells; written to {args.out}")


if __name__ == "__main__":
    main()
