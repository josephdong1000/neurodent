#!/usr/bin/env python
"""Select one artifact detector on the dev features and measure how far its score can be trusted.

Reads the dev parquet written by ``extract_dev_features.py`` (no signal reload) and its provenance file.
Candidates are 5 filters (high_rms, low_rms, logrms_range, high_beta, lof), whose threshold tau is tuned on
labels, and the autoreject grid (``adr.grid_config_names()``), whose masks use no labels. Folds and
bootstrap resamples are by ``cv_group``: animals recorded together from the same files form one group.

For each detector:

- ``all_dev_f1``: tau tuned and scored on all dev cells (in sample; this tau is the one deployed).
- ``logo_f1``: leave-one-group-out F1, with tau refit without the held-out group; for autoreject masks it
  equals ``all_dev_f1``. A 95 percent interval comes from a bootstrap over groups.
- ``lodo_f1``: leave-one-dataset-out F1, with tau tuned on the other datasets.
- Per-group and per-dataset tp, fp, fn, tn, flag rate and F1 (F1 is NaN where a group has neither
  positives nor flags).

Selection rule: the detector with the highest ``logo_f1`` (to 4 decimals) wins; ties go to the earlier
detector in a fixed order, filters first, then the grid order. The paired F1 difference between the winner
and the runner-up is reported as a number only.

The selection effect is measured two ways, each repeating the whole selection inside the training part of
an outer fold and scoring its pick on the held-out part: nested leave-one-group-out, and nested
leave-one-dataset-out. Rater agreement on the dev cells (pairwise Cohen's kappa, overall and per dataset)
is reported as the ceiling.

    uv run python scripts/labeling/select_detectors.py --features results/labeling/dev_features.parquet \\
        --out results/labeling/dev_selection.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import LeaveOneGroupOut

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.scoring import RATER_CATEGORY_PREFIX, pairwise_agreement  # noqa: E402

# Filter or LOF detector: (feature column, reject direction). "hi" rejects scores above tau, "lo" below.
# Autoreject configurations come from the parquet's grid columns.
DETECTORS = {
    "high_rms": ("rms", "hi"),
    "low_rms": ("rms", "lo"),
    "logrms_range": ("logrmsz_abs", "hi"),
    "high_beta": ("beta_wmax", "hi"),
    "lof": ("lof", "hi"),
}
META_COLS = {"recording", "window", "channel", "t_start_s", "y", "animal", "animalday", "dataset", "cv_group",
             "genotype", "timestamp", "isday", "n_raters", "any_event"}
FILTER_COLS = {"rms", "logrmsz_abs", "beta_wmax", "lof"}
GROUP_COL = "cv_group"
# Rater category to binary vote, as scoring.LABEL_MAP: abstentions are NaN.
CATEGORY_VOTE = {"artifact": 1.0, "clean": 0.0, "event": 0.0, "unsure": np.nan, "unlabeled": np.nan}


def score_of(df, col, direction):
    """Return per-cell scores in precision-recall convention (higher means more likely an artifact)."""
    s = df[col].to_numpy(float)
    return s if direction == "hi" else -s


def best_tau(y, s):
    """Return ``(F1, tau)`` for the F1-optimal threshold on ``s`` (reject where s >= tau), over all rows."""
    m = np.isfinite(s)
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return 0.0, np.inf
    p, r, thr = precision_recall_curve(y[m], s[m])
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)[:-1]  # aligned to thr
    if not len(f1):
        return 0.0, np.inf
    k = int(np.argmax(f1))
    return float(f1[k]), float(thr[k])


def predict(s, tau):
    """Return 0/1 rejections, s >= tau; non-finite scores are kept."""
    out = np.zeros(len(s), int)
    m = np.isfinite(s)
    out[m] = (s[m] >= tau).astype(int)
    return out


def _groups(groups):
    uniq = np.array(sorted(pd.unique(groups)))
    return uniq, {a: np.where(groups == a)[0] for a in uniq}


def _cluster_ci(stat, n_groups, n=2000, seed=0):
    """Return a 95 percent interval from ``scipy.stats.bootstrap`` with the group as the resampling unit.

    Whole groups are resampled with replacement and their cells kept together, so the interval respects
    correlation within a group. BCa is tried first, then the percentile method when BCa is undefined (a
    statistic with no jackknife spread).
    """
    gidx = np.arange(n_groups)
    lo = hi = np.nan
    for method in ("BCa", "percentile"):
        ci = bootstrap((gidx,), stat, n_resamples=n, vectorized=False, method=method,
                       random_state=np.random.default_rng(seed)).confidence_interval
        lo, hi = float(ci.low), float(ci.high)
        if np.isfinite(lo) and np.isfinite(hi):
            break
    return lo, hi


def cluster_boot_ci(y, yhat, groups, n=2000, seed=0):
    """Return the 95 percent interval of pooled F1 under a bootstrap over groups."""
    uniq, by = _groups(groups)

    def stat(sampled):
        cells = np.concatenate([by[uniq[int(i)]] for i in np.asarray(sampled).ravel()])
        return f1_score(y[cells], yhat[cells], zero_division=0)

    return _cluster_ci(stat, len(uniq), n, seed)


def paired_delta_ci(y, yhat_a, yhat_b, groups, n=2000, seed=0):
    """Return the 95 percent interval of F1(A) minus F1(B) under a bootstrap over groups."""
    uniq, by = _groups(groups)

    def stat(sampled):
        c = np.concatenate([by[uniq[int(i)]] for i in np.asarray(sampled).ravel()])
        return f1_score(y[c], yhat_a[c], zero_division=0) - f1_score(y[c], yhat_b[c], zero_division=0)

    return _cluster_ci(stat, len(uniq), n, seed)


class _AllInOneFold:
    """A splitter with one fold whose train and test rows are all rows.

    It lets :func:`evaluate` compute the in-sample ``all_dev_f1`` through the same loop that
    ``LeaveOneGroupOut`` uses for ``logo_f1``.
    """

    def split(self, X, y=None, groups=None):
        idx = np.arange(len(X))
        yield idx, idx


def evaluate(s, y, groups, splitter):
    """Fit tau per fold on the training rows, predict the held-out rows, and pool the predictions.

    Args:
        s: Scores, higher meaning more likely an artifact.
        y: 0/1 labels.
        groups: Group of each row, passed to the splitter.
        splitter: ``_AllInOneFold()`` for the in-sample F1, or ``LeaveOneGroupOut()``.

    Returns:
        ``(F1, out-of-fold predictions, {group: F1}, taus per fold)``; with ``_AllInOneFold`` the only tau is
        the deployable one.
    """
    oof = np.zeros(len(s), int)
    per, taus = {}, []
    for tr, te in splitter.split(s, y, groups=groups):
        _, tau = best_tau(y[tr], s[tr])
        oof[te] = predict(s[te], tau)
        taus.append(tau)
        for a in np.unique(groups[te]):
            m = groups[te] == a
            per[str(a)] = round(float(f1_score(y[te][m], oof[te][m], zero_division=0)), 3)
    return float(f1_score(y, oof, zero_division=0)), oof, per, taus


def per_animal_f1(y, yhat, groups):
    """Return ``{group: F1}`` of a fixed prediction."""
    return {str(a): round(float(f1_score(y[groups == a], yhat[groups == a], zero_division=0)), 3)
            for a in sorted(pd.unique(groups))}


def counts(y, yhat):
    """Return tp, fp, fn, tn, n, flag rate and F1 (NaN when there are neither positives nor flags)."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else np.nan
    return {"n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "flag_rate": float(yhat.mean()) if len(y) else np.nan, "f1": f1}


def breakdown(df, yhat, col):
    """Return ``{value of col: counts}`` of a prediction."""
    y = df["y"].to_numpy()
    vals = df[col].astype(str).to_numpy()
    return {v: counts(y[vals == v], yhat[vals == v]) for v in sorted(pd.unique(vals))}


def mask_pred(df, name):
    """Return an autoreject configuration's 0/1 rejections; a cell outside the fit pool (NaN) is kept."""
    return np.nan_to_num(df[name].to_numpy(dtype=float), nan=0.0).astype(int)


def classify_columns(df):
    """Return the autoreject columns of the parquet, refusing any column of unknown role.

    Raises:
        SystemExit: If a column is neither metadata, a filter score, a rater category nor a grid name.
    """
    grid = adr.grid_config_names()
    ar_cols = [c for c in grid if c in df.columns]
    known = META_COLS | FILTER_COLS | set(grid)
    unknown = [c for c in df.columns if c not in known and not c.startswith(RATER_CATEGORY_PREFIX)]
    if unknown:
        raise SystemExit(f"columns of unknown role: {unknown}. Add them to META_COLS or the grid; refusing to "
                         "rank them as detectors.")
    missing = [c for c in grid if c not in df.columns]
    if missing:
        raise SystemExit(f"grid configurations missing from the features: {missing}")
    return ar_cols


def rank(df, ar_cols, group_col=GROUP_COL):
    """Rank every detector on ``df`` by leave-one-group-out F1.

    Returns:
        ``(order, {name: F1})``, ordered by F1 to 4 decimals, ties in the fixed detector order.
    """
    y = df["y"].to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    scores = {}
    for name, (col, direction) in DETECTORS.items():
        scores[name] = evaluate(score_of(df, col, direction), y, groups, LeaveOneGroupOut())[0]
    for name in ar_cols:
        scores[name] = float(f1_score(y, mask_pred(df, name), zero_division=0))
    order = sorted(scores, key=lambda k: -round(scores[k], 4))  # stable: filters first, then grid order
    return order, scores


def fit_and_predict(train, test, name):
    """Predict ``test`` with one detector: a filter's tau is tuned on ``train``; a mask is used as is.

    Returns:
        ``(0/1 predictions, tau in feature units or None)``.
    """
    if name in DETECTORS:
        col, direction = DETECTORS[name]
        _, tau = best_tau(train["y"].to_numpy(), score_of(train, col, direction))
        return predict(score_of(test, col, direction), tau), (tau if direction == "hi" else -tau)
    return mask_pred(test, name), None


def nested(df, ar_cols, outer_col, inner_col=GROUP_COL):
    """Repeat the whole selection inside each outer training part and score its pick on the held-out part.

    Args:
        df: The dev cells.
        ar_cols: Autoreject columns.
        outer_col: Column defining the outer folds (``cv_group`` or ``dataset``).
        inner_col: Column defining the inner folds used to rank.

    Returns:
        ``(pooled predictions, per-fold records)``.
    """
    y = df["y"].to_numpy()
    outer = df[outer_col].astype(str).to_numpy()
    pred = np.zeros(len(df), int)
    folds = []
    for tr, te in LeaveOneGroupOut().split(df, y, groups=outer):
        inner, held = df.iloc[tr], df.iloc[te]
        order, scores = rank(inner, ar_cols, inner_col)
        pick = order[0]
        p, tau = fit_and_predict(inner, held, pick)
        pred[te] = p
        folds.append({"held_out": str(outer[te][0]), "pick": pick, "inner_logo_f1": round(scores[pick], 4),
                      "tau": tau, **counts(y[te], p)})
    return pred, folds


def rater_agreement(df):
    """Return pairwise rater agreement on the cells, overall and per dataset, from the category columns."""
    cat_cols = [c for c in df.columns if c.startswith(RATER_CATEGORY_PREFIX)]
    if len(cat_cols) < 2:
        return {"overall": [], "per_dataset": []}
    long = df.melt(id_vars=["recording", "window", "channel", "dataset"], value_vars=cat_cols,
                   var_name="rater", value_name="category")
    long["rater"] = long["rater"].str[len(RATER_CATEGORY_PREFIX):]
    long["y"] = long["category"].map(CATEGORY_VOTE)
    overall = pairwise_agreement(long.drop(columns="dataset"))
    per_ds = pairwise_agreement(long, by="dataset")
    return {"overall": overall.to_dict("records"), "per_dataset": per_ds.to_dict("records")}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Select one detector on the dev features; report its trust.")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--top", type=int, default=20, help="how many detectors to print")
    args = ap.parse_args(argv)

    from score_detectors import check_split, provenance

    prov_path = Path(str(args.features) + ".provenance.json")
    if not prov_path.exists():
        raise SystemExit(f"missing {prov_path}; features without provenance cannot be selected on")
    features_prov = json.loads(prov_path.read_text())
    df = pd.read_parquet(args.features)
    check_split(set(df["animal"]), args.split, "dev")
    ar_cols = classify_columns(df)
    y = df["y"].to_numpy()
    groups = df[GROUP_COL].astype(str).to_numpy()
    datasets = df["dataset"].astype(str).to_numpy()
    print(f"{len(df)} cells over {df['animal'].nunique()} animals in {len(pd.unique(groups))} groups and "
          f"{len(pd.unique(datasets))} datasets; {len(DETECTORS)} filters + {len(ar_cols)} autoreject configurations\n")

    res = {}
    for name, (col, direction) in DETECTORS.items():
        s = score_of(df, col, direction)
        all_f1, _, _, taus_all = evaluate(s, y, groups, _AllInOneFold())
        logo_f1, oof, per, _ = evaluate(s, y, groups, LeaveOneGroupOut())
        lodo_f1 = evaluate(s, y, datasets, LeaveOneGroupOut())[0]
        lo, hi = cluster_boot_ci(y, oof, groups, n=args.boot)
        res[name] = {"kind": "filter", "col": col, "direction": direction,
                     "tau": (taus_all[0] if direction == "hi" else -taus_all[0]),
                     "all_dev_f1": round(all_f1, 4), "logo_f1": round(logo_f1, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "lodo_f1": round(lodo_f1, 4), "n_score_missing": int((~np.isfinite(s)).sum()),
                     "per_group": breakdown(df, oof, GROUP_COL), "per_dataset": breakdown(df, oof, "dataset"),
                     "_yhat": oof}
    for name in ar_cols:
        yhat = mask_pred(df, name)
        f1 = float(f1_score(y, yhat, zero_division=0))
        lo, hi = cluster_boot_ci(y, yhat, groups, n=args.boot)
        res[name] = {"kind": "autoreject", "col": name, "direction": "reject", "tau": None,
                     "all_dev_f1": round(f1, 4), "logo_f1": round(f1, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "lodo_f1": round(f1, 4), "n_score_missing": int(df[name].isna().sum()),
                     "per_group": breakdown(df, yhat, GROUP_COL), "per_dataset": breakdown(df, yhat, "dataset"),
                     "_yhat": yhat}

    order, _ = rank(df, ar_cols)
    print(f"{'detector':34} {'logo':>7} {'95% CI':>15} {'all-dev':>7} {'lodo':>7}  kind")
    for name in order[:args.top]:
        r = res[name]
        lo, hi = r["ci95"]
        print(f"{name:34} {r['logo_f1']:7.3f}  [{lo:.3f},{hi:.3f}] {r['all_dev_f1']:7.3f} "
              f"{r['lodo_f1']:7.3f}  {r['kind']}")
    if len(order) > args.top:
        print(f"... and {len(order) - args.top} more in {args.out}")

    top, runner = order[0], order[1]
    dlo, dhi = paired_delta_ci(y, res[top]["_yhat"], res[runner]["_yhat"], groups, n=args.boot)
    print(f"\nwinner {top} ({res[top]['kind']}); runner-up {runner} ({res[runner]['kind']}); "
          f"paired F1 difference 95% CI [{dlo:.3f}, {dhi:.3f}]")

    nested_pred, nested_folds = nested(df, ar_cols, GROUP_COL)
    n_lo, n_hi = cluster_boot_ci(y, nested_pred, groups, n=args.boot)
    nested_f1 = float(f1_score(y, nested_pred, zero_division=0))
    lodo_pred, lodo_folds = nested(df, ar_cols, "dataset")
    lodo_sel_f1 = float(f1_score(y, lodo_pred, zero_division=0))
    print(f"nested leave-one-group-out F1 of the whole selection: {nested_f1:.3f} [{n_lo:.3f}, {n_hi:.3f}] "
          f"(winner's logo F1 {res[top]['logo_f1']:.3f})")
    print(f"nested leave-one-dataset-out F1 of the whole selection: {lodo_sel_f1:.3f}")
    for f in lodo_folds:
        print(f"  held out {f['held_out']}: pick {f['pick']}, F1 {f['f1']:.3f}, flag rate {f['flag_rate']:.3f}")

    agreement = rater_agreement(df)
    for row in agreement["overall"]:
        print(f"raters {row['rater_a']} and {row['rater_b']}: kappa {row['cohen_kappa']:.3f} "
              f"over {row['n_cells']} cells")

    winner = {"detector": top, "kind": res[top]["kind"], "col": res[top]["col"], "direction": res[top]["direction"],
              "tau": res[top]["tau"], "logo_f1": res[top]["logo_f1"], "all_dev_f1": res[top]["all_dev_f1"],
              "ci95": res[top]["ci95"], "runner_up": runner, "delta_ci95": [round(dlo, 4), round(dhi, 4)]}
    board = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}
    out = {
        "winner": winner,
        "order": order,
        "leaderboard": board,
        "nested_logo": {"f1": round(nested_f1, 4), "ci95": [round(n_lo, 4), round(n_hi, 4)], "folds": nested_folds},
        "nested_lodo": {"f1": round(lodo_sel_f1, 4), "folds": lodo_folds,
                        "per_dataset": breakdown(df, lodo_pred, "dataset")},
        "rater_agreement": agreement,
        "features": {"path": str(args.features), "sha256": _sha256(args.features), "provenance": features_prov},
        "provenance": provenance({"features": args.features, "split": args.split}, script="select_detectors",
                                 boot=args.boot, group_col=GROUP_COL),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=lambda o: None if o is None else float(o)))
    print(f"\nfrozen pick: {top} ({res[top]['kind']}) written to {args.out}")


if __name__ == "__main__":
    main()
