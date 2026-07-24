#!/usr/bin/env python
"""Select the best filter/LOF detector on the DEV features — whole-dev threshold sweep, leak-free, offline.

Reads `extract_dev_features.py`'s per-cell parquet (no signal reload) and, for each detector, sweeps its one
threshold to the exact whole-dev micro-F1 optimum (`sklearn.metrics.precision_recall_curve`), attaches an
**animal-cluster bootstrap CI** (resample the 14 animals, not cells — the clustering-correct interval), and a
**leave-one-animal-out diagnostic** (`sklearn.model_selection.LeaveOneGroupOut`) for stability + a test preview.
Ranks the detectors, uses a **paired ΔF1 animal-bootstrap** to decide ties, and freezes the winner (config + τ)
to JSON. NOTE: this covers filters + LOF; the autoreject arms join the same leaderboard in Stage 2.

    uv run python scripts/labeling/select_detectors.py --features results/labeling/dev_features_JD.parquet \\
        --out results/labeling/dev_selection_JD.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import LeaveOneGroupOut

# detector -> (feature column, reject direction). "hi": reject if score > τ; "lo": reject if score < τ.
DETECTORS = {
    "high_rms":     ("rms", "hi"),
    "low_rms":      ("rms", "lo"),
    "logrms_range": ("logrmsz_abs", "hi"),
    "high_beta":    ("beta_wmax", "hi"),
    "lof":          ("lof", "hi"),
}


def score_of(df, col, direction):
    """Per-cell score in PR-curve convention (higher = more likely a positive/artifact)."""
    s = df[col].to_numpy(float)
    return s if direction == "hi" else -s


def best_tau(y, s):
    """Exact whole-data F1-optimal threshold on the score `s` (reject where s >= τ)."""
    m = np.isfinite(s)
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return 0.0, np.inf
    p, r, thr = precision_recall_curve(y[m], s[m])
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)[:-1]  # align to thr
    if not len(f1):
        return 0.0, np.inf
    k = int(np.argmax(f1))
    return float(f1[k]), float(thr[k])


def predict(s, tau):
    out = np.zeros(len(s), int)
    m = np.isfinite(s)
    out[m] = (s[m] >= tau).astype(int)
    return out


def _groups(animals):
    uniq = np.array(sorted(pd.unique(animals)))
    return uniq, {a: np.where(animals == a)[0] for a in uniq}


def _cluster_ci(stat, n_animals, n=2000, seed=0):
    """95% CI via `scipy.stats.bootstrap` with the ANIMAL as the resampling unit (a cluster bootstrap: whole
    animals are resampled with replacement and their cells kept together, so the interval respects the
    within-animal correlation). BCa (bias-corrected), falling back to percentile when BCa degenerates — a
    detector with a constant F1 has no jackknife spread and returns nan."""
    aidx = np.arange(n_animals)
    lo = hi = np.nan
    for method in ("BCa", "percentile"):
        ci = bootstrap((aidx,), stat, n_resamples=n, vectorized=False, method=method,
                       random_state=np.random.default_rng(seed)).confidence_interval
        lo, hi = float(ci.low), float(ci.high)
        if np.isfinite(lo) and np.isfinite(hi):
            break
    return lo, hi


def cluster_boot_ci(y, yhat, animals, n=2000, seed=0):
    """95% CI of pooled micro-F1, animal-cluster bootstrap via scipy.stats.bootstrap."""
    uniq, by = _groups(animals)

    def stat(sampled):                                   # sampled: 1-D array of animal indices (with replacement)
        cells = np.concatenate([by[uniq[int(i)]] for i in np.asarray(sampled).ravel()])
        return f1_score(y[cells], yhat[cells], zero_division=0)

    return _cluster_ci(stat, len(uniq), n, seed)


def paired_delta_ci(y, yhat_a, yhat_b, animals, n=2000, seed=0):
    """95% CI of ΔF1 = F1(A) − F1(B), animal-cluster bootstrap. CI includes 0 ⇒ a tie."""
    uniq, by = _groups(animals)

    def stat(sampled):
        c = np.concatenate([by[uniq[int(i)]] for i in np.asarray(sampled).ravel()])
        return f1_score(y[c], yhat_a[c], zero_division=0) - f1_score(y[c], yhat_b[c], zero_division=0)

    return _cluster_ci(stat, len(uniq), n, seed)


def loao_diagnostic(df, col, direction):
    """Leave-one-animal-out: refit τ on the other animals, apply to the held-out one, pool OOF → F1 + per-animal."""
    y = df["y"].to_numpy()
    animals = df["animal"].to_numpy()
    s = score_of(df, col, direction)
    oof = np.zeros(len(df), int)
    per = {}
    for tr, te in LeaveOneGroupOut().split(s, y, groups=animals):
        _, tau = best_tau(y[tr], s[tr])
        oof[te] = predict(s[te], tau)
        per[str(animals[te][0])] = round(float(f1_score(y[te], oof[te], zero_division=0)), 3)
    return float(f1_score(y, oof, zero_division=0)), per


def main():
    ap = argparse.ArgumentParser(description="Whole-dev filter/LOF selection from the extracted features.")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    y = df["y"].to_numpy()
    animals = df["animal"].to_numpy()
    print(f"{len(df)} cells over {len(pd.unique(animals))} animals, {100 * y.mean():.1f}% artifact\n")

    res = {}
    for name, (col, direction) in DETECTORS.items():
        s = score_of(df, col, direction)
        f1, tau = best_tau(y, s)
        yhat = predict(s, tau)
        lo, hi = cluster_boot_ci(y, yhat, animals, n=args.boot)
        diag_f1, per = loao_diagnostic(df, col, direction)
        res[name] = {"col": col, "direction": direction,
                     "tau": (tau if direction == "hi" else -tau),
                     "dev_f1": round(f1, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "loao_f1": round(diag_f1, 4), "per_animal_loao": per, "_yhat": yhat}
        print(f"{name:14} dev-F1={f1:.3f} [{lo:.3f},{hi:.3f}]   LOAO={diag_f1:.3f}   τ={res[name]['tau']:.3g}")

    order = sorted(res, key=lambda k: -res[k]["dev_f1"])
    top, runner = order[0], order[1]
    dlo, dhi = paired_delta_ci(y, res[top]["_yhat"], res[runner]["_yhat"], animals, n=args.boot)
    tie = dlo <= 0 <= dhi
    print(f"\ntop={top} vs {runner}: ΔF1 95% CI [{dlo:.3f}, {dhi:.3f}] → {'TIE (prefer simpler)' if tie else 'top wins'}")

    winner = {"detector": top, "col": res[top]["col"], "direction": res[top]["direction"],
              "tau": res[top]["tau"], "dev_f1": res[top]["dev_f1"], "ci95": res[top]["ci95"],
              "loao_f1": res[top]["loao_f1"], "runner_up": runner, "delta_ci95": [round(dlo, 4), round(dhi, 4)],
              "note": "PROVISIONAL — filters/LOF only; autoreject arms join in Stage 2"}
    board = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}
    args.out.write_text(json.dumps({"leaderboard": board, "winner": winner}, indent=2, default=float))
    print(f"\nfrozen (provisional) winner: {top} (τ={res[top]['tau']:.3g}) → {args.out}")


if __name__ == "__main__":
    main()
