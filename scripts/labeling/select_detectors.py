#!/usr/bin/env python
"""Select the best artifact detector on the dev features: leak-free, offline, filters + LOF + autoreject.

Reads `extract_dev_features.py`'s per-cell parquet (no signal reload). Each filter/LOF detector gets two F1s
through one splitter-parameterized `evaluate` (both call the same `best_tau`/`predict`, so they can't diverge):
  * `all_dev_animals_f1`: τ tuned on all 14 dev animals and scored on them (optimistic; τ is the deployable value).
  * `leave_one_animal_out_f1`: τ refit off each held-out animal (`sklearn.model_selection.LeaveOneGroupOut`),
    the leak-free estimate of performance on a new dataset.
Each autoreject config is scored directly from its per-recording mask (unsupervised, so the two F1s are equal).
Detectors are ranked by `leave_one_animal_out_f1`, the fair cross-family number (filters don't get to peek at
the labels their τ is scored on; autoreject never sees labels), with an animal-cluster bootstrap CI
(`scipy.stats.bootstrap`; resample the 14 animals, not cells) and a paired ΔF1 animal-bootstrap tie-break
(prefer-simpler on a tie). Freezes the winner (detector + deployable τ) to JSON.

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
# To add a filter/LOF detector to the leaderboard, add one line here pointing at a column the extract writes
# (autoreject arms are added on the extract side, see GRID_* in autoreject_detector.py, and auto-discovered).
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
    """95% CI via `scipy.stats.bootstrap` with the animal as the resampling unit (a cluster bootstrap: whole
    animals are resampled with replacement and their cells kept together, so the interval respects the
    within-animal correlation). BCa (bias-corrected), falling back to percentile when BCa degenerates (a
    detector with a constant F1 has no jackknife spread and returns nan)."""
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
    """95% CI of ΔF1 = F1(A) - F1(B), animal-cluster bootstrap. CI includes 0 means a tie."""
    uniq, by = _groups(animals)

    def stat(sampled):
        c = np.concatenate([by[uniq[int(i)]] for i in np.asarray(sampled).ravel()])
        return f1_score(y[c], yhat_a[c], zero_division=0) - f1_score(y[c], yhat_b[c], zero_division=0)

    return _cluster_ci(stat, len(uniq), n, seed)


class _AllInOneFold:
    """Trivial CV splitter: one fold with train == test == all rows. Lets the same :func:`evaluate` loop
    produce the in-sample ``all_dev_animals_f1`` (τ tuned on all 14, scored on all 14) that
    ``LeaveOneGroupOut`` produces as ``leave_one_animal_out_f1``, so the two numbers share one code path."""

    def split(self, X, y=None, groups=None):
        idx = np.arange(len(X))
        yield idx, idx


def evaluate(s, y, animals, splitter):
    """Fit τ per fold (:func:`best_tau` on the train rows), predict the test rows, pool the out-of-fold
    predictions. One function computes both F1s; the only difference is the ``splitter`` (an
    :class:`_AllInOneFold` for ``all_dev_animals_f1`` vs ``LeaveOneGroupOut`` for ``leave_one_animal_out_f1``),
    so the two can never diverge from a hand-copied loop. Returns ``(F1, oof_predictions, per_animal_F1,
    taus_per_fold)``; for the all-in-one fold ``taus_per_fold[0]`` is the deployable whole-dev τ."""
    oof = np.zeros(len(s), int)
    per, taus = {}, []
    for tr, te in splitter.split(s, y, groups=animals):
        _, tau = best_tau(y[tr], s[tr])
        oof[te] = predict(s[te], tau)
        taus.append(tau)
        for a in np.unique(animals[te]):
            m = animals[te] == a
            per[str(a)] = round(float(f1_score(y[te][m], oof[te][m], zero_division=0)), 3)
    return float(f1_score(y, oof, zero_division=0)), oof, per, taus


def per_animal_f1(y, yhat, animals):
    """{animal -> F1} of a fixed prediction (used for the autoreject masks, which have no τ to refit)."""
    return {str(a): round(float(f1_score(y[animals == a], yhat[animals == a], zero_division=0)), 3)
            for a in sorted(pd.unique(animals))}


META_COLS = {"recording", "window", "channel", "t_start_s", "y", "animal"}
FILTER_COLS = {"rms", "logrmsz_abs", "beta_wmax", "lof"}   # anything else is an autoreject config mask


def main():
    ap = argparse.ArgumentParser(description="Unified filter/LOF + autoreject selection from extracted features.")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--top", type=int, default=20, help="how many detectors to print (ranked by honest F1)")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    y = df["y"].to_numpy()
    animals = df["animal"].to_numpy()
    ar_cols = [c for c in df.columns if c not in META_COLS and c not in FILTER_COLS]
    print(f"{len(df)} cells over {len(pd.unique(animals))} animals, {100 * y.mean():.1f}% artifact; "
          f"{len(DETECTORS)} filter/LOF + {len(ar_cols)} autoreject configs\n")

    res = {}
    # ---- filters / LOF: both F1s via the one evaluate(), all-dev fold (tuned, optimistic; also the deploy
    #      τ) and LeaveOneGroupOut (leak-free generalization). Same best_tau/predict, no divergent loop. ----
    for name, (col, direction) in DETECTORS.items():
        s = score_of(df, col, direction)
        all_dev_f1, _, _, taus_all = evaluate(s, y, animals, _AllInOneFold())   # τ tuned + scored on all 14
        loao_f1, oof, per, _ = evaluate(s, y, animals, LeaveOneGroupOut())      # τ refit off the held-out animal
        lo, hi = cluster_boot_ci(y, oof, animals, n=args.boot)                  # CI on the leak-free (OOF) preds
        res[name] = {"kind": "filter", "col": col, "direction": direction,
                     "tau": (taus_all[0] if direction == "hi" else -taus_all[0]),   # deployable whole-dev τ
                     "all_dev_animals_f1": round(all_dev_f1, 4),
                     "leave_one_animal_out_f1": round(loao_f1, 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "per_animal": per, "_yhat": oof}

    # ---- autoreject configs: the leak-free per-recording mask is the prediction; it never sees labels, so
    #      all_dev_animals_f1 == leave_one_animal_out_f1. An unmatched cell (NaN) -> not rejected (0). ----
    for name in ar_cols:
        yhat = np.nan_to_num(df[name].to_numpy(dtype=float), nan=0.0).astype(int)
        f1 = float(f1_score(y, yhat, zero_division=0))
        lo, hi = cluster_boot_ci(y, yhat, animals, n=args.boot)
        res[name] = {"kind": "autoreject", "col": name, "direction": "reject", "tau": None,
                     "all_dev_animals_f1": round(f1, 4), "leave_one_animal_out_f1": round(f1, 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "per_animal": per_animal_f1(y, yhat, animals),
                     "_yhat": yhat}

    # ---- rank by leave_one_animal_out_f1 (fair across families: filters' = OOF, autoreject's = its mask) ----
    order = sorted(res, key=lambda k: -res[k]["leave_one_animal_out_f1"])
    print(f"{'detector':34} {'leave1out':>9} {'95% CI':>15} {'all-dev':>7}  kind")
    for name in order[:args.top]:
        r = res[name]
        lo, hi = r["ci95"]
        print(f"{name:34} {r['leave_one_animal_out_f1']:9.3f}  [{lo:.3f},{hi:.3f}]  "
              f"{r['all_dev_animals_f1']:7.3f}  {r['kind']}")
    if len(order) > args.top:
        print(f"... (+{len(order) - args.top} more configs in {args.out})")

    top, runner = order[0], order[1]
    dlo, dhi = paired_delta_ci(y, res[top]["_yhat"], res[runner]["_yhat"], animals, n=args.boot)
    tie = dlo <= 0 <= dhi
    print(f"\ntop={top} ({res[top]['kind']}) vs {runner} ({res[runner]['kind']}): "
          f"Δleave1out-F1 95% CI [{dlo:.3f}, {dhi:.3f}] -> {'tie (prefer simpler)' if tie else 'top wins'}")

    winner = {"detector": top, "kind": res[top]["kind"], "col": res[top]["col"],
              "direction": res[top]["direction"], "tau": res[top]["tau"],
              "leave_one_animal_out_f1": res[top]["leave_one_animal_out_f1"],
              "all_dev_animals_f1": res[top]["all_dev_animals_f1"], "ci95": res[top]["ci95"],
              "runner_up": runner, "delta_ci95": [round(dlo, 4), round(dhi, 4)],
              "note": "ranked by leave_one_animal_out_f1 (filters: τ refit off the held-out animal; autoreject: "
                      "per-recording unsupervised mask, so it equals all_dev_animals_f1). Provisional dev pick; "
                      "the sealed test is the arbiter."}
    board = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}
    args.out.write_text(json.dumps({"leaderboard": board, "winner": winner}, indent=2, default=float))
    print(f"\nfrozen (provisional) winner: {top} ({res[top]['kind']}) -> {args.out}")


if __name__ == "__main__":
    main()
