"""Unit tests for the detector-selection + feature-alignment logic in ``scripts/labeling/``.

These are the pure, data-free functions behind the artifact-detector leaderboard (issue #208): the threshold
sweep, the ONE splitter-parameterized evaluator that computes both ``all_dev_animals_f1`` and
``leave_one_animal_out_f1``, the animal-cluster bootstrap CIs, and the per-cell feature/mask alignment. The
scripts stay outside the package by convention (CLAUDE.md), so — like ``tests/test_cohort_labeling.py`` — we
import them via ``sys.path.insert`` and exercise them on synthetic numpy/pandas fixtures (no EEG, no dask).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "labeling"))
import select_detectors as SD  # noqa: E402
import extract_dev_features as EDF  # noqa: E402

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- select_detectors: threshold sweep
def test_best_tau_perfectly_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    f1, tau = SD.best_tau(y, s)
    assert f1 == pytest.approx(1.0)
    assert np.all(SD.predict(s, tau) == y)  # the returned τ reproduces the perfect split


def test_best_tau_degenerate_returns_zero_inf():
    y = np.array([0, 1, 0, 1])
    assert SD.best_tau(y, np.full(4, np.nan)) == (0.0, np.inf)      # all-NaN score
    assert SD.best_tau(np.zeros(4, int), np.arange(4.0)) == (0.0, np.inf)  # single-class labels


def test_predict_threshold_and_nan():
    s = np.array([0.1, np.nan, 0.9])
    assert SD.predict(s, 0.5).tolist() == [0, 0, 1]                 # >= τ; NaN -> 0 (not finite)


def test_score_of_direction():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert SD.score_of(df, "x", "hi").tolist() == [1.0, 2.0, 3.0]
    assert SD.score_of(df, "x", "lo").tolist() == [-1.0, -2.0, -3.0]


# --------------------------------------------------------------------------- the ONE unified evaluator
def _separable(n_animals=3, per=6):
    """Per animal, artifacts are the top half of a shared-scale score → one global τ is optimal for all."""
    rng = np.random.default_rng(0)
    s, y, animals = [], [], []
    for a in range(n_animals):
        lo = rng.uniform(0.0, 0.4, per // 2)
        hi = rng.uniform(0.6, 1.0, per // 2)
        s += [*lo, *hi]
        y += [0] * (per // 2) + [1] * (per // 2)
        animals += [f"A{a}"] * per
    return np.array(s), np.array(y), np.array(animals)


def test_evaluate_all_in_one_fold_matches_best_tau():
    """The all-dev fold must go through the SAME best_tau/predict as everything else (no divergent path)."""
    s, y, animals = _separable()
    f1_eval, oof, per, taus = SD.evaluate(s, y, animals, SD._AllInOneFold())
    f1_direct, tau_direct = SD.best_tau(y, s)
    assert f1_eval == pytest.approx(f1_direct)         # unified all-dev == the direct sweep
    assert len(taus) == 1 and taus[0] == pytest.approx(tau_direct)  # one fold -> the deployable whole-dev τ
    assert np.all(oof == y)                            # separable -> perfect


def test_evaluate_loao_folds_and_leak_free():
    s, y, animals = _separable(n_animals=3)
    loao_f1, oof, per, taus = SD.evaluate(s, y, animals, LeaveOneGroupOut())
    assert len(taus) == 3 and set(per) == {"A0", "A1", "A2"}   # one fold per animal
    all_dev_f1 = SD.evaluate(s, y, animals, SD._AllInOneFold())[0]
    assert loao_f1 <= all_dev_f1 + 1e-9    # leak-free never beats in-sample (best_tau picks a data-point τ)
    assert loao_f1 >= 0.8                  # a shared τ still mostly generalizes on separable-per-animal data


# --------------------------------------------------------------------------- animal-cluster bootstrap CIs
def test_cluster_boot_ci_perfect_prediction():
    _, y, animals = _separable(n_animals=4)
    lo, hi = SD.cluster_boot_ci(y, y.copy(), animals, n=200)     # yhat == y -> F1==1 every resample
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)  # degenerate BCa -> percentile fallback


def test_paired_delta_ci_identical_contains_zero():
    _, y, animals = _separable(n_animals=4)
    yhat = (y == 1).astype(int)
    lo, hi = SD.paired_delta_ci(y, yhat, yhat.copy(), animals, n=200)  # identical detectors -> ΔF1 == 0
    assert lo <= 0.0 <= hi


def test_per_animal_f1_perfect():
    _, y, animals = _separable(n_animals=3)
    per = SD.per_animal_f1(y, y.copy(), animals)
    assert per == {"A0": 1.0, "A1": 1.0, "A2": 1.0}


# --------------------------------------------------------------------------- extract_dev_features: pure helpers
def test_beta_prop_grid():
    df_stats = pd.DataFrame({
        "psdband": [{"beta": np.array([1.0, 2.0, 3.0])}, {"beta": np.array([2.0, 4.0, 6.0])}],
        "psdtotal": [np.array([10.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0])],
    })
    out = EDF.beta_prop_grid(df_stats)     # (W, C) beta / total
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])


def test_lof_by_abbrev_remaps_raw_to_abbrev():
    """lof_scores_dict[animalday] is {"lof_scores": [...], "channel_names": [RAW...]} — must zip + remap."""
    war = SimpleNamespace(
        lof_scores_dict={"AD": {"lof_scores": [1.5, 2.5], "channel_names": ["EEG E9-REF2", "EEG E10-REF2"]}},
        channel_names=["EEG E9-REF2", "EEG E10-REF2"],
        channel_abbrevs=["LMot", "RMot"],
    )
    assert EDF.lof_by_abbrev(war, "AD") == {"LMot": 1.5, "RMot": 2.5}
    assert EDF.lof_by_abbrev(war, "missing") == {}          # absent animalday -> empty


def test_lof_by_abbrev_tolerates_already_abbrev_names():
    war = SimpleNamespace(
        lof_scores_dict={"AD": {"lof_scores": [3.0], "channel_names": ["LMot"]}},
        channel_names=["EEG E9-REF2"], channel_abbrevs=["LMot"],
    )
    assert EDF.lof_by_abbrev(war, "AD") == {"LMot": 3.0}     # name already an abbrev -> kept


def test_ar_cell_bits_time_and_channel_match():
    grids = {"cfg": np.array([[True, False], [False, True]])}   # (2 fragments, 2 channels), True=REJECT
    cons = pd.DataFrame([
        {"recording": "a__0", "window": 0, "channel": "LMot", "t_start_s": 0.0},
        {"recording": "a__0", "window": 1, "channel": "RMot", "t_start_s": 4.0},
        {"recording": "a__0", "window": 9, "channel": "LMot", "t_start_s": 999.0},  # beyond tol -> dropped
    ])
    bits = EDF.ar_cell_bits(grids, ["LMot", "RMot"], np.array([0.0, 4.0]), cons, frag_s=4.0)
    assert bits == {(0, "LMot"): {"cfg": 1}, (1, "RMot"): {"cfg": 1}}


def test_align_cells_time_match():
    rms = np.array([[100.0, 200.0], [300.0, 400.0]])
    logrmsz = np.zeros((2, 2))
    beta_wmax = np.array([0.1, 0.2])
    cons = pd.DataFrame([
        {"recording": "a__0", "window": 0, "channel": "LMot", "t_start_s": 0.0, "y_true": 1},
        {"recording": "a__0", "window": 1, "channel": "RMot", "t_start_s": 4.0, "y_true": 0},
        {"recording": "a__0", "window": 5, "channel": "LMot", "t_start_s": 500.0, "y_true": 1},  # dropped
    ])
    rows = EDF.align_cells(rms, logrmsz, beta_wmax, {"LMot": 1.5, "RMot": 2.5},
                           ["LMot", "RMot"], cons, np.array([0.0, 4.0]), frag_s=4.0)
    assert len(rows) == 2
    assert rows[0]["rms"] == 100.0 and rows[0]["y"] == 1 and rows[0]["lof"] == 1.5
    assert rows[1]["rms"] == 400.0 and rows[1]["y"] == 0 and rows[1]["lof"] == 2.5
