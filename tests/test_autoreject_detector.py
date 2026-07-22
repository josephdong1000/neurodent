"""Regression tests for the adapted autoreject detector (:mod:`neurodent.results.autoreject_detector`).

The two invariants the handoff insists on carrying forward:

* **Leak invariant** — corrupting ONLY validation fragments must change **0** training kept-sets (the
  candidate grid and per-fold baseline come from training only). The discarded "v1" whole-pool baseline
  leaks; the ``leaky_baseline`` control proves the test has teeth.
* **Objective equivalence** — the vectorised running-sum kept-means equal a brute-force per-candidate
  recomputation, including tie-heavy inputs.

Plus a smoke test that all six configs produce a ``(F, C)`` reject mask (exercises the LPSD path serially).
"""
import numpy as np
import pytest

from neurodent.results import autoreject_detector as adr


def test_leak_invariant():
    """Corrupt only the validation fragments -> training candidate grid + kept-sets are unchanged.

    Uses the ``np.mean`` (v3) baseline: it is outlier-sensitive, so the leaky control genuinely flips when
    validation leaks in (giving the test teeth). The leak-free path is invariant for ANY baseline because
    it reads only ``pool[tr]``.
    """
    rng = np.random.RandomState(0)
    pool = rng.randn(50, 8)                       # (N fragments, P freq bins): spectral-like reference
    tr, te = np.arange(40), np.arange(40, 50)

    # kept-set membership is determined by the training features feat_tr (kept = feat_tr <= threshold),
    # so feat_tr identical == every kept-set identical. (kept COUNTS `ns` are ~fixed by the quantile grid
    # and are a poor leak detector; feat_tr is the root invariant.)
    _, _, _, feat0 = adr._fold_solve(pool, tr, te, adr.spec_feat, np.mean, n_cand=20)
    corrupt = pool.copy()
    corrupt[te] *= 1000.0                          # blow up ONLY the held-out fragments
    _, _, _, feat1 = adr._fold_solve(corrupt, tr, te, adr.spec_feat, np.mean, n_cand=20)

    assert np.array_equal(feat0, feat1), "training features changed under validation corruption (LEAK)"

    # v1 control: a whole-pool baseline lets validation leak into the training features -> they DO change.
    _, _, _, featL0 = adr._fold_solve(pool, tr, te, adr.spec_feat, np.mean, n_cand=20, leaky_baseline=True)
    _, _, _, featL1 = adr._fold_solve(corrupt, tr, te, adr.spec_feat, np.mean, n_cand=20, leaky_baseline=True)
    assert not np.allclose(featL0, featL1), "leaky control unchanged — the invariant test has no teeth"


def test_objective_equals_bruteforce():
    """The running-sum kept-mean per candidate equals an explicit brute-force select-and-average."""
    rng = np.random.RandomState(1)
    for trial in range(24):
        N, P = int(rng.randint(20, 60)), int(rng.randint(4, 12))
        pool = rng.randn(N, P)
        if trial % 2 == 0:
            pool[:6] = pool[0]                     # inject ties in the feature (identical fragments)
        idx = rng.permutation(N)
        tr, te = idx[: int(0.8 * N)], idx[int(0.8 * N):]

        cands, errs, ns, _ = adr._fold_solve(pool, tr, te, adr.spec_feat, np.median, n_cand=15)

        base = np.median(pool[tr], axis=0)
        feat_tr = adr.spec_feat(pool[tr], base)
        med = np.median(pool[te], axis=0)
        for k, th in enumerate(cands):
            kept = pool[tr][feat_tr <= th]
            assert kept.shape[0] == ns[k], (trial, k)        # kept count matches searchsorted
            if kept.shape[0] < 2:
                assert np.isnan(errs[k])
                continue
            bf = np.sqrt(np.mean((med - kept.mean(0)) ** 2))
            assert np.isclose(errs[k], bf), (trial, k, errs[k], bf)


def test_compute_masks_six_configs_smoke():
    """All six configs run end-to-end (incl. the LPSD spectral path) and return (F, C) reject masks."""
    fs, frag = 250, 5
    F, C, T = 40, 4, fs * frag
    rng = np.random.RandomState(2)
    X = (50e-6 * rng.randn(F, C, T)).astype(float)          # ~clean volts
    X[5, 1] += 5e-3                                          # one obvious blow-out fragment/channel

    masks = adr.compute_masks(X, fs, n_folds=3, n_cand=12, parallel=False)
    assert set(masks) == set(adr.CONFIG_NAMES)
    for name, m in masks.items():
        assert m.shape == (F, C) and m.dtype == bool, name
        # the injected blow-out should be rejected by the amplitude arm (peak-to-peak is unambiguous there)
    assert masks["autoreject/self"][5, 1], "amplitude arm missed the injected blow-out"


@pytest.mark.parametrize("bad_config", ["nope", "v4/self"])
def test_compute_masks_rejects_unknown_config(bad_config):
    X = np.zeros((5, 2, 50))
    with pytest.raises(ValueError, match="unknown config"):
        adr.compute_masks(X, 250, configs=[bad_config], parallel=False)


def test_needed_idx_handles_multielement_array():
    """The fit-pool selection must accept a numpy array of labelled windows (a recording with >1 labelled
    window). ``needed_windows or []`` on a multi-element array raises 'ambiguous truth value' — the exact
    bug that slipped past the 1-window-per-recording integration test."""
    assert adr._needed_idx(None, 40).tolist() == []
    # multi-element array: dedup, sort, drop out-of-range — and crucially do not raise.
    assert adr._needed_idx(np.array([30, 12, 5, 30, 99, -1]), 40).tolist() == [5, 12, 30]


def test_fit_pool_indices_always_includes_labelled_and_caps():
    """Capped fit pool keeps ALL labelled fragments (so scored cells are covered) + fills to the cap; the
    input is a multi-element numpy array (the regression case)."""
    needed = np.array([3, 7, 900, 900])                      # dup 900; all in range for n=1000
    idx = adr.fit_pool_indices(1000, needed, max_fit_fragments=50, seed=0)
    assert idx.size == 50 and idx.tolist() == sorted(idx.tolist())
    assert {3, 7, 900}.issubset(set(idx.tolist())), "labelled fragments must always be in the fit pool"
    # uncapped (or pool <= cap) -> the whole recording
    assert adr.fit_pool_indices(20, np.array([1, 2, 3]), max_fit_fragments=None).tolist() == list(range(20))
    assert adr.fit_pool_indices(20, np.array([1, 2, 3]), max_fit_fragments=100).tolist() == list(range(20))
