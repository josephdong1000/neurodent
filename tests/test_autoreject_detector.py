"""Tests for the adapted autoreject detector (:mod:`neurodent.results.autoreject_detector`).

Two invariants carried from the July work:

- Leak invariant: corrupting only the held-out fragments changes no training kept set, because the
  candidate grid and the per-split baseline come from training rows only. A whole-pool baseline leaks; the
  ``leaky_baseline`` control shows the test can fail.
- Objective equivalence: the running-sum kept means equal a brute-force recomputation per candidate, ties
  included.

The rest characterize behavior on edge cases (dead channels, short pools, few channels, gain), check the
stored-parameter API against the flags, and refuse inputs the search has no rule for. Characterization
tests record current behavior, including its limits, so that a change to the rules shows up here.
"""

import numpy as np
import pytest

from neurodent.results import autoreject_detector as adr

SELF_CONFIGS = ["autoreject/self", "v2/self", "v3/self"]
CMR_CONFIGS = ["autoreject/self+CMR", "v2/self+CMR", "v3/self+CMR"]


def _noise(seed, F, C, T, burst_rate=0.1):
    """Return (F, C, T) white noise in volts with a seeded share of fragments scaled up 6 times."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 50e-6, size=(F, C, T))
    X[rng.random((F, C)) < burst_rate] *= 6
    return X


def test_leak_invariant():
    """Corrupting only the held-out fragments leaves the training features, and so every kept set, unchanged.

    Uses the ``np.mean`` (v3) baseline: it is sensitive to outliers, so the leaky control changes when the
    held-out rows leak in. The leak-free path is invariant for any baseline because it reads only
    ``pool[tr]``.
    """
    rng = np.random.RandomState(0)
    pool = rng.randn(50, 8)  # (N fragments, P bins), spectrum-like rows
    tr, te = np.arange(40), np.arange(40, 50)

    # Kept-set membership is decided by the training features (kept = feat_tr <= candidate), so identical
    # feat_tr means identical kept sets. Kept counts are nearly fixed by the quantile grid and would make a
    # poor leak detector.
    _, _, _, feat0 = adr._fold_solve(pool, tr, te, adr.spec_feat, np.mean, n_cand=20)
    corrupt = pool.copy()
    corrupt[te] *= 1000.0  # change only the held-out fragments
    _, _, _, feat1 = adr._fold_solve(corrupt, tr, te, adr.spec_feat, np.mean, n_cand=20)

    assert np.array_equal(feat0, feat1), "training features changed when only held-out rows changed"

    # Control: a whole-pool baseline lets held-out rows reach the training features.
    _, _, _, featL0 = adr._fold_solve(pool, tr, te, adr.spec_feat, np.mean, n_cand=20, leaky_baseline=True)
    _, _, _, featL1 = adr._fold_solve(corrupt, tr, te, adr.spec_feat, np.mean, n_cand=20, leaky_baseline=True)
    assert not np.allclose(featL0, featL1), "leaky control unchanged, so the invariant test cannot fail"


def test_objective_equals_bruteforce():
    """The running-sum kept mean per candidate equals an explicit select-and-average."""
    rng = np.random.RandomState(1)
    for trial in range(24):
        N, P = int(rng.randint(20, 60)), int(rng.randint(4, 12))
        pool = rng.randn(N, P)
        if trial % 2 == 0:
            pool[:6] = pool[0]  # ties in the feature (identical fragments)
        idx = rng.permutation(N)
        tr, te = idx[: int(0.8 * N)], idx[int(0.8 * N) :]

        cands, errs, ns, _ = adr._fold_solve(pool, tr, te, adr.spec_feat, np.median, n_cand=15)

        base = np.median(pool[tr], axis=0)
        feat_tr = adr.spec_feat(pool[tr], base)
        med = np.median(pool[te], axis=0)
        for k, th in enumerate(cands):
            kept = pool[tr][feat_tr <= th]
            assert kept.shape[0] == ns[k], (trial, k)  # kept count matches searchsorted
            if kept.shape[0] < 2:
                assert np.isnan(errs[k])
                continue
            bf = np.sqrt(np.mean((med - kept.mean(0)) ** 2))
            assert np.isclose(errs[k], bf), (trial, k, errs[k], bf)


def test_compute_masks_six_configs_smoke():
    """All six configs run end to end and return (F, C) masks; a transient is flagged by peak-to-peak."""
    fs, frag = 250, 5
    F, C, T = 40, 4, fs * frag
    rng = np.random.RandomState(2)
    X = (50e-6 * rng.randn(F, C, T)).astype(float)
    X[5, 1, 100:150] += 5e-3  # a 5 mV, 200 ms pulse, which raises that fragment's peak-to-peak

    masks = adr.compute_masks(X, fs, n_folds=3, n_cand=12, parallel=False)
    assert set(masks) == set(adr.CONFIG_NAMES)
    for name, m in masks.items():
        assert m.shape == (F, C) and m.dtype == bool, name
    assert masks["autoreject/self"][5, 1], "peak-to-peak missed the injected pulse"


@pytest.mark.parametrize("bad_config", ["nope", "v4/self"])
def test_compute_masks_rejects_unknown_config(bad_config):
    X = np.zeros((5, 2, 50))
    with pytest.raises(ValueError, match="unknown config"):
        adr.compute_masks(X, 250, configs=[bad_config], parallel=False)


def test_needed_idx_handles_multielement_array():
    """Fit-pool selection accepts a numpy array of labeled windows.

    ``needed_windows or []`` on a multi-element array raises on an ambiguous truth value, a bug the
    one-window-per-recording integration test missed.
    """
    assert adr._needed_idx(None, 40).tolist() == []
    # Deduplicated, sorted, out-of-range dropped, and no error.
    assert adr._needed_idx(np.array([30, 12, 5, 30, 99, -1]), 40).tolist() == [5, 12, 30]


def test_fit_pool_indices_always_includes_labelled_and_caps():
    """A capped pool keeps every labeled fragment and fills to the cap from the rest."""
    needed = np.array([3, 7, 900, 900])  # 900 repeated; all in range for n=1000
    idx = adr.fit_pool_indices(1000, needed, max_fit_fragments=50, seed=0)
    assert idx.size == 50 and idx.tolist() == sorted(idx.tolist())
    assert {3, 7, 900}.issubset(set(idx.tolist())), "labeled fragments must always be in the fit pool"
    # Uncapped, or a recording under the cap, gives the whole recording.
    assert adr.fit_pool_indices(20, np.array([1, 2, 3]), max_fit_fragments=None).tolist() == list(range(20))
    assert adr.fit_pool_indices(20, np.array([1, 2, 3]), max_fit_fragments=100).tolist() == list(range(20))


def test_psd_batch_parallel_matches_serial():
    """The dask path (one delayed task per fragment) returns the same spectra as the serial loop.

    Every other test calls psd_batch with parallel=False, so this is the only coverage of the dask branch
    the real runs use. It runs on dask's default scheduler, without a cluster.
    """
    pytest.importorskip("dask")
    fs, frag = 250, 5
    F, C, T = 6, 3, fs * frag
    X = (50e-6 * np.random.RandomState(7).randn(F, C, T)).astype(float)

    f_par, p_par = adr.psd_batch(X, fs, representation="welch", parallel=True)
    f_ser, p_ser = adr.psd_batch(X, fs, representation="welch", parallel=False)
    np.testing.assert_allclose(f_par, f_ser, rtol=1e-10, err_msg="freqs differ (parallel vs serial)")
    np.testing.assert_allclose(p_par, p_ser, rtol=1e-10, err_msg="psd differs (parallel vs serial)")


def test_grid_names():
    """The grid lists 2 peak-to-peak configurations, then 36 Welch spectral ones, all distinct."""
    names = adr.grid_config_names()
    assert len(names) == 38 and len(set(names)) == 38
    assert names[:2] == ["autoreject/self", "autoreject/self+CMR"]
    assert all(n.startswith("welch/") for n in names[2:])


def test_learn_threshold_flags_a_louder_block():
    """End to end on one channel: 10 fragments at 10 times the amplitude of 90 others are all flagged."""
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1.0, size=(100, 1250))
    loud = rng.choice(100, 10, replace=False)
    ref[loud] *= 10
    th, mask = adr.learn_threshold(ref, adr.amp_feat)
    quiet = np.setdiff1d(np.arange(100), loud)
    assert mask[loud].all(), "a fragment at 10 times the amplitude was kept"
    assert mask[quiet].mean() < 0.05
    assert np.isfinite(th)


def test_fit_matches_masks_and_rescoring():
    """fit_grid's flags equal compute_masks_grid's, and rescoring the pool from stored parameters is exact."""
    X = _noise(4, 30, 4, 1250)
    fits = adr.fit_grid(X, 250.0, parallel=False)
    masks = adr.compute_masks_grid(X, 250.0, parallel=False)
    assert list(fits) == adr.grid_config_names() == list(masks)
    for name, fit in fits.items():
        assert np.array_equal(fit.mask, masks[name]), name
        assert np.array_equal(adr.score_fragments(X, fit, parallel=False), fit.scores), name
        assert np.array_equal(adr.apply_fit(X, fit, parallel=False), fit.mask), name
        assert (fit.baselines is None) == (fit.kind == "amp"), name


def test_apply_fit_scores_new_fragments():
    """Stored parameters score fragments outside the pool, and refuse a different channel count or length."""
    X = _noise(5, 40, 3, 1250)
    fit = adr.fit_configs(X[:30], 250.0, configs=["v2/self"], parallel=False)["v2/self"]
    flags = adr.apply_fit(X[30:], fit, parallel=False)
    assert flags.shape == (10, 3) and flags.dtype == bool
    with pytest.raises(ValueError, match="channels"):
        adr.score_fragments(X[30:, :2], fit, parallel=False)
    with pytest.raises(ValueError, match="samples"):
        adr.score_fragments(X[30:, :, :1000], fit, parallel=False)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_or_inf_input_is_refused(bad):
    """A missing or infinite sample raises instead of silently disabling the cutoff search."""
    X = _noise(6, 20, 3, 1250)
    X[4, 2, 17] = bad
    with pytest.raises(ValueError, match="fragment 4, channel 2"):
        adr.compute_masks(X, 250.0, parallel=False)
    with pytest.raises(ValueError, match="NaN or inf"):
        adr.compute_masks_grid(X, 250.0, parallel=False)


def test_dead_channel_is_never_flagged():
    """Limit of the rule: a channel that is flat throughout is judged only against itself, so never flagged."""
    X = _noise(7, 60, 4, 1250)
    X[:, 2] = 0.0  # exact zero
    X[:, 3] = 1e-3  # constant 1 mV
    masks = adr.compute_masks(X, 250.0, parallel=False)
    for name, m in masks.items():
        assert not m[:, 2].any() and not m[:, 3].any(), name


def test_self_configs_are_gain_invariant():
    """Self configs give a channel the same flags at any gain; powers of two keep the arithmetic exact."""
    X = _noise(8, 60, 3, 1250)
    X[:, 1] = 8 * X[:, 0]
    X[:, 2] = X[:, 0] / 8
    masks = adr.compute_masks(X, 250.0, configs=SELF_CONFIGS, parallel=False)
    for name, m in masks.items():
        assert np.array_equal(m[:, 0], m[:, 1]) and np.array_equal(m[:, 0], m[:, 2]), name


def test_cmr_depends_on_gain():
    """Limit of the rule: with the common-mode pool, a channel scaled up against correlated neighbours is
    flagged throughout, while its self config flags are unchanged."""
    rng = np.random.default_rng(9)
    X = rng.normal(0, 50e-6, size=(60, 1, 1250)) + rng.normal(0, 10e-6, size=(60, 6, 1250))
    loud = X.copy()
    loud[:, 0] *= 8
    before = adr.compute_masks(X, 250.0, configs=["autoreject/self+CMR", "autoreject/self"], parallel=False)
    after = adr.compute_masks(loud, 250.0, configs=["autoreject/self+CMR", "autoreject/self"], parallel=False)
    assert after["autoreject/self+CMR"][:, 0].all()
    assert not before["autoreject/self+CMR"][:, 0].all()
    assert np.array_equal(before["autoreject/self"][:, 0], after["autoreject/self"][:, 0])


@pytest.mark.parametrize("name", adr.CONFIG_NAMES)
def test_one_fragment_raises(name):
    """A pool of one fragment cannot be split, for any configuration."""
    with pytest.raises(ValueError):
        adr.compute_masks(_noise(10, 1, 3, 1250), 250.0, configs=[name], parallel=False)


def test_two_fragments():
    """With two fragments the common-mode configurations raise and the self configurations run."""
    X = _noise(11, 2, 3, 1250)
    for name in CMR_CONFIGS:
        with pytest.raises(ValueError):
            adr.compute_masks(X, 250.0, configs=[name], parallel=False)
    masks = adr.compute_masks(X, 250.0, configs=SELF_CONFIGS, parallel=False)
    assert all(m.shape == (2, 3) for m in masks.values())


def test_common_mode_needs_two_channels():
    """One channel has no other channels to reconstruct it from, so +CMR raises; self configs run."""
    X = _noise(12, 20, 1, 1250)
    for name in CMR_CONFIGS:
        with pytest.raises(ValueError, match="at least 2 channels"):
            adr.compute_masks(X, 250.0, configs=[name], parallel=False)
    assert adr.compute_masks(X, 250.0, configs=SELF_CONFIGS, parallel=False)["autoreject/self"].shape == (20, 1)


def test_two_channel_reconstruction_is_the_other_channel():
    """With two channels, each channel's common-mode reconstruction is the other channel."""
    X = _noise(13, 5, 2, 100)
    assert np.array_equal(adr._cmr_wave(X, 0), X[:, 1])
    assert np.array_equal(adr._cmr_wave(X, 1), X[:, 0])
