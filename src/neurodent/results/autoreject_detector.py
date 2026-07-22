"""Adapted autoreject artifact detector (issue #208).

**Not stock autoreject.** This keeps autoreject's cross-validated per-channel threshold learning, but
**rejects** a fragment over the learned threshold instead of *repairing* it by spatial interpolation — so
**no montage / electrode positions are needed** (rodent iEEG has none). It also operates on **free-running**
5 s fragments, not evoked/epoched trials: the objective's "median fragment" is a *typical-background*
reference (a serviceable "what does clean look like" anchor as long as most fragments on a channel are
clean — see the module note on the median-reference limit for mostly-artifact channels).

Six configs are auditioned = 3 arms × {self, +CMR}:

===================  ========================================  ======================
arm                  feature                                   per-fold baseline
===================  ========================================  ======================
``autoreject``       peak-to-peak amplitude                    none (baseline-free)
``v2``               RMS-dB deviation of the LPSD log-spectrum  training **median** spectrum
``v3``               RMS-dB deviation of the LPSD log-spectrum  training **mean** spectrum
===================  ========================================  ======================

``+CMR`` appends each channel's all-but-one median reconstruction (a spatial common-mode reference) to that
channel's own fragments, stratified real-vs-reconstructed in the CV. The final mask is over the original
fragments only.

**Leak-free CV.** For each candidate threshold, keep the training fragments under it, average, and compare
to the *validation-fold median*; pick ``best_th`` minimising the fold-averaged
``RMS(median(val ref) − mean(kept train ref))``, then apply that one threshold over all fragments on a
full-data baseline (train-then-apply). Both the candidate grid **and** the per-fold baseline come from
**training fragments only** — validation enters solely through the objective's reference median. (A
whole-pool "v1" baseline leaked and is intentionally absent; :func:`_fold_solve`'s ``leaky_baseline`` hook
exists only so the regression test can prove the leak-free path has teeth.)

Two exact algebraic speedups (not approximations): the validation median is threshold-independent (computed
once per fold); the kept set grows monotonically with the threshold, so sorting by feature and taking a
running sum yields every candidate's kept-mean in one pass.

Output of :func:`compute_masks`: ``{config_name: (F, C) bool}`` with **True = REJECT**. Score with
:func:`neurodent.results.scoring.score_mask` (``True = REJECT``), or ``score_keep_mask(~mask)`` for the
``FILTER_REGISTRY`` ``True = KEEP`` convention.

Verified by ``tests/test_autoreject_detector.py`` (leak invariant + objective == brute force). LPSD uses the
reference ``lpsd`` package (a hand-rolled estimator invented a spurious ~110 Hz bump); ~19 log bins in
[1, 200] Hz on a 5 s fragment.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

import lpsd as _lpsd

try:  # dask is a core dependency, but keep the module importable without a cluster
    import dask as _dask
except ImportError:  # pragma: no cover
    _dask = None

# LPSD band + resolution, and CV shape — the handoff's reference values.
FMIN, FMAX = 1.0, 200.0
N_FREQ_DES, N_AVG_DES = 40, 100          # a 5 s fragment supports ~19 log bins in [1, 200] Hz
N_FOLDS, N_CAND = 5, 30


# --------------------------------------------------------------------------- features

def amp_feat(sig: np.ndarray, base=None) -> np.ndarray:
    """``(N, T)`` waveforms → ``(N,)`` peak-to-peak amplitude. Baseline-free (``base`` ignored)."""
    return sig.max(1) - sig.min(1)


def spec_feat(sig: np.ndarray, base: np.ndarray) -> np.ndarray:
    """``(N, P)`` dB log-spectra → ``(N,)`` RMS-dB deviation from the baseline spectrum ``base`` ``(P,)``."""
    return np.sqrt(np.mean((sig - base) ** 2, axis=1))


# --------------------------------------------------------------------------- LPSD

def _lpsd_inband(sig: np.ndarray, fs: float, n_frequencies: int, n_averages: int,
                 fmin: float, fmax: float):
    """One channel-fragment → (freqs, psd) sliced to [fmin, fmax] via the reference ``lpsd`` estimator."""
    idx = np.arange(sig.shape[-1]) / fs
    r = _lpsd.lpsd(pd.Series(sig, index=idx), sample_rate=fs, n_frequencies=n_frequencies,
                   n_averages=n_averages, detrending_order=0)
    rf = r.index.to_numpy()
    band = (rf >= fmin) & (rf <= fmax)
    return rf[band], np.real(r["psd"].to_numpy())[band]


def psd_batch(X: np.ndarray, fs: float, *, fmin=FMIN, fmax=FMAX, n_frequencies=N_FREQ_DES,
              n_averages=N_AVG_DES, parallel=True):
    """``(F, C, T)`` volts → ``(freqs (P,), psd (F, C, P))`` in V²/Hz.

    LPSD is the per-fragment cost of the spectral arms, so the per-fragment work is fanned across dask
    (one task per fragment computes all its channels) when dask is available and ``parallel`` is set —
    reusing whatever :class:`~dask.distributed.Client` the caller has active. Falls back to a serial loop
    otherwise. The amplitude arm never calls this.
    """
    Fn, Cn, _ = X.shape

    def _frag(xf):  # xf: (C, T) -> (freqs, (C, P))
        rows = [_lpsd_inband(xf[c], fs, n_frequencies, n_averages, fmin, fmax) for c in range(xf.shape[0])]
        return rows[0][0], np.stack([p for _, p in rows], axis=0)

    if parallel and _dask is not None and Fn > 1:
        results = _dask.compute(*[_dask.delayed(_frag)(X[f]) for f in range(Fn)])
    else:
        results = [_frag(X[f]) for f in range(Fn)]
    freqs = results[0][0]
    psd = np.stack([r[1] for r in results], axis=0)  # (F, C, P)
    return freqs, psd


# --------------------------------------------------------------------------- leak-free CV

def _fold_solve(pool: np.ndarray, tr: np.ndarray, te: np.ndarray, feat_fn, baseline_fn, n_cand: int,
                leaky_baseline: bool = False):
    """One CV fold. Returns ``(cands (n_cand,), errs (n_cand,), kept_counts (n_cand,), feat_tr (n_tr,))``.

    Leak-free: the candidate grid and the ``baseline_fn`` baseline come from **training** fragments
    (``pool[tr]``) only, so ``feat_tr`` — which determines every kept-SET (membership: fragments with
    feature ≤ a threshold) — is invariant to validation. Validation (``pool[te]``) enters ONLY through the
    objective median. ``errs[k]`` is ``RMS(median(pool[te]) − mean(kept training))`` where "kept" = training
    fragments with feature ≤ ``cands[k]`` (``nan`` when < 2 are kept); ``kept_counts[k]`` is that count.

    ``leaky_baseline=True`` reproduces the discarded "v1" leak (baseline from the whole pool incl.
    validation) — for the regression control ONLY; never use it in production.
    """
    base_src = pool if leaky_baseline else pool[tr]
    base = None if baseline_fn is None else baseline_fn(base_src, axis=0)
    feat_tr = feat_fn(pool[tr], base)
    cands = np.quantile(feat_tr, np.linspace(0.0, 1.0, n_cand))   # full rejection range, quantile-spaced
    med = np.median(pool[te], axis=0)                            # objective reference: validation only
    order = np.argsort(feat_tr, kind="stable")
    csum = np.cumsum(pool[tr][order], axis=0)                    # running sum -> every candidate's kept-mean
    ns = np.searchsorted(feat_tr[order], cands, side="right")    # kept count at each candidate
    errs = np.full(n_cand, np.nan)
    for k, n in enumerate(ns):
        if n < 2:
            continue
        errs[k] = np.sqrt(np.mean((med - csum[n - 1] / n) ** 2))
    return cands, errs, ns, feat_tr


def learn_threshold(ref: np.ndarray, feat_fn, baseline_fn=None, aug_ref=None,
                    n_folds=N_FOLDS, n_cand=N_CAND, seed=0):
    """Leak-free CV per channel → ``(best_th, mask)``.

    Args:
        ref: ``(N, T)`` waveforms (amplitude arm) or ``(N, P)`` dB spectra (spectral arms) for one channel.
        feat_fn: :func:`amp_feat` or :func:`spec_feat`.
        baseline_fn: ``None`` (amplitude) or ``np.median`` / ``np.mean`` (spectral per-fold baseline).
        aug_ref: optional CMR reconstruction ``(N, …)`` appended to ``ref`` as a self+CMR pool, stratified
            real-vs-reconstructed in the split.
        seed, n_folds, n_cand: CV shape.

    Returns:
        ``best_th`` (fold-averaged candidate at the CV argmin) and ``mask`` ``(N,)`` bool, True = REJECT,
        applied to ``ref`` on a full-pool baseline (train-then-apply).
    """
    F0 = ref.shape[0]
    if aug_ref is not None:
        pool = np.concatenate([ref, aug_ref], axis=0)
        y = np.r_[np.zeros(F0), np.ones(F0)]
        splits = StratifiedShuffleSplit(n_folds, test_size=0.2, random_state=seed).split(pool, y)
    else:
        pool = ref
        splits = ShuffleSplit(n_folds, test_size=0.2, random_state=seed).split(pool)

    cand_cols, err_cols = [], []
    for tr, te in splits:
        cands, errs, _, _ = _fold_solve(pool, tr, te, feat_fn, baseline_fn, n_cand)
        cand_cols.append(cands)
        err_cols.append(errs)
    cands = np.stack(cand_cols, axis=1)   # (n_cand, n_folds)
    errs = np.stack(err_cols, axis=1)
    with warnings.catch_warnings():
        # the smallest candidate keeps <2 fragments in every fold -> an all-NaN row; nanmean warns benignly
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_err = np.nanmean(errs, axis=1)
    best_k = n_cand - 1 if np.all(np.isnan(mean_err)) else int(np.nanargmin(mean_err))
    best_th = np.nanmean(cands[best_k])

    base_full = None if baseline_fn is None else baseline_fn(pool, axis=0)
    return best_th, feat_fn(ref, base_full) > best_th


# --------------------------------------------------------------------------- fit-pool selection

def _needed_idx(needed_windows, n_fragments: int) -> np.ndarray:
    """Labelled window indices in ``[0, n_fragments)``, sorted + unique.

    MUST be robust to a numpy-array ``needed_windows`` — the caller passes ``consensus["window"].unique()``,
    and ``needed_windows or []`` would evaluate an array's ambiguous truth value once it has >1 element
    (the bug this replaced). ``None`` → empty.
    """
    if needed_windows is None:
        return np.array([], dtype=int)
    return np.array(sorted({int(w) for w in needed_windows if 0 <= int(w) < n_fragments}), dtype=int)


def fit_pool_indices(n_fragments: int, needed_windows=None, max_fit_fragments=None, seed=0) -> np.ndarray:
    """Fragment indices to fit the per-recording CV on: **always** the labelled ``needed_windows`` (so the
    scored fragments are covered), plus a capped random subsample of the rest when ``max_fit_fragments``
    bounds a long recording. Returns a sorted int array; the whole recording when uncapped.

    The CV threshold needs only a representative, mostly-clean pool, so a random subsample of a 24 h
    recording (~17k fragments) is statistically equivalent while bounding memory/compute.
    """
    needed = _needed_idx(needed_windows, n_fragments)
    if not max_fit_fragments or n_fragments <= max_fit_fragments:
        return np.arange(n_fragments)
    others = np.setdiff1d(np.arange(n_fragments), needed)
    k = max(0, int(max_fit_fragments) - needed.size)
    extra = others if k >= others.size else np.random.RandomState(seed).choice(others, size=k, replace=False)
    return np.sort(np.concatenate([needed, extra]).astype(int))


# --------------------------------------------------------------------------- the 6 configs

# name -> (ref_kind, feat_fn, baseline_fn, cmr):  ref_kind selects waveform (amp) vs dB-spectrum (spec);
# cmr in {None, "wave", "spec"} picks the CMR reconstruction domain.
_CONFIGS = {
    "autoreject/self":     ("amp",  amp_feat,  None,      None),
    "autoreject/self+CMR": ("amp",  amp_feat,  None,      "wave"),
    "v2/self":             ("spec", spec_feat, np.median, None),
    "v2/self+CMR":         ("spec", spec_feat, np.median, "spec"),
    "v3/self":             ("spec", spec_feat, np.mean,   None),
    "v3/self+CMR":         ("spec", spec_feat, np.mean,   "spec"),
}
CONFIG_NAMES = list(_CONFIGS)


def _cmr_wave(X: np.ndarray, c: int) -> np.ndarray:
    """All-but-one median waveform for channel ``c``: ``(F, T)``."""
    return np.median(np.delete(X, c, axis=1), axis=1)


def compute_masks(X: np.ndarray, fs: float, *, configs=CONFIG_NAMES, n_folds=N_FOLDS, n_cand=N_CAND,
                  seed=0, fmin=FMIN, fmax=FMAX, n_frequencies=N_FREQ_DES, n_averages=N_AVG_DES,
                  parallel=True) -> dict[str, np.ndarray]:
    """Run the requested configs on ``X`` ``(F, C, T)`` float **volts**.

    Returns ``{config_name: (F, C) bool}`` with True = REJECT. LPSD spectra (and their all-but-one CMR
    reconstructions) are computed once, only if a spectral config is requested. Each config runs an
    independent per-channel leak-free CV (:func:`learn_threshold`).
    """
    X = np.asarray(X, dtype=float)
    F, C, _ = X.shape
    unknown = [n for n in configs if n not in _CONFIGS]
    if unknown:
        raise ValueError(f"unknown config(s): {unknown}. Available: {CONFIG_NAMES}")

    need_spec = any(_CONFIGS[n][0] == "spec" for n in configs)
    S = recon_spec = None
    if need_spec:
        _, psd = psd_batch(X, fs, fmin=fmin, fmax=fmax, n_frequencies=n_frequencies,
                           n_averages=n_averages, parallel=parallel)
        S = 10 * np.log10(psd + 1e-20)                                              # (F, C, P) dB
        recon_spec = np.stack([np.median(np.delete(S, c, axis=1), axis=1) for c in range(C)], axis=1)

    masks = {}
    for name in configs:
        ref_kind, feat_fn, baseline_fn, cmr = _CONFIGS[name]
        ref_all = X if ref_kind == "amp" else S
        mask = np.zeros((F, C), dtype=bool)
        for c in range(C):
            if cmr == "wave":
                aug = _cmr_wave(X, c)
            elif cmr == "spec":
                aug = recon_spec[:, c, :]
            else:
                aug = None
            _, mask[:, c] = learn_threshold(ref_all[:, c, :], feat_fn, baseline_fn=baseline_fn,
                                            aug_ref=aug, n_folds=n_folds, n_cand=n_cand, seed=seed)
        masks[name] = mask
    return masks
