"""Adapted autoreject artifact detector.

The detector flags a fragment on a channel when the fragment's score exceeds a cutoff learned from that
channel's own fragments. It keeps autoreject's cross-validated per-channel cutoff search (Jas et al.,
2017) but drops the repair by spatial interpolation and the consensus vote across channels, so it needs no
electrode positions. Fragments are free-running windows (5 s by default), not epochs locked to an event.

A configuration combines an arm, a spectral resolution for the spectral arms, and an optional common-mode
comparison.

Arms:

- ``autoreject``: peak-to-peak amplitude of the fragment.
- ``v2``: RMS over frequency bins of the fragment's Welch dB spectrum minus the channel's median spectrum.
- ``v3``: the same, against the channel's mean spectrum.

With ``+CMR``, a channel's pool also holds one reconstruction per fragment built from the other channels:
the per-sample median of their waveforms for peak-to-peak, or the per-bin median of their dB spectra for
the spectral arms. The splits stratify real against reconstructed rows, and only real rows are flagged.

Cutoff search, per channel: 5 random 80/20 splits of the pool; 30 candidates at quantiles 0 to 1 of the
training scores; a candidate's error is the RMS of the median of the held-out rows minus the mean of the
training rows scoring at or under it. The candidate index with the lowest mean error over the splits wins,
and the cutoff is the mean over splits of that candidate. Baselines and candidates come from training rows
only, so held-out rows enter only through the median. A fragment is flagged when its score, measured against
a baseline over the whole pool, exceeds the cutoff.

Two shortcuts are exact: the held-out median does not depend on the candidate, so it is computed once per
split; and the kept set grows with the candidate, so one sort and a running sum give every candidate's kept
mean.

:func:`fit_configs` and :func:`fit_grid` return a :class:`DetectorFit` per configuration with each channel's
cutoff, baseline spectrum and fragment scores. :func:`score_fragments` and :func:`apply_fit` score and flag
any fragments with those stored parameters. :func:`compute_masks` and :func:`compute_masks_grid` return
only the flags, ``{config: (F, C) bool}`` with True meaning reject; score them with
:func:`neurodent.results.scoring.score_mask`, or pass ``~mask`` to ``score_keep_mask``.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import NamedTuple

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

try:  # dask is a core dependency; the module still imports without it
    import dask as _dask
except ImportError:  # pragma: no cover
    _dask = None

# Spectral band and resolution, and the shape of the cutoff search.
FMIN, FMAX = 1.0, 200.0
N_FREQ_DES, N_AVG_DES = 40, 100
N_FOLDS, N_CAND = 5, 30
# Fragments per recording the cutoff is fit on, shared by every caller (labeled windows are always kept).
MAX_FIT_FRAGMENTS = 5000


# Features


def amp_feat(sig: np.ndarray, base=None) -> np.ndarray:
    """Return the peak-to-peak amplitude of each row.

    Args:
        sig: ``(N, T)`` waveforms.
        base: Ignored; peak-to-peak has no baseline.

    Returns:
        ``(N,)`` peak-to-peak amplitudes.
    """
    return sig.max(1) - sig.min(1)


def spec_feat(sig: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Return each row's RMS deviation in dB from a baseline spectrum.

    Args:
        sig: ``(N, P)`` dB spectra.
        base: ``(P,)`` baseline dB spectrum.

    Returns:
        ``(N,)`` RMS over bins of ``sig - base``.
    """
    return np.sqrt(np.mean((sig - base) ** 2, axis=1))


def _welch_inband(sig: np.ndarray, fs: float, n_frequencies: int, n_averages: int, fmin: float, fmax: float):
    """Return one channel-fragment's Welch spectrum within ``[fmin, fmax]``.

    The grid's ``(n_frequencies, n_averages)`` are mapped onto Welch's ``(nperseg, noverlap)``: ``nperseg``
    is chosen so about ``n_frequencies`` linear bins land in the band, and ``noverlap`` so about
    ``n_averages`` segments are averaged, inverting ``K = (N - noverlap) / (nperseg - noverlap)``, both
    clamped to what a short fragment supports.

    Returns:
        ``(freqs, psd)`` restricted to the band, psd in V^2/Hz.
    """
    from scipy.signal import welch

    N = int(sig.shape[-1])
    nperseg = int(np.clip(round(n_frequencies * fs / (fmax - fmin)), 16, N))
    K = max(2, int(n_averages))  # target number of segments to average
    noverlap = int(np.clip((K * nperseg - N) // (K - 1), 0, nperseg - 1))  # invert K, clamp to [0, nperseg)
    f, p = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    band = (f >= fmin) & (f <= fmax)
    return f[band], np.real(p)[band]


def psd_batch(
    X: np.ndarray,
    fs: float,
    *,
    representation="welch",
    fmin=FMIN,
    fmax=FMAX,
    n_frequencies=N_FREQ_DES,
    n_averages=N_AVG_DES,
    parallel=True,
):
    """Return the in-band power spectrum of every fragment and channel.

    With dask available and ``parallel`` set, each fragment is one delayed task computing all its channels,
    run on whatever dask client the caller has active. The fragment travels in the task graph rather than as
    a scattered future, so a lost worker is recomputed instead of cancelling the run. Otherwise a serial
    loop runs. The peak-to-peak arm never calls this.

    Args:
        X: ``(F, C, T)`` fragments in volts.
        fs: Sampling rate in Hz.
        representation: Spectral estimator; only ``"welch"``.
        fmin: Lowest frequency kept, Hz.
        fmax: Highest frequency kept, Hz.
        n_frequencies: Approximate number of in-band bins.
        n_averages: Approximate number of segments averaged.
        parallel: Fan fragments out over dask when available.

    Returns:
        ``(freqs (P,), psd (F, C, P))`` with psd in V^2/Hz.

    Raises:
        ValueError: If ``representation`` is not ``"welch"``.
    """
    Fn, Cn, _ = X.shape
    inband = {"welch": _welch_inband}.get(representation)
    if inband is None:
        raise ValueError(f"unknown representation {representation!r}; use 'welch'")

    def _frag(xf):  # xf: (C, T) -> (freqs, (C, P))
        rows = [inband(xf[c], fs, n_frequencies, n_averages, fmin, fmax) for c in range(xf.shape[0])]
        return rows[0][0], np.stack([p for _, p in rows], axis=0)

    if parallel and _dask is not None and Fn > 1:
        results = _dask.compute(*[_dask.delayed(_frag)(X[f]) for f in range(Fn)])
    else:
        results = [_frag(X[f]) for f in range(Fn)]
    freqs = results[0][0]
    psd = np.stack([r[1] for r in results], axis=0)  # (F, C, P)
    return freqs, psd


def log_spectra(X: np.ndarray, fs: float, *, parallel=True, **spectral):
    """Return every fragment and channel's in-band spectrum in dB.

    Args:
        X: ``(F, C, T)`` fragments in volts.
        fs: Sampling rate in Hz.
        parallel: Passed to :func:`psd_batch`.
        **spectral: ``representation``, ``fmin``, ``fmax``, ``n_frequencies`` and ``n_averages`` for
            :func:`psd_batch`.

    Returns:
        ``(freqs (P,), spectra (F, C, P))`` with spectra in dB, floored at -200 dB.
    """
    freqs, psd = psd_batch(X, fs, parallel=parallel, **spectral)
    return freqs, 10 * np.log10(psd + 1e-20)


# Cutoff search


def _fold_solve(
    pool: np.ndarray, tr: np.ndarray, te: np.ndarray, feat_fn, baseline_fn, n_cand: int, leaky_baseline: bool = False
):
    """Evaluate every candidate cutoff on one split.

    The candidates and the baseline come from training rows (``pool[tr]``) only, so ``feat_tr``, which
    decides every kept set, does not depend on the held-out rows. Held-out rows (``pool[te]``) enter only
    through the median. ``errs[k]`` is the RMS of ``median(pool[te])`` minus the mean of the training rows
    whose feature is at or under ``cands[k]``, or NaN when fewer than 2 are kept.

    Args:
        pool: ``(N, ...)`` rows of one channel (waveforms or dB spectra).
        tr: Training row indices.
        te: Held-out row indices.
        feat_fn: :func:`amp_feat` or :func:`spec_feat`.
        baseline_fn: None, ``np.median`` or ``np.mean``.
        n_cand: Number of candidate cutoffs.
        leaky_baseline: Take the baseline over the whole pool, held-out rows included. This exists only so
            a test can show that the training-only baseline matters; never use it otherwise.

    Returns:
        ``(cands, errs, kept_counts, feat_tr)``: candidates and errors ``(n_cand,)``, the number of training
        rows kept at each candidate ``(n_cand,)``, and the training features ``(n_tr,)``.
    """
    base_src = pool if leaky_baseline else pool[tr]
    base = None if baseline_fn is None else baseline_fn(base_src, axis=0)
    feat_tr = feat_fn(pool[tr], base)
    cands = np.quantile(feat_tr, np.linspace(0.0, 1.0, n_cand))  # quantile-spaced over the full range
    med = np.median(pool[te], axis=0)  # the reference: held-out rows only
    order = np.argsort(feat_tr, kind="stable")
    csum = np.cumsum(pool[tr][order], axis=0)  # running sum gives every candidate's kept mean
    ns = np.searchsorted(feat_tr[order], cands, side="right")  # kept count at each candidate
    errs = np.full(n_cand, np.nan)
    for k, n in enumerate(ns):
        if n < 2:
            continue
        errs[k] = np.sqrt(np.mean((med - csum[n - 1] / n) ** 2))
    return cands, errs, ns, feat_tr


class ChannelFit(NamedTuple):
    """One channel's fitted cutoff, baseline and fragment scores (the return of :func:`fit_channel`).

    Attributes:
        threshold: The learned cutoff.
        baseline: ``(P,)`` baseline dB spectrum over the whole pool for spectral arms, None for peak-to-peak.
        scores: ``(N,)`` score of each real fragment against ``baseline``.
    """

    threshold: float
    baseline: np.ndarray | None
    scores: np.ndarray


def fit_channel(ref: np.ndarray, feat_fn, baseline_fn=None, aug_ref=None, n_folds=N_FOLDS, n_cand=N_CAND, seed=0):
    """Learn one channel's cutoff by the cross-validated search described in the module docstring.

    Args:
        ref: ``(N, T)`` waveforms (peak-to-peak) or ``(N, P)`` dB spectra (spectral arms) of one channel.
        feat_fn: :func:`amp_feat` or :func:`spec_feat`.
        baseline_fn: None for peak-to-peak, ``np.median`` or ``np.mean`` for the spectral arms.
        aug_ref: Optional ``(N, ...)`` common-mode reconstruction appended to ``ref``; the splits then
            stratify real against reconstructed rows.
        n_folds: Number of random 80/20 splits.
        n_cand: Number of candidate cutoffs.
        seed: Seed of the splits.

    Returns:
        A :class:`ChannelFit`; ``scores > threshold`` are the flagged fragments of ``ref``.
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
    cands = np.stack(cand_cols, axis=1)  # (n_cand, n_folds)
    errs = np.stack(err_cols, axis=1)
    with warnings.catch_warnings():
        # The smallest candidate keeps fewer than 2 rows in every split, an all-NaN row that nanmean warns on.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_err = np.nanmean(errs, axis=1)
    best_k = n_cand - 1 if np.all(np.isnan(mean_err)) else int(np.nanargmin(mean_err))
    best_th = np.nanmean(cands[best_k])

    base_full = None if baseline_fn is None else baseline_fn(pool, axis=0)
    return ChannelFit(best_th, base_full, feat_fn(ref, base_full))


def learn_threshold(ref: np.ndarray, feat_fn, baseline_fn=None, aug_ref=None, n_folds=N_FOLDS, n_cand=N_CAND, seed=0):
    """Learn one channel's cutoff and flag its fragments.

    Args:
        ref: See :func:`fit_channel`.
        feat_fn: See :func:`fit_channel`.
        baseline_fn: See :func:`fit_channel`.
        aug_ref: See :func:`fit_channel`.
        n_folds: See :func:`fit_channel`.
        n_cand: See :func:`fit_channel`.
        seed: See :func:`fit_channel`.

    Returns:
        ``(threshold, mask)`` with ``mask`` ``(N,)`` bool, True meaning reject.
    """
    fit = fit_channel(ref, feat_fn, baseline_fn=baseline_fn, aug_ref=aug_ref, n_folds=n_folds, n_cand=n_cand, seed=seed)
    return fit.threshold, fit.scores > fit.threshold


# Fit-pool selection


def _needed_idx(needed_windows, n_fragments: int) -> np.ndarray:
    """Return the labeled window indices within ``[0, n_fragments)``, sorted and unique.

    Accepts a numpy array (callers pass ``consensus["window"].unique()``), where ``needed_windows or []``
    would raise on an ambiguous truth value. None gives an empty array.
    """
    if needed_windows is None:
        return np.array([], dtype=int)
    return np.array(sorted({int(w) for w in needed_windows if 0 <= int(w) < n_fragments}), dtype=int)


def fit_pool_indices(n_fragments: int, needed_windows=None, max_fit_fragments=None, seed=0) -> np.ndarray:
    """Return the fragment indices a recording's cutoffs are fit on.

    The labeled ``needed_windows`` are always included, so every scored fragment is in the pool; when
    ``max_fit_fragments`` bounds a long recording, a seeded random subsample of the other fragments fills
    the pool to the cap.

    Args:
        n_fragments: Fragments in the recording.
        needed_windows: Labeled window indices to always include, or None.
        max_fit_fragments: Pool cap; None or 0 uses the whole recording.
        seed: Seed of the subsample.

    Returns:
        Sorted int indices; the whole recording when uncapped or when it fits under the cap.
    """
    needed = _needed_idx(needed_windows, n_fragments)
    if not max_fit_fragments or n_fragments <= max_fit_fragments:
        return np.arange(n_fragments)
    others = np.setdiff1d(np.arange(n_fragments), needed)
    k = max(0, int(max_fit_fragments) - needed.size)
    extra = others if k >= others.size else np.random.RandomState(seed).choice(others, size=k, replace=False)
    return np.sort(np.concatenate([needed, extra]).astype(int))


class FitEpochs(NamedTuple):
    """One recording's fit-pool fragments (the return of :func:`build_fit_epochs`).

    Attributes:
        X: ``(n_pool, C, T)`` fragments in volts.
        channels: Canonical channel abbreviations aligned to ``X``'s channel axis.
        fs: Integer sampling rate in Hz.
        grid_times: Start of each kept fragment in seconds.
    """

    X: np.ndarray
    channels: list
    fs: int
    grid_times: np.ndarray


def build_fit_epochs(
    lro,
    needed_windows=None,
    *,
    fragment_len_s=5.0,
    apply_notch_filter=True,
    max_fit_fragments=MAX_FIT_FRAGMENTS,
    seed=0,
) -> "FitEpochs":
    """Cut one recording into the fragments its cutoffs are fit on.

    Fragments come from :class:`~neurodent.analysis.long_recording_analyzer.LongRecordingAnalyzer`, which
    applies the 60 Hz notch to each fragment separately, as the rater view does. The labeled
    ``needed_windows`` are always kept (:func:`fit_pool_indices`) along with a seeded subsample of the rest
    up to ``max_fit_fragments``. A ragged final fragment is skipped so every fragment has the same length.
    The detector reads no labels, so keeping labeled windows in the pool does not fit the cutoffs to them.

    Args:
        lro: A LongRecordingOrganizer.
        needed_windows: Labeled window indices to always include, or None.
        fragment_len_s: Fragment length in seconds.
        apply_notch_filter: Notch each fragment at 60 Hz.
        max_fit_fragments: Pool cap; None or 0 uses the whole recording.
        seed: Seed of the subsample.

    Returns:
        A :class:`FitEpochs`.

    Raises:
        ValueError: If the recording has no full-length fragment.
    """
    # Lazy imports: neurodent.analysis imports neurodent.results, so a top-level import would be circular.
    from neurodent.analysis.long_recording_analyzer import LongRecordingAnalyzer
    from neurodent.core.utils import resolve_channels

    lan = LongRecordingAnalyzer(lro, fragment_len_s=fragment_len_s, apply_notch_filter=apply_notch_filter)
    idxs = fit_pool_indices(lan.n_fragments, needed_windows, max_fit_fragments, seed)
    win = int(round(fragment_len_s * lan.f_s))
    frags, keep = [], []
    for i in idxs:
        f = lan.get_fragment_np(int(i))  # (n_samples, C) in uV
        if f.shape[0] != win:  # ragged final fragment; labeled windows are never at the edge
            continue
        frags.append(f.T * 1e-6)  # (C, T) in volts
        keep.append(int(i))
    if not frags:
        raise ValueError(f"no full-length fragments (win={win} samples) to fit on")
    X = np.stack(frags, axis=0)  # (n_pool, C, T) in volts
    ch_names = resolve_channels(list(lan.channel_names))
    grid_times = np.array(keep, dtype=float) * fragment_len_s
    return FitEpochs(X, ch_names, int(lan.f_s), grid_times)


# Configurations

# name: (kind, feat_fn, baseline_fn, cmr). kind is "amp" (waveforms) or "spec" (dB spectra); cmr is None,
# "wave" or "spec", the domain of the common-mode reconstruction.
_CONFIGS = {
    "autoreject/self": ("amp", amp_feat, None, None),
    "autoreject/self+CMR": ("amp", amp_feat, None, "wave"),
    "v2/self": ("spec", spec_feat, np.median, None),
    "v2/self+CMR": ("spec", spec_feat, np.median, "spec"),
    "v3/self": ("spec", spec_feat, np.mean, None),
    "v3/self+CMR": ("spec", spec_feat, np.mean, "spec"),
}
CONFIG_NAMES = list(_CONFIGS)


def _cmr_wave(X: np.ndarray, c: int) -> np.ndarray:
    """Return channel ``c``'s all-but-one median waveform, ``(F, T)``."""
    return np.median(np.delete(X, c, axis=1), axis=1)


def _validated(X) -> np.ndarray:
    """Return ``X`` as float64 ``(F, C, T)`` after refusing missing or infinite samples.

    The cutoff search has no rule for missing samples: one NaN turns a channel's cutoff into NaN or the
    largest training score, and through the common-mode medians it reaches every other channel.

    Raises:
        ValueError: If ``X`` is not 3-D or holds NaN or inf.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"expected (fragments, channels, samples), got shape {X.shape}")
    finite = np.isfinite(X)
    if not finite.all():
        bad_cells = ~finite.all(axis=2)
        f, c = (int(i) for i in np.argwhere(bad_cells)[0])
        raise ValueError(
            f"input holds NaN or inf samples in {int(bad_cells.sum())} fragment-channel cells, first at "
            f"fragment {f}, channel {c}; drop or repair them before fitting"
        )
    return X


@dataclasses.dataclass(frozen=True)
class DetectorFit:
    """One configuration fitted on one recording's pool.

    Attributes:
        config: Configuration name, as listed by :func:`grid_config_names` or ``CONFIG_NAMES``.
        kind: ``"amp"`` for peak-to-peak, ``"spec"`` for the spectral arms.
        fs: Sampling rate the fit used, Hz.
        n_samples: Samples per fragment the fit used.
        thresholds: ``(C,)`` learned cutoff per channel.
        scores: ``(F, C)`` score of each pool fragment.
        baselines: ``(C, P)`` baseline dB spectrum per channel; None for peak-to-peak.
        freqs: ``(P,)`` bin frequencies of ``baselines``; None for peak-to-peak.
        spectral: Spectral settings (``representation``, ``fmin``, ``fmax``, ``n_frequencies``,
            ``n_averages``); None for peak-to-peak.
    """

    config: str
    kind: str
    fs: float
    n_samples: int
    thresholds: np.ndarray
    scores: np.ndarray
    baselines: np.ndarray | None = None
    freqs: np.ndarray | None = None
    spectral: dict | None = None

    @property
    def mask(self) -> np.ndarray:
        """``(F, C)`` flags on the pool fragments, True meaning reject."""
        return self.scores > self.thresholds


def _fit_configs(
    X, fs, configs, *, representation, n_folds, n_cand, seed, fmin, fmax, n_frequencies, n_averages, parallel
):
    """Fit ``configs`` on already validated ``X``; see :func:`fit_configs`."""
    F, C, T = X.shape
    unknown = [n for n in configs if n not in _CONFIGS]
    if unknown:
        raise ValueError(f"unknown config(s): {unknown}. Available: {CONFIG_NAMES}")
    if C < 2 and any(_CONFIGS[n][3] is not None for n in configs):
        # The reconstruction from zero other channels is all NaN, which would silently disable the search.
        raise ValueError(f"common-mode (+CMR) configurations need at least 2 channels, got {C}")

    spectral = dict(
        representation=representation, fmin=fmin, fmax=fmax, n_frequencies=n_frequencies, n_averages=n_averages
    )
    S = recon_spec = freqs = None
    if any(_CONFIGS[n][0] == "spec" for n in configs):
        freqs, S = log_spectra(X, fs, parallel=parallel, **spectral)  # (F, C, P) dB
        recon_spec = np.stack([np.median(np.delete(S, c, axis=1), axis=1) for c in range(C)], axis=1)

    fits = {}
    for name in configs:
        kind, feat_fn, baseline_fn, cmr = _CONFIGS[name]
        ref_all = X if kind == "amp" else S
        thresholds = np.empty(C)
        scores = np.empty((F, C))
        baselines = None if kind == "amp" else np.empty((C, S.shape[2]))
        for c in range(C):
            if cmr == "wave":
                aug = _cmr_wave(X, c)
            elif cmr == "spec":
                aug = recon_spec[:, c, :]
            else:
                aug = None
            ch = fit_channel(
                ref_all[:, c, :], feat_fn, baseline_fn=baseline_fn, aug_ref=aug,
                n_folds=n_folds, n_cand=n_cand, seed=seed,
            )
            thresholds[c] = ch.threshold
            scores[:, c] = ch.scores
            if baselines is not None:
                baselines[c] = ch.baseline
        fits[name] = DetectorFit(
            config=name,
            kind=kind,
            fs=float(fs),
            n_samples=int(T),
            thresholds=thresholds,
            scores=scores,
            baselines=baselines,
            freqs=None if kind == "amp" else freqs,
            spectral=None if kind == "amp" else dict(spectral),
        )
    return fits


def fit_configs(
    X: np.ndarray,
    fs: float,
    *,
    configs=CONFIG_NAMES,
    representation="welch",
    n_folds=N_FOLDS,
    n_cand=N_CAND,
    seed=0,
    fmin=FMIN,
    fmax=FMAX,
    n_frequencies=N_FREQ_DES,
    n_averages=N_AVG_DES,
    parallel=True,
) -> dict[str, DetectorFit]:
    """Fit the requested configurations on one recording's pool.

    Spectra and their common-mode reconstructions are computed once and shared by every spectral
    configuration. Each configuration runs its own per-channel cutoff search (:func:`fit_channel`).

    Args:
        X: ``(F, C, T)`` fragments in volts, all finite.
        fs: Sampling rate in Hz.
        configs: Names from ``CONFIG_NAMES``.
        representation: Spectral estimator; only ``"welch"``.
        n_folds: Number of random 80/20 splits.
        n_cand: Number of candidate cutoffs.
        seed: Seed of the splits.
        fmin: Lowest frequency kept, Hz.
        fmax: Highest frequency kept, Hz.
        n_frequencies: Approximate number of in-band bins.
        n_averages: Approximate number of segments averaged.
        parallel: Fan spectral work out over dask when available.

    Returns:
        ``{config: DetectorFit}``.

    Raises:
        ValueError: On an unknown configuration, a common-mode configuration with fewer than 2 channels, or
            input that is not 3-D or holds NaN or inf.
    """
    return _fit_configs(
        _validated(X),
        fs,
        list(configs),
        representation=representation,
        n_folds=n_folds,
        n_cand=n_cand,
        seed=seed,
        fmin=fmin,
        fmax=fmax,
        n_frequencies=n_frequencies,
        n_averages=n_averages,
        parallel=parallel,
    )


def compute_masks(
    X: np.ndarray,
    fs: float,
    *,
    configs=CONFIG_NAMES,
    representation="welch",
    n_folds=N_FOLDS,
    n_cand=N_CAND,
    seed=0,
    fmin=FMIN,
    fmax=FMAX,
    n_frequencies=N_FREQ_DES,
    n_averages=N_AVG_DES,
    parallel=True,
) -> dict[str, np.ndarray]:
    """Flag the pool fragments under the requested configurations.

    Args:
        X: See :func:`fit_configs`.
        fs: See :func:`fit_configs`.
        configs: See :func:`fit_configs`.
        representation: See :func:`fit_configs`.
        n_folds: See :func:`fit_configs`.
        n_cand: See :func:`fit_configs`.
        seed: See :func:`fit_configs`.
        fmin: See :func:`fit_configs`.
        fmax: See :func:`fit_configs`.
        n_frequencies: See :func:`fit_configs`.
        n_averages: See :func:`fit_configs`.
        parallel: See :func:`fit_configs`.

    Returns:
        ``{config: (F, C) bool}``, True meaning reject.
    """
    fits = fit_configs(
        X,
        fs,
        configs=configs,
        representation=representation,
        n_folds=n_folds,
        n_cand=n_cand,
        seed=seed,
        fmin=fmin,
        fmax=fmax,
        n_frequencies=n_frequencies,
        n_averages=n_averages,
        parallel=parallel,
    )
    return {name: fit.mask for name, fit in fits.items()}


def score_fragments(X: np.ndarray, fit: DetectorFit, *, parallel=True) -> np.ndarray:
    """Score fragments with a fitted configuration's stored parameters.

    Peak-to-peak needs only the samples; the spectral arms recompute each fragment's dB spectrum with the
    fit's settings and measure it against the stored baselines. On the fit's own pool this reproduces
    ``fit.scores`` exactly.

    Args:
        X: ``(F, C, T)`` fragments in volts, all finite, with the fit's channels, sampling rate and length.
        fit: A :class:`DetectorFit`.
        parallel: Fan spectral work out over dask when available.

    Returns:
        ``(F, C)`` scores.

    Raises:
        ValueError: If ``X`` holds NaN or inf, or its channels or length differ from the fit's.
    """
    X = _validated(X)
    F, C, T = X.shape
    if C != fit.thresholds.size or T != fit.n_samples:
        raise ValueError(
            f"fragments are (channels={C}, samples={T}) but {fit.config} was fit on "
            f"(channels={fit.thresholds.size}, samples={fit.n_samples})"
        )
    if fit.kind == "amp":
        return X.max(axis=2) - X.min(axis=2)
    freqs, S = log_spectra(X, fit.fs, parallel=parallel, **fit.spectral)
    if not np.array_equal(freqs, fit.freqs):
        raise ValueError(f"spectral bins differ from those {fit.config} was fit on")
    scores = np.empty((F, C))
    for c in range(C):
        scores[:, c] = spec_feat(S[:, c, :], fit.baselines[c])
    return scores


def apply_fit(X: np.ndarray, fit: DetectorFit, *, parallel=True) -> np.ndarray:
    """Flag fragments with a fitted configuration's stored cutoffs.

    Args:
        X: See :func:`score_fragments`.
        fit: A :class:`DetectorFit`.
        parallel: See :func:`score_fragments`.

    Returns:
        ``(F, C)`` bool, True meaning reject.
    """
    return score_fragments(X, fit, parallel=parallel) > fit.thresholds


# The full grid

# The peak-to-peak arms involve no spectrum, so they run once. The spectral arms run for every
# (representation, n_frequencies, n_averages); baseline (v2 median, v3 mean) and common-mode on or off are
# the 4 _SPEC_CONFIGS. Extending a tuple below adds configurations to grid_config_names, fit_grid and
# compute_masks_grid, and select_detectors picks up the new mask columns.
GRID_REPRESENTATIONS = ("welch",)
GRID_NFREQ = (20, 40, 80)
GRID_NAVG = (50, 100, 200)
_AMP_CONFIGS = ["autoreject/self", "autoreject/self+CMR"]
_SPEC_CONFIGS = ["v2/self", "v2/self+CMR", "v3/self", "v3/self+CMR"]


def grid_config_names(*, representations=GRID_REPRESENTATIONS, n_freqs=GRID_NFREQ, n_avgs=GRID_NAVG):
    """List the full grid's configuration names in the order :func:`fit_grid` emits them.

    Peak-to-peak configurations come first, then ``{representation}/{config}@nf{n}_na{m}``. Computes nothing.

    Returns:
        A list of names (38 with the default grid).
    """
    names = list(_AMP_CONFIGS)
    for rep in representations:
        for nf in n_freqs:
            for na in n_avgs:
                names += [f"{rep}/{base}@nf{nf}_na{na}" for base in _SPEC_CONFIGS]
    return names


def fit_grid(
    X: np.ndarray,
    fs: float,
    *,
    representations=GRID_REPRESENTATIONS,
    n_freqs=GRID_NFREQ,
    n_avgs=GRID_NAVG,
    n_folds=N_FOLDS,
    n_cand=N_CAND,
    seed=0,
    parallel=True,
) -> dict[str, DetectorFit]:
    """Fit every configuration of the grid on one recording's pool.

    One spectral transform runs per (representation, n_frequencies, n_averages) and is shared by its 4
    spectral configurations.

    Args:
        X: ``(F, C, T)`` fragments in volts, all finite.
        fs: Sampling rate in Hz.
        representations: Spectral estimators in the grid.
        n_freqs: Grid values of ``n_frequencies``.
        n_avgs: Grid values of ``n_averages``.
        n_folds: Number of random 80/20 splits.
        n_cand: Number of candidate cutoffs.
        seed: Seed of the splits.
        parallel: Fan spectral work out over dask when available.

    Returns:
        ``{name: DetectorFit}`` keyed as :func:`grid_config_names`.
    """
    X = _validated(X)
    common = dict(n_folds=n_folds, n_cand=n_cand, seed=seed, fmin=FMIN, fmax=FMAX, parallel=parallel)
    fits = _fit_configs(
        X, fs, _AMP_CONFIGS, representation="welch", n_frequencies=N_FREQ_DES, n_averages=N_AVG_DES, **common
    )
    for rep in representations:
        for nf in n_freqs:
            for na in n_avgs:
                spec = _fit_configs(X, fs, _SPEC_CONFIGS, representation=rep, n_frequencies=nf, n_averages=na, **common)
                for base, fit in spec.items():
                    name = f"{rep}/{base}@nf{nf}_na{na}"
                    fits[name] = dataclasses.replace(fit, config=name)
    return fits


def compute_masks_grid(
    X: np.ndarray,
    fs: float,
    *,
    representations=GRID_REPRESENTATIONS,
    n_freqs=GRID_NFREQ,
    n_avgs=GRID_NAVG,
    n_folds=N_FOLDS,
    n_cand=N_CAND,
    seed=0,
    parallel=True,
) -> dict[str, np.ndarray]:
    """Flag the pool fragments under every configuration of the grid.

    Args:
        X: See :func:`fit_grid`.
        fs: See :func:`fit_grid`.
        representations: See :func:`fit_grid`.
        n_freqs: See :func:`fit_grid`.
        n_avgs: See :func:`fit_grid`.
        n_folds: See :func:`fit_grid`.
        n_cand: See :func:`fit_grid`.
        seed: See :func:`fit_grid`.
        parallel: See :func:`fit_grid`.

    Returns:
        ``{name: (F, C) bool}`` keyed as :func:`grid_config_names`, True meaning reject.
    """
    fits = fit_grid(
        X,
        fs,
        representations=representations,
        n_freqs=n_freqs,
        n_avgs=n_avgs,
        n_folds=n_folds,
        n_cand=n_cand,
        seed=seed,
        parallel=parallel,
    )
    return {name: fit.mask for name, fit in fits.items()}
