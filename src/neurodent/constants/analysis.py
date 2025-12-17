LINEAR_FEATURES = [
    "rms",
    "ampvar",
    "psdtotal",
    "psdslope",
    "nspike",
    "logrms",
    "logampvar",
    "logpsdtotal",
    "lognspike",
]
"""List of linear (scalar) feature names computed per channel."""

BAND_FEATURES = ["psdband", "psdfrac"] + ["logpsdband", "logpsdfrac"]
"""List of frequency-band feature names (one value per band)."""

MATRIX_FEATURES = ["cohere", "zcohere", "imcoh", "zimcoh", "pcorr", "zpcorr"]
"""List of connectivity/matrix feature names (channel pairs)."""

HIST_FEATURES = ["psd"]
"""List of histogram/spectral feature names."""

FEATURES = LINEAR_FEATURES + BAND_FEATURES + MATRIX_FEATURES + HIST_FEATURES
"""Complete list of all available features."""

WAR_FEATURES = [f for f in FEATURES if "nspike" not in f]
"""Features available in WindowAnalysisResult (excludes spike-related)."""

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 25),
    "gamma": (25, 40),
}
"""Dictionary mapping frequency band names to (min_hz, max_hz) tuples.

Delta band adjusted to 1-4 Hz (changed from 0.1-4 Hz) for reliable coherence
estimation with short epochs and to avoid insufficient cycles warnings.
"""

BAND_NAMES = [k for k, _ in FREQ_BANDS.items()]
"""Ordered list of frequency band names: ['delta', 'theta', 'alpha', 'beta', 'gamma']."""

FREQ_BAND_TOTAL = (1, 40)
"""Total frequency range covered by all bands (min, max) in Hz."""

FREQ_MINS = [v[0] for _, v in FREQ_BANDS.items()]
"""List of minimum frequencies for each band."""

FREQ_MAXS = [v[1] for _, v in FREQ_BANDS.items()]
"""List of maximum frequencies for each band."""
