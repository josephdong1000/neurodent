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
BAND_FEATURES = ["psdband", "psdfrac"] + ["logpsdband", "logpsdfrac"]
MATRIX_FEATURES = ["cohere", "zcohere", "imcoh", "zimcoh", "pcorr", "zpcorr"]
HIST_FEATURES = ["psd"]
FEATURES = LINEAR_FEATURES + BAND_FEATURES + MATRIX_FEATURES + HIST_FEATURES
WAR_FEATURES = [f for f in FEATURES if "nspike" not in f]

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 25),
    "gamma": (25, 40),
}
"""Dictionary of frequency band ranges in Hz.

Delta band adjusted to 1-4 Hz (changed from 0.1-4 Hz) for reliable coherence
estimation with short epochs and to avoid insufficient cycles warnings.
"""

BAND_NAMES = [k for k, _ in FREQ_BANDS.items()]

FREQ_BAND_TOTAL = (1, 40)  # Updated to match new delta band minimum
FREQ_MINS = [v[0] for _, v in FREQ_BANDS.items()]
FREQ_MAXS = [v[1] for _, v in FREQ_BANDS.items()]
