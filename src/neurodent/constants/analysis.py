from enum import Enum


class FeatureType(Enum):
    """Enumeration of feature data shapes.

    Each feature in the system has exactly one :class:`FeatureType` that describes
    how its data is stored and how common operations (averaging, filtering,
    channel remapping, plotting) should handle it.

    Using :func:`classify_feature` with this enum eliminates the need for
    scattered ``match`` / ``if`` blocks that hard-code feature name strings.

    Example::

        from neurodent.constants import FeatureType, classify_feature

        ftype = classify_feature("rms")
        assert ftype is FeatureType.LINEAR
        assert not ftype.is_dict_stored
    """

    LINEAR = "linear"
    """Scalar per channel.  Stored as a list of floats (one per channel)."""

    LINEAR_2D = "linear_2d"
    """Multi-component per channel.  Stored as a 2-D array ``(n_channels, n_components)``."""

    BAND = "band"
    """Dict keyed by frequency-band name, values are per-channel arrays."""

    BANDED_MATRIX = "banded_matrix"
    """Dict keyed by frequency-band name, values are 2-D (channel × channel) matrices."""

    SIMPLE_MATRIX = "simple_matrix"
    """2-D (channel × channel) matrix without a frequency-band dimension."""

    HIST = "hist"
    """Histogram / spectral data stored as a ``(frequencies, values)`` tuple."""

    # -- convenience predicates --------------------------------------------------

    @property
    def is_linear(self) -> bool:
        """``True`` for per-channel features (scalar or multi-component)."""
        return self in (FeatureType.LINEAR, FeatureType.LINEAR_2D)

    @property
    def is_matrix(self) -> bool:
        """``True`` for features stored as channel × channel matrices."""
        return self in (FeatureType.BANDED_MATRIX, FeatureType.SIMPLE_MATRIX)

    @property
    def is_dict_stored(self) -> bool:
        """``True`` for features stored as dicts keyed by frequency band."""
        return self in (FeatureType.BAND, FeatureType.BANDED_MATRIX)


LINEAR_FEATURES = [
    "rms",
    "ampvar",
    "psdtotal",
    "nspike",
    "logrms",
    "logampvar",
    "logpsdtotal",
    "lognspike",
]
"""List of linear (scalar) feature names computed per channel."""

LINEAR_2D_FEATURES = ["psdslope"]
"""List of multi-component linear features (stored as 2-D arrays per channel)."""

BAND_FEATURES = ["psdband", "psdfrac"] + ["logpsdband", "logpsdfrac"]
"""List of frequency-band feature names (one value per band)."""

MATRIX_FEATURES = ["cohere", "zcohere", "imcoh", "zimcoh", "pcorr", "zpcorr"]
"""List of connectivity/matrix feature names (channel pairs)."""

BANDED_MATRIX_FEATURES = ["cohere", "zcohere", "imcoh", "zimcoh"]
"""Matrix features with frequency band dimension (stored as dict with band keys)."""

SIMPLE_MATRIX_FEATURES = ["pcorr", "zpcorr"]
"""Matrix features without frequency bands (single 2D correlation matrix)."""


HIST_FEATURES = ["psd"]
"""List of histogram/spectral feature names."""

FEATURES = LINEAR_FEATURES + LINEAR_2D_FEATURES + BAND_FEATURES + MATRIX_FEATURES + HIST_FEATURES
"""Complete list of all available features."""

WAR_FEATURES = [f for f in FEATURES if "nspike" not in f]
"""Features available in WindowAnalysisResult (excludes spike-related)."""

# ---------------------------------------------------------------------------
# Centralised feature-name → FeatureType mapping
# ---------------------------------------------------------------------------

FEATURE_TYPES: dict[str, FeatureType] = {
    **{f: FeatureType.LINEAR for f in LINEAR_FEATURES},
    **{f: FeatureType.LINEAR_2D for f in LINEAR_2D_FEATURES},
    **{f: FeatureType.BAND for f in BAND_FEATURES},
    **{f: FeatureType.BANDED_MATRIX for f in BANDED_MATRIX_FEATURES},
    **{f: FeatureType.SIMPLE_MATRIX for f in SIMPLE_MATRIX_FEATURES},
    **{f: FeatureType.HIST for f in HIST_FEATURES},
}
"""Mapping from every known feature name to its :class:`FeatureType`."""


def classify_feature(feature_name: str) -> FeatureType:
    """Return the :class:`FeatureType` for a given feature name.

    Args:
        feature_name: Name of the feature (e.g. ``'rms'``, ``'psdband'``, ``'cohere'``).

    Returns:
        The corresponding :class:`FeatureType` enum member.

    Raises:
        ValueError: If *feature_name* is not a recognised feature.
    """
    try:
        return FEATURE_TYPES[feature_name]
    except KeyError:
        raise ValueError(
            f"Unknown feature '{feature_name}'. "
            f"Known features: {sorted(FEATURE_TYPES.keys())}"
        ) from None

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
