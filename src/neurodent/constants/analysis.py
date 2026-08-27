from enum import Enum
from typing import Literal, get_args


class FeatureType(Enum):
    """Enumeration of feature data shapes.

    Each feature in the system has exactly one :class:`FeatureType` that describes
    how its data is stored and how common operations (averaging, filtering,
    channel remapping, plotting) should handle it.

    Using :func:`classify_feature` with this enum eliminates the need for
    scattered ``match`` / ``if`` blocks that hard-code feature name strings.

    Canonical extracted shapes (after stacking W windows via ``extract_*``):

    .. code-block:: text

        Type              Extracted shape       Axis semantics
        ──────────────    ────────────────      ──────────────────────────
        LINEAR            (W, C)                windows, channels
        LINEAR_2D         (W, C, K)             windows, channels, components
        BAND              (W, C, B)             windows, channels, bands
        SIMPLE_MATRIX     (W, C, C)             windows, ch_row, ch_col
        BANDED_MATRIX     (W, C, C, B)          windows, ch_row, ch_col, bands
        HIST              (W, C, F)             windows, channels, freq_bins

    **Convention**: ``W`` is always axis 0.  Channel axes (``C``) always come
    next (axis 1, and axis 2 for matrix types).  Feature-specific semantic
    dimensions (bands ``B``, components ``K``, frequency bins ``F``) are last.

    This means:

    - **Channel collapse** is always ``axis=1`` (or ``axis=(1, 2)`` for matrices).
    - ``vals[:, ch_idx]`` gives one channel's data for every feature type.
    - Matches per-cell storage order: LINEAR stores ``(C,)``, matrices store
      ``(C, C)``, so stacking windows on axis 0 produces the canonical shape.

    Example::

        from neurodent.constants import FeatureType, classify_feature

        ftype = classify_feature("rms")
        assert ftype is FeatureType.LINEAR
        assert ftype.extracted_shape == "W, C"
        assert ftype.channel_axes == (1,)
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

    # -- shape metadata properties -----------------------------------------------

    @property
    def extracted_shape(self) -> str:
        """Symbolic shape string after extraction (e.g. ``'W, C, B'``)."""
        return FEATURE_SHAPES[self]["extracted_shape"]

    @property
    def channel_axes(self) -> tuple[int, ...]:
        """Axis indices corresponding to channel dimensions."""
        return FEATURE_SHAPES[self]["channel_axes"]

    @property
    def semantic_axes(self) -> dict[str, int]:
        """Mapping of semantic dimension name to axis index."""
        return FEATURE_SHAPES[self]["semantic_axes"]

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


# ---------------------------------------------------------------------------
# Canonical shape registry  (defined after FeatureType so members are available)
# ---------------------------------------------------------------------------

FEATURE_SHAPES: dict[FeatureType, dict] = {
    FeatureType.LINEAR: {
        "extracted_shape": "W, C",
        "cell_shape": "C",
        "channel_axes": (1,),
        "semantic_axes": {},
        "description": "Scalar per channel (e.g. rms, ampvar)",
    },
    FeatureType.LINEAR_2D: {
        "extracted_shape": "W, C, K",
        "cell_shape": "C, K",
        "channel_axes": (1,),
        "semantic_axes": {"components": 2},
        "description": "K components per channel (e.g. psdslope)",
    },
    FeatureType.BAND: {
        "extracted_shape": "W, C, B",
        "cell_shape": "{band: (C,)}",
        "channel_axes": (1,),
        "semantic_axes": {"bands": 2},
        "description": "One value per band per channel (e.g. psdband, logpsdband)",
    },
    FeatureType.SIMPLE_MATRIX: {
        "extracted_shape": "W, C, C",
        "cell_shape": "C, C",
        "channel_axes": (1, 2),
        "semantic_axes": {},
        "description": "Channel x channel matrix (e.g. pcorr, zpcorr)",
    },
    FeatureType.BANDED_MATRIX: {
        "extracted_shape": "W, C, C, B",
        "cell_shape": "{band: (C, C)}",
        "channel_axes": (1, 2),
        "semantic_axes": {"bands": 3},
        "description": "Channel x channel matrix per band (e.g. cohere, zcohere)",
    },
    FeatureType.HIST: {
        "extracted_shape": "W, C, F",
        "cell_shape": "(F,), (C, F)",
        "channel_axes": (1,),
        "semantic_axes": {"freq_bins": 2},
        "description": "Spectral histogram per channel (e.g. psd)",
    },
}
"""Canonical shape metadata for each :class:`FeatureType`.

Keys per entry:

- ``extracted_shape``: Symbolic shape after stacking W windows.
- ``cell_shape``: Shape of a single cell in the WAR DataFrame.
- ``channel_axes``: Tuple of axis indices that are channel dimensions.
- ``semantic_axes``: Dict mapping semantic dimension names to axis indices.
- ``description``: Human-readable description.

**Shape convention**: ``(W, C, [C₂], ...semantic)``.
Windows (W) is always axis 0.  Channel axes (C) always follow immediately.
Semantic dimensions (bands, components, freq bins) are trailing.
"""


LinearFeatureName = Literal[
    "rms",
    "ampvar",
    "psdtotal",
    "nspike",
    "logrms",
    "logampvar",
    "logpsdtotal",
    "lognspike",
]
"""Valid names for linear (scalar) features."""

LINEAR_FEATURES = list(get_args(LinearFeatureName))
"""List of linear (scalar) feature names computed per channel."""

Linear2DFeatureName = Literal["psdslope"]
"""Valid names for multi-component linear features."""

LINEAR_2D_FEATURES = list(get_args(Linear2DFeatureName))
"""List of multi-component linear features (stored as 2-D arrays per channel)."""

BandFeatureName = Literal["psdband", "psdfrac", "logpsdband", "logpsdfrac"]
"""Valid names for frequency-band features."""

BAND_FEATURES = list(get_args(BandFeatureName))
"""List of frequency-band feature names (one value per band)."""

BandedMatrixFeatureName = Literal["cohere", "zcohere", "imcoh", "zimcoh"]
"""Valid names for matrix features with a frequency band dimension."""

BANDED_MATRIX_FEATURES = list(get_args(BandedMatrixFeatureName))
"""Matrix features with frequency band dimension (stored as dict with band keys)."""

SimpleMatrixFeatureName = Literal["pcorr", "zpcorr"]
"""Valid names for matrix features without a frequency band dimension."""

SIMPLE_MATRIX_FEATURES = list(get_args(SimpleMatrixFeatureName))
"""Matrix features without frequency bands (single 2D correlation matrix)."""

MatrixFeatureName = Literal[BandedMatrixFeatureName, SimpleMatrixFeatureName]
"""Valid names for connectivity/matrix features."""

MATRIX_FEATURES = list(get_args(MatrixFeatureName))
"""List of connectivity/matrix feature names (channel pairs)."""

HistFeatureName = Literal["psd"]
"""Valid names for histogram/spectral features."""

HIST_FEATURES = list(get_args(HistFeatureName))
"""List of histogram/spectral feature names."""

PlottableFeatureName = Literal[
    LinearFeatureName,
    Linear2DFeatureName,
    BandFeatureName,
    MatrixFeatureName,
]
"""Feature names the general-purpose plot functions accept.

Excludes the histogram features, whose ``freq_bins`` axis holds around a hundred values
where every other semantic axis holds two to five. They are plotted by the dedicated
:meth:`~neurodent.plotting.AnimalPlotter.plot_psd_histogram` and
:meth:`~neurodent.plotting.AnimalPlotter.plot_psd_spectrogram` instead.
"""

FeatureName = Literal[
    PlottableFeatureName,
    HistFeatureName,
]
"""Every valid feature name. Annotate feature parameters with this."""

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

# ---------------------------------------------------------------------------
# Semantic dimension labels
# ---------------------------------------------------------------------------

COMPONENT_LABELS = {
    "psdslope": ["slope", "intercept"],
}
"""Mapping of feature names to their component dimension labels.

For LINEAR_2D features that have multiple components (e.g., psdslope has slope
and intercept), this provides human-readable labels for each component.
"""

# ---------------------------------------------------------------------------
# Frequency band parameters
# ---------------------------------------------------------------------------

FREQ_BAND_TOTAL = (1, 40)
"""Total frequency range covered by all bands (min, max) in Hz."""

FREQ_MINS = [v[0] for _, v in FREQ_BANDS.items()]
"""List of minimum frequencies for each band."""

FREQ_MAXS = [v[1] for _, v in FREQ_BANDS.items()]
"""List of maximum frequencies for each band."""
