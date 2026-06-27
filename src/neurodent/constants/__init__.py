"""**Constants** used throughout NeuRodent.

This module centralizes all configuration values, feature definitions, and lookup tables
used by the :mod:`neurodent.core` and :mod:`neurodent.visualization` modules.

**Quick Reference:**

.. code-block:: python

    from neurodent import constants

    # Frequency bands (Hz ranges)
    constants.FREQ_BANDS
    # {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 25), 'gamma': (25, 40)}

    # Available features
    constants.LINEAR_FEATURES   # ['rms', 'ampvar', 'psdtotal', 'nspike', ...]
    constants.LINEAR_2D_FEATURES  # ['psdslope']
    constants.BAND_FEATURES     # ['psdband', 'psdfrac', 'logpsdband', 'logpsdfrac']
    constants.MATRIX_FEATURES   # ['cohere', 'zcohere', 'imcoh', 'zimcoh', 'pcorr', 'zpcorr']

    # Global settings
    constants.GLOBAL_SAMPLING_RATE  # 1000 Hz
    constants.LINE_FREQ             # 60 Hz (for notch filter)

    # Colorblind-friendly colors
    constants.OKABE_ITO_COLORS["blue"]   # '#0072B2'
    constants.OKABE_ITO_COLORS["orange"] # '#E69F00'

**Customization:**

To override defaults, import and modify before running analysis:

.. code-block:: python

    from neurodent.constants import config
    config.GLOBAL_SAMPLING_RATE = 500  # If your data is 500 Hz

**See Also:**

- :doc:`/quickstart/configuration` - Full customization guide
- :mod:`neurodent.constants.analysis` - Feature and frequency definitions
- :mod:`neurodent.constants.config` - Sampling parameters
"""

# Re-export everything for backward compatibility
from .mappings import (
    GENOTYPE_ALIASES,
    GENE_ALIASES,
    SEX_ALIASES,
    CHANNEL_MAP,
    CHANNEL_ABBREVS,
    CHANNEL_ABBREV_BY_RAW,
    DF_SORT_ORDER,
    DATEPARSER_PATTERNS_TO_REMOVE,
    DEFAULT_DAY,
    FEATURE_LABELS,
)

# ANIMAL_METADATA is injected at runtime by apply_samples_config()
# Default is empty dict until populated
ANIMAL_METADATA: dict = {}


def _recompute_channel_map_derived() -> None:
    """Rebuild channel-derived constants from :data:`CHANNEL_MAP`.

    Call after assigning a new ``CHANNEL_MAP`` (e.g. via :func:`set_channel_map`
    or ``apply_samples_config``). Updates :data:`CHANNEL_ABBREVS`,
    :data:`CHANNEL_ABBREV_BY_RAW`, and the ``channel`` entry of :data:`DF_SORT_ORDER` so
    every reader sees the new channel map.

    Raises:
        ValueError: if a raw channel name is mapped to more than one abbreviation (a
            config error — surfaced loudly rather than silently resolving to one).
    """
    global CHANNEL_ABBREVS, CHANNEL_ABBREV_BY_RAW
    CHANNEL_ABBREVS = list(CHANNEL_MAP)
    DF_SORT_ORDER["channel"] = ["average", "all", *CHANNEL_ABBREVS]
    reverse: dict = {}
    for abbrev, raws in CHANNEL_MAP.items():
        for raw in raws:
            if raw in reverse and reverse[raw] != abbrev:
                raise ValueError(
                    f"Raw channel name {raw!r} is mapped to both {reverse[raw]!r} and "
                    f"{abbrev!r} in CHANNEL_MAP; raw names must map to exactly one channel."
                )
            reverse[raw] = abbrev
    CHANNEL_ABBREV_BY_RAW = reverse


def set_channel_map(channels: dict) -> None:
    """Set the canonical channel map (the single source of truth) and derive the rest.

    This is the package-level front door for declaring custom channels; the Snakemake
    pipeline reaches the same state through ``apply_samples_config``. Both update
    :data:`CHANNEL_MAP` and then call :func:`_recompute_channel_map_derived`, so
    :data:`CHANNEL_ABBREVS`, :data:`CHANNEL_ABBREV_BY_RAW`, the standardization target, the
    ``channel`` sort order, and the LOF evaluation set all follow from one place.

    Channel resolution is **exact** (``resolve_channel`` looks a raw name up in
    :data:`CHANNEL_ABBREV_BY_RAW`); a label is never inferred from a name. List every raw
    spelling a dataset's data presents under its abbreviation.

    Args:
        channels: Ordered ``{abbrev: [raw names]}`` mapping. Order defines the
            canonical channel order. Each abbreviation (e.g. ``"LMot"``) is the atomic
            channel identity; left/right is part of the name, not a separate axis.

    Example:
        >>> from neurodent import set_channel_map
        >>> set_channel_map({"LMot": ["LMot", "L Motor Ctx"], "RMot": ["RMot", "R Motor Ctx"]})
    """
    global CHANNEL_MAP
    CHANNEL_MAP = dict(channels)
    _recompute_channel_map_derived()

from .analysis import (
    FeatureType,
    FEATURE_SHAPES,
    LINEAR_FEATURES,
    LINEAR_2D_FEATURES,
    BAND_FEATURES,
    MATRIX_FEATURES,
    BANDED_MATRIX_FEATURES,
    SIMPLE_MATRIX_FEATURES,
    HIST_FEATURES,
    FEATURES,
    WAR_FEATURES,
    FEATURE_TYPES,
    classify_feature,

    FREQ_BANDS,
    BAND_NAMES,
    COMPONENT_LABELS,
    FREQ_BAND_TOTAL,
    FREQ_MINS,
    FREQ_MAXS,
)
from .plotting import (
    FEATURE_PLOT_HEIGHT_RATIOS,
    OKABE_ITO_COLORS,
)

from .config import (
    GLOBAL_SAMPLING_RATE,
    GLOBAL_DTYPE,
    LINE_FREQ,
    NEURODENT_SIDECAR_NAME,
)

__all__ = [
    # Mappings
    "GENOTYPE_ALIASES",
    "GENE_ALIASES",
    "SEX_ALIASES",
    "ANIMAL_METADATA",
    "CHANNEL_MAP",
    "CHANNEL_ABBREVS",
    "CHANNEL_ABBREV_BY_RAW",
    "set_channel_map",
    "DF_SORT_ORDER",
    "DATEPARSER_PATTERNS_TO_REMOVE",
    "DEFAULT_DAY",
    "FEATURE_LABELS",
    # Feature Typing
    "FeatureType",
    "FEATURE_SHAPES",
    "FEATURE_TYPES",
    "classify_feature",
    # Feature & Frequency Definitions
    "LINEAR_FEATURES",
    "LINEAR_2D_FEATURES",
    "BAND_FEATURES",
    "MATRIX_FEATURES",
    "BANDED_MATRIX_FEATURES",
    "SIMPLE_MATRIX_FEATURES",
    "HIST_FEATURES",
    "FEATURES",
    "WAR_FEATURES",
    "FREQ_BANDS",
    "BAND_NAMES",
    "COMPONENT_LABELS",
    "FREQ_BAND_TOTAL",
    "FREQ_MINS",
    "FREQ_MAXS",
    # Plotting
    "FEATURE_PLOT_HEIGHT_RATIOS",
    "OKABE_ITO_COLORS",
    # Global Config
    "GLOBAL_SAMPLING_RATE",
    "GLOBAL_DTYPE",
    "LINE_FREQ",
    "NEURODENT_SIDECAR_NAME",
]
