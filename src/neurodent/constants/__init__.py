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
    DEFAULT_ID_TO_LR,
    GENOTYPE_ALIASES,
    GENE_ALIASES,
    SEX_ALIASES,
    CHNAME_ALIASES,
    LR_ALIASES,
    DEFAULT_ID_TO_NAME,
    DF_SORT_ORDER,
    DATEPARSER_PATTERNS_TO_REMOVE,
    DEFAULT_DAY,
    FEATURE_LABELS,
)

# ANIMAL_METADATA is injected at runtime by inject_config_aliases()
# Default is empty dict until populated
ANIMAL_METADATA: dict = {}

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
    "DEFAULT_ID_TO_LR",
    "GENOTYPE_ALIASES",
    "GENE_ALIASES",
    "SEX_ALIASES",
    "ANIMAL_METADATA",
    "CHNAME_ALIASES",
    "LR_ALIASES",
    "DEFAULT_ID_TO_NAME",
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
