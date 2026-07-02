"""**Core module** for loading, organizing, and analyzing EEG data.

This module provides the fundamental building blocks for the NeuRodent analysis pipeline,
handling everything from raw data loading to feature extraction.

**Typical Workflow:**

.. code-block:: python

    from neurodent import visualization

    # 1. Organize recordings for an animal
    ao = visualization.AnimalOrganizer(
        data_path, animal_id, mode="nest",
    )

    # 2. Run windowed feature analysis
    war = ao.compute_windowed_analysis(
        features=['psdband', 'cohere'],
        window_s=5,
        multiprocess_mode="dask",
    )

    # 3. Access results (WindowAnalysisResult)
    df = war.result  # pandas DataFrame with all features
    war.save_parquet_and_json('output/')

The lower-level classes can also be used directly:

.. code-block:: python

    from neurodent import core

    # Load a single recording
    lro = core.LongRecordingOrganizer(day_folder_path)

    # Analyze in fragments
    analyzer = core.LongRecordingAnalyzer(lro, fragment_len_s=5)
    rms = analyzer.compute_rms(fragment_index=0)
    psd = analyzer.compute_psdband(fragment_index=0)

**What Gets Computed:**

- **Power features**: RMS, amplitude variance, band power (delta, theta, alpha, beta, gamma)
- **Connectivity**: Coherence, imaginary coherence, Pearson correlation between channels
- **Spikes**: Spike counts via frequency-domain detection

**See Also:**

- :doc:`/quickstart/basic_usage` - Getting started guide
- :doc:`/tutorials/windowed_analysis` - Detailed analysis examples
- :mod:`neurodent.constants` - Configure frequency bands and parameters
"""

import os
import tempfile

# Ensure a usable temporary directory is available for downstream modules
if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = tempfile.gettempdir()

# Core classes
from .core import (
    LongRecordingOrganizer,
    RecordingMetadata,
    DDFBinaryMetadata,  # Deprecated, kept for backward compatibility
    convert_ddfcolbin_to_ddfrowbin,
    convert_ddfrowbin_to_si,
    split_recording,
)
from .analysis import LongRecordingAnalyzer
from .analyze_frag import FragmentAnalyzer
from .frequency_domain_spike_detection import FrequencyDomainSpikeDetector
from .utils import (
    get_temp_directory,
    set_temp_directory,
    validate_timestamps,
    nanaverage,
    resolve_channel,
    resolve_channels,
    parse_str_to_day,
    log_transform,
    should_use_cache_unified,
    get_feature_label,
    get_cache_status_message,
    slugify,
)

from .zeitgeber import (
    ZeitgeberAnalysisResult,
    run_zeitgeber_pipeline,
    get_expanded_feature_names,
    transform_time_axis,
    expand_zt_axis,
)
from . import utils
from . import discovery

# Loading orchestration (imported after backend + results so their names are bound)
from .loading import AnimalOrganizer

__all__ = [
    # Data loading
    "AnimalOrganizer",
    "RecordingMetadata",
    "DDFBinaryMetadata",  # Deprecated, kept for backward compatibility
    "convert_ddfcolbin_to_ddfrowbin",
    "convert_ddfrowbin_to_si",
    "split_recording",
    "LongRecordingOrganizer",
    # Analysis
    "LongRecordingAnalyzer",
    "FragmentAnalyzer",
    "FrequencyDomainSpikeDetector",
    "ZeitgeberAnalysisResult",
    "run_zeitgeber_pipeline",
    "get_expanded_feature_names",
    "transform_time_axis",
    "expand_zt_axis",
    # Utilities
    "convert_ddfcolbin_to_ddfrowbin",
    "convert_ddfrowbin_to_si",
    "get_temp_directory",
    "set_temp_directory",
    "validate_timestamps",
    "nanaverage",
    "resolve_channel",
    "resolve_channels",
    "parse_str_to_day",
    "log_transform",
    "get_feature_label",
    "get_cache_status_message",
    "should_use_cache_unified",
    "slugify",
    "utils",
]
