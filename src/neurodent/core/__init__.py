"""**Core module** for loading, organizing, and analyzing EEG data.

This module provides the fundamental building blocks for the NeuRodent analysis pipeline,
handling everything from raw data loading to feature extraction.

**Typical Workflow:**

.. code-block:: python

    from neurodent import core

    # 1. Load and organize recordings
    organizer = core.LongRecordingOrganizer(data_path)
    organizer.parse_dates_from_filenames()
    organizer.infer_channel_names()

    # 2. Run windowed feature analysis
    analyzer = core.LongRecordingAnalyzer(organizer)
    results = analyzer.run_analysis(
        window_sec=4,
        step_sec=4,
        features=['psdband', 'cohere'],
    )

    # 3. Access results
    df = results.features_df  # pandas DataFrame with all features
    results.save('output.pkl')

**What Gets Computed:**

- **Power features**: RMS, amplitude variance, band power (delta, theta, alpha, beta, gamma)
- **Connectivity**: Coherence, imaginary coherence, Pearson correlation between channels
- **Spikes**: Spike counts via MountainSort5 or frequency-domain detection

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
    DDFBinaryMetadata,
    convert_ddfcolbin_to_ddfrowbin,
    convert_ddfrowbin_to_si,
    split_recording,
)
from .analysis import LongRecordingAnalyzer
from .analyze_frag import FragmentAnalyzer
from .analyze_sort import MountainSortAnalyzer
from .frequency_domain_spike_detection import FrequencyDomainSpikeDetector
from .utils import (
    get_temp_directory,
    set_temp_directory,
    parse_path_to_animalday,
    validate_timestamps,
    nanaverage,
    parse_chname_to_abbrev,
    log_transform,
    should_use_cache_unified,
    get_feature_label,
    get_cache_status_message,
)

from .zeitgeber import (
    ZeitgeberAnalysisResult,
    run_zeitgeber_pipeline,
    get_expanded_feature_names,
    transform_time_axis,
)
from . import utils

__all__ = [
    # Data loading
    "DDFBinaryMetadata",
    "convert_ddfcolbin_to_ddfrowbin",
    "convert_ddfrowbin_to_si",
    "split_recording",
    "LongRecordingOrganizer",
    # Analysis
    "LongRecordingAnalyzer",
    "FragmentAnalyzer",
    "MountainSortAnalyzer",
    "FrequencyDomainSpikeDetector",
    "ZeitgeberAnalysisResult",
    "run_zeitgeber_pipeline",
    "get_expanded_feature_names",
    "transform_time_axis",
    # Utilities
    "convert_ddfcolbin_to_ddfrowbin",
    "convert_ddfrowbin_to_si",
    "get_temp_directory",
    "set_temp_directory",
    "parse_path_to_animalday",
    "validate_timestamps",
    "nanaverage",
    "parse_chname_to_abbrev",
    "log_transform",
    "get_feature_label",
    "get_cache_status_message",
    "should_use_cache_unified",
    "utils",
]
