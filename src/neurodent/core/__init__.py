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
)
from .analysis import LongRecordingAnalyzer
from .analyze_frag import FragmentAnalyzer
from .analyze_sort import MountainSortAnalyzer
from .frequency_domain_spike_detection import FrequencyDomainSpikeDetector

# Essential utilities for users
from .utils import (
    get_temp_directory,
    set_temp_directory,
    parse_chname_to_abbrev,
    parse_path_to_animalday,
    validate_timestamps,
    nanaverage,
    log_transform,
    get_cache_status_message,
    should_use_cache_unified,
)

from . import utils

__all__: list[str] = [
    # === PUBLIC API ===
    # Core classes
    "LongRecordingOrganizer",
    "DDFBinaryMetadata",
    "convert_ddfcolbin_to_ddfrowbin",
    "convert_ddfrowbin_to_si",
    "LongRecordingAnalyzer",
    "FragmentAnalyzer",
    "MountainSortAnalyzer",
    "FrequencyDomainSpikeDetector",
    # Essential utilities
    "get_temp_directory",
    "set_temp_directory",
    "parse_chname_to_abbrev",
    "parse_path_to_animalday",
    "validate_timestamps",
    "nanaverage",
    "log_transform",
    "get_cache_status_message",
    "should_use_cache_unified",
    # === INTERNAL/ADVANCED ===
    "utils",  # Access via core.utils.function_name for internal functions
]
