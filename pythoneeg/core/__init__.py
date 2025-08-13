import os
import tempfile

# Ensure a usable temporary directory is available for downstream modules
if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = tempfile.gettempdir()

# Import core classes and functions immediately for better IDE support
# Note: This will import heavier dependencies (MNE, SpikeInterface) but provides
# better documentation and autocomplete experience

# === PUBLIC API ===
# Core classes
from .core import (
    LongRecordingOrganizer,
    DDFBinaryMetadata,
    convert_ddfcolbin_to_ddfrowbin,
    convert_ddfrowbin_to_si,
)
from .analyze_frag import FragmentAnalyzer
from .analysis import LongRecordingAnalyzer

# Essential utilities for users
from .utils import (
    get_temp_directory,
    set_temp_directory,
    parse_chname_to_abbrev,
    parse_path_to_animalday,
    validate_timestamps,
    nanaverage,
    log_transform,
)

# === INTERNAL/ADVANCED UTILITIES ===
# Import the utils module for internal functions
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
    # Essential utilities
    "get_temp_directory",
    "set_temp_directory", 
    "parse_chname_to_abbrev",
    "parse_path_to_animalday",
    "validate_timestamps",
    "nanaverage",
    "log_transform",
    # === INTERNAL/ADVANCED ===
    "utils",  # Access via core.utils.function_name for internal functions
]