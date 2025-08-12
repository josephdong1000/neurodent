import os
import tempfile

# Ensure a usable temporary directory is available for downstream modules
if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = tempfile.gettempdir()

# Import core classes and functions immediately for better IDE support
# Note: This will import heavier dependencies (MNE, SpikeInterface) but provides
# better documentation and autocomplete experience
from .core import (
    LongRecordingOrganizer,
    DDFBinaryMetadata,
    convert_ddfcolbin_to_ddfrowbin,
    convert_ddfrowbin_to_si,
)
from .analyze_frag import FragmentAnalyzer
from .utils import (
    get_temp_directory,
    set_temp_directory,
    nanaverage,
    log_transform,
    parse_chname_to_abbrev,
    parse_path_to_animalday,
)
# Import analysis after core to avoid circular imports
from .analysis import LongRecordingAnalyzer

__all__: list[str] = [
    # From .core (immediate imports)
    "LongRecordingOrganizer",
    "DDFBinaryMetadata",
    "convert_ddfcolbin_to_ddfrowbin",
    "convert_ddfrowbin_to_si",
    # From .analysis (immediate import)
    "LongRecordingAnalyzer",
    # From .analyze_frag (immediate import)
    "FragmentAnalyzer",
    # From .utils (immediate imports)
    "get_temp_directory",
    "nanaverage",
    "log_transform",
    "parse_chname_to_abbrev",
    "parse_path_to_animalday",
]