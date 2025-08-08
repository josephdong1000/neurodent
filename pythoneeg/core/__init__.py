import os
import tempfile
from typing import Any

# Ensure a usable temporary directory is available for downstream modules
if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = tempfile.gettempdir()

# Public API for pythoneeg.core
#
# Use lazy attribute loading to avoid importing heavy optional dependencies
# (e.g., MNE, SpikeInterface) unless actually requested by the caller.
# This keeps `import pythoneeg.core` lightweight while still enabling the
# ergonomic style `core.LongRecordingOrganizer` and
# `from pythoneeg.core import FragmentAnalyzer`.

__all__: list[str] = [
    # From .core
    "LongRecordingOrganizer",
    "DDFBinaryMetadata",
    "convert_ddfcolbin_to_ddfrowbin",
    "convert_ddfrowbin_to_si",
    # From .analysis
    "LongRecordingAnalyzer",
    # From .analyze_frag
    "FragmentAnalyzer",
    # From .utils
    "get_temp_directory",
    "nanaverage",
    "log_transform",
    "parse_chname_to_abbrev",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy imports
    if name in {
        "LongRecordingOrganizer",
        "DDFBinaryMetadata",
        "convert_ddfcolbin_to_ddfrowbin",
        "convert_ddfrowbin_to_si",
    }:
        from .core import (
            LongRecordingOrganizer,
            DDFBinaryMetadata,
            convert_ddfcolbin_to_ddfrowbin,
            convert_ddfrowbin_to_si,
        )

        return {
            "LongRecordingOrganizer": LongRecordingOrganizer,
            "DDFBinaryMetadata": DDFBinaryMetadata,
            "convert_ddfcolbin_to_ddfrowbin": convert_ddfcolbin_to_ddfrowbin,
            "convert_ddfrowbin_to_si": convert_ddfrowbin_to_si,
        }[name]

    if name == "LongRecordingAnalyzer":
        from .analysis import LongRecordingAnalyzer

        return LongRecordingAnalyzer

    if name == "FragmentAnalyzer":
        from .analyze_frag import FragmentAnalyzer

        return FragmentAnalyzer

    if name in {"get_temp_directory", "set_temp_directory", "nanaverage", "log_transform", "parse_chname_to_abbrev"}:
        from .utils import (
            get_temp_directory,
            set_temp_directory,
            nanaverage,
            log_transform,
            parse_chname_to_abbrev,
        )

        return {
            "get_temp_directory": get_temp_directory,
            "set_temp_directory": set_temp_directory,
            "nanaverage": nanaverage,
            "log_transform": log_transform,
            "parse_chname_to_abbrev": parse_chname_to_abbrev,
        }[name]

    raise AttributeError(f"module 'pythoneeg.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)