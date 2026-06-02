"""Backward-compatible re-export shim.

This module historically contained multiple large classes.
It is kept as a thin compatibility layer so existing imports and tests keep working.
"""

from __future__ import annotations

# These imports are kept at module scope for backward-compatible patch paths in tests.
import dask
import dask.array as da
from dask import delayed

from .. import constants, core

from .feature_parser import AnimalFeatureParser, _sanitize_feature_request
from .animal_organizer import AnimalOrganizer
from .window_analysis_result import WindowAnalysisResult, bin_spike_times, _bin_spike_df

__all__ = [
    "AnimalFeatureParser",
    "AnimalOrganizer",
    "WindowAnalysisResult",
    "_sanitize_feature_request",
    "bin_spike_times",
    "_bin_spike_df",
]
