"""Backwards-compatible re-export shim for the former monolithic ``results.py``.

The public objects now live in dedicated modules (issue #134):

- :class:`~neurodent.visualization.animal_organizer.AnimalOrganizer`
- :class:`~neurodent.visualization.window_analysis_result.WindowAnalysisResult`

and the feature-averaging helper moved to
:func:`neurodent.visualization.feature_utils.average_feature` (issue #136).

This module is kept so ``from neurodent.visualization.results import ...`` and the
existing test suite keep working unchanged. The ``constants``/``core``/``dask``/``da``
imports below are intentional: tests patch names like
``neurodent.visualization.results.core.LongRecordingOrganizer``, which mutate the
shared module objects that ``animal_organizer`` also references.
"""

import dask
import dask.array as da
from dask import delayed

from .. import constants, core
from .animal_organizer import AnimalOrganizer
from .window_analysis_result import (
    WindowAnalysisResult,
    _sanitize_feature_request,
    bin_spike_times,
    _bin_spike_df,
)

__all__ = ["AnimalOrganizer", "WindowAnalysisResult", "bin_spike_times"]
