"""Backward-compat shim; canonical home is neurodent.core.results.window_analysis_result (#110)."""
from ..core.results.window_analysis_result import *  # noqa: F401,F403
from ..core.results.window_analysis_result import _sanitize_feature_request, _bin_spike_df  # noqa: F401
