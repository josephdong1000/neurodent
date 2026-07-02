"""Result containers and their support utilities (issue #110).

A neutral shared layer: the loading/analysis code and the plotting code both
depend downward into it. Holds ``WindowAnalysisResult``,
``FrequencyDomainSpikeAnalysisResult``, ``ZeitgeberAnalysisResult`` and their
feature/filter/streaming helpers, so nothing here imports ``neurodent.visualization``.
"""

from .window_analysis_result import (
    WindowAnalysisResult,
    bin_spike_times,
    _bin_spike_df,
    _sanitize_feature_request,
)
from .frequency_domain_results import FrequencyDomainSpikeAnalysisResult
from .feature_utils import (
    average_feature,
    extract_linear_array,
    extract_band_dict,
    repack_band_dict,
    extract_hist_data,
    extract_feature,
    format_channel_data,
    flatten_feature_for_plotting,
    collapse_feature_channels,
)
from .feature_handlers import FEATURE_HANDLERS, handler_for
from .filters import (
    FILTER_REGISTRY,
    ChannelInfo,
    FilterScope,
    compute_filter_mask,
    update_bad_channels_dict_from_config,
)
from .streaming import LazyWindowAnalysisResult
from .zeitgeber import (
    ZeitgeberAnalysisResult,
    run_zeitgeber_pipeline,
    get_expanded_feature_names,
    transform_time_axis,
    expand_zt_axis,
)

__all__ = [
    "WindowAnalysisResult",
    "FrequencyDomainSpikeAnalysisResult",
    "ZeitgeberAnalysisResult",
    "LazyWindowAnalysisResult",
    "average_feature",
    "bin_spike_times",
    "run_zeitgeber_pipeline",
    "get_expanded_feature_names",
    "transform_time_axis",
    "expand_zt_axis",
    "FEATURE_HANDLERS",
    "handler_for",
    "FILTER_REGISTRY",
    "ChannelInfo",
    "FilterScope",
    "compute_filter_mask",
    "update_bad_channels_dict_from_config",
]
