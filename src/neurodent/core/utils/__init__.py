"""Shared low-level utilities (issue #69).

``core.utils`` was split from a single 1900-line module into cohesive
submodules. These names are re-exported here for internal convenience so
``from neurodent.core.utils import X`` keeps working; this is package-internal
convenience, not a back-compat shim.
"""

from .units import (
    convert_units_to_multiplier,
    extract_mne_unit_info,
    log_transform,
)
from .time import (
    is_day,
    parse_str_to_day,
    _clean_str_for_date,
    TimestampMapper,
    validate_timestamps,
)
from .naming import (
    resolve_channel,
    resolve_channels,
    parse_str_to_animal,
    normalize_value_from_aliases,
    rename_mne_channels,
    slugify,
    get_feature_label,
)
from .paths import (
    set_temp_directory,
    get_temp_directory,
    safe_unlink,
    is_si_recording_folder,
    safe_rmtree,
    atomic_output_path,
    atomic_write_json,
    convert_colpath_to_rowpath,
    filepath_to_index,
    get_file_stem,
)
from .caching import (
    cache_fragments_to_zarr,
    stream_fragments_to_zarr,
    stream_recording_to_zarr,
    should_use_cached_file,
    get_cache_status_message,
    should_use_cache_unified,
)
from .dataframe import (
    nanaverage,
    nanmean_series_of_np,
    sort_dataframe_by_plot_order,
    _get_groupby_keys,
    _get_pairwise_combinations,
)
from .neighbors import (
    Natural_Neighbor,
    chunked_channel_distance_matrix,
)
from .misc import (
    parse_truncate,
    _HiddenPrints,
)

__all__ = [
    "convert_units_to_multiplier",
    "extract_mne_unit_info",
    "log_transform",
    "is_day",
    "parse_str_to_day",
    "TimestampMapper",
    "validate_timestamps",
    "resolve_channel",
    "resolve_channels",
    "parse_str_to_animal",
    "normalize_value_from_aliases",
    "rename_mne_channels",
    "slugify",
    "get_feature_label",
    "set_temp_directory",
    "get_temp_directory",
    "safe_unlink",
    "is_si_recording_folder",
    "safe_rmtree",
    "atomic_output_path",
    "atomic_write_json",
    "convert_colpath_to_rowpath",
    "filepath_to_index",
    "get_file_stem",
    "cache_fragments_to_zarr",
    "stream_fragments_to_zarr",
    "stream_recording_to_zarr",
    "should_use_cached_file",
    "get_cache_status_message",
    "should_use_cache_unified",
    "nanaverage",
    "nanmean_series_of_np",
    "sort_dataframe_by_plot_order",
    "Natural_Neighbor",
    "chunked_channel_distance_matrix",
    "parse_truncate",
]
