"""Workflow utility functions."""

from .config import (
    load_samples_config,
    resolve_samples_config,
    apply_samples_config,
    deep_merge_dict,
    merge_dataset_config,
    load_dataset_config,
    analysis_config,
    enumerate_cohort,
    expand_animals_config,
    apply_path_overrides,
    format_config_value,
)
from .discovery import (
    get_discovery_animal_filter,
    resolve_animal_pattern,
)
from .animal_loading import load_animal_recordings
from .quality_filter import (
    write_exclusion_record,
    clear_exclusion_record,
    quality_filter_outcome,
    format_exclusion_summary,
)
from .logging import (
    setup_snakemake_logging,
    rule_log,
    increment_memory,
)
from .resources import (
    count_war_values,
    war_stream_bytes,
    war_input_mem_mb,
    one_war_at_a_time_time,
    rule_resources,
)
from .plotting_helpers import (
    iter_wars,
    load_wars,
    extend_plot_order_from_attr,
    create_genotype_color_scale,
    create_sex_marker_scale,
)

__all__ = [
    "load_samples_config",
    "resolve_samples_config",
    "apply_samples_config",
    "deep_merge_dict",
    "merge_dataset_config",
    "load_dataset_config",
    "analysis_config",
    "enumerate_cohort",
    "expand_animals_config",
    "apply_path_overrides",
    "format_config_value",
    "get_discovery_animal_filter",
    "resolve_animal_pattern",
    "load_animal_recordings",
    "write_exclusion_record",
    "clear_exclusion_record",
    "quality_filter_outcome",
    "format_exclusion_summary",
    "setup_snakemake_logging",
    "rule_log",
    "increment_memory",
    "count_war_values",
    "war_stream_bytes",
    "war_input_mem_mb",
    "one_war_at_a_time_time",
    "rule_resources",
    "iter_wars",
    "load_wars",
    "extend_plot_order_from_attr",
    "create_genotype_color_scale",
    "create_sex_marker_scale",
]
