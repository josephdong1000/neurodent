"""neurodent.workflow.utils - workflow utility functions.

Split from a single 916-line module into cohesive submodules. All 14 public
names are re-exported here so ``from neurodent.workflow.utils import X`` keeps
working for every consumer; package-internal convenience, not a back-compat shim.
"""

from .config import (
    load_samples_config,
    resolve_samples_config,
    apply_samples_config,
    deep_merge_dict,
    expand_animals_config,
    apply_path_overrides,
    format_config_value,
)
from .discovery import (
    get_discovery_animal_filter,
    resolve_animal_pattern,
)
from .logging import (
    setup_snakemake_logging,
    increment_memory,
)
from .plotting_helpers import (
    load_wars,
    extend_plot_order_from_attr,
    create_sex_marker_scale,
)

__all__ = [
    "load_samples_config",
    "resolve_samples_config",
    "apply_samples_config",
    "deep_merge_dict",
    "expand_animals_config",
    "apply_path_overrides",
    "format_config_value",
    "get_discovery_animal_filter",
    "resolve_animal_pattern",
    "setup_snakemake_logging",
    "increment_memory",
    "load_wars",
    "extend_plot_order_from_attr",
    "create_sex_marker_scale",
]
