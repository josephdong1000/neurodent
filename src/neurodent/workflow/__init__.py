"""
Workflow Utilities
==================

Utilities for Snakemake workflow scripts and general data loading.

This module provides:

- :func:`setup_snakemake_logging`: Configure logging for Snakemake scripts
- :func:`load_wars`: Load multiple WindowAnalysisResult objects
- :func:`expand_animals_config`: Expand unified ``animals`` list config into legacy keys
"""

from .utils import (
    setup_snakemake_logging,
    load_wars,
    apply_samples_config,
    expand_animals_config,
    extend_plot_order_from_attr,
    create_sex_marker_scale,
)

__all__ = [
    "setup_snakemake_logging",
    "load_wars",
    "apply_samples_config",
    "expand_animals_config",
    "extend_plot_order_from_attr",
    "create_sex_marker_scale",
]
