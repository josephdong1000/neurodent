"""
Workflow Utilities
==================

Utilities for Snakemake workflow scripts and general data loading.

This module provides:

- :func:`setup_snakemake_logging`: Configure logging for Snakemake scripts
- :func:`load_wars`: Load multiple WindowAnalysisResult objects
"""

from .utils import setup_snakemake_logging, load_wars, inject_config_aliases, build_discovery_pattern

__all__ = ["setup_snakemake_logging", "load_wars", "inject_config_aliases", "build_discovery_pattern"]
