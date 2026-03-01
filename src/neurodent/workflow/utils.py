"""
Workflow utility functions.

This module provides utilities that reduce boilerplate in Snakemake workflow scripts.
"""

import copy
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurodent.visualization import WindowAnalysisResult


def setup_snakemake_logging(snakemake) -> logging.Logger:
    """Configure logging to write to the Snakemake log file.

    This replaces the common boilerplate pattern in workflow scripts that
    redirects stdout/stderr to the log file specified in the Snakemake rule.

    Args:
        snakemake: The global snakemake object injected by Snakemake.
            Must have a ``log`` attribute containing the log file path.

    Returns:
        logging.Logger: A configured logger instance.

    Example:
        In a Snakemake script::

            from neurodent.workflow import setup_snakemake_logging

            def main():
                logger = setup_snakemake_logging(snakemake)
                logger.info("Starting processing...")

            if __name__ == "__main__":
                main()

    Note:
        The log file path is determined by the ``log:`` directive in your
        Snakemake rule. For example::

            rule my_rule:
                log: "logs/my_rule.log"
                script: "scripts/my_script.py"

        The logger will write to ``logs/my_rule.log``.
    """
    log_path = snakemake.log[0]
    log_file = open(log_path, "w")

    # Redirect stdout and stderr to the log file
    sys.stdout = log_file
    sys.stderr = log_file

    # Configure logging to use the redirected stdout
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )

    return logging.getLogger(__name__)


def inject_config_aliases(samples_config: dict):
    """Inject aliases from samples_config into the global neurodent.constants.

    This ensures that custom aliases for genotypes, channel names, and L/R labels
    are available across all modules in the pipeline. This should be called at the
    beginning of every Snakemake script that loads WindowAnalysisResults or uses
    channel name parsing.

    Args:
        samples_config (dict): Configuration dictionary loaded from samples.json
    """
    from neurodent import constants
    from neurodent.core import metadata as metadata_module

    # Legacy: GENOTYPE_ALIASES for file path parsing (parse_str_to_genotype)
    if "GENOTYPE_ALIASES" in samples_config:
        constants.GENOTYPE_ALIASES = samples_config["GENOTYPE_ALIASES"]
    if "CHNAME_ALIASES" in samples_config:
        constants.CHNAME_ALIASES = samples_config["CHNAME_ALIASES"]
    if "LR_ALIASES" in samples_config:
        constants.LR_ALIASES = samples_config["LR_ALIASES"]
    
    # New: ANIMAL_METADATA for sex/gene enrichment (required)
    if "ANIMAL_METADATA" in samples_config:
        constants.ANIMAL_METADATA = metadata_module.load_animal_metadata(samples_config)


def load_wars(
    pkl_paths: list[str | Path],
    json_paths: list[str | Path] | None = None,
) -> list["WindowAnalysisResult"]:
    """Load multiple WindowAnalysisResult objects from pickle/json file pairs.

    General-purpose utility for loading WAR files. Works with any list of paths,
    not tied to Snakemake.

    Args:
        pkl_paths: Paths to .pkl files containing WindowAnalysisResult data.
        json_paths: Optional paths to corresponding .json metadata files.
            If None, assumes json files are in the same directory as pkl files
            with the same basename but .json extension.

    Returns:
        List of loaded WindowAnalysisResult objects.

    Raises:
        FileNotFoundError: If a pkl or json file does not exist.
        RuntimeError: If no WARs could be loaded.

    Example:
        Load WARs from explicit paths::

            from neurodent.workflow import load_wars

            wars = load_wars(
                pkl_paths=["data/animal1/war.pkl", "data/animal2/war.pkl"],
                json_paths=["data/animal1/war.json", "data/animal2/war.json"],
            )

        Load WARs with auto-detected json paths::

            wars = load_wars(pkl_paths=["data/animal1/war.pkl"])
            # Automatically looks for data/animal1/war.json
    """
    from neurodent import visualization

    # If json_paths not provided, derive from pkl_paths
    if json_paths is None:
        json_paths = [Path(p).with_suffix(".json") for p in pkl_paths]

    if len(pkl_paths) != len(json_paths):
        raise ValueError(
            f"pkl_paths ({len(pkl_paths)}) and json_paths ({len(json_paths)}) "
            "must have the same length"
        )

    wars = []
    for pkl_path, json_path in zip(pkl_paths, json_paths):
        pkl_path = Path(pkl_path)
        json_path = Path(json_path)

        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=pkl_path.parent,
            pickle_name=pkl_path.name,
            json_name=json_path.name,
        )
        wars.append(war)

    if not wars:
        raise RuntimeError("No WARs were successfully loaded")

    return wars


def deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    This function performs a deep merge, recursively merging nested dictionaries.
    Non-dict values in override will replace corresponding values in base.

    Used in the Snakefile to merge dataset-specific configurations into the main
    configuration, allowing any nested parameter to be overridden.

    Args:
        base: Base dictionary to merge into
        override: Dictionary with override values

    Returns:
        Merged dictionary with values from both base and override

    Examples:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"d": 4, "e": 5}, "f": 6}
        >>> deep_merge_dict(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 4, 'e': 5}, 'f': 6}

        Real-world config merge::

            # Main config
            base = {
                "samples": {"quality_filter": {"exclude_unknown": True}},
                "analysis": {
                    "war_generation": {
                        "day_sep": None,
                        "lro_kwargs": {"multiprocess_mode": "dask"}
                    }
                }
            }

            # Dataset override
            override = {
                "samples": {"samples_file": "config/custom.json"},
                "analysis": {
                    "war_generation": {
                        "mode": "base",
                        "lro_kwargs": {"extract_func": "read_intan"}
                    }
                }
            }

            # Result preserves nested values from both
            merged = deep_merge_dict(base, override)
            # merged["samples"]["quality_filter"]["exclude_unknown"] == True (preserved)
            # merged["samples"]["samples_file"] == "config/custom.json" (added)
            # merged["analysis"]["war_generation"]["day_sep"] == None (preserved)
            # merged["analysis"]["war_generation"]["mode"] == "base" (added)
            # merged["analysis"]["war_generation"]["lro_kwargs"]["multiprocess_mode"] == "dask" (preserved)
            # merged["analysis"]["war_generation"]["lro_kwargs"]["extract_func"] == "read_intan" (added)

    Note:
        This function does NOT mutate the input dictionaries - it returns a new dict.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dict(result[key], value)
        else:
            # Override value (or add new key)
            result[key] = value
    return result


def build_discovery_pattern(
    base_path: str | Path,
    mode: str | None = None,
    file_pattern: str | None = None,
    pattern: str | list[str] | None = None,
) -> str | list[str]:
    """Convert pipeline config to AnimalOrganizer discovery pattern.

    Supports both the new ``pattern`` config key and backward-compatible
    conversion from the old ``mode`` + ``file_pattern`` style.

    When ``pattern`` is provided it takes precedence: the template is
    prepended with *base_path* and returned directly.

    Otherwise the function builds a pattern string from the legacy *mode*
    and *file_pattern* parameters.

    For ``"nest"`` mode the file-name portion uses an ``{index}`` placeholder
    so that the ``FileDiscoverer`` regex can match discovered paths.  For
    ``"base"``/``"concat"`` modes, plain globs (no placeholders) are used
    instead, which lets ``FileDiscoverer`` return paths without metadata.

    Args:
        base_path: Root folder for this session (e.g.
            ``data_parent_folder / folder_path``).
        mode: Legacy AnimalOrganizer mode (``"nest"``, ``"base"``,
            ``"concat"``).  Used only when *pattern* is ``None``.
        file_pattern: Legacy file glob (e.g. ``"*.rhd"``).  Used only
            when *pattern* is ``None``.
        pattern: New-style pattern template **relative** to *base_path*.
            May contain ``{animal}``, ``{session}``, ``{index}``
            placeholders.  May be a list for multi-file patterns.

    Returns:
        A discovery pattern string (or list of strings) suitable for
        ``AnimalOrganizer(pattern=...)``.

    Examples:
        New-style with explicit pattern::

            >>> build_discovery_pattern("/data/s1", pattern="{animal}/{session}/{index}.bin")
            '/data/s1/{animal}/{session}/{index}.bin'

        Legacy nest mode::

            >>> build_discovery_pattern("/data/s1", mode="nest")
            '/data/s1/{animal}/{session}/{index}'

        Legacy base mode with file filter::

            >>> build_discovery_pattern("/data/s1", mode="base", file_pattern="*.rhd")
            '/data/s1/*.rhd'

    Raises:
        ValueError: If neither *pattern* nor *mode* is provided.
    """
    base = str(base_path).rstrip("/")

    # New-style: explicit pattern template
    if pattern is not None:
        if isinstance(pattern, list):
            return [f"{base}/{p}" for p in pattern]
        return f"{base}/{pattern}"

    # Legacy conversion
    fp = file_pattern or "*"

    if mode == "nest":
        # Nest mode uses {animal}/{session} placeholders, so the file-name
        # portion must also be a placeholder for the regex to work.
        # Convert glob wildcards (e.g. "*.rhd") to "{index}.rhd".
        fp_placeholder = _glob_to_index_placeholder(fp)
        return f"{base}/{{animal}}/{{session}}/{fp_placeholder}"
    elif mode in ("base", "concat"):
        # Flat modes: no placeholders needed, pure glob
        return f"{base}/{fp}"
    elif mode is not None:
        return f"{base}/{fp}"
    else:
        raise ValueError(
            "Either 'pattern' or 'mode' must be provided in the "
            "war_generation config to build a discovery pattern."
        )


def _glob_to_index_placeholder(file_pattern: str) -> str:
    """Convert a file glob pattern to use an ``{index}`` placeholder.

    This is needed when the pattern already contains ``{animal}``/``{session}``
    placeholders because ``FileDiscoverer`` escapes literal ``*`` in the regex.

    Only the **first** ``*`` in *file_pattern* is replaced.  Patterns with
    multiple wildcards (e.g. ``*_data_*.bin``) are uncommon in practice;
    use an explicit ``pattern`` config key with named placeholders for
    such cases instead of relying on the legacy ``mode`` + ``file_pattern``
    conversion.

    Examples:
        >>> _glob_to_index_placeholder("*")
        '{index}'
        >>> _glob_to_index_placeholder("*.rhd")
        '{index}.rhd'
        >>> _glob_to_index_placeholder("*_ColMajor.bin")
        '{index}_ColMajor.bin'
    """
    if "*" in file_pattern:
        return file_pattern.replace("*", "{index}", 1)
    return file_pattern


def apply_path_overrides(base_config: dict, overrides: dict) -> dict:
    """Apply path-based overrides to a config dictionary using deep merge.

    This function allows overriding nested configuration values using dotted path notation,
    enabling flexible session-specific, animal-specific, or other granular overrides in the
    Snakemake pipeline.

    Args:
        base_config: Base configuration dictionary
        overrides: Dict mapping dotted paths to values
                  e.g., {"analysis.war_generation.file_pattern": "*.EDF"}

    Returns:
        Merged configuration with overrides applied

    Raises:
        KeyError: If a path references a non-dict intermediate value
        ValueError: If override path is empty or malformed

    Examples:
        Basic usage with nested paths::

            >>> config = {"analysis": {"war_generation": {"mode": "base"}}}
            >>> overrides = {"analysis.war_generation.file_pattern": "*.EDF"}
            >>> result = apply_path_overrides(config, overrides)
            >>> result["analysis"]["war_generation"]["file_pattern"]
            '*.EDF'

        Creating new nested keys::

            >>> config = {"existing": "value"}
            >>> overrides = {"new.nested.key": "new_value"}
            >>> result = apply_path_overrides(config, overrides)
            >>> result["new"]["nested"]["key"]
            'new_value'

        Real-world session-specific override::

            # Base dataset config
            config = {
                "analysis": {
                    "war_generation": {
                        "mode": "base",
                        "lro_kwargs": {"mode": "si", "input_type": "files"}
                    }
                }
            }

            # Session-specific overrides for EDF format
            overrides = {
                "analysis.war_generation.file_pattern": "*.EDF",
                "analysis.war_generation.lro_kwargs.extract_func": "read_edf"
            }

            result = apply_path_overrides(config, overrides)
            # result["analysis"]["war_generation"]["file_pattern"] == "*.EDF"
            # result["analysis"]["war_generation"]["lro_kwargs"]["extract_func"] == "read_edf"
            # result["analysis"]["war_generation"]["lro_kwargs"]["mode"] == "si" (preserved)

    Note:
        This function does NOT mutate the input config - it returns a new deep copy.
    """
    if not overrides:
        return copy.deepcopy(base_config)

    result = copy.deepcopy(base_config)

    for path, value in overrides.items():
        if not path:
            raise ValueError("Override path cannot be empty")

        keys = path.split('.')
        target = result

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            elif not isinstance(target[key], dict):
                raise KeyError(
                    f"Cannot override '{path}': intermediate key '{key}' "
                    f"is {type(target[key]).__name__}, not dict"
                )
            target = target[key]

        # Set the value
        target[keys[-1]] = value

    return result
