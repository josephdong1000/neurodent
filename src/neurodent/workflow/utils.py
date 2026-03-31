"""
Workflow utility functions.

This module provides utilities that reduce boilerplate in Snakemake workflow scripts.
"""

import copy
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from neurodent.visualization import WindowAnalysisResult


class AnimalFolder(NamedTuple):
    """A single folder/session entry for an animal in the pipeline.

    Attributes:
        folder_path: Path to the data folder. Empty string for regular animals
            that use pattern-based discovery.
        animal_id: The original (non-slugified) animal ID string.
        session_key: Session identifier. Empty string for regular animals;
            matches the joint-session name for joint-session animals.
    """

    folder_path: str
    animal_id: str
    session_key: str


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
        Snakemake rule. Log files use ``.out`` and ``.err`` extensions and
        are organized under ``logs/<rule_group>/``. For example::

            rule my_rule:
                log:
                    stdout="logs/my_rule/{animal}.out",
                    stderr="logs/my_rule/{animal}.err",
                script: "scripts/my_script.py"

        The logger will write to ``logs/my_rule/{animal}.out``.
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
                        "pattern": "{index}.rhd",
                        "lro_kwargs": {"extract_func": "read_intan"}
                    }
                }
            }

            # Result preserves nested values from both
            merged = deep_merge_dict(base, override)
            # merged["samples"]["quality_filter"]["exclude_unknown"] == True (preserved)
            # merged["samples"]["samples_file"] == "config/custom.json" (added)
            # merged["analysis"]["war_generation"]["day_sep"] == None (preserved)
            # merged["analysis"]["war_generation"]["pattern"] == "{index}.rhd" (added)
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


def expand_animals_config(samples_config: dict) -> dict:
    """Expand the unified ``animals`` list into pipeline keys.

    When the samples config contains an ``animals`` key (a list of animal
    dicts), this function produces:

    * ``ANIMAL_METADATA`` – list of ``{id, gene, sex, ...}`` dicts
    * ``manual_datetimes`` – animal_id → datetime string mapping
    * ``GENOTYPE_ALIASES`` – gene → [animal_id, …] mapping (auto-generated
      from ``gene`` field unless already present)
    * ``bad_channels`` – animal_id → {session → [channels]} mapping
      (built from per-animal ``bad_channels`` entries)
    * ``_animal_overrides`` – animal_id → per-animal overrides dict (pattern,
      lro_kwargs, day_parse_kwargs)

    ``data_root`` is the canonical path key.  If the legacy key
    ``data_parent_folder`` is present it is migrated to ``data_root``.

    If ``animals`` is not present the config is returned unchanged (a deep
    copy is always made so the caller never sees mutations).

    Parameters
    ----------
    samples_config : dict
        Samples configuration loaded from a ``samples_*.json`` file.

    Returns
    -------
    dict
        Expanded configuration with pipeline keys populated.

    Examples
    --------
    Minimal config with two animals::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "gene": "WT", "sex": "M"},
        ...         {"id": "F22", "gene": "KO", "sex": "F"},
        ...     ],
        ... })
        >>> cfg["data_root"]
        '/data'
        >>> "A10" in dict([(e["id"], e) for e in cfg["ANIMAL_METADATA"]])
        True

    Per-animal overrides::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "X1", "gene": "WT", "sex": "M",
        ...          "pattern": "{data_root}/custom/{animal}_{index}.rhd",
        ...          "lro_kwargs": {"mode": "si"},
        ...          "manual_datetime": "2025-01-01 10:00:00"},
        ...     ],
        ... })
        >>> cfg["_animal_overrides"]["X1"]["pattern"]
        '{data_root}/custom/{animal}_{index}.rhd'
        >>> cfg["manual_datetimes"]["X1"]
        '2025-01-01 10:00:00'

    Bad channels (list format for all sessions)::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "gene": "WT", "sex": "M",
        ...          "bad_channels": ["LHip", "RHip"]},
        ...     ],
        ... })
        >>> cfg["bad_channels"]["A10"]
        {'_all': ['LHip', 'RHip']}

    Bad channels (dict format for per-session)::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "gene": "WT", "sex": "M",
        ...          "bad_channels": {"Session1": ["LHip"], "Session2": ["RMot"]}},
        ...     ],
        ... })
        >>> cfg["bad_channels"]["A10"]
        {'Session1': ['LHip'], 'Session2': ['RMot']}
    """
    result = copy.deepcopy(samples_config)

    # Migrate legacy data_parent_folder → data_root
    if "data_parent_folder" in result and "data_root" not in result:
        result["data_root"] = result.pop("data_parent_folder")

    if "animals" not in result:
        return result

    animals_list = result["animals"]

    # Keys that are per-animal overrides (not core metadata)
    _OVERRIDE_KEYS = {"pattern", "lro_kwargs", "day_parse_kwargs", "manual_datetime", "datetimes_are_start", "bad_channels"}
    _METADATA_SKIP = _OVERRIDE_KEYS  # excluded from ANIMAL_METADATA entries

    # --- Build ANIMAL_METADATA ---
    if "ANIMAL_METADATA" not in result:
        result["ANIMAL_METADATA"] = []
    existing_ids = {e["id"] for e in result["ANIMAL_METADATA"]}

    for animal in animals_list:
        if animal["id"] not in existing_ids:
            meta_entry = {k: v for k, v in animal.items() if k not in _METADATA_SKIP}
            result["ANIMAL_METADATA"].append(meta_entry)
            existing_ids.add(animal["id"])

    # --- Build manual_datetimes ---
    if "manual_datetimes" not in result:
        result["manual_datetimes"] = {}
    for animal in animals_list:
        if "manual_datetime" in animal:
            result["manual_datetimes"][animal["id"]] = animal["manual_datetime"]

    # --- Auto-generate GENOTYPE_ALIASES from gene field ---
    if "GENOTYPE_ALIASES" not in result:
        gene_to_animals: dict[str, list[str]] = {}
        for animal in animals_list:
            gene = animal.get("gene")
            if gene:
                gene_to_animals.setdefault(gene, []).append(animal["id"])
        if gene_to_animals:
            result["GENOTYPE_ALIASES"] = gene_to_animals

    # --- Build bad_channels ---
    if "bad_channels" not in result:
        result["bad_channels"] = {}
    for animal in animals_list:
        if "bad_channels" in animal:
            bc = animal["bad_channels"]
            if isinstance(bc, list):
                # List format: channels bad across all sessions
                result["bad_channels"][animal["id"]] = {"_all": bc}
            elif isinstance(bc, dict):
                # Dict format: session → bad channels mapping
                result["bad_channels"][animal["id"]] = bc

    # --- Build _animal_overrides ---
    overrides: dict[str, dict] = {}
    for animal in animals_list:
        animal_overrides = {}
        for key in ("pattern", "lro_kwargs", "day_parse_kwargs"):
            if key in animal:
                animal_overrides[key] = animal[key]
        # Propagate datetimes_are_start into lro_kwargs override
        if "datetimes_are_start" in animal:
            lro_kw = animal_overrides.setdefault("lro_kwargs", {})
            lro_kw.setdefault("datetimes_are_start", animal["datetimes_are_start"])
        if animal_overrides:
            overrides[animal["id"]] = animal_overrides
    if overrides:
        result["_animal_overrides"] = overrides

    return result


def resolve_animal_pattern(
    pattern_config,
    animal_id: str,
    data_root: str,
) -> "str | list[str]":
    """Resolve a discovery pattern for a specific animal.

    Substitutes ``{data_root}`` in patterns with the actual data root path.

    Supports two formats for ``pattern_config``:

    * **Shared** (``str`` or ``list[str]``): every animal uses the same pattern(s).
    * **Per-animal** (``dict[str, str | list[str]]``): each animal has its own
      pattern(s), enabling heterogeneous file structures in a single dataset.

    Parameters
    ----------
    pattern_config : str | list[str] | dict[str, str | list[str]]
        Either a shared pattern (string or list), or a dict mapping
        ``animal_id → pattern(s)``.  Patterns may contain ``{data_root}``
        which will be replaced with the *data_root* value.
    animal_id : str
        The animal to resolve the pattern for.
    data_root : str
        Absolute path to the data root directory, substituted for
        ``{data_root}`` in patterns.

    Returns
    -------
    str | list[str]
        Absolute discovery pattern(s) for the given animal.

    Raises
    ------
    KeyError
        If ``pattern_config`` is a dict and ``animal_id`` is not found.

    Examples
    --------
    Pattern with ``{data_root}``::

        >>> resolve_animal_pattern(
        ...     "{data_root}/session1/{animal}/{index}.nwb", "A10", "/data"
        ... )
        '/data/session1/{animal}/{index}.nwb'

    List of patterns::

        >>> resolve_animal_pattern(
        ...     ["{data_root}/{animal}/{index}.bin", "{data_root}/{animal}/{index}.csv"],
        ...     "A10",
        ...     "/data",
        ... )
        ['/data/{animal}/{index}.bin', '/data/{animal}/{index}.csv']

    Per-animal pattern dict::

        >>> resolve_animal_pattern(
        ...     {"A10": "{data_root}/{animal}/{index}.rhd"},
        ...     "A10",
        ...     "/data",
        ... )
        '/data/{animal}/{index}.rhd'
    """
    # Per-animal patterns: dict mapping animal_id → pattern(s)
    if isinstance(pattern_config, dict):
        if animal_id not in pattern_config:
            raise KeyError(
                f"Animal '{animal_id}' not found in per-animal pattern config. "
                f"Available animals: {list(pattern_config.keys())}"
            )
        pattern = pattern_config[animal_id]
    else:
        # Shared pattern (string or list)
        pattern = pattern_config

    def _resolve(p: str) -> str:
        return p.replace("{data_root}", data_root)

    if isinstance(pattern, list):
        return [_resolve(p) for p in pattern]
    else:
        return _resolve(pattern)


def apply_path_overrides(base_config: dict, overrides: dict) -> dict:
    """Apply path-based overrides to a config dictionary using deep merge.

    This function allows overriding nested configuration values using dotted path notation,
    enabling flexible session-specific, animal-specific, or other granular overrides in the
    Snakemake pipeline.

    Args:
        base_config: Base configuration dictionary
        overrides: Dict mapping dotted paths to values
                  e.g., {"analysis.war_generation.pattern": "{index}.EDF"}

    Returns:
        Merged configuration with overrides applied

    Raises:
        KeyError: If a path references a non-dict intermediate value
        ValueError: If override path is empty or malformed

    Examples:
        Basic usage with nested paths::

            >>> config = {"analysis": {"war_generation": {"pattern": "{index}"}}}
            >>> overrides = {"analysis.war_generation.pattern": "{index}.EDF"}
            >>> result = apply_path_overrides(config, overrides)
            >>> result["analysis"]["war_generation"]["pattern"]
            '{index}.EDF'

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
                        "pattern": "{index}",
                        "lro_kwargs": {"mode": "si"}
                    }
                }
            }

            # Session-specific overrides for EDF format
            overrides = {
                "analysis.war_generation.pattern": "{index}.EDF",
                "analysis.war_generation.lro_kwargs.mode": "mne",
                "analysis.war_generation.lro_kwargs.extract_func": "read_raw_edf"
            }

            result = apply_path_overrides(config, overrides)
            # result["analysis"]["war_generation"]["pattern"] == "{index}.EDF"
            # result["analysis"]["war_generation"]["lro_kwargs"]["extract_func"] == "read_raw_edf"
            # result["analysis"]["war_generation"]["lro_kwargs"]["mode"] == "mne" (overridden)

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


# ---------------------------------------------------------------------------
# Snakefile helpers
# ---------------------------------------------------------------------------
# These pure helpers are used by the Snakefile but live here so they can be
# imported and unit-tested without requiring the Snakemake runtime.


def format_config_value(value, indent=4):
    """Format a config value for display (handles nested dicts, lists, etc.)."""
    spaces = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for k, v in value.items():
            formatted_val = format_config_value(v, indent + 2)
            if "\n" in formatted_val:
                lines.append(f"{spaces}{k}:")
                lines.append(formatted_val)
            else:
                lines.append(f"{spaces}{k}: {formatted_val}")
        return "\n".join(lines)
    elif isinstance(value, list):
        if not value:
            return "[]"
        return f"[{', '.join(repr(v) for v in value)}]"
    elif isinstance(value, str):
        return f'"{value}"'
    elif value is None:
        return "null"
    else:
        return str(value)


def increment_memory(base_memory):
    """Return a callable ``mem(wildcards, attempt)`` that doubles on each retry.

    Used by Snakemake rules to exponentially increase memory on retries::

        resources:
            mem_mb=increment_memory(4000),
    """
    def mem(wildcards, attempt):
        return base_memory * (2 ** (attempt - 1))
    return mem
