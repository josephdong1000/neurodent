"""
Workflow utility functions.

This module provides utilities that reduce boilerplate in Snakemake workflow scripts.
"""

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurodent.visualization import WindowAnalysisResult


def load_samples_config(path: "str | Path") -> dict:
    """Load a samples config from a ``.yaml``/``.yml`` or ``.json`` file.

    The format is chosen from the file suffix, so samples configs can be
    migrated to YAML one dataset at a time while older ``.json`` configs keep
    working unchanged.

    Args:
        path: Path to the samples config file. Must end in ``.yaml``, ``.yml``,
            or ``.json``.

    Returns:
        dict: The parsed samples config.

    Raises:
        ValueError: If the file suffix is not a supported config format.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)
    if suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(
        f"Unsupported samples config format: '{path.suffix}' ({path}). "
        "Expected .yaml, .yml, or .json."
    )


def resolve_samples_config(config: dict) -> dict:
    """Resolve the samples config from a merged pipeline ``config``.

    Supports two dataset shapes:

    - **Single-file** datasets carry the samples inventory inline under a
      top-level ``samples_data`` key (the dataset config and the samples
      inventory live in one file).
    - **Two-file** datasets carry a ``samples.samples_file`` path pointing at a
      separate ``.json``/``.yaml`` samples file (loaded via
      :func:`load_samples_config`).

    Args:
        config: The merged Snakemake pipeline config dict.

    Returns:
        dict: The (unexpanded) samples config.

    Raises:
        KeyError: If neither ``samples_data`` nor ``samples.samples_file`` is present.
    """
    if config.get("samples_data") is not None:
        return config["samples_data"]
    samples = config.get("samples", {})
    if samples.get("samples_file"):
        return load_samples_config(samples["samples_file"])
    raise KeyError(
        "Config needs a top-level 'samples_data' block (inline) or "
        "'samples.samples_file' (path)."
    )


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
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
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

    This ensures that custom aliases for genotypes, channel names, L/R labels, and the
    ``sex``/``gene`` metadata value normalizers (``SEX_ALIASES``/``GENE_ALIASES``) are
    available across all modules in the pipeline. This should be called at the
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
    # Value normalizers for the `sex` / `gene` metadata fields (parallel mechanisms);
    # injected before load_animal_metadata below so the normalization picks them up.
    if "SEX_ALIASES" in samples_config:
        constants.SEX_ALIASES = samples_config["SEX_ALIASES"]
    if "GENE_ALIASES" in samples_config:
        constants.GENE_ALIASES = samples_config["GENE_ALIASES"]

    # New: ANIMAL_METADATA for sex/gene enrichment (required)
    if "ANIMAL_METADATA" in samples_config:
        constants.ANIMAL_METADATA = metadata_module.load_animal_metadata(samples_config)


def load_wars(
    parquet_paths: list[str | Path],
    json_paths: list[str | Path] | None = None,
) -> list["WindowAnalysisResult"]:
    """Load multiple WindowAnalysisResult objects from parquet/json file pairs.

    General-purpose utility for loading WAR files. Works with any list of paths,
    not tied to Snakemake.

    Args:
        parquet_paths: Paths to .parquet files containing WindowAnalysisResult data.
            For backward compatibility, legacy .pkl paths are also accepted — the
            loader will resolve the corresponding .parquet file next to them and
            fall back to the pickle only if the parquet is missing.
        json_paths: Optional paths to corresponding .json metadata files.
            If None, assumes json files are in the same directory as the parquet
            files with the same basename but .json extension.

    Returns:
        List of loaded WindowAnalysisResult objects.

    Raises:
        FileNotFoundError: If a parquet or json file does not exist.
        RuntimeError: If no WARs could be loaded.

    Example:
        Load WARs from explicit paths::

            from neurodent.workflow import load_wars

            wars = load_wars(
                parquet_paths=["data/animal1/war.parquet", "data/animal2/war.parquet"],
                json_paths=["data/animal1/war.json", "data/animal2/war.json"],
            )

        Load WARs with auto-detected json paths::

            wars = load_wars(parquet_paths=["data/animal1/war.parquet"])
            # Automatically looks for data/animal1/war.json
    """
    from neurodent import visualization

    # If json_paths not provided, derive from parquet_paths
    if json_paths is None:
        json_paths = [Path(p).with_suffix(".json") for p in parquet_paths]

    if len(parquet_paths) != len(json_paths):
        raise ValueError(
            f"parquet_paths ({len(parquet_paths)}) and json_paths ({len(json_paths)}) "
            "must have the same length"
        )

    wars = []
    for parquet_path, json_path in zip(parquet_paths, json_paths):
        parquet_path = Path(parquet_path)
        json_path = Path(json_path)

        # Accept legacy .pkl input by swapping the suffix
        if parquet_path.suffix == ".pkl":
            parquet_path = parquet_path.with_suffix(".parquet")

        war = visualization.WindowAnalysisResult.load_parquet_and_json(
            folder_path=parquet_path.parent,
            parquet_name=parquet_path.name,
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
    * ``bad_channels`` – animal_id → {session → [channels]} mapping
      (built from per-animal ``bad_channels`` entries)
    * ``_animal_overrides`` – animal_id → per-animal overrides dict (pattern,
      lro_kwargs, day_parse_kwargs)
    * ``_animal_channels`` – animal_id → channel list mapping (for joint sessions)
    * ``_animal_groups`` – animal_id → group string mapping (for joint sessions)

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

    Joint sessions with channels::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "gene": "WT", "sex": "M",
        ...          "channels": ["Ch0", "Ch1", "Ch2", "Ch3"]},
        ...     ],
        ... })
        >>> cfg["_animal_channels"]["A10"]
        ['Ch0', 'Ch1', 'Ch2', 'Ch3']

    Joint sessions with group::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "gene": "WT", "sex": "M",
        ...          "channels": ["Ch0", "Ch1"],
        ...          "group": "SharedGroup"},
        ...     ],
        ... })
        >>> cfg["_animal_groups"]["A10"]
        'SharedGroup'
    """
    result = copy.deepcopy(samples_config)

    # Migrate legacy data_parent_folder → data_root
    if "data_parent_folder" in result and "data_root" not in result:
        result["data_root"] = result.pop("data_parent_folder")

    if "animals" not in result:
        return result

    # Filter out excluded animals from the working list and from result["animals"]
    # so downstream consumers (e.g. the Snakefile) don't need their own exclude check.
    # The excluded entries are preserved in the on-disk samples.json for documentation.
    animals_list = [a for a in result["animals"] if not a.get("exclude", False)]
    result["animals"] = animals_list

    # Keys that are per-animal overrides (not core metadata)
    _OVERRIDE_KEYS = {"pattern", "lro_kwargs", "day_parse_kwargs", "manual_datetime", "datetimes_are_start", "bad_channels", "exclude", "channels", "group"}
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

    # --- Build _animal_channels and _animal_groups ---
    animal_channels: dict[str, list[str]] = {}
    animal_groups: dict[str, str] = {}

    for animal in animals_list:
        if "channels" in animal:
            animal_channels[animal["id"]] = animal["channels"]
        if "group" in animal:
            animal_groups[animal["id"]] = animal["group"]

    # Validate no overlapping channels within the same group
    if animal_channels and animal_groups:
        # Group animals by their group name
        groups_to_animals: dict[str, list[str]] = {}
        for animal_id, group_name in animal_groups.items():
            groups_to_animals.setdefault(group_name, []).append(animal_id)

        # Check for channel overlaps within each group
        for group_name, animal_ids in groups_to_animals.items():
            # Get all channels for animals in this group
            all_channels_in_group: list[tuple[str, str]] = []  # (animal_id, channel)
            for animal_id in animal_ids:
                if animal_id in animal_channels:
                    for channel in animal_channels[animal_id]:
                        all_channels_in_group.append((animal_id, channel))

            # Check for duplicates
            seen_channels: dict[str, str] = {}  # channel -> first animal_id
            for animal_id, channel in all_channels_in_group:
                if channel in seen_channels:
                    raise ValueError(
                        f"Channel '{channel}' is assigned to both '{seen_channels[channel]}' "
                        f"and '{animal_id}' in group '{group_name}'. "
                        f"Animals in the same joint recording cannot share channels."
                    )
                seen_channels[channel] = animal_id

    if animal_channels:
        result["_animal_channels"] = animal_channels
    if animal_groups:
        result["_animal_groups"] = animal_groups

    # --- Backward compatibility: derive channels from legacy joint_sessions ---
    # Note: This only derives _animal_channels, not _animal_groups.
    # For legacy configs where folder names don't contain animal IDs,
    # migration to the new format with explicit 'group' fields is required.
    if "joint_sessions" in result and result["joint_sessions"]:
        # Check if any animals already have channels defined
        has_new_format = any("channels" in a for a in animals_list)

        if not has_new_format:
            # Auto-derive from legacy format with deprecation warning
            import warnings
            warnings.warn(
                "The 'joint_sessions' configuration format is deprecated. "
                "Please migrate to the unified 'animals' format by adding 'channels' "
                "and optionally 'group' fields to animal entries. "
                "See the animals configuration documentation for details.",
                DeprecationWarning,
                stacklevel=2
            )

            # Derive channels from joint_sessions
            if "_animal_channels" not in result:
                result["_animal_channels"] = {}

            for session_name, animals_dict in result["joint_sessions"].items():
                for animal_id, channels in animals_dict.items():
                    if animal_id in result["_animal_channels"]:
                        # Verify consistency
                        if result["_animal_channels"][animal_id] != channels:
                            raise ValueError(
                                f"Inconsistent channel lists for {animal_id} across joint sessions. "
                                f"Expected {result['_animal_channels'][animal_id]}, got {channels} in {session_name}."
                            )
                    else:
                        result["_animal_channels"][animal_id] = channels

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


def get_discovery_animal_filter(
    source_animal_id: str,
    is_joint: bool,
    animal_groups: dict[str, str],
) -> str:
    """Determine the animal filter value for discovery.

    For joint sessions with 'group', use the group name for {animal} placeholder.
    For joint sessions without 'group', use the animal id.
    For non-joint sessions, use the animal id as usual.

    Parameters
    ----------
    source_animal_id : str
        The animal ID from the configuration
    is_joint : bool
        Whether this is a joint session
    animal_groups : dict[str, str]
        Mapping of animal_id to group name

    Returns
    -------
    str
        The value to use for {animal} placeholder in discovery pattern

    Examples
    --------
    Regular non-joint animal::

        >>> get_discovery_animal_filter("A10", False, {})
        'A10'

    Joint session with group (e.g., arx_rosa)::

        >>> groups = {"ArxRosa-1017": "Arx Rosa 1017 1015", "ArxRosa-1015": "Arx Rosa 1017 1015"}
        >>> get_discovery_animal_filter("ArxRosa-1017", True, groups)
        'Arx Rosa 1017 1015'

    Joint session without group (e.g., jess_rhd where folders contain animal IDs)::

        >>> get_discovery_animal_filter("AP3B2het-207-M", True, {})
        'AP3B2het-207-M'
    """
    if is_joint and source_animal_id in animal_groups:
        # Use group name for discovery (folder contains group name, not individual animal ID)
        return animal_groups[source_animal_id]
    else:
        # Joint session without group OR regular non-joint session: use animal ID
        return source_animal_id


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


def extend_plot_order_from_attr(wars, attr: str, base_order):
    """Extend a plot-order list with the values of *attr* observed on *wars*.

    Mirrors the dynamic-extension pattern used in EP plotting scripts:
    start from a base order (typically ``constants.DF_SORT_ORDER[attr]``)
    and append any values seen on the loaded WARs that aren't already in
    that base.  This keeps strict plot-order validation happy for datasets
    with non-default category values (e.g. arxrosa, where every animal has
    ``sex='Unknown'``).

    Args:
        wars: Iterable of objects exposing ``attr`` (typically
            :class:`WindowAnalysisResult` instances).
        attr: The attribute / column name to extend (``"genotype"``,
            ``"sex"``, ...).
        base_order: Starting list of category values; not mutated.

    Returns:
        list: ``list(base_order)`` with any newly-observed truthy values of
            ``getattr(war, attr)`` appended, preserving insertion order
            relative to *base_order*.

    Example::

        >>> from neurodent import constants
        >>> base = constants.DF_SORT_ORDER["sex"]   # ["Male", "Female"]
        >>> class W: pass
        >>> w1, w2 = W(), W()
        >>> w1.sex, w2.sex = "Male", "Unknown"
        >>> extend_plot_order_from_attr([w1, w2], "sex", base)
        ['Male', 'Female', 'Unknown']
    """
    order = list(base_order)
    seen = set(order)
    for war in wars:
        v = getattr(war, attr, None)
        if not v or v in seen:
            continue
        logging.info(f"Adding unknown {attr} '{v}' to plot order")
        order.append(v)
        seen.add(v)
    return order


def build_sex_marker_scale(df, plot_lib=None):
    """Build a seaborn-objects marker scale for the sex column of *df*.

    Preserves the canonical Female=circle (``"o"``), Male=square (``"s"``)
    mapping when those values are present, and assigns a diamond (``"D"``)
    fallback marker for any non-canonical sex value (e.g. arxrosa's
    ``"Unknown"``).

    Why this exists: ep_figures plots use ``so.Plot(..., marker="sex")``
    with a static ``so.Nominal(["o", "s"], order=["Female", "Male"])``
    scale. seaborn-objects **silently drops** any row whose sex value
    isn't listed in ``order``. Datasets with non-canonical sex (arxrosa)
    therefore produce blank plots with no error. This helper makes the
    scale's order/markers track what's actually in ``df``.

    Args:
        df: DataFrame with a ``"sex"`` column.
        plot_lib: Optional reference to ``seaborn.objects``. If ``None``,
            it's imported lazily so this util doesn't require seaborn at
            module-load time (useful for tests that don't render plots).

    Returns:
        A ``seaborn.objects.Nominal`` scale instance.

    Example::

        >>> import pandas as pd
        >>> df = pd.DataFrame({"sex": ["Female", "Male", "Female"]})
        >>> scale = build_sex_marker_scale(df)
        >>> scale.order
        ['Female', 'Male']
        >>> scale.values
        ['o', 's']
    """
    if plot_lib is None:
        import seaborn.objects as so
        plot_lib = so

    sex_marker_map = {"Female": "o", "Male": "s"}
    fallback_marker = "D"
    observed = list(df["sex"].dropna().unique())
    # Preserve canonical Female/Male ordering when present; append the rest.
    order = [s for s in ["Female", "Male"] if s in observed] + [
        s for s in observed if s not in ("Female", "Male")
    ]
    markers = [sex_marker_map.get(s, fallback_marker) for s in order]
    return plot_lib.Nominal(markers, order=order)
