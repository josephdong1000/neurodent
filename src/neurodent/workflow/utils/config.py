"""Samples-config lifecycle and config-dict helpers for Snakemake workflows."""

import copy
import json
from pathlib import Path


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


def apply_samples_config(samples_config: dict):
    """Apply a samples config to the global ``neurodent.constants`` (pipeline front door).

    Installs the per-dataset globals from one samples config: the channel map
    (:data:`CHANNEL_MAP`, via :func:`~neurodent.set_channel_map`), the exact
    ``genotype``/``sex`` value maps (``GENOTYPE_MAP``/``SEX_MAP``), and
    ``ANIMAL_METADATA``. Completes the ``load_samples_config`` → ``resolve_samples_config``
    → ``apply_samples_config`` lifecycle, and should be called at the start of every
    Snakemake script that loads WindowAnalysisResults or resolves channel names.

    The ``genotype``/``sex`` value maps are set **completely**: a dataset that omits a
    ``GENOTYPE_MAP``/``SEX_MAP`` block resets that map to its module default
    (:data:`~neurodent.constants.DEFAULT_GENOTYPE_MAP` / :data:`~neurodent.constants.DEFAULT_SEX_MAP`),
    so applying config A then config B in the same process never leaks A's map into B.
    ``CHANNEL_MAP`` and ``ANIMAL_METADATA`` remain sticky (only updated when present) —
    every real dataset declares both, and a stale ``ANIMAL_METADATA`` surfaces loudly as a
    ``KeyError`` rather than a silent mis-normalization.

    Args:
        samples_config (dict): Configuration dictionary loaded from samples.json
    """
    from neurodent import constants
    from neurodent.core import metadata as metadata_module

    # Channels: the flat CHANNEL_MAP is the single source of truth. set_channel_map()
    # derives CHANNEL_ABBREVS / CHANNEL_ABBREV_BY_RAW / DF_SORT_ORDER / standardization target
    # / LOF channels from it. A dataset declares its channels under `channels`.
    channels = samples_config.get("channels")
    if channels:
        constants.set_channel_map(channels)
    # Exact value maps for the `sex` / `genotype` metadata fields (parallel mechanisms);
    # applied before load_animal_metadata below so the normalization picks them up.
    # Assigned UNCONDITIONALLY (complete, not sticky): a dataset that omits a block resets
    # that map to its module default, so applying config A then config B in one process
    # never leaks A's map into B (which, under strict normalization, would wrongly raise).
    # Deep-copied to avoid aliasing the config dict or the canonical default.
    constants.SEX_MAP = copy.deepcopy(samples_config.get("SEX_MAP", constants.DEFAULT_SEX_MAP))
    constants.GENOTYPE_MAP = copy.deepcopy(samples_config.get("GENOTYPE_MAP", constants.DEFAULT_GENOTYPE_MAP))

    # New: ANIMAL_METADATA for sex/genotype enrichment (required)
    if "ANIMAL_METADATA" in samples_config:
        constants.ANIMAL_METADATA = metadata_module.load_animal_metadata(samples_config)


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
                        "skip_sessions": ["*bad*"],
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
            # merged["analysis"]["war_generation"]["skip_sessions"] == ["*bad*"] (preserved)
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


def merge_dataset_config(config: dict, dataset: str, datasets_dir="config/datasets"):
    """Deep-merge a dataset's config file into a base pipeline config.

    Resolves ``{datasets_dir}/{dataset}.yaml``, parses it, and deep-merges it into
    ``config`` (dataset values win). Shared by the Snakemake workflow and
    standalone tools so the two never drift on how a dataset config is applied.

    Args:
        config (dict): Base pipeline config (already loaded, e.g. from config.yaml
            + config.local.yaml).
        dataset (str): Dataset name (the ``config/datasets/{name}.yaml`` stem).
        datasets_dir (str | Path): Directory holding dataset config files.

    Returns:
        tuple[dict, dict]: ``(merged_config, dataset_config)`` where
        ``dataset_config`` is the raw parsed dataset file (useful for logging the
        overrides that were applied).

    Raises:
        FileNotFoundError: If the dataset config file does not exist; the message
            lists the available datasets.
    """
    import yaml

    datasets_dir = Path(datasets_dir)
    dataset_config_file = datasets_dir / f"{dataset}.yaml"
    if not dataset_config_file.exists():
        available = sorted(p.stem for p in datasets_dir.glob("*.yaml")) if datasets_dir.is_dir() else []
        raise FileNotFoundError(
            f"Dataset config file not found: {dataset_config_file}\n"
            f"Available datasets: {', '.join(available) if available else 'None'}"
        )
    with open(dataset_config_file, "r") as f:
        dataset_config = yaml.safe_load(f) or {}
    return deep_merge_dict(config, dataset_config), dataset_config


def load_dataset_config(
    dataset: str,
    config_path="config/config.yaml",
    local_path="config/config.local.yaml",
    datasets_dir="config/datasets",
) -> dict:
    """Assemble the merged pipeline config for a dataset, outside Snakemake.

    Reproduces the Snakefile's config assembly for standalone scripts: load the
    base ``config.yaml``, deep-merge ``config.local.yaml`` when it parses to a
    dict, then deep-merge ``config/datasets/{dataset}.yaml`` (via
    :func:`merge_dataset_config`). Uses the same :func:`deep_merge_dict` as the
    workflow so a standalone tool and the pipeline resolve identical config.

    Note: unlike the Snakefile, this does not run the ``config.schema.yaml``
    validation (which requires Snakemake); callers that need strict validation
    should validate separately.

    Args:
        dataset (str): Dataset name (``config/datasets/{name}.yaml`` stem).
        config_path (str | Path): Base config file.
        local_path (str | Path): Optional local-override file; ignored if missing
            or not a mapping.
        datasets_dir (str | Path): Directory holding dataset config files.

    Returns:
        dict: The merged pipeline config.

    Raises:
        FileNotFoundError: If ``config_path`` or the dataset config is missing.
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    local_path = Path(local_path)
    if local_path.exists():
        with open(local_path, "r") as f:
            local_config = yaml.safe_load(f)
        if isinstance(local_config, dict):
            config = deep_merge_dict(config, local_config)

    merged, _ = merge_dataset_config(config, dataset, datasets_dir=datasets_dir)
    return merged


def enumerate_cohort(samples_config: dict) -> list:
    """Return the ordered, de-duplicated list of animal ids in a samples config.

    Mirrors the Snakefile's cohort enumeration: read ids from the unified
    ``animals`` list. Shared with standalone tools (e.g. the labeling cohort
    bundler) so they iterate the same animals as the pipeline.

    Args:
        samples_config (dict): Samples config (expanded or not) containing an
            ``animals`` list of ``{"id": ...}`` dicts.

    Returns:
        list[str]: Animal ids, order-preserving, duplicates removed.

    Raises:
        KeyError: If the samples config has no ``animals`` list.
    """
    if "animals" not in samples_config:
        raise KeyError("Samples config must contain an 'animals' list")
    ids = {}
    for entry in samples_config["animals"]:
        ids.setdefault(entry["id"], None)
    return list(ids)


def expand_animals_config(samples_config: dict) -> dict:
    """Expand the unified ``animals`` list into pipeline keys.

    When the samples config contains an ``animals`` key (a list of animal
    dicts), this function produces:

    * ``ANIMAL_METADATA`` – list of ``{id, genotype, sex, ...}`` dicts
    * ``manual_datetimes`` – animal_id → datetime string mapping
    * ``bad_channels`` – animal_id → {session → [channels]} mapping
      (built from per-animal ``bad_channels`` entries)
    * ``_animal_overrides`` – animal_id → per-animal overrides dict (pattern,
      lro_kwargs)
    * ``_animal_channel_subsets`` – animal_id → channel list mapping (for joint sessions)
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
        ...         {"id": "A10", "genotype": "WT", "sex": "M"},
        ...         {"id": "F22", "genotype": "KO", "sex": "F"},
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
        ...         {"id": "X1", "genotype": "WT", "sex": "M",
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
        ...         {"id": "A10", "genotype": "WT", "sex": "M",
        ...          "bad_channels": ["LHip", "RHip"]},
        ...     ],
        ... })
        >>> cfg["bad_channels"]["A10"]
        {'_all': ['LHip', 'RHip']}

    Bad channels (dict format for per-session)::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "genotype": "WT", "sex": "M",
        ...          "bad_channels": {"Session1": ["LHip"], "Session2": ["RMot"]}},
        ...     ],
        ... })
        >>> cfg["bad_channels"]["A10"]
        {'Session1': ['LHip'], 'Session2': ['RMot']}

    Joint sessions with a channel subset::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "genotype": "WT", "sex": "M",
        ...          "channel_subset": ["Ch0", "Ch1", "Ch2", "Ch3"]},
        ...     ],
        ... })
        >>> cfg["_animal_channel_subsets"]["A10"]
        ['Ch0', 'Ch1', 'Ch2', 'Ch3']

    Joint sessions with group::

        >>> cfg = expand_animals_config({
        ...     "data_root": "/data",
        ...     "animals": [
        ...         {"id": "A10", "genotype": "WT", "sex": "M",
        ...          "channel_subset": ["Ch0", "Ch1"],
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
    _OVERRIDE_KEYS = {"pattern", "lro_kwargs", "manual_datetime", "datetimes_are_start", "bad_channels", "exclude", "channel_subset", "group"}
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

    # --- Build _animal_channel_subsets and _animal_groups ---
    animal_channel_subsets: dict[str, list[str]] = {}
    animal_groups: dict[str, str] = {}

    for animal in animals_list:
        if "channel_subset" in animal:
            animal_channel_subsets[animal["id"]] = animal["channel_subset"]
        if "group" in animal:
            animal_groups[animal["id"]] = animal["group"]

    # Validate no overlapping channels within the same group
    if animal_channel_subsets and animal_groups:
        # Group animals by their group name
        groups_to_animals: dict[str, list[str]] = {}
        for animal_id, group_name in animal_groups.items():
            groups_to_animals.setdefault(group_name, []).append(animal_id)

        # Check for channel overlaps within each group
        for group_name, animal_ids in groups_to_animals.items():
            # Get all channels for animals in this group
            all_channels_in_group: list[tuple[str, str]] = []  # (animal_id, channel)
            for animal_id in animal_ids:
                if animal_id in animal_channel_subsets:
                    for channel in animal_channel_subsets[animal_id]:
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

    if animal_channel_subsets:
        result["_animal_channel_subsets"] = animal_channel_subsets
    if animal_groups:
        result["_animal_groups"] = animal_groups

    # --- Build _animal_overrides ---
    overrides: dict[str, dict] = {}
    for animal in animals_list:
        animal_overrides = {}
        for key in ("pattern", "lro_kwargs"):
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
