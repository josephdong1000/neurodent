"""Discovery pattern/filter resolution for the Snakemake pipeline."""


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
