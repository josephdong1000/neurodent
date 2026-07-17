"""Shared per-animal recording loader.

The per-animal *load-and-consolidate* step used by WAR generation is factored out
here so that both the pipeline (``workflow/scripts/generate_wars.py``) and
standalone tools (e.g. the labeling cohort bundler) build recordings the same
way, with no risk of the two implementations drifting apart.

This is a workflow-layer helper: it depends on :mod:`neurodent.loading`
(``AnimalOrganizer``) and the sibling discovery/config utilities. It performs no
Dask / analysis / WAR computation — only file discovery, loading, optional
joint-session channel splitting, and consolidation into a single
:class:`~neurodent.loading.AnimalOrganizer`.
"""

import logging
from pathlib import Path

from neurodent import constants
from neurodent.loading import AnimalOrganizer

from .config import apply_path_overrides
from .discovery import get_discovery_animal_filter, resolve_animal_pattern


def load_animal_recordings(
    samples_config,
    config,
    animal_folders,
    animal_id,
    channel_subset=None,
    logger=None,
):
    """Discover, load, and consolidate one animal's recordings into an ``AnimalOrganizer``.

    Mirrors the loading half of the WAR-generation pipeline: for each source
    folder/session it resolves the discovery pattern, applies session- and
    per-animal config overrides (``datetimes_are_start``, ``manual_datetimes``,
    ``lro_kwargs``, ``pattern``), builds an :class:`~neurodent.loading.AnimalOrganizer`,
    optionally splits a joint session down to this animal's channels, and finally
    consolidates every discovered ``LongRecordingOrganizer`` into one
    ``AnimalOrganizer`` via :meth:`AnimalOrganizer.from_lros`.

    Genotype and sex are resolved from
    :data:`neurodent.constants.ANIMAL_METADATA` (populated by
    ``apply_samples_config``), which must therefore have been applied before
    calling this function.

    Args:
        samples_config (dict): Expanded samples config (must contain ``data_root``
            and the ``_animal_*`` keys produced by ``expand_animals_config``).
        config (dict): Merged pipeline config; reads
            ``config["analysis"]["war_generation"]`` and, when present,
            ``config["overrides"]["by_session"]``.
        animal_folders (list[tuple]): ``(folder_path, source_animal_id,
            session_key)`` tuples for this animal (as built by the Snakefile; a
            standalone caller passes ``[("", animal_id, "")]``).
        animal_id (str): Canonical animal id; must be present in
            ``constants.ANIMAL_METADATA``.
        channel_subset (list[str] | None, optional): Channels assigned to this
            animal for a joint session; ``None`` for a non-joint animal.
        logger (logging.Logger | None, optional): Logger for progress messages; a
            module logger is used when ``None``.

    Returns:
        AnimalOrganizer: Consolidated organizer with ``genotype``/``sex`` set and
        ``long_recordings`` populated.

    Raises:
        KeyError: If ``animal_id`` is not in ``constants.ANIMAL_METADATA`` or a
            session has no ``pattern`` configured.
        ValueError: If no recordings are discovered for the animal.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Set up paths
    data_root = Path(samples_config.get("data_root", samples_config.get("data_parent_folder", "")))

    all_lros = []
    analysis_config = config["analysis"]["war_generation"]

    # Resolve genotype from metadata (Metadata-First)
    if animal_id not in constants.ANIMAL_METADATA:
        raise KeyError(
            f"Animal '{animal_id}' (from {animal_folders[0][0]}) not found in ANIMAL_METADATA. "
            "All animals in the pipeline must be defined in the metadata for reliable processing."
        )

    meta = constants.ANIMAL_METADATA[animal_id]
    genotype = meta.get("genotype", "Unknown")
    sex = meta.get("sex", "Unknown")
    logger.info(f"Resolved genotype '{genotype}' and sex '{sex}' for {animal_id} from ANIMAL_METADATA")

    # Load data from all source folders
    for folder_info in animal_folders:
        # Unpack tuple from Snakefile
        folder_path, source_animal_id, session_key = folder_info

        logger.info(f"Loading session: {folder_path} (ID in metadata: {source_animal_id})")

        # Check if this animal has channels defined (indicates joint session)
        is_joint = source_animal_id in samples_config.get("_animal_channel_subsets", {})

        # Apply session-specific overrides from dataset config
        session_analysis_config = analysis_config.copy()

        if "overrides" in config and "by_session" in config["overrides"]:
            session_overrides = config["overrides"]["by_session"].get(session_key, {})
            if session_overrides:
                logger.info(f"  -> Applying session overrides: {list(session_overrides.keys())}")
                # Apply path-based overrides to the full config
                overridden_config = apply_path_overrides(config, session_overrides)
                session_analysis_config = overridden_config["analysis"]["war_generation"]

        # Prepare kwargs for this specific session
        session_lro_kwargs = dict(session_analysis_config.get("lro_kwargs", {}))

        # Propagate datetimes_are_start from war_generation config into lro_kwargs
        # (it lives at the war_generation level, not inside lro_kwargs)
        if "datetimes_are_start" in session_analysis_config:
            session_lro_kwargs.setdefault("datetimes_are_start", session_analysis_config["datetimes_are_start"])

        # Apply per-animal overrides from unified animals config
        animal_overrides = samples_config.get("_animal_overrides", {}).get(animal_id, {})
        if animal_overrides:
            logger.info(f"  -> Applying per-animal overrides: {list(animal_overrides.keys())}")
            if "lro_kwargs" in animal_overrides:
                session_lro_kwargs.update(animal_overrides["lro_kwargs"])

        # Sessions to skip: dataset-level plus any per-animal override (e.g. a setup/test stub that
        # a single animal has but isn't a real recording day). Both are fnmatch patterns on {session}.
        skip_sessions = list(session_analysis_config.get("skip_sessions", session_analysis_config.get("skip_days", [])))
        skip_sessions += list(animal_overrides.get("skip_sessions", []))

        # Resolve manual_datetimes for this session. The per-animal value may be a
        # scalar (one start time), a dict (keyed per session/file), or a list (per-recording
        # order, possibly nested); AnimalOrganizer distributes it across discovered sessions.
        if "manual_datetimes" in samples_config:
            all_manual_dts = samples_config["manual_datetimes"]
            if animal_id in all_manual_dts:
                session_lro_kwargs["manual_datetimes"] = all_manual_dts[animal_id]
                logger.info(f"  -> Using manual datetimes for {animal_id}")

        # Build absolute discovery pattern from the config's relative pattern
        # Per-animal pattern override takes precedence over session/default config
        effective_pattern = animal_overrides.get("pattern", session_analysis_config.get("pattern"))
        if effective_pattern is None:
            raise KeyError(
                f"Missing 'pattern' key in war_generation config for session '{session_key}'. "
                "Each dataset config must specify 'pattern' (e.g. '{{animal}}/{{session}}/{{index}}.nwb' "
                "or '{{index}}.rhd')."
            )

        logger.info(f"  -> File pattern: {effective_pattern}")
        discovery_pattern = resolve_animal_pattern(
            effective_pattern,
            source_animal_id,
            data_root=str(data_root),
        )
        logger.info(f"  -> Discovery pattern: {discovery_pattern}")

        # Determine the animal filter value for discovery
        animal_groups = samples_config.get("_animal_groups", {})
        discovery_animal_filter = get_discovery_animal_filter(
            source_animal_id, is_joint, animal_groups
        )
        if is_joint and source_animal_id in animal_groups:
            logger.info(f"  -> Using group '{discovery_animal_filter}' for {{animal}} placeholder in discovery")
        elif is_joint:
            logger.info(f"  -> Using animal ID '{discovery_animal_filter}' for discovery (joint session without group)")

        # Create AO for this session using pattern-based discovery
        session_ao = AnimalOrganizer(
            discovery_pattern,
            animal_id=discovery_animal_filter,
            skip_sessions=skip_sessions,
            lro_kwargs=session_lro_kwargs,
        )

        if is_joint and channel_subset is not None:
            logger.info(f"  -> Joint session detected. Filtering to channels: {channel_subset}")
            # Split to only the channels assigned to this animal
            # source_animal_id is the key in the splits dict
            splits = session_ao.split(groups={source_animal_id: channel_subset})
            session_ao = splits[source_animal_id]

        # Collect LROs
        all_lros.extend(session_ao.long_recordings)

    # Consolidate into single AnimalOrganizer
    logger.info(f"Consolidating {len(all_lros)} recordings into single AnimalOrganizer for {animal_id}")
    if not all_lros:
        raise ValueError(f"No recordings found for {animal_id}")

    ao = AnimalOrganizer.from_lros(
        all_lros,
        animal_id=animal_id,
        genotype=genotype,
        sex=sex,
    )
    return ao
