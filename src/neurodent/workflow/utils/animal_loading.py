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

from .config import apply_path_overrides, resolve_source_subset
from .discovery import get_discovery_animal_filter, resolve_animal_pattern


def load_animal_recordings(
    samples_config,
    config,
    animal_folders,
    animal_id,
    channel_subset=None,
    logger=None,
    validate_only=False,
):
    """Discover, load, and consolidate one animal's recordings into an ``AnimalOrganizer``.

    Mirrors the loading half of the WAR-generation pipeline: for each source
    folder/session it resolves the discovery pattern, applies session- and
    per-animal config overrides (``datetimes_are_start``, ``manual_datetimes``,
    ``lro_kwargs``, ``pattern``), builds an :class:`~neurodent.loading.AnimalOrganizer`,
    splits it down to this animal's channels when that source's files hold several
    animals, and finally consolidates every discovered ``LongRecordingOrganizer`` into one
    ``AnimalOrganizer`` via :meth:`AnimalOrganizer.from_lros`.

    Whether a source is split is decided per source by
    :func:`~neurodent.workflow.utils.config.resolve_source_subset`: the source's own
    ``channel_subset`` if it declares one (null means it is not split), otherwise the
    animal's. The reader and file format play no part. Only a source that is split looks
    for its files under the animal's ``group`` name, because only it reads files shared with
    other animals.

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
        channel_subset (list[str] | None, optional): A cross-check only. The dataset
            config decides which channels belong to an animal, and they are read from it
            here, so ``None`` is always correct. A caller that passes a list must pass
            exactly the configured one.
        logger (logging.Logger | None, optional): Logger for progress messages; a
            module logger is used when ``None``.
        validate_only (bool, optional): Discover and check without loading anything; see
            Returns. Defaults to False.

    Returns:
        AnimalOrganizer: Consolidated organizer with ``genotype``/``sex`` set and
        ``long_recordings`` populated. When ``validate_only=True`` no data is loaded and a
        summary ``dict`` (``{"n_sessions", "n_files", "sessions"}``) is returned instead.
        That runs discovery, skip_sessions and every ``manual_datetimes`` check that needs no
        file contents, raising as the real load would. Anything that reads a file (the
        reader, units, channel names, the split, merges, time overlaps) is only checked by a
        real load. ``snakemake --dry-run`` does not call this at all; the preflight script
        ``scripts/iqsec2/validate_config.py`` does.

    Raises:
        KeyError: If ``animal_id`` is not in ``constants.ANIMAL_METADATA`` or a
            session has no ``pattern`` configured.
        ValueError: If no recordings are discovered for the animal, or if a
            ``channel_subset`` is passed that differs from the configured one.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Set up paths
    data_root = Path(samples_config.get("data_root", samples_config.get("data_parent_folder", "")))

    all_lros = []
    val_sessions, val_files = set(), 0   # validate_only accounting (no data load)
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

    # Per-animal overrides are constant across the loop, so resolve them once.
    animal_overrides = samples_config.get("_animal_overrides", {}).get(animal_id, {})

    # Which channels belong to the animal is configuration. A caller that left it out used
    # to get every channel of a multi-animal file without a word, so it is read here and a
    # caller's value is only compared against it.
    animal_subset = samples_config.get("_animal_channel_subsets", {}).get(animal_id)
    if channel_subset is not None and (
        animal_subset is None or list(channel_subset) != list(animal_subset)
    ):
        raise ValueError(
            f"{animal_id}: channel_subset {list(channel_subset)} was passed, but the dataset "
            f"config assigns {animal_subset}. The dataset config decides which channels "
            "belong to an animal; pass None, or exactly the configured list."
        )

    # An animal may declare several SOURCES, each with its own pattern, reader and
    # datetime convention. IQSEC2's dual-tree cohorts need this: the same animal was
    # recorded sequentially by two acquisition systems, so one entry draws from an Intan
    # tree whose filenames carry start times and a DataWave tree whose sidecars carry end
    # times. Each source produces its own LRO(s) and they are consolidated into a single
    # AnimalOrganizer below, so the animal stays one animal for statistics rather than
    # being split into two entries.
    source_specs = animal_overrides.get("sources")
    if source_specs:
        base_info = animal_folders[0]
        work_items = [(base_info, spec) for spec in source_specs]
        logger.info(f"{animal_id}: loading from {len(source_specs)} configured source(s)")
    else:
        work_items = [(info, None) for info in animal_folders]

    for folder_info, source_spec in work_items:
        # Unpack tuple from Snakefile
        folder_path, source_animal_id, session_key = folder_info

        logger.info(f"Loading session: {folder_path} (ID in metadata: {source_animal_id})")

        # Decided per source: a file holding several animals needs splitting whatever
        # format it is in, and one holding this animal alone does not.
        subset = resolve_source_subset(animal_subset, source_spec)
        splits_source = subset is not None

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

        # Apply per-animal overrides from unified animals config, then the per-source
        # spec on top so a source's own reader and datetimes_are_start win.
        if animal_overrides:
            logger.info(f"  -> Applying per-animal overrides: {list(animal_overrides.keys())}")
            if "lro_kwargs" in animal_overrides:
                session_lro_kwargs.update(animal_overrides["lro_kwargs"])
        if source_spec and "lro_kwargs" in source_spec:
            logger.info(f"  -> Applying per-source lro_kwargs: {list(source_spec['lro_kwargs'])}")
            session_lro_kwargs.update(source_spec["lro_kwargs"])

        # Sessions to skip: dataset-level plus any per-animal override (e.g. a setup/test stub that
        # a single animal has but isn't a real recording day). Both are fnmatch patterns on {session}.
        skip_sessions = list(session_analysis_config.get("skip_sessions", session_analysis_config.get("skip_days", [])))
        skip_sessions += list(animal_overrides.get("skip_sessions", []))
        if source_spec:
            skip_sessions += list(source_spec.get("skip_sessions", []))

        # Resolve manual_datetimes for this session. The per-animal value may be a
        # scalar (one start time), a dict (keyed per session/file), or a list (per-recording
        # order, possibly nested); AnimalOrganizer distributes it across discovered sessions.
        if source_spec and "manual_datetime" in source_spec:
            session_lro_kwargs["manual_datetimes"] = source_spec["manual_datetime"]
            logger.info(f"  -> Using per-source manual datetimes for {animal_id}")
        elif "manual_datetimes" in samples_config:
            all_manual_dts = samples_config["manual_datetimes"]
            if animal_id in all_manual_dts:
                session_lro_kwargs["manual_datetimes"] = all_manual_dts[animal_id]
                logger.info(f"  -> Using manual datetimes for {animal_id}")

        # Build absolute discovery pattern from the config's relative pattern
        # Per-animal pattern override takes precedence over session/default config
        effective_pattern = session_analysis_config.get("pattern")
        if "pattern" in animal_overrides:
            effective_pattern = animal_overrides["pattern"]
        if source_spec and "pattern" in source_spec:
            effective_pattern = source_spec["pattern"]
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
            source_animal_id, splits_source, animal_groups
        )
        if splits_source:
            logger.info(f"  -> Split to this animal's channels: {subset}")
            if source_animal_id in animal_groups:
                logger.info(f"  -> Using group '{discovery_animal_filter}' for {{animal}} placeholder in discovery")
        else:
            logger.info("  -> Not split: every channel in these files belongs to this animal")

        # Create AO for this session using pattern-based discovery
        session_ao = AnimalOrganizer(
            discovery_pattern,
            animal_id=discovery_animal_filter,
            skip_sessions=skip_sessions,
            lro_kwargs=session_lro_kwargs,
            validate_only=validate_only,
        )

        if validate_only:
            # Discovery, skip_sessions and the file-free manual_datetime checks ran above,
            # with no file read. Record what would load and move on: no split, no merge.
            groups = session_ao._animalday_folder_groups
            val_sessions.update(groups.keys())
            val_files += sum(len(v) for v in groups.values())
            continue

        if splits_source:
            # source_animal_id is the key in the splits dict
            session_ao = session_ao.split(groups={source_animal_id: subset})[source_animal_id]

        # Collect LROs
        all_lros.extend(session_ao.long_recordings)

    if validate_only:
        return {"n_sessions": len(val_sessions), "n_files": val_files,
                "sessions": sorted(val_sessions)}

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
