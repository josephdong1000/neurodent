#!/usr/bin/env python
"""
WAR Generation Script
====================

Generate Windowed Analysis Results (WARs) from raw EEG data.
This script is a refactored version of the pipeline-war-* scripts,
designed to work with the Snakemake workflow.

Input: Raw EEG data files
Output: WAR parquet and JSON files
"""

import warnings
import os
from pathlib import Path

from dask.distributed import Client, LocalCluster

from neurodent import constants, core, visualization
from neurodent.workflow import setup_snakemake_logging, apply_samples_config
from neurodent.workflow.utils import apply_path_overrides, resolve_animal_pattern, get_discovery_animal_filter


def load_samples_and_config():
    """Load sample configuration and pipeline config"""
    # Get parameters from Snakemake
    samples_config = snakemake.params.samples_config
    config = snakemake.params.config
    animal_folders = snakemake.params.animal_folders # List of (folder, animal_id, session_key)
    animal_id = snakemake.params.animal_id
    channel_subset = snakemake.params.channel_subset  # None for regular, list for joint sessions

    return samples_config, config, animal_folders, animal_id, channel_subset


def generate_war_for_animal(samples_config, config, animal_folders, animal_id, channel_subset, logger):
    """Generate WAR for a specific animal, aggregating across multiple folders/sessions.
    
    Args:
        animal_folders: List of (folder_path, source_animal_id, session_key) tuples.
        channel_subset: Global channel subset for this animal if it is part of a joint session.
    """

    # Set up paths and parameters
    data_root = Path(samples_config.get("data_root", samples_config.get("data_parent_folder", "")))

    # Set temp directory
    core.set_temp_directory(config["temp_directory"])

    # Set aliases
    apply_samples_config(samples_config)
    
    # Logging key
    animal_key = f"{animal_id} (across {len(animal_folders)} folders)"

    try:
        with (
            LocalCluster(
                n_workers=snakemake.threads,
                threads_per_worker=1,
                interface=config["cluster"]["war_generation"]["interface"],
            ) as cluster,
            Client(cluster) as client,
        ):
            logger.info(f"\n\nLocal Dask cluster dashboard: {cluster.dashboard_link}")
            logger.info(f"Number of workers: {len(client.scheduler_info()['workers'])}")
            
            logger.info(f"Processing {animal_id} across {len(animal_folders)} sessions")
            
            all_lros = []
            analysis_config = config["analysis"]["war_generation"]

            # Resolve genotype from metadata (Metadata-First)
            if animal_id not in constants.ANIMAL_METADATA:
                 raise KeyError(
                     f"Animal '{animal_id}' (from {animal_folders[0][0]}) not found in ANIMAL_METADATA. "
                     "All animals in the pipeline must be defined in the metadata for reliable processing."
                 )
            
            meta = constants.ANIMAL_METADATA[animal_id]
            genotype = meta.get("gene", "Unknown")
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
                session_ao = visualization.AnimalOrganizer(
                    discovery_pattern,
                    animal_id=discovery_animal_filter,
                    skip_sessions=session_analysis_config.get("skip_sessions", session_analysis_config.get("skip_days", [])),
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

            ao = visualization.AnimalOrganizer.from_lros(
                all_lros,
                animal_id=animal_id,
                genotype=genotype,
                sex=sex,
            )

            # Compute bad channels
            logger.info(f"Computing bad channels for {animal_key}")
            lof_config = config["analysis"]["channel_filter_config"]["lof"]
            lof_threshold = lof_config.get("reject_lof_threshold")
            lof_chunk_duration_s = lof_config.get("lof_chunk_duration_s", 60)
            ao.compute_bad_channels(
                lof_threshold=lof_threshold,
                lof_chunk_duration_s=lof_chunk_duration_s,
            )

            # Generate WAR using Dask
            logger.info(f"Computing windowed analysis for {animal_key}")
            cwa_config = analysis_config.get("compute_windowed_analysis", {})
            cwa_features = cwa_config.get("features", ["all"])
            cwa_multiprocess_mode = cwa_config.get("multiprocess_mode", "dask")
            cwa_chunk_duration_s = cwa_config.get("chunk_duration_s", 3600)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*fmin=.*Spectrum estimate will be unreliable.*",
                    category=RuntimeWarning,
                )
                war = ao.compute_windowed_analysis(
                    cwa_features,
                    multiprocess_mode=cwa_multiprocess_mode,
                    chunk_duration_s=cwa_chunk_duration_s,
                )

            # Frequency-domain spike detection
            logger.info(f"Computing frequency-domain spike detection for {animal_key}")
            fdsar_config = config["analysis"]["frequency_domain_spike_detection"]
            detection_params = fdsar_config["default_params"]
            multiprocess_mode = fdsar_config.get("multiprocess_mode", "serial")
            fdsar_chunk_duration_s = fdsar_config.get("chunk_duration_s", 3600)

            fdsar_list = ao.compute_frequency_domain_spike_analysis(
                detection_params=detection_params, multiprocess_mode=multiprocess_mode,
                chunk_duration_s=fdsar_chunk_duration_s,
            )

            # Integrate spike features into WAR
            logger.info(f"Integrating spike features into WAR for {animal_key}")
            war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=True)

            # Release AnimalOrganizer to free memmap-backed recording references.
            # war is self-contained (DataFrame + metadata), fdsar_list SAs still
            # hold per-channel recording refs needed for .fif export.
            del ao, all_lros

        return war, fdsar_list
    except Exception as e:
        logger.error(f"Failed to generate WAR for {animal_key}: {e}")
        raise
    finally:
        cluster.close()


def main():
    """Main execution function"""
    global snakemake

    logger = setup_snakemake_logging(snakemake)
    logger.info("WAR generation script started successfully")

    # Load configuration
    samples_config, config, animal_folder, animal_id, channel_subset = load_samples_and_config()

    # Generate WAR with integrated spike detection
    war, fdsar_list = generate_war_for_animal(samples_config, config, animal_folder, animal_id, channel_subset, logger)

    # Save WAR (now includes nspike/lognspike features)
    war.save_parquet_and_json(Path(snakemake.output.war_parquet).parent, filename="war", slugify_filename=False)
    logger.info(f"Successfully saved WAR for {animal_folder} {animal_id}")
    del war  # Free WAR DataFrame before FDSAR saves

    # Save FDSAR results - each animalday gets its own subdirectory
    fdsar_base_dir = Path(snakemake.output.fdsar_dir)
    fdsar_base_dir.mkdir(parents=True, exist_ok=True)
    fdsar_config = config["analysis"]["frequency_domain_spike_detection"]
    fdsar_export_chunk_duration_s = fdsar_config.get("export_chunk_duration_s", 60)

    for fdsar in fdsar_list:
        # path_safe_save_stem returns slugified ``{animal_id}-{genotype}-{animal_day}``
        # so genotypes containing "/" (e.g. ``Arx(F/y); Rosa(+/wt)``) can't break
        # path construction.  See ``slugify`` for the convention.
        animalday_dir = fdsar_base_dir / fdsar.path_safe_save_stem
        animalday_dir.mkdir(parents=True, exist_ok=True)

        fdsar.save_fif_and_json(animalday_dir, convert_to_mne=True, overwrite=True, chunk_duration_s=fdsar_export_chunk_duration_s)
        logger.info(f"Saved FDSAR for {fdsar.animal_id} {fdsar.animal_day} to {animalday_dir}")
        fdsar.result_sas = None  # Release memmap-backed recording references

    logger.info(f"Successfully saved {len(fdsar_list)} FDSAR results to {fdsar_base_dir}")


if __name__ == "__main__":

    memray_enabled = os.environ.get("NEURODENT_MEMRAY")
    profile_enabled = os.environ.get("NEURODENT_PROFILE")

    # Optional memray memory profiling (context manager wraps main)
    # memray is Linux-only; gracefully disable on other platforms
    if memray_enabled:
        import sys
        if sys.platform == "linux":
            import memray
            memray_path = Path(snakemake.output.war_parquet).parent / "memray.bin"
            tracker_ctx = memray.Tracker(
                destination=memray.FileDestination(str(memray_path), overwrite=True)
            )
        else:
            from contextlib import nullcontext
            tracker_ctx = nullcontext()
            print(f"Warning: NEURODENT_MEMRAY set but memray is Linux-only (current platform: {sys.platform}). Profiling disabled.")
    else:
        from contextlib import nullcontext
        tracker_ctx = nullcontext()

    with tracker_ctx:
        if profile_enabled:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
            main()
            profiler.disable()
            prof_path = Path(snakemake.output.war_parquet).parent / "profile.prof"
            profiler.dump_stats(str(prof_path))
            print(f"Profile saved to {prof_path}")
            print(f"Analyze with: python -m snakeviz {prof_path}")
        else:
            main()

    if memray_enabled:
        print(f"Memray profile saved to {memray_path}")
        print(f"Generate flamegraph: python -m memray flamegraph {memray_path}")
