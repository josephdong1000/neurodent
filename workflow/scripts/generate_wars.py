#!/usr/bin/env python
"""
WAR Generation Script
====================

Generate Windowed Analysis Results (WARs) from raw EEG data.
This script is a refactored version of the pipeline-war-* scripts,
designed to work with the Snakemake workflow.

Input: Raw EEG data files
Output: WAR pickle and JSON files
"""

import warnings
import os
from pathlib import Path

from dask.distributed import Client, LocalCluster

from neurodent import constants, core, visualization
from neurodent.workflow import setup_snakemake_logging, inject_config_aliases
from neurodent.workflow.utils import apply_path_overrides, resolve_animal_pattern


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
    data_parent_folder = Path(samples_config["data_parent_folder"])

    # Set temp directory
    core.set_temp_directory(config["temp_directory"])

    # Set aliases
    inject_config_aliases(samples_config)
    
    # Logging key
    animal_key = f"{animal_id} (across {len(animal_folders)} folders)"

    try:
        with (
            LocalCluster(
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
            logger.info(f"Resolved genotype '{genotype}' for {animal_id} from ANIMAL_METADATA")

            # Load data from all source folders
            for folder_info in animal_folders:
                # Unpack tuple from Snakefile
                folder_path, source_animal_id, session_key = folder_info
                
                logger.info(f"Loading session: {folder_path} (ID in metadata: {source_animal_id})")
                
                # Check if this specific session is a joint session
                # We check if the session_key exists in the joint_sessions config
                is_joint = session_key in samples_config.get("joint_sessions", {})

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

                # Apply per-animal overrides from unified animals config
                animal_overrides = samples_config.get("_animal_overrides", {}).get(animal_id, {})
                if animal_overrides:
                    logger.info(f"  -> Applying per-animal overrides: {list(animal_overrides.keys())}")
                    if "lro_kwargs" in animal_overrides:
                        session_lro_kwargs.update(animal_overrides["lro_kwargs"])

                # Resolve manual_datetimes for this session
                if "manual_datetimes" in samples_config:
                    all_manual_dts = samples_config["manual_datetimes"]
                    if animal_id in all_manual_dts:
                        spec = all_manual_dts[animal_id]
                        if isinstance(spec, list):
                            # Distribute: each AO gets one timestamp from the list
                            # (one per session folder, matched by index order)
                            if len(spec) != len(animal_folders):
                                raise ValueError(
                                    f"Length of manual_datetimes list ({len(spec)}) for {animal_id} "
                                    f"does not match number of session folders ({len(animal_folders)})"
                                )
                            current_dt = spec[animal_folders.index(folder_info)]
                            session_lro_kwargs["manual_datetimes"] = current_dt
                            logger.info(f"  -> Using specific timestamp from list: {current_dt}")
                        else:
                            # Single string/scalar/dict: pass directly
                            session_lro_kwargs["manual_datetimes"] = spec
                            logger.info(f"  -> Using manual datetime: {spec}")

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
                base_path = str(data_parent_folder / folder_path)
                discovery_pattern = resolve_animal_pattern(
                    effective_pattern,
                    source_animal_id,
                    base_path,
                )
                logger.info(f"  -> Discovery pattern: {discovery_pattern}")

                # For joint sessions, don't filter by animal_id during discovery
                # since all files belong to all animals in the joint session
                ao_animal_id = None if is_joint else source_animal_id

                # Create AO for this session using pattern-based discovery
                session_ao = visualization.AnimalOrganizer(
                    discovery_pattern,
                    animal_id=ao_animal_id,
                    skip_sessions=session_analysis_config.get("skip_sessions", session_analysis_config.get("skip_days", [])),
                    assume_from_number=session_analysis_config["assume_from_number"],
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
            )

            # Compute bad channels
            logger.info(f"Computing bad channels for {animal_key}")
            lof_threshold = config["analysis"]["channel_filter_config"]["lof"].get("reject_lof_threshold")
            ao.compute_bad_channels(lof_threshold=lof_threshold)

            # Generate WAR using Dask
            logger.info(f"Computing windowed analysis for {animal_key}")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*fmin=.*Spectrum estimate will be unreliable.*",
                    category=RuntimeWarning,
                )
                war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")

            # Frequency-domain spike detection
            logger.info(f"Computing frequency-domain spike detection for {animal_key}")
            fdsar_config = config["analysis"]["frequency_domain_spike_detection"]
            detection_params = fdsar_config["default_params"]
            multiprocess_mode = fdsar_config.get("multiprocess_mode", "serial")

            fdsar_list = ao.compute_frequency_domain_spike_analysis(
                detection_params=detection_params, multiprocess_mode=multiprocess_mode
            )

            # Integrate spike features into WAR
            logger.info(f"Integrating spike features into WAR for {animal_key}")
            war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=True)

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
    war.save_pickle_and_json(Path(snakemake.output.war_pkl).parent, filename="war", slugify_filename=False)
    logger.info(f"Successfully saved WAR for {animal_folder} {animal_id}")

    # Save FDSAR results - each animalday gets its own subdirectory
    fdsar_base_dir = Path(snakemake.output.fdsar_dir)
    fdsar_base_dir.mkdir(parents=True, exist_ok=True)

    for fdsar in fdsar_list:
        # Create subdirectory for this animalday
        animalday_dir = fdsar_base_dir / f"{fdsar.animal_id}-{fdsar.genotype}-{fdsar.animal_day}"
        animalday_dir.mkdir(parents=True, exist_ok=True)

        fdsar.save_fif_and_json(animalday_dir, convert_to_mne=True, slugify_filebase=False, overwrite=True)
        logger.info(f"Saved FDSAR for {fdsar.animal_id} {fdsar.animal_day} to {animalday_dir}")

    logger.info(f"Successfully saved {len(fdsar_list)} FDSAR results to {fdsar_base_dir}")


if __name__ == "__main__":
    if os.environ.get("NEURODENT_PROFILE"):
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        prof_path = Path(snakemake.output.war_pkl).parent / "profile.prof"
        profiler.dump_stats(str(prof_path))
        print(f"Profile saved to {prof_path}")
        print("Analyze with: python -m snakeviz {prof_path}")
    else:
        main()
