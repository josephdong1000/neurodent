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

from neurodent import core
from neurodent.analysis import AnimalAnalyzer
from neurodent.workflow import setup_snakemake_logging, apply_samples_config
from neurodent.workflow.utils import load_animal_recordings


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

    # Set temp directory
    core.utils.set_temp_directory(config["temp_directory"])

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
            
            ao = load_animal_recordings(
                samples_config, config, animal_folders, animal_id,
                channel_subset=channel_subset, logger=logger,
            )
            az = AnimalAnalyzer(ao)

            # Compute bad channels
            logger.info(f"Computing bad channels for {animal_key}")
            lof_config = config["analysis"]["channel_filter_config"]["lof"]
            lof_threshold = lof_config.get("reject_lof_threshold")
            lof_chunk_duration_s = lof_config.get("lof_chunk_duration_s", 60)
            az.compute_bad_channels(
                lof_threshold=lof_threshold,
                lof_chunk_duration_s=lof_chunk_duration_s,
            )

            # Generate WAR using Dask
            logger.info(f"Computing windowed analysis for {animal_key}")
            cwa_config = config["analysis"]["war_generation"].get("compute_windowed_analysis", {})
            cwa_features = cwa_config.get("features", ["all"])
            cwa_multiprocess_mode = cwa_config.get("multiprocess_mode", "dask")
            cwa_chunk_duration_s = cwa_config.get("chunk_duration_s", 3600)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*fmin=.*Spectrum estimate will be unreliable.*",
                    category=RuntimeWarning,
                )
                war = az.compute_windowed_analysis(
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

            fdsar_list = az.compute_frequency_domain_spike_analysis(
                detection_params=detection_params, multiprocess_mode=multiprocess_mode,
                chunk_duration_s=fdsar_chunk_duration_s,
            )

            # Integrate spike features into WAR
            logger.info(f"Integrating spike features into WAR for {animal_key}")
            war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=True)

            # Release AnimalOrganizer to free memmap-backed recording references.
            # war is self-contained (DataFrame + metadata), fdsar_list SAs still
            # hold per-channel recording refs needed for .fif export.
            del ao

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
