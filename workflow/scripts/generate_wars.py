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
from pathlib import Path

from dask.distributed import Client, LocalCluster

from neurodent import constants, core, visualization
from neurodent.workflow import setup_snakemake_logging, inject_config_aliases


def load_samples_and_config():
    """Load sample configuration and pipeline config"""
    # Get parameters from Snakemake
    samples_config = snakemake.params.samples_config
    config = snakemake.params.config
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id

    return samples_config, config, animal_folder, animal_id


def generate_war_for_animal(samples_config, config, animal_folder, animal_id, logger):
    """Generate WAR for a specific animal"""

    # Set up paths and parameters
    data_parent_folder = Path(samples_config["data_parent_folder"])

    # Set temp directory
    core.set_temp_directory(config["temp_directory"])

    # Set aliases
    inject_config_aliases(samples_config)
    animal_key = f"{animal_folder} {animal_id}"

    try:
        with (
            LocalCluster(
                interface=config["cluster"]["war_generation"]["interface"],
            ) as cluster,
            Client(cluster) as client,
        ):
            logger.info(f"\n\nLocal Dask cluster dashboard: {cluster.dashboard_link}")
            logger.info(f"Number of workers: {len(client.scheduler_info()['workers'])}")
            for worker, info in client.scheduler_info()["workers"].items():
                print(f"Worker {worker}: {info['memory_limit']}, CPUs: {info['nthreads']}")
            print("\n")

            logger.info(f"Processing {animal_folder} - {animal_id}")

            # Create AnimalOrganizer
            analysis_config = config["analysis"]["war_generation"]
            
            # Check if this is a split recording (saved as zarr)
            # Split recordings need mode="si" to use load_extractor for zarr
            lro_kwargs = dict(analysis_config.get("lro_kwargs", {}))
            if snakemake.params.is_split_recording:
                lro_kwargs["mode"] = "si"  # Override to read zarr via load_extractor
                logger.info(f"Detected split recording, using mode='si' for zarr loading")
            
            # Use built-in AnimalOrganizer timestamp resolution if manual_datetimes in JSON
            if "manual_datetimes" in samples_config:
                lro_kwargs["manual_datetimes"] = samples_config["manual_datetimes"]
                logger.info("Passing manual_datetimes from JSON to AnimalOrganizer")
            
            ao = visualization.AnimalOrganizer(
                data_parent_folder / animal_folder,
                animal_id,
                mode=analysis_config["mode"],
                file_pattern=analysis_config.get("file_pattern"),
                day_sep=analysis_config.get("day_sep"),
                assume_from_number=analysis_config["assume_from_number"],
                skip_days=analysis_config["skip_days"],
                lro_kwargs=lro_kwargs,
                day_parse_kwargs=analysis_config.get("day_parse_kwargs", {}),
            )

            # Compute bad channels
            logger.info(f"Computing bad channels for {animal_key}")
            ao.compute_bad_channels()

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
    samples_config, config, animal_folder, animal_id = load_samples_and_config()

    # Generate WAR with integrated spike detection
    war, fdsar_list = generate_war_for_animal(samples_config, config, animal_folder, animal_id, logger)

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
    main()
