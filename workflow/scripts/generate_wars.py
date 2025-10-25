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

import json
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from neurodent import constants, core, visualization


def load_samples_and_config():
    """Load sample configuration and pipeline config"""
    # Get parameters from Snakemake
    samples_config = snakemake.params.samples_config
    config = snakemake.params.config
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id

    return samples_config, config, animal_folder, animal_id


def generate_war_for_animal(samples_config, config, animal_folder, animal_id):
    """Generate WAR for a specific animal"""

    # Set up paths and parameters
    base_folder = Path(config["base_folder"])
    data_parent_folder = Path(samples_config["data_parent_folder"])

    # Set temp directory
    core.set_temp_directory(config["temp_directory"])

    # Set genotype aliases
    constants.GENOTYPE_ALIASES = samples_config["GENOTYPE_ALIASES"]
    animal_key = f"{animal_folder} {animal_id}"

    try:
        with (
            LocalCluster(
                interface=config["cluster"]["war_generation"]["interface"],
            ) as cluster,
            Client(cluster) as client,
        ):
            logging.info(f"\n\nLocal Dask cluster dashboard: {cluster.dashboard_link}")
            logging.info(f"Number of workers: {len(client.scheduler_info()['workers'])}")
            for worker, info in client.scheduler_info()["workers"].items():
                print(f"Worker {worker}: {info['memory_limit']}, CPUs: {info['nthreads']}")
            print("\n")

            logging.info(f"Processing {animal_folder} - {animal_id}")

            # Create AnimalOrganizer
            analysis_config = config["analysis"]["war_generation"]
            ao = visualization.AnimalOrganizer(
                data_parent_folder / animal_folder,
                animal_id,
                mode=analysis_config["mode"],
                assume_from_number=analysis_config["assume_from_number"],
                skip_days=analysis_config["skip_days"],
                lro_kwargs=analysis_config["lro_kwargs"],
            )

            # Compute bad channels
            logging.info(f"Computing bad channels for {animal_key}")
            ao.compute_bad_channels()

            # Generate WAR using Dask
            logging.info(f"Computing windowed analysis for {animal_key}")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*fmin=.*Spectrum estimate will be unreliable.*",
                    category=RuntimeWarning,
                )
                war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")

            # Frequency-domain spike detection
            logging.info(f"Computing frequency-domain spike detection for {animal_key}")
            fdsar_config = config["analysis"]["frequency_domain_spike_detection"]
            detection_params = fdsar_config["default_params"]
            multiprocess_mode = fdsar_config.get("multiprocess_mode", "serial")

            fdsar_list = ao.compute_frequency_domain_spike_analysis(
                detection_params=detection_params, multiprocess_mode=multiprocess_mode
            )

            # Integrate spike features into WAR
            logging.info(f"Integrating spike features into WAR for {animal_key}")
            war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=True)

        return war, fdsar_list
    except Exception as e:
        logging.error(f"Failed to generate WAR for {animal_key}: {e}")
        raise
    finally:
        cluster.close()


def main():
    """Main execution function"""
    global snakemake

    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f

        # Set up logging to the redirected stdout
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.DEBUG,
            stream=sys.stdout,
            force=True,
        )

        logging.info("WAR generation script started successfully")

        # Load configuration
        samples_config, config, animal_folder, animal_id = load_samples_and_config()

        # Generate WAR with integrated spike detection
        war, fdsar_list = generate_war_for_animal(samples_config, config, animal_folder, animal_id)

        # Save WAR (now includes nspike/lognspike features)
        war.save_pickle_and_json(Path(snakemake.output.war_pkl).parent, filename="war", slugify_filename=False)
        logging.info(f"Successfully saved WAR for {animal_folder} {animal_id}")

        # Save FDSAR results - each animalday gets its own subdirectory
        fdsar_base_dir = Path(snakemake.output.fdsar_dir)
        fdsar_base_dir.mkdir(parents=True, exist_ok=True)

        for fdsar in fdsar_list:
            # Create subdirectory for this animalday
            animalday_dir = fdsar_base_dir / f"{fdsar.animal_id}-{fdsar.genotype}-{fdsar.animal_day}"
            animalday_dir.mkdir(parents=True, exist_ok=True)

            fdsar.save_fif_and_json(animalday_dir, convert_to_mne=True, slugify_filebase=False, overwrite=True)
            logging.info(f"Saved FDSAR for {fdsar.animal_id} {fdsar.animal_day} to {animalday_dir}")

        logging.info(f"Successfully saved {len(fdsar_list)} FDSAR results to {fdsar_base_dir}")


if __name__ == "__main__":
    main()
