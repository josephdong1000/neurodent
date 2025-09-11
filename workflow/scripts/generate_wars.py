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
from pathlib import Path

from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from pythoneeg import constants, core, visualization


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
            war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")

            # TODO move this into fragment filtering stage, not at the war generation stage
            # Apply bad channel filtering if defined
            bad_channels = samples_config.get("bad_channels", {})
            logging.info(f"Looking for bad channels for: '{animal_key}'")

            # First try exact match
            if animal_key in bad_channels:
                logging.info(f"Found exact match - filtering bad channels: {bad_channels[animal_key]}")
                war = war.filter_reject_channels_by_session(bad_channels[animal_key])
            else:
                logging.info(f"No bad channels defined for {animal_key}")

        return war

    finally:
        cluster.close()


def main():
    """Main execution function"""

    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f

        # Set up logging to the redirected stdout
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.DEBUG,
            stream=sys.stdout,
            force=True,
        )

        try:
            logging.info("WAR generation script started successfully")

            # Load configuration
            samples_config, config, animal_folder, animal_id = load_samples_and_config()

            # Generate WAR
            war = generate_war_for_animal(samples_config, config, animal_folder, animal_id)

            # Save WAR
            output_dir = Path(snakemake.output.war_pkl).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            war.save_pickle_and_json(output_dir, filename="war", slugify_filename=False)

            logging.info(f"Successfully saved WAR for {animal_folder} {animal_id}")

        except Exception as e:
            # Try to log to logger if available, otherwise print to stdout
            error_msg = f"Error in WAR generation: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logging.error(error_msg)
            raise


if __name__ == "__main__":
    main()
