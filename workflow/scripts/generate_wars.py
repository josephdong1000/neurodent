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

import os
import sys
import logging
import json
import traceback
from pathlib import Path
import logging
from tqdm import tqdm

from dask.distributed import Client, LocalCluster

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants


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

    # Note: Bad animaldays are now filtered at the Snakemake level, so they won't reach this script
    animal_key = f"{animal_folder} {animal_id}"

    # Get resource allocation from SLURM environment or use defaults (30 cores, 100GB)
    available_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 30))
    available_memory_gb = int(os.environ.get("SLURM_MEM_PER_NODE", 102400)) // 1000  # Convert MB to GB

    # Set up local Dask cluster
    n_workers = max(1, available_cores - 2)
    memory_per_worker = f"{int(available_memory_gb * 0.9 / n_workers)}GB"

    try:
        with (
            LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
                processes=True,
                memory_limit=memory_per_worker,
                silence_logs=logging.WARN,
            ) as cluster,
            Client(cluster) as client,
        ):
            logging.info(f"\n\nLocal Dask cluster dashboard: {cluster.dashboard_link}")
            logging.info(f"Workers: {n_workers}, Memory per worker: {memory_per_worker}")
            logging.info(f"Total resources: {available_cores} cores, {available_memory_gb}GB memory\n")

            logging.info(f"Processing {animal_folder} - {animal_id}")
            logging.info(f"Using {len(client.scheduler_info()['workers'])} Dask workers")

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

            # Apply bad channel filtering if defined
            bad_channels = samples_config.get("bad_channels", {})
            logging.info(f"Looking for bad channels for: '{animal_key}'")

            # First try exact match
            if animal_key in bad_channels:
                logging.info(f"Found exact match - filtering bad channels: {bad_channels[animal_key]}")
                war = war.filter_reject_channels_by_session(bad_channels[animal_key])
            else:
                # Try alternative key constructions if exact match fails
                logging.info(f"No exact match found. Available keys: {list(bad_channels.keys())[:5]}...")

                # Alternative approach: search for keys that end with the animal_id
                matching_keys = [key for key in bad_channels.keys() if key.endswith(f" {animal_id}")]
                if matching_keys:
                    # Use the first matching key
                    matching_key = matching_keys[0]
                    logging.info(
                        f"Found alternative match '{matching_key}' - filtering bad channels: {bad_channels[matching_key]}"
                    )
                    war = war.filter_reject_channels_by_session(bad_channels[matching_key])
                else:
                    logging.info(
                        f"No bad channels defined for {animal_key} (tried exact match and ending with ' {animal_id}')"
                    )

            return war

    finally:
        cluster.close()


def main():
    """Main execution function"""

    with open(snakemake.log, "w") as f:
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

            # Save metadata for downstream rules
            metadata = {
                "animal_folder": animal_folder,
                "animal_id": animal_id,
                "animal_key": f"{animal_folder} {animal_id}",
                "original_combined_name": f"{animal_folder} {animal_id}",
                "slugified_name": snakemake.wildcards.animal,
            }
            with open(snakemake.output.metadata, "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)

            logging.info(f"Successfully saved WAR and metadata for {animal_folder} {animal_id}")

        except Exception as e:
            # Try to log to logger if available, otherwise print to stdout
            error_msg = f"Error in WAR generation: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logging.error(error_msg)
            raise


if __name__ == "__main__":
    main()
