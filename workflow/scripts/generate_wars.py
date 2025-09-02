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
from pathlib import Path
from tqdm import tqdm

from dask.distributed import Client, LocalCluster

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants


def setup_logging():
    """Set up logging configuration"""
    # Log to both the Snakemake log file and stdout/stderr for SLURM
    log_file = getattr(snakemake, 'log', [None])[0] if hasattr(snakemake, 'log') else None
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", 
        level=logging.DEBUG, 
        handlers=handlers,
        force=True
    )
    return logging.getLogger()


# def setup_local_cluster(available_cores, available_memory_gb):
#     """Set up Local Dask cluster using full resources within this SLURM job"""
#     # Use almost all available cores, leaving 2 for overhead
#     n_workers = max(1, available_cores - 2)

#     # Distribute memory among workers, leaving some overhead for the main process
#     memory_per_worker = f"{int(available_memory_gb * 0.85 / n_workers)}GB"

#     cluster = LocalCluster(
#         n_workers=n_workers,
#         threads_per_worker=1,  # Single-threaded workers for better memory isolation
#         memory_limit=memory_per_worker,
#         processes=True,  # Use separate processes for better memory management
#         silence_logs=True,  # Reduce log noise
#     )

#     print(f"\n\nLocal Dask cluster dashboard: {cluster.dashboard_link}")
#     print(f"Workers: {n_workers}, Memory per worker: {memory_per_worker}")
#     print(f"Total resources: {available_cores} cores, {available_memory_gb}GB memory\n")

#     return cluster


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
    base_folder = Path(config["base_folder"])
    data_parent_folder = Path(samples_config["data_parent_folder"])

    # Set temp directory
    core.set_temp_directory(config["temp_directory"])

    # Set genotype aliases
    constants.GENOTYPE_ALIASES = samples_config["GENOTYPE_ALIASES"]

    # Check if this animal/folder combo should be skipped
    animal_key = f"{animal_folder} {animal_id}"
    bad_folder_animalday = config.get("bad_folder_animalday", [])
    if animal_key in bad_folder_animalday:
        logger.warning(f"Skipping {animal_key} because it is in bad_folder_animalday")
        return None

    # Get resource allocation from SLURM environment or use defaults (30 cores, 100GB)
    available_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 30))
    available_memory_gb = int(os.environ.get("SLURM_MEM_PER_NODE", 102400)) // 1000  # Convert MB to GB

    # Set up local Dask cluster
    n_workers = max(1, available_cores - 2)
    memory_per_worker = f"{int(available_memory_gb * 0.9 / n_workers)}GB"

    try:
        with (
            LocalCluster(
                n_workers= n_workers,
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

            logger.info(f"Processing {animal_folder} - {animal_id}")
            logger.info(f"Using {len(client.scheduler_info()['workers'])} Dask workers")

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
            logger.info(f"Computing bad channels for {animal_key}")
            ao.compute_bad_channels()

            # Generate WAR using Dask
            logger.info(f"Computing windowed analysis for {animal_key}")
            war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")

            # Apply bad channel filtering if defined
            bad_channels = samples_config.get("bad_channels", {})
            if animal_key in bad_channels:
                logger.info(f"Filtering bad channels for {animal_key}: {bad_channels[animal_key]}")
                war = war.filter_reject_channels_by_session(bad_channels[animal_key])
            else:
                logger.info(f"No bad channels defined for {animal_key}")

            return war

    finally:
        cluster.close()


def main():
    """Main execution function"""
    logger = setup_logging()

    try:
        # Load configuration
        samples_config, config, animal_folder, animal_id = load_samples_and_config()

        # Generate WAR
        war = generate_war_for_animal(samples_config, config, animal_folder, animal_id, logger)

        if war is None:
            logger.error(f"Failed to generate WAR for {animal_folder} {animal_id}")
            sys.exit(1)

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
        with open(snakemake.output.metadata, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully saved WAR and metadata for {animal_folder} {animal_id}")

    except Exception as e:
        logger.error(f"Error in WAR generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
