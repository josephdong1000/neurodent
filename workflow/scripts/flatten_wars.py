#!/usr/bin/env python3
"""
WAR Flattening Script
====================

This script aggregates time windows from filtered WARs and saves flattened results
for individual animals. Based on pipeline-epfig-so.py but adapted for Snakemake
workflow integration.

Input: Filtered WARs (filtering and channel reordering already applied)
Output: Individual aggregated WARs saved as pickle and json in wars_flattened/
"""

from pathlib import Path

from neurodent import visualization
from neurodent.workflow import setup_snakemake_logging, inject_config_aliases


def main():
    """Main flattening function for single animal (1-to-1 operation)"""
    global snakemake

    logger = setup_snakemake_logging(snakemake)

    # Get parameters from snakemake
    input_war_dir = Path(snakemake.input.war_pkl).parent
    war_pkl_name = Path(snakemake.input.war_pkl).name
    war_json_name = Path(snakemake.input.war_json).name
    output_war_pkl = snakemake.output.war_pkl
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    inject_config_aliases(samples_config)

    # Get animal name from wildcards
    animal_name = snakemake.wildcards.animal

    # Get groupby parameters from config
    groupby_params = config["analysis"]["aggregation"]["groupby"]
    logger.info(f"Processing animal: {animal_name}")
    logger.info(f"Using groupby parameters: {groupby_params}")

    try:
        # Load the filtered WAR
        logger.info(f"Loading WAR from: {input_war_dir}")
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=input_war_dir, pickle_name=war_pkl_name, json_name=war_json_name
        )

        # Aggregate time windows using configurable groupby
        war.aggregate_time_windows(groupby=groupby_params)

        # Save aggregated WAR as both pickle and json
        war.save_pickle_and_json(Path(output_war_pkl).parent)

        logger.info(f"Successfully aggregated and saved {animal_name}")

    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        raise

    logger.info("WAR flattening script completed successfully")


if __name__ == "__main__":
    main()
