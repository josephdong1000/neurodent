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

import logging
import sys
from pathlib import Path
import json

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization

def main():
    """Main flattening function for single animal (1-to-1 operation)"""
    global snakemake

    # Get parameters from snakemake
    input_war_dir = Path(snakemake.input.war_pkl).parent
    war_pkl_name = Path(snakemake.input.war_pkl).name
    war_json_name = Path(snakemake.input.war_json).name
    output_war_pkl = snakemake.output.war_pkl
    config = snakemake.params.config

    # Get animal name from wildcards
    animal_name = snakemake.wildcards.animal

    # Get groupby parameters from config
    groupby_params = config["analysis"]["aggregation"]["groupby"]
    logging.info(f"Processing animal: {animal_name}")
    logging.info(f"Using groupby parameters: {groupby_params}")

    try:
        # Load the filtered WAR
        logging.info(f"Loading WAR from: {input_war_dir}")
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=input_war_dir, pickle_name=war_pkl_name, json_name=war_json_name
        )

        # Aggregate time windows using configurable groupby
        war.aggregate_time_windows(groupby=groupby_params)

        # Create output directory
        output_dir = Path(output_war_pkl).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated WAR as both pickle and json
        war.save_pickle_and_json(output_dir)

        logging.info(f"Successfully aggregated and saved {animal_name}")

    except Exception as e:
        logging.error(f"Failed to process {animal_name}: {str(e)}")
        raise

    logging.info("WAR flattening script completed successfully")


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            stream=sys.stdout,
            force=True
        )
        main()
