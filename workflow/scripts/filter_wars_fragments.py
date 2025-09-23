#!/usr/bin/env python3
"""
WAR Fragment Filtering Script
============================

This script applies fragment-level filtering to standardized WARs.
Only applies fragment filters (temporal artifact removal), not channel filtering.

Input: Standardized WARs (channel reordering/padding already applied)
Output: Fragment-filtered WARs (ready for channel filtering)
"""

import logging
import sys
from pathlib import Path

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization


def main():
    """Main fragment filtering function for single animal (1-to-1 operation)"""
    global snakemake

    # Get parameters from snakemake
    logging.debug(f"snakemake.input.war_pkl: {snakemake.input.war_pkl}")
    logging.debug(f"snakemake.input.war_json: {snakemake.input.war_json}")

    # Handle both string and list inputs
    war_pkl_path = snakemake.input.war_pkl[0] if isinstance(snakemake.input.war_pkl, list) else snakemake.input.war_pkl
    war_json_path = (
        snakemake.input.war_json[0] if isinstance(snakemake.input.war_json, list) else snakemake.input.war_json
    )
    input_war_dir = Path(war_pkl_path).parent
    war_pkl_name = Path(war_pkl_path).name
    war_json_name = Path(war_json_path).name

    output_war_pkl = snakemake.output.war_pkl
    config = snakemake.params.config
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id

    # Get animal name from wildcards and construct the animal key
    animal_name = snakemake.wildcards.animal
    animal_key = f"{animal_folder} {animal_id}"

    logging.info(f"Processing animal: {animal_name}")
    logging.info(f"Animal key: {animal_key}")

    try:
        # Load the standardized WAR
        logging.info(f"Loading WAR from: {input_war_dir}")
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=input_war_dir, pickle_name=war_pkl_name, json_name=war_json_name
        )

        # Apply fragment filtering only (no channel filtering)
        logging.info("Applying fragment filtering only")

        # Get fragment filter configuration from config
        fragment_filter_config = config["analysis"]["fragment_filter_config"].copy()

        logging.info(f"Fragment filter configuration: {fragment_filter_config}")

        # Apply filters using configuration-based approach
        war = war.apply_filters(filter_config=fragment_filter_config, min_valid_channels=3)

        # Save fragment-filtered WAR as both pickle and json
        war.save_pickle_and_json(Path(output_war_pkl).parent)

        logging.info(f"Successfully filtered and saved {animal_name}")

    except Exception as e:
        logging.error(f"Failed to process {animal_name}: {str(e)}")
        raise

    logging.info("WAR fragment filtering script completed successfully")


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
        )
        main()
