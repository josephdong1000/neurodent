#!/usr/bin/env python3
"""
WAR Standardization Script
==========================

This script handles the standardization steps for quality-filtered WARs:
- Channel reordering and padding
- Unique hash addition

This is separated from fragment filtering to enable modular pipeline organization.

Input: Quality-filtered WARs (genotype/bad animal filtering already applied)
Output: Standardized WARs ready for fragment filtering
"""

import logging
import sys
from pathlib import Path

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization


def main():
    """Main standardization function for single animal (1-to-1 operation)"""
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

    # Get standardization parameters from config
    standardization_params = config["analysis"]["standardization"]

    # Get channel reordering parameters
    channel_reorder = standardization_params.get("channel_reorder")
    use_abbrevs = standardization_params.get("use_abbrevs", True)

    # Get unique hash parameters
    add_unique_hash = standardization_params.get("add_unique_hash", False)
    unique_hash_length = standardization_params.get("unique_hash_length", 4)

    logging.info(f"Processing animal: {animal_name}")
    logging.info(f"Animal key: {animal_key}")
    logging.info(f"Channel reorder: {channel_reorder}")
    logging.info(f"Use abbreviations: {use_abbrevs}")
    logging.info(f"Add unique hash: {add_unique_hash}")
    if add_unique_hash:
        logging.info(f"Unique hash length: {unique_hash_length}")

    try:
        # Load the quality-filtered WAR
        logging.info(f"Loading WAR from: {input_war_dir}")
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=input_war_dir, pickle_name=war_pkl_name, json_name=war_json_name
        )

        # Apply channel standardization for all downstream steps
        logging.info("Applying channel reordering and padding")
        war.reorder_and_pad_channels(channel_reorder, use_abbrevs=use_abbrevs)

        # Add unique hash if requested
        if add_unique_hash:
            logging.info(f"Adding unique hash with length {unique_hash_length}")
            war.add_unique_hash(unique_hash_length)

        # Create output directory
        output_dir = Path(output_war_pkl).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessed WAR as both pickle and json
        war.save_pickle_and_json(output_dir)

        logging.info(f"Successfully standardized and saved {animal_name}")

    except Exception as e:
        logging.error(f"Failed to standardize {animal_name}: {str(e)}")
        raise

    logging.info("WAR standardization script completed successfully")


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
        )
        main()
