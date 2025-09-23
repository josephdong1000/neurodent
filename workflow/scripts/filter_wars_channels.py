#!/usr/bin/env python3
"""
WAR Channel Filtering Script
============================

This script applies channel-level filtering to fragment-filtered WARs.
Supports both manual bad channel lists and LOF-based filtering.

Input: Fragment-filtered WARs (temporal artifacts already removed)
Output: Channel-filtered WARs ready for flattening
"""

import logging
import sys
from pathlib import Path

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization


def main():
    """Main channel filtering function for single animal (1-to-1 operation)"""
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
    samples_config = snakemake.params.samples_config
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id
    filter_type = snakemake.params.filter_type  # "manual" or "lof"

    # Get animal name from wildcards and construct the animal key
    animal_name = snakemake.wildcards.animal
    animal_key = f"{animal_folder} {animal_id}"

    logging.info(f"Processing animal: {animal_name}")
    logging.info(f"Animal key: {animal_key}")
    logging.info(f"Channel filter type: {filter_type}")

    try:
        # Load the fragment-filtered WAR
        logging.info(f"Loading fragment-filtered WAR from: {input_war_dir}")
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=input_war_dir, pickle_name=war_pkl_name, json_name=war_json_name
        )

        if filter_type not in ["manual", "lof"]:
            raise ValueError(f"Unknown filter_type: {filter_type}. Must be 'manual' or 'lof'")

        filter_config = {}

        # read in bad channels -- both pipelines
        channel_filter_config = config["analysis"]["channel_filter_config"][filter_type].copy()
        bad_channels = channel_filter_config.get("reject_channels", [])
        logging.info(f"{filter_type} - Reject channels: {bad_channels}")
        filter_config["reject_channels"] = {"bad_channels": bad_channels}

        if filter_type == "manual":
            # read in bad channels by session -- manual only
            reject_channels_by_session = channel_filter_config["reject_channels_by_session"]

            if reject_channels_by_session:
                samples_bad_channels = samples_config.get("bad_channels", {})
                bad_channels_dict_manual = samples_bad_channels.get(animal_key, {})
                logging.info(f"{filter_type} - Reject channels by session: {bad_channels_dict_manual}")
                filter_config["reject_channels_by_session"] = {"bad_channels_dict": bad_channels_dict_manual}

            min_valid_channels = channel_filter_config["min_valid_channels"]
            logging.info(f"{filter_type} - Minimum valid channels: {min_valid_channels}")

        elif filter_type == "lof":
            # apply lof-based channel filtering -- lof only
            lof_threshold = channel_filter_config["reject_lof_threshold"]
            logging.debug(f"LOF threshold: {lof_threshold}")
            logging.debug(f"LOF scores dict: {war.lof_scores_dict}")

            bad_channels_dict_lof = war.get_bad_channels_by_lof_threshold(lof_threshold)
            logging.info(f"{filter_type} - Reject channels by LOF threshold: {bad_channels_dict_lof}")
            filter_config["reject_channels_by_session"] = {"bad_channels_dict": bad_channels_dict_lof}

            min_valid_channels = channel_filter_config["min_valid_channels"]
            logging.info(f"{filter_type} - Minimum valid channels: {min_valid_channels}")

        # Apply filters
        war = war.apply_filters(
            filter_config=filter_config,
            min_valid_channels=min_valid_channels,
        )
        logging.info(f"{filter_type} - Applied channel filtering")

        # Create output directory
        output_dir = Path(output_war_pkl).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save channel-filtered WAR as both pickle and json
        war.save_pickle_and_json(output_dir)

        logging.info(f"Successfully channel-filtered ({filter_type}) and saved {animal_name}")

    except Exception as e:
        logging.error(f"Failed to process {animal_name}: {str(e)}")
        raise

    logging.info(f"WAR channel filtering ({filter_type}) script completed successfully")


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
        )
        main()
