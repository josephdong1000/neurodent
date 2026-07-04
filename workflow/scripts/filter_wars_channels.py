#!/usr/bin/env python3
"""
WAR Channel Filtering Script
============================

This script applies channel-level filtering to fragment-filtered WARs.
Supports both manual bad channel lists and LOF-based filtering.

Input: Fragment-filtered WARs (temporal artifacts already removed)
Output: Channel-filtered WARs ready for flattening
"""

from pathlib import Path

from neurodent.results import WindowAnalysisResult
from neurodent.workflow import setup_snakemake_logging, apply_samples_config


def main():
    """Main channel filtering function for single animal (1-to-1 operation)"""
    global snakemake

    logger = setup_snakemake_logging(snakemake)

    # Get parameters from snakemake
    logger.debug(f"snakemake.input.war_parquet: {snakemake.input.war_parquet}")
    logger.debug(f"snakemake.input.war_json: {snakemake.input.war_json}")

    # Handle both string and list inputs
    war_parquet_path = (
        snakemake.input.war_parquet[0]
        if isinstance(snakemake.input.war_parquet, list)
        else snakemake.input.war_parquet
    )
    war_json_path = (
        snakemake.input.war_json[0] if isinstance(snakemake.input.war_json, list) else snakemake.input.war_json
    )
    input_war_dir = Path(war_parquet_path).parent
    war_parquet_name = Path(war_parquet_path).name
    war_json_name = Path(war_json_path).name

    output_war_parquet = snakemake.output.war_parquet
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id
    filter_type = snakemake.params.filter_type  # "manual" or "lof"

    # Inject aliases
    apply_samples_config(samples_config)

    # Get animal name from wildcards and construct the animal key
    animal_name = snakemake.wildcards.animal
    animal_key = f"{animal_folder} {animal_id}"

    logger.info(f"Processing animal: {animal_name}")
    logger.info(f"Animal key: {animal_key}")
    logger.info(f"Channel filter type: {filter_type}")

    try:
        if filter_type not in ["manual", "lof"]:
            raise ValueError(f"Unknown filter_type: {filter_type}. Must be 'manual' or 'lof'")

        src_filename = Path(war_parquet_name).stem
        dst_filename = Path(output_war_parquet).stem
        logger.info(
            f"Stream-filtering channels ({filter_type}): {input_war_dir} -> {Path(output_war_parquet).parent}"
        )
        war = WindowAnalysisResult.scan_parquet_and_json(
            input_war_dir, filename=src_filename
        )

        filter_config = {}

        # read in bad channels -- both pipelines
        channel_filter_config = config["analysis"]["channel_filter_config"][filter_type].copy()
        bad_channels = channel_filter_config.get("reject_channels", [])
        logger.info(f"{filter_type} - Reject channels: {bad_channels}")
        filter_config["reject_channels"] = {"bad_channels": bad_channels}

        if filter_type == "manual":
            # Per-session bad channels declared in the dataset config are ALWAYS applied
            # (empty dict = no-op); there is no on/off toggle. The global reject_channels
            # read above applies on top of these.
            samples_bad_channels = samples_config.get("bad_channels", {})
            bad_channels_dict_manual = samples_bad_channels.get(animal_id, {})
            # Expand "_all" key: merge into every other session entry
            if "_all" in bad_channels_dict_manual:
                all_bad = bad_channels_dict_manual.pop("_all")
                if bad_channels_dict_manual:
                    for session_key in bad_channels_dict_manual:
                        merged = list(dict.fromkeys(
                            bad_channels_dict_manual[session_key] + all_bad
                        ))
                        bad_channels_dict_manual[session_key] = merged
                else:
                    existing = filter_config["reject_channels"].get("bad_channels", [])
                    filter_config["reject_channels"]["bad_channels"] = list(
                        dict.fromkeys(existing + all_bad)
                    )
            logger.info(f"{filter_type} - Reject channels by session: {bad_channels_dict_manual}")
            filter_config["reject_channels_by_session"] = {"bad_channels_dict": bad_channels_dict_manual}

            min_valid_channels = channel_filter_config["min_valid_channels"]
            logger.info(f"{filter_type} - Minimum valid channels: {min_valid_channels}")

        elif filter_type == "lof":
            lof_threshold = channel_filter_config["reject_lof_threshold"]
            logger.debug(f"LOF threshold: {lof_threshold}")
            logger.debug(f"LOF scores dict: {war.lof_scores_dict}")
            bad_channels_dict_lof = war.get_bad_channels_by_lof_threshold(lof_threshold)
            logger.info(f"{filter_type} - Reject channels by LOF threshold: {bad_channels_dict_lof}")
            filter_config["reject_channels_by_session"] = {"bad_channels_dict": bad_channels_dict_lof}

            min_valid_channels = channel_filter_config["min_valid_channels"]
            logger.info(f"{filter_type} - Minimum valid channels: {min_valid_channels}")

        war.apply_filters(
            filter_config=filter_config,
            min_valid_channels=min_valid_channels,
        )
        war.save_parquet_and_json(
            Path(output_war_parquet).parent, filename=dst_filename
        )

        logger.info(f"Successfully channel-filtered ({filter_type}) and saved {animal_name}")

    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        raise

    logger.info(f"WAR channel filtering ({filter_type}) script completed successfully")


if __name__ == "__main__":
    main()
