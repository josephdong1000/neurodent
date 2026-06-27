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

from pathlib import Path

from neurodent import visualization
from neurodent.workflow import setup_snakemake_logging, apply_samples_config


def main():
    """Main standardization function for single animal (1-to-1 operation)"""
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
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id
    samples_config = snakemake.params.samples_config

    # Inject aliases from config
    apply_samples_config(samples_config)

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

    logger.info(f"Processing animal: {animal_name}")
    logger.info(f"Animal key: {animal_key}")
    logger.info(f"Channel reorder: {channel_reorder}")
    logger.info(f"Use abbreviations: {use_abbrevs}")
    logger.info(f"Add unique hash: {add_unique_hash}")
    if add_unique_hash:
        logger.info(f"Unique hash length: {unique_hash_length}")

    try:
        src_filename = Path(war_parquet_name).stem
        dst_filename = Path(output_war_parquet).stem
        logger.info(
            f"Stream-standardising WAR: {input_war_dir} -> {Path(output_war_parquet).parent}"
        )
        war = visualization.WindowAnalysisResult.scan_parquet_and_json(
            input_war_dir, filename=src_filename
        )
        war.reorder_and_pad_channels(channel_reorder, use_abbrevs=use_abbrevs)
        if add_unique_hash:
            war.add_unique_hash(unique_hash_length)
        war.save_parquet_and_json(
            Path(output_war_parquet).parent, filename=dst_filename
        )

        logger.info(f"Successfully standardized and saved {animal_name}")

    except Exception as e:
        logger.error(f"Failed to standardize {animal_name}: {str(e)}")
        raise

    logger.info("WAR standardization script completed successfully")


if __name__ == "__main__":
    main()
