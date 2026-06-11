#!/usr/bin/env python3
"""
WAR Fragment Filtering Script
============================

This script applies fragment-level filtering to standardized WARs.
Only applies fragment filters (temporal artifact removal), not channel filtering.

Input: Standardized WARs (channel reordering/padding already applied)
Output: Fragment-filtered WARs (ready for channel filtering)
"""

from pathlib import Path

from neurodent import visualization
from neurodent.workflow import setup_snakemake_logging, inject_config_aliases


def main():
    """Main fragment filtering function for single animal (1-to-1 operation)"""
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

    # Inject aliases
    inject_config_aliases(samples_config)
    animal_id = snakemake.params.animal_id

    # Get animal name from wildcards and construct the animal key
    animal_name = snakemake.wildcards.animal
    animal_key = f"{animal_folder} {animal_id}"

    logger.info(f"Processing animal: {animal_name}")
    logger.info(f"Animal key: {animal_key}")

    try:
        fragment_filter_config = config["analysis"]["fragment_filter_config"].copy()
        logger.info(f"Fragment filter configuration: {fragment_filter_config}")

        src_filename = Path(war_parquet_name).stem
        dst_filename = Path(output_war_parquet).stem
        logger.info(
            f"Stream-filtering fragments: {input_war_dir} -> {Path(output_war_parquet).parent}"
        )
        war = visualization.WindowAnalysisResult.scan_parquet_and_json(
            input_war_dir, filename=src_filename
        )
        war.apply_filters(filter_config=fragment_filter_config, min_valid_channels=3)
        war.save_parquet_and_json(
            Path(output_war_parquet).parent, filename=dst_filename
        )

        logger.info(f"Successfully filtered and saved {animal_name}")

    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        raise

    logger.info("WAR fragment filtering script completed successfully")


if __name__ == "__main__":
    main()
