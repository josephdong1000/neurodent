#!/usr/bin/env python3
"""
WAR Flattening Script
====================

This script aggregates time windows from filtered WARs and saves flattened results
for individual animals. Based on pipeline-epfig-so.py but adapted for Snakemake
workflow integration.

Input: Filtered WARs (filtering and channel reordering already applied)
Output: Individual aggregated WARs saved as parquet and json in wars_flattened/
"""

from pathlib import Path

from neurodent.results import WindowAnalysisResult
from neurodent.workflow import setup_snakemake_logging, apply_samples_config


def main():
    """Main flattening function for single animal (1-to-1 operation)"""
    global snakemake

    logger = setup_snakemake_logging(snakemake)

    # Get parameters from snakemake
    input_war_dir = Path(snakemake.input.war_parquet).parent
    war_parquet_name = Path(snakemake.input.war_parquet).name
    output_war_parquet = snakemake.output.war_parquet
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    apply_samples_config(samples_config)

    # Get animal name from wildcards
    animal_name = snakemake.wildcards.animal

    # Get groupby parameters from config
    groupby_params = config["analysis"]["aggregation"]["groupby"]
    logger.info(f"Processing animal: {animal_name}")
    logger.info(f"Using groupby parameters: {groupby_params}")

    try:
        src_filename = Path(war_parquet_name).stem
        dst_filename = Path(output_war_parquet).stem
        logger.info(
            f"Stream-flattening: {input_war_dir} -> {Path(output_war_parquet).parent}"
        )
        war = WindowAnalysisResult.scan_parquet_and_json(
            input_war_dir, filename=src_filename
        )
        war.aggregate_time_windows(groupby=groupby_params)
        war.save_parquet_and_json(
            Path(output_war_parquet).parent, filename=dst_filename
        )

        logger.info(f"Successfully aggregated and saved {animal_name}")

    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        raise

    logger.info("WAR flattening script completed successfully")


if __name__ == "__main__":
    main()
