#!/usr/bin/env python3
"""
Zeitgeber Feature Extraction Script
===================================

This script implements the pipeline-alphadelta.py functionality in the Snakemake workflow.
It processes fragment-filtered WARs to extract features with zeitgeber time information
and creates a single concatenated dataframe across all animals.

Based on: notebooks/examples/pipeline-alphadelta.py
"""

import logging
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

# Import the new zeitgeber module
from neurodent.core import get_expanded_feature_names
from neurodent.core.zeitgeber import _load_war_for_zeitgeber
from neurodent.workflow import setup_snakemake_logging, inject_config_aliases

logger = logging.getLogger(__name__)


def main():
    """Main zeitgeber feature extraction function"""
    global snakemake

    setup_snakemake_logging(snakemake)

    # Get parameters from snakemake
    input_war_parquets = snakemake.input.war_parquet
    input_war_jsons = snakemake.input.war_json
    output_pkl = snakemake.output.zeitgeber_features
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    inject_config_aliases(samples_config)

    # Get zeitgeber processing parameters from config
    zeitgeber_params = config["analysis"]["zeitgeber"]
    features_to_extract = zeitgeber_params["features"]
    threads = snakemake.threads

    logger.info(f"Processing {len(input_war_parquets)} fragment-filtered WARs")
    logger.info(f"Features to extract: {features_to_extract}")
    logger.info(f"Using {threads} threads")

    # Validate that parquet and JSON inputs match
    if len(input_war_parquets) != len(input_war_jsons):
        raise ValueError(
            f"Mismatch between parquet files ({len(input_war_parquets)}) and JSON files ({len(input_war_jsons)})"
        )

    # Create animal name to file path mappings
    parquet_animals = {}
    json_animals = {}

    for parquet_path in input_war_parquets:
        parquet_path_obj = Path(parquet_path)
        animal_name = parquet_path_obj.parent.name
        parquet_animals[animal_name] = parquet_path_obj

    for json_path in input_war_jsons:
        json_path_obj = Path(json_path)
        animal_name = json_path_obj.parent.name
        json_animals[animal_name] = json_path_obj

    # Validate that all animals have both parquet and JSON files
    parquet_animal_set = set(parquet_animals.keys())
    json_animal_set = set(json_animals.keys())

    if parquet_animal_set != json_animal_set:
        missing_json = parquet_animal_set - json_animal_set
        missing_parquet = json_animal_set - parquet_animal_set
        error_msg = []
        if missing_json:
            error_msg.append(f"Animals missing JSON files: {missing_json}")
        if missing_parquet:
            error_msg.append(f"Animals missing parquet files: {missing_parquet}")
        raise ValueError("; ".join(error_msg))

    logger.info(f"Validated {len(parquet_animal_set)} animals have both parquet and JSON files")

    # Prepare war information for processing
    war_infos = []
    # Configure pipeline for extraction:
    # - shift_for_48h=False: We want canonical features here, not plotting data
    # - interval_minutes: Ensure it matches what we want for aggregation
    pipeline_config = zeitgeber_params.copy()
    pipeline_config["shift_for_48h"] = False
    
    # Use ANIMAL_METADATA (injected by inject_config_aliases, required)
    from neurodent import constants
    pipeline_config["animal_metadata"] = constants.ANIMAL_METADATA
    
    for animal_name in sorted(parquet_animal_set):
        parquet_path = parquet_animals[animal_name]
        json_path = json_animals[animal_name]
        war_infos.append((parquet_path, json_path, features_to_extract, animal_name, pipeline_config))

    # Process WARs to extract features (parallel processing)
    dfs = []
    if threads > 1:
        with Pool(threads) as pool:
            # Use the new module function
            for df in tqdm(
                pool.imap(_load_war_for_zeitgeber, war_infos),
                total=len(war_infos),
                desc="Loading WARs for zeitgeber analysis",
            ):
                if df is not None:
                    dfs.append(df)
    else:
        # Single-threaded processing
        for war_info in tqdm(war_infos, desc="Loading WARs for zeitgeber analysis"):
            df = _load_war_for_zeitgeber(war_info)
            if df is not None:
                dfs.append(df)

    if not dfs:
        logger.error("No valid WARs were processed!")
        raise RuntimeError("No valid WARs found for zeitgeber analysis")

    logger.info(f"Successfully processed {len(dfs)} WARs")

    # Concatenate all dataframes (already channel-averaged by get_channel_averaged_result())
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataframe shape: {df.shape}")

    
    # Expand features (e.g. logpsdband -> logpsdband_delta, etc.)
    expanded_features = get_expanded_feature_names(features_to_extract)
    
    # Also include baseline-subtracted versions
    expanded_features_nobase = [f"{f}_nobase" for f in expanded_features]
    
    # Combine and intersect with actual dataframe columns
    expected_features = set(expanded_features + expanded_features_nobase)
    feature_cols = [col for col in df.columns if col in expected_features]
    
    # Identify metadata columns as everything else
    metadata_cols = [col for col in df.columns if col not in feature_cols]

    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Metadata columns ({len(metadata_cols)}): {metadata_cols}")
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Log column dtypes to catch any non-numeric issues
    logger.info("Column dtypes before aggregation:")
    for col in feature_cols:
        if col in df.columns:
            logger.info(f"  {col}: {df[col].dtype}")

    # Select final columns (only keep those that exist in dataframe)
    final_columns = [col for col in metadata_cols + feature_cols if col in df.columns]
    df = df[final_columns]

    # Aggregate by time windows (following alphadelta pipeline)
    logger.info("Aggregating by time windows")
    
    # Only aggregate numeric feature columns (double check)
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[int, float]).columns.tolist()
    agg_dict = {feature: "mean" for feature in numeric_feature_cols}

    try:
        # Group by all metadata columns present (except those that vary per row like timestamp/minute if we are binning)
        # We want to group by: animal, genotype, sex, gene, zt_minutes
        # And potentially other static metadata.
        # However, 'zt_minutes' is the binning key.

        # Define grouping columns based on what's available and what should be grouped
        # We want to keep animal-level metadata and the time bin.
        potential_group_cols = ["animal", "genotype", "sex", "gene", "zt_minutes", "daynight"]
        group_cols = [c for c in potential_group_cols if c in df.columns]
        
        df = df.groupby(group_cols).agg(agg_dict).reset_index()
        logger.info(f"✓ Aggregation successful! Aggregated dataframe shape: {df.shape}")
    except Exception as e:
        logger.error(f"✗ Aggregation failed with error: {str(e)}")
        logger.error(f"Feature columns being aggregated: {numeric_feature_cols}")
        logger.error("Sample values from first row:")
        for col in numeric_feature_cols:
            if col in df.columns:
                logger.error(f"  {col}: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")
        raise

    # Create output directory
    output_dir = Path(output_pkl).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to pickle
    df.to_pickle(output_pkl)
    logger.info(f"Saved zeitgeber features to: {output_pkl}")

    logger.info("Zeitgeber feature extraction completed successfully")


if __name__ == "__main__":
    main()
