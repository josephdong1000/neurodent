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
import sys
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

# Import the new zeitgeber module
from neurodent.analysis import zeitgeber

logger = logging.getLogger(__name__)


def main():
    """Main zeitgeber feature extraction function"""
    global snakemake

    # Get parameters from snakemake
    input_war_pkls = snakemake.input.war_pkl
    input_war_jsons = snakemake.input.war_json
    output_pkl = snakemake.output.zeitgeber_features
    config = snakemake.params.config

    # Get zeitgeber processing parameters from config
    zeitgeber_params = config["analysis"]["zeitgeber"]
    features_to_extract = zeitgeber_params["features"]
    threads = snakemake.threads

    logger.info(f"Processing {len(input_war_pkls)} fragment-filtered WARs")
    logger.info(f"Features to extract: {features_to_extract}")
    logger.info(f"Using {threads} threads")

    # Validate that PKL and JSON inputs match
    if len(input_war_pkls) != len(input_war_jsons):
        raise ValueError(f"Mismatch between PKL files ({len(input_war_pkls)}) and JSON files ({len(input_war_jsons)})")

    # Create animal name to file path mappings
    pkl_animals = {}
    json_animals = {}

    for pkl_path in input_war_pkls:
        pkl_path_obj = Path(pkl_path)
        animal_name = pkl_path_obj.parent.name
        pkl_animals[animal_name] = pkl_path_obj

    for json_path in input_war_jsons:
        json_path_obj = Path(json_path)
        animal_name = json_path_obj.parent.name
        json_animals[animal_name] = json_path_obj

    # Validate that all animals have both PKL and JSON files
    pkl_animal_set = set(pkl_animals.keys())
    json_animal_set = set(json_animals.keys())

    if pkl_animal_set != json_animal_set:
        missing_json = pkl_animal_set - json_animal_set
        missing_pkl = json_animal_set - pkl_animal_set
        error_msg = []
        if missing_json:
            error_msg.append(f"Animals missing JSON files: {missing_json}")
        if missing_pkl:
            error_msg.append(f"Animals missing PKL files: {missing_pkl}")
        raise ValueError("; ".join(error_msg))

    logger.info(f"Validated {len(pkl_animal_set)} animals have both PKL and JSON files")

    # Prepare war information for processing
    war_infos = []
    for animal_name in sorted(pkl_animal_set):
        pkl_path = pkl_animals[animal_name]
        json_path = json_animals[animal_name]
        war_infos.append((pkl_path, json_path, features_to_extract, animal_name))

    # Process WARs to extract features (parallel processing)
    dfs = []
    if threads > 1:
        with Pool(threads) as pool:
            # Use the new module function
            for df in tqdm(
                pool.imap(zeitgeber.load_war_for_zeitgeber, war_infos),
                total=len(war_infos),
                desc="Loading WARs for zeitgeber analysis",
            ):
                if df is not None:
                    dfs.append(df)
    else:
        # Single-threaded processing
        for war_info in tqdm(war_infos, desc="Loading WARs for zeitgeber analysis"):
            df = zeitgeber.load_war_for_zeitgeber(war_info)
            if df is not None:
                dfs.append(df)

    if not dfs:
        logger.error("No valid WARs were processed!")
        raise RuntimeError("No valid WARs found for zeitgeber analysis")

    logger.info(f"Successfully processed {len(dfs)} WARs")

    # Concatenate all dataframes (already channel-averaged by get_channel_averaged_result())
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataframe shape: {df.shape}")

    # Convert to zeitgeber time using new module
    df = zeitgeber.convert_to_zeitgeber_time(df)

    # Identify feature columns (exclude ALL metadata)
    metadata_cols = [
        "timestamp",
        "animal",
        "genotype",  # Base identifiers
        "animalday",
        "day",
        "duration",
        "endfile",
        "isday",  # WAR metadata columns
        "hour",
        "minute",
        "total_minutes",  # Zeitgeber time columns
    ]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

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
    agg_dict = {feature: "mean" for feature in feature_cols}

    try:
        df = df.groupby(["animal", "genotype", "total_minutes"]).agg(agg_dict).reset_index()
        logger.info(f"✓ Aggregation successful! Aggregated dataframe shape: {df.shape}")
    except Exception as e:
        logger.error(f"✗ Aggregation failed with error: {str(e)}")
        logger.error(f"Feature columns being aggregated: {feature_cols}")
        logger.error("Sample values from first row:")
        for col in feature_cols:
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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
    )
    main()
