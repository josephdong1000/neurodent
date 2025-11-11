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

from neurodent import visualization

logger = logging.getLogger(__name__)


def load_war_for_zeitgeber(war_path_info):
    """
    Load a fragment-filtered WAR and extract channel-averaged features for zeitgeber analysis

    Args:
        war_path_info: Tuple of (war_pkl_path, war_json_path, features_to_extract, animal_name)

    Returns:
        pd.DataFrame: Processed dataframe with channel-averaged zeitgeber features, or None if failed
    """
    war_pkl_path, war_json_path, features_to_extract, animal_name = war_path_info

    try:
        logger.info(f"Loading {animal_name}")

        # Load fragment-filtered WAR using explicit PKL and JSON paths
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=war_pkl_path.parent, pickle_name=war_pkl_path.name, json_name=war_json_path.name
        )

        # Channel standardization already done in fragment filtering step

        # Extract features for zeitgeber analysis WITH CHANNEL AVERAGING
        # This single method call replaces all the band extraction and averaging logic
        df = war.get_channel_averaged_result(features=features_to_extract)
        df["animal"] = animal_name

        # Clean up memory
        del war

        return df

    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        raise


def convert_to_zeitgeber_time(df):
    """
    Convert timestamps to zeitgeber time representation
    (Implementation from pipeline-alphadelta.py)
    """
    logger.info("Converting to zeitgeber time")

    # Extract hour and minute from timestamp
    df["hour"] = df["timestamp"].dt.hour.copy()
    df["minute"] = df["timestamp"].dt.minute.copy()

    # Create total_minutes representation (rounded to nearest hour)
    df["total_minutes"] = 60 * (round((df["hour"] * 60 + df["minute"]) / 60) % 24)

    return df


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
            for df in tqdm(
                pool.imap(load_war_for_zeitgeber, war_infos),
                total=len(war_infos),
                desc="Loading WARs for zeitgeber analysis",
            ):
                if df is not None:
                    dfs.append(df)
    else:
        # Single-threaded processing
        for war_info in tqdm(war_infos, desc="Loading WARs for zeitgeber analysis"):
            df = load_war_for_zeitgeber(war_info)
            if df is not None:
                dfs.append(df)

    if not dfs:
        logger.error("No valid WARs were processed!")
        raise RuntimeError("No valid WARs found for zeitgeber analysis")

    logger.info(f"Successfully processed {len(dfs)} WARs")

    # Concatenate all dataframes (already channel-averaged by get_channel_averaged_result())
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataframe shape: {df.shape}")

    # Convert to zeitgeber time
    df = convert_to_zeitgeber_time(df)

    # Identify feature columns (exclude metadata)
    metadata_cols = ["timestamp", "animal", "genotype", "hour", "minute", "total_minutes"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    logger.info(f"Found {len(feature_cols)} feature columns: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    # Select final columns
    final_columns = metadata_cols + feature_cols
    df = df[final_columns]

    # Aggregate by time windows (following alphadelta pipeline)
    logger.info("Aggregating by time windows")
    agg_dict = {feature: "mean" for feature in feature_cols}
    df = df.groupby(["animal", "genotype", "total_minutes"]).agg(agg_dict).reset_index()

    logger.info(f"Final aggregated dataframe shape: {df.shape}")

    # Create output directory
    output_dir = Path(output_pkl).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to pickle
    df.to_pickle(output_pkl)
    logger.info(f"Saved zeitgeber features to: {output_pkl}")

    logger.info("Zeitgeber feature extraction completed successfully")


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
        )
        main()
