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
import numpy as np
import pandas as pd

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

def load_war_for_zeitgeber(war_path_info):
    """
    Load a fragment-filtered WAR and extract features for zeitgeber analysis
    
    Args:
        war_path_info: Tuple of (war_path, features_to_extract, animal_name)
        
    Returns:
        pd.DataFrame: Processed dataframe with zeitgeber features, or None if failed
    """
    war_path, features_to_extract, animal_name = war_path_info
    
    try:
        logger.info(f"Loading {animal_name}")
        
        # Load fragment-filtered WAR
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=war_path.parent, 
            pickle_name=war_path.name, 
            json_name=war_path.with_suffix('.json').name
        )
        
        # Extract features for zeitgeber analysis
        df = war.get_result(features=features_to_extract)
        df["animal"] = animal_name
        
        # Clean up memory
        del war
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        return None

def process_alphadelta_features(df):
    """
    Process logpsdband features to create alphadelta ratio
    (Implementation from pipeline-alphadelta.py)
    """
    logger.info("Processing alphadelta features")
    
    # Extract band features
    df_bands = pd.DataFrame(df["logpsdband"].tolist())
    alpha_array = np.stack(df_bands["alpha"].values)
    delta_array = np.stack(df_bands["delta"].values)
    
    # Create alphadelta ratio and individual band features
    df["alphadelta"] = (alpha_array / delta_array).tolist()
    df["delta"] = delta_array.tolist()
    df["alpha"] = alpha_array.tolist()
    
    return df

def average_features_across_channels(df, features):
    """
    Average each feature across channels
    (Implementation from pipeline-alphadelta.py)
    """
    logger.info("Averaging features across channels")
    
    for feature in features:
        if feature in df.columns:
            feature_arrays = np.stack(df[feature].values)  # Shape: (time_points, channels)
            feature_avg = np.nanmean(feature_arrays, axis=1)  # Average across channels
            df[f"{feature}"] = feature_avg
    
    return df

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
    
    # Get parameters from snakemake
    input_wars = snakemake.input.wars
    output_pkl = snakemake.output.zeitgeber_features
    config = snakemake.params.config
    
    # Get zeitgeber processing parameters from config
    zeitgeber_params = config["analysis"]["zeitgeber"]
    features_to_extract = zeitgeber_params["features"]
    threads = snakemake.threads
    
    logger.info(f"Processing {len(input_wars)} fragment-filtered WARs")
    logger.info(f"Features to extract: {features_to_extract}")
    logger.info(f"Using {threads} threads")
    
    # Prepare war information for processing
    war_infos = []
    for war_pkl_path in input_wars:
        war_path = Path(war_pkl_path)
        animal_name = war_path.parent.name
        war_infos.append((war_path, features_to_extract, animal_name))
    
    # Process WARs to extract features (parallel processing)
    dfs = []
    if threads > 1:
        with Pool(threads) as pool:
            for df in tqdm(
                pool.imap(load_war_for_zeitgeber, war_infos), 
                total=len(war_infos), 
                desc="Loading WARs for zeitgeber analysis"
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
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataframe shape: {df.shape}")
    
    # Process alphadelta features if logpsdband is in features
    if "logpsdband" in features_to_extract:
        df = process_alphadelta_features(df)
    
    # Define features to average across channels
    features_to_average = []
    if "logpsdband" in features_to_extract:
        features_to_average.extend(["alphadelta", "delta", "alpha"])
    if "logrms" in features_to_extract:
        features_to_average.append("logrms")
    if "zpcorr" in features_to_extract:
        features_to_average.append("zpcorr")
    
    # Average features across channels
    df = average_features_across_channels(df, features_to_average)
    
    # Convert to zeitgeber time
    df = convert_to_zeitgeber_time(df)
    
    # Select final columns (following alphadelta pipeline)
    final_columns = ["timestamp", "animal", "genotype", "hour", "minute", "total_minutes"]
    final_columns.extend(features_to_average)
    df = df[final_columns]
    
    # Aggregate by time windows (following alphadelta pipeline)
    logger.info("Aggregating by time windows")
    agg_dict = {feature: "mean" for feature in features_to_average}
    df = (
        df.groupby(["animal", "genotype", "total_minutes"])
        .agg(agg_dict)
        .reset_index()
    )
    
    logger.info(f"Final aggregated dataframe shape: {df.shape}")
    
    # Create output directory
    output_dir = Path(output_pkl).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to pickle
    df.to_pickle(output_pkl)
    logger.info(f"Saved zeitgeber features to: {output_pkl}")
    
    logger.info("Zeitgeber feature extraction completed successfully")

if __name__ == "__main__":
    main()