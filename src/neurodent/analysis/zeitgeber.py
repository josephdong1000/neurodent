"""
Zeitgeber Time (ZT) Analysis Module
===================================
This module provides functionality for processing and analyzing data in Zeitgeber Time (ZT).
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from .. import visualization

logger = logging.getLogger(__name__)

def load_war_for_zeitgeber(war_path_info):
    """
    Load a fragment-filtered WAR and extract channel-averaged features for zeitgeber analysis.

    Args:
        war_path_info (tuple): Tuple containing:
            - war_pkl_path (Path): Path to the WAR pickle file.
            - war_json_path (Path): Path to the WAR JSON metadata file.
            - features_to_extract (list[str]): List of features to extract.
            - animal_name (str): Identifier for the animal.

    Returns:
        pd.DataFrame: DataFrame containing channel-averaged features and animal identifier.
                      The DataFrame will have columns for each extracted feature and an 'animal' column.
    """
    war_pkl_path, war_json_path, features_to_extract, animal_name = war_path_info

    try:
        logger.info(f"Loading {animal_name}")
        
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=war_pkl_path.parent, pickle_name=war_pkl_path.name, json_name=war_json_path.name
        )

        df = war.get_channel_averaged_result(features=features_to_extract)
        df["animal"] = animal_name
        del war
        return df

    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        raise

def convert_to_zeitgeber_time(df):
    """
    Convert timestamps to zeitgeber time representation.

    Args:
        df (pd.DataFrame): DataFrame containing a 'timestamp' column.
                           - 'timestamp': datetime64[ns] series representing the absolute time.

    Returns:
        pd.DataFrame: DataFrame with added time columns:
                      - 'hour' (int): Hour of the day (0-23).
                      - 'minute' (int): Minute of the hour (0-59).
                      - 'total_minutes' (int): Time of day in minutes from midnight (0-1440).
    """
    if df is None or df.empty:
        return df
        
    logger.info("Converting to zeitgeber time")

    df["hour"] = df["timestamp"].dt.hour.copy()
    df["minute"] = df["timestamp"].dt.minute.copy()
    
    # Total minutes (0-1440)
    df["total_minutes"] = 60 * (round((df["hour"] * 60 + df["minute"]) / 60) % 24)

    return df

def baseline_correct_features(df, baseline_hours=12, exclude_from_baseline=None):
    """
    Apply baseline correction to numeric features in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing feature columns and 'total_minutes'.
                           Required format:
                           - 'total_minutes' (numeric): Time of day in minutes.
                           - 'animal', 'genotype', 'sex' (optional): Used for grouping if present.
                           - Feature columns (numeric): Columns to be baseline corrected.
        baseline_hours (int, optional): Number of hours from start of day (ZT0) to use as baseline. Defaults to 12.
        exclude_from_baseline (list[str], optional): List of column names to exclude from baseline correction. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with new columns suffixed with '_nobase', containing baseline-corrected values.
                      Formula: value - mean(value where total_minutes <= baseline_hours * 60)
    """
    if exclude_from_baseline is None:
        exclude_from_baseline = []

    metadata_cols = ["timestamp", "animal", "genotype", "hour", "minute", "total_minutes", "sex", "gene"]
    
    features_to_baseline = [
        col for col in df.columns
        if col not in metadata_cols
        and col not in exclude_from_baseline
        and not col.endswith("_nobase")
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    logger.info(f"Baseline-correcting {len(features_to_baseline)} features using first {baseline_hours} hours")

    result_df = df.copy()
    group_cols = [c for c in ["animal", "gene", "sex"] if c in df.columns]

    for feature in features_to_baseline:
        if group_cols:
            def calc_baseline(g):
                # Calculate baseline from data <= baseline_hours
                baseline_data = g.loc[g["total_minutes"] <= baseline_hours * 60, feature]
                if len(baseline_data) > 0:
                    return g[feature] - baseline_data.mean()
                else:
                    return g[feature] * np.nan 
            
            result_df[f"{feature}_nobase"] = result_df.groupby(group_cols).apply(calc_baseline).reset_index(level=list(range(len(group_cols))), drop=True)
        else:
            baseline_data = df.loc[df["total_minutes"] <= baseline_hours * 60, feature]
            if len(baseline_data) > 0:
                 baseline_mean = baseline_data.mean()
                 result_df[f"{feature}_nobase"] = df[feature] - baseline_mean
            else:
                 result_df[f"{feature}_nobase"] = np.nan
        
        logger.debug(f"Created baseline-corrected version for {feature}")
        
    return result_df

def prepare_plot_data(df, shift_for_48h=True, perform_zt_shift=False):
    """
    Prepare data for plotting: enrichment, sorting, and optional 48h view duplication.

    Args:
        df (pd.DataFrame): Input dataframe.
                           Expected format:
                           - 'genotype' (str, optional): Strain info (e.g., 'M_WT', 'F_Mut').
                               Used to derive 'sex' and 'gene' if not present.
                           - 'total_minutes' (numeric): Time of day in minutes.
        shift_for_48h (bool, optional): If True, duplicates data shifted by 24h (1440 min) to create a 48h view. Defaults to True.
        perform_zt_shift (bool, optional): If True, shifts 'total_minutes' by -6 hours (Clock -> ZT conversion). Defaults to False.

    Returns:
        pd.DataFrame: Processed dataframe ready for plotting. 
                      Adds 'sex', 'gene', and temporary sorting columns if applicable.
    """
    df = df.copy()
    
    # Metadata enrichment
    if "genotype" in df.columns and "sex" not in df.columns:
        df["sex"] = df["genotype"].str[0].map({"F": "Female", "M": "Male"})
        
    if "genotype" in df.columns and "gene" not in df.columns:
        df["gene"] = df["genotype"].str[1:]
        
    # Align to ZT standard if requested (Clock -> ZT)
    if perform_zt_shift and "total_minutes" in df.columns:
         df['total_minutes'] = (df['total_minutes'] - 6 * 60) % 1440
    
    if shift_for_48h:
        df2 = df.copy()
        df2['total_minutes'] = df2['total_minutes'] + 1440
        df = pd.concat([df, df2], ignore_index=True)
    
    # Sorting
    sort_cols = []
    if "gene" in df.columns:
        genotype_order = {'WT': 0, 'Het': 1, 'Mut': 2}
        df['genotype_order'] = df['gene'].map(genotype_order).fillna(99)
        sort_cols.append('genotype_order')
        
    if "sex" in df.columns:
        sex_order = {'Male': 0, 'Female': 1}
        df['sex_order'] = df['sex'].map(sex_order).fillna(99)
        # Sort by sex first
        sort_cols.insert(0, 'sex_order')
    
    if sort_cols:
        df = df.sort_values(sort_cols).drop(columns=sort_cols)
        
    return df

def process_zeitgeber_data(df, config=None, baseline_hours=12, exclude_from_baseline=None):
    """
    Main orchestration function for processing zeitgeber data.

    Performs the following steps:
    1. Enrichment of metadata (sex, gene) from genotype.
    2. Conversion to ZT (shifting total_minutes by -6h).
    3. Baseline correction of numeric features.
    4. Preparation for plotting (sorting, 48h duplication).

    Args:
        df (pd.DataFrame): Input zeitgeber features dataframe.
                           Required columns:
                           - 'timestamp' (datetime64): For initial ZT conversion if needed.
                           - 'total_minutes' (int/float): For processing.
                           - 'genotype' (str, optional): e.g. 'M_WT', 'F_Mut'.
        config (dict, optional): Configuration dictionary. Can override baseline settings. Defaults to None.
        baseline_hours (int, optional): Hours to use for baseline correction. Defaults to 12.
        exclude_from_baseline (list[str], optional): Columns to exclude from baseline correction. Defaults to None.

    Returns:
        pd.DataFrame: Processed dataframe ready for plotting.
    """
    logger.info("Processing zeitgeber data for temporal analysis")
    
    if config and "analysis" in config and "zeitgeber_plots" in config["analysis"]:
        zt_config = config["analysis"]["zeitgeber_plots"]
        baseline_hours = zt_config.get("baseline_hours", baseline_hours)
        exclude_from_baseline = zt_config.get("exclude_from_baseline", exclude_from_baseline)
        
    df_processed = df.copy()
    
    # 1. Enrich (Sex/Gene) for proper grouping in baseline correction
    if "genotype" in df_processed.columns:
        if "sex" not in df_processed.columns:
             df_processed["sex"] = df_processed["genotype"].str[0].map({"F": "Female", "M": "Male"})
        if "gene" not in df_processed.columns:
             df_processed["gene"] = df_processed["genotype"].str[1:]

    # 2. Shift to ZT for baseline logic (assuming baseline is first 12h of ZT)
    if "total_minutes" in df_processed.columns:
         df_processed['total_minutes'] = (df_processed['total_minutes'] - 6 * 60) % 1440
         
    # 3. Baseline Correction
    df_processed = baseline_correct_features(
        df_processed, 
        baseline_hours=baseline_hours, 
        exclude_from_baseline=exclude_from_baseline
    )
    
    # 4. Prepare for Plotting (48h expansion + sorting)
    # Note: we already shifted to ZT, so perform_zt_shift=False
    df_final = prepare_plot_data(df_processed, shift_for_48h=True, perform_zt_shift=False)
    
    if "animal" in df_final.columns:
        logger.info(f"Processed data for {df_final['animal'].nunique()} unique animals")
        
    return df_final
