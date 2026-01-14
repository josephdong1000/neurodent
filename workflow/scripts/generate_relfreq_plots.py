#!/usr/bin/env python3
"""
Relative Frequency Plots Generation Script
==========================================

Generate relative frequency (distribution) plots from channel-filtered,
non-flattened WAR files. These plots show empirical distributions of features
across all time windows, providing much richer distributions than plots from
flattened data (n_animals × windows_per_animal datapoints vs. n_animals datapoints).

This pipeline operates on channel-filtered WARs before the flattening step,
similar to the zeitgeber feature extraction pipeline.

Input: Channel-filtered WAR pickle and JSON files from all animals
Output: Relative frequency distribution plots (histograms) and CSV data exports

Memory Optimization: Uses streaming parallel extraction - workers extract features
from WARs and return small DataFrames, avoiding loading all WARs into memory.
"""

import gc
import logging
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from neurodent.workflow import setup_snakemake_logging

from neurodent.constants import blue, green, orange, purple, red
from neurodent import visualization, constants

logger = logging.getLogger(__name__)


def log_memory_usage(logger, label=""):
    """Log current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage{' (' + label + ')' if label else ''}: {mem_mb:.1f} MB")


# Use constants for banded features - imported from neurodent.constants at top of file
# BANDED_FEATURES: Features that have per-band columns (e.g. cohere_delta, cohere_theta)
# These are constants.BAND_FEATURES + constants.BANDED_MATRIX_FEATURES


def melt_banded_feature(df, feature):
    """
    Melt banded feature columns from wide to long format.
    
    Converts columns like 'cohere_delta', 'cohere_theta', etc. into
    a single 'cohere' column with a 'band' column indicating the band.
    
    Args:
        df: DataFrame with banded feature columns (e.g. cohere_delta, cohere_theta)
        feature: Base feature name (e.g. 'cohere')
    
    Returns:
        DataFrame in long format with 'band' column and single feature column
    """
    # Identify band-specific columns using constants
    band_cols = [f"{feature}_{band}" for band in constants.BAND_NAMES if f"{feature}_{band}" in df.columns]
    
    if not band_cols:
        # No band columns found, return as-is
        return df
    
    # Identify metadata columns (everything that's not a band-specific feature column)
    metadata_cols = [col for col in df.columns if col not in band_cols]
    
    # Melt the band columns into long format
    df_melted = df.melt(
        id_vars=metadata_cols,
        value_vars=band_cols,
        var_name='band_col',
        value_name=feature
    )
    
    # Extract band name from column name (e.g. 'cohere_delta' -> 'delta')
    df_melted['band'] = df_melted['band_col'].str.replace(f"{feature}_", "", regex=False)
    df_melted = df_melted.drop(columns=['band_col'])
    
    return df_melted


def extract_feature_from_war(args):
    """
    Worker function: Load WAR, extract feature data, return small DataFrame only.
    
    WAR is garbage collected when function returns, keeping only the extracted data.
    
    Args:
        args: Tuple of (war_path_info, feature, collapse_channels)
            war_path_info: Tuple of (war_pkl_path, war_json_path, animal_name)
            feature: Feature name to extract
            collapse_channels: Whether to average across channels
    
    Returns:
        pd.DataFrame: Small DataFrame with just the extracted feature data
    """
    war_path_info, feature, collapse_channels = args
    war_pkl_path, war_json_path, animal_name = war_path_info
    
    try:
        # Load WAR
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=war_pkl_path.parent, 
            pickle_name=war_pkl_path.name, 
            json_name=war_json_path.name
        )
        
        # Extract just the feature data using channel averaging
        if collapse_channels:
            df = war.get_channel_averaged_result(features=[feature])
        else:
            df = war.get_windowed_result(features=[feature])
        
        # Add metadata columns
        df['animal'] = war.animal_id
        df['genotype'] = war.genotype
        # Note: 'animalday' column should already exist in df from get_channel_averaged_result
        # if it was present in war.result. No need to add it separately.
        
        # Ensure isday column exists
        if 'isday' not in df.columns and hasattr(war, 'result') and 'isday' in war.result.columns:
            df['isday'] = war.result['isday'].values[:len(df)]
        
        # WAR will be garbage collected after return - only small DF sent back
        return df
        
    except Exception as e:
        logger.error(f"Failed to extract {feature} from {animal_name}: {str(e)}")
        raise


def process_feature_dataframe(df):
    """Process feature dataframe by adding categorical columns.

    Args:
        df (pd.DataFrame): Input dataframe with feature data

    Returns:
        pd.DataFrame: Processed dataframe with sex and gene columns
    """
    df = df.copy()
    
    # Add categorical columns based on genotype
    df["sex"] = df["genotype"].map(
        lambda x: "Male" if x in ["MWT", "MHet", "MMut"] else "Female" if x in ["FWT", "FHet", "FMut"] else None
    )
    df["gene"] = df["genotype"].map(
        lambda x: "WT"
        if x in ["MWT", "FWT"]
        else "Het"
        if x in ["MHet", "FHet"]
        else "Mut"
        if x in ["MMut", "FMut"]
        else x
    )

    if "isday" in df.columns:
        df["isday"] = df["isday"].map(lambda x: "Day" if x else "Night")

    return df


def add_animal_weights(df):
    """
    Add weights to dataframe so each animal contributes equally.

    For use with seaborn histplot to ensure equal contribution from each animal
    regardless of sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with 'animal' column

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'weight' column
    """

    # Count samples per animal
    animal_counts = df.groupby("animal").size()

    # Calculate weight for each animal (1 / n_samples) so each animal sums to 1
    df = df.copy()
    df["weight"] = df["animal"].map(lambda a: 1.0 / animal_counts[a])

    return df


def create_relfreq_plot(df, feature, feature_label, hue, hue_order, palette, log_scale, output_path, dpi):
    """
    Create a relative frequency plot using FacetGrid and histplot with weighted data.

    Parameters
    ----------
    df : pd.DataFrame
        Weighted dataframe with 'weight' column
    feature : str
        Feature column name
    feature_label : str
        Label for x-axis
    hue : str
        Column name for hue (e.g., 'gene' or 'band')
    hue_order : list
        Order of hue categories
    palette : list
        Color palette
    log_scale : bool
        Whether to use log scale
    output_path : Path
        Output file path
    dpi : int
        DPI for output figure
    """
    logger = logging.getLogger(__name__)

    # Compute bins once across entire dataset to ensure consistency across and within plots
    bins = np.histogram_bin_edges(df[feature].dropna(), bins="auto").tolist()
    logger.info(f"\tBins: {len(bins)} bins")

    g = sns.FacetGrid(
        df,
        col="sex",
        row="isday",
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        height=4,
        aspect=1.2,
    )
    g.map_dataframe(
        sns.histplot,
        x=feature,
        weights="weight",
        bins=bins,
        stat="density",
        element="step",
        fill=True,
        alpha=0.6,
        log_scale=log_scale,
    )
    g.add_legend(title=hue.capitalize())
    g.set_axis_labels(feature_label, "Relative Frequency")
    g.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def create_relfreq_plots_from_df(df_weighted, feature, feature_label, output_dir, relfreq_config):
    """Create relative frequency plots for a specific feature from pre-extracted DataFrame"""
    
    logger = logging.getLogger(__name__)
    
    # Get format parameters from config
    figure_format = relfreq_config.get("figure_format", "png")
    dpi = relfreq_config.get("dpi", 300)
    
    # Create relative frequency distribution plots
    # Check if this is a banded feature using constants
    banded_features = set(constants.BAND_FEATURES) | set(constants.BANDED_MATRIX_FEATURES)
    if feature in banded_features:
        # For band features, create per-band plots
        for band in constants.BAND_NAMES:
            df_band = df_weighted[df_weighted["band"] == band]
            if len(df_band) == 0:
                logger.warning(f"No data for {feature} band {band}")
                continue
            create_relfreq_plot(
                df=df_band,
                feature=feature,
                feature_label=f"{feature_label} ({band})",
                hue="gene",
                hue_order=["WT", "Het", "Mut"],
                palette=["blue", "blueviolet", "red"],
                log_scale=False,
                output_path=output_dir / f"{feature}_relfreq_{band}.{figure_format}",
                dpi=dpi,
            )

        # Also create combined band comparison plot
        create_relfreq_plot(
            df=df_weighted,
            feature=feature,
            feature_label=feature_label,
            hue="band",
            hue_order=constants.BAND_NAMES,
            palette=[blue, orange, red, green, purple],
            log_scale=False,
            output_path=output_dir / f"{feature}_relfreq_byband.{figure_format}",
            dpi=dpi,
        )
    else:
        # For non-band features, single plot
        create_relfreq_plot(
            df=df_weighted,
            feature=feature,
            feature_label=feature_label,
            hue="gene",
            hue_order=["WT", "Het", "Mut"],
            palette=["blue", "blueviolet", "red"],
            log_scale=False,
            output_path=output_dir / f"{feature}_relfreq.{figure_format}",
            dpi=dpi,
        )

    logger.info(f"Successfully created plots for feature: {feature}")


def main():
    """Main relative frequency plots generation function"""
    global snakemake
    logger = setup_snakemake_logging(snakemake)
    logger.info("Relative frequency plots generation started")
    log_memory_usage(logger, "startup")

    # Get parameters from snakemake
    war_pkl_files = snakemake.input.war_pkl
    war_json_files = snakemake.input.war_json
    config = snakemake.params.config

    # Create output directories
    output_dir = Path(snakemake.output.figure_dir)
    data_dir = Path(snakemake.output.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(war_pkl_files)} channel-filtered WARs")

    # Get number of threads for parallel extraction
    threads = snakemake.threads
    logger.info(f"Using {threads} threads for parallel feature extraction")

    # Validate that PKL and JSON inputs match
    if len(war_pkl_files) != len(war_json_files):
        raise ValueError(
            f"Mismatch between PKL files ({len(war_pkl_files)}) and JSON files ({len(war_json_files)})"
        )

    # Prepare WAR information for parallel extraction
    war_infos = []
    for pkl_file, json_file in zip(war_pkl_files, war_json_files):
        pkl_path = Path(pkl_file)
        json_path = Path(json_file)
        animal_name = pkl_path.parent.name
        war_infos.append((pkl_path, json_path, animal_name))

    # Get relfreq configuration
    relfreq_config = config["analysis"]["relfreq_plots"]
    features = relfreq_config["features"]
    data_format = relfreq_config.get("data_format", "csv")

    # Feature to label mapping
    feature_to_label = {
        "pcorr": "PCC",
        "cohere": "|Coherency|",
        "imcoh": "Imaginary Coherencey",
        "zpcorr": "z(PCC)",
        "zcohere": "z(|Coherencey|)",
        "zimcoh": "z(Imaginary Coherencey)",
        "logpsdfrac": "Log Percent Power",
        "logpsdband": "Log Band Power",
        "psdband": "Band Power ($\\mu V^2$)",
        "nspike": "n_spike / t_window",
        "lognspike": "Log(n_spike / t_window)",
    }

    logger.info(f"Will process {len(features)} features: {features}")
    log_memory_usage(logger, "before feature loop")

    # Process each feature using streaming parallel extraction
    for feature in features:
        logger.info(f"=== Processing feature: {feature} ===")
        log_memory_usage(logger, f"start {feature}")
        
        feature_label = feature_to_label.get(feature, feature)
        
        # Prepare extraction arguments
        args = [(info, feature, True) for info in war_infos]  # True = collapse_channels
        
        # Parallel extraction - workers extract and return small DataFrames
        dfs = []
        if threads > 1:
            with Pool(threads) as pool:
                for df in tqdm(
                    pool.imap(extract_feature_from_war, args),
                    total=len(args),
                    desc=f"Extracting {feature}",
                ):
                    if df is not None:
                        dfs.append(df)
        else:
            # Single-threaded extraction
            for arg in tqdm(args, desc=f"Extracting {feature}"):
                df = extract_feature_from_war(arg)
                if df is not None:
                    dfs.append(df)
        
        if not dfs:
            logger.warning(f"No data extracted for feature {feature}, skipping")
            continue
        
        logger.info(f"Extracted {len(dfs)} DataFrames for {feature}")
        log_memory_usage(logger, f"after extraction {feature}")
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        del dfs  # Free memory
        gc.collect()
        
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        log_memory_usage(logger, f"after concat {feature}")
        
        # For banded features (BAND_FEATURES + BANDED_MATRIX_FEATURES), melt from wide to long format
        # SIMPLE_MATRIX_FEATURES (pcorr, zpcorr) are NOT banded and don't need melting
        banded_features = set(constants.BAND_FEATURES) | set(constants.BANDED_MATRIX_FEATURES)
        if feature in banded_features:
            logger.info(f"Melting banded feature {feature} from wide to long format")
            combined_df = melt_banded_feature(combined_df, feature)
            logger.info(f"After melt, DataFrame shape: {combined_df.shape}")
        
        # Process dataframe (add sex, gene columns)
        df_processed = process_feature_dataframe(combined_df)
        del combined_df
        gc.collect()
        
        # Add weights for equal animal contribution
        df_weighted = add_animal_weights(df_processed)
        del df_processed
        gc.collect()
        
        log_memory_usage(logger, f"after processing {feature}")
        
        # Save data in configured format
        if data_format == "csv":
            df_weighted.to_csv(data_dir / f"{feature}_relfreq.csv", index=False)
        else:  # default to pkl
            df_weighted.to_pickle(data_dir / f"{feature}_relfreq.pkl")
        
        # Create plots
        create_relfreq_plots_from_df(df_weighted, feature, feature_label, output_dir, relfreq_config)
        
        # Free memory before next feature
        del df_weighted
        gc.collect()
        
        log_memory_usage(logger, f"end {feature}")
        logger.info(f"=== Completed feature: {feature} ===\n")

    log_memory_usage(logger, "finished")
    logger.info(f"Successfully generated relative frequency plots for {len(features)} features")


if __name__ == "__main__":
    main()
