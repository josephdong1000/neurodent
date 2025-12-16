#!/usr/bin/env python3
"""
Zeitgeber Temporal Plots Generation Script
==========================================

Generate zeitgeber time (ZT) temporal analysis plots showing circadian patterns.
Based on the alphadelta example pipeline with seaborn objects plotting.

Input: Zeitgeber features pickle file from all animals
Output: Temporal figure files and processed data exports
"""

import sys
import logging
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import seaborn as sns
from seaborn import axes_style

# Import neurodent modules
from neurodent.core import zeitgeber
from neurodent.visualization.plotting import ZeitgeberPlotter
from neurodent import constants


logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    Load zeitgeber features dataframe.
    
    Args:
        file_path (Path or str): Path to the pickle file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    logger.info(f"Loading zeitgeber features from {file_path}")
    df = pd.read_pickle(file_path)
    logger.info(f"Loaded zeitgeber data with shape: {df.shape}")
    return df


def generate_plots(df, output_dir, data_dir, zt_config):
    """
    Generate all zeitgeber plots configured.
    
    Args:
        df (pd.DataFrame): Processed dataframe.
        output_dir (Path): Directory for figures.
        data_dir (Path): Directory for data exports.
        zt_config (dict): Configuration dictionary.
    """
    # Get format parameters
    figure_format = zt_config.get("figure_format", "png")
    data_format = zt_config.get("data_format", "csv")
    dpi = zt_config.get("dpi", 300)
    figsize = zt_config.get("figsize", [20, 20])
    
    # Instantiate plotter with dataframe
    plotter = ZeitgeberPlotter(df)
    
    # Identify feature columns (numeric columns excluding metadata)
    metadata_cols = ['animal', 'genotype', 'sex', 'gene', 'total_minutes', 'hour', 'minute',
                     'genotype_order', 'sex_order', 'timestamp']
    available_features = [col for col in df.columns 
                          if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    logger.info(f"Creating zeitgeber plots for {len(available_features)} features")
    
    # Save processed data
    if data_format == "csv":
        df.to_csv(data_dir / "zeitgeber_processed.csv", index=False)
    else:
        df.to_pickle(data_dir / "zeitgeber_processed.pkl")
    
    # Log animal counts for reference
    animal_counts = df.groupby(['gene', 'sex'])['animal'].nunique()
    logger.info(f"Animal counts by genotype and sex:\n{animal_counts}")
    
    # Generate all plots
    for i, feature in enumerate(available_features):
        logger.info(f"Creating zeitgeber plot for {feature}")
        output_path = output_dir / f"{i:02d}_{feature}.{figure_format}"
        
        # Use plotter instance
        plotter.plot_feature(feature, output_path, figsize, dpi)



def main():
    """Main zeitgeber plots generation function"""
    # Global snakemake object is injected by Snakemake execution
    global snakemake
    
    logger.info("Zeitgeber temporal plots generation started")

    # Get inputs and config
    zeitgeber_file = snakemake.input.zeitgeber_features
    config = snakemake.params.config
    
    # Create output directories
    output_dir = Path(snakemake.output.figure_dir)
    data_dir = Path(snakemake.output.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df = load_data(zeitgeber_file)

    # 2. Process Data (48h expansion)
    # Note: Data is already ZT-shifted and baseline-subtracted by extract_zeitgeber_features.py
    
    # 2. Process Data (48h expansion)
    # Note: Data is already ZT-shifted and baseline-subtracted by extract_zeitgeber_features.py
    df_processed = zeitgeber.prepare_plot_data(
        df, 
        shift_for_48h=True, 
        perform_zt_shift=False
    )

    # 3. Generate Plots
    zt_config = config["analysis"]["zeitgeber_plots"]
    generate_plots(df_processed, output_dir, data_dir, zt_config)
    
    logger.info("Successfully generated zeitgeber temporal plots")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )
    main()
