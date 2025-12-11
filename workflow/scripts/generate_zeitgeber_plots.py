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

# Import the new zeitgeber module
from neurodent.analysis import zeitgeber


def create_zeitgeber_plots(df, output_dir, data_dir, zt_config):
    """Create zeitgeber temporal plots using seaborn objects"""

    logger = logging.getLogger(__name__)

    # Get format parameters from config
    figure_format = zt_config.get("figure_format", "png")
    data_format = zt_config.get("data_format", "csv")
    dpi = zt_config.get("dpi", 300)
    figsize = zt_config.get("figsize", [20, 20])
    
    # Feature to label mapping
    feature_to_label = {
        'logrms': "Log(RMS)",
        'alphadelta': "Alpha/Delta ratio",
        'delta': "Log Delta band power",
        'alpha': "Log Alpha band power",
        'logpsdband': "Log Band Power",
        'zpcorr': "Z-transformed PCC",
        'logrms_nobase': "Log(RMS) - Baseline",
        'alphadelta_nobase': "Alpha/Delta ratio - Baseline",
        'delta_nobase': "Log Delta band power - Baseline",
        'alpha_nobase': "Log Alpha band power - Baseline",
        'logpsdband_nobase': "Log Band Power - Baseline",
        'zpcorr_nobase': "Z-transformed PCC - Baseline",
    }

    # Add band-specific labels for logpsdband and logpsdfrac
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        feature_to_label[f"logpsdband_{band}"] = f"Log Power - {band.capitalize()} band"
        feature_to_label[f"logpsdband_{band}_nobase"] = f"Log Power - {band.capitalize()} band - Baseline"
        feature_to_label[f"logpsdfrac_{band}"] = f"Log Power Fraction - {band.capitalize()} band"
        feature_to_label[f"logpsdfrac_{band}_nobase"] = f"Log Power Fraction - {band.capitalize()} band - Baseline"
        feature_to_label[f"zcohere_{band}"] = f"Z-Coherence - {band.capitalize()} band"
        feature_to_label[f"zcohere_{band}_nobase"] = f"Z-Coherence - {band.capitalize()} band - Baseline"
        feature_to_label[f"zimcoh_{band}"] = f"Z-ImCoh - {band.capitalize()} band"
        feature_to_label[f"zimcoh_{band}_nobase"] = f"Z-ImCoh - {band.capitalize()} band - Baseline"
    
    # Get available features from dataframe
    available_features = [col for col in df.columns 
                         if col in feature_to_label or col.endswith('_nobase')]
    
    logger.info(f"Creating zeitgeber plots for {len(available_features)} features")
    
    # Save processed data
    if data_format == "csv":
        df.to_csv(data_dir / "zeitgeber_processed.csv", index=False)
    else:
        df.to_pickle(data_dir / "zeitgeber_processed.pkl")
    
    # Log animal counts for reference
    animal_counts = df.groupby(['gene', 'sex'])['animal'].nunique()
    logger.info(f"Animal counts by genotype and sex:\n{animal_counts}")
    
    for i, feature in enumerate(available_features):
        logger.info(f"Creating zeitgeber plot for {feature}")
        
        try:
            p = (
                so.Plot(df, x="total_minutes", y=feature, color="gene")
                .facet(col="sex", row="gene")
                .add(so.Line(linewidth=2), so.Agg())
                .add(so.Dot(), so.Agg())
                .add(so.Band(), so.Est())
                .layout(size=(1, 1))
                .theme(axes_style("ticks") | sns.plotting_context("poster"))
                .label(y=feature_to_label.get(feature, feature))
            )
            
            fig = mpl.figure.Figure(figsize=figsize)
            p.on(fig).plot()
            
            # Add ZT formatting and day/night shading
            for ax in fig.axes:
                # Shade night periods (12-24h and 36-48h)
                ax.axvspan(xmin=12 * 60, xmax=24 * 60, alpha=0.1, color='grey')
                ax.axvspan(xmin=36 * 60, xmax=48 * 60, alpha=0.1, color='grey')
                
                # Set ticks every 6 hours
                ax.set_xticks(np.arange(0, 49 * 60, 6 * 60))
                new_labels = [(x/60) % 24 for x in ax.get_xticks()]
                ax.set_xticklabels([f"{x:.0f}" for x in new_labels])
                ax.set_xlabel("ZT")
            
            fig.tight_layout()
            fig.savefig(output_dir / f"{i:02d}_{feature}.{figure_format}", 
                       bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            
            logger.info(f"Successfully created zeitgeber plot for {feature}")
            
        except Exception as e:
            logger.error(f"Failed to create zeitgeber plot for {feature}: {str(e)}")
            raise


def main():
    """Main zeitgeber plots generation function"""
    global snakemake
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info("Zeitgeber temporal plots generation started")

    # Get parameters from snakemake
    zeitgeber_file = snakemake.input.zeitgeber_features
    config = snakemake.params.config

    # Create output directories
    output_dir = Path(snakemake.output.figure_dir)
    data_dir = Path(snakemake.output.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading zeitgeber features from {zeitgeber_file}")

    # Load zeitgeber features dataframe
    df = pd.read_pickle(zeitgeber_file)
    logger.info(f"Loaded zeitgeber data with shape: {df.shape}")

    # Process data for temporal plotting using new module
    df_processed = zeitgeber.process_zeitgeber_data(df, config)

    # Get zeitgeber plots configuration
    zt_config = config["analysis"]["zeitgeber_plots"]

    # Create zeitgeber temporal plots
    create_zeitgeber_plots(df_processed, output_dir, data_dir, zt_config)

    logger.info("Successfully generated zeitgeber temporal plots")


if __name__ == "__main__":
    main()
