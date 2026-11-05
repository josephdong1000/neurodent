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

Memory: each WAR is loaded in full exactly once, one animal at a time, through
neurodent.workflow.iter_wars, and only the per-window frame of each configured feature is
kept from it. A loaded WAR holds every feature as nested Python lists and is many times
larger than its parquet file. The earlier design, a pool that loaded up to `threads` WARs
at once and loaded every WAR again for each feature, ran out of memory on large datasets,
and a pool worker killed for using too much memory left the pool waiting until the job's
time limit. The rule's threads setting is no longer used.
"""

import gc
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from neurodent.workflow import setup_snakemake_logging, apply_samples_config, iter_wars
from neurodent.core import metadata as metadata_module

from neurodent.constants import OKABE_ITO_COLORS
from neurodent import constants

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


def extract_features_from_war(war, features):
    """
    Return the per-window, channel-averaged frame of every requested feature of one WAR.

    Each frame is the frame the earlier one-feature-per-load extraction returned: the
    channel-averaged result for that one feature with every non-feature column of the
    WAR, plus 'animal' and 'genotype'. Extraction does not modify the WAR, so taking
    every feature from a single load gives the same frames as loading the WAR once per
    feature. The frames do not refer to the WAR, so it can be released afterwards.

    This replaced extract_feature_from_war, which loaded the WAR itself, one feature per
    load, and had a branch for not averaging channels. That branch called
    get_windowed_result, which does not exist, and nothing used it.

    Args:
        war (WindowAnalysisResult): A loaded, non-flattened WAR.
        features (list[str]): Feature names to extract.

    Returns:
        dict[str, pd.DataFrame]: Feature name to that feature's per-window frame.
    """
    try:
        frames = {}
        for feature in features:
            # Extract just the feature data using channel averaging
            df = war.get_channel_averaged_result(features=[feature])

            # Add metadata columns
            df['animal'] = war.animal_id
            df['genotype'] = war.genotype
            # Note: 'animalday' column should already exist in df from get_channel_averaged_result
            # if it was present in war.result. No need to add it separately.

            # Ensure isday column exists
            if 'isday' not in df.columns and hasattr(war, 'result') and 'isday' in war.result.columns:
                df['isday'] = war.result['isday'].values[:len(df)]

            frames[feature] = df

        return frames

    except Exception as e:
        logger.error(f"Failed to extract {features} from {war.animal_id}: {str(e)}")
        raise


def process_feature_dataframe(df, samples_config):
    """Process feature dataframe by adding categorical columns using samples mapping.

    Args:
        df (pd.DataFrame): Input dataframe with feature data
        samples_config (dict): Samples configuration containing ANIMAL_METADATA

    Returns:
        pd.DataFrame: Processed dataframe with sex and gene columns
    """
    df = df.copy()
    
    # Use ANIMAL_METADATA for explicit sex/gene lookup (required)
    animal_metadata = metadata_module.load_animal_metadata(samples_config)
    df = metadata_module.enrich_metadata(df, animal_metadata)

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
        Column name for hue (e.g., 'genotype' or 'band')
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


def genotype_hue(df, column="genotype"):
    """Return ``(hue_order, palette)`` sized to the genotypes actually present.

    FacetGrid cycles a palette list that is shorter than ``hue_order`` silently, so the
    three fixed colours this module used gave two pairs of genotypes the same colour once a
    dataset had more than three. Sizing the palette to the levels removes the failure mode
    rather than relying on a warning that this code path does not emit.

    Order follows :data:`~neurodent.constants.DF_SORT_ORDER` where it applies, so these
    plots agree with the ep_figures ordering instead of applying a separate ASCII sort.
    """
    observed = sorted(df[column].dropna().unique().tolist())
    preferred = [value for value in constants.DF_SORT_ORDER.get(column, []) if value in observed]
    hue_order = preferred + [value for value in observed if value not in preferred]

    base = ["blue", "blueviolet", "red"]
    if len(hue_order) <= len(base):
        palette = base[: len(hue_order)]
    else:
        palette = sns.color_palette("husl", len(hue_order)).as_hex()
    return hue_order, palette


def create_relfreq_plots_from_df(df_weighted, feature, feature_label, output_dir, relfreq_config):
    """Create relative frequency plots for a specific feature from pre-extracted DataFrame"""

    logger = logging.getLogger(__name__)

    # Get format parameters from config
    figure_format = relfreq_config.get("figure_format", "png")
    dpi = relfreq_config.get("dpi", 300)

    hue_order, palette = genotype_hue(df_weighted)

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
                hue="genotype",
                hue_order=hue_order,
                palette=palette,
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
            palette=[
                OKABE_ITO_COLORS["blue"],
                OKABE_ITO_COLORS["orange"],
                OKABE_ITO_COLORS["red"],
                OKABE_ITO_COLORS["green"],
                OKABE_ITO_COLORS["purple"],
            ],
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
            hue="genotype",
            hue_order=hue_order,
            palette=palette,
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
    war_parquet_files = snakemake.input.war_parquet
    war_json_files = snakemake.input.war_json
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    apply_samples_config(samples_config)

    # Create output directories
    output_dir = Path(snakemake.output.figure_dir)
    data_dir = Path(snakemake.output.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(war_parquet_files)} channel-filtered WARs")

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
    log_memory_usage(logger, "before extraction")

    # Load each WAR once, one animal at a time, in input order, and keep only the
    # per-window frame of each feature. The per-feature lists below hold the same
    # frames, in the same order, as the earlier per-feature extraction passes.
    frames_by_feature = {feature: [] for feature in features}
    for parquet_path, war in iter_wars(war_parquet_files, war_json_files):
        frames = extract_features_from_war(war, features)
        del war  # iter_wars loads the next WAR only after this one is released
        for feature in features:
            frames_by_feature[feature].append(frames[feature])
        del frames
        log_memory_usage(logger, f"after extracting {parquet_path.parent.name}")

    # Process each feature from the extracted frames
    for feature in features:
        logger.info(f"=== Processing feature: {feature} ===")
        log_memory_usage(logger, f"start {feature}")

        feature_label = feature_to_label.get(feature, feature)

        # This feature's frames, one per animal; popped so they are freed after concat
        dfs = frames_by_feature.pop(feature)

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
        df_processed = process_feature_dataframe(combined_df, samples_config)
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
