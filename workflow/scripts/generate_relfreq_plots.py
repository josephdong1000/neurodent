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
"""

import sys
import logging
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from okabeito import blue, green, orange, purple, red

from pythoneeg import visualization, constants

logger = logging.getLogger(__name__)


def load_war_for_relfreq(war_path_info):
    """
    Load a channel-filtered WAR for relative frequency plotting

    Args:
        war_path_info: Tuple of (war_pkl_path, war_json_path, animal_name)

    Returns:
        visualization.WindowAnalysisResult: Loaded WAR object
    """
    war_pkl_path, war_json_path, animal_name = war_path_info

    try:
        logger.info(f"Loading {animal_name}")

        # Load channel-filtered WAR using explicit PKL and JSON paths
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=war_pkl_path.parent, pickle_name=war_pkl_path.name, json_name=war_json_path.name
        )

        logger.info(f"Loaded WAR for {war.animal_id} ({war.genotype})")
        return war

    except Exception as e:
        logger.error(f"Failed to load {animal_name}: {str(e)}")
        raise


def process_feature_dataframe(df):
    """Process feature dataframe by adding categorical columns.

    Args:
        df (pd.DataFrame): Input dataframe with feature data

    Returns:
        pd.DataFrame: Processed dataframe with sex and gene columns
    """
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
    logger.info(f"\tBins: {bins}")

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


def create_relfreq_plots_for_feature(ep, feature, feature_label, output_dir, data_dir, relfreq_config):
    """Create relative frequency plots for a specific feature"""

    logger = logging.getLogger(__name__)
    logger.info(f"Processing feature: {feature}")

    # Get format parameters from config
    figure_format = relfreq_config.get("figure_format", "png")
    data_format = relfreq_config.get("data_format", "csv")
    dpi = relfreq_config.get("dpi", 300)

    try:
        # Pull raw timeseries data without averaging (key difference from EP plots)
        df_raw = ep.pull_timeseries_dataframe(
            feature=feature, groupby=["animal", "genotype", "isday"], collapse_channels=True, average_groupby=False
        )

        # Process raw dataframe (adds sex and gene columns)
        df_processed = process_feature_dataframe(df_raw)

        # Add weights for equal animal contribution to histogram
        df_weighted = add_animal_weights(df_processed)

        # Save data in configured format
        if data_format == "csv":
            df_weighted.to_csv(data_dir / f"{feature}_relfreq.csv", index=False)
        else:  # default to pkl
            df_weighted.to_pickle(data_dir / f"{feature}_relfreq.pkl")

        # Create relative frequency distribution plots
        if feature in ["logpsdfrac", "logpsdband", "psdband", "cohere", "zcohere", "imcoh", "zimcoh"]:
            # For band features, create per-band plots
            bands = ["delta", "theta", "alpha", "beta", "gamma"]
            for band in bands:
                df_band = df_weighted[df_weighted["band"] == band]
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
                hue_order=["delta", "theta", "alpha", "beta", "gamma"],
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

        logger.info(f"Successfully processed feature: {feature}")

    except Exception as e:
        logger.error(f"Failed to process feature {feature}: {str(e)}")
        raise


def main():
    """Main relative frequency plots generation function"""
    global snakemake
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            stream=sys.stdout,
            force=True,
        )
        logger = logging.getLogger(__name__)

        logger.info("Relative frequency plots generation started")

        # Get parameters from snakemake
        war_pkl_files = snakemake.input.war_pkl
        war_json_files = snakemake.input.war_json
        config = snakemake.params.config

        # Create output directories
        output_dir = Path(snakemake.output.figure_dir)
        data_dir = Path(snakemake.output.data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {len(war_pkl_files)} channel-filtered WARs")

        # Get number of threads for parallel loading
        threads = snakemake.threads
        logger.info(f"Using {threads} threads for parallel WAR loading")

        # Validate that PKL and JSON inputs match
        if len(war_pkl_files) != len(war_json_files):
            raise ValueError(
                f"Mismatch between PKL files ({len(war_pkl_files)}) and JSON files ({len(war_json_files)})"
            )

        # Prepare WAR information for parallel loading
        war_infos = []
        for pkl_file, json_file in zip(war_pkl_files, war_json_files):
            pkl_path = Path(pkl_file)
            json_path = Path(json_file)
            animal_name = pkl_path.parent.name
            war_infos.append((pkl_path, json_path, animal_name))

        # Load WARs in parallel
        wars = []
        if threads > 1:
            with Pool(threads) as pool:
                for war in tqdm(
                    pool.imap(load_war_for_relfreq, war_infos),
                    total=len(war_infos),
                    desc="Loading WARs for relative frequency plots",
                ):
                    if war is not None:
                        wars.append(war)
        else:
            # Single-threaded loading
            for war_info in tqdm(war_infos, desc="Loading WARs for relative frequency plots"):
                war = load_war_for_relfreq(war_info)
                if war is not None:
                    wars.append(war)

        if not wars:
            raise RuntimeError("No WARs were successfully loaded")

        logger.info(f"Successfully loaded {len(wars)} WARs")

        # Get relfreq configuration
        relfreq_config = config["analysis"]["relfreq_plots"]
        features = relfreq_config["features"]

        # Create genotype ordering
        genotype_order = ["MWT", "MHet", "MMut", "FWT", "FHet", "FMut"]
        plot_order = constants.DF_SORT_ORDER.copy()
        plot_order["genotype"] = genotype_order

        # Create ExperimentPlotter
        logger.info("Creating ExperimentPlotter")
        ep = visualization.ExperimentPlotter(wars=wars, exclude=None, plot_order=plot_order)

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

        # Process each feature
        for feature in features:
            if feature in feature_to_label:
                feature_label = feature_to_label[feature]
            else:
                feature_label = feature

            create_relfreq_plots_for_feature(ep, feature, feature_label, output_dir, data_dir, relfreq_config)

        logger.info(f"Successfully generated relative frequency plots for {len(features)} features")


if __name__ == "__main__":
    main()
