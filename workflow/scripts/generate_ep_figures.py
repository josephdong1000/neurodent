#!/usr/bin/env python3
"""
EP Statistical Figures Generation Script
=======================================

Generate experiment-level statistical figures using ExperimentPlotter and seaborn objects.
Based on the seaborn objects pipeline from notebooks/tests/ep figures example.py.

Input: Flattened WAR pickle and JSON files from all animals
Output: Statistical figure files (TIF) and CSV data exports
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style

from neurodent import visualization, constants
from neurodent.workflow import (
    setup_snakemake_logging,
    load_wars,
    apply_samples_config,
    extend_plot_order_from_attr,
    create_sex_marker_scale,
)

def infer_metadata_columns(df):
    """No-op passthrough: ``genotype`` is the canonical metadata column and is
    already present on WAR-derived frames (``sex`` likewise). Formerly added a
    redundant alias column."""
    return df.copy()

def process_feature_dataframe(df, feature):
    """Process feature dataframe by adding categorical columns and pivoting.

    Based on the process_feature_dataframe function from EP example.

    Args:
        df (pd.DataFrame): Input dataframe with feature data
        feature (str): Name of feature being processed

    Returns:
        tuple: (processed_df, pivoted_df)
    """
    if feature == "normpsd":
        ftype = constants.FeatureType.HIST
    else:
        ftype = constants.classify_feature(feature)

    if ftype.is_dict_stored:
        groupby = ["animal", "isday", "band"]
    elif ftype is constants.FeatureType.LINEAR_2D:
        raise ValueError(f"LINEAR_2D features (e.g. psdslope) not yet supported for EP plots")
    else:
        groupby = ["animal", "isday"]

    if "isday" not in df.columns:
        groupby.remove("isday")

    # Infer metadata (sex, gene) from genotype/animal ID
    df = infer_metadata_columns(df)

    if "isday" in df.columns:
        df["isday"] = df["isday"].map(lambda x: "Day" if x else "Night")

    # Create pivot table
    pivot_index = ["animal", "genotype", "sex"] if "genotype" in df.columns and "sex" in df.columns else ["animal"]
    if "freq" in df.columns:
        pivot_index.append("freq")
        
    pivot_columns = []
    if "isday" in df.columns:
        pivot_columns.append("isday")
    if "band" in df.columns:
        pivot_columns.append("band")
    if not pivot_columns:
        pivot_columns = None

    df_pivot = df.pivot_table(
        index=pivot_index,
        columns=pivot_columns,
        values=feature,
        aggfunc="mean",
        observed=True,
    ).reset_index()

    if isinstance(df_pivot.columns, pd.MultiIndex):
        df_pivot.columns = [
            "-".join(str(x) for x in col if x != "") if isinstance(col, tuple) else col for col in df_pivot.columns
        ]
    df_pivot.columns.name = None

    return df, df_pivot

def create_ep_plots(ep, feature, feature_label, output_dir, data_dir, ep_config):
    """Create plots for a specific feature using seaborn objects"""

    logger = logging.getLogger(__name__)
    logger.info(f"Processing feature: {feature}")

    # Get format parameters from config
    figure_format = ep_config.get("figure_format", "png")
    data_format = ep_config.get("data_format", "csv")
    dpi = ep_config.get("dpi", 300)

    try:
        # Pipeline 1: Pull averaged data for traditional plots (1 point per animal)
        if feature == "normpsd":
            df_avg = ep.pull_timeseries_dataframe(
                feature="psd", groupby=["animal", "genotype", "sex", "isday"], collapse_channels=True, average_groupby=True
            )
            df_total = ep.pull_timeseries_dataframe(
                feature="psdtotal",
                groupby=["animal", "genotype", "sex", "isday"],
                collapse_channels=True,
                average_groupby=True,
            )

            df_avg = df_avg.merge(df_total, on=["animal", "genotype", "sex", "channel"], suffixes=("", "_total"))
            df_avg["normpsd"] = df_avg["psd"] / df_avg["psdtotal"]
        else:
            df_avg = ep.pull_timeseries_dataframe(
                feature=feature, groupby=["animal", "genotype", "sex", "isday"], collapse_channels=True, average_groupby=True
            )

        # Process averaged dataframe (adds sex and gene columns)
        df, df_pivot = process_feature_dataframe(df_avg, feature)

        # Save data in configured format
        if data_format == "csv":
            df.to_csv(data_dir / f"{feature}.csv", index=False)
            df_pivot.to_csv(data_dir / f"{feature}-pivot.csv", index=False)
        else:  # default to pkl
            df.to_pickle(data_dir / f"{feature}.pkl")
            df_pivot.to_pickle(data_dir / f"{feature}-pivot.pkl")

        # Create plots based on feature type
        if feature == "normpsd":
            ftype = constants.FeatureType.HIST
        else:
            ftype = constants.classify_feature(feature)

        if ftype in (constants.FeatureType.LINEAR, constants.FeatureType.SIMPLE_MATRIX):
            # Bar plot with individual points
            p = (
                so.Plot(df, x="sex", y=feature, color="genotype", marker="sex")
                .facet(col="isday")
                .add(so.Dash(color="k"), so.Agg(), so.Dodge(empty="drop", gap=0.2))
                .add(so.Range(color="k"), so.Est(errorbar="sd"), so.Dodge(empty="drop", gap=0.2))
                .add(so.Dot(), so.Dodge(empty="drop", gap=0.2), so.Jitter(0.75, seed=42))
                .scale(marker=create_sex_marker_scale(df, plot_lib=so))
                .theme(
                    axes_style("ticks")
                    | sns.plotting_context("talk")
                    | {"axes.prop_cycle": plt.cycler(color=["blue", "blueviolet", "red"])}
                    | {"axes.spines.right": False, "axes.spines.top": False}
                )
                .layout(size=(6, 6))
                .label(y=feature_label)
            )
            p.save(output_dir / f"{feature}.{figure_format}", bbox_inches="tight", dpi=dpi)

        elif ftype.is_dict_stored:
            # By band plot — explicit Nominal(order=BAND_NAMES) on x because
            # seaborn-objects' Dodge/Jitter/Est can drop pd.Categorical order.
            # See ExperimentPlotter.band_scale docstring.
            p1 = (
                so.Plot(df, x="band", y=feature, color="genotype", marker="sex")
                .facet(col="isday")
                .add(so.Dash(color="k"), so.Agg(), so.Dodge())
                .add(so.Range(color="k"), so.Est(errorbar="sd"), so.Dodge())
                .add(so.Dot(), so.Dodge(), so.Jitter(0.75, seed=42))
                .scale(
                    x=visualization.ExperimentPlotter.band_scale(plot_lib=so),
                    marker=create_sex_marker_scale(df, plot_lib=so),
                )
                .theme(
                    axes_style("ticks")
                    | sns.plotting_context("notebook")
                    | {"axes.prop_cycle": plt.cycler(color=["blue", "blueviolet", "red", "blue", "blueviolet", "red"])}
                    | {"axes.spines.right": False, "axes.spines.top": False}
                )
                .label(x="Frequency band", y=feature_label)
                .layout(size=(10, 6), engine="tight")
            )
            p1.save(output_dir / f"byband-{feature}.{figure_format}", bbox_inches="tight", dpi=dpi)

            # By genotype plot — same Nominal(order=...) treatment for the
            # band-as-colour mapping so the legend reads in canonical order.
            p2 = (
                so.Plot(df, x="genotype", y=feature, color="band", marker="sex")
                .facet(col="isday")
                .add(so.Dash(color="k"), so.Agg(), so.Dodge())
                .add(so.Range(color="k"), so.Est(errorbar="sd"), so.Dodge())
                .add(so.Dot(), so.Dodge(), so.Jitter(0.75, seed=42))
                .scale(color=visualization.ExperimentPlotter.band_scale(plot_lib=so))
                .theme(
                    axes_style("ticks")
                    | sns.plotting_context("notebook")
                    | {"axes.spines.right": False, "axes.spines.top": False}
                )
                .layout(size=(10, 6), engine="tight")
                .label(x="Genotype", y=feature_label)
            )
            p2.save(output_dir / f"bygeno-{feature}.{figure_format}", bbox_inches="tight", dpi=dpi)

        elif ftype is constants.FeatureType.HIST:
            ylim = (1e-4, 1) if feature == "normpsd" else (0.3, 1e5)
            for scale in [so.Continuous(), "log"]:
                p = (
                    so.Plot(df, x="freq", y=feature, color="genotype")
                    .facet(col="sex", row="isday")
                    .add(so.Line(), so.Agg())
                    .add(so.Band(), so.Est())
                    .theme(
                        axes_style("ticks")
                        | sns.plotting_context("notebook")
                        | {"axes.prop_cycle": plt.cycler(color=["blue", "blueviolet", "red"])}
                        | {"axes.spines.right": False, "axes.spines.top": False}
                    )
                    .scale(x=scale, y=scale)
                    .limit(x=(lambda x: (1, 60) if callable(x) else (1, 100))(scale), y=ylim)
                    .layout(size=(10, 6))
                    .label(x="Frequency (Hz)", y=feature_label)
                )
                scale_name = "linear" if callable(scale) else scale
                p.save(output_dir / f"{feature}-{scale_name}.{figure_format}", bbox_inches="tight", dpi=dpi)

        logger.info(f"Successfully processed feature: {feature}")

    except Exception as e:
        logger.error(f"Failed to process feature {feature}: {str(e)}")
        raise

def main():
    """Main EP figures generation function"""
    global snakemake
    logger = setup_snakemake_logging(snakemake)
    logger.info("EP statistical figures generation started")

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

    logger.info(f"Loading {len(war_parquet_files)} flattened WARs")

    # Load WARs using the workflow utility
    wars = load_wars(war_parquet_files, war_json_files)
    for war in wars:
        logger.info(f"Loaded WAR for {war.animal_id} ({war.genotype})")

    logger.info(f"Successfully loaded {len(wars)} WARs")

    # Get EP configuration
    ep_config = config["analysis"]["ep_figures"]
    features = ep_config["features"]
    exclude_features = ep_config.get("exclude_features", [])

    # Extend the genotype/sex plot orders with any values observed on the
    # loaded WARs that aren't in the default DF_SORT_ORDER.  Without this,
    # datasets like arxrosa (every animal has sex='Unknown', genotype='UNKNOWN')
    # fail strict plot_order validation in sort_dataframe_by_plot_order.
    plot_order = constants.DF_SORT_ORDER.copy()
    plot_order["genotype"] = extend_plot_order_from_attr(
        wars, "genotype", constants.DF_SORT_ORDER.get("genotype", [])
    )
    plot_order["sex"] = extend_plot_order_from_attr(
        wars, "sex", constants.DF_SORT_ORDER.get("sex", [])
    )

    # Create ExperimentPlotter
    logger.info("Creating ExperimentPlotter")
    ep = visualization.ExperimentPlotter(wars=wars, exclude=exclude_features, plot_order=plot_order)

    feature_to_label = {
        **constants.FEATURE_LABELS,
        "normpsd": "Normalized PSD",
    }

    # Process each feature
    for feature in features:
        if feature in feature_to_label:
            feature_label = feature_to_label[feature]
        else:
            feature_label = feature

        create_ep_plots(ep, feature, feature_label, output_dir, data_dir, ep_config)
    logger.info(f"Successfully generated EP statistical figures for {len(features)} features")

if __name__ == "__main__":
    main()