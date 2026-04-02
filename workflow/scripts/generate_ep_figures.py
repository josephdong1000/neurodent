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
from neurodent.visualization.results import WindowAnalysisResult
from neurodent.workflow import setup_snakemake_logging, load_wars, inject_config_aliases


def infer_metadata_columns(df):
    """
    Infer metadata columns (sex, gene) from genotype and animal ID.
    This logic is isolated as it may change depending on project conventions.
    """
    df = df.copy()

    # Add categorical columns based on genotype
    if "genotype" in df.columns:
        # First try to map from standard genotypes
        df["sex"] = df["genotype"].map(
            lambda x: "Male" if x in ["MWT", "MHet", "MMut"] else "Female" if x in ["FWT", "FHet", "FMut"] else None
        )
        
        # If sex is missing, try to infer from animal ID (e.g. "...-M" or "...-F")
        if df["sex"].isnull().any() and "animal" in df.columns:
            def infer_sex_from_animal(row):
                if pd.notna(row["sex"]):
                    return row["sex"]
                animal = str(row["animal"]).upper()
                if animal.endswith("-M") or animal.endswith("_M"):
                    return "Male"
                if animal.endswith("-F") or animal.endswith("_F"):
                    return "Female"
                return "Unknown"
            
            df["sex"] = df.apply(infer_sex_from_animal, axis=1)

        # Map gene/genotype
        df["gene"] = df["genotype"].map(
            lambda x: "WT"
            if x in ["MWT", "FWT", "WT"]  # Added WT explicitly
            else "Het"
            if x in ["MHet", "FHet"]
            else "Mut"
            if x in ["MMut", "FMut", "HOMO"] # Added HOMO explicitly if possible, or just fallback
            else x
        )
    return df


def process_feature_dataframe(df, feature):
    """Process feature dataframe by adding categorical columns and pivoting.

    Based on the process_feature_dataframe function from EP example.

    Args:
        df (pd.DataFrame): Input dataframe with feature data
        feature (str): Name of feature being processed

    Returns:
        tuple: (processed_df, pivoted_df)
    """
    if feature in ["logpsdfrac", "logpsdband", "psdband", "cohere", "zcohere", "imcoh", "zimcoh"]:
        groupby = ["animal", "isday", "band"]
    elif feature in ["pcorr", "zpcorr", "psd", "normpsd", "nspike", "lognspike"]:
        groupby = ["animal", "isday"]
    else:
        raise ValueError(f"Feature {feature} not supported")

    if "isday" not in df.columns:
        groupby.remove("isday")

    # Infer metadata (sex, gene) from genotype/animal ID
    df = infer_metadata_columns(df)

    if "isday" in df.columns:
        df["isday"] = df["isday"].map(lambda x: "Day" if x else "Night")

    # Create pivot table
    pivot_index = ["animal", "gene", "sex"] if "gene" in df.columns and "sex" in df.columns else ["animal"]
    if "genotype" in df.columns and "gene" not in df.columns:
        pivot_index.append("genotype")
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
                feature="psd", groupby=["animal", "genotype", "isday"], collapse_channels=True, average_groupby=True
            )
            df_total = ep.pull_timeseries_dataframe(
                feature="psdtotal",
                groupby=["animal", "genotype", "isday"],
                collapse_channels=True,
                average_groupby=True,
            )

            df_avg = df_avg.merge(df_total, on=["animal", "genotype", "channel"], suffixes=("", "_total"))
            df_avg["normpsd"] = df_avg["psd"] / df_avg["psdtotal"]
        else:
            df_avg = ep.pull_timeseries_dataframe(
                feature=feature, groupby=["animal", "genotype", "isday"], collapse_channels=True, average_groupby=True
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
        if feature in ["pcorr", "zpcorr", "nspike", "lognspike"]:
            # Bar plot with individual points
            p = (
                so.Plot(df, x="sex", y=feature, color="gene", marker="sex")
                .facet(col="isday")
                .add(so.Dash(color="k"), so.Agg(), so.Dodge(empty="drop", gap=0.2))
                .add(so.Range(color="k"), so.Est(errorbar="sd"), so.Dodge(empty="drop", gap=0.2))
                .add(so.Dot(), so.Dodge(empty="drop", gap=0.2), so.Jitter(0.75, seed=42))
                .scale(marker=so.Nominal(["o", "s"], order=["Female", "Male"]))
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

        elif feature in ["logpsdfrac", "logpsdband", "psdband", "cohere", "zcohere", "imcoh", "zimcoh"]:
            # By band plot
            p1 = (
                so.Plot(df, x="band", y=feature, color="gene", marker="sex")
                .facet(col="isday")
                .add(so.Dash(color="k"), so.Agg(), so.Dodge())
                .add(so.Range(color="k"), so.Est(errorbar="sd"), so.Dodge())
                .add(so.Dot(), so.Dodge(), so.Jitter(0.75, seed=42))
                .scale(marker=so.Nominal(["o", "s"], order=["Female", "Male"]))
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

            # By genotype plot
            p2 = (
                so.Plot(df, x="gene", y=feature, color="band", marker="sex")
                .facet(col="isday")
                .add(so.Dash(color="k"), so.Agg(), so.Dodge())
                .add(so.Range(color="k"), so.Est(errorbar="sd"), so.Dodge())
                .add(so.Dot(), so.Dodge(), so.Jitter(0.75, seed=42))
                .theme(
                    axes_style("ticks")
                    | sns.plotting_context("notebook")
                    | {"axes.spines.right": False, "axes.spines.top": False}
                )
                .layout(size=(10, 6), engine="tight")
                .label(x="Genotype", y=feature_label)
            )
            p2.save(output_dir / f"bygeno-{feature}.{figure_format}", bbox_inches="tight", dpi=dpi)

        elif feature == "psd" or feature == "normpsd":
            ylim = (1e-4, 1) if feature == "normpsd" else (0.3, 3000)
            for scale in [so.Continuous(), "log"]:
                p = (
                    so.Plot(df, x="freq", y=feature, color="gene")
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
    war_pkl_files = snakemake.input.war_pkl
    war_json_files = snakemake.input.war_json
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    inject_config_aliases(samples_config)

    # Create output directories
    output_dir = Path(snakemake.output.figure_dir)
    data_dir = Path(snakemake.output.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {len(war_pkl_files)} flattened WARs")

    # Load WARs using the workflow utility
    wars = load_wars(war_pkl_files, war_json_files)
    # Ensure we have WindowAnalysisResult objects (backwards-compatibility)
    normalized_wars = []
    for w in wars:
        if isinstance(w, WindowAnalysisResult):
            normalized_wars.append(w)
            continue
        # If w is a tuple or mapping with pkl/json paths, try to load via loader
        try:
            # support (pkl_path, json_path) tuples or objects with attributes
            if isinstance(w, (tuple, list)) and len(w) >= 2:
                pkl_path, json_path = w[0], w[1]
            else:
                pkl_path = getattr(w, "pkl", None) or getattr(w, "pkl_path", None) or getattr(w, "pickle", None)
                json_path = getattr(w, "json", None) or getattr(w, "json_path", None)
            if pkl_path is not None and json_path is not None:
                loaded = WindowAnalysisResult.load_pickle_and_json(
                    folder_path=Path(pkl_path).parent,
                    pickle_name=Path(pkl_path).name,
                    json_name=Path(json_path).name,
                )
                normalized_wars.append(loaded)
            else:
                normalized_wars.append(w)
        except Exception:
            normalized_wars.append(w)
    wars = normalized_wars

    # Get EP configuration
    ep_config = config["analysis"]["ep_figures"]
    features = ep_config["features"]
    exclude_features = ep_config.get("exclude_features", [])

    # Create genotype ordering - ensure we include all genotypes present in the data
    genotype_order = ["MWT", "MHet", "MMut", "FWT", "FHet", "FMut"]
    
    # Check for genotypes in loaded WARs that aren't in the default list
    found_genotypes = set()
    for war in wars:
        if hasattr(war, "genotype") and war.genotype:
            found_genotypes.add(war.genotype)
    
    # Add any missing genotypes to the order list
    for gt in found_genotypes:
        if gt not in genotype_order:
            logging.info(f"Adding unknown genotype '{gt}' to plot order")
            genotype_order.append(gt)
            
    plot_order = constants.DF_SORT_ORDER.copy()
    plot_order["genotype"] = genotype_order

    # Create ExperimentPlotter
    logger.info("Creating ExperimentPlotter")
    ep = visualization.ExperimentPlotter(wars=wars, exclude=exclude_features, plot_order=plot_order)

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
        "psd": "PSD ($\\mu V^2/Hz$)",
        "normpsd": "Normalized PSD",
        "nspike": "n_spike / t_window",
        "lognspike": "Log(n_spike / t_window)",
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
