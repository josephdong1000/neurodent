#!/usr/bin/env python3
"""
EP Statistical Figures Generation Script
=======================================

Generate experiment-level statistical figures using ExperimentPlotter and seaborn objects.
Based on the seaborn objects pipeline from notebooks/tests/ep figures example.py.

Input: Flattened WAR pickle and JSON files from all animals
Output: Statistical figure files (TIF) and CSV data exports
"""

import sys
import logging
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from okabeito import black, blue, green, lightblue, orange, purple, red, yellow
from seaborn import axes_style

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization, constants


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

    # Create pivot table
    df_pivot = df.pivot_table(
        index=["animal", "gene", "sex"] if "freq" not in df.columns else ["animal", "gene", "sex", "freq"],
        columns=["isday", "band"] if ("isday" in df.columns and "band" in df.columns) else "band" if "band" in df.columns else "isday" if "isday" in df.columns else None,
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


def compute_pmf_per_animal(df, feature, groupby_cols):
    """
    Compute PMF for each animal separately, then average the PMFs.

    Designed for discrete spike count data (nspike, lognspike). This prevents
    bias from unequal sample sizes across animals by computing per-animal PMFs
    first, then averaging them.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with individual samples (not averaged)
    feature : str
        Feature column name (e.g., 'nspike', 'lognspike')
    groupby_cols : list
        Columns to group by (e.g., ['genotype', 'isday', 'sex', 'gene'])

    Returns
    -------
    pd.DataFrame
        DataFrame with averaged PMF values for each unique spike count value
    """
    import logging
    logger = logging.getLogger(__name__)

    # Get data range across all animals to create consistent bins
    data_min = df[feature].min()
    data_max = df[feature].max()

    # Create bin edges with consistent spacing (default 0.1 bin width)
    bin_width = 0.1
    bin_edges = np.arange(data_min, data_max + bin_width, bin_width)

    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Debug logging
    logger.info(f"PMF computation for {feature}:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Value range: {data_min:.6f} to {data_max:.6f}")
    logger.info(f"  Bin width: {bin_width}")
    logger.info(f"  Number of bins: {len(bin_centers)}")
    logger.info(f"  Data type: {df[feature].dtype}")

    # For each animal, compute the PMF
    pmf_list = []

    for animal, animal_df in df.groupby("animal"):
        # Get the groupby metadata for this animal
        animal_meta = animal_df[groupby_cols].iloc[0].to_dict()

        # Compute histogram with consistent bins across all animals
        counts, _ = np.histogram(animal_df[feature].dropna(), bins=bin_edges)

        # Normalize to get PMF (sum to 1)
        total_counts = counts.sum()
        if total_counts > 0:
            pmf_values = counts / total_counts
        else:
            pmf_values = np.zeros_like(counts, dtype=float)

        # Convert to dataframe
        animal_pmf = pd.DataFrame({feature: bin_centers, "pmf": pmf_values, "animal": animal})

        # Add metadata
        for col, val in animal_meta.items():
            animal_pmf[col] = val

        pmf_list.append(animal_pmf)

    # Concatenate all animal PMFs
    all_pmfs = pd.concat(pmf_list, ignore_index=True)

    # Average the PMFs across animals within each group
    # This is the key step: average PMFs, not raw counts
    # All animals have same bin_centers, so groupby will align correctly
    averaged_pmf = all_pmfs.groupby(groupby_cols + [feature])["pmf"].mean().reset_index()

    return averaged_pmf


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

        # Pipeline 2: Pull raw timeseries data for PMF plots (except psd/normpsd)
        if feature not in ["psd", "normpsd"]:
            # Get raw data without averaging
            df_raw = ep.pull_timeseries_dataframe(
                feature=feature, groupby=["animal", "genotype", "isday"], collapse_channels=True, average_groupby=False
            )

            # Process raw dataframe (adds sex and gene columns)
            df_raw_processed, _ = process_feature_dataframe(df_raw, feature)

            # Compute PMF per animal, then average PMFs
            # For band features, include band in groupby to get separate PMFs per band
            if feature in ["logpsdfrac", "logpsdband", "psdband", "cohere", "zcohere", "imcoh", "zimcoh"]:
                df_pmf = compute_pmf_per_animal(df_raw_processed, feature, ["genotype", "isday", "sex", "gene", "band"])
            else:
                df_pmf = compute_pmf_per_animal(df_raw_processed, feature, ["genotype", "isday", "sex", "gene"])

        # Save data in configured format
        if data_format == "csv":
            df.to_csv(data_dir / f"{feature}.csv", index=False)
            df_pivot.to_csv(data_dir / f"{feature}-pivot.csv", index=False)
            if feature not in ["psd", "normpsd"]:
                df_pmf.to_csv(data_dir / f"{feature}-pmf.csv", index=False)
        else:  # default to pkl
            df.to_pickle(data_dir / f"{feature}.pkl")
            df_pivot.to_pickle(data_dir / f"{feature}-pivot.pkl")
            if feature not in ["psd", "normpsd"]:
                df_pmf.to_pickle(data_dir / f"{feature}-pmf.pkl")

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
                    | {"axes.prop_cycle": plt.cycler(color=[blue, orange, red, green, purple, yellow, lightblue, black])}
                    | {"axes.spines.right": False, "axes.spines.top": False}
                )
                .layout(size=(10, 6), engine="tight")
                .label(x="Genotype", y=feature_label)
            )
            p2.save(output_dir / f"bygeno-{feature}.{figure_format}", bbox_inches="tight", dpi=dpi)

        elif feature == "psd" or feature == "normpsd":
            ylim = (1e-4, 1) if feature == "normpsd" else (0.3, 3000)
            for scale in [so.Continuous(), 'log']:
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
                    .limit(x=(lambda x: (1,60) if callable(x) else (1,100))(scale), y=ylim)
                    .layout(size=(10, 6))
                    .label(x="Frequency (Hz)", y=feature_label)
                )
                scale_name = 'linear' if callable(scale) else scale
                p.save(output_dir / f"{feature}-{scale_name}.{figure_format}", bbox_inches="tight", dpi=dpi)

        # For all features except psd/normpsd, also create PMF distribution plots
        if feature not in ["psd", "normpsd"]:
            if feature in ["logpsdfrac", "logpsdband", "psdband", "cohere", "zcohere", "imcoh", "zimcoh"]:
                # For band features, create per-band PMF plots
                bands = ["delta", "theta", "alpha", "beta", "gamma"]
                for band in bands:
                    df_pmf_band = df_pmf[df_pmf["band"] == band]
                    for scale in [so.Continuous(), "log"]:
                        p = (
                            so.Plot(df_pmf_band, x=feature, y="pmf", color="gene")
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
                            .layout(size=(10, 6))
                            .label(x=f"{feature_label} ({band})", y="Probability Mass")
                        )
                        scale_name = "linear" if callable(scale) else scale
                        p.save(
                            output_dir / f"{feature}-pmf-{band}-{scale_name}.{figure_format}",
                            bbox_inches="tight",
                            dpi=dpi,
                        )

                # Also create combined band comparison plot
                for scale in [so.Continuous(), "log"]:
                    p = (
                        so.Plot(df_pmf, x=feature, y="pmf", color="band")
                        .facet(col="sex", row="isday")
                        .add(so.Line(), so.Agg())
                        .add(so.Band(), so.Est())
                        .theme(
                            axes_style("ticks")
                            | sns.plotting_context("notebook")
                            | {"axes.prop_cycle": plt.cycler(color=[blue, orange, red, green, purple])}
                            | {"axes.spines.right": False, "axes.spines.top": False}
                        )
                        .scale(x=scale, y=scale)
                        .layout(size=(10, 6))
                        .label(x=feature_label, y="Probability Mass")
                    )
                    scale_name = "linear" if callable(scale) else scale
                    p.save(
                        output_dir / f"{feature}-pmf-byband-{scale_name}.{figure_format}", bbox_inches="tight", dpi=dpi
                    )
            else:
                # For non-band features, single PMF plot
                for scale in [so.Continuous(), "log"]:
                    p = (
                        so.Plot(df_pmf, x=feature, y="pmf", color="gene")
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
                        .layout(size=(10, 6))
                        .label(x=feature_label, y="Probability Mass")
                    )
                    scale_name = "linear" if callable(scale) else scale
                    p.save(output_dir / f"{feature}-pmf-{scale_name}.{figure_format}", bbox_inches="tight", dpi=dpi)

        logger.info(f"Successfully processed feature: {feature}")
        
    except Exception as e:
        logger.error(f"Failed to process feature {feature}: {str(e)}")
        raise


def main():
    """Main EP figures generation function"""
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

        logger.info("EP statistical figures generation started")
        
        # Get parameters from snakemake
        war_pkl_files = snakemake.input.war_pkl
        war_json_files = snakemake.input.war_json
        config = snakemake.params.config
        
        # Create output directories
        output_dir = Path(snakemake.output.figure_dir)
        data_dir = Path(snakemake.output.data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading {len(war_pkl_files)} flattened WARs")
        
        # Load WARs - let failures be visible rather than silently continuing
        wars = []
        for pkl_file, json_file in zip(war_pkl_files, war_json_files):
            war = visualization.WindowAnalysisResult.load_pickle_and_json(
                folder_path=Path(pkl_file).parent,
                pickle_name=Path(pkl_file).name,
                json_name=Path(json_file).name
            )
            
            wars.append(war)
            logger.info(f"Loaded WAR for {war.animal_id} ({war.genotype})")
        
        if not wars:
            raise RuntimeError("No WARs were successfully loaded")
        
        logger.info(f"Successfully loaded {len(wars)} WARs")
        
        # Get EP configuration
        ep_config = config["analysis"]["ep_figures"]
        features = ep_config["features"]
        exclude_features = ep_config.get("exclude_features", [])
        
        # Create genotype ordering
        genotype_order = ['MWT', 'MHet', 'MMut', 'FWT', 'FHet', 'FMut']
        plot_order = constants.DF_SORT_ORDER.copy()
        plot_order['genotype'] = genotype_order
        
        # Create ExperimentPlotter
        logger.info("Creating ExperimentPlotter")
        ep = visualization.ExperimentPlotter(
            wars=wars,
            exclude=exclude_features,
            plot_order=plot_order
        )
        
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