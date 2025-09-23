#!/usr/bin/env python3
"""
Filtering Comparison Analysis Script
===================================

This script compares manual vs LOF channel filtering approaches by analyzing
differences in extracted features between the two methods.

Input: Flattened WARs from both manual and LOF filtering pipelines
Output: Comparison plots and statistical analysis
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization


def load_wars_from_paths(war_paths, label):
    """Load multiple WARs and combine into a single dataset with labels"""
    wars = []
    for path in war_paths:
        try:
            war_dir = Path(path).parent
            war = visualization.WindowAnalysisResult.load_pickle_and_json(
                folder_path=war_dir, pickle_name="war.pkl", json_name="war.json"
            )

            # Add filtering method label to the dataframe
            war_df = war.result.copy()
            war_df["filtering_method"] = label
            war_df["animal"] = war.animal_id
            wars.append(war_df)

            logging.info(f"Loaded {label} WAR for {war.animal_id}: {len(war_df)} records")

        except Exception as e:
            logging.error(f"Failed to load WAR from {path}: {e}")
            continue

    if wars:
        combined_df = pd.concat(wars, ignore_index=True)
        logging.info(f"Combined {label} dataset: {len(combined_df)} total records from {len(wars)} animals")
        return combined_df
    else:
        logging.warning(f"No {label} WARs loaded successfully")
        return pd.DataFrame()


def generate_feature_scatter_plots(manual_df, lof_df, features, output_dir):
    """Generate scatter plots comparing features between manual and LOF filtering"""

    logging.info("Generating feature scatter plots")

    # Merge datasets on common identifiers for direct comparison
    manual_agg = manual_df.groupby(["animal", "animalday", "channel"])[features].mean().reset_index()
    lof_agg = lof_df.groupby(["animal", "animalday", "channel"])[features].mean().reset_index()

    merged = pd.merge(
        manual_agg, lof_agg, on=["animal", "animalday", "channel"], suffixes=("_manual", "_lof"), how="inner"
    )

    if merged.empty:
        logging.warning("No matching records found between manual and LOF datasets")
        return

    logging.info(f"Found {len(merged)} matching records for comparison")

    # Create scatter plots for each feature
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        manual_vals = merged[f"{feature}_manual"]
        lof_vals = merged[f"{feature}_lof"]

        # Remove infinite and NaN values
        valid_mask = np.isfinite(manual_vals) & np.isfinite(lof_vals)
        manual_clean = manual_vals[valid_mask]
        lof_clean = lof_vals[valid_mask]

        if len(manual_clean) > 0:
            # Scatter plot
            ax.scatter(manual_clean, lof_clean, alpha=0.6, s=20)

            # Add diagonal line (perfect correlation)
            min_val = min(manual_clean.min(), lof_clean.min())
            max_val = max(manual_clean.max(), lof_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="Perfect correlation")

            # Calculate correlation
            if len(manual_clean) > 1:
                correlation = stats.pearsonr(manual_clean, lof_clean)[0]
                rmse = np.sqrt(mean_squared_error(manual_clean, lof_clean))
                ax.text(
                    0.05,
                    0.95,
                    f"r = {correlation:.3f}\nRMSE = {rmse:.3f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            ax.set_xlabel(f"{feature} (Manual Filtering)")
            ax.set_ylabel(f"{feature} (LOF Filtering)")
            ax.set_title(f"{feature} Comparison")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center")
            ax.set_title(f"{feature} (No Data)")

    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_scatter_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    logging.info("Feature scatter plots saved")


def generate_channel_impact_analysis(manual_df, lof_df, features, output_dir):
    """Analyze the impact of filtering on different channels"""

    logging.info("Generating channel impact analysis")

    # Calculate mean features per channel for each filtering method
    manual_channel = manual_df.groupby("channel")[features].mean()
    lof_channel = lof_df.groupby("channel")[features].mean()

    # Calculate differences
    common_channels = manual_channel.index.intersection(lof_channel.index)
    if len(common_channels) == 0:
        logging.warning("No common channels found between filtering methods")
        return

    channel_diff = lof_channel.loc[common_channels] - manual_channel.loc[common_channels]

    # Create heatmap of differences
    plt.figure(figsize=(12, 8))
    sns.heatmap(channel_diff.T, annot=True, cmap="RdBu_r", center=0, fmt=".3f", cbar_kws={"label": "LOF - Manual"})
    plt.title("Channel-wise Feature Differences (LOF - Manual Filtering)")
    plt.xlabel("Channel")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "channel_impact_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save channel impact data
    channel_diff.to_csv(output_dir / "channel_impact_differences.csv")

    logging.info("Channel impact analysis saved")


def generate_animal_correlation_analysis(manual_df, lof_df, features, output_dir):
    """Analyze correlation of animal-level aggregated features"""

    logging.info("Generating animal-level correlation analysis")

    # Aggregate features by animal
    manual_animal = manual_df.groupby("animal")[features].mean()
    lof_animal = lof_df.groupby("animal")[features].mean()

    common_animals = manual_animal.index.intersection(lof_animal.index)
    if len(common_animals) == 0:
        logging.warning("No common animals found between filtering methods")
        return

    # Calculate correlations for each feature
    correlations = {}
    for feature in features:
        manual_vals = manual_animal.loc[common_animals, feature]
        lof_vals = lof_animal.loc[common_animals, feature]

        valid_mask = np.isfinite(manual_vals) & np.isfinite(lof_vals)
        if valid_mask.sum() > 1:
            corr, p_val = stats.pearsonr(manual_vals[valid_mask], lof_vals[valid_mask])
            correlations[feature] = {"correlation": corr, "p_value": p_val, "n_animals": valid_mask.sum()}

    # Create correlation summary plot
    if correlations:
        corr_df = pd.DataFrame(correlations).T

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            range(len(corr_df)),
            corr_df["correlation"],
            color=["green" if p < 0.05 else "orange" for p in corr_df["p_value"]],
        )
        plt.xlabel("Feature")
        plt.ylabel("Correlation (Manual vs LOF)")
        plt.title("Animal-Level Feature Correlations Between Filtering Methods")
        plt.xticks(range(len(corr_df)), corr_df.index, rotation=45)
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.grid(True, alpha=0.3)

        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, corr_df["p_value"])):
            if p_val < 0.001:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, "***", ha="center", va="bottom")
            elif p_val < 0.01:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, "**", ha="center", va="bottom")
            elif p_val < 0.05:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, "*", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_dir / "animal_correlations.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Save correlation data
        corr_df.to_csv(output_dir / "animal_correlations.csv")

        logging.info("Animal correlation analysis saved")


def generate_summary_statistics(manual_df, lof_df, features, output_dir):
    """Generate summary statistics comparing the two filtering methods"""

    logging.info("Generating summary statistics")

    summary_stats = {
        "feature": [],
        "manual_mean": [],
        "manual_std": [],
        "lof_mean": [],
        "lof_std": [],
        "mean_difference": [],
        "std_difference": [],
        "effect_size": [],
    }

    for feature in features:
        manual_vals = manual_df[feature].dropna()
        lof_vals = lof_df[feature].dropna()

        if len(manual_vals) > 0 and len(lof_vals) > 0:
            manual_mean = manual_vals.mean()
            manual_std = manual_vals.std()
            lof_mean = lof_vals.mean()
            lof_std = lof_vals.std()

            mean_diff = lof_mean - manual_mean
            std_diff = lof_std - manual_std

            # Calculate Cohen's d (effect size)
            pooled_std = np.sqrt(
                ((len(manual_vals) - 1) * manual_std**2 + (len(lof_vals) - 1) * lof_std**2)
                / (len(manual_vals) + len(lof_vals) - 2)
            )
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

            summary_stats["feature"].append(feature)
            summary_stats["manual_mean"].append(manual_mean)
            summary_stats["manual_std"].append(manual_std)
            summary_stats["lof_mean"].append(lof_mean)
            summary_stats["lof_std"].append(lof_std)
            summary_stats["mean_difference"].append(mean_diff)
            summary_stats["std_difference"].append(std_diff)
            summary_stats["effect_size"].append(effect_size)

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)

    logging.info(f"Summary statistics saved for {len(summary_df)} features")


def main():
    """Main comparison analysis function"""
    global snakemake

    # Get parameters from snakemake
    manual_war_paths = snakemake.input.manual_wars
    lof_war_paths = snakemake.input.lof_wars
    config = snakemake.params.config

    # Create output directories
    comparison_dir = Path(snakemake.output.comparison_dir)
    data_dir = Path(snakemake.output.comparison_data)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get comparison parameters
    comparison_config = config["analysis"]["filtering_comparison"]
    features_to_compare = comparison_config["features_to_compare"]
    plot_types = comparison_config["plot_types"]

    logging.info(f"Starting filtering comparison analysis")
    logging.info(f"Manual WARs: {len(manual_war_paths)}")
    logging.info(f"LOF WARs: {len(lof_war_paths)}")
    logging.info(f"Features to compare: {features_to_compare}")
    logging.info(f"Plot types: {plot_types}")

    # Load datasets
    manual_df = load_wars_from_paths(manual_war_paths, "manual")
    lof_df = load_wars_from_paths(lof_war_paths, "lof")

    if manual_df.empty or lof_df.empty:
        logging.error("Failed to load data from one or both filtering methods")
        return

    # Filter to only include requested features that exist in both datasets
    available_features = set(manual_df.columns) & set(lof_df.columns) & set(features_to_compare)
    features_to_analyze = list(available_features)

    if not features_to_analyze:
        logging.error(f"No requested features found in both datasets: {features_to_compare}")
        return

    logging.info(f"Analyzing features: {features_to_analyze}")

    # Generate analyses based on requested plot types
    if "feature_scatter" in plot_types:
        generate_feature_scatter_plots(manual_df, lof_df, features_to_analyze, comparison_dir)

    if "channel_impact" in plot_types:
        generate_channel_impact_analysis(manual_df, lof_df, features_to_analyze, comparison_dir)

    if "animal_correlation" in plot_types:
        generate_animal_correlation_analysis(manual_df, lof_df, features_to_analyze, comparison_dir)

    # Always generate summary statistics
    generate_summary_statistics(manual_df, lof_df, features_to_analyze, data_dir)

    # Save combined dataset for further analysis
    combined_df = pd.concat([manual_df, lof_df], ignore_index=True)
    combined_df.to_csv(data_dir / "combined_filtering_comparison_data.csv", index=False)

    logging.info("Filtering comparison analysis completed successfully")


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
        )
        main()
