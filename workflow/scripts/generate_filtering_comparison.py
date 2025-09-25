#!/usr/bin/env python3
"""
Filtering Comparison Analysis Script
===================================

This script compares manual vs LOF channel filtering approaches by analyzing
differences in extracted features between the two methods.

Input: Flattened WARs from both manual and LOF filtering pipelines
Output: Comparison plots and statistical analysis
"""

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
from pythoneeg import visualization, constants


def load_wars_from_paths(war_paths, label):
    """Load multiple WARs and create ExperimentPlotter instance"""
    wars = []
    path_to_hash_mapping = {}  # Store mapping from file path identifier to hash

    for path in war_paths:
        try:
            war_dir = Path(path).parent
            war = visualization.WindowAnalysisResult.load_pickle_and_json(
                folder_path=war_dir, pickle_name="war.pkl", json_name="war.json"
            )
            war.add_unique_hash()

            # Extract path identifier for matching (use the parent directory name as identifier)
            path_identifier = war_dir.name
            path_to_hash_mapping[path_identifier] = war.animal_id

            wars.append(war)
            logging.info(f"Loaded {label} WAR for {war.animal_id}: {len(war.result)} rows")
            logging.debug(f"Path identifier '{path_identifier}' -> hash '{war.animal_id}'")

        except Exception as e:
            logging.error(f"Failed to load WAR from {path}: {e}")
            continue

    if wars:
        logging.info(f"Creating ExperimentPlotter for {label} dataset: {len(wars)} animals")
        ep = visualization.ExperimentPlotter(wars)
        # Store the path mapping for later use
        ep.path_to_hash_mapping = path_to_hash_mapping
        return ep
    else:
        logging.warning(f"No {label} WARs loaded successfully")
        return None


def create_hash_mapping(manual_ep, lof_ep):
    """
    Create a mapping between manual and LOF hashes using file path identifiers

    Parameters:
    -----------
    manual_ep : ExperimentPlotter
        Manual filtering ExperimentPlotter with path_to_hash_mapping
    lof_ep : ExperimentPlotter
        LOF filtering ExperimentPlotter with path_to_hash_mapping

    Returns:
    --------
    dict
        Dictionary mapping manual_hash -> lof_hash for matching animals
    """
    hash_mapping = {}

    if not hasattr(manual_ep, "path_to_hash_mapping") or not hasattr(lof_ep, "path_to_hash_mapping"):
        logging.warning("Path mappings not available - cannot create hash mapping")
        return hash_mapping

    manual_paths = manual_ep.path_to_hash_mapping
    lof_paths = lof_ep.path_to_hash_mapping

    # Find common path identifiers
    common_paths = set(manual_paths.keys()).intersection(set(lof_paths.keys()))
    logging.info(f"Found {len(common_paths)} common path identifiers for hash mapping")

    for path_id in common_paths:
        manual_hash = manual_paths[path_id]
        lof_hash = lof_paths[path_id]
        hash_mapping[manual_hash] = lof_hash
        logging.debug(f"Mapped: {manual_hash} -> {lof_hash} (path: {path_id})")

    logging.info(f"Created hash mapping for {len(hash_mapping)} animals")
    return hash_mapping


def calculate_feature_correlation(manual_vals, lof_vals, feature_name, hash_mapping=None):
    """
    Calculate correlation between manual and LOF feature values, handling hash mapping

    Parameters:
    -----------
    manual_vals : pd.Series
        Manual filtering feature values indexed by animal ID
    lof_vals : pd.Series
        LOF filtering feature values indexed by animal ID
    feature_name : str
        Name of the feature for logging
    hash_mapping : dict, optional
        Mapping from manual animal IDs to LOF animal IDs

    Returns:
    --------
    dict or None
        Dictionary with correlation, p_value, and n_animals if successful, None otherwise
    """
    try:
        if hash_mapping:
            # Use hash mapping to match animals
            manual_animals_with_mapping = [animal for animal in manual_vals.index if animal in hash_mapping]
            matched_manual_animals = [animal for animal in manual_animals_with_mapping
                                    if hash_mapping[animal] in lof_vals.index]

            if len(matched_manual_animals) == 0:
                logging.warning(f"No animals could be matched via hash mapping for feature {feature_name}")
                return None

            # Get values for matched animals
            manual_common = manual_vals.loc[matched_manual_animals]
            lof_matched_animals = [hash_mapping[animal] for animal in matched_manual_animals]
            lof_common = lof_vals.loc[lof_matched_animals]

            logging.debug(f"Feature {feature_name}: {len(matched_manual_animals)} animals matched via hash mapping")
        else:
            # Direct matching by animal ID (fallback)
            common_animals = manual_vals.index.intersection(lof_vals.index)
            if len(common_animals) == 0:
                logging.warning(f"No common animals found for feature {feature_name}")
                return None

            manual_common = manual_vals.loc[common_animals]
            lof_common = lof_vals.loc[common_animals]

            logging.debug(f"Feature {feature_name}: {len(common_animals)} common animals found")

        # Remove infinite and NaN values - use .values to avoid index alignment issues
        valid_mask = np.isfinite(manual_common.values) & np.isfinite(lof_common.values)
        n_valid = valid_mask.sum()

        if n_valid > 1:
            corr, p_val = stats.pearsonr(manual_common.values[valid_mask], lof_common.values[valid_mask])
            return {"correlation": corr, "p_value": p_val, "n_animals": n_valid}
        else:
            logging.warning(f"Insufficient valid data for correlation: {feature_name} (n_valid={n_valid})")
            return None

    except Exception as e:
        logging.error(f"Error calculating correlation for {feature_name}: {e}")
        return None


def extract_feature_dataframe(ep, feature, label):
    """
    Extract feature data using ExperimentPlotter's built-in methods with proper animal-level aggregation

    Parameters:
    -----------
    ep : ExperimentPlotter
        PyEEG ExperimentPlotter instance containing the WAR data
    feature : str
        Feature name to extract (e.g., 'logpsdband', 'rms', 'cohere')
    label : str
        Label for filtering method ('manual' or 'lof')

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with animal-level aggregated feature data, or None if extraction fails
        - For LINEAR_FEATURES (rms, ampvar, etc.): One row per animal
        - For BAND_FEATURES (logpsdband, psdfrac, etc.): Multiple rows per animal (one per frequency band)
        - For MATRIX_FEATURES (cohere, pcorr, etc.): Multiple rows per animal (one per frequency band for spectral features)
    """
    try:
        # Use animal-level aggregation across channels and time periods
        # This gives us ~48-62 data points (one per animal) for linear features
        # For band/matrix features, we get multiple rows per animal (one per frequency band)
        df = ep.pull_timeseries_dataframe(
            feature=feature,
            groupby=["animal"],
            collapse_channels=True,
            average_groupby=True,  # Average across time periods for each animal
        )

        # Add filtering method label
        df["filtering_method"] = label

        # Add genotype information if available
        if hasattr(ep, "results") and ep.results:
            # Create a mapping from animal to genotype
            animal_to_genotype = {}
            for war in ep.results:
                animal_to_genotype[war.animal_id] = war.genotype
            logging.info(f"Retrieved genotype information for {len(animal_to_genotype)} animals")

            # Add genotype column to the dataframe
            if animal_to_genotype:
                df["genotype"] = df["animal"].map(animal_to_genotype)
                logging.info(f"Added genotype information for {len(animal_to_genotype)} animals")

        # Log the result structure
        if "band" in df.columns:
            bands = sorted(df["band"].unique())
            animals = df["animal"].nunique()
            logging.info(
                f"Extracted {label} feature '{feature}': {len(df)} records ({animals} animals × {len(bands)} bands)"
            )
        else:
            animals = df["animal"].nunique()
            logging.info(f"Extracted {label} feature '{feature}': {len(df)} records ({animals} animals)")

        return df

    except Exception as e:
        logging.error(f"Failed to extract feature '{feature}' from {label} data: {e}")
        return None


def generate_feature_scatter_plots(manual_ep, lof_ep, features, output_dir, hash_mapping=None):
    """
    Generate individual scatter plots comparing features between manual and LOF filtering methods

    Creates separate plots for each frequency band of band/matrix features, enabling clear comparison
    of specific frequency components between the two filtering approaches.

    Parameters:
    -----------
    manual_ep : ExperimentPlotter
        PyEEG ExperimentPlotter instance with manually filtered data
    lof_ep : ExperimentPlotter
        PyEEG ExperimentPlotter instance with LOF-filtered data
    features : list[str]
        List of feature names to compare (e.g., ['logpsdband', 'rms', 'cohere'])
    output_dir : Path
        Directory to save scatter plot PNG files

    Output Files:
    -------------
    For LINEAR_FEATURES (rms, ampvar, psdtotal, etc.):
        - scatter_{feature}.png (e.g., scatter_rms.png)
        - One plot per feature, ~48-62 data points (one per animal)

    For BAND_FEATURES (logpsdband, psdband, psdfrac):
        - scatter_{feature}_{band}.png (e.g., scatter_logpsdband_delta.png)
        - Five plots per feature (delta, theta, alpha, beta, gamma)
        - Each plot: ~48-62 data points (one per animal for that frequency band)
        - Delta: 1-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-25 Hz, Gamma: 25-40 Hz

    For MATRIX_FEATURES with bands (cohere, imcoh, zcohere, zimcoh):
        - scatter_{feature}_{band}.png (e.g., scatter_cohere_theta.png)
        - Five plots per feature (delta, theta, alpha, beta, gamma)
        - Each plot: ~48-62 data points (one per animal for that frequency band)
        - Represents coherence/imaginary coherence averaged across channel pairs in that band

    For MATRIX_FEATURES without bands (pcorr, zpcorr):
        - scatter_{feature}.png (e.g., scatter_pcorr.png)
        - One plot per feature, ~48-62 data points (one per animal)
        - Represents correlation averaged across channel pairs and time

    Plot Content:
    -------------
    - X-axis: Manual filtering values
    - Y-axis: LOF filtering values
    - Red dashed line: Perfect correlation (y=x)
    - Text box: Pearson correlation coefficient and RMSE
    - Each data point: One animal's aggregated value for that feature/band combination
    """

    logging.info("Generating feature scatter plots using PyEEG ExperimentPlotter methods")

    # Use the provided hash mapping for merging manual and LOF data
    if hash_mapping is None:
        hash_mapping = create_hash_mapping(manual_ep, lof_ep)

    try:
        # Extract data and organize plots (separate plots for each frequency band)
        logging.info("Extracting and organizing feature data for individual band plotting")
        plots_to_create = {}  # Will store plot_name -> {"manual": df, "lof": df, "feature_col": str, "title": str}

        for i, feature in enumerate(features):
            logging.debug(f"Processing feature {i + 1}/{len(features)}: {feature}")

            # Extract manual data
            manual_df = extract_feature_dataframe(manual_ep, feature, "manual")
            if manual_df is None:
                logging.warning(f"Failed to extract manual data for {feature}, skipping")
                continue

            # Extract LOF data
            lof_df = extract_feature_dataframe(lof_ep, feature, "lof")
            if lof_df is None:
                logging.warning(f"Failed to extract LOF data for {feature}, skipping")
                continue

            # Check if this is a band feature (has 'band' column)
            if "band" in manual_df.columns and "band" in lof_df.columns:
                # Band feature - create separate plots for each band
                bands = sorted(manual_df["band"].unique())
                logging.debug(f"Feature '{feature}' is a band feature with bands: {bands}")

                for band in bands:
                    # Filter data for this specific band
                    manual_band = manual_df[manual_df["band"] == band].copy()
                    lof_band = lof_df[lof_df["band"] == band].copy()

                    # Create unique plot name for this band
                    plot_name = f"{feature}_{band}"
                    freq_range = constants.FREQ_BANDS.get(band, "Unknown")
                    freq_str = (
                        f"{freq_range[0]}-{freq_range[1]} Hz" if isinstance(freq_range, tuple) else str(freq_range)
                    )

                    plots_to_create[plot_name] = {
                        "manual": manual_band,
                        "lof": lof_band,
                        "feature_col": feature,  # The actual column name to plot
                        "title": f"{feature} - {band.capitalize()} Band ({freq_str})",
                    }
                    logging.debug(
                        f"Prepared plot '{plot_name}' with {len(manual_band)} manual and {len(lof_band)} LOF records"
                    )
            else:
                # Linear feature - single plot
                plot_name = feature
                plots_to_create[plot_name] = {
                    "manual": manual_df,
                    "lof": lof_df,
                    "feature_col": feature,
                    "title": f"{feature} Comparison",
                }
                logging.debug(f"Prepared plot '{plot_name}' with {len(manual_df)} manual and {len(lof_df)} LOF records")

        if not plots_to_create:
            logging.error("No plots could be prepared")
            return

        logging.info(f"Successfully prepared {len(plots_to_create)} plots: {list(plots_to_create.keys())}")

        # Create individual scatter plots
        logging.info("Creating individual scatter plots")

        for i, (plot_name, plot_data) in enumerate(plots_to_create.items()):
            logging.debug(f"Creating plot {i + 1}/{len(plots_to_create)}: {plot_name}")

            # Create individual figure for this plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.set_box_aspect(1)

            try:
                manual_df = plot_data["manual"]
                lof_df = plot_data["lof"]
                feature_col = plot_data["feature_col"]
                plot_title = plot_data["title"]

                # Data is already animal-level aggregated by PyEEG
                # For band features, we may have multiple rows per animal (one per band)
                # For linear features, we have one row per animal

                # For band-separated plots, we only have data for one specific band now
                # Use hash mapping to merge manual and LOF data
                logging.debug(
                    f"Hash mapping available: {hash_mapping is not None}, size: {len(hash_mapping) if hash_mapping else 0}"
                )
                if hash_mapping:
                    # Create a reverse mapping for LOF data
                    lof_to_manual_mapping = {v: k for k, v in hash_mapping.items()}

                    # Add mapping columns to enable merging
                    manual_df_temp = manual_df.copy()
                    lof_df_temp = lof_df.copy()

                    # Map LOF hashes to manual hashes for merging
                    lof_df_temp["manual_hash"] = lof_df_temp["animal"].map(lof_to_manual_mapping)
                    manual_df_temp["manual_hash"] = manual_df_temp["animal"]

                    # Check for mapping failures
                    manual_na_count = manual_df_temp["manual_hash"].isna().sum()
                    lof_na_count = lof_df_temp["manual_hash"].isna().sum()
                    if manual_na_count > 0 or lof_na_count > 0:
                        logging.warning(
                            f"Hash mapping failures - Manual NAs: {manual_na_count}, LOF NAs: {lof_na_count}"
                        )

                    # Merge on the mapped manual hash
                    merged = pd.merge(
                        manual_df_temp, lof_df_temp, on=["manual_hash"], suffixes=("_manual", "_lof"), how="inner"
                    )
                    logging.debug(f"Hash-based merge for {plot_name}: {len(merged)} records")
                else:
                    logging.error(f"Hash mapping not available for {plot_name}")
                    raise ValueError("Hash mapping not available")

                if merged.empty:
                    logging.warning(f"No matching records for {plot_name}")
                    ax.text(0.5, 0.5, "No matching data", transform=ax.transAxes, ha="center")
                    ax.set_title(f"{plot_name} (No Data)")
                    continue

                # Extract values for plotting using the correct feature column name
                manual_vals = merged[f"{feature_col}_manual"]
                lof_vals = merged[f"{feature_col}_lof"]

                # Remove infinite and NaN values
                valid_mask = np.isfinite(manual_vals) & np.isfinite(lof_vals)
                manual_clean = manual_vals[valid_mask]
                lof_clean = lof_vals[valid_mask]

                # Log data point information
                # Use manual_hash column to count unique animals since 'animal' column is now split
                if "manual_hash" in merged.columns:
                    unique_animals = merged["manual_hash"].nunique()
                else:
                    unique_animals = len(merged)
                logging.debug(f"Plot {plot_name}: {len(manual_clean)} valid data points ({unique_animals} animals)")

                if len(manual_clean) > 0:
                    # Create scatter plot with genotype-based coloring
                    if "genotype" in merged.columns:
                        # Get unique genotypes and create color mapping
                        unique_genotypes = sorted(merged["genotype"].dropna().unique())
                        genotype_colors = {gt: f"C{i}" for i, gt in enumerate(unique_genotypes)}

                        # Plot each genotype with different colors
                        for genotype in unique_genotypes:
                            genotype_mask = merged["genotype"] == genotype
                            valid_genotype_mask = valid_mask & genotype_mask

                            if valid_genotype_mask.sum() > 0:
                                manual_genotype = manual_vals[valid_genotype_mask]
                                lof_genotype = lof_vals[valid_genotype_mask]
                                ax.scatter(
                                    manual_genotype,
                                    lof_genotype,
                                    alpha=0.6,
                                    s=20,
                                    c=genotype_colors[genotype],
                                    label=f"{genotype} (n={len(manual_genotype)})",
                                )
                    else:
                        # No genotype information available, use default color
                        ax.scatter(manual_clean, lof_clean, alpha=0.6, s=20, c="C0")

                    # Add diagonal line (perfect correlation)
                    min_val = min(manual_clean.min(), lof_clean.min())
                    max_val = max(manual_clean.max(), lof_clean.max())
                    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="Perfect correlation")

                    # Calculate statistics
                    if len(manual_clean) > 1:
                        correlation = stats.pearsonr(manual_clean, lof_clean)[0]
                        rmse = np.sqrt(mean_squared_error(manual_clean, lof_clean))
                        logging.debug(f"Plot {plot_name}: correlation={correlation:.3f}, RMSE={rmse:.3f}")

                        ax.text(
                            0.05,
                            0.95,
                            f"r = {correlation:.3f}\nRMSE = {rmse:.3f}",
                            transform=ax.transAxes,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                        )

                    ax.set_xlabel(f"{plot_title} (Manual Filtering)")
                    ax.set_ylabel(f"{plot_title} (LOF Filtering)")
                    ax.set_title(plot_title)
                    ax.legend()

                else:
                    ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center")
                    ax.set_title(f"{plot_name} (No Data)")

            except Exception as e:
                logging.error(f"Error processing plot {plot_name}: {str(e)}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes, ha="center")
                ax.set_title(f"{plot_name} (Error)")

            # Save individual plot
            fig.tight_layout()
            output_filename = f"scatter_{plot_name}.png"
            fig.savefig(output_dir / output_filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logging.debug(f"Saved scatter plot: {output_filename}")

        logging.info("All scatter plots saved successfully")

    except Exception as e:
        logging.error(f"Critical error in generate_feature_scatter_plots: {str(e)}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


def generate_channel_impact_analysis(manual_ep, lof_ep, features, output_dir, hash_mapping=None):
    """Analyze the impact of filtering on different channels using PyEEG ExperimentPlotter methods"""

    logging.info("Generating channel impact analysis using PyEEG ExperimentPlotter methods")

    try:
        logging.info("Step 1: Extracting feature data for comparison analysis")

        # We'll use band-wise analysis as it provides the most meaningful comparison
        group_col = "band"
        group_label = "Band"

        manual_feature_data = {}
        lof_feature_data = {}

        # Extract data for each feature using PyEEG's proper methods
        for i, feature in enumerate(features):
            logging.info(f"Processing feature {i + 1}/{len(features)}: {feature}")

            # Extract manual data
            manual_df = extract_feature_dataframe(manual_ep, feature, "manual")
            if manual_df is None:
                logging.warning(f"Failed to extract manual data for {feature}, skipping")
                continue

            # Extract LOF data
            lof_df = extract_feature_dataframe(lof_ep, feature, "lof")
            if lof_df is None:
                logging.warning(f"Failed to extract LOF data for {feature}, skipping")
                continue

            manual_feature_data[feature] = manual_df
            lof_feature_data[feature] = lof_df

        if not manual_feature_data or not lof_feature_data:
            logging.error("No feature data could be extracted for channel impact analysis")
            return

        logging.info(f"Successfully extracted data for {len(manual_feature_data)} features")

        # Determine grouping based on available columns in the extracted data
        sample_df = next(iter(manual_feature_data.values()))
        available_cols = sample_df.columns.tolist()
        logging.info(f"Available columns in extracted data: {available_cols}")

        if "band" in available_cols:
            group_col = "band"
            group_label = "Band"
        elif "isday" in available_cols:
            group_col = "isday"
            group_label = "DayNight"
        elif "genotype" in available_cols:
            group_col = "genotype"
            group_label = "Genotype"
        else:
            logging.warning("No suitable grouping column found (neither 'band', 'isday', nor 'genotype')")
            return

        logging.info(f"Using '{group_col}' for grouping analysis")

        # Build aggregated data for each feature
        manual_grouped_data = {}
        lof_grouped_data = {}

        for feature in manual_feature_data.keys():
            logging.info(f"Processing feature '{feature}' for {group_label.lower()}-wise analysis")

            manual_df = manual_feature_data[feature]
            lof_df = lof_feature_data[feature]

            # Process features: create per-band entries for band features
            if "band" in manual_df.columns and "band" in lof_df.columns:
                # Band feature - create separate entries for each band
                for band in manual_df['band'].unique():
                    if band in lof_df['band'].unique():
                        manual_band_data = manual_df[manual_df['band'] == band]
                        lof_band_data = lof_df[lof_df['band'] == band]

                        # Group by the grouping column within this band
                        manual_grouped = manual_band_data.groupby(group_col)[feature].mean()
                        lof_grouped = lof_band_data.groupby(group_col)[feature].mean()

                        manual_grouped_data[f"{feature}_{band}"] = manual_grouped
                        lof_grouped_data[f"{feature}_{band}"] = lof_grouped
            else:
                # Linear feature - group directly by grouping column
                if group_col in manual_df.columns and group_col in lof_df.columns:
                    manual_grouped = manual_df.groupby(group_col)[feature].mean()
                    lof_grouped = lof_df.groupby(group_col)[feature].mean()
                    manual_grouped_data[feature] = manual_grouped
                    lof_grouped_data[feature] = lof_grouped
                else:
                    logging.warning(f"Feature '{feature}' missing '{group_col}' column - skipping")
                    continue

            # Count what we added
            current_keys = [k for k in manual_grouped_data.keys() if k.startswith(feature)]
            logging.info(f"Feature '{feature}': created {len(current_keys)} analysis groups")

        if not manual_grouped_data or not lof_grouped_data:
            logging.error(f"No features could be grouped by {group_col}")
            return

        logging.info(f"Successfully grouped {len(manual_grouped_data)} features")

        # Create combined grouped dataframes
        logging.info("Step 2: Creating combined grouped dataframes")
        manual_grouped_df = pd.DataFrame(manual_grouped_data)
        lof_grouped_df = pd.DataFrame(lof_grouped_data)

        logging.info(f"Manual grouped data shape: {manual_grouped_df.shape}")
        logging.info(f"LOF grouped data shape: {lof_grouped_df.shape}")

        # Find common groups
        logging.info(f"Step 3: Finding common {group_label.lower()} categories")
        common_groups = manual_grouped_df.index.intersection(lof_grouped_df.index)
        logging.info(f"Found {len(common_groups)} common {group_label.lower()} categories: {list(common_groups)}")

        if len(common_groups) == 0:
            logging.warning(f"No common {group_label.lower()} categories found between filtering methods")
            return

        logging.info(f"Step 4: Calculating {group_label.lower()} differences")
        group_diff = lof_grouped_df.loc[common_groups] - manual_grouped_df.loc[common_groups]
        logging.info(f"{group_label} difference matrix shape: {group_diff.shape}")

        logging.info("Step 5: Creating three versions of the heatmap")

        # Create three versions: absolute differences, z-score-like, and log2 fold-change
        from matplotlib.colors import TwoSlopeNorm

        versions = [
            ("absolute", "Absolute Differences (LOF - Manual)", group_diff),
            ("zscore", "Z-score-like Differences", None),  # Will calculate below
            ("log2fc", "Log2 Fold-Change log2(LOF/Manual)", None)  # Will calculate below
        ]

        # Calculate z-score-like version (divide by std of differences)
        group_diff_zscore = group_diff / group_diff.std()
        versions[1] = ("zscore", "Z-score-like Differences", group_diff_zscore)

        # Calculate log2 fold-change version
        # Need original values, not differences, for fold-change calculation
        # log2(LOF/Manual) = log2(LOF) - log2(Manual)
        manual_vals = manual_grouped_df.loc[common_groups]
        lof_vals = lof_grouped_df.loc[common_groups]

        # Avoid log of zero or negative values by adding small epsilon
        epsilon = 1e-10
        manual_safe = np.maximum(manual_vals, epsilon)
        lof_safe = np.maximum(lof_vals, epsilon)

        group_diff_log2fc = np.log2(lof_safe) - np.log2(manual_safe)
        versions[2] = ("log2fc", "Log2 Fold-Change log2(LOF/Manual)", group_diff_log2fc)

        for version_name, version_title, version_data in versions:
            if version_data is None:
                continue

            logging.info(f"Creating {version_name} heatmap")

            plt.figure(figsize=(12, max(8, len(version_data.columns) * 0.6)))

            # Calculate symmetric limits for proper centering
            abs_max = np.abs(version_data.values).max()
            vmin, vmax = -abs_max, abs_max

            # Use TwoSlopeNorm for proper centering at zero with red-blue colormap
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            sns.heatmap(version_data.T, annot=True, cmap="RdBu_r", norm=norm, fmt=".3f",
                       cbar_kws={"label": f"{version_title}"})

            plt.title(f"{group_label}-wise Feature {version_title}")
            plt.xlabel(group_label)
            plt.ylabel("Feature")
            plt.tight_layout()

            # Save heatmap
            heatmap_filename = f"{group_label.lower()}_impact_heatmap_{version_name}.png"
            plt.savefig(output_dir / heatmap_filename, dpi=300, bbox_inches="tight")
            plt.close()

            # Save data
            csv_filename = f"{group_label.lower()}_impact_differences_{version_name}.csv"
            version_data.to_csv(output_dir / csv_filename)

        logging.info(f"{group_label} impact analysis saved successfully")

    except Exception as e:
        logging.error(f"Critical error in generate_channel_impact_analysis: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


def generate_animal_correlation_analysis(manual_ep, lof_ep, features, output_dir, hash_mapping=None):
    """Analyze correlation of animal-level aggregated features using PyEEG ExperimentPlotter methods"""

    logging.info("Generating animal-level correlation analysis using PyEEG ExperimentPlotter methods")

    try:
        logging.info("Step 1: Extracting feature data for correlation analysis")

        manual_feature_data = {}
        lof_feature_data = {}

        # Extract data for each feature using PyEEG's proper methods
        for i, feature in enumerate(features):
            logging.info(f"Processing feature {i + 1}/{len(features)}: {feature}")

            # Extract manual data
            manual_df = extract_feature_dataframe(manual_ep, feature, "manual")
            if manual_df is None:
                logging.warning(f"Failed to extract manual data for {feature}, skipping")
                continue

            # Extract LOF data
            lof_df = extract_feature_dataframe(lof_ep, feature, "lof")
            if lof_df is None:
                logging.warning(f"Failed to extract LOF data for {feature}, skipping")
                continue

            manual_feature_data[feature] = manual_df
            lof_feature_data[feature] = lof_df

        if not manual_feature_data or not lof_feature_data:
            logging.error("No feature data could be extracted for correlation analysis")
            return

        logging.info(f"Successfully extracted data for {len(manual_feature_data)} features")

        logging.info("Aggregating features by animal for correlation analysis (per-band for band features)")
        manual_animal_data = {}
        lof_animal_data = {}

        for feature in manual_feature_data.keys():
            logging.debug(f"Processing feature '{feature}' for animal-level correlation analysis")

            manual_df = manual_feature_data[feature]
            lof_df = lof_feature_data[feature]

            # Check if animal column exists
            if "animal" not in manual_df.columns or "animal" not in lof_df.columns:
                logging.warning(f"Missing 'animal' column for feature {feature}, skipping")
                continue

            # For band features, create separate correlations for each band
            if "band" in manual_df.columns and "band" in lof_df.columns:
                for band in manual_df['band'].unique():
                    if band in lof_df['band'].unique():
                        manual_band_data = manual_df[manual_df['band'] == band]
                        lof_band_data = lof_df[lof_df['band'] == band]

                        manual_animal = manual_band_data.set_index("animal")[feature]
                        lof_animal = lof_band_data.set_index("animal")[feature]

                        manual_animal_data[f"{feature}_{band}"] = manual_animal
                        lof_animal_data[f"{feature}_{band}"] = lof_animal

                        logging.info(f"Feature '{feature}_{band}': manual animals={len(manual_animal)}, lof animals={len(lof_animal)}")
            else:
                # Linear features already have one value per animal
                manual_animal = manual_df.set_index("animal")[feature]
                lof_animal = lof_df.set_index("animal")[feature]

                manual_animal_data[feature] = manual_animal
                lof_animal_data[feature] = lof_animal

                logging.info(f"Feature '{feature}': manual animals={len(manual_animal)}, lof animals={len(lof_animal)}")

        if not manual_animal_data or not lof_animal_data:
            logging.error("No features could be aggregated by animal")
            return

        logging.info(f"Successfully aggregated {len(manual_animal_data)} features by animal")

        logging.info("Step 3: Calculating correlations for each feature using hash mapping")
        correlations = {}

        if not hash_mapping:
            logging.error("Hash mapping not available for correlation analysis")
            return

        for i, feature in enumerate(manual_animal_data.keys()):
            logging.info(f"Calculating correlation for feature {i + 1}/{len(manual_animal_data)}: {feature}")

            manual_vals = manual_animal_data[feature]
            lof_vals = lof_animal_data[feature]

            # Use the shared correlation calculation function
            correlation_result = calculate_feature_correlation(manual_vals, lof_vals, feature, hash_mapping)

            if correlation_result:
                correlations[feature] = correlation_result
                logging.info(f"Feature {feature}: correlation={correlation_result['correlation']:.3f}, p_value={correlation_result['p_value']:.6f}, n_animals={correlation_result['n_animals']}")
            else:
                logging.warning(f"Could not calculate correlation for feature {feature}")

        logging.info(f"Calculated correlations for {len(correlations)} features")

        # Create correlation summary plot
        if correlations:
            logging.info("Step 4: Creating correlation dataframe")
            corr_df = pd.DataFrame(correlations).T
            logging.info(f"Correlation dataframe shape: {corr_df.shape}")

            logging.info("Step 5: Creating figure for correlation plot")
            plt.figure(figsize=(10, 6))

            logging.info("Step 6: Creating bar plot")
            bars = plt.bar(
                range(len(corr_df)),
                corr_df["correlation"],
                color=["green" if p < 0.05 else "orange" for p in corr_df["p_value"]],
            )

            logging.info("Step 7: Setting plot labels and formatting")
            plt.xlabel("Feature")
            plt.ylabel("Correlation (Manual vs LOF)")
            plt.title("Animal-Level Feature Correlations Between Filtering Methods")
            plt.xticks(range(len(corr_df)), corr_df.index, rotation=45)
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.grid(True, alpha=0.3)

            logging.info("Step 8: Adding significance indicators")
            for i, (bar, p_val) in enumerate(zip(bars, corr_df["p_value"])):
                if p_val < 0.001:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, "***", ha="center", va="bottom"
                    )
                elif p_val < 0.01:
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, "**", ha="center", va="bottom")
                elif p_val < 0.05:
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, "*", ha="center", va="bottom")

            logging.info("Step 9: Applying tight layout")
            plt.tight_layout()

            logging.info("Step 10: Saving correlation plot")
            plt.savefig(output_dir / "animal_correlations.png", dpi=300, bbox_inches="tight")

            logging.info("Step 11: Closing figure")
            plt.close()

            logging.info("Step 12: Saving correlation data to CSV")
            corr_df.to_csv(output_dir / "animal_correlations.csv")

            logging.info("Animal correlation analysis saved successfully")

    except Exception as e:
        logging.error(f"Critical error in generate_animal_correlation_analysis: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


def generate_summary_statistics(manual_ep, lof_ep, features, output_dir, hash_mapping=None):
    """Generate summary statistics comparing the two filtering methods using PyEEG ExperimentPlotter methods"""

    logging.info("Generating summary statistics using PyEEG ExperimentPlotter methods")

    try:
        logging.info("Step 1: Extracting feature data for summary statistics")
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

        # Extract data for each feature using PyEEG's proper methods
        for i, feature in enumerate(features):
            logging.info(f"Processing feature {i + 1}/{len(features)}: {feature}")

            # Extract manual data
            manual_df = extract_feature_dataframe(manual_ep, feature, "manual")
            if manual_df is None:
                logging.warning(f"Failed to extract manual data for {feature}, skipping")
                continue

            # Extract LOF data
            lof_df = extract_feature_dataframe(lof_ep, feature, "lof")
            if lof_df is None:
                logging.warning(f"Failed to extract LOF data for {feature}, skipping")
                continue

            # Get feature values (already properly extracted by PyEEG methods)
            manual_vals = manual_df[feature].dropna()
            lof_vals = lof_df[feature].dropna()
            logging.info(f"Feature {feature}: manual={len(manual_vals)} values, lof={len(lof_vals)} values")

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

                logging.info(
                    f"Feature {feature}: manual_mean={manual_mean:.3f}, lof_mean={lof_mean:.3f}, effect_size={effect_size:.3f}"
                )

        logging.info("Step 2: Creating summary dataframe")
        summary_df = pd.DataFrame(summary_stats)
        logging.info(f"Summary dataframe shape: {summary_df.shape}")

        if summary_df.empty:
            logging.error("No summary statistics could be generated")
            return

        logging.info("Step 3: Saving summary statistics to CSV")
        output_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(output_path, index=False)
        logging.info(f"Summary statistics saved for {len(summary_df)} features successfully")

    except Exception as e:
        logging.error(f"Critical error in generate_summary_statistics: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


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

    logging.info("Starting filtering comparison analysis")
    logging.info(f"Manual WARs: {len(manual_war_paths)}")
    logging.info(f"LOF WARs: {len(lof_war_paths)}")
    logging.info(f"Features to compare: {features_to_compare}")
    logging.info(f"Plot types: {plot_types}")

    # Load datasets as ExperimentPlotter instances
    manual_ep = load_wars_from_paths(manual_war_paths, "manual")
    lof_ep = load_wars_from_paths(lof_war_paths, "lof")

    if manual_ep is None or lof_ep is None:
        logging.error("Failed to load data from one or both filtering methods")
        return

    # Use features_to_compare directly since PyEEG methods handle feature validation
    features_to_analyze = features_to_compare
    logging.info(f"Analyzing features: {features_to_analyze}")

    # Create hash mapping for merging manual and LOF data
    hash_mapping = create_hash_mapping(manual_ep, lof_ep)
    logging.info(f"Main function hash mapping created: {len(hash_mapping) if hash_mapping else 0} mappings")

    # Generate analyses based on requested plot types
    logging.info("Starting plot generation phase")

    try:
        if "feature_scatter" in plot_types:
            logging.info("Calling generate_feature_scatter_plots...")
            generate_feature_scatter_plots(manual_ep, lof_ep, features_to_analyze, comparison_dir, hash_mapping)
            logging.info("generate_feature_scatter_plots completed successfully")

        if "channel_impact" in plot_types:
            logging.info("Calling generate_channel_impact_analysis...")
            generate_channel_impact_analysis(manual_ep, lof_ep, features_to_analyze, comparison_dir, hash_mapping)
            logging.info("generate_channel_impact_analysis completed successfully")

        if "animal_correlation" in plot_types:
            logging.info("Calling generate_animal_correlation_analysis...")
            generate_animal_correlation_analysis(manual_ep, lof_ep, features_to_analyze, comparison_dir, hash_mapping)
            logging.info("generate_animal_correlation_analysis completed successfully")

        # Always generate summary statistics
        logging.info("Calling generate_summary_statistics...")
        generate_summary_statistics(manual_ep, lof_ep, features_to_analyze, data_dir, hash_mapping)
        logging.info("generate_summary_statistics completed successfully")

        logging.info("Filtering comparison analysis completed successfully")

    except Exception as e:
        logging.error(f"Critical error in main analysis phase: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    # Check if running under Snakemake (snakemake object will be injected)
    if "snakemake" in globals():
        # Access snakemake safely since we know it exists
        snakemake_obj = globals()["snakemake"]
        with open(snakemake_obj.log[0], "w") as f:
            sys.stderr = sys.stdout = f
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
            )
            main()
    else:
        # Running standalone - just setup basic logging
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
        print("This script is designed to be run by Snakemake.")
        print("The 'snakemake' object with input/output paths is not available in standalone mode.")
