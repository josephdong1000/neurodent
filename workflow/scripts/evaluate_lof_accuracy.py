#!/usr/bin/env python3
"""
LOF Accuracy Evaluation Script
==============================

This script evaluates the accuracy of LOF (Local Outlier Factor) bad channel
detection by comparing automated predictions against ground truth annotations
from samples.json across all flattened WARs.

Input: All flattened WARs + samples.json configuration
Output: F-score vs threshold analysis with CSV results and plot
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization, core

# Logging setup moved to if __name__ == "__main__" block


def get_ground_truth_bad_channels(samples_config, animal_folder, animal_id):
    """Extract ground truth bad channels for a specific animal from samples.json"""
    animal_key = f"{animal_folder} {animal_id}"
    samples_bad_channels = samples_config.get("bad_channels", {})
    bad_channels_dict = samples_bad_channels.get(animal_key, {})

    # Convert to set format for each animal-day
    # Use the full animalday_session key as-is since it matches LOF scores format
    ground_truth = {}
    for animalday_session, bad_channels_list in bad_channels_dict.items():
        ground_truth[animalday_session] = set(bad_channels_list)

    return ground_truth


def evaluate_lof_accuracy_across_animals(
    flattened_wars, samples_config, animal_folder_map, animal_id_map, thresholds, evaluation_channels
):
    """Evaluate LOF accuracy across all animals and thresholds"""

    results = []

    for threshold in thresholds:
        logging.info(f"Evaluating threshold: {threshold:.2f}")

        # Collect all y_true and y_pred across all animals
        all_y_true = []
        all_y_pred = []

        for animal_name, (pkl_file, json_file) in flattened_wars.items():
            try:
                # Load the flattened WAR
                war = visualization.WindowAnalysisResult.load_pickle_and_json(
                    folder_path=Path(pkl_file).parent, pickle_name=Path(pkl_file).name, json_name=Path(json_file).name
                )

                # Get animal folder and ID
                animal_folder = animal_folder_map[animal_name]
                animal_id = animal_id_map[animal_name]

                # Extract ground truth bad channels
                ground_truth = get_ground_truth_bad_channels(samples_config, animal_folder, animal_id)

                # Debug: Log before evaluation
                logging.debug(
                    f"Animal {animal_name}: ground_truth keys = {list(ground_truth.keys()) if ground_truth else 'None'}"
                )
                logging.debug(
                    f"Animal {animal_name}: has lof_scores_dict = {hasattr(war, 'lof_scores_dict') and bool(war.lof_scores_dict)}"
                )
                if hasattr(war, "lof_scores_dict") and war.lof_scores_dict:
                    logging.debug(f"Animal {animal_name}: lof_scores_dict keys = {list(war.lof_scores_dict.keys())}")

                # Evaluate this threshold for this animal
                # Use ground truth from samples.json, but could fall back to war.bad_channels_dict
                y_true, y_pred = war.evaluate_lof_threshold_binary(
                    ground_truth_bad_channels=ground_truth if ground_truth else None,
                    threshold=threshold,
                    evaluation_channels=evaluation_channels,
                )

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

                logging.debug(f"Animal {animal_name}: {len(y_true)} evaluation points returned")
                if len(y_true) == 0:
                    logging.warning(f"Animal {animal_name}: No evaluation points returned for threshold {threshold}")

            except Exception as e:
                logging.error(f"Failed to evaluate {animal_name}: {str(e)}")
                raise

        # Debug: Log what we collected for this threshold
        logging.debug(
            f"Threshold {threshold:.2f}: Collected {len(all_y_true)} total evaluation points from {len([name for name, _ in flattened_wars.items()])} animals"
        )

        # Calculate metrics across all animals
        if len(all_y_true) > 0:
            # Handle edge case where all predictions are the same
            if len(set(all_y_pred)) == 1:
                f1 = 0.0 if all_y_pred[0] != all_y_true[0] else 1.0
                precision = 0.0 if all_y_pred[0] != all_y_true[0] else 1.0
                recall = 0.0 if all_y_pred[0] != all_y_true[0] else 1.0
            else:
                f1 = f1_score(all_y_true, all_y_pred, average="binary", zero_division=0)
                precision = precision_score(all_y_true, all_y_pred, average="binary", zero_division=0)
                recall = recall_score(all_y_true, all_y_pred, average="binary", zero_division=0)

            results.append(
                {
                    "threshold": threshold,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "total_channels": len(all_y_true),
                    "true_positives": sum(1 for t, p in zip(all_y_true, all_y_pred) if t == 1 and p == 1),
                    "false_positives": sum(1 for t, p in zip(all_y_true, all_y_pred) if t == 0 and p == 1),
                    "false_negatives": sum(1 for t, p in zip(all_y_true, all_y_pred) if t == 1 and p == 0),
                    "true_negatives": sum(1 for t, p in zip(all_y_true, all_y_pred) if t == 0 and p == 0),
                }
            )

            logging.info(f"Threshold {threshold:.2f}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        else:
            logging.warning(f"No evaluation points for threshold {threshold:.2f}")

    return pd.DataFrame(results)


def create_lof_accuracy_plot(results_df, output_path, config):
    """Create F-score vs threshold plot"""

    plt.figure(figsize=(10, 6))

    # Main F1 score plot - only if we have data
    if len(results_df) > 0 and not results_df.empty:
        plt.plot(results_df["threshold"], results_df["f1_score"], "b-o", linewidth=2, markersize=6, label="F1 Score")
        plt.plot(results_df["threshold"], results_df["precision"], "g--s", linewidth=1, markersize=4, label="Precision")
        plt.plot(results_df["threshold"], results_df["recall"], "r--^", linewidth=1, markersize=4, label="Recall")
    else:
        # Plot empty placeholder
        plt.text(
            0.5,
            0.5,
            "No evaluation data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )

    plt.xlabel("LOF Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("LOF Bad Channel Detection Accuracy vs Threshold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Find optimal threshold (max F1 score) - skip if no results
    if len(results_df) > 0 and not results_df.empty:
        best_idx = results_df["f1_score"].idxmax()
        best_threshold = results_df.loc[best_idx, "threshold"]
        best_f1 = results_df.loc[best_idx, "f1_score"]

        plt.axvline(
            x=best_threshold,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"Optimal: {best_threshold:.2f} (F1={best_f1:.3f})",
        )

    plt.legend()

    # Add evaluation info
    total_channels = results_df["total_channels"].iloc[0] if len(results_df) > 0 else 0
    evaluation_channels = config["analysis"]["lof_evaluation"]["evaluation_channels"]

    plt.figtext(
        0.02,
        0.02,
        f"Evaluation channels: {evaluation_channels}\nTotal evaluation points: {total_channels}",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=config["analysis"]["lof_evaluation"]["dpi"], bbox_inches="tight")
    plt.close()

    logging.info(f"F-score plot saved to: {output_path}")


def create_lof_channel_barplot(
    flattened_wars, animal_folder_map, animal_id_map, evaluation_channels, output_path, config
):
    """Create LOF scores vs channels barplot with individual animal bars

    Args:
        flattened_wars: Dictionary mapping animal names to (pkl_file, json_file) tuples
        animal_folder_map: Mapping from animal names to folder names
        animal_id_map: Mapping from animal names to IDs
        evaluation_channels: List of channel abbreviations to include
        output_path: Output path for the plot
        config: Configuration dictionary
    """

    # Collect LOF scores per animal (averaged across days only)
    animal_channel_data = []  # List of (animal_name, channel_scores_dict)
    animals_processed = 0

    for animal_name, (pkl_file, json_file) in flattened_wars.items():
        try:
            # Load the flattened WAR
            war = visualization.WindowAnalysisResult.load_pickle_and_json(
                folder_path=Path(pkl_file).parent, pickle_name=Path(pkl_file).name, json_name=Path(json_file).name
            )

            # Check if LOF scores are available
            if not hasattr(war, "lof_scores_dict") or not war.lof_scores_dict:
                logging.warning(f"Animal {animal_name}: No LOF scores available")
                continue

            # Collect LOF scores for this animal across all animal-days
            animal_channel_scores = {channel: [] for channel in evaluation_channels}

            for animalday, lof_data in war.lof_scores_dict.items():
                if "lof_scores" not in lof_data or "channel_names" not in lof_data:
                    continue

                scores = lof_data["lof_scores"]
                channel_names = lof_data["channel_names"]

                # Map scores to evaluation channels
                for i, channel in enumerate(channel_names):
                    if i < len(scores):
                        # Try direct match first
                        if channel in evaluation_channels:
                            animal_channel_scores[channel].append(scores[i])
                        elif core.parse_chname_to_abbrev(channel, strict_matching=False) in evaluation_channels:
                            animal_channel_scores[core.parse_chname_to_abbrev(channel, strict_matching=False)].append(
                                scores[i]
                            )

            # Average across days for this animal only
            animal_avg_scores = {}
            for channel in evaluation_channels:
                if animal_channel_scores[channel]:  # If we have scores for this channel
                    animal_avg_scores[channel] = np.mean(animal_channel_scores[channel])

            # Only include animals that have at least some channel data
            if animal_avg_scores:
                animal_channel_data.append((animal_name, animal_avg_scores))
                animals_processed += 1
                logging.debug(f"Processed LOF scores for animal {animal_name}")

        except Exception as e:
            logging.error(f"Failed to process LOF scores for {animal_name}: {str(e)}")
            raise

    logging.info(f"Creating LOF channel barplot with {len(evaluation_channels)} channels from {animals_processed} animals")

    # Create the barplot
    plt.figure(figsize=(12, 6))

    if animal_channel_data:
        # Prepare data for plotting - each bar represents one animal's one channel
        all_scores = []

        for animal_idx, (animal_name, channel_scores) in enumerate(animal_channel_data):
            for channel in evaluation_channels:
                if channel in channel_scores:
                    all_scores.append(channel_scores[channel])

        # Filter out NaN values
        valid_scores = [score for score in all_scores if not np.isnan(score)]

        # Create adjacent bars - all blue (C0)
        x_positions = np.arange(len(valid_scores))
        plt.bar(x_positions, valid_scores, color='C0')

        # Customize the plot
        plt.xlabel("Channels", fontsize=12)
        plt.ylabel("LOF Score", fontsize=12)
        plt.title("LOF Scores by Channel (Each Animal Averaged Across Days)", fontsize=14)

        # Remove x tick marks completely
        plt.xticks([])

        # Add grid for readability
        plt.grid(True, alpha=0.3, axis="y")

    else:
        # No data available
        plt.text(
            0.5,
            0.5,
            "No LOF score data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        plt.xlabel("Channels", fontsize=12)
        plt.ylabel("LOF Score", fontsize=12)
        plt.title("LOF Scores by Channel (No Data Available)", fontsize=14)

    # Add evaluation info
    total_bars = len(valid_scores) if animal_channel_data else 0
    plt.figtext(
        0.02,
        0.02,
        f"Evaluation channels: {evaluation_channels}\nAnimals processed: {animals_processed}\nTotal bars: {total_bars}",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=config["analysis"]["lof_evaluation"]["dpi"], bbox_inches="tight")
    plt.close()

    logging.info(f"LOF channel barplot saved to: {output_path}")


def main():
    """Main execution function"""

    # Get parameters from snakemake
    global snakemake
    war_pkl_files = snakemake.input.war_pkls
    war_json_files = snakemake.input.war_jsons
    samples_config = snakemake.params.samples_config
    config = snakemake.params.config
    animal_folder_map = snakemake.params.animal_folder_map
    animal_id_map = snakemake.params.animal_id_map

    output_csv = snakemake.output.results_csv
    output_plot = snakemake.output.plot_png
    output_barplot = snakemake.output.barplot_png

    # Get evaluation parameters
    lof_params = config["analysis"]["lof_evaluation"]
    threshold_range = lof_params["threshold_range"]
    evaluation_channels = lof_params["evaluation_channels"]

    # Generate threshold array
    thresholds = np.arange(
        threshold_range["min"], threshold_range["max"] + threshold_range["step"], threshold_range["step"]
    )

    logging.info("Starting LOF accuracy evaluation")
    logging.info(
        f"Threshold range: {threshold_range['min']:.1f} to {threshold_range['max']:.1f}, step {threshold_range['step']:.1f}"
    )
    logging.info(f"Evaluation channels: {evaluation_channels}")
    logging.info(f"Number of animals: {len(war_pkl_files)}")

    # Create mapping from paired pkl/json files to animal names
    flattened_wars = {}
    for pkl_file, json_file in zip(war_pkl_files, war_json_files):
        # Extract animal name from path: results/wars_flattened/{animal}/war.pkl
        animal_name = Path(pkl_file).parent.name
        flattened_wars[animal_name] = (pkl_file, json_file)

    try:
        # Evaluate LOF accuracy across all thresholds and animals
        results_df = evaluate_lof_accuracy_across_animals(
            flattened_wars=flattened_wars,
            samples_config=samples_config,
            animal_folder_map=animal_folder_map,
            animal_id_map=animal_id_map,
            thresholds=thresholds,
            evaluation_channels=evaluation_channels,
        )

        # Debug: Log results DataFrame info
        logging.info(f"Results DataFrame shape: {results_df.shape}")
        logging.info(f"Results DataFrame columns: {list(results_df.columns)}")
        if len(results_df) > 0:
            logging.info(f"Results DataFrame head:\n{results_df.head()}")
            logging.info(f"F1 scores: {results_df['f1_score'].tolist()}")
        else:
            logging.warning("Results DataFrame is empty!")

        # Create output directory
        output_dir = Path(output_csv).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results CSV
        results_df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to: {output_csv}")

        # Create and save F-score plot
        create_lof_accuracy_plot(results_df, output_plot, config)

        # Create and save LOF channel barplot
        create_lof_channel_barplot(
            flattened_wars=flattened_wars,
            animal_folder_map=animal_folder_map,
            animal_id_map=animal_id_map,
            evaluation_channels=evaluation_channels,
            output_path=output_barplot,
            config=config,
        )

        if len(results_df) > 0 and not results_df.empty and not results_df["f1_score"].isna().all():
            try:
                best_idx = results_df["f1_score"].idxmax()
                best_threshold = results_df.loc[best_idx, "threshold"]
                best_f1 = results_df.loc[best_idx, "f1_score"]
                logging.info(f"Optimal threshold: {best_threshold:.2f} (F1 score: {best_f1:.3f})")
            except Exception as e:
                logging.error(f"Error finding optimal threshold: {str(e)}")
                logging.warning("Cannot determine optimal threshold from results")
                raise
        else:
            logging.warning("No valid results available for summary")

        logging.info("LOF accuracy evaluation completed successfully")

    except Exception as e:
        logging.error(f"Failed to complete LOF accuracy evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG, stream=sys.stdout, force=True
        )
        main()
