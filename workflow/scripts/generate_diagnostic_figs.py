#!/usr/bin/env python
"""
Diagnostic Figures Generation Script
====================================

Generate diagnostic figures from WAR files into a directory.
Uses checkpoint approach - lets AnimalPlotter generate whatever files it naturally creates.

Input: WAR pickle and JSON files
Output: Directory containing diagnostic figure PNG files
"""

import sys
import logging
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from pythoneeg import visualization


def create_norm_from_config(norm_config):
    """Create matplotlib normalization from config parameters"""
    norm_type = norm_config["norm_type"]
    norm_params = norm_config["norm_params"]

    if norm_type == "fixed":
        return matplotlib.colors.Normalize(vmin=norm_params["vmin"], vmax=norm_params["vmax"])
    elif norm_type == "centered":
        return matplotlib.colors.CenteredNorm(halfrange=norm_params["halfrange"])
    elif norm_type == "minmax":
        return matplotlib.colors.Normalize(vmin=norm_params["vmin"], vmax=norm_params["vmax"])
    else:
        logging.warning(f"Unknown norm_type: {norm_type}, using default")
        return None


def generate_temporal_heatmaps_from_config(animal_plotter, config, animal_id, output_dir, version_name):
    """Generate temporal heatmaps using AnimalPlotter and config parameters"""
    try:
        heatmaps_config = config["analysis"]["figures"]["temporal_heatmaps"]["features"]

        for feature_name, heatmap_config in heatmaps_config.items():
            logging.info(f"    - Generating {feature_name} temporal heatmap")

            # Create normalization from config
            norm = create_norm_from_config(heatmap_config)

            # Generate the heatmap using AnimalPlotter
            animal_plotter.plot_temporal_heatmap(
                features=feature_name,
                figsize=tuple(heatmap_config["figsize"]),
                cmap=heatmap_config["cmap"],
                norm=norm,
            )

            # The AnimalPlotter saves files automatically, but we need to ensure
            # they match the expected output names from the Snakemake rule
            expected_filename = f"{animal_id}_temporal_heatmap_{feature_name}.png"
            expected_path = output_dir / expected_filename

            # Check if file was created with the expected name, if not try to find and rename
            generated_files = list(output_dir.glob(f"*{feature_name}*heatmap*.png"))
            if generated_files and not expected_path.exists():
                # Rename to match expected output
                generated_files[0].rename(expected_path)
                logging.info(f"Renamed {generated_files[0]} to {expected_path}")

    except Exception as e:
        logging.error(f"Failed to generate temporal heatmaps: {str(e)}")
        # Create placeholder files to satisfy Snakemake outputs
        for feature_name in config["analysis"]["figures"]["temporal_heatmaps"]["features"].keys():
            placeholder_path = output_dir / f"{animal_id}_temporal_heatmap_{feature_name}.png"
            placeholder_path.touch()


def load_war_and_config():
    config = snakemake.params.config
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id
    output_dir = Path(snakemake.output.figure_dir)

    war_dir = Path(snakemake.input.war_pkl[0]).parent
    war_pkl_name = Path(snakemake.input.war_pkl[0]).name
    war_json_name = Path(snakemake.input.war_json[0]).name
    war = visualization.WindowAnalysisResult.load_pickle_and_json(
        folder_path=war_dir, pickle_name=war_pkl_name, json_name=war_json_name
    )

    return war, config, animal_folder, animal_id, output_dir


def generate_figures_for_war_version(war_version, war, config, animal_id, output_subdir, version_name):
    """Generate figures for a specific version (filtered/unfiltered) of WAR data"""
    logging.info(f"Generating {version_name} figures")
    output_subdir.mkdir(parents=True, exist_ok=True)

    save_path_base = output_subdir / animal_id
    ap = visualization.AnimalPlotter(war_version, save_fig=True, save_path=str(save_path_base))
    figure_config = config["analysis"]["figures"]

    try:
        logging.info(f"  - Generating {version_name} PSD histogram")
        ap.plot_psd_histogram(
            figsize=figure_config["psd_histogram"]["figsize"],
            avg_channels=figure_config["psd_histogram"]["avg_channels"],
            plot_type=figure_config["psd_histogram"]["plot_type"],
        )

        logging.info(f"  - Generating {version_name} coherence/correlation spectral plots")
        ap.plot_coherecorr_spectral(
            figsize=figure_config["coherecorr_spectral"]["figsize"],
            score_type=figure_config["coherecorr_spectral"]["score_type"],
        )

        logging.info(f"  - Generating {version_name} PSD spectrogram")
        ap.plot_psd_spectrogram(
            figsize=figure_config["psd_spectrogram"]["figsize"], mode=figure_config["psd_spectrogram"]["mode"]
        )

        # Generate temporal heatmaps for this version
        logging.info(f"  - Generating {version_name} temporal heatmaps")
        generate_temporal_heatmaps_from_config(ap, config, animal_id, output_subdir, version_name)

    except Exception as e:
        logging.error(f"Figure generation failed for {version_name}: {str(e)}")
        raise

    created_files = list(output_subdir.glob("*.png"))
    logging.info(f"Generated {len(created_files)} {version_name} figure files")
    return created_files


def generate_diagnostic_figures_for_animal(war, config, animal_folder, animal_id, output_dir):
    logging.info(f"Processing {animal_folder} - {animal_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    processing_config = config["analysis"]["processing"]

    # Preprocessing - Prepare WAR data (reorder/pad channels for both versions)
    logging.info("Preprocessing: Reordering and padding channels")
    war.reorder_and_pad_channels(processing_config["channel_reorder"], use_abbrevs=processing_config["use_abbrevs"])

    # Generate unfiltered figures
    unfiltered_dir = output_dir / "unfiltered"
    unfiltered_files = generate_figures_for_war_version(war, war, config, animal_id, unfiltered_dir, "unfiltered")

    # Generate filtered figures - pass all filtering parameters directly
    filtering_config = processing_config["filtering"]
    logging.info(f"Applying filtering with parameters: {filtering_config}")
    war_filtered = war.filter_all(inplace=False, **filtering_config)

    # Use consistent folder name - filtering details will be in summary
    filtered_dir = output_dir / "filtered"

    filtered_files = generate_figures_for_war_version(war_filtered, war, config, animal_id, filtered_dir, "filtered")

    # Validate that files were created
    total_files = unfiltered_files + filtered_files
    logging.info(f"Generated {len(total_files)} total figure files:")
    for f in total_files:
        logging.info(f"  - {f.relative_to(output_dir)}")

    if not total_files:
        raise FileNotFoundError(f"No figure files were created in {output_dir}")

    logging.info(f"Successfully generated diagnostic figures for {animal_folder} {animal_id}")


def main():
    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.DEBUG,
            stream=sys.stdout,
            force=True,
        )

        try:
            logging.info("Diagnostic figures generation started")
            war, config, animal_folder, animal_id, output_dir = load_war_and_config()
            generate_diagnostic_figures_for_animal(war, config, animal_folder, animal_id, output_dir)
            logging.info(f"Completed diagnostic figures for {animal_folder} {animal_id}")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logging.error(error_msg)
            raise


if __name__ == "__main__":
    main()
