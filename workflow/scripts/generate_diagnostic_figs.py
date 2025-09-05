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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from pythoneeg import visualization


def load_war_and_config():
    config = snakemake.params.config
    animal_folder = snakemake.params.animal_folder
    animal_id = snakemake.params.animal_id
    output_dir = Path(snakemake.output.figure_dir)
    
    war_dir = Path(snakemake.input.war_pkl).parent
    war_pkl_name = Path(snakemake.input.war_pkl).name
    war_json_name = Path(snakemake.input.war_json).name
    war = visualization.WindowAnalysisResult.load_pickle_and_json(
        folder_path=war_dir,
        pickle_name=war_pkl_name,
        json_name=war_json_name
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
            plot_type=figure_config["psd_histogram"]["plot_type"]
        )
        
        logging.info(f"  - Generating {version_name} coherence/correlation spectral plots")
        ap.plot_coherecorr_spectral(
            figsize=figure_config["coherecorr_spectral"]["figsize"],
            score_type=figure_config["coherecorr_spectral"]["score_type"]
        )
        
        logging.info(f"  - Generating {version_name} PSD spectrogram")
        ap.plot_psd_spectrogram(
            figsize=figure_config["psd_spectrogram"]["figsize"],
            mode=figure_config["psd_spectrogram"]["mode"]
        )
        
    except Exception as e:
        logging.error(f"AnimalPlotter failed for {version_name}: {str(e)}")
        raise
    
    created_files = list(output_subdir.glob("*.png"))
    logging.info(f"Generated {len(created_files)} {version_name} figure files")
    return created_files


def generate_diagnostic_figures_for_animal(war, config, animal_folder, animal_id, output_dir):
    logging.info(f"Processing {animal_folder} - {animal_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if war.genotype == "Unknown":
        logging.info(f"Skipping {animal_id} because genotype is Unknown")
        (output_dir / "empty_genotype_unknown.txt").write_text("Genotype unknown - no figures generated")
        return
    
    processing_config = config["analysis"]["processing"]
    
    # Preprocessing - Prepare WAR data (reorder/pad channels for both versions)
    logging.info("Preprocessing: Reordering and padding channels")
    war.reorder_and_pad_channels(
        processing_config["channel_reorder"], 
        use_abbrevs=processing_config["use_abbrevs"]
    )
    
    # Generate unfiltered figures
    unfiltered_dir = output_dir / "unfiltered"
    unfiltered_files = generate_figures_for_war_version(
        war, war, config, animal_id, unfiltered_dir, "unfiltered"
    )
    
    # Generate filtered figures - pass all filtering parameters directly
    filtering_config = processing_config["filtering"]
    logging.info(f"Applying filtering with parameters: {filtering_config}")
    war_filtered = war.filter_all(inplace=False, **filtering_config)
    
    # Use consistent folder name - filtering details will be in summary
    filtered_dir = output_dir / "filtered"
    
    filtered_files = generate_figures_for_war_version(
        war_filtered, war, config, animal_id, filtered_dir, "filtered"
    )
    
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