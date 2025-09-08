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

import numpy as np
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


def generate_temporal_heatmap(war_data, config, output_dir, animal_id, version_name):
    """Generate temporal heatmap visualization"""
    try:
        heatmap_config = config["analysis"]["figures"]["temporal_heatmap"]
        feature_type = heatmap_config["feature_type"]
        
        # Extract the appropriate feature for temporal visualization
        if feature_type == "rms_amplitude" and hasattr(war_data, 'rms_amplitude') and war_data.rms_amplitude is not None:
            data = war_data.rms_amplitude
            ylabel = "RMS Amplitude"
        elif feature_type == "amplitude_variance" and hasattr(war_data, 'amplitude_variance') and war_data.amplitude_variance is not None:
            data = war_data.amplitude_variance
            ylabel = "Amplitude Variance" 
        elif feature_type == "power_bands" and hasattr(war_data, 'power_bands') and war_data.power_bands is not None:
            # Use total power across bands
            data = np.sum(war_data.power_bands, axis=-1) if len(war_data.power_bands.shape) > 2 else war_data.power_bands
            ylabel = "Power (All Bands)"
        else:
            logging.warning(f"Feature type '{feature_type}' not available or no data found")
            return
        
        # Create the temporal heatmap
        fig, ax = plt.subplots(figsize=heatmap_config["figsize"])
        
        # Transpose for proper orientation (channels x time)
        if len(data.shape) == 2:
            heatmap_data = data.T
        else:
            heatmap_data = data
            
        im = ax.imshow(heatmap_data, 
                      aspect='auto', 
                      cmap=heatmap_config["colormap"],
                      interpolation=heatmap_config["interpolation"])
        
        ax.set_xlabel('Time Windows')
        ax.set_ylabel('Channels')
        ax.set_title(f'Temporal Heatmap - {ylabel} ({version_name.title()})')
        
        # Add colorbar if requested
        if heatmap_config["include_colorbar"]:
            plt.colorbar(im, ax=ax, label=ylabel)
        
        plt.tight_layout()
        
        # Save the heatmap
        output_path = output_dir / f"{animal_id}_{version_name}_temporal_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Temporal heatmap saved: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to generate temporal heatmap: {str(e)}")


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
        
        # Generate temporal heatmap for this version
        logging.info(f"  - Generating {version_name} temporal heatmap")
        generate_temporal_heatmap(war_version, config, output_subdir, animal_id, version_name)
        
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