#!/usr/bin/env python3
"""
WAR Fragment Filtering Script
============================

This script applies filter_all() to quality-filtered WARs to perform channel
and fragment-level filtering. This is a separate step to ensure all downstream
analysis uses consistently filtered data.

Input: Quality-filtered WARs (genotype/bad animal filtering already applied)
Output: Fragment-filtered WARs with filter_all() applied
"""

import logging
import sys
from pathlib import Path

# Add pythoneeg to path
sys.path.insert(0, str(Path("pythoneeg").resolve()))
from pythoneeg import visualization

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True
)

def main():
    """Main fragment filtering function for single animal (1-to-1 operation)"""
    
    # Get parameters from snakemake
    logging.debug(f"snakemake.input.war_pkl: {snakemake.input.war_pkl}")
    logging.debug(f"snakemake.input.war_json: {snakemake.input.war_json}")

    # Handle both string and list inputs
    war_pkl_path = snakemake.input.war_pkl[0] if isinstance(snakemake.input.war_pkl, list) else snakemake.input.war_pkl
    war_json_path = (
        snakemake.input.war_json[0] if isinstance(snakemake.input.war_json, list) else snakemake.input.war_json
    )
    input_war_dir = Path(war_pkl_path).parent
    war_pkl_name = Path(war_pkl_path).name
    war_json_name = Path(war_json_path).name

    output_war_pkl = snakemake.output.war_pkl
    config = snakemake.params.config
    
    # Get animal name from wildcards
    animal_name = snakemake.wildcards.animal
    
    # Get fragment filtering parameters from config
    filtering_params = config["analysis"]["fragment_filtering"]
    bad_channels = filtering_params["bad_channels"]
    morphological_smoothing_seconds = filtering_params["morphological_smoothing_seconds"]
    
    # Get channel reordering parameters
    channel_reorder = filtering_params.get("channel_reorder", ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"])
    use_abbrevs = filtering_params.get("use_abbrevs", True)
    
    # Get unique hash parameters
    add_unique_hash = filtering_params.get("add_unique_hash", False)
    unique_hash_length = filtering_params.get("unique_hash_length", 4)
    
    logging.info(f"Processing animal: {animal_name}")
    logging.info(f"Bad channels to filter: {bad_channels}")
    logging.info(f"Morphological smoothing: {morphological_smoothing_seconds}")
    logging.info(f"Channel reorder: {channel_reorder}")
    logging.info(f"Use abbreviations: {use_abbrevs}")
    logging.info(f"Add unique hash: {add_unique_hash}")
    if add_unique_hash:
        logging.info(f"Unique hash length: {unique_hash_length}")
    
    try:
        # Load the quality-filtered WAR
        logging.info(f"Loading WAR from: {input_war_dir}")
        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=input_war_dir, 
            pickle_name=war_pkl_name, 
            json_name=war_json_name
        )
        
        # Apply channel standardization for all downstream steps
        logging.info("Applying channel reordering and padding")
        war.reorder_and_pad_channels(channel_reorder, use_abbrevs=use_abbrevs)
        
        # Add unique hash if requested
        if add_unique_hash:
            logging.info(f"Adding unique hash with length {unique_hash_length}")
            war.add_unique_hash(unique_hash_length)
        
        # Apply fragment and channel filtering
        logging.info("Applying filter_all() with configured parameters")
        war.filter_all(
            bad_channels=bad_channels,
            morphological_smoothing_seconds=morphological_smoothing_seconds
        )
        
        # Create output directory
        output_dir = Path(output_war_pkl).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save fragment-filtered WAR as both pickle and json
        war.save_pickle_and_json(output_dir)
        
        logging.info(f"Successfully filtered and saved {animal_name}")
        
    except Exception as e:
        logging.error(f"Failed to process {animal_name}: {str(e)}")
        raise
    
    logging.info("WAR fragment filtering script completed successfully")

if __name__ == "__main__":
    main()