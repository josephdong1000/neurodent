"""
PyEEG Snakemake Pipeline
========================

This pipeline processes raw EEG data through multiple analysis stages:
1. Raw files → WARs (Windowed Analysis Results)
2. WARs → Temporal heatmaps (animal-level diagnostics)  
3. WARs → Diagnostic figures
4. WARs → Flattened WARs → Final EP figures

Usage:
    snakemake --profile slurm all
    snakemake --profile slurm war_generation
    snakemake --profile slurm final_analysis
"""

from pathlib import Path
import pandas as pd

# Load configuration
configfile: "workflow/config/config.yaml"
samples_file = config.get("samples_file", "workflow/config/samples.json")

# Load sample definitions
import json
import re
import os
import sys
from datetime import datetime
from django.utils.text import slugify

# Set up timestamped logging for this run
def setup_run_logging():
    """Set up timestamped logging directory and environment variables"""
    # Check if run ID already exists (prevents multiple timestamps)
    if "SNAKEMAKE_RUN_ID" in os.environ:
        return os.environ["SNAKEMAKE_RUN_ID"]
    
    # Get target rule from command line arguments - look for known rule names
    target_rule = "run"
    known_rules = ["all", "wars_only", "temporal_only", "diagnostics_only", "flatten_only", "final_only"]
    
    for arg in sys.argv[1:]:
        if arg in known_rules:
            target_rule = arg
            break
    
    # Create run identifier: timestamp + target rule (only once)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_{target_rule}"
    
    # Set environment variables for SLURM profile
    os.environ["SNAKEMAKE_RUN_ID"] = run_id
    os.environ["SNAKEMAKE_TARGET"] = target_rule
    
    # Create log directory
    log_dir = Path(f"logs/slurm/{run_id}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Write run ID to a temp file for consistency across all processes
    with open(".snakemake_run_id", "w") as f:
        f.write(run_id)
    
    print(f"📁 Logs for this run: logs/slurm/{run_id}/")
    return run_id

# Initialize run logging
RUN_ID = setup_run_logging()

# Create safe animal identifiers for file paths while preserving originals

with open(samples_file, 'r') as f:
    samples_config = json.load(f)

# Extract sample information  
DATA_FOLDERS = list(samples_config["data_folders_to_animal_ids"].keys())
ANIMALS = []
ANIMAL_TO_FOLDER_MAP = {}  # Maps slugified name back to (original_folder, original_animal_id)
SLUGIFIED_TO_ORIGINAL = {}  # Maps slugified name back to original combined name

# Get bad animaldays to filter out at job submission level
bad_folder_animalday = samples_config.get("bad_folder_animalday", [])

for folder, animals in samples_config["data_folders_to_animal_ids"].items():
    for animal in animals:
        combined_name = f"{folder} {animal}"
        
        # Skip bad animaldays entirely - prevents job submission
        if combined_name in bad_folder_animalday:
            print(f"⚠️  Skipping bad animalday: {combined_name}")
            continue
            
        slugified_name = slugify(combined_name, allow_unicode=True)
        
        ANIMALS.append(slugified_name)  # Use slugified names for file paths
        ANIMAL_TO_FOLDER_MAP[slugified_name] = (folder, animal)
        SLUGIFIED_TO_ORIGINAL[slugified_name] = combined_name

def get_animal_folder(wildcards):
    """Get the data folder for an animal from the combined name"""
    return ANIMAL_TO_FOLDER_MAP[wildcards.animal][0]

def get_animal_id(wildcards):
    """Get the animal ID for an animal from the combined name"""
    return ANIMAL_TO_FOLDER_MAP[wildcards.animal][1]

# Include rule definitions
include: "workflow/rules/war_generation.smk"
include: "workflow/rules/temporal_diagnostics.smk" 
include: "workflow/rules/diagnostic_figures.smk"
include: "workflow/rules/war_flattening.smk"
include: "workflow/rules/final_analysis.smk"

# Target rules - these are convenience rules for running specific pipeline stages
rule all:
    input:
        # WARs generated with json metadata
        expand("results/wars/{animal}/war.pkl", animal=ANIMALS),
        expand("results/wars/{animal}/war.json", animal=ANIMALS),
        # Temporal heatmaps
        expand("results/temporal_heatmaps/{animal}/heatmap.png", animal=ANIMALS),
        # Diagnostic figures  
        expand("results/diagnostic_figures/{animal}/coherecorr_spectral.png", animal=ANIMALS),
        expand("results/diagnostic_figures/{animal}/psd_histogram.png", animal=ANIMALS),
        expand("results/diagnostic_figures/{animal}/psd_spectrogram.png", animal=ANIMALS),
        # Flattened WARs
        "results/flattened_wars/combined_wars.pkl",
        # Final analysis outputs
        "results/final_analysis/experiment_plots_complete.flag"

rule wars_only:
    """Generate WARs from raw data only"""
    input:
        expand("results/wars/{animal}/war.pkl", animal=ANIMALS),
        expand("results/wars/{animal}/war.json", animal=ANIMALS),

rule temporal_only:
    """Generate temporal diagnostic heatmaps only"""
    input:
        expand("results/temporal_heatmaps/{animal}/heatmap.png", animal=ANIMALS)

rule diagnostics_only:
    """Generate diagnostic figures only"""
    input:
        expand("results/diagnostic_figures/{animal}/coherecorr_spectral.png", animal=ANIMALS),
        expand("results/diagnostic_figures/{animal}/psd_histogram.png", animal=ANIMALS),
        expand("results/diagnostic_figures/{animal}/psd_spectrogram.png", animal=ANIMALS)

rule flatten_only:
    """Generate flattened WARs only"""
    input:
        "results/flattened_wars/combined_wars.pkl"

rule final_only:
    """Generate final EP figures and analysis only"""
    input:
        "results/final_analysis/experiment_plots_complete.flag"

rule clean:
    """Clean all generated files"""
    shell:
        "rm -rf results/ logs/"

# Configuration validation
def validate_config():
    required_keys = ["base_folder", "data_parent_folder", "temp_directory"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

validate_config()