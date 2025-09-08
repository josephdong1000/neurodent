"""
PyEEG Snakemake Pipeline
========================

This pipeline processes raw EEG data through multiple analysis stages:
1. Raw files → WARs (Windowed Analysis Results)
2. WARs → Temporal heatmaps (animal-level diagnostics)  
3. WARs → Diagnostic figures
4. WARs → Flattened WARs → Final EP figures
"""

from pathlib import Path
import pandas as pd

# Load configuration
configfile: "workflow/config/config.yaml"
samples_file = config["samples"]["samples_file"]

workflow.rerun_incomplete = True
workflow.default_latency_wait = 120

# Load sample definitions
import json
import re
import os
import sys
from datetime import datetime
from django.utils.text import slugify

# Load samples config
with open(samples_file, 'r') as f:
    samples_config = json.load(f)

# Extract sample information  
DATA_FOLDERS = list(samples_config["data_folders_to_animal_ids"].keys())
ANIMALS = []
ANIMAL_TO_FOLDER_MAP = {}  # Maps slugified name back to (original_folder, original_animal_id)
SLUGIFIED_TO_ORIGINAL = {}  # Maps slugified name back to original combined name

for folder, animals in samples_config["data_folders_to_animal_ids"].items():
    for animal in animals:
        combined_name = f"{folder} {animal}"
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

def increment_memory(base_memory):
    def mem(wildcards, attempt):
        return base_memory * (2 ** (attempt - 1))
    return mem

def get_filtered_animals(wildcards):
    """Discover animals that passed quality filtering"""
    import os
    filtered_animals = []
    filtered_dir = "results/wars_filtered"
    if os.path.exists(filtered_dir):
        for item in os.listdir(filtered_dir):
            if os.path.isdir(os.path.join(filtered_dir, item)):
                war_file = os.path.join(filtered_dir, item, "war.pkl")
                if os.path.exists(war_file):
                    filtered_animals.append(item)
    return filtered_animals

def animal_passed_filtering(animal):
    """Check if a specific animal passed quality filtering"""
    import os
    war_file = os.path.join("results", "wars_filtered", animal, "war.pkl")
    return os.path.exists(war_file)

def filtered_input(filename):
    """Helper to create input function that only includes files for animals that passed filtering"""
    def input_func(wildcards):
        if animal_passed_filtering(wildcards.animal):
            return f"results/wars_filtered/{wildcards.animal}/{filename}"
        return []
    return input_func

def filtered_war_inputs():
    """Convenient function to get all standard filtered WAR inputs"""
    return {
        'war_pkl': filtered_input('war.pkl'),
        'war_json': filtered_input('war.json'), 
        'metadata': filtered_input('metadata.json')
    }

def filtered_war_pkl():
    """Convenience function for just the filtered WAR pickle"""
    return filtered_input('war.pkl')

def filtered_war_json():
    """Convenience function for just the filtered WAR JSON"""
    return filtered_input('war.json')

# Wildcard constraints to prevent conflicts
wildcard_constraints:
    animal="[^/]+"  # Animal names cannot contain slashes

# Include rule definitions  
include: "workflow/rules/war_generation.smk"
include: "workflow/rules/war_quality_filter.smk"
include: "workflow/rules/diagnostic_figures.smk"
include: "workflow/rules/war_flattening.smk"
include: "workflow/rules/final_analysis.smk"

# Target rules - these are convenience rules for running specific pipeline stages
rule all:
    input:
        # WARs generated with json metadata  
        # expand("results/wars/{animal}/war.pkl", animal=ANIMALS),
        # expand("results/wars/{animal}/war.json", animal=ANIMALS),
        # Quality filtered WARs first - need to ensure filtering happens for all animals
        expand("results/wars_filtered/{animal}", animal=ANIMALS),
        # Then diagnostic figures - only for animals that pass filtering
        # get_diagnostic_figures
        lambda wc: expand("results/diagnostic_figures/{animal}", animal=get_filtered_animals(wc))


        # Temporal heatmaps (using filtered WARs)
        # lambda wildcards: expand("results/temporal_heatmaps/{animal}/heatmap.png", animal=get_filtered_animals(wildcards)),
        # Flattened WARs
        # "results/flattened_wars/combined_wars.pkl",
        # Final analysis outputs
        # "results/final_analysis/experiment_plots_complete.flag"

rule wars_filtered_only:
    """Generate quality-filtered WARs"""
    input:
        expand("results/wars_filtered/{animal}", animal=ANIMALS)
        # get_filtered_animals

# rule diagnostics_only:
#     """Generate diagnostic figures only (quality-filtered WARs)"""
#     input:
#         lambda wildcards: expand("results/diagnostic_figures/{animal}", animal=get_filtered_animals(wildcards))
        
# rule temporal_only:
#     """Generate temporal diagnostic heatmaps only (quality-filtered WARs)"""
#     input:
#         lambda wildcards: expand("results/temporal_heatmaps/{animal}/heatmap.png", animal=get_filtered_animals(wildcards))

# rule flatten_only:
#     """Generate flattened WARs only"""
#     input:
#         "results/flattened_wars/combined_wars.pkl"

# rule final_only:
#     """Generate final EP figures and analysis only"""
#     input:
#         "results/final_analysis/experiment_plots_complete.flag"

# rule clean:
#     """Clean all generated files"""
#     shell:
#         "rm -rf results/ logs/"

# Configuration validation
def validate_config():
    required_keys = ["base_folder", "data_parent_folder", "temp_directory"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

validate_config()