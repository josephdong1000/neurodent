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
configfile: "config/config.yaml"


samples_file = config["samples"]["samples_file"]


# Load sample definitions
import json
import re
import os
import sys
from datetime import datetime
from django.utils.text import slugify
from snakemake.io import glob_wildcards

# Load samples config
with open(samples_file, "r") as f:
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


def get_animal_quality_filtered_pkl(wildcards):
    checkpoint_output = checkpoints.war_quality_filter.get(**wildcards).output[0]
    filenames = glob_wildcards(os.path.join(checkpoint_output, "{filename}.pkl")).filename
    return expand(os.path.join(checkpoint_output, "{filename}.pkl"), filename=filenames)


def get_animal_quality_filtered_json(wildcards):
    checkpoint_output = checkpoints.war_quality_filter.get(**wildcards).output[0]
    filenames = glob_wildcards(os.path.join(checkpoint_output, "{filename}.pkl")).filename
    return expand(os.path.join(checkpoint_output, "{filename}.json"), filename=filenames)


# These should be defined because this is an aggregation step with an upstream checkpointed step
def get_fragment_filtered_pkl(wildcards):
    all_fragment_filtered_files = []
    for anim in ANIMALS:
        ck_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        filename = glob_wildcards(os.path.join(f"results/wars_quality_filtered/{anim}", "{filename}.pkl")).filename
        all_fragment_filtered_files.extend(expand(Path("results/wars_fragment_filtered") / anim / "{filename}.pkl", filename=filename))
    return all_fragment_filtered_files


def get_fragment_filtered_json(wildcards):
    all_fragment_filtered_files = []
    for anim in ANIMALS:
        ck_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        filename = glob_wildcards(os.path.join(f"results/wars_quality_filtered/{anim}", "{filename}.pkl")).filename
        all_fragment_filtered_files.extend(expand(Path("results/wars_fragment_filtered") / anim / "{filename}.json", filename=filename))
    return all_fragment_filtered_files


def get_flattened_wars_pkl(wildcards):
    all_flattened_wars_files = []
    for anim in ANIMALS:
        ck_output = checkpoints.flatten_wars.get(animal=anim).output[0]
        filename = glob_wildcards(os.path.join(f"results/wars_flattened/{anim}", "{filename}.pkl")).filename
        all_flattened_wars_files.extend(expand(Path("results/wars_flattened") / anim / "{filename}.pkl", filename=filename))
    return all_flattened_wars_files


def get_flattened_wars_json(wildcards):
    all_flattened_wars_files = []
    for anim in ANIMALS:
        ck_output = checkpoints.flatten_wars.get(animal=anim).output[0]
        filename = glob_wildcards(os.path.join(f"results/wars_flattened/{anim}", "{filename}.json")).filename
        all_flattened_wars_files.extend(expand(Path("results/wars_flattened") / anim / "{filename}.json", filename=filename))
    return all_flattened_wars_files


# Wildcard constraints to prevent conflicts
wildcard_constraints:
    animal="[^/]+",  # Animal names cannot contain slashes


# Include rule definitions
include: "workflow/rules/war_generation.smk"
include: "workflow/rules/war_quality_filter.smk"
include: "workflow/rules/war_fragment_filtering.smk"
include: "workflow/rules/diagnostic_figures.smk"
include: "workflow/rules/war_flattening.smk"
include: "workflow/rules/war_zeitgeber.smk"
include: "workflow/rules/zeitgeber_plots.smk"
include: "workflow/rules/ep_analysis.smk"
include: "workflow/rules/final_analysis.smk"


rule all:
    input:
        # Pipeline visualization
        'results/graphs/rulegraph.png',
        'results/graphs/filegraph.png',
        'results/graphs/dag.png',
        # WAR generation and prefiltering
        expand("results/wars_quality_filtered/{animal}", animal=ANIMALS),
        # WAR per-animal diagnostic plots
        lambda wc: expand("results/diagnostic_figures/{animal}/", animal=glob_wildcards("results/wars_quality_filtered/{animal}/war.pkl").animal),
        # ZT time-based features
        "results/wars_zeitgeber/zeitgeber_features.pkl",
        "results/zeitgeber_plots/",
        # EP full experiment plots
        lambda wc: expand("results/wars_flattened/{animal}/war.pkl", animal=glob_wildcards("results/wars_quality_filtered/{animal}/war.pkl").animal),
        "results/ep_figures/",
        "results/ep_heatmaps/",


rule rulegraph:
    output: "results/graphs/rulegraph.png"
    shell: "snakemake --rulegraph | dot -Tpng > {output}"


rule filegraph:
    output: "results/graphs/filegraph.png"
    shell: "snakemake --filegraph | dot -Tpng > {output}"


rule dag:
    output: "results/graphs/dag.png"
    shell: "snakemake --dag | dot -Tpng > {output}"


# Configuration validation
# FIXME better to define in a json/yaml schema
def validate_config():
    required_keys = ["base_folder", "data_parent_folder", "temp_directory"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


validate_config()
