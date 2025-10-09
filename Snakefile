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
import glob
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


def get_all_fragment_filtered_pkl(wildcards):
    out = []
    for anim in ANIMALS:
        # Only process animals that have quality-filtered output
        qual_filter_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        qual_filenames = glob_wildcards(os.path.join(qual_filter_output, "{filename}.pkl")).filename
        if qual_filenames:  # Only if quality filtering produced files
            out.append(f"results/wars_fragment_filtered/{Path(qual_filter_output).name}/war.pkl")
    return out


def get_all_fragment_filtered_json(wildcards):
    out = []
    for anim in ANIMALS:
        qual_filter_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        qual_filenames = glob_wildcards(os.path.join(qual_filter_output, "{filename}.json")).filename
        if qual_filenames:
            out.append(f"results/wars_fragment_filtered/{Path(qual_filter_output).name}/war.json")
    return out


# def get_flattened_wars_pkl(wildcards):
#     out = []
#     for anim in ANIMALS:
#         checkpoint_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
#         qual_filenames = glob_wildcards(os.path.join(checkpoint_output, "{filename}.pkl")).filename
#         if qual_filenames:
#             out.append(f"results/wars_flattened/{Path(checkpoint_output).name}/war.pkl")
#     return out


# def get_flattened_wars_json(wildcards):
#     out = []
#     for anim in ANIMALS:
#         checkpoint_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
#         qual_filenames = glob_wildcards(os.path.join(checkpoint_output, "{filename}.json")).filename
#         if qual_filenames:
#             out.append(f"results/wars_flattened/{Path(checkpoint_output).name}/war.json")
#     return out

def get_wars_after_quality_filtered(wildcards, filepath_prepend, filepath_append):
    """General case function to get any desired WAR files for steps after quality filter"""
    out = []
    for anim in ANIMALS:
        checkpoint_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        animal_name = Path(checkpoint_output).name
        qual_filenames = glob.glob(os.path.join(checkpoint_output, "war.pkl"))
        if qual_filenames:
            out.append(str(Path(filepath_prepend) / animal_name / filepath_append))
    return out

def get_all_flattened_manual_wars(wildcards):
    """Get all manually channel-filtered flattened WAR paths"""
    out = []
    for anim in ANIMALS:
        checkpoint_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        qual_filenames = glob_wildcards(os.path.join(checkpoint_output, "{filename}.pkl")).filename
        if qual_filenames:
            out.append(f"results/wars_flattened_manual/{Path(checkpoint_output).name}/war.pkl")
    return out


def get_all_flattened_lof_wars(wildcards):
    """Get all LOF channel-filtered flattened WAR paths"""
    out = []
    for anim in ANIMALS:
        checkpoint_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        qual_filenames = glob_wildcards(os.path.join(checkpoint_output, "{filename}.pkl")).filename
        if qual_filenames:
            out.append(f"results/wars_flattened_lof/{Path(checkpoint_output).name}/war.pkl")
    return out


def get_diagnostic_figures_unfiltered(wildcards):
    outputs = []
    for anim in ANIMALS:
        checkpoint_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        hypothetical_war_path = Path(checkpoint_output) / "war.pkl"
        if os.path.exists(hypothetical_war_path):
            # print(f"thing exists: {hypothetical_war_path}")
            # outputs.extend(expand("results/diagnostic_figures/{a}/unfiltered", a=glob_wildcards()))
            outputs.append(f"results/diagnostic_figures/{Path(checkpoint_output).name}/unfiltered")
    return outputs

def get_diagnostic_figures_filtered(wildcards):
    outputs = []
    for anim in ANIMALS:
        checkpoint_output = checkpoints.war_quality_filter.get(animal=anim).output[0]
        hypothetical_war_path = Path(checkpoint_output) / "war.pkl"
        if os.path.exists(hypothetical_war_path):
            # print(f"thing exists: {hypothetical_war_path}")
            # outputs.extend(expand("results/diagnostic_figures/{a}/unfiltered", a=glob_wildcards()))
            outputs.append(f"results/diagnostic_figures/{Path(checkpoint_output).name}/filtered")
        # else:
        #     print(f"does NOT exist: {hypothetical_war_path}")
    return outputs



# def get_diagnostic_figures_filtered(wildcards):
#     """Get filtered diagnostic figure directories for fragment-filtered animals"""  
#     filtered_dirs = []
#     for anim in ANIMALS:
#         ck_output = checkpoints.make_diagnostic_figures_filtered.get(animal=anim).output
#         if ck_output:
#             filtered_dirs.append(f"results/diagnostic_figures/{anim}/filtered/")
#     return filtered_dirs


# Wildcard constraints to prevent conflicts
wildcard_constraints:
    animal="[^/]+",  # Animal names cannot contain slashes


# Include rule definitions
include: "workflow/rules/war_generation.smk"
include: "workflow/rules/fdsar_diagnostics.smk"
include: "workflow/rules/war_quality_filter.smk"
include: "workflow/rules/war_standardize.smk"
include: "workflow/rules/war_fragment_filtering.smk"
include: "workflow/rules/war_channel_filtering.smk"
include: "workflow/rules/diagnostic_figures.smk"
include: "workflow/rules/war_flattening.smk"
include: "workflow/rules/war_zeitgeber.smk"
include: "workflow/rules/zeitgeber_plots.smk"
include: "workflow/rules/ep_analysis.smk"
include: "workflow/rules/lof_evaluation.smk"
include: "workflow/rules/filtering_comparison.smk"
include: "workflow/rules/notebook.smk"


rule all:
    input:
        # Pipeline visualization
        'results/graphs/rulegraph.png',
        'results/graphs/filegraph.png',
        'results/graphs/dag.png',

        # WAR generation and prefiltering (includes spike detection)
        expand("results/wars_quality_filtered/{animal}", animal=ANIMALS),

        # FDSAR spike detection diagnostics
        expand("results/fdsar_diagnostics/{animal}", animal=ANIMALS), # FIXME this crashes the repository

        # WAR per-animal diagnostic plots (unfiltered)
        # NOTE also trigger fragment filtering + diagnostic figures filter unfiltered
        get_diagnostic_figures_unfiltered,

        # WAR per-animal diagnostic plots (filtered)
        get_diagnostic_figures_filtered,

        # ZT time-based features
        "results/wars_zeitgeber/zeitgeber_features.pkl",
        "results/zeitgeber_plots/",

        # EP full experiment plots
        # expand("results/wars_flattened/{animal}/war.pkl", animal=glob_wildcards("results/wars_fragment_filtered/{animal}/war.pkl").animal),
        # expand("results/wars_flattened/{animal}/war.pkl", animal=glob_wildcards("results/wars_fragment_filtered/{animal}/war.pkl").animal),
        # get_fragment_filtered_pkl,
        "results/ep_figures/",
        "results/ep_heatmaps/",

        # LOF accuracy evaluation
        "results/lof_evaluation/lof_accuracy_results.csv",
        "results/lof_evaluation/lof_fscore_vs_threshold.png",

        # Filtering comparison analysis (manual vs LOF)
        "results/filtering_comparison_plots/",

        # Interactive analysis notebooks
        # "results/notebooks/war_data_explorer.ipynb",

rule graphs:
    input:
        'results/graphs/rulegraph.png',
        'results/graphs/filegraph.png',
        'results/graphs/dag.png',

rule rulegraph:
    output: "results/graphs/rulegraph.png"
    shell: "snakemake --rulegraph --forceall | dot -Tpng > {output}"


rule filegraph:
    output: "results/graphs/filegraph.png"
    shell: "snakemake --filegraph --forceall | dot -Tpng > {output}"


rule dag:
    output: "results/graphs/dag.png"
    shell: "snakemake --dag --forceall | dot -Tpng > {output}"



# Configuration validation
# FIXME better to define in a json/yaml schema
def validate_config():
    required_keys = ["base_folder", "data_parent_folder", "temp_directory"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


validate_config()
