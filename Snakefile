"""
NeuRodent Snakemake Pipeline
========================

This pipeline processes raw EEG data through multiple analysis stages:
1. Raw files → WARs (Windowed Analysis Results)
2. WARs → Temporal heatmaps (animal-level diagnostics)
3. WARs → Diagnostic figures
4. WARs → Flattened WARs → Final EP figures
"""

from pathlib import Path
import pandas as pd
import json
import re
import os
import sys
import glob

# Import workflow utilities
from neurodent.workflow.utils import deep_merge_dict


# Load configuration
configfile: "config/config.yaml"

# Load local override if it exists
if os.path.exists("config/config.local.yaml"):
    configfile: "config/config.local.yaml"


# Apply dataset-specific configuration
active_dataset = os.environ.get("NEURODENT_DATASET", config.get("active_dataset", "sox5_nwb"))
dataset_config_file = f"config/datasets/{active_dataset}.yaml"

if os.path.exists(dataset_config_file):
    # Load dataset-specific config from file
    import yaml
    with open(dataset_config_file, 'r') as f:
        dataset_config = yaml.safe_load(f) or {}

    # Deep merge dataset config into main config
    # This allows datasets to override ANY configuration, not just specific keys
    config = deep_merge_dict(config, dataset_config)

    # Report dataset configuration in user-readable format
    def format_config_value(value, indent=4):
        """Format a config value for display (handles nested dicts, lists, etc.)."""
        spaces = " " * indent
        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = []
            for k, v in value.items():
                formatted_val = format_config_value(v, indent + 2)
                if "\n" in formatted_val:
                    lines.append(f"{spaces}{k}:")
                    lines.append(formatted_val)
                else:
                    lines.append(f"{spaces}{k}: {formatted_val}")
            return "\n".join(lines)
        elif isinstance(value, list):
            if not value:
                return "[]"
            return f"[{', '.join(repr(v) for v in value)}]"
        elif isinstance(value, str):
            return f'"{value}"'
        elif value is None:
            return "null"
        else:
            return str(value)

    print(f"✓ Using dataset: {active_dataset}")
    print(f"  Config file: {dataset_config_file}")
    print(f"\n  Dataset configuration overrides:")
    print(format_config_value(dataset_config, indent=4))
    print()
else:
    # List available datasets by scanning config/datasets/ directory
    available_datasets = []
    if os.path.exists("config/datasets"):
        available_datasets = [f.replace('.yaml', '') for f in os.listdir("config/datasets") if f.endswith('.yaml')]

    raise FileNotFoundError(
        f"Dataset config file not found: {dataset_config_file}\n"
        f"Available datasets: {', '.join(available_datasets) if available_datasets else 'None'}"
    )

samples_file = config["samples"]["samples_file"]


# Load sample definitions
from datetime import datetime
from django.utils.text import slugify

# Verify snakemake is installed before importing from it
try:
    from snakemake.io import glob_wildcards
except ImportError:
    raise ImportError(
        "Snakemake is required for pipeline functionality.\n"
        "Install with either:\n"
        "  pip install neurodent[pipeline]\n"
        "  uv pip install neurodent[pipeline]\n"
    )

# Load samples config
with open(samples_file, "r") as f:
    samples_config = json.load(f)

# Extract sample information
DATA_FOLDERS = list(samples_config["data_folders_to_animal_ids"].keys())
ANIMALS = []
ANIMAL_TO_FOLDERS_MAP = {}  # Maps slugified animal_id -> list of (folder, animal_id, original_session_key)
ANIMAL_TO_FULL_ID_MAP = {} # Maps slugified animal_id -> original animal_id string

for folder, animals in samples_config["data_folders_to_animal_ids"].items():
    for animal in animals:
        # We group by the animal ID to merge split sessions
        slugified_name = slugify(animal, allow_unicode=True)

        if slugified_name not in ANIMALS:
            ANIMALS.append(slugified_name)
            ANIMAL_TO_FOLDERS_MAP[slugified_name] = []
            ANIMAL_TO_FULL_ID_MAP[slugified_name] = animal
        
        # Store tuple of (real_folder_path, original_animal_id, config_session_key)
        # config_session_key is 'folder' here
        ANIMAL_TO_FOLDERS_MAP[slugified_name].append((folder, animal, folder))

# Build mapping for animals from joint sessions
# These animals will have their data read from split output folders
JOINT_ANIMAL_TO_SESSION = {}  # Maps slugified animal name -> (session, original_animal_id)

for session, animals_dict in samples_config.get("joint_sessions", {}).items():
    for animal_id in animals_dict.keys():
        slugified_name = slugify(animal_id, allow_unicode=True)
        
        JOINT_ANIMAL_TO_SESSION[slugified_name] = (session, animal_id)
        
        if slugified_name not in ANIMALS:
            ANIMALS.append(slugified_name)
            ANIMAL_TO_FOLDERS_MAP[slugified_name] = []
            ANIMAL_TO_FULL_ID_MAP[slugified_name] = animal_id
            
        # For joint sessions, 'session' is the folder
        ANIMAL_TO_FOLDERS_MAP[slugified_name].append((session, animal_id, session))


def get_animal_folders(wildcards):
    """Get the list of (data_folder, animal_id, config_key) tuples for an animal."""
    return ANIMAL_TO_FOLDERS_MAP[wildcards.animal]

def get_animal_folder(wildcards):
    """Get the primary (first) data folder for an animal.
    
    Used for backward compatibility with downstream rules (e.g. log paths).
    """
    return ANIMAL_TO_FOLDERS_MAP[wildcards.animal][0][0]


def get_joint_session_channels(wildcards):
    """Get channel subset for joint session animals, or None for regular animals.
    
    Now looks up based on the animal ID slug. 
    Assumes if ANY session for this animal is joint, we return the channels for that session.
    (Merging joint + non-joint sessions for same animal is complex, assumed uniform).
    """
    if wildcards.animal in JOINT_ANIMAL_TO_SESSION:
        session, animal_id = JOINT_ANIMAL_TO_SESSION[wildcards.animal]
        return samples_config["joint_sessions"][session][animal_id]
    return None


def get_animal_id(wildcards):
    """Get the original animal ID string."""
    return ANIMAL_TO_FULL_ID_MAP[wildcards.animal]


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
include: "workflow/rules/war_relfreq_plots.smk"
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
        expand("results/fdsar_diagnostics/{animal}", animal=ANIMALS), # FIXME this crashes my VDI - perhaps a logic issue
        
        # WAR per-animal diagnostic plots (unfiltered)
        get_diagnostic_figures_unfiltered,

        # WAR per-animal diagnostic plots (filtered)
        get_diagnostic_figures_filtered,

        # ZT time-based features
        "results/wars_zeitgeber/zeitgeber_features.pkl",
        "results/zeitgeber_plots/",

        # Relative frequency distribution plots
        "results/relfreq_plots/",

        # EP full experiment plots
        "results/ep_figures/",
        "results/ep_heatmaps/",

        # LOF accuracy evaluation
        "results/lof_evaluation/lof_accuracy_results.csv",
        "results/lof_evaluation/lof_fscore_vs_threshold.png",

        # Filtering comparison analysis (manual vs LOF)
        "results/filtering_comparison_plots/",

        # Interactive analysis notebooks
        # "results/notebooks/war_data_explorer.ipynb", # FIXME configure the notebook so that it runs on Snakemake

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
    required_keys = ["temp_directory"]  # base_folder and data_parent_folder now in samples.json
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


validate_config()
