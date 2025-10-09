"""
EP (ExperimentPlotter) Analysis Rules
=====================================

Rules for final experiment-level analysis using ExperimentPlotter.
Generates statistical figures and correlation/coherence matrix heatmaps
from flattened WARs across all animals.
"""


rule generate_ep_figures:
    """
    Generate experiment-level statistical figures using ExperimentPlotter
    """
    input:
        war_pkl=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_flattened_manual", filepath_append="war.pkl"),
        war_json=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_flattened_manual", filepath_append="war.json"),
    output:
        figure_dir=directory("results/ep_figures/"),
        data_dir=directory("results/ep_data/"),
    params:
        config=config,
    threads:
        config["cluster"]["ep_figures"]["threads"]
    resources:
        time=config["cluster"]["ep_figures"]["time"],
        mem_mb=increment_memory(config["cluster"]["ep_figures"]["mem_mb"]),
        nodes=config["cluster"]["ep_figures"]["nodes"],
    log:
        "logs/ep_analysis/generate_ep_figures.log",
    script:
        "../scripts/generate_ep_figures.py"


rule generate_ep_heatmaps:
    """
    Generate experiment-level correlation/coherence matrix heatmaps
    """
    input:
        # war_pkl=get_flattened_wars_pkl,
        # war_json=get_flattened_wars_json,
        war_pkl=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_flattened_manual", filepath_append="war.pkl"),
        war_json=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_flattened_manual", filepath_append="war.json"),
    output:
        heatmap_dir=directory("results/ep_heatmaps/"),
        heatmap_data_dir=directory("results/ep_heatmap_data/"),
    params:
        config=config,
    threads:
        config["cluster"]["ep_heatmaps"]["threads"]
    resources:
        time=config["cluster"]["ep_heatmaps"]["time"],
        mem_mb=increment_memory(config["cluster"]["ep_heatmaps"]["mem_mb"]),
        nodes=config["cluster"]["ep_heatmaps"]["nodes"],
    log:
        "logs/ep_analysis/generate_ep_heatmaps.log",
    script:
        "../scripts/generate_ep_heatmaps.py"