"""
Notebook Rules
==============

Rules for executing Jupyter notebooks as part of the analysis pipeline.
These notebooks provide interactive exploration and reporting capabilities.
"""


rule war_explorer_notebook:
    """
    Execute the WAR data explorer notebook for interactive analysis of flattened WARs.
    
    This notebook loads all flattened WAR files and provides statistical analysis
    and visualization using ExperimentPlotter. The executed notebook with embedded
    outputs serves as a comprehensive analysis report.
    """
    input:
        war_files=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_flattened_manual", filepath_append="war.pkl"),
    output:
        # Executed notebook with embedded outputs
        notebook="results/notebooks/war_data_explorer.ipynb"
    log:
        stdout="logs/notebooks/war_data_explorer.stdout",
        stderr="logs/notebooks/war_data_explorer.stderr"
    threads: config["cluster"]["notebook"]["threads"]
    retries:
        config["cluster"]["notebook"]["retries"]
    params:
        # Pass configuration for resource allocation
        config=config,
        samples_config=samples_config,
    resources:
        time=config["cluster"]["notebook"]["time"],
        mem_mb=increment_memory(config["cluster"]["notebook"]["mem_mb"]),
        nodes=config["cluster"]["notebook"]["nodes"],
    notebook:
        # Source notebook to execute
        "../notebooks/war_data_explorer.ipynb"


rule all_notebooks:
    """
    Execute all analysis notebooks.
    """
    input:
        "results/notebooks/war_data_explorer.ipynb"