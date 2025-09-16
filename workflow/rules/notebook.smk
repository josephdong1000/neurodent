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
        # All flattened WAR files from all animals
        war_files=get_flattened_wars_pkl
    output:
        # Executed notebook with embedded outputs
        notebook="results/notebooks/war_data_explorer.ipynb"
    log:
        # Log file for notebook execution
        "logs/notebooks/war_data_explorer.log"
    threads: config["cluster"]["notebook"]["threads"]
    retries: 1
    params:
        # Pass configuration for resource allocation
        config=config,
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