"""
FDSAR Diagnostic Rules
=====================

Rules for generating diagnostic plots from frequency-domain spike analysis results (FDSARs).
This includes spike-averaged traces and epoch data for visual inspection and validation.
"""


rule fdsar_diagnostics:
    """
    Generate diagnostic plots from FDSAR results for a specific animal.

    This rule:
    1. Loads FDSAR results from results/fdsars/{animal}/
    2. Generates spike-averaged trace plots for each channel
    3. Saves epoch data (.fif) and plots (.png) to results/fdsar_diagnostics/{animal}/

    Note: Input changed from directory to WAR JSON file to avoid checkpoint-related
    DAG deadlocks. The FDSAR directory is still accessed via the script, but Snakemake
    only tracks the WAR completion as the trigger.
    """
    input:
        war_json="results/wars/{animal}/war.json",
    output:
        diagnostics_dir=directory("results/fdsar_diagnostics/{animal}"),
    params:
        config=config,
        samples_config=samples_config,
        fdsar_dir="results/fdsars/{animal}",
    threads: config["cluster"]["spike_averaged_traces"]["threads"]
    retries:
        config["cluster"]["spike_averaged_traces"]["retries"]
    resources:
        time=config["cluster"]["spike_averaged_traces"]["time"],
        mem_mb=increment_memory(config["cluster"]["spike_averaged_traces"]["mem_mb"]),
        nodes=config["cluster"]["spike_averaged_traces"]["nodes"],
    log:
        stdout="logs/fdsar_diagnostics/{animal}.out",
        stderr="logs/fdsar_diagnostics/{animal}.err",
    script:
        "../scripts/generate_fdsar_diagnostics.py"
