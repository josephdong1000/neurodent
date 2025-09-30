"""
FDSAR Diagnostic Rules
=====================

Rules for generating diagnostic plots from frequency-domain spike analysis results (FDSARs).
This includes spike-averaged traces and epoch data for visual inspection and validation.
"""


rule make_fdsar_diagnostics:
    """
    Generate diagnostic plots from FDSAR results for a specific animal.

    This rule:
    1. Loads FDSAR results from results/fdsars/{animal}/
    2. Generates spike-averaged trace plots for each channel
    3. Saves epoch data (.fif) and plots (.png) to results/fdsar_diagnostics/{animal}/
    """
    input:
        fdsar_dir="results/fdsars/{animal}",
    output:
        diagnostics_dir=directory("results/fdsar_diagnostics/{animal}"),
    params:
        config=config,
    threads: config["cluster"]["spike_averaged_traces"]["threads"]
    resources:
        time=config["cluster"]["spike_averaged_traces"]["time"],
        mem_mb=increment_memory(config["cluster"]["spike_averaged_traces"]["mem_mb"]),
        nodes=config["cluster"]["spike_averaged_traces"]["nodes"],
    log:
        "logs/fdsar_diagnostics/{animal}.log",
    script:
        "../scripts/generate_fdsar_diagnostics.py"
