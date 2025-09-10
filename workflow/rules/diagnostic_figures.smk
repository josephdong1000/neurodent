"""
Diagnostic Figures Rules
=======================

Rules for generating diagnostic figures from WARs using checkpoints.
This allows AnimalPlotter to generate variable numbers of files naturally.
"""


checkpoint make_diagnostic_figures:
    """
    Generate diagnostic figures for a specific animal into subdirectories (filtered/unfiltered)
    """
    input:
        war_pkl=get_animal_quality_filtered_pkl,
        war_json=get_animal_quality_filtered_json
    output:
        figure_dir=directory("results/diagnostic_figures/{animal}/"),
        unfiltered_dir=directory("results/diagnostic_figures/{animal}/unfiltered"),
        filtered_dir=directory("results/diagnostic_figures/{animal}/filtered"),
    params:
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        samples_config=samples_config,
        config=config,
    threads: config["cluster"]["diagnostic_figures"]["threads"]
    retries: 2
    resources:
        time=config["cluster"]["diagnostic_figures"]["time"],
        mem_mb=increment_memory(config["cluster"]["diagnostic_figures"]["mem_mb"]),
        nodes=config["cluster"]["diagnostic_figures"]["nodes"],
    log:
        "logs/diagnostic_figures/{animal}.log",
    script:
        "../scripts/generate_diagnostic_figs.py"