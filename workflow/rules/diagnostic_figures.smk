"""
Diagnostic Figures Rules
=======================

Rules for generating diagnostic figures from WARs using checkpoints.
This allows AnimalPlotter to generate variable numbers of files naturally.
"""


rule diagnostic_figures_unfiltered:
    """
    Generate diagnostic figures from quality-filtered (unfiltered) data
    """
    input:
        war_parquet=lambda wc: Path(checkpoints.war_quality_filter.get(**wc).output[0]) / "war.parquet",
        war_json=lambda wc: Path(checkpoints.war_quality_filter.get(**wc).output[0]) / "war.json",
    output:
        figure_dir=directory("results/diagnostic_figures/{animal}/unfiltered/"),
    params:
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        samples_config=samples_config,
        config=config,
    threads: config["cluster"]["diagnostic_figures"]["threads"]
    retries:
        config["cluster"]["diagnostic_figures"]["retries"]
    resources:
        time=config["cluster"]["diagnostic_figures"]["time"],
        mem_mb=increment_memory(config["cluster"]["diagnostic_figures"]["mem_mb"]),
        nodes=config["cluster"]["diagnostic_figures"]["nodes"],
    log:
        stdout="logs/diagnostic_figures_unfiltered/{animal}.out",
        stderr="logs/diagnostic_figures_unfiltered/{animal}.err",
    script:
        "../scripts/generate_diagnostic_figs.py"


rule diagnostic_figures_filtered:
    """
    Generate diagnostic figures from fragment-filtered data
    """
    input:
        war_parquet=lambda wc: checkpoints.war_fragment_filtering.get(**wc).output.war_parquet,
        war_json=lambda wc: checkpoints.war_fragment_filtering.get(**wc).output.war_json,
    output:
        figure_dir=directory("results/diagnostic_figures/{animal}/filtered/"),
    params:
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        samples_config=samples_config,
        config=config,
    threads: config["cluster"]["diagnostic_figures"]["threads"]
    retries:
        config["cluster"]["diagnostic_figures"]["retries"]
    resources:
        time=config["cluster"]["diagnostic_figures"]["time"],
        mem_mb=increment_memory(config["cluster"]["diagnostic_figures"]["mem_mb"]),
        nodes=config["cluster"]["diagnostic_figures"]["nodes"],
    log:
        stdout="logs/diagnostic_figures_filtered/{animal}.out",
        stderr="logs/diagnostic_figures_filtered/{animal}.err",
    script:
        "../scripts/generate_diagnostic_figs.py"