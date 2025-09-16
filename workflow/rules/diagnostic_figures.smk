"""
Diagnostic Figures Rules
=======================

Rules for generating diagnostic figures from WARs using checkpoints.
This allows AnimalPlotter to generate variable numbers of files naturally.
"""


rule make_diagnostic_figures_unfiltered:
    """
    Generate diagnostic figures from quality-filtered (unfiltered) data
    """
    input:
        war_pkl="results/wars_quality_filtered/{animal}/war.pkl",
        war_json="results/wars_quality_filtered/{animal}/war.json",
        # war_pkl=lambda wc: Path(checkpoints.war_quality_filter.get(**wc).output[0]) / "war.pkl",
        # war_json=lambda wc: Path(checkpoints.war_quality_filter.get(**wc).output[0]) / "war.json",
    output:
        figure_dir=directory("results/diagnostic_figures/{animal}/unfiltered/"),
    params:
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        samples_config=samples_config,
        config=config,
    threads: config["cluster"]["diagnostic_figures"]["threads"]
    retries: 1
    resources:
        time=config["cluster"]["diagnostic_figures"]["time"],
        mem_mb=increment_memory(config["cluster"]["diagnostic_figures"]["mem_mb"]),
        nodes=config["cluster"]["diagnostic_figures"]["nodes"],
    log:
        "logs/diagnostic_figures/{animal}_unfiltered.log",
    script:
        "../scripts/generate_diagnostic_figs.py"


rule make_diagnostic_figures_filtered:
    """
    Generate diagnostic figures from fragment-filtered data
    """
    input:
        war_pkl="results/wars_fragment_filtered/{animal}/war.pkl",
        war_json="results/wars_fragment_filtered/{animal}/war.json",
        # war_pkl=lambda wc: Path(checkpoints.war_fragment_filter.get(**wc).output[0]) / "war.pkl",
        # war_json=lambda wc: Path(checkpoints.war_fragment_filter.get(**wc).output[0]) / "war.json",
    output:
        figure_dir=directory("results/diagnostic_figures/{animal}/filtered/"),
    params:
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        samples_config=samples_config,
        config=config,
    threads: config["cluster"]["diagnostic_figures"]["threads"]
    retries: 1
    resources:
        time=config["cluster"]["diagnostic_figures"]["time"],
        mem_mb=increment_memory(config["cluster"]["diagnostic_figures"]["mem_mb"]),
        nodes=config["cluster"]["diagnostic_figures"]["nodes"],
    log:
        "logs/diagnostic_figures/{animal}_filtered.log",
    script:
        "../scripts/generate_diagnostic_figs.py"