"""
WAR Standardization Rules
========================

Rules for standardizing quality-filtered WARs including channel reordering,
padding, and unique hash addition. This separates standardization from filtering
to enable modular pipeline organization.
"""


rule war_standardize:
    """
    Standardize quality-filtered WARs: channel reordering, padding, unique hash
    """
    input:
        war_parquet="results/wars_quality_filtered/{animal}/war.parquet",
        war_json="results/wars_quality_filtered/{animal}/war.json",
    output:
        war_parquet="results/wars_standardized/{animal}/war.parquet",
        war_json="results/wars_standardized/{animal}/war.json",
    threads: 1
    retries:
        config["cluster"]["war_standardize"]["retries"]
    params:
        config=config,
        samples_config=samples_config,
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
    resources:
        time=config["cluster"]["war_standardize"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_standardize"]["mem_mb"]),
        nodes=config["cluster"]["war_standardize"]["nodes"],
    log:
        stdout="logs/war_standardize/{animal}.out",
        stderr="logs/war_standardize/{animal}.err",
    script:
        "../scripts/standardize_wars.py"