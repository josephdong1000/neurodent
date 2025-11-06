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
        war_pkl="results/wars_quality_filtered/{animal}/war.pkl",
        war_json="results/wars_quality_filtered/{animal}/war.json",
    output:
        war_pkl="results/wars_standardized/{animal}/war.pkl",
        war_json="results/wars_standardized/{animal}/war.json",
    threads: 1
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
        "logs/war_standardize/{animal}.log",
    script:
        "../scripts/standardize_wars.py"