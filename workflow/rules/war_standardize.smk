"""
WAR Standardization Rules
========================

Rules for standardizing quality-filtered WARs including channel reordering,
padding, and unique hash addition. This separates standardization from filtering
to enable modular pipeline organization.
"""

import os


def _war_quality_filtered_parquet(wildcards):
    """Return path to quality-filtered war.parquet after the checkpoint has run."""
    checkpoint_output = checkpoints.war_quality_filter.get(animal=wildcards.animal).output[0]
    return os.path.join(checkpoint_output, "war.parquet")


def _war_quality_filtered_json(wildcards):
    """Return path to quality-filtered war.json after the checkpoint has run."""
    checkpoint_output = checkpoints.war_quality_filter.get(animal=wildcards.animal).output[0]
    return os.path.join(checkpoint_output, "war.json")


rule war_standardize:
    """
    Standardize quality-filtered WARs: channel reordering, padding, unique hash
    """
    input:
        war_parquet=_war_quality_filtered_parquet,
        war_json=_war_quality_filtered_json,
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