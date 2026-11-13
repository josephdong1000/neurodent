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

    The script streams the WAR, so its memory follows one row group and one batch of it
    (rule_resources, war_loading="streamed"). war_generation writes each WAR in one row group,
    so for this rule that is the whole WAR's uncompressed size.
    """
    input:
        war_parquet=_war_quality_filtered_parquet,
        war_json=_war_quality_filtered_json,
    output:
        war_parquet="results/wars_standardized/{animal}/war.parquet",
        war_json="results/wars_standardized/{animal}/war.json",
    threads: config["cluster"]["war_standardize"]["threads"]
    retries:
        config["cluster"]["war_standardize"]["retries"]
    params:
        config=analysis_config(config),
        samples_config=samples_config,
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
    resources:
        **rule_resources(config["cluster"]["war_standardize"], war_loading="streamed"),
    log:
        stdout="logs/war_standardize/{animal}.out",
        stderr="logs/war_standardize/{animal}.err",
    script:
        "../scripts/standardize_wars.py"