"""
WAR Flattening Rules
===================

Rules for flattening WARs and preparing them for final analysis.
This corresponds to the pipeline-epfig-so functionality in the original workflow.
"""


def _war_fragment_filtered_parquet(wildcards):
    """Return path to fragment-filtered war.parquet after the checkpoint has run."""
    return checkpoints.war_fragment_filtering.get(animal=wildcards.animal).output.war_parquet


def _war_fragment_filtered_json(wildcards):
    """Return path to fragment-filtered war.json after the checkpoint has run."""
    return checkpoints.war_fragment_filtering.get(animal=wildcards.animal).output.war_json


checkpoint war_flattening:
    """
    Flatten filtered WARs by aggregating time windows for each animal individually
    """
    input:
        war_parquet=_war_fragment_filtered_parquet,
        war_json=_war_fragment_filtered_json,
    output:
        war_parquet="results/wars_flattened/{animal}/war.parquet",
        war_json="results/wars_flattened/{animal}/war.json",
    threads: config["cluster"]["war_flattening"]["threads"]
    retries:
        config["cluster"]["war_flattening"]["retries"]
    params:
        config=config,
        samples_config=samples_config,
    resources:
        time=config["cluster"]["war_flattening"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_flattening"]["mem_mb"]),
        nodes=config["cluster"]["war_flattening"]["nodes"],
    log:
        stdout="logs/war_flattening/{animal}.out",
        stderr="logs/war_flattening/{animal}.err",
    script:
        "../scripts/flatten_wars.py"


rule war_flattening_manual:
    """
    Flatten manually channel-filtered WARs by aggregating time windows
    """
    input:
        war_parquet="results/wars_channel_filtered_manual/{animal}/war.parquet",
        war_json="results/wars_channel_filtered_manual/{animal}/war.json",
    output:
        war_parquet="results/wars_flattened_manual/{animal}/war.parquet",
        war_json="results/wars_flattened_manual/{animal}/war.json",
    threads: config["cluster"]["war_flattening"]["threads"]
    retries:
        config["cluster"]["war_flattening"]["retries"]
    params:
        config=config,
        samples_config=samples_config,
    resources:
        time=config["cluster"]["war_flattening"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_flattening"]["mem_mb"]),
        nodes=config["cluster"]["war_flattening"]["nodes"],
    log:
        stdout="logs/war_flattening_manual/{animal}.out",
        stderr="logs/war_flattening_manual/{animal}.err",
    script:
        "../scripts/flatten_wars.py"


rule war_flattening_lof:
    """
    Flatten LOF channel-filtered WARs by aggregating time windows
    """
    input:
        war_parquet="results/wars_channel_filtered_lof/{animal}/war.parquet",
        war_json="results/wars_channel_filtered_lof/{animal}/war.json",
    output:
        war_parquet="results/wars_flattened_lof/{animal}/war.parquet",
        war_json="results/wars_flattened_lof/{animal}/war.json",
    threads: config["cluster"]["war_flattening"]["threads"]
    retries:
        config["cluster"]["war_flattening"]["retries"]
    params:
        config=config,
        samples_config=samples_config,
    resources:
        time=config["cluster"]["war_flattening"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_flattening"]["mem_mb"]),
        nodes=config["cluster"]["war_flattening"]["nodes"],
    log:
        stdout="logs/war_flattening_lof/{animal}.out",
        stderr="logs/war_flattening_lof/{animal}.err",
    script:
        "../scripts/flatten_wars.py"