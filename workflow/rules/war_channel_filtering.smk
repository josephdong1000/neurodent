"""
WAR Channel Filtering Rules
===========================

Rules for applying different channel filtering approaches to fragment-filtered WARs.
This enables comparison between manual bad channel lists and LOF-based detection.
"""


def _war_fragment_filtered_parquet(wildcards):
    """Return path to fragment-filtered war.parquet after the checkpoint has run."""
    return checkpoints.war_fragment_filtering.get(animal=wildcards.animal).output.war_parquet


def _war_fragment_filtered_json(wildcards):
    """Return path to fragment-filtered war.json after the checkpoint has run."""
    return checkpoints.war_fragment_filtering.get(animal=wildcards.animal).output.war_json


rule war_channel_filtering_manual:
    """
    Apply manual bad channel filtering using config/samples.json bad channel lists
    """
    input:
        war_parquet=_war_fragment_filtered_parquet,
        war_json=_war_fragment_filtered_json,
    output:
        war_parquet="results/wars_channel_filtered_manual/{animal}/war.parquet",
        war_json="results/wars_channel_filtered_manual/{animal}/war.json",
    threads: 1
    retries:
        config["cluster"]["war_channel_filtering"]["retries"]
    params:
        config=config,
        samples_config=samples_config,
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        filter_type="manual",
    resources:
        time=config["cluster"]["war_channel_filtering"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_channel_filtering"]["mem_mb"]),
        nodes=config["cluster"]["war_channel_filtering"]["nodes"],
    log:
        stdout="logs/war_channel_filtering_manual/{animal}.out",
        stderr="logs/war_channel_filtering_manual/{animal}.err",
    script:
        "../scripts/filter_wars_channels.py"


rule war_channel_filtering_lof:
    """
    Apply LOF-based bad channel filtering using pre-computed LOF scores
    """
    input:
        war_parquet=_war_fragment_filtered_parquet,
        war_json=_war_fragment_filtered_json,
    output:
        war_parquet="results/wars_channel_filtered_lof/{animal}/war.parquet",
        war_json="results/wars_channel_filtered_lof/{animal}/war.json",
    threads: 1
    retries:
        config["cluster"]["war_channel_filtering"]["retries"]
    params:
        config=config,
        samples_config=samples_config,
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        filter_type="lof",
    resources:
        time=config["cluster"]["war_channel_filtering"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_channel_filtering"]["mem_mb"]),
        nodes=config["cluster"]["war_channel_filtering"]["nodes"],
    log:
        stdout="logs/war_channel_filtering_lof/{animal}.out",
        stderr="logs/war_channel_filtering_lof/{animal}.err",
    script:
        "../scripts/filter_wars_channels.py"

