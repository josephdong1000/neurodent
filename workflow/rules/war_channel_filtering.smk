"""
WAR Channel Filtering Rules
===========================

Rules for applying different channel filtering approaches to fragment-filtered WARs.
This enables comparison between manual bad channel lists and LOF-based detection.
"""


rule war_channel_filter_manual:
    """
    Apply manual bad channel filtering using config/samples.json bad channel lists
    """
    input:
        war_pkl="results/wars_fragment_filtered/{animal}/war.pkl",
        war_json="results/wars_fragment_filtered/{animal}/war.json",
    output:
        war_pkl="results/wars_channel_filtered_manual/{animal}/war.pkl",
        war_json="results/wars_channel_filtered_manual/{animal}/war.json",
    threads: 1
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
        "logs/war_channel_filtering/{animal}_manual.log",
    script:
        "../scripts/filter_wars_channels.py"


rule war_channel_filter_lof:
    """
    Apply LOF-based bad channel filtering using pre-computed LOF scores
    """
    input:
        war_pkl="results/wars_fragment_filtered/{animal}/war.pkl",
        war_json="results/wars_fragment_filtered/{animal}/war.json",
    output:
        war_pkl="results/wars_channel_filtered_lof/{animal}/war.pkl",
        war_json="results/wars_channel_filtered_lof/{animal}/war.json",
    threads: 1
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
        "logs/war_channel_filtering/{animal}_lof.log",
    script:
        "../scripts/filter_wars_channels.py"


rule flatten_wars_manual:
    """
    Flatten manually channel-filtered WARs by aggregating time windows
    """
    input:
        war_pkl="results/wars_channel_filtered_manual/{animal}/war.pkl",
        war_json="results/wars_channel_filtered_manual/{animal}/war.json",
    output:
        war_pkl="results/wars_flattened_manual/{animal}/war.pkl",
        war_json="results/wars_flattened_manual/{animal}/war.json",
    threads: config["cluster"]["war_flattening"]["threads"]
    params:
        samples_config=samples_config,
        config=config,
    resources:
        time=config["cluster"]["war_flattening"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_flattening"]["mem_mb"]),
        nodes=config["cluster"]["war_flattening"]["nodes"],
    log:
        "logs/war_flattening/{animal}_manual.log",
    script:
        "../scripts/flatten_wars.py"


rule flatten_wars_lof:
    """
    Flatten LOF channel-filtered WARs by aggregating time windows
    """
    input:
        war_pkl="results/wars_channel_filtered_lof/{animal}/war.pkl",
        war_json="results/wars_channel_filtered_lof/{animal}/war.json",
    output:
        war_pkl="results/wars_flattened_lof/{animal}/war.pkl",
        war_json="results/wars_flattened_lof/{animal}/war.json",
    threads: config["cluster"]["war_flattening"]["threads"]
    params:
        samples_config=samples_config,
        config=config,
    resources:
        time=config["cluster"]["war_flattening"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_flattening"]["mem_mb"]),
        nodes=config["cluster"]["war_flattening"]["nodes"],
    log:
        "logs/war_flattening/{animal}_lof.log",
    script:
        "../scripts/flatten_wars.py"