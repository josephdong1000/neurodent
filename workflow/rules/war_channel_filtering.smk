"""
WAR Channel Filtering Rules
===========================

Rules for applying different channel filtering approaches to fragment-filtered WARs.
This enables comparison between manual bad channel lists and LOF-based detection.
"""


rule war_channel_filtering_manual:
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
        stdout="logs/war_channel_filtering/{animal}-manual.out",
        stderr="logs/war_channel_filtering/{animal}-manual.err",
    script:
        "../scripts/filter_wars_channels.py"


rule war_channel_filtering_lof:
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
        stdout="logs/war_channel_filtering/{animal}-lof.out",
        stderr="logs/war_channel_filtering/{animal}-lof.err",
    script:
        "../scripts/filter_wars_channels.py"

