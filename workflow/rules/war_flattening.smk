"""
WAR Flattening Rules
===================

Rules for flattening WARs and preparing them for final analysis.
This corresponds to the pipeline-epfig-so functionality in the original workflow.
"""


checkpoint flatten_wars:
    """
    Flatten filtered WARs by aggregating time windows for each animal individually
    """
    input:
        war_pkl="results/wars_fragment_filtered/{animal}/war.pkl",
        war_json="results/wars_fragment_filtered/{animal}/war.json",
    output:
        war_pkl="results/wars_flattened/{animal}/war.pkl",
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
        stdout="logs/war_flattening/{animal}.stdout",
        stderr="logs/war_flattening/{animal}.stderr",
    script:
        "../scripts/flatten_wars.py"


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
        stdout="logs/war_flattening/{animal}_manual.stdout",
        stderr="logs/war_flattening/{animal}_manual.stderr",
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
        stdout="logs/war_flattening/{animal}_lof.stdout",
        stderr="logs/war_flattening/{animal}_lof.stderr",
    script:
        "../scripts/flatten_wars.py"