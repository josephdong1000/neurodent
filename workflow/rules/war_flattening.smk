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
    params:
        samples_config=samples_config,
        config=config,
    resources:
        time=config["cluster"]["war_flattening"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_flattening"]["mem_mb"]),
        nodes=config["cluster"]["war_flattening"]["nodes"],
    log:
        "logs/war_flattening/{animal}.log",
    script:
        "../scripts/flatten_wars.py"
