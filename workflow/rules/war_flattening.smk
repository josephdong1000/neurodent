"""
WAR Flattening Rules
===================

Rules for flattening WARs and preparing them for final analysis.
This corresponds to the pipeline-epfig-so functionality in the original workflow.
"""


rule flatten_wars:
    """
    Flatten and combine filtered WARs for final analysis
    """
    input:
        wars=lambda wildcards: [f"results/wars_filtered/{animal}/war.pkl" for animal in get_filtered_animals(wildcards)],
    output:
        flattened_wars="results/flattened_wars/combined_wars.pkl",
        processing_log="results/flattened_wars/processing.log",
    threads: config["cluster"]["war_flattening"]["threads"]
    params:
        samples_config=samples_config,
        config=config,
    resources:
        time=config["cluster"]["war_flattening"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_flattening"]["mem_mb"]),
        nodes=config["cluster"]["war_flattening"]["nodes"],
    log:
        "logs/war_flattening/flatten_wars.log",
    script:
        "../scripts/flatten_wars.py"
