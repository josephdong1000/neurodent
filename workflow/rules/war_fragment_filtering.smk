"""
WAR Fragment Filtering Rules
============================

Rules for applying fragment and channel filtering to quality-filtered WARs.
This applies filter_all() with configurable parameters as a separate step
to ensure all downstream analysis uses consistently filtered data.
"""


rule filter_wars_fragments:
    """
    Apply fragment and channel filtering to quality-filtered WARs
    """
    input:
        war_pkl="results/wars_quality_filtered/{animal}/war.pkl",
        war_json="results/wars_quality_filtered/{animal}/war.json",
    output:
        war_pkl="results/wars_fragment_filtered/{animal}/war.pkl",
        war_json="results/wars_fragment_filtered/{animal}/war.json",
    threads:
        config["cluster"]["war_fragment_filtering"]["threads"]
    params:
        config=config,
    resources:
        time=config["cluster"]["war_fragment_filtering"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_fragment_filtering"]["mem_mb"]),
        nodes=config["cluster"]["war_fragment_filtering"]["nodes"],
    log:
        "logs/war_fragment_filtering/{animal}.log",
    script:
        "../scripts/filter_wars_fragments.py"