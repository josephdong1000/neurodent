"""
WAR Fragment Filtering Rules
============================

Rules for applying fragment and channel filtering to quality-filtered WARs.
This applies filter_all() with configurable parameters as a separate step
to ensure all downstream analysis uses consistently filtered data.
"""


checkpoint war_fragment_filter:
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
        config["cluster"]["war_fragment_filter"]["threads"]
    params:
        config=config,
        samples_config=samples_config,
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
    resources:
        time=config["cluster"]["war_fragment_filter"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_fragment_filter"]["mem_mb"]),
        nodes=config["cluster"]["war_fragment_filter"]["nodes"],
    log:
        "logs/war_fragment_filter/{animal}.log",
    script:
        "../scripts/filter_wars_fragments.py"