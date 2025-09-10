"""
WAR Generation Rules
===================

Rules for generating Windowed Analysis Results (WARs) from raw EEG data.
This corresponds to the pipeline-war-* scripts in the original workflow.
"""


rule make_war:
    """
    Generate WAR (Windowed Analysis Results) for a specific animal
    """
    output:
        war_pkl="results/wars/{animal}/war.pkl",
        war_json="results/wars/{animal}/war.json",
    params:
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        samples_config=samples_config,
        config=config,
    threads: config["cluster"]["war_generation"]["threads"]
    resources:
        time=config["cluster"]["war_generation"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_generation"]["mem_mb"]),
        nodes=config["cluster"]["war_generation"]["nodes"],
    log:
        "logs/war_generation/{animal}.log",
    script:
        "../scripts/generate_wars.py"
