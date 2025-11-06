"""
WAR Generation Rules
===================

Rules for generating Windowed Analysis Results (WARs) from raw EEG data.
This corresponds to the pipeline-war-* scripts in the original workflow.
"""


rule make_war:
    """
    Generate WAR (Windowed Analysis Results) with integrated spike detection for a specific animal.

    This rule:
    1. Generates WAR with all frequency-domain features
    2. Runs frequency-domain spike detection (FDSAR)
    3. Integrates spike features (nspike, lognspike) into WAR
    4. Saves both WAR and FDSAR results
    """
    output:
        war_pkl="results/wars/{animal}/war.pkl",
        war_json="results/wars/{animal}/war.json",
        fdsar_dir=directory("results/fdsars/{animal}"),
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
