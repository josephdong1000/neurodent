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
        animal_folders=get_animal_folders,
        animal_id=get_animal_id,
        is_split_recording=lambda wc: wc.animal in JOINT_ANIMAL_TO_SESSION,
        channel_subset=get_joint_session_channels,  # None for regular animals, list for joint sessions
        config=config,
        samples_config=samples_config,
    threads: config["cluster"]["war_generation"]["threads"]
    retries:
        config["cluster"]["war_generation"]["retries"]
    resources:
        time=config["cluster"]["war_generation"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_generation"]["mem_mb"]),
        nodes=config["cluster"]["war_generation"]["nodes"],
    log:
        stdout="logs/war_generation/{animal}.stdout",
        stderr="logs/war_generation/{animal}.stderr",
    script:
        "../scripts/generate_wars.py"
