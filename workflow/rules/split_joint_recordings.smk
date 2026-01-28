"""
Joint Recording Splitting Rules
==============================

Rules for splitting joint multi-animal recordings into per-animal zarr files.
This must run BEFORE war_generation for any sessions listed in
samples.json["joint_sessions"].

Configuration:
--------------
In samples.json, add a "joint_sessions" section:

{
    "joint_sessions": {
        "session_folder_name": {
            "AnimalA": ["Ch0", "Ch1", "Ch2", "Ch3"],
            "AnimalB": ["Ch4", "Ch5", "Ch6", "Ch7"]
        }
    }
}

Sessions NOT listed here are processed normally (single-animal).
"""


def get_joint_sessions():
    """Get list of session folders that need splitting."""
    return list(samples_config.get("joint_sessions", {}).keys())


rule split_joint_recordings:
    """
    Split joint multi-animal recordings into per-animal zarr files.
    
    This rule:
    1. Loads the joint recording as an AnimalOrganizer
    2. Splits by channel groups defined in samples.json
    3. Persists each animal's data as zarr
    
    Downstream rules (make_war) will automatically read from split outputs
    for animals that come from joint sessions.
    """
    output:
        directory("results/split_recordings/{session}")
    params:
        session=lambda wc: wc.session,
        joint_config=lambda wc: samples_config["joint_sessions"][wc.session],
        data_parent=samples_config["data_parent_folder"],
        split_config=config["analysis"]["split_recordings"],
        samples_config=samples_config,
    threads: config["cluster"]["split_joint_recordings"]["threads"]
    resources:
        time=config["cluster"]["split_joint_recordings"]["time"],
        mem_mb=increment_memory(config["cluster"]["split_joint_recordings"]["mem_mb"]),
        nodes=config["cluster"]["split_joint_recordings"]["nodes"],
    log:
        stdout="logs/split_joint_recordings/{session}.stdout",
        stderr="logs/split_joint_recordings/{session}.stderr",
    script:
        "../scripts/split_joint_recordings.py"


def get_split_sessions_to_run():
    """Get list of session wildcards for all joint sessions."""
    return expand(
        "results/split_recordings/{session}",
        session=get_joint_sessions()
    )
