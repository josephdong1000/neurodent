"""
WAR Zeitgeber Analysis Rules
============================

Rules for extracting zeitgeber time features from fragment-filtered WARs.
This implements the pipeline-alphadelta.py functionality to process features
with respect to zeitgeber time rather than fragment index.
"""

rule extract_zeitgeber_features:
    """
    Extract zeitgeber time features from all fragment-filtered WARs
    """
    input:
        war_pkl=get_all_fragment_filtered_pkl,
        war_json=get_all_fragment_filtered_json,
    output:
        zeitgeber_features="results/wars_zeitgeber/zeitgeber_features.pkl",
    threads:
        config["cluster"]["war_zeitgeber"]["threads"]
    params:
        config=config,
    resources:
        time=config["cluster"]["war_zeitgeber"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_zeitgeber"]["mem_mb"]),
        nodes=config["cluster"]["war_zeitgeber"]["nodes"],
    log:
        "logs/war_zeitgeber/extract_zeitgeber_features.log",
    script:
        "../scripts/extract_zeitgeber_features.py"