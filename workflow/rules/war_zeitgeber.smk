"""
WAR Zeitgeber Analysis Rules
============================

Rules for extracting zeitgeber time features from fragment-filtered WARs.
This implements the pipeline-alphadelta.py functionality to process features
with respect to zeitgeber time rather than fragment index.
"""

rule war_zeitgeber:
    """
    Extract zeitgeber time features from all fragment-filtered WARs
    """
    input:
        war_parquet=get_all_fragment_filtered_parquet,
        war_json=get_all_fragment_filtered_json,
    output:
        zeitgeber_features="results/wars_zeitgeber/zeitgeber_features.pkl",
    threads:
        config["cluster"]["war_zeitgeber"]["threads"]
    retries:
        config["cluster"]["war_zeitgeber"]["retries"]
    params:
        config=config,
        samples_config=samples_config,
    resources:
        time=config["cluster"]["war_zeitgeber"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_zeitgeber"]["mem_mb"]),
        nodes=config["cluster"]["war_zeitgeber"]["nodes"],
    log:
        stdout="logs/war_zeitgeber/war_zeitgeber.out",
        stderr="logs/war_zeitgeber/war_zeitgeber.err",
    script:
        "../scripts/extract_zeitgeber_features.py"