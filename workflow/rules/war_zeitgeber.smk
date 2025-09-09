"""
WAR Zeitgeber Analysis Rules
============================

Rules for extracting zeitgeber time features from fragment-filtered WARs.
This implements the pipeline-alphadelta.py functionality to process features
with respect to zeitgeber time rather than fragment index.
"""


def get_fragment_filtered_animals(wildcards):
    """Discover animals that have fragment-filtered WARs available"""
    import os
    filtered_animals = []
    filtered_dir = "results/wars_fragment_filtered"
    if os.path.exists(filtered_dir):
        for item in os.listdir(filtered_dir):
            if os.path.isdir(os.path.join(filtered_dir, item)):
                war_file = os.path.join(filtered_dir, item, "war.pkl")
                if os.path.exists(war_file):
                    filtered_animals.append(item)
    return filtered_animals


rule extract_zeitgeber_features:
    """
    Extract zeitgeber time features from all fragment-filtered WARs
    """
    input:
        wars=lambda wildcards: [f"results/wars_fragment_filtered/{animal}/war.pkl" 
                               for animal in get_fragment_filtered_animals(wildcards)], # FIXME maybe more appropriate as a glob operation, this feels ad-hoc
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