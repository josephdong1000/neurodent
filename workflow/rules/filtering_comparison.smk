"""
Filtering Comparison Rules
==========================

Rules for comparing manual vs LOF channel filtering approaches.
Generates analysis plots showing differences in feature extraction
between the two filtering methods.
"""


rule generate_filtering_comparison:
    """
    Generate comparison plots between manual and LOF channel filtering
    """
    input:
        manual_wars=get_all_flattened_manual_wars,
        lof_wars=get_all_flattened_lof_wars,
    output:
        comparison_dir=directory("results/filtering_comparison_plots/"),
        comparison_data=directory("results/filtering_comparison_data/"),
    params:
        config=config,
        samples_config=samples_config,
    threads: config["cluster"]["filtering_comparison"]["threads"]
    retries: 0
    resources:
        time=config["cluster"]["filtering_comparison"]["time"],
        mem_mb=increment_memory(config["cluster"]["filtering_comparison"]["mem_mb"]),
        nodes=config["cluster"]["filtering_comparison"]["nodes"],
    log:
        "logs/filtering_comparison/generate_comparison.log",
    script:
        "../scripts/generate_filtering_comparison.py"