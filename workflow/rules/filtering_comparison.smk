"""
Filtering Comparison Rules
==========================

Rules for comparing manual vs LOF channel filtering approaches.
Generates analysis plots showing differences in feature extraction
between the two filtering methods.
"""


rule filtering_comparison:
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
        config=analysis_config(config),
        samples_config=samples_config,
    threads: config["cluster"]["filtering_comparison"]["threads"]
    retries:
        config["cluster"]["filtering_comparison"]["retries"]
    resources:
        **rule_resources(config["cluster"]["filtering_comparison"]),
    log:
        stdout="logs/filtering_comparison/filtering_comparison.out",
        stderr="logs/filtering_comparison/filtering_comparison.err",
    script:
        "../scripts/generate_filtering_comparison.py"