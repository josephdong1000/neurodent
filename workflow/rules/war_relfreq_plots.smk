"""
WAR Relative Frequency Plots Rules
===================================

Rules for generating relative frequency (distribution) plots from
channel-filtered, non-flattened WARs. These plots show the empirical
distributions of features across all time windows, providing richer
distributions than flattened data (n_animals × windows_per_animal datapoints).

Sister to extract_zeitgeber_features - both operate on channel-filtered,
non-flattened WARs before the flattening step.
"""


rule generate_relfreq_plots:
    """
    Generate relative frequency distribution plots from channel-filtered WARs
    """
    input:
        war_pkl=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_channel_filtered_manual", filepath_append="war.pkl"),
        war_json=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_channel_filtered_manual", filepath_append="war.json"),
    output:
        figure_dir=directory("results/relfreq_plots/"),
        data_dir=directory("results/relfreq_plot_data/"),
    params:
        config=config,
    threads:
        config["cluster"]["relfreq_plots"]["threads"]
    retries: 1
    resources:
        time=config["cluster"]["relfreq_plots"]["time"],
        mem_mb=increment_memory(config["cluster"]["relfreq_plots"]["mem_mb"]),
        nodes=config["cluster"]["relfreq_plots"]["nodes"],
    log:
        "logs/relfreq_plots/generate_relfreq_plots.log",
    script:
        "../scripts/generate_relfreq_plots.py"
