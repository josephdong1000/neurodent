"""
WAR Relative Frequency Plots Rules
===================================

Rules for generating relative frequency (distribution) plots from
channel-filtered, non-flattened WARs. These plots show the empirical
distributions of features across all time windows, providing richer
distributions than flattened data (n_animals × windows_per_animal datapoints).

Sister to war_zeitgeber - both operate on channel-filtered,
non-flattened WARs before the flattening step.
"""


rule war_relfreq_plots:
    """
    Generate relative frequency distribution plots from channel-filtered WARs

    The script loads the WARs one at a time, each once, so the rule needs one core and
    memory for the largest single WAR, about 47 GiB for IQSEC2. It used to load 15 at a
    time, each once per feature, and ran out of memory at 150 and then 300 GB.
    """
    input:
        war_parquet=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_channel_filtered_manual", filepath_append="war.parquet"),
        war_json=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_channel_filtered_manual", filepath_append="war.json"),
    output:
        figure_dir=directory("results/relfreq_plots/"),
        data_dir=directory("results/relfreq_plot_data/"),
    params:
        config=config,
        samples_config=samples_config,
    threads:
        config["cluster"]["relfreq_plots"]["threads"]
    retries:
        config["cluster"]["relfreq_plots"]["retries"]
    resources:
        time=config["cluster"]["relfreq_plots"]["time"],
        mem_mb=increment_memory(config["cluster"]["relfreq_plots"]["mem_mb"]),
        nodes=config["cluster"]["relfreq_plots"]["nodes"],
    log:
        stdout="logs/war_relfreq_plots/war_relfreq_plots.out",
        stderr="logs/war_relfreq_plots/war_relfreq_plots.err",
    script:
        "../scripts/generate_relfreq_plots.py"
