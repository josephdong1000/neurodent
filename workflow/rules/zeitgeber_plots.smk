"""
Zeitgeber Temporal Analysis Rules
=================================

Rules for zeitgeber time (ZT) temporal analysis and visualization.
Generates circadian plots showing features over 24-48 hour cycles 
from zeitgeber-processed features.
"""


rule generate_zeitgeber_plots:
    """
    Generate zeitgeber time temporal plots showing circadian patterns
    """
    input:
        zeitgeber_features="results/wars_zeitgeber/zeitgeber_features.pkl",
    output:
        figure_dir=directory("results/zeitgeber_plots/"),
        data_dir=directory("results/zeitgeber_plot_data/"),
    params:
        config=config,
    threads:
        config["cluster"]["zeitgeber_plots"]["threads"]
    retries:
        config["cluster"]["zeitgeber_plots"]["retries"]
    resources:
        time=config["cluster"]["zeitgeber_plots"]["time"],
        mem_mb=increment_memory(config["cluster"]["zeitgeber_plots"]["mem_mb"]),
        nodes=config["cluster"]["zeitgeber_plots"]["nodes"],
    log:
        stdout="logs/zeitgeber_analysis/generate_zeitgeber_plots.stdout",
        stderr="logs/zeitgeber_analysis/generate_zeitgeber_plots.stderr",
    script:
        "../scripts/generate_zeitgeber_plots.py"