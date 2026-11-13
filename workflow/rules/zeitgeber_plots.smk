"""
Zeitgeber Temporal Analysis Rules
=================================

Rules for zeitgeber time (ZT) temporal analysis and visualization.
Generates circadian plots showing features over 24-48 hour cycles 
from zeitgeber-processed features.
"""


rule zeitgeber_plots:
    """
    Generate zeitgeber time temporal plots showing circadian patterns
    """
    input:
        zeitgeber_features="results/wars_zeitgeber/zeitgeber_features.pkl",
    output:
        figure_dir=directory("results/zeitgeber_plots/"),
        data_dir=directory("results/zeitgeber_plot_data/"),
    params:
        config=analysis_config(config),
        samples_config=samples_config,
    threads:
        config["cluster"]["zeitgeber_plots"]["threads"]
    retries:
        config["cluster"]["zeitgeber_plots"]["retries"]
    resources:
        **rule_resources(config["cluster"]["zeitgeber_plots"]),
    log:
        stdout="logs/zeitgeber_plots/zeitgeber_plots.out",
        stderr="logs/zeitgeber_plots/zeitgeber_plots.err",
    script:
        "../scripts/generate_zeitgeber_plots.py"