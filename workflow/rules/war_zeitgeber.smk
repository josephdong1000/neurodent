"""
WAR Zeitgeber Analysis Rules
============================

Rules for extracting zeitgeber time features from fragment-filtered WARs.
This implements the pipeline-alphadelta.py functionality to process features
with respect to zeitgeber time rather than fragment index.
"""

rule war_zeitgeber:
    """
    Extract zeitgeber time features from the manually channel-filtered WARs.

    The same branch EP and relfreq read, so the configured bad channels are left out here too.
    This rule used to read the fragment-filtered WARs, one step earlier, so electrodes the config
    marks as bad were in every zeitgeber figure while every other figure left them out. Channel
    averaging uses nanmean, so a rejected channel drops out of the average instead of blanking it.
    """
    input:
        war_parquet=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_channel_filtered_manual", filepath_append="war.parquet"),
        war_json=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_channel_filtered_manual", filepath_append="war.json"),
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