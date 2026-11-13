"""
WAR Zeitgeber Analysis Rules
============================

Rules for extracting zeitgeber time features from manually channel-filtered WARs.
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

    The script loads the WARs one at a time, each once, so the rule needs one core. Its memory
    and time are computed from the input WARs when the job is scheduled, from the largest WAR
    and from all windows together (see neurodent.workflow.utils.resources), so they follow the
    dataset. On IQSEC2 the largest WAR alone peaks at about 47 GiB. The script used to load 15
    at a time and peaked at 270 of its 293 GiB.
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
        config=analysis_config(config),
        samples_config=samples_config,
    resources:
        **rule_resources(config["cluster"]["war_zeitgeber"], war_loading="whole"),
    log:
        stdout="logs/war_zeitgeber/war_zeitgeber.out",
        stderr="logs/war_zeitgeber/war_zeitgeber.err",
    script:
        "../scripts/extract_zeitgeber_features.py"