"""
WAR Fragment Filtering Rules
============================

Rules for applying fragment and channel filtering to quality-filtered WARs.
This applies filter_all() with configurable parameters as a separate step
to ensure all downstream analysis uses consistently filtered data.
"""


checkpoint war_fragment_filtering:
    """
    Apply fragment filtering to standardized WARs (no channel filtering)

    The script streams the WAR, so its memory follows one row group and one batch of it,
    plus what the filters keep per window (rule_resources, war_loading="streamed").
    """
    input:
        war_parquet="results/wars_standardized/{animal}/war.parquet",
        war_json="results/wars_standardized/{animal}/war.json",
    output:
        war_parquet="results/wars_fragment_filtered/{animal}/war.parquet",
        war_json="results/wars_fragment_filtered/{animal}/war.json",
    threads:
        config["cluster"]["war_fragment_filter"]["threads"]
    retries:
        config["cluster"]["war_fragment_filter"]["retries"]
    params:
        config=analysis_config(config),
        samples_config=samples_config,
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
    resources:
        **rule_resources(config["cluster"]["war_fragment_filter"], war_loading="streamed"),
    log:
        stdout="logs/war_fragment_filtering/{animal}.out",
        stderr="logs/war_fragment_filtering/{animal}.err",
    script:
        "../scripts/filter_wars_fragments.py"