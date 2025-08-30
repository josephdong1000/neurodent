"""
Temporal Diagnostics Rules
=========================

Rules for generating temporal heatmaps and diagnostic plots at the animal level.
This corresponds to the pipeline-artifact functionality in the original workflow.
"""

rule generate_temporal_heatmap:
    """
    Generate temporal heatmap for a specific animal
    """
    input:
        war_pkl="results/wars/{animal}/war.pkl",
        war_json="results/wars/{animal}/war.json",
        metadata="results/wars/{animal}/metadata.json"
    output:
        heatmap="results/temporal_heatmaps/{animal}/heatmap.png",
        processed_war="results/temporal_heatmaps/{animal}/processed_war.pkl"
    params:
        config=config
    resources:
        mem_mb=51200,  # 50GB
        cpus_per_task=4,
        runtime=720,   # 12 hours
    log:
        "logs/temporal_diagnostics/{animal}.log"
    script:
        "../scripts/create_temporal_heatmaps.py"

rule temporal_diagnostics_summary:
    """
    Create a summary report of temporal diagnostics generation
    """
    input:
        heatmaps=expand("results/temporal_heatmaps/{animal}/heatmap.png", animal=ANIMALS)
    output:
        summary="results/temporal_heatmaps/diagnostics_summary.txt"
    shell:
        """
        echo "Temporal Diagnostics Complete" > {output.summary}
        echo "Total heatmaps generated: $(ls {input.heatmaps} | wc -l)" >> {output.summary}
        echo "Generated at: $(date)" >> {output.summary}
        """