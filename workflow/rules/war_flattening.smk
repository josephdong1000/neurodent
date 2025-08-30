"""
WAR Flattening Rules
===================

Rules for flattening WARs and preparing them for final analysis.
This corresponds to the pipeline-epfig-so functionality in the original workflow.
"""

rule flatten_wars:
    """
    Flatten and combine all WARs for final analysis
    """
    input:
        wars=expand("results/wars/{animal}/war.pkl", animal=ANIMALS)
    output:
        flattened_wars="results/flattened_wars/combined_wars.pkl",
        processing_log="results/flattened_wars/processing.log"
    params:
        samples_config=samples_config,
        config=config
    resources:
        mem_mb=409600,  # 400GB
        cpus_per_task=14,
        runtime=1440,   # 24 hours
    log:
        "logs/war_flattening/flatten_wars.log"
    script:
        "../scripts/flatten_wars.py"

rule war_flattening_summary:
    """
    Create a summary report of WAR flattening
    """
    input:
        flattened_wars="results/flattened_wars/combined_wars.pkl"
    output:
        summary="results/flattened_wars/flattening_summary.txt"
    shell:
        """
        echo "WAR Flattening Complete" > {output.summary}
        echo "Combined WAR file: {input.flattened_wars}" >> {output.summary}
        echo "File size: $(du -h {input.flattened_wars} | cut -f1)" >> {output.summary}
        echo "Generated at: $(date)" >> {output.summary}
        """