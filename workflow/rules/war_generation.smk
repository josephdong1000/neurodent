"""
WAR Generation Rules
===================

Rules for generating Windowed Analysis Results (WARs) from raw EEG data.
This corresponds to the pipeline-war-* scripts in the original workflow.
"""

rule generate_war:
    """
    Generate WAR (Windowed Analysis Results) for a specific animal
    """
    output:
        war_pkl = "results/wars/{animal}/war.pkl",
        war_json = "results/wars/{animal}/war.json",
        metadata = "results/wars/{animal}/metadata.json"
    params:
        animal_folder = get_animal_folder,
        animal_id = get_animal_id,
        samples_config = samples_config,
        config = config,
    threads:
        config['cluster']['war_generation']['cpu']
    resources:
        time = config["cluster"]["war_generation"]["time"],
        mem_mb = increment_memory(config["cluster"]["war_generation"]["mem"]),
        nodes = config["cluster"]["war_generation"]["nodes"],
    log:
        "logs/war_generation/{animal}.log",
    script:
        "../scripts/generate_wars.py"

rule war_generation_summary:
    """
    Create a summary report of WAR generation
    """
    input:
        wars = expand("results/wars/{animal}/war.pkl", animal=ANIMALS)
    output:
        summary = "results/wars/generation_summary.txt"
    shell:
        """
        echo "WAR Generation Complete" > {output.summary}
        echo "Total WARs generated: $(ls {input.wars} | wc -l)" >> {output.summary}
        echo "Generated at: $(date)" >> {output.summary}
        """