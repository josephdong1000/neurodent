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
        config["cluster"]["war_generation"]["threads"]
    resources:
        time = config["cluster"]["war_generation"]["time"],
        mem_mb = increment_memory(config["cluster"]["war_generation"]["mem_mb"]),
        nodes = config["cluster"]["war_generation"]["nodes"],
    log:
        "logs/war_generation/{animal}.log",
    script:
        "../scripts/generate_wars.py"

rule war_generation_summary:
    """
    Create a summary report of WAR generation with validation
    """
    input:
        wars_pkl = expand("results/wars/{animal}/war.pkl", animal=ANIMALS),
        wars_json = expand("results/wars/{animal}/war.json", animal=ANIMALS)
    output:
        summary = "results/wars/generation_summary.txt"
    localrule:
        True
    run:
        from pathlib import Path
        
        missing_files = []
        for animal in ANIMALS:
            pkl_file = Path(f"results/wars/{animal}/war.pkl")
            json_file = Path(f"results/wars/{animal}/war.json")
            if not pkl_file.exists():
                missing_files.append(str(pkl_file))
            if not json_file.exists():
                missing_files.append(str(json_file))
        
        if missing_files:
            raise FileNotFoundError(f"Missing WAR files: {missing_files}")
            
        with open(output.summary, 'w') as f:
            f.write("WAR Generation Complete\n")
            f.write(f"Total animals processed: {len(ANIMALS)}\n")
            f.write(f"WAR files generated: {len(input.wars_pkl)}\n") 
            f.write(f"JSON files generated: {len(input.wars_json)}\n")
            f.write(f"✓ All required files validated\n")
            f.write(f"Generated at: {__import__('datetime').datetime.now()}\n")