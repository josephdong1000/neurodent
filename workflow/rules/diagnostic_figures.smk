"""
Diagnostic Figures Rules
=======================

Rules for generating diagnostic figures from WARs using checkpoints.
This allows AnimalPlotter to generate variable numbers of files naturally.
"""

checkpoint generate_diagnostic_figures:
    """
    Generate diagnostic figures for a specific animal into subdirectories (filtered/unfiltered)
    """
    input:
        war_pkl="results/wars/{animal}/war.pkl",
        war_json="results/wars/{animal}/war.json",
        metadata="results/wars/{animal}/metadata.json"
    output:
        figure_dir=directory("results/diagnostic_figures/{animal}"),
        unfiltered_dir=directory("results/diagnostic_figures/{animal}/unfiltered"),
        filtered_dir=directory("results/diagnostic_figures/{animal}/filtered")
    params:
        animal_folder=get_animal_folder,
        animal_id=get_animal_id,
        samples_config=samples_config,
        config=config,
    threads:
        config["cluster"]["diagnostic_figures"]["threads"],
    retries:
        1
    resources:
        time=config["cluster"]["diagnostic_figures"]["time"],
        mem_mb=increment_memory(config["cluster"]["diagnostic_figures"]["mem_mb"]),
        nodes=config["cluster"]["diagnostic_figures"]["nodes"],
    log:
        "logs/diagnostic_figures/{animal}.log",
    script:
        "../scripts/generate_diagnostic_figs.py"


rule diagnostic_figures_summary:
    """
    Create a summary report of diagnostic figures generation with validation
    """
    input:
        figure_dirs=expand("results/diagnostic_figures/{animal}", animal=ANIMALS)
    output:
        summary="results/diagnostic_figures/figures_summary.txt"
    localrule:
        True
    run:
        from pathlib import Path
        
        required_types = ["psd_histogram", "coherecorr_spectral", "psd_spectrogram"]
        expected_subdirs = ["unfiltered", "filtered"]
        
        # Get processing configuration details for summary
        processing_config = config.get("analysis", {}).get("processing", {})
        filtering_config = processing_config.get("filtering", {})
        
        total_figures = {subdir: {fig_type: 0 for fig_type in required_types} for subdir in expected_subdirs}
        missing_types = []
        animals_processed = 0
        
        for animal in ANIMALS:
            fig_dir = Path(f"results/diagnostic_figures/{animal}")
            if not fig_dir.exists():
                missing_types.append(f"Directory missing: {fig_dir}")
                continue
                
            animals_processed += 1
            
            # Check each expected subdirectory
            for subdir in expected_subdirs:
                subdir_path = fig_dir / subdir
                if not subdir_path.exists():
                    missing_types.append(f"Subdirectory missing: {subdir_path}")
                    continue
                    
                # Check for required figure types in this subdirectory
                for fig_type in required_types:
                    matching_files = list(subdir_path.glob(f"*{fig_type}*"))
                    if not matching_files:
                        missing_types.append(f"No {fig_type} figure found in {animal}/{subdir}")
                    else:
                        total_figures[subdir][fig_type] += len(matching_files)
        
        if missing_types:
            raise FileNotFoundError(f"Missing diagnostic figures: {missing_types}")
        
        with open(output.summary, 'w') as f:
            f.write("Diagnostic Figures Summary\n")
            f.write("=" * 26 + "\n\n")
            f.write(f"Animals processed: {animals_processed}\n")
            f.write(f"Structure: Both filtered and unfiltered versions\n\n")
            
            # Processing configuration details
            f.write("Processing Configuration:\n")
            f.write("=" * 25 + "\n")
            
            # Preprocessing parameters
            f.write("Preprocessing (channel setup):\n")
            f.write(f"  channel_reorder: {processing_config.get('channel_reorder', [])}\n")
            f.write(f"  use_abbrevs: {processing_config.get('use_abbrevs', True)}\n")
            f.write("\n")
            
            # Filtering parameters - report all specified parameters
            f.write("Filtering parameters passed to filter_all():\n")
            if filtering_config:
                for param, value in filtering_config.items():
                    f.write(f"  {param}: {value}\n")
            else:
                f.write("  (no filtering parameters specified - using filter_all defaults)\n")
            f.write("\n")
            
            # Aggregation parameters
            aggregation_config = processing_config.get("aggregation", {})
            f.write("Post-processing (aggregation):\n")
            if aggregation_config:
                for param, value in aggregation_config.items():
                    f.write(f"  {param}: {value}\n")
            else:
                f.write("  (no aggregation parameters specified)\n")
            f.write("\n")
            
            f.write("Figure Counts:\n")
            f.write("=" * 14 + "\n")
            f.write("Unfiltered Figures (raw data with channel reordering only):\n")
            f.write(f"  - Histogram figures: {total_figures['unfiltered']['psd_histogram']}\n")
            f.write(f"  - Coherecorr figures: {total_figures['unfiltered']['coherecorr_spectral']}\n")  
            f.write(f"  - Spectrogram figures: {total_figures['unfiltered']['psd_spectrogram']}\n")
            f.write("\n")
            
            f.write("Filtered Figures (with artifact rejection and smoothing):\n")
            f.write(f"  - Histogram figures: {total_figures['filtered']['psd_histogram']}\n")
            f.write(f"  - Coherecorr figures: {total_figures['filtered']['coherecorr_spectral']}\n")  
            f.write(f"  - Spectrogram figures: {total_figures['filtered']['psd_spectrogram']}\n")
            f.write("\n")
            
            f.write("✓ All required figure types validated for both filtered and unfiltered\n")
            f.write(f"\nGenerated at: {__import__('datetime').datetime.now()}\n")