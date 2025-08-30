"""
Final Analysis Rules
===================

Rules for final analysis and EP figure generation.
This integrates with the test_Joseph notebooks for final outputs.
"""

rule generate_final_analysis:
    """
    Generate final EP figures and statistical analysis
    """
    input:
        flattened_wars="results/flattened_wars/combined_wars.pkl"
    output:
        completion_flag="results/final_analysis/experiment_plots_complete.flag",
        plots_dir=directory("results/final_analysis/plots"),
        stats_summary="results/final_analysis/statistical_summary.txt"
    params:
        samples_config=samples_config,
        config=config
    resources:
        mem_mb=409600,  # 400GB
        cpus_per_task=14,
        runtime=1440,   # 24 hours
    log:
        "logs/final_analysis/final_analysis.log"
    script:
        "../scripts/final_analysis.py"

rule final_analysis_summary:
    """
    Create a comprehensive summary report of the entire pipeline
    """
    input:
        wars_summary="results/wars/generation_summary.txt",
        temporal_summary="results/temporal_heatmaps/diagnostics_summary.txt",
        figures_summary="results/diagnostic_figures/figures_summary.txt",
        flattening_summary="results/flattened_wars/flattening_summary.txt",
        final_flag="results/final_analysis/experiment_plots_complete.flag"
    output:
        pipeline_summary="results/pipeline_summary.txt"
    shell:
        """
        echo "PyEEG Snakemake Pipeline Summary" > {output.pipeline_summary}
        echo "===============================" >> {output.pipeline_summary}
        echo "" >> {output.pipeline_summary}
        echo "Pipeline completed at: $(date)" >> {output.pipeline_summary}
        echo "" >> {output.pipeline_summary}
        
        echo "WAR Generation:" >> {output.pipeline_summary}
        cat {input.wars_summary} | tail -n +2 >> {output.pipeline_summary}
        echo "" >> {output.pipeline_summary}
        
        echo "Temporal Diagnostics:" >> {output.pipeline_summary}
        cat {input.temporal_summary} | tail -n +2 >> {output.pipeline_summary}
        echo "" >> {output.pipeline_summary}
        
        echo "Diagnostic Figures:" >> {output.pipeline_summary}
        cat {input.figures_summary} | tail -n +2 >> {output.pipeline_summary}
        echo "" >> {output.pipeline_summary}
        
        echo "WAR Flattening:" >> {output.pipeline_summary}
        cat {input.flattening_summary} | tail -n +2 >> {output.pipeline_summary}
        echo "" >> {output.pipeline_summary}
        
        echo "Final Analysis: Complete" >> {output.pipeline_summary}
        echo "All pipeline stages completed successfully." >> {output.pipeline_summary}
        """