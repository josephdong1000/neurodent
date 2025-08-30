"""
Diagnostic Figures Rules
=======================

Rules for generating diagnostic figures from WARs.
This corresponds to the pipeline-apfig-* functionality in the original workflow.
"""

rule generate_diagnostic_figures:
    """
    Generate diagnostic figures for a specific animal
    """
    input:
        war_pkl="results/wars/{animal}/war.pkl",
        war_json="results/wars/{animal}/war.json",
        metadata="results/wars/{animal}/metadata.json"
    output:
        coherecorr_spectral="results/diagnostic_figures/{animal}/coherecorr_spectral.png",
        psd_histogram="results/diagnostic_figures/{animal}/psd_histogram.png",
        psd_spectrogram="results/diagnostic_figures/{animal}/psd_spectrogram.png"
    params:
        samples_config=samples_config,
        config=config
    resources:
        mem_mb=102400,  # 100GB  
        cpus_per_task=4,
        runtime=2880,   # 48 hours
    log:
        "logs/diagnostic_figures/{animal}.log"
    script:
        "../scripts/generate_diagnostic_figs.py"

rule diagnostic_figures_summary:
    """
    Create a summary report of diagnostic figures generation
    """
    input:
        coherecorr=expand("results/diagnostic_figures/{animal}/coherecorr_spectral.png", animal=ANIMALS),
        histograms=expand("results/diagnostic_figures/{animal}/psd_histogram.png", animal=ANIMALS),
        spectrograms=expand("results/diagnostic_figures/{animal}/psd_spectrogram.png", animal=ANIMALS)
    output:
        summary="results/diagnostic_figures/figures_summary.txt"
    shell:
        """
        echo "Diagnostic Figures Complete" > {output.summary}
        echo "Total coherecorr figures: $(ls {input.coherecorr} | wc -l)" >> {output.summary}
        echo "Total histogram figures: $(ls {input.histograms} | wc -l)" >> {output.summary}
        echo "Total spectrogram figures: $(ls {input.spectrograms} | wc -l)" >> {output.summary}
        echo "Generated at: $(date)" >> {output.summary}
        """