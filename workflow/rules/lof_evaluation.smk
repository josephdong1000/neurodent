"""
LOF Evaluation Rules
===================

Rules for evaluating LOF (Local Outlier Factor) bad channel detection accuracy
against ground truth annotations from samples.json. Generates F-score vs threshold
analysis across all flattened WARs.
"""


rule lof_evaluation:
    """
    Evaluate LOF accuracy across all flattened WARs using ground truth bad channels
    """
    input:
        war_parquets=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_flattened_manual", filepath_append="war.parquet"),
        war_jsons=lambda wc: get_wars_after_quality_filtered(wc, filepath_prepend="results/wars_flattened_manual", filepath_append="war.json"),
    output:
        results_csv="results/lof_evaluation/lof_accuracy_results.csv",
        plot_png="results/lof_evaluation/lof_fscore_vs_threshold.png",
        barplot_png="results/lof_evaluation/lof_scores_by_channel.png",
    params:
        config=config,
        samples_config=samples_config,
        animal_folder_map=lambda wildcards: {animal: get_animal_folder(type('', (), {'animal': animal})) for animal in ANIMALS},
        animal_id_map=lambda wildcards: {animal: get_animal_id(type('', (), {'animal': animal})) for animal in ANIMALS},
    threads:
        config["cluster"]["lof_evaluation"]["threads"]
    retries:
        config["cluster"]["lof_evaluation"]["retries"]
    resources:
        time=config["cluster"]["lof_evaluation"]["time"],
        mem_mb=increment_memory(config["cluster"]["lof_evaluation"]["mem_mb"]),
        nodes=config["cluster"]["lof_evaluation"]["nodes"],
    log:
        stdout="logs/lof_evaluation/lof_evaluation.out",
        stderr="logs/lof_evaluation/lof_evaluation.err",
    script:
        "../scripts/evaluate_lof_accuracy.py"