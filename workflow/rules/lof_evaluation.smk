"""
LOF Evaluation Rules
===================

Rules for evaluating LOF (Local Outlier Factor) bad channel detection accuracy
against ground truth annotations from samples.json. Generates F-score vs threshold
analysis across all flattened WARs.
"""


rule evaluate_lof_accuracy:
    """
    Evaluate LOF accuracy across all flattened WARs using ground truth bad channels
    """
    input:
        war_pkls=get_flattened_wars_pkl,
        war_jsons=get_flattened_wars_json,
    output:
        results_csv="results/lof_evaluation/lof_accuracy_results.csv",
        plot_png="results/lof_evaluation/lof_fscore_vs_threshold.png",
    params:
        config=config,
        samples_config=samples_config,
        animal_folder_map=lambda wildcards: {animal: get_animal_folder(type('', (), {'animal': animal})) for animal in ANIMALS},
        animal_id_map=lambda wildcards: {animal: get_animal_id(type('', (), {'animal': animal})) for animal in ANIMALS},
    threads:
        config["cluster"]["lof_evaluation"]["threads"]
    resources:
        time=config["cluster"]["lof_evaluation"]["time"],
        mem_mb=increment_memory(config["cluster"]["lof_evaluation"]["mem_mb"]),
        nodes=config["cluster"]["lof_evaluation"]["nodes"],
    log:
        "logs/lof_evaluation/evaluate_lof_accuracy.log",
    script:
        "../scripts/evaluate_lof_accuracy.py"