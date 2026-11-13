#!/usr/bin/env python3
"""
Zeitgeber Feature Extraction Script
===================================

This script implements the pipeline-alphadelta.py functionality in the Snakemake workflow.
It processes the manually channel-filtered WARs to extract features with zeitgeber time
information and creates a single concatenated dataframe across all animals.

Memory: each WAR is loaded in full exactly once, one animal at a time in sorted animal
order, through neurodent.workflow.iter_wars, and only its zeitgeber bins are kept: its
channel-averaged windows, averaged per bin by aggregate_zeitgeber_bins. This replaced a pool
that loaded up to `threads` WARs at once, which on IQSEC2 peaked at 270 of the 293 GiB the
rule had. The rule's threads setting is no longer used. The order is sorted, so the output
does not depend on the order of the inputs.

Every configured feature must be in every WAR: a missing column raises, naming the animal.
The columns are named by channel_averaged_columns, so psdslope gives psdslope_slope and
psdslope_intercept. Every animal's windows used to be kept and averaged together at the end,
with columns picked by a rule that left out psdslope's two, and columns it did not find were
left out without a word.

Based on: notebooks/examples/pipeline-alphadelta.py
"""

import logging
from pathlib import Path
import pandas as pd

from neurodent.results.zeitgeber import _extract_zeitgeber_frame
from neurodent.workflow import setup_snakemake_logging, apply_samples_config, iter_wars

logger = logging.getLogger(__name__)


def main():
    """Main zeitgeber feature extraction function"""
    global snakemake

    setup_snakemake_logging(snakemake)

    # Get parameters from snakemake
    input_war_parquets = snakemake.input.war_parquet
    input_war_jsons = snakemake.input.war_json
    output_pkl = snakemake.output.zeitgeber_features
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    apply_samples_config(samples_config)

    # Get zeitgeber processing parameters from config
    zeitgeber_params = config["analysis"]["zeitgeber"]
    features_to_extract = zeitgeber_params["features"]

    logger.info(f"Processing {len(input_war_parquets)} channel-filtered WARs")
    logger.info(f"Features to extract: {features_to_extract}")

    # Validate that parquet and JSON inputs match
    if len(input_war_parquets) != len(input_war_jsons):
        raise ValueError(
            f"Mismatch between parquet files ({len(input_war_parquets)}) and JSON files ({len(input_war_jsons)})"
        )

    # Create animal name to file path mappings
    parquet_animals = {}
    json_animals = {}

    for parquet_path in input_war_parquets:
        parquet_path_obj = Path(parquet_path)
        animal_name = parquet_path_obj.parent.name
        parquet_animals[animal_name] = parquet_path_obj

    for json_path in input_war_jsons:
        json_path_obj = Path(json_path)
        animal_name = json_path_obj.parent.name
        json_animals[animal_name] = json_path_obj

    # Validate that all animals have both parquet and JSON files
    parquet_animal_set = set(parquet_animals.keys())
    json_animal_set = set(json_animals.keys())

    if parquet_animal_set != json_animal_set:
        missing_json = parquet_animal_set - json_animal_set
        missing_parquet = json_animal_set - parquet_animal_set
        error_msg = []
        if missing_json:
            error_msg.append(f"Animals missing JSON files: {missing_json}")
        if missing_parquet:
            error_msg.append(f"Animals missing parquet files: {missing_parquet}")
        raise ValueError("; ".join(error_msg))

    logger.info(f"Validated {len(parquet_animal_set)} animals have both parquet and JSON files")

    # Every other key of the zeitgeber block is a pipeline setting (interval_minutes,
    # baseline_hours, exclude_from_baseline, ...), and ZeitgeberAnalysisResult refuses a key it
    # does not read, as the config schema does.
    pipeline_config = {key: value for key, value in zeitgeber_params.items() if key != "features"}
    logger.info(f"Zeitgeber settings: {pipeline_config}")
    # shift_for_48h=False: We want canonical features here, not plotting data
    pipeline_config["shift_for_48h"] = False

    # Use ANIMAL_METADATA (injected by apply_samples_config, required)
    from neurodent import constants
    pipeline_config["animal_metadata"] = constants.ANIMAL_METADATA

    # Load each WAR once, one animal at a time, in sorted animal order, and keep only its
    # zeitgeber bins
    animals = sorted(parquet_animal_set)
    dfs = []
    for parquet_path, war in iter_wars(
        [parquet_animals[animal] for animal in animals],
        [json_animals[animal] for animal in animals],
    ):
        dfs.append(
            _extract_zeitgeber_frame(
                war, parquet_path.parent.name, features_to_extract, pipeline_config
            )
        )
        del war  # iter_wars loads the next WAR only after this one is released

    if not dfs:
        logger.error("No valid WARs were processed!")
        raise RuntimeError("No valid WARs found for zeitgeber analysis")

    logger.info(f"Successfully processed {len(dfs)} WARs")

    # Each animal's frame is already averaged per bin, by aggregate_zeitgeber_bins
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Aggregated dataframe shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Create output directory
    output_dir = Path(output_pkl).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to pickle
    df.to_pickle(output_pkl)
    logger.info(f"Saved zeitgeber features to: {output_pkl}")

    logger.info("Zeitgeber feature extraction completed successfully")


if __name__ == "__main__":
    main()
