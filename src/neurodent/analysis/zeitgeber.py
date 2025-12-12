"""
Zeitgeber Time (ZT) Analysis Module
===================================
This module provides functionality for processing and analyzing data in Zeitgeber Time (ZT).
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from .. import visualization

logger = logging.getLogger(__name__)


def load_war_for_zeitgeber(war_path_info):
    """
    Load a fragment-filtered WAR and extract channel-averaged features for zeitgeber analysis.

    Args:
        war_path_info (tuple): Tuple containing:
            - war_pkl_path (Path): Path to the WAR pickle file.
            - war_json_path (Path): Path to the WAR JSON metadata file.
            - features_to_extract (list[str]): List of features to extract.
            - animal_name (str): Identifier for the animal.

    Returns:
        pd.DataFrame: DataFrame containing channel-averaged features and animal identifier.
                      The DataFrame will have columns for each extracted feature and an 'animal' column.
    """
    war_pkl_path, war_json_path, features_to_extract, animal_name = war_path_info

    try:
        logger.info(f"Loading {animal_name}")

        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=war_pkl_path.parent,
            pickle_name=war_pkl_path.name,
            json_name=war_json_path.name,
        )

        df = war.get_channel_averaged_result(features=features_to_extract)
        df["animal"] = animal_name
        del war
        return df

    except Exception as e:
        logger.error(f"Failed to process {animal_name}: {str(e)}")
        raise


def add_zeitgeber_time_columns(df, interval_minutes=60):
    """
    Convert timestamps to zeitgeber time representation with specified binning interval.

    Args:
        df (pd.DataFrame): DataFrame containing a 'timestamp' column.
            Required columns:
            * 'timestamp': datetime64[ns] series representing the absolute time.
        interval_minutes (int, optional): Binning interval in minutes. Defaults to 60.
            Must evenly divide 1440 (24 hours).

    Returns:
        pd.DataFrame: DataFrame with added time columns:
            * 'hour' (int): Hour of the day (0-23).
            * 'minute' (int): Minute of the hour (0-59).
            * 'total_minutes' (int): Time of day in minutes from midnight (0-1440),
              binned to the nearest 'interval_minutes'.

    Raises:
        ValueError: If interval_minutes does not evenly divide 24 hours (1440 minutes).
    """
    if df is None or df.empty:
        return df

    if 1440 % interval_minutes != 0:
        raise ValueError(
            f"interval_minutes ({interval_minutes}) must evenly divide 24 hours (1440 minutes)."
        )

    logger.info(f"Adding binned zeitgeber time columns with {interval_minutes}min bins")

    df["hour"] = df["timestamp"].dt.hour.copy()
    df["minute"] = df["timestamp"].dt.minute.copy()

    raw_minutes = df["hour"] * 60 + df["minute"]
    binned_minutes = interval_minutes * (np.round(raw_minutes / interval_minutes))

    # Modulo 1440 to handle wraparound (e.g., 23:59 rounding up to 24:00 -> 0:00)
    df["total_minutes"] = binned_minutes % 1440

    return df


def subtract_zeitgeber_baseline(
    df, baseline_hours=12, baseline_window=None, exclude_from_baseline=None
):
    """
    Subtract baseline from numeric features in the dataframe.

    Baseline is calculated as the mean value within a specified window of time (ZT).
    Requires 'total_minutes' column to be present (usually added by add_zeitgeber_time_columns).

    Args:
        df (pd.DataFrame): DataFrame containing features and 'total_minutes'.
        baseline_hours (int, optional): Number of hours from start of ZT0 to use as baseline.
            Defaults to 12. Ignored if baseline_window is set.
        baseline_window (tuple | str, optional): Explicit window for baseline.
            Overrides baseline_hours. Defaults to None. Options:
            * tuple: (start_hour, end_hour)
            * "day": (0, 12)
            * "night": (12, 24)
        exclude_from_baseline (list[str], optional): Columns to exclude. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with new columns per feature: '{feature}_nobase'
            representing the baseline-corrected values.
    """
    if df.empty:
        return df

    if exclude_from_baseline is None:
        exclude_from_baseline = []

    if baseline_window is not None:
        if isinstance(baseline_window, str):
            if baseline_window == "day":
                start_min, end_min = 0, 12 * 60
            elif baseline_window == "night":
                start_min, end_min = 12 * 60, 24 * 60
            else:
                raise ValueError(f"Unknown baseline_window alias: {baseline_window}")
        elif isinstance(baseline_window, (tuple, list)) and len(baseline_window) == 2:
            start_min, end_min = baseline_window[0] * 60, baseline_window[1] * 60
        else:
            raise ValueError(
                "baseline_window must be 'day', 'night', or a (start, end) tuple."
            )

        logger.info(
            f"Using explicit baseline window: {baseline_window} ({start_min}-{end_min} min)"
        )
    else:
        start_min, end_min = 0, baseline_hours * 60
        logger.info(
            f"Using default baseline: first {baseline_hours} hours ({start_min}-{end_min} min)"
        )

    if "total_minutes" not in df.columns:
        logger.warning("Skipping baseline subtraction: 'total_minutes' not found.")
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip_cols = [
        "hour",
        "minute",
        "total_minutes",
        "animal_idx",
        "day_idx",
        "epoch_idx",
    ] + exclude_from_baseline

    features_to_correct = [
        c for c in numeric_cols if c not in skip_cols and not c.endswith("_nobase")
    ]

    group_cols = [c for c in ["animal", "sex", "gene"] if c in df.columns]

    result_df = df.copy()

    for feature in features_to_correct:
        if group_cols:

            def get_group_mean(g):
                vals = g.loc[
                    g["total_minutes"].between(start_min, end_min, inclusive="left"),
                    feature,
                ]
                return vals.mean() if not vals.empty else np.nan

            group_means = result_df.groupby(group_cols).apply(get_group_mean)

            # Note: set_index matches rows to the group index
            aligned_means = result_df.set_index(group_cols).index.map(group_means)

            result_df[f"{feature}_nobase"] = result_df[feature] - np.array(
                aligned_means
            )

        else:
            baseline_data = df.loc[
                df["total_minutes"].between(start_min, end_min, inclusive="left"),
                feature,
            ]
            if not baseline_data.empty:
                mean_val = baseline_data.mean()
                result_df[f"{feature}_nobase"] = result_df[feature] - mean_val
            else:
                logger.warning(
                    f"No data found in baseline window for feature {feature}"
                )
                result_df[f"{feature}_nobase"] = np.nan

    return result_df


def prepare_plot_data(df, shift_for_48h=True, perform_zt_shift=False):
    """
    Prepare data for plotting: enrichment, sorting, and optional 48h view duplication.

    Args:
        df (pd.DataFrame): Input dataframe.
            Expected format:
            * 'genotype' (str, optional): Strain info (e.g., 'M_WT', 'F_Mut').
              Used to derive 'sex' and 'gene' if not present.
            * 'total_minutes' (numeric): Time of day in minutes.
        shift_for_48h (bool, optional): If True, duplicates data shifted by 24h (1440 min)
            to create a 48h view. Defaults to True.
        perform_zt_shift (bool, optional): If True, shifts 'total_minutes' by -6 hours
            (Clock -> ZT conversion). Defaults to False.

    Returns:
        pd.DataFrame: Processed dataframe ready for plotting.
            Adds 'sex', 'gene', and temporary sorting columns if applicable.
    """
    df = df.copy()

    if "genotype" in df.columns and "sex" not in df.columns:
        df["sex"] = df["genotype"].str[0].map({"F": "Female", "M": "Male"})

    if "genotype" in df.columns and "gene" not in df.columns:
        df["gene"] = df["genotype"].str[2:]

    if perform_zt_shift and "total_minutes" in df.columns:
        df["total_minutes"] = (df["total_minutes"] - 6 * 60) % 1440

    if shift_for_48h and "total_minutes" in df.columns:
        df2 = df.copy()
        df2["total_minutes"] = df2["total_minutes"] + 1440
        df = pd.concat([df, df2], ignore_index=True)

    sort_cols = []
    if "gene" in df.columns:
        genotype_order = {"WT": 0, "Het": 1, "Mut": 2}
        df["genotype_order"] = df["gene"].map(genotype_order).fillna(99)
        sort_cols.append("genotype_order")

    if "sex" in df.columns:
        sex_order = {"Male": 0, "Female": 1}
        df["sex_order"] = df["sex"].map(sex_order).fillna(99)
        # Sort by sex first
        sort_cols.insert(0, "sex_order")

    if sort_cols:
        df = df.sort_values(sort_cols).drop(columns=sort_cols)

    return df


def enrich_genotype_metadata(df, genotype_pattern=None, sex_mapper=None):
    """
    Extract 'sex' and 'gene' from 'genotype' column if present.

    Supports custom regex pattern extraction. Defaults to standard 'M_WT' format if no pattern provided.

    Args:
        df (pd.DataFrame): DataFrame potentially containing 'genotype'.
        genotype_pattern (str, optional): Regex pattern with named groups to extract metadata.
            Example: r"(?P<sex>[MF])_(?P<gene>.+)"
            If None, uses default logic: index 0 is Sex, index 2+ is Gene.
        sex_mapper (dict, optional): Dictionary to map extracted sex abbreviations to full names.
            Defaults to {"M": "Male", "F": "Female", "m": "Male", "f": "Female"}.
            Pass an empty dict to disable mapping.

    Returns:
        pd.DataFrame: DataFrame with added 'sex' and 'gene' columns (if extraction succeeds).
    """
    if sex_mapper is None:
        sex_mapper = {"M": "Male", "F": "Female", "m": "Male", "f": "Female"}

    if "genotype" in df.columns:
        if df.empty:
            # If empty, just return (or ensure columns if needed contextually,
            # but usually empty in = empty out without side effects is fine,
            # though tests might expect columns. Let's create columns to be safe).
            for col in ["sex", "gene"]:
                if col not in df.columns:
                    df[col] = pd.Series([], dtype=object)
            return df

        # Ensure genotype is string for .str accessor
        if not pd.api.types.is_string_dtype(df["genotype"]):
            df["genotype"] = df["genotype"].astype(str)

        if genotype_pattern:
            logger.info(f"Extracting metadata with pattern: {genotype_pattern}")
            extracted = df["genotype"].str.extract(genotype_pattern)
            for col in extracted.columns:
                if col not in df.columns:
                    df[col] = extracted[col]

            if "sex" in df.columns and sex_mapper:
                mask = df["sex"].isin(sex_mapper.keys())
                if mask.any():
                    df.loc[mask, "sex"] = df.loc[mask, "sex"].map(sex_mapper)

        else:
            # Default Strategy (M_WT)
            if "sex" not in df.columns:
                df["sex"] = df["genotype"].str[0].map({"F": "Female", "M": "Male"})
            if "gene" not in df.columns:
                df["gene"] = df["genotype"].str[2:]
    return df


def shift_to_zeitgeber_reference(df, shift_hours=6):
    """
    Shift 'total_minutes' to start at ZT0.

    Typically, the experimental clock starts at a certain time (e.g. 6:00 AM).
    ZT0 corresponds to "Lights On" which is often 6:00 AM.
    So Clock 6:00 -> ZT 0.

    Formula: (total_minutes - shift_hours * 60) % 1440

    Args:
        df (pd.DataFrame): DataFrame with 'total_minutes'.
        shift_hours (int, float): Diff between Clock 0:00 and ZT0 in hours.
                                  Defaults to 6 (so 06:00 -> Z0).
    Returns:
        pd.DataFrame: DataFrame with shifted 'total_minutes'.
    """
    if "total_minutes" in df.columns:
        df["total_minutes"] = (df["total_minutes"] - int(shift_hours * 60)) % 1440
    return df


def run_zeitgeber_pipeline(
    df,
    baseline_hours=12,
    baseline_window=None,
    exclude_from_baseline=None,
    interval_minutes=60,
    zeitgeber_shift_hours=6,
    genotype_pattern=None,
    sex_mapper=None,
):
    """
    Main orchestration function for processing zeitgeber data.

    The pipeline performs the following steps:
    1. Enrich metadata (sex, gene).
    2. Shift to Zeitgeber Time (ZT) reference.
    3. Subtract baseline.
    4. Prepare for plotting (48h expansion).

    Args:
        df (pd.DataFrame): Input dataframe with 'total_minutes', 'genotype'.
        baseline_hours (int): Baseline duration from ZT0. Default 12.
        baseline_window (tuple | str): Explicit baseline window. Override.
        exclude_from_baseline (list): Columns to skip.
        interval_minutes (int): Binning interval. Ensure data is correctly binned if this is passed.
            Note: This function doesn't re-bin, just passes it if needed
            (though here it's mostly for signature compatibility).
        zeitgeber_shift_hours (int): Shift applied to align Clock Time to ZT. Default 6.
        genotype_pattern (str, optional): Regex pattern for metadata extraction. See `enrich_genotype_metadata`.
        sex_mapper (dict, optional): Mapper for sex abbreviations. See `enrich_genotype_metadata`.

    Returns:
        pd.DataFrame: Fully processed dataframe.
    """
    logger.info("Running zeitgeber analysis pipeline...")

    df_processed = df.copy()

    # 1. Enrich Metadata
    df_processed = enrich_genotype_metadata(
        df_processed, genotype_pattern=genotype_pattern, sex_mapper=sex_mapper
    )

    # 2. Shift to ZT
    df_processed = shift_to_zeitgeber_reference(
        df_processed, shift_hours=zeitgeber_shift_hours
    )

    # 3. Baseline Subtraction
    df_processed = subtract_zeitgeber_baseline(
        df_processed,
        baseline_hours=baseline_hours,
        baseline_window=baseline_window,
        exclude_from_baseline=exclude_from_baseline,
    )

    # 4. Prepare for Plotting (48h expansion + sorting)
    # Note: we already shifted to ZT, so perform_zt_shift=False
    df_final = prepare_plot_data(
        df_processed, shift_for_48h=True, perform_zt_shift=False
    )

    if "animal" in df_final.columns:
        logger.info(f"Processed data for {df_final['animal'].nunique()} unique animals")

    return df_final


class ZeitgeberAnalysisResult:
    """
    Proxy wrapper for WindowAnalysisResult that applies Zeitgeber processing on the fly.

    Duck-types WindowAnalysisResult to be compatible with ExperimentPlotter and AnimalPlotter,
    intercepting data retrieval methods to inject the ZT pipeline.

    Args:
        war (WindowAnalysisResult): The implementation-specific analysis result object.
        **pipeline_config: Keywords arguments passed to `run_zeitgeber_pipeline`.
    """

    def __init__(self, war, **pipeline_config):
        self.war = war
        self.config = pipeline_config

    def __getattr__(self, name):
        """Delegate attribute access to the underlying WAR object."""
        return getattr(self.war, name)

    def _apply_pipeline(self, df):
        """Helper to apply ZT pipeline to a dataframe."""
        if df.empty:
            return df

        # Ensure total_minutes exists (requires 'timestamp')
        if "total_minutes" not in df.columns:
            # Try to add it if timestamp exists
            if "timestamp" in df.columns:
                # Use interval from config if present, else default
                interval = self.config.get("interval_minutes", 60)
                df = add_zeitgeber_time_columns(df, interval_minutes=interval)

        return run_zeitgeber_pipeline(df, **self.config)

    def get_result(self, *args, **kwargs):
        """Intercepts get_result and applies ZT pipeline."""
        df = self.war.get_result(*args, **kwargs)
        return self._apply_pipeline(df)

    def get_grouprows_result(self, *args, **kwargs):
        """Intercepts get_grouprows_result and applies ZT pipeline."""
        df = self.war.get_grouprows_result(*args, **kwargs)
        return self._apply_pipeline(df)

    def get_groupavg_result(self, *args, **kwargs):
        """Intercepts get_groupavg_result and applies ZT pipeline."""
        # NOTE: get_groupavg_result returns a dataframe with MultiIndex or index that might
        # complicate things if metadata columns needed for pipeline (like genotype)
        # are now part of the index.
        # However, run_zeitgeber_pipeline expects columns.
        # Ideally, we process on raw data (get_result) or row-grouped (get_grouprows).
        # Averaged results might LOSE time info needed for ZT shifting if averaged over time?
        # If averaging over time, ZT pipeline doesn't make sense post-hoc.
        # But AnimalPlotter calls this. Let's see if it works.
        # If it's just animal level feature averages, we might not be able to timeline shift.
        # Usually ZT analysis is for time-series.

        # If this is being called, we assume the user knows what they are doing OR
        # the result still contains time info (e.g. grouped by animalday + hour).
        df = self.war.get_groupavg_result(*args, **kwargs)

        # If it's a completely aggregated result (no time), the pipeline might skip ZT shifting
        # but still run baseline subtraction if 'total_minutes' is preserved?
        # Unlikely 'total_minutes' is preserved in a generic group-avg unless grouped by it.

        # For now, we attempt to run it. If it fails due to missing cols,
        # logic in pipeline should be robust (checks `if "total_minutes" in df.columns`).
        return self._apply_pipeline(df)

    def get_channel_averaged_result(self, *args, **kwargs):
        """Intercepts get_channel_averaged_result and applies ZT pipeline."""
        # This was part of the original plan, though maybe not strict WAR protocol.
        # Good to have if WAR supports it.
        if hasattr(self.war, "get_channel_averaged_result"):
            df = self.war.get_channel_averaged_result(*args, **kwargs)
            return self._apply_pipeline(df)
        else:
            raise NotImplementedError(
                "Underlying WAR does not support get_channel_averaged_result"
            )
