"""
Zeitgeber Time (ZT) Analysis Module
===================================
This module provides functionality for processing and analyzing data in Zeitgeber Time (ZT).
"""

import inspect
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from .. import constants, visualization
from . import metadata as metadata_module
from .utils import normalize_value_from_aliases

logger = logging.getLogger(__name__)


def get_expanded_feature_names(base_features):
    """
    Expand base feature names into their full column names (e.g. bands, matrix features).
    
    Uses neurodent.constants to determine expansion rules.
    
    - BAND_FEATURES and BANDED_MATRIX_FEATURES get expanded to per-band columns (e.g. zcohere_delta)
    - SIMPLE_MATRIX_FEATURES, LINEAR_FEATURES, and HIST_FEATURES remain as single columns (e.g. zpcorr)

    Args:
        base_features (list[str]): List of base feature names (e.g. ['logpsdband', 'rms', 'zpcorr']).

    Returns:
        list[str]: List of expanded column names.
    """
    from .. import constants
    
    expanded_features = []
    
    for feature in base_features:
        if feature in constants.BAND_FEATURES or feature in constants.BANDED_MATRIX_FEATURES:
            # Expand into per-band features (e.g. zcohere -> zcohere_delta, zcohere_theta, ...)
            for band in constants.BAND_NAMES:
                expanded_features.append(f"{feature}_{band}")
        elif feature in constants.SIMPLE_MATRIX_FEATURES:
            # Simple matrix features are NOT banded - keep as single column (e.g. zpcorr, pcorr)
            expanded_features.append(feature)
        elif feature in constants.LINEAR_FEATURES or feature in constants.HIST_FEATURES:
            # Keep as is
            expanded_features.append(feature)
        else:
            # Unknown feature type, assume it's a single column
            expanded_features.append(feature)
            
    return expanded_features


def _load_war_for_zeitgeber(war_path_info):
    """
    Load a fragment-filtered WAR and extract channel-averaged features for zeitgeber analysis.

    Args:
        war_path_info (tuple): Tuple containing:
            - war_pkl_path (Path): Path to the WAR pickle file.
            - war_json_path (Path): Path to the WAR JSON metadata file.
            - features_to_extract (list[str]): List of features to extract.
            - animal_name (str): Identifier for the animal.
            - pipeline_config (dict): Configuration for the zeitgeber pipeline.

    Returns:
        pd.DataFrame: DataFrame containing channel-averaged features and animal identifier.
                      The DataFrame will have columns for each extracted feature and an 'animal' column.
    """
    war_pkl_path, war_json_path, features_to_extract, animal_name, pipeline_config = war_path_info

    try:
        logger.info(f"Loading {animal_name}")

        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=war_pkl_path.parent,
            pickle_name=war_pkl_path.name,
            json_name=war_json_path.name,
        )
        
        # Wrap with ZeitgeberAnalysisResult to apply pipeline on the fly
        zar = ZeitgeberAnalysisResult(war, **pipeline_config)

        df = zar.get_channel_averaged_result(features=features_to_extract)
        df["animal"] = animal_name
        del war
        del zar
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

            group_means = result_df.groupby(group_cols).apply(
                get_group_mean, include_groups=False
            )

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


def transform_time_axis(df, time_range=(0, 48), shift=0):
    """
    Transform time axis for plotting: enrichment, sorting, and time range expansion.

    Args:
        df (pd.DataFrame): Input dataframe.
            Expected format:
            * 'genotype' (str, optional): Strain info (e.g., 'M_WT', 'F_Mut').
              Used to derive 'sex' and 'gene' if not present.
            * 'total_minutes' (numeric): Time of day in minutes (0-1440).
        time_range (tuple, optional): Hours to display as (start_hr, end_hr). Defaults to (0, 48).
            Any range is valid as long as start < end. If end > 24, the function
            duplicates data to fill the extended range (e.g., (0, 48) repeats 0-24h
            as 24-48h, (0, 72) would repeat twice, etc.).
        shift (float, optional): Hours to shift the time axis. Defaults to 0.
            - **Negative shift** moves times earlier: a data point at 6:00 with shift=-6
              becomes 0:00. Use this when your data starts at clock time 6:00 but you
              want it to appear at hour 0 on the plot.
            - **Positive shift** moves times later: a data point at 0:00 with shift=6
              becomes 6:00.
            - Common use: shift=-6 aligns "lights on at 6am" to hour 0 (Zeitgeber Time).

    Returns:
        pd.DataFrame: Processed dataframe ready for plotting.
            Adds 'sex', 'gene', and temporary sorting columns if applicable.

    Raises:
        ValueError: If time_range[0] >= time_range[1].
    """
    if time_range[0] >= time_range[1]:
        raise ValueError(f"time_range start ({time_range[0]}) must be less than end ({time_range[1]})")
    df = df.copy()

    if "genotype" in df.columns and "sex" not in df.columns:
        df["sex"] = df["genotype"].str[0].apply(
            lambda x: normalize_value_from_aliases(x, constants.SEX_ALIASES)
        )

    if "genotype" in df.columns and "gene" not in df.columns:
        df["gene"] = df["genotype"].str[2:]

    # Apply time shift
    if shift != 0 and "total_minutes" in df.columns:
        df["total_minutes"] = (df["total_minutes"] + shift * 60) % 1440

    # Duplicate for 48h view if time_range extends beyond 24h
    if time_range[1] > 24 and "total_minutes" in df.columns:
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


def enrich_genotype_metadata(df, genotype_pattern=None, sex_mapper=None, genotype_aliases=None, animal_metadata=None):
    """
    DEPRECATED: Use neurodent.core.metadata.enrich_metadata instead.
    
    This function is kept for backward compatibility but will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "enrich_genotype_metadata is deprecated. Use neurodent.core.metadata.enrich_metadata instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # If new-style animal_metadata is provided, use the new module
    if animal_metadata is not None:
        return metadata_module.enrich_metadata(df, animal_metadata)
    
    # Legacy fallback: use old logic with genotype_aliases
    # This is a simplified pass-through for backward compat
    if genotype_aliases and "animal" in df.columns:
        # Build reverse map
        animal_to_genotype = {}
        for genotype_group, animals in genotype_aliases.items():
            for animal in animals:
                animal_to_genotype[animal] = genotype_group
        
        # Convert to new format
        animal_metadata_converted = {}
        for animal_id, genotype_key in animal_to_genotype.items():
            # Parse genotype key to extract sex and gene
            if "_" in genotype_key:
                parts = genotype_key.split("_", 1)
                sex_char, gene = parts[0], parts[1]
            elif len(genotype_key) >= 2 and genotype_key[0].upper() in ("M", "F"):
                sex_char, gene = genotype_key[0], genotype_key[1:]
            else:
                sex_char, gene = None, genotype_key
            
            sex = normalize_value_from_aliases(sex_char, constants.SEX_ALIASES) if sex_char else None
            animal_metadata_converted[animal_id] = {"sex": sex, "gene": gene}
        
        return metadata_module.enrich_metadata(df, animal_metadata_converted)
    
    # No metadata provided, return as-is
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
    shift_for_48h=True,
    animal_metadata=None,
    # Deprecated params (kept for backward compat)
    genotype_pattern=None,
    sex_mapper=None,
    genotype_aliases=None,
):
    """
    Main orchestration function for processing zeitgeber data.

    The pipeline performs the following steps:
    1. Enrich metadata (sex, gene) from ANIMAL_METADATA.
    2. Shift to Zeitgeber Time (ZT) reference.
    3. Subtract baseline.
    4. Prepare for plotting (48h expansion).

    Args:
        df (pd.DataFrame): Input dataframe with 'total_minutes', 'animal'.
        baseline_hours (int): Baseline duration from ZT0. Default 12.
        baseline_window (tuple | str): Explicit baseline window. Override.
        exclude_from_baseline (list): Columns to skip.
        interval_minutes (int): Binning interval.
        zeitgeber_shift_hours (int): Shift applied to align Clock Time to ZT. Default 6.
        shift_for_48h (bool, optional): Whether to duplicate data for 48h plotting. Defaults to True.
        animal_metadata (dict, optional): Dict of animal_id -> {sex, gene} from load_animal_metadata().
        genotype_pattern (str, optional): DEPRECATED.
        sex_mapper (dict, optional): DEPRECATED.
        genotype_aliases (dict, optional): DEPRECATED. Use animal_metadata instead.

    Returns:
        pd.DataFrame: Fully processed dataframe.
    """
    logger.info("Running zeitgeber analysis pipeline...")

    df_processed = df.copy()

    # 1. Enrich Metadata
    if animal_metadata is not None:
        df_processed = metadata_module.enrich_metadata(df_processed, animal_metadata)
    elif genotype_aliases is not None:
        # Legacy path
        df_processed = enrich_genotype_metadata(
            df_processed, 
            genotype_pattern=genotype_pattern, 
            sex_mapper=sex_mapper,
            genotype_aliases=genotype_aliases
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

    # 4. Transform time axis for plotting (48h expansion + sorting)
    # Note: we already shifted to ZT, so shift=0
    time_range = (0, 48) if shift_for_48h else (0, 24)
    df_final = transform_time_axis(
        df_processed, time_range=time_range, shift=0
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

        # Filter config to only include valid kwargs for run_zeitgeber_pipeline
        sig = inspect.signature(run_zeitgeber_pipeline)
        valid_kwargs = {p for p in sig.parameters if p != 'df'}
        filtered_config = {k: v for k, v in self.config.items() if k in valid_kwargs}

        return run_zeitgeber_pipeline(df, **filtered_config)

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
        # Note: If aggregation removes time information (timestamp/total_minutes),
        # the ZT pipeline will gracefully skip time-dependent steps like shifting
        # and baseline subtraction.
        df = self.war.get_groupavg_result(*args, **kwargs)
        return self._apply_pipeline(df)

    def get_channel_averaged_result(self, *args, **kwargs):
        """Intercepts get_channel_averaged_result and applies ZT pipeline."""
        if hasattr(self.war, "get_channel_averaged_result"):
            df = self.war.get_channel_averaged_result(*args, **kwargs)
            return self._apply_pipeline(df)
        else:
            raise NotImplementedError(
                "Underlying WAR does not support get_channel_averaged_result"
            )

