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
from .. import constants
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
        ftype = constants.FEATURE_TYPES.get(feature)
        if ftype is not None and ftype.is_dict_stored:
            # Expand into per-band features (e.g. zcohere -> zcohere_delta, zcohere_theta, ...)
            for band in constants.BAND_NAMES:
                expanded_features.append(f"{feature}_{band}")
        else:
            # LINEAR, SIMPLE_MATRIX, HIST, and unknown features remain as single columns
            expanded_features.append(feature)
            
    return expanded_features


def _load_war_for_zeitgeber(war_path_info):
    """
    Load a fragment-filtered WAR and extract channel-averaged features for zeitgeber analysis.

    Args:
        war_path_info (tuple): Tuple containing:
            - war_parquet_path (Path): Path to the WAR parquet file.
            - war_json_path (Path): Path to the WAR JSON metadata file.
            - features_to_extract (list[str]): List of features to extract.
            - animal_name (str): Identifier for the animal.
            - pipeline_config (dict): Configuration for the zeitgeber pipeline.

    Returns:
        pd.DataFrame: DataFrame containing channel-averaged features and animal identifier.
                      The DataFrame will have columns for each extracted feature and an 'animal' column.
    """
    war_parquet_path, war_json_path, features_to_extract, animal_name, pipeline_config = war_path_info

    try:
        logger.info(f"Loading {animal_name}")

        # Local import breaks the core->visualization layer edge (a result container
        # loaded lazily only when this pipeline helper actually runs).
        from ..visualization import WindowAnalysisResult

        war = WindowAnalysisResult.load_parquet_and_json(
            folder_path=war_parquet_path.parent,
            parquet_name=war_parquet_path.name,
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


def _compute_daynight(zt_minutes):
    """Return ``"Day"``/``"Night"`` labels for each ZT minute.

    Single source of truth for the day/night phase boundary, used by
    :func:`shift_to_zeitgeber_reference`, :func:`transform_time_axis`, and
    :func:`expand_zt_axis` so the rule never drifts across call sites.

    ``"Day"`` corresponds to ZT 0:00–11:59 (light phase, ``zt_minutes %
    1440 < 720``); ``"Night"`` to ZT 12:00–23:59 (dark phase).  The
    ``% 1440`` lets this work over multi-day expansions (e.g. ZT 1500 ->
    "Day" because 1500 % 1440 = 60).
    """
    arr = np.asarray(zt_minutes)
    return np.where(arr % 1440 < 720, "Day", "Night")


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
            * 'zt_minutes' (float): Minutes-of-day, binned to the nearest
              ``interval_minutes``.  **Initially populated from raw clock
              time** (= hour * 60 + minute); call
              :func:`shift_to_zeitgeber_reference` afterwards to convert
              this column to true zeitgeber-time minutes from lights-on.
              The ``daynight`` companion column is added at that point.

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
    df["zt_minutes"] = binned_minutes % 1440

    return df


def subtract_zeitgeber_baseline(
    df, baseline_hours=12, baseline_window=None, exclude_from_baseline=None
):
    """
    Subtract baseline from numeric features in the dataframe.

    Baseline is calculated as the mean value within a specified window of time (ZT).
    Requires 'zt_minutes' column to be present (usually added by add_zeitgeber_time_columns).

    Args:
        df (pd.DataFrame): DataFrame containing features and 'zt_minutes'.
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

    if "zt_minutes" not in df.columns:
        logger.warning("Skipping baseline subtraction: 'zt_minutes' not found.")
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip_cols = [
        "hour",
        "minute",
        "zt_minutes",
        "animal_idx",
        "day_idx",
        "epoch_idx",
    ] + exclude_from_baseline

    features_to_correct = [
        c for c in numeric_cols if c not in skip_cols and not c.endswith("_nobase")
    ]

    group_cols = [c for c in ["animal", "sex", "genotype"] if c in df.columns]

    result_df = df.copy()

    for feature in features_to_correct:
        if group_cols:

            def get_group_mean(g):
                vals = g.loc[
                    g["zt_minutes"].between(start_min, end_min, inclusive="left"),
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
                df["zt_minutes"].between(start_min, end_min, inclusive="left"),
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


def transform_time_axis(df, time_range=(0, 24), shift=0):
    """
    Transform time axis for plotting: enrichment, optional time shift, and sorting.

    Args:
        df (pd.DataFrame): Input dataframe.
            Expected format:
            * 'genotype' (str, optional): Genotype label (e.g., 'WT', 'KO').
            * 'sex' (str, optional): 'Male'/'Female'. Both 'genotype' and 'sex'
              are expected to already be present (added by enrich_metadata).
            * 'zt_minutes' (numeric): Zeitgeber-time minutes (0-1440).
        time_range (tuple, optional): No-op since the data layer no longer
            duplicates rows for multi-day plotting.  Kept for backward
            compatibility with callers that still pass it.  Multi-day
            expansion is now the plotter's job — use
            :func:`expand_zt_axis` to get a 48h/72h/... view at render
            time.  Defaults to (0, 24).
        shift (float, optional): Hours to shift the time axis. Defaults to 0.
            - **Negative shift** moves times earlier: a data point at 6:00 with shift=-6
              becomes 0:00. Use this when your data starts at clock time 6:00 but you
              want it to appear at hour 0 on the plot.
            - **Positive shift** moves times later: a data point at 0:00 with shift=6
              becomes 6:00.
            - Common use: shift=-6 aligns "lights on at 6am" to hour 0 (Zeitgeber Time).
            When non-zero, also recomputes ``daynight`` so the labels stay
            consistent with the shifted ``zt_minutes``.

    Returns:
        pd.DataFrame: Processed dataframe ready for plotting.
            Adds temporary sorting columns (genotype_order/sex_order) if applicable.
            Row count is unchanged — call :func:`expand_zt_axis` to
            duplicate rows across multiple ZT cycles.

    Raises:
        ValueError: If time_range[0] >= time_range[1].
    """
    if time_range[0] >= time_range[1]:
        raise ValueError(f"time_range start ({time_range[0]}) must be less than end ({time_range[1]})")
    df = df.copy()

    # Apply time shift.  When non-zero, recompute daynight so labels stay
    # consistent with the new zt_minutes.
    if shift != 0 and "zt_minutes" in df.columns:
        df["zt_minutes"] = (df["zt_minutes"] + shift * 60) % 1440
        if "daynight" in df.columns:
            df["daynight"] = _compute_daynight(df["zt_minutes"])

    sort_cols = []
    if "genotype" in df.columns:
        genotype_order = {"WT": 0, "Het": 1, "KO": 2, "Mut": 2}
        df["genotype_order"] = df["genotype"].map(genotype_order).fillna(99)
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
            # Parse genotype key to extract sex and genotype
            if "_" in genotype_key:
                parts = genotype_key.split("_", 1)
                sex_char, genotype = parts[0], parts[1]
            elif len(genotype_key) >= 2 and genotype_key[0].upper() in ("M", "F"):
                sex_char, genotype = genotype_key[0], genotype_key[1:]
            else:
                sex_char, genotype = None, genotype_key

            sex = normalize_value_from_aliases(sex_char, constants.SEX_MAP) if sex_char else None
            animal_metadata_converted[animal_id] = {"sex": sex, "genotype": genotype}
        
        return metadata_module.enrich_metadata(df, animal_metadata_converted)
    
    # No metadata provided, return as-is
    return df


def shift_to_zeitgeber_reference(df, shift_hours=6):
    """
    Shift ``zt_minutes`` to start at ZT0 and add the ``daynight`` label.

    Typically, the experimental clock starts at a certain time (e.g. 6:00 AM).
    ZT0 corresponds to "Lights On" which is often 6:00 AM.
    So Clock 6:00 -> ZT 0.

    Formula: ``(zt_minutes - shift_hours * 60) % 1440``

    After the shift the values represent true zeitgeber-time minutes from
    lights-on, so this is also where the ``daynight`` companion column is
    populated (``"Day"`` for ZT 0:00–11:59, ``"Night"`` for ZT 12:00–23:59).

    Args:
        df (pd.DataFrame): DataFrame with ``zt_minutes``.
        shift_hours (int, float): Diff between Clock 0:00 and ZT0 in hours.
                                  Defaults to 6 (so 06:00 -> ZT0).
    Returns:
        pd.DataFrame: DataFrame with shifted ``zt_minutes`` and a fresh
        ``daynight`` column.
    """
    if "zt_minutes" in df.columns:
        df["zt_minutes"] = (df["zt_minutes"] - int(shift_hours * 60)) % 1440
        df["daynight"] = _compute_daynight(df["zt_minutes"])
    return df


def expand_zt_axis(df, n_days=2):
    """Duplicate *df* across *n_days* ZT cycles for multi-day plotting.

    Each copy's ``zt_minutes`` is offset by ``1440 * i``; the ``daynight``
    column is recomputed so day/night labels stay correct in the expanded
    range.  Used by ``ZeitgeberPlotter`` (and any other plotter that
    wants a multi-day view) — the persisted zeitgeber CSV stays a clean
    24 h on disk; this helper materialises the wider view only at render
    time.

    Args:
        df (pd.DataFrame): DataFrame with ``zt_minutes``.  Should already
            be ZT-aligned (i.e. produced after
            :func:`shift_to_zeitgeber_reference`).
        n_days (int): Number of ZT cycles to span.  Defaults to ``2``
            (today's 48h plot behaviour).  Must be ``>= 1``; ``n_days=1``
            is a no-op pass-through.

    Returns:
        pd.DataFrame: New dataframe with ``len(df) * n_days`` rows.

    Raises:
        ValueError: If ``n_days < 1``.
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    if "zt_minutes" not in df.columns:
        return df.copy()
    if n_days == 1:
        return df.copy()
    copies = []
    for i in range(n_days):
        c = df.copy()
        c["zt_minutes"] = c["zt_minutes"] + 1440 * i
        copies.append(c)
    out = pd.concat(copies, ignore_index=True)
    out["daynight"] = _compute_daynight(out["zt_minutes"])
    return out


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
    1. Enrich metadata (sex, genotype) from ANIMAL_METADATA.
    2. Shift to Zeitgeber Time (ZT) reference (also adds ``daynight``).
    3. Subtract baseline.
    4. Sort + sex/genotype enrichment for plot readiness.

    Args:
        df (pd.DataFrame): Input dataframe with 'zt_minutes', 'animal'.
        baseline_hours (int): Baseline duration from ZT0. Default 12.
        baseline_window (tuple | str): Explicit baseline window. Override.
        exclude_from_baseline (list): Columns to skip.
        interval_minutes (int): Binning interval.
        zeitgeber_shift_hours (int): Shift applied to align Clock Time to ZT. Default 6.
        shift_for_48h (bool, optional): **DEPRECATED no-op.**  Multi-day
            expansion has moved out of the data layer to
            :func:`expand_zt_axis`, called by plotters at render time.
            Kept in the signature for backward compatibility; ignored.
        animal_metadata (dict, optional): Dict of animal_id -> {sex, genotype} from load_animal_metadata().
        genotype_pattern (str, optional): DEPRECATED.
        sex_mapper (dict, optional): DEPRECATED.
        genotype_aliases (dict, optional): DEPRECATED. Use animal_metadata instead.

    Returns:
        pd.DataFrame: Fully processed 24h dataframe.  Row count equals
            input row count — plotters that want a multi-day view call
            :func:`expand_zt_axis` on the result.
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

    # 2. Shift to ZT (also adds the daynight column).
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

    # 4. Sort + sex/genotype enrichment.  No multi-day expansion here — the
    # data layer stays 24h; plotters use expand_zt_axis() at render time.
    df_final = transform_time_axis(df_processed, shift=0)

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

        # Ensure zt_minutes exists (requires 'timestamp')
        if "zt_minutes" not in df.columns:
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
        # Note: If aggregation removes time information (timestamp/zt_minutes),
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

