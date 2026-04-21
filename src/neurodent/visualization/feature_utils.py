"""Shared utilities for extracting and reshaping feature data.

These functions consolidate common patterns that previously appeared in
``results.py``, ``plotting/experiment.py``, and ``plotting/animal.py``.
Each utility operates on a pandas Series (one column of a WAR DataFrame)
and returns numpy arrays ready for downstream processing.

All functions are stateless and side-effect-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import constants


def _ensure_dense(arr: np.ndarray, error_msg: str) -> np.ndarray:
    """Validate that *arr* is a dense (non-object) numpy array.

    Raises ``ValueError`` with *error_msg* when the array has ``dtype=object``,
    which indicates ragged (non-uniform) nested structure.
    """
    if arr.dtype == object:
        raise ValueError(error_msg)
    return arr


# ---------------------------------------------------------------------------
# Linear / array-valued features
# ---------------------------------------------------------------------------

def extract_linear_array(
    series: pd.Series,
    ftype: constants.FeatureType | None = None,
) -> np.ndarray:
    """Convert a Series of per-channel arrays into a dense numpy array.

    Works for LINEAR features (scalar per channel), LINEAR_2D features
    (multi-component per channel, e.g. psdslope), and matrix features
    stored as raw arrays.

    Parameters
    ----------
    series : pd.Series
        Each element is a list/array of consistent shape.
    ftype : constants.FeatureType, optional
        When provided, the expected ndim is validated against the feature type.

    Returns
    -------
    np.ndarray
        For LINEAR features: shape ``(n_windows, n_channels)``.
        For LINEAR_2D features: shape ``(n_windows, n_channels, n_components)``.
        Higher-dimensional inputs are supported as long as all rows share
        the same shape.

    Raises
    ------
    ValueError
        If the per-row arrays have inconsistent shapes (ragged data),
        or if *ftype* is provided and the extracted ndim doesn't match.
    """
    try:
        result = np.asarray(series.tolist())
    except ValueError:
        shapes = {np.asarray(row).shape for row in series}
        raise ValueError(
            f"Ragged input: per-row shapes are not uniform ({shapes}). "
            f"All rows must have the same shape."
        )
    result = _ensure_dense(
        result,
        "Ragged input: per-row shapes are not uniform. "
        "All rows must have the same shape.",
    )
    if ftype is not None:
        expected_ndim = len(ftype.extracted_shape.split(","))
        if result.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim}-D array for {ftype.name} "
                f"but got {result.ndim}-D (shape {result.shape})."
            )
    return result


# ---------------------------------------------------------------------------
# Dict-stored (banded) features
# ---------------------------------------------------------------------------

def extract_band_dict(
    series: pd.Series,
    ftype: constants.FeatureType | None = None,
) -> tuple[np.ndarray, list]:
    """Unpack a Series of band dicts into an array plus ordered key list.

    Parameters
    ----------
    series : pd.Series
        Each element is a dict ``{band_name: array}``.
    ftype : constants.FeatureType, optional
        When provided, the expected ndim is validated against the feature type
        instead of being inferred from the array shape alone.

    Returns
    -------
    vals : np.ndarray
        Array of shape ``(n_windows, n_channels, n_bands)`` for BAND features,
        or ``(n_windows, n_channels, n_channels, n_bands)`` for BANDED_MATRIX.
        Canonical shape: channels before semantic (band) dimensions.
    keys : list
        Ordered band names extracted from the first row.

    Raises
    ------
    ValueError
        If band values have inconsistent shapes across windows (ragged data),
        or if *ftype* is provided and the extracted ndim doesn't match.
    """
    _RAGGED_BAND_MSG = (
        "Ragged input: band values have inconsistent shapes across windows. "
        "All windows must have the same shape per band."
    )
    df_bands = pd.DataFrame(series.tolist())
    keys = list(df_bands.columns)
    try:
        vals = np.asarray(df_bands.values.tolist())
    except ValueError:
        raise ValueError(_RAGGED_BAND_MSG)
    vals = _ensure_dense(vals, _RAGGED_BAND_MSG)

    # Determine expected ndim from ftype or infer from array shape.
    if ftype is not None:
        expected_ndim = len(ftype.extracted_shape.split(","))
        if vals.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim}-D array for {ftype.name} "
                f"but got {vals.ndim}-D (shape {vals.shape})."
            )
    # Transpose to canonical (W, C, ..., B) shape: move band axis to last position.
    # Raw stacking yields (W, B, C) for BAND or (W, B, C, C) for BANDED_MATRIX.
    if vals.ndim == 3:
        # BAND: (W, B, C) → (W, C, B)
        vals = vals.transpose((0, 2, 1))
    elif vals.ndim == 4:
        # BANDED_MATRIX: (W, B, C, C) → (W, C, C, B)
        vals = vals.transpose((0, 2, 3, 1))
    return vals, keys


def repack_band_dict(vals: np.ndarray, keys: list) -> list[dict]:
    """Repack a numpy array back into a list of band dicts.

    This is the inverse of :func:`extract_band_dict`.

    Parameters
    ----------
    vals : np.ndarray
        Canonical array of shape ``(n_windows, n_channels, n_bands)`` for BAND,
        or ``(n_windows, n_channels, n_channels, n_bands)`` for BANDED_MATRIX.
    keys : list
        Ordered band names (same length as the last axis of ``vals``).

    Returns
    -------
    list[dict]
        One dict per window, each mapping band names to their arrays.
    """
    # Transpose canonical (W, C, B) → (W, B, C) or (W, C, C, B) → (W, B, C, C)
    # so that iterating over axis 1 yields per-band slices.
    if vals.ndim == 3:
        vals_banded = vals.transpose((0, 2, 1))   # (W, C, B) → (W, B, C)
    elif vals.ndim == 4:
        vals_banded = vals.transpose((0, 3, 1, 2))  # (W, C, C, B) → (W, B, C, C)
    else:
        vals_banded = vals
    return [dict(zip(keys, row)) for row in vals_banded]


# ---------------------------------------------------------------------------
# Histogram / spectral (PSD) features
# ---------------------------------------------------------------------------

def extract_hist_data(
    series: pd.Series,
    ftype: constants.FeatureType | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a Series of ``(coords, values)`` histogram tuples.

    Handles both pickle-origin tuples and parquet-origin lists.

    Parameters
    ----------
    series : pd.Series
        Each element is a ``(coords, values)`` tuple or list pair.
    ftype : constants.FeatureType, optional
        When provided, validates that *ftype* is ``FeatureType.HIST``.

    Returns
    -------
    coords : np.ndarray
        Frequency coordinates, shape ``(n_windows, n_freq_bins)``.
    values : np.ndarray
        Spectral values, shape ``(n_windows, n_channels, n_freq_bins)``.
        Canonical shape: channels axis 1, freq_bins axis 2.

    Raises
    ------
    ValueError
        If histogram entries have inconsistent shapes across windows
        (ragged data), or if *ftype* is provided and is not ``HIST``.
    """
    if ftype is not None and ftype is not constants.FeatureType.HIST:
        raise ValueError(
            f"extract_hist_data expects FeatureType.HIST but got {ftype.name}."
        )
    _RAGGED_HIST_MSG = (
        "Ragged input: histogram entries have inconsistent shapes across windows. "
        "All windows must have the same shape."
    )
    data = series.tolist()
    try:
        coords = np.asarray([
            np.asarray(item[0]) if isinstance(item, (tuple, list)) and len(item) == 2
            else np.asarray(item)
            for item in data
        ])
        values = np.asarray([
            np.asarray(item[1]) if isinstance(item, (tuple, list)) and len(item) == 2
            else np.asarray(item)
            for item in data
        ])
    except ValueError:
        raise ValueError(_RAGGED_HIST_MSG)
    _ensure_dense(coords, _RAGGED_HIST_MSG)
    _ensure_dense(values, _RAGGED_HIST_MSG)
    # Raw stacking yields (W, F, C) when each cell stores (F, C).
    # Transpose to canonical (W, C, F) — channels axis 1, freq_bins axis 2.
    if values.ndim == 3:
        values = values.transpose((0, 2, 1))
    # If each cell stored a 1-D (F,) vector (single-channel), stacking
    # produces (W, F). Insert a singleton channel axis to match the
    # canonical (W, C, F) shape expected by downstream code.
    elif values.ndim == 2:
        values = values[:, np.newaxis, :]
    return coords, values


# ---------------------------------------------------------------------------
# Channel reshaping utilities
# ---------------------------------------------------------------------------

def extract_feature(
    series: pd.Series,
    ftype: constants.FeatureType,
) -> tuple[np.ndarray, list | None]:
    """Extract feature data from a Series, dispatching on FeatureType.

    Uses :func:`extract_band_dict` for dict-stored features and
    :func:`extract_linear_array` for all others, returning the canonical
    extracted array for the given feature type.

    Parameters
    ----------
    series : pd.Series
        One column of a WAR DataFrame.
    ftype : constants.FeatureType
        The classified feature type.

    Returns
    -------
    vals : np.ndarray
        Canonical extracted array (shape depends on *ftype*).
    keys : list or None
        Band keys for dict-stored features, ``None`` otherwise.
    """
    if ftype.is_dict_stored:
        vals, keys = extract_band_dict(series, ftype=ftype)
    else:
        vals = extract_linear_array(series, ftype=ftype)
        keys = None
    return vals, keys


def format_channel_data(
    vals: np.ndarray,
    ftype: constants.FeatureType,
    collapse_channels: bool,
    ch_to_idx: dict[str, int] | None = None,
    channels: list[str] | None = None,
    ch_names: list[str] | None = None,
) -> dict[str, list]:
    """Format extracted feature data as a channel-keyed dict.

    Handles both the *collapse_channels* path (average across channels)
    and the per-channel path (index by channel name).

    Parameters
    ----------
    vals : np.ndarray
        Canonical extracted array.
    ftype : constants.FeatureType
        The classified feature type.
    collapse_channels : bool
        If ``True``, average across channels via
        :func:`collapse_feature_channels` and return ``{"average": ...}``.
        If ``False``, return per-channel data indexed by channel name,
        or ``{"all": ...}`` for matrix features.
    ch_to_idx : dict, optional
        Channel name → column index mapping (required when *collapse_channels*
        is ``False`` and *ftype* is not a matrix type).
    channels : list[str], optional
        Which channels to include (required when *collapse_channels* is
        ``False`` and *ftype* is not a matrix type).
    ch_names : list[str], optional
        All channel names in the WAR (required when *collapse_channels* is
        ``False`` and *ftype* is not a matrix type).

    Returns
    -------
    dict[str, list]
        Channel-keyed dict suitable for DataFrame construction.
    """
    if collapse_channels:
        collapsed = collapse_feature_channels(vals, ftype)
        return {"average": collapsed.tolist()}

    if ftype.is_matrix:
        return {"all": vals.tolist()}

    # Per-channel indexing for non-matrix features.
    # Features with semantic trailing axes (BAND, LINEAR_2D, HIST) need
    # ``[:, idx, :]`` while scalar-per-channel (LINEAR) needs ``[:, idx]``.
    has_semantic = bool(ftype.semantic_axes)
    result: dict[str, list] = {}
    for ch in channels:
        if ch in ch_names:
            idx = ch_to_idx[ch]
            if has_semantic:
                result[ch] = vals[:, idx, :].tolist()
            else:
                result[ch] = vals[:, idx].tolist()
    return result


def flatten_feature_for_plotting(
    vals: np.ndarray,
    ftype: constants.FeatureType,
    triag: bool = True,
) -> np.ndarray:
    """Reshape extracted feature data into a 3-D array for plotting.

    Converts raw extracted feature arrays into a three-axis shape
    ``(n_time, n_features, n_components)`` used by per-channel visualization.
    The middle axis represents channels, channel-pairs, or other spatial
    units depending on the feature type.

    Parameters
    ----------
    vals : np.ndarray
        Feature array from :func:`extract_linear_array` or
        :func:`extract_band_dict`.
    ftype : constants.FeatureType
        The classified feature type.
    triag : bool, optional
        For matrix features, whether to extract only the lower-triangular
        channel pairs (default ``True``).

    Returns
    -------
    np.ndarray
        Shape ``(n_time, n_features, n_components)`` where:

        - LINEAR: ``(n_time, n_channels, 1)``
        - LINEAR_2D: ``(n_time, n_channels, n_components)``
        - BAND: ``(n_time, n_channels, n_bands)``
        - BANDED_MATRIX: ``(n_time, n_pairs, n_bands)``
        - SIMPLE_MATRIX: ``(n_time, n_pairs, 1)``

    Raises
    ------
    ValueError
        If *ftype* is not a supported feature type.
    """
    if ftype is constants.FeatureType.LINEAR:
        return np.expand_dims(vals, axis=-1)

    elif ftype is constants.FeatureType.LINEAR_2D:
        return vals

    elif ftype is constants.FeatureType.BAND:
        # (n_time, n_chan, n_bands) — already canonical, no transpose needed
        return vals

    elif ftype is constants.FeatureType.BANDED_MATRIX:
        # (n_time, n_chan, n_chan, n_bands) → (n_time, n_pairs, n_bands)
        n_time, n_chan, _, n_bands = vals.shape
        if triag:
            tril = np.tril_indices(n_chan, k=-1)
            return vals[:, tril[0], tril[1], :]  # (n_time, n_pairs, n_bands)
        else:
            return vals.reshape(n_time, n_chan * n_chan, n_bands)

    elif ftype is constants.FeatureType.SIMPLE_MATRIX:
        # (n_time, n_chan, n_chan) → (n_time, n_pairs, 1)
        n_time, n_chan, _ = vals.shape
        if triag:
            tril = np.tril_indices(n_chan, k=-1)
            result = vals[:, tril[0], tril[1]]  # (n_time, n_pairs)
        else:
            result = vals.reshape(n_time, n_chan * n_chan)
        return np.expand_dims(result, axis=-1)

    else:
        raise ValueError(f"Unsupported FeatureType for flatten: {ftype}")


def collapse_feature_channels(
    vals: np.ndarray,
    ftype: constants.FeatureType,
) -> np.ndarray:
    """Average feature data across channels, collapsing the channel dimension.

    For non-matrix features the single channel axis (``ftype.channel_axes[0]``)
    is averaged directly.  For matrix features the lower-triangular channel
    pairs are extracted first using the two channel axes, then averaged.

    Parameters
    ----------
    vals : np.ndarray
        Feature array in its canonical extracted shape:

        - LINEAR: ``(n_windows, n_channels)``
        - LINEAR_2D: ``(n_windows, n_channels, n_components)``
        - BAND: ``(n_windows, n_channels, n_bands)``
        - SIMPLE_MATRIX: ``(n_windows, n_channels, n_channels)``
        - BANDED_MATRIX: ``(n_windows, n_channels, n_channels, n_bands)``
        - HIST: ``(n_windows, n_channels, n_freq_bins)``
    ftype : constants.FeatureType
        The classified feature type.  The ``channel_axes`` property is used
        to determine which axes to collapse; no axis positions are hardcoded.

    Returns
    -------
    np.ndarray
        Array with the channel dimension collapsed:

        - LINEAR: ``(n_windows,)``
        - LINEAR_2D: ``(n_windows, n_components)``
        - BAND: ``(n_windows, n_bands)``
        - SIMPLE_MATRIX: ``(n_windows,)``
        - BANDED_MATRIX: ``(n_windows, n_bands)``
        - HIST: ``(n_windows, n_freq_bins)``

    Raises
    ------
    ValueError
        If *ftype* is not a supported feature type.
    """
    channel_axes = ftype.channel_axes

    if not ftype.is_matrix:
        # Single channel axis — average over it.  axis index comes from
        # channel_axes so no position is hardcoded here.
        return np.nanmean(vals, axis=channel_axes[0])

    else:
        # Matrix types (SIMPLE_MATRIX, BANDED_MATRIX): channel_axes = (1, 2).
        # Extract the lower-triangular channel pairs then average over them.
        # n_chan is derived from the first channel axis via channel_axes.
        n_chan = vals.shape[channel_axes[0]]
        tril = np.tril_indices(n_chan, k=-1)
        # Advanced indexing on the two channel axes simultaneously; any
        # trailing semantic axes (e.g. bands) are preserved automatically.
        tril_vals = vals[:, tril[0], tril[1]]  # (W, n_pairs[, ...semantic])
        return np.nanmean(tril_vals, axis=1)

