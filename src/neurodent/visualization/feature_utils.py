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

def extract_linear_array(series: pd.Series) -> np.ndarray:
    """Convert a Series of per-channel arrays into a dense numpy array.

    Works for LINEAR features (scalar per channel), LINEAR_2D features
    (multi-component per channel, e.g. psdslope), and matrix features
    stored as raw arrays.

    Parameters
    ----------
    series : pd.Series
        Each element is a list/array of consistent shape.

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
        If the per-row arrays have inconsistent shapes (ragged data).
    """
    try:
        result = np.asarray(series.tolist())
    except ValueError:
        shapes = {np.asarray(row).shape for row in series}
        raise ValueError(
            f"Ragged input: per-row shapes are not uniform ({shapes}). "
            f"All rows must have the same shape."
        )
    return _ensure_dense(
        result,
        "Ragged input: per-row shapes are not uniform. "
        "All rows must have the same shape.",
    )


# ---------------------------------------------------------------------------
# Dict-stored (banded) features
# ---------------------------------------------------------------------------

def extract_band_dict(series: pd.Series) -> tuple[np.ndarray, list]:
    """Unpack a Series of band dicts into an array plus ordered key list.

    Parameters
    ----------
    series : pd.Series
        Each element is a dict ``{band_name: array}``.

    Returns
    -------
    vals : np.ndarray
        Array of shape ``(n_windows, n_bands, ...)``.
    keys : list
        Ordered band names extracted from the first row.

    Raises
    ------
    ValueError
        If band values have inconsistent shapes across windows (ragged data).
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
    return _ensure_dense(vals, _RAGGED_BAND_MSG), keys


def repack_band_dict(vals: np.ndarray, keys: list) -> list[dict]:
    """Repack a numpy array back into a list of band dicts.

    This is the inverse of :func:`extract_band_dict`.

    Parameters
    ----------
    vals : np.ndarray
        Array of shape ``(n_windows, n_bands, ...)``.
    keys : list
        Ordered band names (same length as ``vals.shape[1]``).

    Returns
    -------
    list[dict]
        One dict per window, each mapping band names to their arrays.
    """
    return [dict(zip(keys, row)) for row in vals]


# ---------------------------------------------------------------------------
# Histogram / spectral (PSD) features
# ---------------------------------------------------------------------------

def extract_hist_data(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a Series of ``(coords, values)`` histogram tuples.

    Handles both pickle-origin tuples and parquet-origin lists.

    Parameters
    ----------
    series : pd.Series
        Each element is a ``(coords, values)`` tuple or list pair.

    Returns
    -------
    coords : np.ndarray
        Frequency coordinates, shape ``(n_windows, n_freq_bins)``
        (or higher-dimensional if multi-channel).
    values : np.ndarray
        Spectral values, shape ``(n_windows, n_freq_bins, n_channels)``
        (or similar, depending on upstream layout).

    Raises
    ------
    ValueError
        If histogram entries have inconsistent shapes across windows
        (ragged data).
    """
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
    return coords, values


# ---------------------------------------------------------------------------
# Channel reshaping utilities
# ---------------------------------------------------------------------------

def flatten_feature_for_plotting(
    vals: np.ndarray,
    ftype: constants.FeatureType,
    triag: bool = True,
) -> np.ndarray:
    """Reshape extracted feature data into a uniform 3-D array for plotting.

    Converts raw extracted feature arrays into the common shape
    ``(n_time, n_features, n_components)`` used by per-channel visualization.

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
        # (n_time, n_bands, n_chan) → (n_time, n_chan, n_bands)
        return vals.transpose((0, 2, 1))

    elif ftype is constants.FeatureType.BANDED_MATRIX:
        # (n_time, n_bands, n_chan, n_chan) → (n_bands, n_chan, n_chan, n_time)
        result = np.moveaxis(vals, 0, -1)
        if triag:
            tril = np.tril_indices(result.shape[1], k=-1)
            result = result[:, tril[0], tril[1], :]
        result = result.reshape(result.shape[0], -1, result.shape[-1])
        # (n_bands, n_pairs, n_time) → (n_time, n_pairs, n_bands)
        return np.transpose(result)

    elif ftype is constants.FeatureType.SIMPLE_MATRIX:
        # (n_time, n_chan, n_chan) → (n_chan, n_chan, n_time)
        result = np.moveaxis(vals, 0, -1)
        if triag:
            tril = np.tril_indices(result.shape[0], k=-1)
            result = result[tril[0], tril[1], :]
        result = result.reshape(-1, result.shape[-1])
        # (n_pairs, n_time) → (n_time, n_pairs)
        result = result.transpose()
        return np.expand_dims(result, axis=-1)

    else:
        raise ValueError(f"Unsupported FeatureType for flatten: {ftype}")


def collapse_feature_channels(
    vals: np.ndarray,
    ftype: constants.FeatureType,
) -> np.ndarray:
    """Average feature data across channels, collapsing the channel dimension.

    For array features the channel axis is averaged directly.  For matrix
    features the lower-triangular channel pairs are extracted first, then
    averaged across those pairs.

    Parameters
    ----------
    vals : np.ndarray
        Feature array in its canonical extracted shape:

        - LINEAR: ``(n_windows, n_channels)``
        - LINEAR_2D: ``(n_windows, n_channels, n_components)``
        - BAND: ``(n_windows, n_bands, n_channels)``
        - SIMPLE_MATRIX: ``(n_windows, n_channels, n_channels)``
        - BANDED_MATRIX: ``(n_windows, n_bands, n_channels, n_channels)``
        - HIST: ``(n_windows, n_channels, n_freq)`` (after transpose)
    ftype : constants.FeatureType
        The classified feature type.

    Returns
    -------
    np.ndarray
        Array with the channel dimension collapsed:

        - LINEAR: ``(n_windows,)``
        - LINEAR_2D: ``(n_windows, n_components)``
        - BAND: ``(n_windows, n_bands)``
        - SIMPLE_MATRIX: ``(n_windows,)``
        - BANDED_MATRIX: ``(n_windows, n_bands)``
        - HIST: ``(n_windows, n_freq)``

    Raises
    ------
    ValueError
        If *ftype* is not a supported feature type.
    """
    if ftype in (constants.FeatureType.LINEAR, constants.FeatureType.LINEAR_2D):
        # (n_windows, n_channels[, n_components]) → average over channels
        return np.nanmean(vals, axis=1)

    elif ftype is constants.FeatureType.BAND:
        # (n_windows, n_bands, n_channels) → average over channels
        return np.nanmean(vals, axis=2)

    elif ftype is constants.FeatureType.HIST:
        # (n_windows, n_channels, n_freq) → average over channels
        return np.nanmean(vals, axis=1)

    elif ftype is constants.FeatureType.SIMPLE_MATRIX:
        # (n_windows, n_chan, n_chan) → extract lower tril, mean across pairs
        tril_indices = np.tril_indices(vals.shape[1], k=-1)
        return np.nanmean(vals[:, tril_indices[0], tril_indices[1]], axis=-1)

    elif ftype is constants.FeatureType.BANDED_MATRIX:
        # (n_windows, n_bands, n_chan, n_chan) → extract lower tril per band, mean
        tril_indices = np.tril_indices(vals.shape[2], k=-1)
        return np.nanmean(vals[:, :, tril_indices[0], tril_indices[1]], axis=-1)

    else:
        raise ValueError(f"Unsupported FeatureType for channel collapse: {ftype}")
