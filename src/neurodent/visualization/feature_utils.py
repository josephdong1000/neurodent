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
    if result.dtype == object:
        shapes = {np.asarray(row).shape for row in series}
        raise ValueError(
            f"Ragged input: per-row shapes are not uniform ({shapes}). "
            f"All rows must have the same shape."
        )
    return result


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
    df_bands = pd.DataFrame(series.tolist())
    keys = list(df_bands.columns)
    try:
        vals = np.asarray(df_bands.values.tolist())
    except ValueError:
        raise ValueError(
            "Ragged input: band values have inconsistent shapes across windows. "
            "All windows must have the same shape per band."
        )
    if vals.dtype == object:
        raise ValueError(
            "Ragged input: band values have inconsistent shapes across windows. "
            "All windows must have the same shape per band."
        )
    return vals, keys


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
        raise ValueError(
            "Ragged input: histogram entries have inconsistent shapes across windows. "
            "All windows must have the same shape."
        )
    if coords.dtype == object or values.dtype == object:
        raise ValueError(
            "Ragged input: histogram entries have inconsistent shapes across windows. "
            "All windows must have the same shape."
        )
    return coords, values
