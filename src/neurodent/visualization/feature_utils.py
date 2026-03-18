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
    """Convert a Series of per-channel arrays into a 2-D numpy array.

    Works for LINEAR features (scalar per channel) and LINEAR_2D features
    (multi-component per channel, e.g. psdslope).

    Parameters
    ----------
    series : pd.Series
        Each element is a list/array of shape ``(n_channels,)`` or
        ``(n_channels, n_components)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, n_channels)`` or
        ``(n_windows, n_channels, n_components)``.
    """
    return np.array(series.tolist())


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
    """
    df_bands = pd.DataFrame(series.tolist())
    keys = list(df_bands.columns)
    vals = np.array(df_bands.values.tolist())
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
    """
    data = series.tolist()
    coords = np.array([
        np.asarray(item[0]) if isinstance(item, (tuple, list)) and len(item) == 2
        else np.asarray(item)
        for item in data
    ])
    values = np.array([
        np.asarray(item[1]) if isinstance(item, (tuple, list)) and len(item) == 2
        else np.asarray(item)
        for item in data
    ])
    return coords, values
