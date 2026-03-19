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
        Array of shape ``(n_windows, n_channels, n_bands)`` for BAND features,
        or ``(n_windows, n_channels, n_channels, n_bands)`` for BANDED_MATRIX.
        Canonical shape: channels before semantic (band) dimensions.
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
    vals = _ensure_dense(vals, _RAGGED_BAND_MSG)
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
        Frequency coordinates, shape ``(n_windows, n_freq_bins)``.
    values : np.ndarray
        Spectral values, shape ``(n_windows, n_channels, n_freq_bins)``.
        Canonical shape: channels axis 1, freq_bins axis 2.

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
    # Raw stacking yields (W, F, C) since each cell stores (F, C).
    # Transpose to canonical (W, C, F) — channels axis 1, freq_bins axis 2.
    if values.ndim == 3:
        values = values.transpose((0, 2, 1))
    return coords, values


# ---------------------------------------------------------------------------
# Channel reshaping utilities
# ---------------------------------------------------------------------------

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

