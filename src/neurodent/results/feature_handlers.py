"""Per-``FeatureType`` dispatch for column-level operations.

Each handler encapsulates how to:

* **reorder_pad** — rebuild a feature column for a new channel ordering, padding
  missing target channels with NaN.
* **apply_mask** — NaN-poison cells where a ``(W, C)`` bool mask is False.
* **accumulate** / **finalize** — fold a batch of per-row cells into a
  duration-weighted average state and emit the final aggregated cell.

The eager mutators on :class:`WindowAnalysisResult` and the streaming
``Transform`` classes share these handlers, so per-FeatureType logic lives in
one place. Adding a new ``FeatureType`` = registering one handler in
:data:`FEATURE_HANDLERS`.
"""

from __future__ import annotations

import abc
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from neurodent import constants
from .feature_utils import (
    extract_band_dict,
    extract_hist_data,
    extract_linear_array,
    repack_band_dict,
)


def _reorder_along_axis(
    arr: np.ndarray,
    axis: int,
    channel_map: dict[str, int],
    source_channels: list[str],
    n_target: int,
) -> np.ndarray:
    """Build a new array padded with NaN along `axis`, copying source channels
    into their target positions per ``channel_map``."""
    new_shape = list(arr.shape)
    new_shape[axis] = n_target
    out = np.full(new_shape, np.nan)
    for i, ch in enumerate(source_channels):
        if ch in channel_map:
            src_idx = [slice(None)] * arr.ndim
            dst_idx = [slice(None)] * arr.ndim
            src_idx[axis] = i
            dst_idx[axis] = channel_map[ch]
            out[tuple(dst_idx)] = arr[tuple(src_idx)]
    return out


def _broadcast_mask(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a ``(W, C)`` mask to a higher-rank ``(W, C, ...)`` shape."""
    extra = len(target_shape) - 2
    expanded = mask[(slice(None), slice(None)) + (np.newaxis,) * extra]
    return np.broadcast_to(expanded, target_shape)


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Element-wise division; cells where ``denom == 0`` become NaN."""
    out = np.full(num.shape, np.nan, dtype=float)
    nz = denom != 0
    out[nz] = num[nz] / denom[nz]
    return out


def _weighted_nansum(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell weighted sum + per-cell active-weight sum, ignoring NaN cells.

    ``values`` has shape ``(W, ...)``; ``weights`` has shape ``(W,)``.
    Returns ``(weighted_sum, active_weight_sum)`` each with shape ``(...,)``.
    """
    valid = ~np.isnan(values)
    # Broadcast weights to the cell shape so NaN cells contribute zero weight.
    w_b = np.broadcast_to(
        weights[(slice(None),) + (np.newaxis,) * (values.ndim - 1)], values.shape
    )
    w_valid = np.where(valid, w_b, 0.0)
    v_filled = np.where(valid, values, 0.0)
    return (v_filled * w_valid).sum(axis=0), w_valid.sum(axis=0)


# ---------------------------------------------------------------------------
# Handler base class
# ---------------------------------------------------------------------------


class FeatureHandler(abc.ABC):
    """Per-FeatureType operations used by eager + streaming engines."""

    @abc.abstractmethod
    def reorder_pad(
        self,
        cells: pd.Series,
        channel_map: dict[str, int],
        source_channels: list[str],
        target_channels: list[str],
    ) -> list:
        """Return a new column-value list for the target channel ordering."""

    @abc.abstractmethod
    def apply_mask(self, cells: pd.Series, mask: np.ndarray) -> list:
        """Return a new column-value list with cells NaN'd where ``mask`` is False."""

    @abc.abstractmethod
    def accumulate(self, state: dict, cells: pd.Series, weights: np.ndarray) -> None:
        """Fold a batch slice into the aggregator state (initialised lazily)."""

    @abc.abstractmethod
    def finalize(self, state: dict) -> Any:
        """Emit the final aggregated cell value from the accumulated state."""


# ---------------------------------------------------------------------------
# LINEAR — scalar per channel, cell shape (C,), extracted shape (W, C)
# ---------------------------------------------------------------------------


class LinearHandler(FeatureHandler):
    def reorder_pad(self, cells, channel_map, source_channels, target_channels):
        vals = extract_linear_array(cells)
        new_vals = _reorder_along_axis(vals, 1, channel_map, source_channels, len(target_channels))
        return new_vals.tolist()

    def apply_mask(self, cells, mask):
        vals = extract_linear_array(cells).astype(float, copy=False)
        vals[~mask] = np.nan
        return vals.tolist()

    def accumulate(self, state, cells, weights):
        vals = extract_linear_array(cells).astype(float, copy=False)
        wsum, ws = _weighted_nansum(vals, np.asarray(weights, dtype=float))
        state["sum"] = wsum if "sum" not in state else state["sum"] + wsum
        state["w"] = ws if "w" not in state else state["w"] + ws

    def finalize(self, state):
        return _safe_divide(state["sum"], state["w"])


# ---------------------------------------------------------------------------
# LINEAR_2D — per-channel vector, cell shape (C, K), extracted shape (W, C, K)
# ---------------------------------------------------------------------------


class Linear2DHandler(FeatureHandler):
    def reorder_pad(self, cells, channel_map, source_channels, target_channels):
        vals = extract_linear_array(cells)
        new_vals = _reorder_along_axis(vals, 1, channel_map, source_channels, len(target_channels))
        return new_vals.tolist()

    def apply_mask(self, cells, mask):
        vals = extract_linear_array(cells).astype(float, copy=False)
        m = _broadcast_mask(mask, vals.shape)
        vals[~m] = np.nan
        return vals.tolist()

    def accumulate(self, state, cells, weights):
        vals = extract_linear_array(cells).astype(float, copy=False)
        wsum, ws = _weighted_nansum(vals, np.asarray(weights, dtype=float))
        state["sum"] = wsum if "sum" not in state else state["sum"] + wsum
        state["w"] = ws if "w" not in state else state["w"] + ws

    def finalize(self, state):
        return _safe_divide(state["sum"], state["w"])


# ---------------------------------------------------------------------------
# BAND — dict-stored {band: (C,)} per cell; extracted (W, C, B)
# ---------------------------------------------------------------------------


class BandHandler(FeatureHandler):
    def reorder_pad(self, cells, channel_map, source_channels, target_channels):
        vals, keys = extract_band_dict(cells)
        new_vals = _reorder_along_axis(vals, 1, channel_map, source_channels, len(target_channels))
        return repack_band_dict(new_vals, keys)

    def apply_mask(self, cells, mask):
        vals, keys = extract_band_dict(cells)
        vals = vals.astype(float, copy=False)
        m = _broadcast_mask(mask, vals.shape)
        vals[~m] = np.nan
        return repack_band_dict(vals, keys)

    def accumulate(self, state, cells, weights):
        vals, keys = extract_band_dict(cells)
        vals = vals.astype(float, copy=False)
        wsum, ws = _weighted_nansum(vals, np.asarray(weights, dtype=float))
        state["keys"] = keys
        state["sum"] = wsum if "sum" not in state else state["sum"] + wsum
        state["w"] = ws if "w" not in state else state["w"] + ws

    def finalize(self, state):
        avg = _safe_divide(state["sum"], state["w"])  # (C, B)
        keys = state["keys"]
        return {keys[i]: avg[..., i] for i in range(len(keys))}


# ---------------------------------------------------------------------------
# SIMPLE_MATRIX — (C, C) per cell; extracted (W, C, C)
# ---------------------------------------------------------------------------


class SimpleMatrixHandler(FeatureHandler):
    def reorder_pad(self, cells, channel_map, source_channels, target_channels):
        vals = extract_linear_array(cells)
        # Reorder both channel axes in turn.
        v = _reorder_along_axis(vals, 1, channel_map, source_channels, len(target_channels))
        v = _reorder_along_axis(v, 2, channel_map, source_channels, len(target_channels))
        return v.tolist()

    def apply_mask(self, cells, mask):
        vals = extract_linear_array(cells).astype(float, copy=False)
        m = _broadcast_mask(mask, vals.shape)
        vals[~m] = np.nan
        vals[~m.transpose(0, 2, 1)] = np.nan
        return vals.tolist()

    def accumulate(self, state, cells, weights):
        vals = extract_linear_array(cells).astype(float, copy=False)
        wsum, ws = _weighted_nansum(vals, np.asarray(weights, dtype=float))
        state["sum"] = wsum if "sum" not in state else state["sum"] + wsum
        state["w"] = ws if "w" not in state else state["w"] + ws

    def finalize(self, state):
        return _safe_divide(state["sum"], state["w"])


# ---------------------------------------------------------------------------
# BANDED_MATRIX — dict-stored {band: (C, C)}; extracted (W, C, C, B)
# ---------------------------------------------------------------------------


class BandedMatrixHandler(FeatureHandler):
    def reorder_pad(self, cells, channel_map, source_channels, target_channels):
        vals, keys = extract_band_dict(cells)
        v = _reorder_along_axis(vals, 1, channel_map, source_channels, len(target_channels))
        v = _reorder_along_axis(v, 2, channel_map, source_channels, len(target_channels))
        return repack_band_dict(v, keys)

    def apply_mask(self, cells, mask):
        vals, keys = extract_band_dict(cells)
        vals = vals.astype(float, copy=False)
        # Mask broadcast to (W, C, C, B) via (W, C) → (W, C, 1, 1) → broadcast.
        m_cc = _broadcast_mask(mask, vals.shape[:-1])  # (W, C, C)
        m_full = m_cc[..., np.newaxis]
        vals[~np.broadcast_to(m_full, vals.shape)] = np.nan
        vals[~np.broadcast_to(m_cc.transpose(0, 2, 1)[..., np.newaxis], vals.shape)] = np.nan
        return repack_band_dict(vals, keys)

    def accumulate(self, state, cells, weights):
        vals, keys = extract_band_dict(cells)
        vals = vals.astype(float, copy=False)
        wsum, ws = _weighted_nansum(vals, np.asarray(weights, dtype=float))
        state["keys"] = keys
        state["sum"] = wsum if "sum" not in state else state["sum"] + wsum
        state["w"] = ws if "w" not in state else state["w"] + ws

    def finalize(self, state):
        avg = _safe_divide(state["sum"], state["w"])  # (C, C, B)
        keys = state["keys"]
        return {keys[i]: avg[..., i] for i in range(len(keys))}


# ---------------------------------------------------------------------------
# HIST — per cell (coords, vals_FxC); extracted (W, C, F) values + coords
# ---------------------------------------------------------------------------


class HistHandler(FeatureHandler):
    def reorder_pad(self, cells, channel_map, source_channels, target_channels):
        coords, vals = extract_hist_data(cells)
        new_vals = _reorder_along_axis(vals, 1, channel_map, source_channels, len(target_channels))
        return [(coords[i], new_vals[i].T) for i in range(len(coords))]

    def apply_mask(self, cells, mask):
        coords, vals = extract_hist_data(cells)
        vals = vals.astype(float, copy=False)
        m = _broadcast_mask(mask, vals.shape)
        vals[~m] = np.nan
        return [(coords[i], vals[i].T) for i in range(len(coords))]

    def accumulate(self, state, cells, weights):
        coords, vals = extract_hist_data(cells)
        vals = vals.astype(float, copy=False)
        wsum, ws = _weighted_nansum(vals, np.asarray(weights, dtype=float))
        # All rows in a group share the same coords by construction; keep the first.
        state.setdefault("coords", coords[0])
        state["sum"] = wsum if "sum" not in state else state["sum"] + wsum
        state["w"] = ws if "w" not in state else state["w"] + ws

    def finalize(self, state):
        avg = _safe_divide(state["sum"], state["w"])  # (C, F)
        return (state["coords"], avg.T)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


__all__ = [
    "FEATURE_HANDLERS",
    "FeatureHandler",
    "handler_for",
]


# Backing store — populated below at module import.  Consumers see the
# read-only view :data:`FEATURE_HANDLERS`.
_FEATURE_HANDLERS: dict[constants.FeatureType, FeatureHandler] = {
    constants.FeatureType.LINEAR: LinearHandler(),
    constants.FeatureType.LINEAR_2D: Linear2DHandler(),
    constants.FeatureType.BAND: BandHandler(),
    constants.FeatureType.SIMPLE_MATRIX: SimpleMatrixHandler(),
    constants.FeatureType.BANDED_MATRIX: BandedMatrixHandler(),
    constants.FeatureType.HIST: HistHandler(),
}

#: Read-only mapping from :class:`~neurodent.constants.FeatureType` to its
#: :class:`FeatureHandler`.  Lookup / iteration / membership work as for a
#: dict; direct assignment raises ``TypeError``.  Adding a new FeatureType
#: requires editing this module.
FEATURE_HANDLERS: Mapping[constants.FeatureType, FeatureHandler] = MappingProxyType(_FEATURE_HANDLERS)


def _assert_complete_coverage() -> None:
    """Fail fast at import if any FeatureType is missing a handler.

    Adding a new FeatureType requires:
    1. A new enum value + entry in :data:`neurodent.constants.FEATURE_SHAPES`
       (in ``constants/analysis.py``).
    2. A new :class:`FeatureHandler` subclass registered in
       :data:`FEATURE_HANDLERS` below.

    This check catches the second step's drift — if anyone adds a
    FeatureType but forgets the handler, neurodent fails at import rather
    than at the first call site that touches the unhandled type.
    """
    missing = [ft for ft in constants.FeatureType if ft not in FEATURE_HANDLERS]
    if missing:
        raise RuntimeError(
            "FEATURE_HANDLERS is missing entries for: "
            + ", ".join(ft.name for ft in missing)
            + ". Register a FeatureHandler subclass for each in "
            "src/neurodent/visualization/feature_handlers.py."
        )


_assert_complete_coverage()


def handler_for(feature: str) -> FeatureHandler:
    """Resolve the handler for a feature column by classifying its name."""
    ftype = constants.classify_feature(feature)
    if ftype not in FEATURE_HANDLERS:
        raise ValueError(
            f"Unsupported FeatureType {ftype} for {feature}"
        )
    return FEATURE_HANDLERS[ftype]
