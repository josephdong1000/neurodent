"""Filter registry for ``WindowAnalysisResult.apply_filters``.

Each filter is a pure function that takes a stats DataFrame (the cheap
columns it needs) plus channel metadata and returns a ``(W, C)`` boolean
mask: ``True`` = keep, ``False`` = mask out.

The registry decouples filter logic from ``WindowAnalysisResult``, so the
same filter set drives both the eager DataFrame path and the streaming
parquet path. Adding a new filter is a single ``register_filter(...)``
call.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening
from scipy.stats import zscore

from ..core.utils import parse_chname_to_abbrev
from .feature_utils import extract_linear_array


__all__ = [
    "ChannelInfo",
    "FILTER_REGISTRY",
    "FilterScope",
    "FilterSpec",
    "compute_filter_mask",
    "register_filter",
    "required_columns_for_config",
    "update_bad_channels_dict_from_config",
]


class FilterScope(Enum):
    """When the filter's mask is computed relative to the full (W, C) result."""

    PER_ROW = "per_row"
    """``mask[i, j]`` depends only on row ``i`` — streams in a single batch pass."""

    CROSS_ROW = "cross_row"
    """Mask depends on reductions over all rows (e.g. z-score normalisation)."""

    MASK_POST = "mask_post"
    """Operates on an already-computed ``(W, C)`` mask (e.g. morphological smoothing)."""


class ChannelInfo(NamedTuple):
    """Channel metadata passed to filters."""

    channel_names: list[str]
    channel_abbrevs: list[str]
    assume_from_number: bool


@dataclass(frozen=True)
class FilterSpec:
    """Declarative spec for a filter."""

    name: str
    required_columns: tuple[str, ...]
    """Columns the filter reads from the stats DataFrame.

    For filters that have alternative inputs (e.g. ``high_beta`` reads
    ``psdfrac`` *or* ``psdband`` + ``psdtotal``), list every column the
    filter might touch — the streaming engine loads the intersection with
    columns actually present in the parquet schema.
    """
    scope: FilterScope
    apply: Callable[..., np.ndarray]
    """Per-scope signatures:

    - ``PER_ROW`` / ``CROSS_ROW``: ``apply(df_stats, channel_info, n_windows, **params) -> (W, C) bool``
    - ``MASK_POST``: ``apply(filter_mask, df_stats, channel_info, **params) -> (W, C) bool``
    """


# Backing store — mutated only by :func:`register_filter` at module import.
# Consumers see :data:`FILTER_REGISTRY` (the read-only view below).
_FILTER_REGISTRY: dict[str, FilterSpec] = {}

#: Read-only view of the filter registry.  Lookups (``FILTER_REGISTRY["name"]``,
#: iteration, ``"name" in FILTER_REGISTRY``) work as for a dict; direct
#: assignment (``FILTER_REGISTRY["x"] = ...``) raises ``TypeError``.  Use
#: :func:`register_filter` to add entries.
FILTER_REGISTRY: Mapping[str, FilterSpec] = MappingProxyType(_FILTER_REGISTRY)


def register_filter(spec: FilterSpec) -> None:
    """Register a filter. Raises if the name is already taken."""
    if spec.name in _FILTER_REGISTRY:
        raise ValueError(f"Filter {spec.name!r} is already registered")
    _FILTER_REGISTRY[spec.name] = spec


# ---------------------------------------------------------------------------
# Pure filter implementations
# ---------------------------------------------------------------------------


def _filter_logrms_range(
    df_stats: pd.DataFrame,
    channel_info: ChannelInfo,
    n_windows: int,
    *,
    z_range: float = 3,
    **_kwargs: Any,
) -> np.ndarray:
    z = abs(z_range)
    np_rms = extract_linear_array(df_stats["rms"])
    np_logrms = np.log(np_rms)
    np_logrmsz = zscore(np_logrms, axis=0, nan_policy="omit")
    out = np.full(np_logrms.shape, True)
    out[(np_logrmsz > z) | (np_logrmsz < -z)] = False
    return out


def _filter_high_rms(
    df_stats: pd.DataFrame,
    channel_info: ChannelInfo,
    n_windows: int,
    *,
    max_rms: float = 500,
    **_kwargs: Any,
) -> np.ndarray:
    np_rms = extract_linear_array(df_stats["rms"])
    out = np.full(np_rms.shape, True)
    out[np_rms > max_rms] = False
    return out


def _filter_low_rms(
    df_stats: pd.DataFrame,
    channel_info: ChannelInfo,
    n_windows: int,
    *,
    min_rms: float = 30,
    **_kwargs: Any,
) -> np.ndarray:
    np_rms = extract_linear_array(df_stats["rms"])
    out = np.full(np_rms.shape, True)
    out[np_rms < min_rms] = False
    return out


def _filter_high_beta(
    df_stats: pd.DataFrame,
    channel_info: ChannelInfo,
    n_windows: int,
    *,
    max_beta_prop: float = 0.4,
    **_kwargs: Any,
) -> np.ndarray:
    if "psdfrac" in df_stats.columns:
        df_psdfrac = pd.DataFrame(df_stats["psdfrac"].tolist())
        np_prop = np.array(df_psdfrac["beta"].tolist())
    elif "psdband" in df_stats.columns and "psdtotal" in df_stats.columns:
        df_psdband = pd.DataFrame(df_stats["psdband"].tolist())
        np_beta = np.array(df_psdband["beta"].tolist())
        np_total = np.array(df_stats["psdtotal"].tolist())
        np_prop = np_beta / np_total
    else:
        raise ValueError(
            "psdfrac or psdband+psdtotal required for beta power filtering"
        )
    out = np.full(np_prop.shape, True)
    out[np_prop > max_beta_prop] = False
    out = np.broadcast_to(np.all(out, axis=-1)[:, np.newaxis], out.shape)
    return out


def _resolve_channel_indices(
    bad_channels: list[str] | None,
    channel_info: ChannelInfo,
    use_abbrevs: bool | None,
) -> tuple[list[int], list[str], list[str]]:
    """Map ``bad_channels`` → column indices.

    Returns ``(indices, normalised_bad_channels, channel_targets)`` so callers
    can both poison columns and persist the normalised list (e.g. into
    ``bad_channels_dict``).
    """
    bad_channels = [] if bad_channels is None else list(bad_channels)
    channel_targets = (
        channel_info.channel_abbrevs
        if use_abbrevs or use_abbrevs is None
        else channel_info.channel_names
    )
    if use_abbrevs is None:
        bad_channels = [
            parse_chname_to_abbrev(ch, assume_from_number=channel_info.assume_from_number)
            for ch in bad_channels
        ]
    indices: list[int] = []
    for ch in bad_channels:
        if ch in channel_targets:
            indices.append(channel_targets.index(ch))
        else:
            warnings.warn(f"Channel {ch} not found in {channel_targets}")
    return indices, bad_channels, channel_targets


def _filter_reject_channels(
    df_stats: pd.DataFrame,
    channel_info: ChannelInfo,
    n_windows: int,
    *,
    bad_channels: list[str] | None = None,
    use_abbrevs: bool | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    n_channels = len(channel_info.channel_names)
    mask = np.ones((n_windows, n_channels), dtype=bool)
    if bad_channels is None:
        return mask
    indices, _, _ = _resolve_channel_indices(bad_channels, channel_info, use_abbrevs)
    for ch_idx in indices:
        mask[:, ch_idx] = False
    return mask


def _filter_reject_channels_by_session(
    df_stats: pd.DataFrame,
    channel_info: ChannelInfo,
    n_windows: int,
    *,
    bad_channels_dict: dict[str, list[str]] | None = None,
    use_abbrevs: bool | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    n_channels = len(channel_info.channel_names)
    mask = np.ones((n_windows, n_channels), dtype=bool)
    if not bad_channels_dict:
        return mask
    if "animalday" not in df_stats.columns:
        raise ValueError(
            "reject_channels_by_session filter requires 'animalday' column in df_stats"
        )

    # Group row indices by animalday without copying feature columns.
    animaldays = df_stats["animalday"]
    seen = set()
    for animalday in animaldays.unique():
        if animalday not in bad_channels_dict:
            raise ValueError(
                f"No bad channels specified for recording session {animalday}. "
                f"Check that all days are present in bad_channels_dict"
            )
        bad_channels = bad_channels_dict[animalday]
        seen.add(animalday)
        indices, _, channel_targets = _resolve_channel_indices(
            bad_channels, channel_info, use_abbrevs
        )
        session_rows = np.flatnonzero((animaldays == animalday).to_numpy())
        for ch_idx in indices:
            mask[session_rows, ch_idx] = False
    return mask


def _filter_morphological_smoothing(
    filter_mask: np.ndarray,
    df_stats: pd.DataFrame,
    channel_info: ChannelInfo,
    *,
    smoothing_seconds: float,
    **_kwargs: Any,
) -> np.ndarray:
    if "duration" not in df_stats.columns:
        raise ValueError(
            "Cannot calculate window duration - 'duration' column missing"
        )
    window_duration = df_stats["duration"].median()
    structure_size = max(1, int(smoothing_seconds / window_duration))
    if structure_size <= 1:
        return filter_mask
    smoothed_mask = filter_mask.copy()
    structure = np.ones(structure_size)
    for ch_idx in range(filter_mask.shape[1]):
        ch_mask = filter_mask[:, ch_idx]
        ch_mask = binary_opening(ch_mask, structure=structure)
        ch_mask = binary_closing(ch_mask, structure=structure)
        smoothed_mask[:, ch_idx] = ch_mask
    return smoothed_mask


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------


register_filter(FilterSpec(
    name="logrms_range",
    required_columns=("rms",),
    scope=FilterScope.CROSS_ROW,
    apply=_filter_logrms_range,
))
register_filter(FilterSpec(
    name="high_rms",
    required_columns=("rms",),
    scope=FilterScope.PER_ROW,
    apply=_filter_high_rms,
))
register_filter(FilterSpec(
    name="low_rms",
    required_columns=("rms",),
    scope=FilterScope.PER_ROW,
    apply=_filter_low_rms,
))
register_filter(FilterSpec(
    name="high_beta",
    required_columns=("psdfrac", "psdband", "psdtotal"),
    scope=FilterScope.PER_ROW,
    apply=_filter_high_beta,
))
register_filter(FilterSpec(
    name="reject_channels",
    required_columns=(),
    scope=FilterScope.PER_ROW,
    apply=_filter_reject_channels,
))
register_filter(FilterSpec(
    name="reject_channels_by_session",
    required_columns=("animalday",),
    scope=FilterScope.PER_ROW,
    apply=_filter_reject_channels_by_session,
))
register_filter(FilterSpec(
    name="morphological_smoothing",
    required_columns=("duration",),
    scope=FilterScope.MASK_POST,
    apply=_filter_morphological_smoothing,
))


# ---------------------------------------------------------------------------
# Driver (eager + streaming share this)
# ---------------------------------------------------------------------------


def required_columns_for_config(
    filter_config: dict,
    available_columns: set[str] | None = None,
) -> set[str]:
    """Union of columns needed by every enabled filter.

    If *available_columns* is given, the result is restricted to that set
    (so callers asking for an alternative input — e.g. ``psdband`` when only
    ``psdfrac`` is present — don't trigger missing-column errors).
    """
    needed: set[str] = set()
    for name in filter_config:
        spec = FILTER_REGISTRY.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown filter: {name}. Available: {sorted(FILTER_REGISTRY)}"
            )
        needed.update(spec.required_columns)
    if available_columns is not None:
        needed &= available_columns
    return needed


def compute_filter_mask(
    df_stats: pd.DataFrame,
    filter_config: dict,
    channel_info: ChannelInfo,
    n_windows: int | None = None,
) -> np.ndarray:
    """Combine every enabled filter into a single ``(W, C)`` mask.

    Walks ``FILTER_REGISTRY`` — no filter-specific branching here.
    """
    if n_windows is None:
        n_windows = len(df_stats)
    n_channels = len(channel_info.channel_names)

    masks: list[np.ndarray] = []
    mask_post_specs: list[tuple[FilterSpec, dict]] = []

    for name, params in filter_config.items():
        spec = FILTER_REGISTRY.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown filter: {name}. Available: {sorted(FILTER_REGISTRY)}"
            )
        params = params or {}
        if spec.scope is FilterScope.MASK_POST:
            mask_post_specs.append((spec, params))
            continue
        m = spec.apply(df_stats, channel_info, n_windows, **params)
        masks.append(m)
        logging.info(
            f"{name}: filtered {m.size - np.count_nonzero(m)}/{m.size}"
        )

    if masks:
        combined = np.prod(np.stack(masks, axis=-1), axis=-1).astype(bool)
    else:
        combined = np.ones((n_windows, n_channels), dtype=bool)

    for spec, params in mask_post_specs:
        combined = spec.apply(combined, df_stats, channel_info, **params)
        logging.info(f"{spec.name}: applied (smoothing)")

    return combined


def _normalise_to_abbrevs(
    channels: list[str], channel_info: ChannelInfo
) -> list[str]:
    return [
        parse_chname_to_abbrev(ch, assume_from_number=channel_info.assume_from_number)
        for ch in channels
    ]


def update_bad_channels_dict_from_config(
    existing_dict: dict[str, list[str]],
    filter_config: dict,
    channel_info: ChannelInfo,
    animaldays: list[str],
) -> dict[str, list[str]]:
    """Apply ``save_bad_channels`` side effects from a filter_config to a dict.

    Mirrors the legacy behaviour previously embedded in
    ``get_filter_reject_channels`` and
    ``get_filter_reject_channels_by_recording_session``: ``reject_channels``
    normalises to abbrevs before saving (unless ``use_abbrevs is False``);
    ``reject_channels_by_session`` stores the raw per-session channel lists.

    Returns a new dict; ``existing_dict`` is not mutated.
    """
    updated = {k: list(v) for k, v in existing_dict.items()}

    for name in ("reject_channels", "reject_channels_by_session"):
        if name not in filter_config:
            continue
        params = filter_config[name] or {}
        save = params.get("save_bad_channels", "union")
        if save is None:
            continue
        use_abbrevs = params.get("use_abbrevs")

        if name == "reject_channels":
            bad_channels = list(params.get("bad_channels") or [])
            if not bad_channels:
                continue
            channels_to_save = (
                bad_channels
                if use_abbrevs is False
                else _normalise_to_abbrevs(bad_channels, channel_info)
            )
            if save == "overwrite":
                updated = {ad: list(channels_to_save) for ad in animaldays}
            elif save == "union":
                for ad in animaldays:
                    if ad in updated:
                        updated[ad] = sorted(set(updated[ad]) | set(channels_to_save))
                    else:
                        updated[ad] = list(channels_to_save)

        else:  # reject_channels_by_session
            session_dict = params.get("bad_channels_dict") or {}
            if not session_dict:
                continue
            if save == "overwrite":
                updated = {ad: list(chs) for ad, chs in session_dict.items()}
            elif save == "union":
                for ad, chs in session_dict.items():
                    if ad in updated:
                        updated[ad] = sorted(set(updated[ad]) | set(chs))
                    else:
                        updated[ad] = list(chs)

    return updated
