"""Utility functions for windowed analysis results.

The full WindowAnalysisResult class implementation lives in results.py.
This module provides standalone helper functions used by that class and by
other parts of the visualization pipeline.
"""
import copy

import numpy as np
import pandas as pd

from .. import constants


def _sanitize_feature_request(
    features: list[str] | str | None, exclude: list[str] | str | None = None
):
    """
    Sanitizes a list of requested features for WindowAnalysisResult

    Args:
        features (list[str] | str | None): List of features to include, a single feature
            name as a string, or None to include all features. If ``"all"``, include all
            features in constants.FEATURES except for those in ``exclude``.
        exclude (list[str] | str | None, optional): Feature or list of features to exclude.
            Defaults to None.

    Returns:
        list[str]: Sanitized list of features.
    """
    if features is None:
        features = ["all"]
    if isinstance(features, str):
        features = [features]
    if isinstance(exclude, str):
        exclude = [exclude]
    if features == ["all"]:
        feat = copy.deepcopy(constants.FEATURES)
    elif not features:
        raise ValueError("Features cannot be empty")
    else:
        if not all(f in constants.FEATURES for f in features):
            raise ValueError(f"Available features are: {constants.FEATURES}")
        feat = copy.deepcopy(features)
    if exclude is not None:
        for e in exclude:
            try:
                feat.remove(e)
            except ValueError:
                pass
    return feat


def bin_spike_times(
    spike_times: list[float], fragment_durations: list[float]
) -> list[int]:
    """Bin spike times into counts based on fragment durations.

    Args:
        spike_times (list[float]): List of spike timestamps in seconds
        fragment_durations (list[float]): List of fragment durations in seconds

    Returns:
        list[int]: List of spike counts per fragment
    """
    bin_edges = np.cumsum([0] + fragment_durations)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.tolist()


def _bin_spike_df(df: pd.DataFrame, spikes_channel: list[list[float]]) -> np.ndarray:
    """
    Bins spike times into a matrix of shape (n_windows, n_channels), based on duration of each window in df
    """
    durations = df["duration"].tolist()
    out = np.empty((len(durations), len(spikes_channel)))
    for i, spike_times in enumerate(spikes_channel):
        out[:, i] = bin_spike_times(spike_times, durations)
    return out


__all__ = ["bin_spike_times", "_bin_spike_df", "_sanitize_feature_request"]
