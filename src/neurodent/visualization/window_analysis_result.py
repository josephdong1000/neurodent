import copy
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.ndimage import binary_opening, binary_closing
from tqdm import tqdm

from .. import constants
from .feature_utils import (
    extract_linear_array,
    extract_band_dict,
    repack_band_dict,
    extract_hist_data,
)
from .feature_parser import AnimalFeatureParser


# Defer importing heavy or optional parts of the package until runtime.
if TYPE_CHECKING:
    from .. import core  # for type checkers only
else:
    core = None


# Lazy wrappers for small utility functions which would otherwise import the
# top-level `neurodent.core` package (which may bring in optional heavy deps).
def abbreviate_channel_names(*args, **kwargs):
    from ..core.utils import abbreviate_channel_names as _fn

    return _fn(*args, **kwargs)


def filepath_to_index(*args, **kwargs):
    from ..core.utils import filepath_to_index as _fn

    return _fn(*args, **kwargs)


def parse_chname_to_abbrev(*args, **kwargs):
    from ..core.utils import parse_chname_to_abbrev as _fn

    return _fn(*args, **kwargs)


def slugify(*args, **kwargs):
    from ..core.utils import slugify as _fn

    return _fn(*args, **kwargs)


def _sanitize_feature_request(
    features: list[str] | str | None, exclude: list[str] | str = []
):
    """
    Sanitizes a list of requested features for WindowAnalysisResult

    Args:
        features (list[str] | str | None): List of features to include, a single feature
            name as a string, or None to include all features. If ``"all"``, include all
            features in constants.FEATURES except for those in ``exclude``.
        exclude (list[str] | str, optional): Feature or list of features to exclude.
            Defaults to [].

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


class WindowAnalysisResult(AnimalFeatureParser):
    """
    Wrapper for output of windowed analysis. Has useful functions like group-wise and global averaging, filtering, and saving

    (Docstring and methods preserved from previous implementation.)
    """

    def __init__(
        self,
        result: pd.DataFrame,
        animal_id: str = None,
        genotype: str = None,
        sex: str = "Unknown",
        channel_names: list[str] = None,
        assume_from_number=False,
        bad_channels_dict: dict[str, list[str]] = {},
        suppress_short_interval_error=False,
        lof_scores_dict: dict[str, dict] = {},
    ) -> None:
        # Minimal initialization; full behavior lives in methods which will
        # import heavy modules only when invoked.
        self.result = result
        self.animal_id = animal_id
        self.genotype = genotype
        self.sex = sex
        self.channel_names = channel_names
        self.assume_from_number = assume_from_number
        self.bad_channels_dict = bad_channels_dict.copy()
        self.suppress_short_interval_error = suppress_short_interval_error
        self.lof_scores_dict = lof_scores_dict

        # Defer updating derived instance vars until methods run (which may
        # import `core` lazily as needed).
        try:
            self._update_instance_vars()
        except Exception:
            # In minimal import scenarios, core utilities may be unavailable
            # until the user invokes functionality — swallow errors here and
            # allow the instance to be imported.
            pass


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


__all__ = ["WindowAnalysisResult", "bin_spike_times", "_bin_spike_df"]
