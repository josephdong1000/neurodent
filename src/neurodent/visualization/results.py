import copy
import fnmatch
import glob
import json
import logging
import os
import re
import tempfile
import time
import warnings
import dateutil.parser
from datetime import datetime, timedelta
from pathlib import Path
from ..core.utils import abbreviate_channel_names, filepath_to_index, parse_chname_to_abbrev, slugify
import numpy as np
import pandas as pd
from typing import Any, Literal, Optional, Union

from .. import constants, core
from scipy.stats import zscore
from scipy.ndimage import binary_opening, binary_closing
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frequency_domain_results import FrequencyDomainSpikeAnalysisResult

import mne
from .feature_utils import extract_linear_array, extract_band_dict, repack_band_dict, extract_hist_data
from .window_analysis_result import bin_spike_times, _bin_spike_df

# Implementation moved to separate modules to keep this file smaller.
# Import and re-export the moved classes to preserve the original public API.
from .feature_parser import AnimalFeatureParser
from .animal_organizer import AnimalOrganizer



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


class WindowAnalysisResult(AnimalFeatureParser):
    """
    Wrapper for output of windowed analysis. Has useful functions like group-wise and global averaging, filtering, and saving

    Args:
        result (pd.DataFrame): Result comes from AnimalOrganizer.compute_windowed_analysis()
        animal_id (str, optional): Identifier for the animal where result was computed from. Defaults to None.
        genotype (str, optional): Genotype of animal. Defaults to None.
        channel_names (list[str], optional): List of channel names. Defaults to None.
        assume_channels (bool, optional): If true, assumes channel names according to AnimalFeatureParser.DEFAULT_CHNUM_TO_NAME. Defaults to False.
        bad_channels_dict (dict[str, list[str]], optional): Dictionary of channels to reject for each recording session. Defaults to {}.
        suppress_short_interval_error (bool, optional): If True, suppress ValueError for short intervals between timestamps. Useful for aggregated WARs with large window sizes. Defaults to False.

    Attributes:
        result (pd.DataFrame): DataFrame containing the windowed analysis results.
        animal_id (str): Identifier for the animal.
        genotype (str): Genotype of the animal.
        channel_names (list[str]): List of channel names.
        channel_abbrevs (list[str]): Abbreviated channel names.
        bad_channels_dict (dict): Dictionary mapping sessions to bad channel names.
        lof_scores_dict (dict): Dictionary of LOF scores for outage detection.
    """

    def __init__(
        self,
        result: pd.DataFrame,
        animal_id: str = None,
        genotype: str = None,
        sex: str = "Unknown",
        channel_names: list[str] = None,
        assume_from_number=False,
        bad_channels_dict: dict[str, list[str]] | None = None,
        suppress_short_interval_error=False,
        lof_scores_dict: dict[str, dict] | None = None,
    ) -> None:
        self.result = result
        self.animal_id = animal_id
        self.genotype = genotype
        self.sex = sex
        self.channel_names = channel_names
        self.assume_from_number = assume_from_number
        self.bad_channels_dict = bad_channels_dict.copy() if bad_channels_dict is not None else {}
        self.suppress_short_interval_error = suppress_short_interval_error
        self.lof_scores_dict = lof_scores_dict if lof_scores_dict is not None else {}

        self._update_instance_vars()

        logging.info(f"Channel names: \t{self.channel_names}")
        logging.info(f"Channel abbreviations: \t{self.channel_abbrevs}")

    def __str__(self) -> str:
        return f"{self.animaldays}"

    def copy(self):
        """
        Create a deep copy of the WindowAnalysisResult object.

        Returns:
            WindowAnalysisResult: A deep copy of the current instance with all attributes copied.
        """
        return WindowAnalysisResult(
            result=self.result.copy(deep=True),
            animal_id=self.animal_id,
            genotype=self.genotype,
            sex=self.sex,
            channel_names=(
                self.channel_names.copy() if self.channel_names is not None else None
            ),
            assume_from_number=self.assume_from_number,
            bad_channels_dict=copy.deepcopy(self.bad_channels_dict),
            suppress_short_interval_error=self.suppress_short_interval_error,
            lof_scores_dict=copy.deepcopy(self.lof_scores_dict),
        )

    @classmethod
    def _from_existing(
        cls, source: "WindowAnalysisResult", result: pd.DataFrame
    ) -> "WindowAnalysisResult":
        """Create a new WindowAnalysisResult by copying metadata from an existing instance.

        This is a shallow copy path: it reuses the source's metadata (animal_id, genotype,
        channel_names, etc.) with a new result DataFrame, without re-running __init__ logging.
        Used by filtering methods to avoid redundant log output during chained operations.

        Args:
            source: The existing WindowAnalysisResult to copy metadata from.
            result: The new result DataFrame for the new instance.

        Returns:
            A new WindowAnalysisResult with the given result and source's metadata.
        """
        new_war = cls.__new__(cls)
        new_war.result = result
        new_war.animal_id = source.animal_id
        new_war.genotype = source.genotype
        new_war.sex = source.sex
        new_war.channel_names = source.channel_names
        new_war.assume_from_number = source.assume_from_number
        new_war.bad_channels_dict = source.bad_channels_dict.copy()
        new_war.suppress_short_interval_error = source.suppress_short_interval_error
        new_war.lof_scores_dict = source.lof_scores_dict.copy()
        new_war._update_instance_vars()
        return new_war

    def _update_instance_vars(self):
        """Run after updating self.result, or other init values"""
        if "index" in self.result.columns:
            warnings.warn("Dropping column 'index'")
            self.result = self.result.drop(columns=["index"])

        # Check if timestamps are sorted and sort if needed
        if "timestamp" in self.result.columns:
            if not self.result["timestamp"].is_monotonic_increasing:
                warnings.warn(
                    "Timestamps are not sorted. Sorting result DataFrame by timestamp."
                )
                self.result = self.result.sort_values("timestamp")

        # Check for unusually short intervals between timestamps
        if "timestamp" in self.result.columns and "duration" in self.result.columns:
            median_duration = self.result["duration"].median()
            timestamp_diffs = self.result["timestamp"].diff()
            short_intervals = timestamp_diffs < pd.Timedelta(seconds=median_duration)

            # Skip first row since diff() produces NaT
            short_intervals = short_intervals[1:]

            if short_intervals.any():
                n_short = short_intervals.sum()
                pct_short = (n_short / len(short_intervals)) * 100

                warning_msg = (
                    f"Found {n_short} intervals ({pct_short:.1f}%) between timestamps "
                    f"that are shorter than the median duration of {median_duration:.1f}s"
                )

                if (
                    pct_short > 1.0 and not self.suppress_short_interval_error
                ):  # More than 1% of intervals are short
                    raise ValueError(warning_msg)
                elif not self.suppress_short_interval_error:
                    warnings.warn(warning_msg)

        if "animal" in self.result.columns:
            unique_animals = self.result["animal"].unique()
            if len(unique_animals) > 1:
                raise ValueError(f"Multiple animals found in result: {unique_animals}")
            if unique_animals[0] != self.animal_id:
                raise ValueError(
                    f"Animal ID mismatch: result has {unique_animals[0]}, but self.animal_id is {self.animal_id}"
                )

        self._feature_columns = [
            x for x in self.result.columns if x in constants.FEATURES
        ]
        self._nonfeature_columns = [
            x for x in self.result.columns if x not in constants.FEATURES
        ]
        self.animaldays = self.result.loc[:, "animalday"].unique()

        # Ensure bad_channels_dict and lof_scores_dict have entries for all animaldays
        # This fixes the issue where windowed analysis creates per-date animaldays
        # but bad_channels_dict only has LRO-level (per-folder) entries
        for animalday in self.animaldays:
            if animalday not in self.bad_channels_dict:
                # Add missing animalday with empty bad channels list
                self.bad_channels_dict[animalday] = []
                logging.info(
                    f"Added missing animalday to bad_channels_dict: {animalday}"
                )

            if animalday not in self.lof_scores_dict:
                # Add missing animalday with empty LOF scores
                # NOTE: Both lof_scores AND channel_names must be empty to maintain invariant!
                self.lof_scores_dict[animalday] = {
                    "lof_scores": [],
                    "channel_names": [],  # Must be empty to match empty lof_scores!
                }
                logging.warning(
                    f"Added missing animalday to lof_scores_dict: {animalday}. "
                    f"This indicates LOF scores were not computed for this session. "
                    f"It will be excluded from LOF-based analysis."
                )

        try:
            self.channel_abbrevs = [
                core.parse_chname_to_abbrev(x, assume_from_number=self.assume_from_number)
                for x in self.channel_names
            ]
        except (ValueError, KeyError) as e:
            raise type(e)(
                f"{e}\n\nChannel names in data: {self.channel_names}"
            ) from e

    def reorder_and_pad_channels(
        self, target_channels: list[str], use_abbrevs: bool = True, inplace: bool = True
    ) -> pd.DataFrame:
        """Reorder and pad channels to match a target channel list.

        This method ensures that the data has a consistent channel order and structure
        by reordering existing channels and padding missing channels with NaNs.

        Args:
            target_channels (list[str]): List of target channel names to match
            use_abbrevs (bool, optional): If True, target channel names are read as channel abbreviations instead of channel names. Defaults to True.
            inplace (bool, optional): If True, modify the result in place. Defaults to True.
        Returns:
            pd.DataFrame: DataFrame with reordered and padded channels
        """
        duplicates = [ch for ch in target_channels if target_channels.count(ch) > 1]
        if duplicates:
            raise ValueError(
                f"Target channels must be unique. Found duplicates: {duplicates}"
            )

        result = self.result.copy()

        channel_map = {ch: i for i, ch in enumerate(target_channels)}
        channel_names = self.channel_names if not use_abbrevs else self.channel_abbrevs

        valid_channels = [ch for ch in channel_names if ch in channel_map]
        if not valid_channels:
            warnings.warn(
                f"None of the channel names {channel_names} were found in target channels {target_channels}. Is use_abbrevs correctly set?"
            )

        for feature in self._feature_columns:
            ftype = constants.classify_feature(feature)

            if ftype in (constants.FeatureType.LINEAR, constants.FeatureType.LINEAR_2D, constants.FeatureType.BAND):
                if ftype is constants.FeatureType.BAND:
                    vals, keys = extract_band_dict(result[feature])
                    # vals is canonical (W, C, B) — no transpose needed
                else:
                    vals = extract_linear_array(result[feature])

                # vals has shape (n_rows, n_channels, *extra_dims). We allocate an array
                # with the same leading and trailing dimensions but with the channel axis
                # sized to len(target_channels). Missing channels are padded with NaN and
                # existing channels are copied in via channel_map below.
                new_vals = np.full(
                    (vals.shape[0], len(target_channels), *vals.shape[2:]), np.nan
                )

                for i, ch in enumerate(channel_names):
                    if ch in channel_map:
                        new_vals[:, channel_map[ch]] = vals[:, i]

                if ftype is constants.FeatureType.BAND:
                    # new_vals is (W, n_target, B) — canonical, pass directly to repack
                    result[feature] = repack_band_dict(new_vals, keys)
                else:
                    result[feature] = [list(x) for x in new_vals]

            elif ftype.is_matrix:
                if ftype is constants.FeatureType.BANDED_MATRIX:
                    vals, keys = extract_band_dict(result[feature])
                    # vals is canonical (W, C, C, B)
                    logging.debug(f"vals.shape: {vals.shape}")
                    n_bands = vals.shape[ftype.semantic_axes["bands"]]
                    new_shape = [vals.shape[0], len(target_channels), len(target_channels), n_bands]
                    new_vals = np.full(new_shape, np.nan)
                    for i, ch1 in enumerate(channel_names):
                        if ch1 in channel_map:
                            for j, ch2 in enumerate(channel_names):
                                if ch2 in channel_map:
                                    new_vals[:, channel_map[ch1], channel_map[ch2], :] = vals[:, i, j, :]
                    result[feature] = repack_band_dict(new_vals, keys)
                else:
                    vals = extract_linear_array(result[feature])
                    # vals is canonical (W, C, C) for SIMPLE_MATRIX
                    logging.debug(f"vals.shape: {vals.shape}")
                    new_shape = [vals.shape[0], len(target_channels), len(target_channels)]
                    new_vals = np.full(new_shape, np.nan)
                    for i, ch1 in enumerate(channel_names):
                        if ch1 in channel_map:
                            for j, ch2 in enumerate(channel_names):
                                if ch2 in channel_map:
                                    new_vals[:, channel_map[ch1], channel_map[ch2]] = vals[:, i, j]
                    result[feature] = [list(x) for x in new_vals]

            elif ftype is constants.FeatureType.HIST:
                coords, vals = extract_hist_data(result[feature])
                # vals is canonical (W, C, F)
                new_vals = np.full(
                    (vals.shape[0], len(target_channels), vals.shape[ftype.semantic_axes["freq_bins"]]), np.nan
                )

                for i, ch in enumerate(channel_names):
                    if ch in channel_map:
                        new_vals[:, channel_map[ch], :] = vals[:, i, :]

                # Repack as (F, C) per cell to preserve per-cell storage format
                result[feature] = [
                    (coords[i], new_vals[i].T) for i in range(len(coords))
                ]

            else:
                raise ValueError(
                    f"Unsupported FeatureType {ftype} for channel remapping: {feature}"
                )

        if inplace:
            self.result = result

            logging.debug(f"Old channel names: {self.channel_names}")
            self.channel_names = target_channels
            logging.debug(f"New channel names: {self.channel_names}")

            logging.debug(f"Old channel abbreviations: {self.channel_abbrevs}")
            self._update_instance_vars()
            logging.debug(f"New channel abbreviations: {self.channel_abbrevs}")

        return result

    def read_sars_spikes(
        self,
        sars: list["FrequencyDomainSpikeAnalysisResult"],
        read_mode: Literal["sa", "mne"] = "sa",
        inplace=True,
    ):
        """
        Integrate spike analysis results into WAR by adding nspike/lognspike features.

        This method extracts spike timing information from spike detection results and bins
        them according to the WAR's time windows, adding spike count features to each row.

        Args:
            sars: List of FrequencyDomainSpikeAnalysisResult objects.
                  One result per recording session (animalday).
            read_mode: Mode for extracting spike data:
                - "sa": Read from SortingAnalyzer objects (result_sas attribute)
                - "mne": Read from MNE RawArray objects (result_mne attribute)
            inplace: If True, modifies self.result and returns self.
                    If False, returns a new WindowAnalysisResult.

        Returns:
            WindowAnalysisResult: WAR object with added spike features (nspike, lognspike).
                - If inplace=True: returns self with modified result DataFrame
                - If inplace=False: returns new WAR object with enhanced result DataFrame

        Notes:
            - The number of sars must match the number of unique animaldays in self.result
            - Spikes are binned into time windows matching the existing WAR fragments
            - nspike: array of spike counts per channel for each time window
            - lognspike: log-transformed spike counts using core.log_transform()

        Example:
            >>> # After computing WAR and spike detection
            >>> enhanced_war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=False)
            >>> enhanced_war.result['nspike']  # Spike counts per channel per window
        """
        match read_mode:
            case "sa":
                spikes_all = []
                for sar in sars:  # for each continuous recording session
                    spikes_channel = []
                    for i, sa in enumerate(sar.result_sas):  # for each channel
                        spike_times = []
                        for unit in sa.sorting.get_unit_ids():  # Flatten units
                            spike_times.extend(
                                sa.sorting.get_unit_spike_train(unit_id=unit).tolist()
                            )
                        spike_times = (
                            np.array(spike_times) / sa.sorting.get_sampling_frequency()
                        )
                        spikes_channel.append(spike_times)
                    spikes_all.append(spikes_channel)
                return self._read_from_spikes_all(spikes_all, inplace=inplace)
            case "mne":
                raws = [sar.result_mne for sar in sars]
                return self.read_mnes_spikes(raws, inplace=inplace)
            case _:
                raise ValueError(f"Invalid read_mode: {read_mode}")

    def read_mnes_spikes(self, raws: list[mne.io.RawArray], inplace=True):
        """
        Extract spike features from MNE RawArray objects with spike annotations.

        This method extracts spike timing from MNE annotations (where spikes are marked
        with channel-specific event labels) and bins them into WAR time windows.

        Args:
            raws: List of MNE RawArray objects with spike annotations. One per recording
                  session (animalday). Each should have annotations with channel names
                  as event labels (e.g., 'LMot', 'RMot', etc.).
            inplace: If True, modifies self.result and returns self.
                    If False, returns a new WindowAnalysisResult.

        Returns:
            WindowAnalysisResult: WAR object with added spike features (nspike, lognspike).

        Notes:
            - Expects MNE annotations with channel names as event descriptions
            - Spike times are extracted from event onsets and binned to WAR windows
            - Channels not found in annotations will have empty spike arrays
            - Delegates to _read_from_spikes_all() for the actual binning logic

        Example:
            >>> # From MNE spike annotations
            >>> enhanced_war = war.read_mnes_spikes([mne_raw1, mne_raw2], inplace=False)
        """
        spikes_all = []
        for raw in raws:
            # each mne is a contiguous recording session
            events, event_id = mne.events_from_annotations(raw)
            event_id = {k.item(): v for k, v in event_id.items()}

            spikes_channel = []
            for channel in raw.ch_names:
                if channel not in event_id.keys():
                    logging.warning(f"Channel {channel} not found in event_id")
                    spikes_channel.append([])
                    continue
                event_id_channel = event_id[channel]
                spike_times = events[events[:, 2] == event_id_channel, 0]
                spike_times = spike_times / raw.info["sfreq"]
                spikes_channel.append(spike_times)
            spikes_all.append(spikes_channel)
        return self._read_from_spikes_all(spikes_all, inplace=inplace)

    def _read_from_spikes_all(self, spikes_all: list[list[list[float]]], inplace=True):
        """
        Internal method to bin spike times into WAR time windows and add as features.

        This is the common endpoint for both read_sars_spikes() and read_mnes_spikes().
        It bins spike times according to the WAR's time windows and adds nspike/lognspike
        features to the result DataFrame.

        Args:
            spikes_all: Nested list structure of spike times in seconds:
                - Outer list: recording sessions (one per animalday)
                - Middle list: channels (one per EEG channel)
                - Inner list/array: spike times in seconds for that channel
                Example: [[[0.5, 1.2], [0.8]], [[1.1, 2.3], []]]
                         = 2 sessions, 2 channels each
            inplace: If True, modifies self.result and returns self.
                    If False, returns a new WindowAnalysisResult with enhanced data.

        Returns:
            WindowAnalysisResult: WAR object with spike features added to result DataFrame.

        Notes:
            - Groups self.result by 'animalday' and matches to spikes_all by index
            - Uses _bin_spike_df() helper to count spikes within each time window
            - Adds two new columns:
                - 'nspike': array of spike counts per channel for each window
                - 'lognspike': log-transformed spike counts via core.log_transform()
            - Warns if spike count size doesn't match result DataFrame size
        """
        # Each groupby animalday is a recording session
        grouped = self.result.groupby("animalday")
        animaldays = grouped.groups.keys()
        logging.debug(f"Animal days: {animaldays}")
        spike_counts = dict(zip(animaldays, spikes_all))
        spike_counts = grouped.apply(
            lambda x: _bin_spike_df(x, spikes_channel=spike_counts[x.name])
        )
        spike_counts: pd.Series = spike_counts.explode()

        if spike_counts.size != self.result.shape[0]:
            logging.warning(
                f"Spike counts size {spike_counts.size} does not match result size {self.result.shape[0]}"
            )

        result = self.result.copy()
        result["nspike"] = spike_counts.tolist()
        result["lognspike"] = list(
            core.log_transform(np.stack(result["nspike"].tolist(), axis=0))
        )
        if inplace:
            self.result = result
            return self
        else:
            # Create a new WindowAnalysisResult
            new_war = copy.deepcopy(self)
            new_war.result = result
            return new_war

    def get_info(self):
        """Returns a formatted string with basic information about the WindowAnalysisResult object"""
        info = []
        info.append(f"feature names: {', '.join(self._feature_columns)}")
        info.append(f"animaldays: {', '.join(self.result['animalday'].unique())}")
        info.append(
            f"animal_id: {self.result['animal'].unique()[0] if 'animal' in self.result.columns else self.animal_id}"
        )
        info.append(
            f"genotype: {self.result['genotype'].unique()[0] if 'genotype' in self.result.columns else self.genotype}"
        )
        info.append(f"sex: {self.sex}")
        info.append(
            f"channel_names: {', '.join(self.channel_names) if self.channel_names else 'None'}"
        )

        return "\n".join(info)

    def get_result(
        self,
        features: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        allow_missing=False,
    ):
        """Get windowed analysis result dataframe, with helpful filters

        Args:
            features (list[str] | str | None, optional): Feature name, list of feature names,
                or None to return all features. Defaults to None (all features).
            exclude (list[str] | str, optional): Feature name or list of feature names to
                exclude from result; will override the features parameter. Defaults to [].
            allow_missing (bool, optional): If True, will return all requested features as columns regardless if they exist in result. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with features in columns and windows in rows
        """
        features = _sanitize_feature_request(features, exclude)
        if not allow_missing:
            return self.result.loc[:, self._nonfeature_columns + features]
        else:
            return self.result.reindex(columns=self._nonfeature_columns + features)

    def get_groupavg_result(
        self,
        features: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        df: pd.DataFrame = None,
        groupby="animalday",
    ):
        """Group result and average within groups. Preserves data structure and shape for each feature.

        Args:
            features (list[str] | str | None, optional): Feature name, list of feature names,
                or None to return all features. Defaults to None (all features).
            exclude (list[str] | str, optional): Feature name or list of feature names to
                exclude from result. Will override the features parameter. Defaults to [].
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            groupby (str, optional): Feature or list of features to group by before averaging. Passed to the `by` parameter in pd.DataFrame.groupby(). Defaults to "animalday".

        Returns:
            pd.DataFrame: Result grouped by `groupby` and averaged for each group.
        """
        result_grouped, result_validcols = self.__get_groups(
            features=features, exclude=exclude, df=df, groupby=groupby
        )
        features = _sanitize_feature_request(features, exclude)

        avg_results = []
        for f in features:
            if f in result_validcols:
                avg_result_col = result_grouped.apply(
                    self._average_feature, f, "duration", include_groups=False
                )
                avg_result_col.name = f
                avg_results.append(avg_result_col)
            else:
                logging.warning(f"{f} not calculated, skipping")

        return pd.concat(avg_results, axis=1)

    def __get_groups(
        self,
        features: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        df: pd.DataFrame = None,
        groupby="animalday",
    ):
        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df
        return result_win.groupby(groupby), result_win.columns

    def get_grouprows_result(
        self,
        features: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        df: pd.DataFrame = None,
        multiindex=["animalday", "animal", "genotype"],
        include=["duration", "endfile"],
    ):
        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df
        result_win = result_win.filter(features + multiindex + include)
        return result_win.set_index(multiindex)

    def get_channel_averaged_result(
        self,
        features: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Get windowed analysis result with features averaged across channels.

        This method collapses the channel dimension for all requested features,
        converting multi-channel data to scalar values per time window. It handles
        three types of features differently:

        1. **Linear features** (logrms, rms, etc.): Simple average across channels
        2. **Band features** (logpsdband, logpsdfrac, etc.): Extracts each frequency
           band (delta, theta, alpha, beta, gamma) and averages across channels.
           Creates columns like: logpsdband_delta, logpsdband_theta, etc.
        3. **Matrix features** (zcohere, zimcoh, cohere, imcoh): Extracts each
           frequency band's connectivity matrix and averages the upper triangle
           (excluding diagonal). Creates columns like: zcohere_delta, zcohere_theta, etc.

        Args:
            features (list[str] | str | None, optional): Feature name, list of feature names,
                or None to return all features. Can include any combination of linear, band,
                or matrix features. Defaults to None (all features).
            exclude (list[str] | str, optional): Feature name or list of feature names to
                exclude. Defaults to [].
            df (pd.DataFrame, optional): If provided, use this dataframe instead of
                self.result. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with all features averaged to scalars per time window.
                - Non-feature columns (timestamp, animalday, etc.) are preserved
                - Band features expanded to 5 columns per feature (one per frequency band)
                - Matrix features expanded to 5 columns per feature (one per frequency band)
                - All feature values are scalars (float)

        Example:
            >>> war = WindowAnalysisResult.load_pickle_and_json(folder_path, "war.pkl", "war_metadata.json")
            >>> # Get channel-averaged zeitgeber features
            >>> df = war.get_channel_averaged_result(["logpsdband", "zcohere", "logrms"])
            >>> print(df.columns)
            ['timestamp', 'animalday', 'genotype', 'logrms',
             'logpsdband_delta', 'logpsdband_theta', 'logpsdband_alpha', 'logpsdband_beta', 'logpsdband_gamma',
             'zcohere_delta', 'zcohere_theta', 'zcohere_alpha', 'zcohere_beta', 'zcohere_gamma']
            >>> # All feature values are scalars
            >>> df['logpsdband_delta'].iloc[0]  # Returns a single float

        Note:
            This method is designed for temporal analyses (like zeitgeber) where you want
            to analyze feature trends over time without the channel dimension.
            For analyses that need channel information, use get_result() instead.

        See Also:
            - get_result(): Get features with full channel information
            - get_groupavg_result(): Average features across time windows (preserves channels)
        """
        from neurodent import constants

        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df

        # Filter to only features that exist in the dataframe
        available_features = [f for f in features if f in result_win.columns]

        # Get the base result with requested features
        df_result = result_win.loc[
            :, self._nonfeature_columns + available_features
        ].copy()

        # Classify features by type
        band_features_in_data = [
            f for f in available_features if f in constants.BAND_FEATURES
        ]
        banded_matrix_features_in_data = [
            f for f in available_features if f in constants.BANDED_MATRIX_FEATURES
        ]
        simple_matrix_features_in_data = [
            f for f in available_features if f in constants.SIMPLE_MATRIX_FEATURES
        ]
        simple_features_in_data = [
            f for f in available_features if f in constants.LINEAR_FEATURES
        ]

        # Process band features - extract all 5 bands
        for band_feature in band_features_in_data:
            if band_feature in df_result.columns:
                df_result = self._extract_band_features(
                    df_result, band_feature, constants.BAND_NAMES
                )

        # Process banded matrix features - extract all 5 bands
        for matrix_feature in banded_matrix_features_in_data:
            if matrix_feature in df_result.columns:
                df_result = self._extract_banded_matrix_features(
                    df_result, matrix_feature, constants.BAND_NAMES
                )

        # Build list of features to average
        features_to_average = []
        features_to_average.extend(simple_features_in_data)
        features_to_average.extend(
            simple_matrix_features_in_data
        )  # pcorr, zpcorr (no bands)

        for band_feature in band_features_in_data:
            for band in constants.BAND_NAMES:
                features_to_average.append(f"{band_feature}_{band}")

        for matrix_feature in banded_matrix_features_in_data:
            for band in constants.BAND_NAMES:
                features_to_average.append(f"{matrix_feature}_{band}")

        # Average all features across channels
        df_result = self._average_across_channels(df_result, features_to_average)

        # Drop original band/banded-matrix features (now that bands are extracted into separate columns)
        # These are no longer needed and cannot be aggregated (contain dicts/arrays)
        features_to_drop = band_features_in_data + banded_matrix_features_in_data
        df_result = df_result.drop(columns=features_to_drop, errors="ignore")

        return df_result

    def _extract_band_features(
        self, df: pd.DataFrame, feature_name: str, band_names: list[str]
    ) -> pd.DataFrame:
        """Extract individual frequency bands from band features.

        Band features (logpsdband, logpsdfrac, etc.) are stored as dicts with
        band names as keys and channel arrays as values.

        Args:
            df: DataFrame containing the band feature
            feature_name: Name of the band feature column
            band_names: List of band names to extract

        Returns:
            DataFrame with new columns for each band (feature_name_bandname format)
        """
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        if feature_name not in df.columns:
            return df

        # Determine number of windows and channels from first element
        first_element = df[feature_name].iloc[0]
        if not isinstance(first_element, dict):
            raise ValueError(
                f"Band feature {feature_name} must be a dictionary of bands. "
                f"Got {type(first_element)}. If this is a linear feature, fix constants."
            )

        # Pre-allocate columns for all expected bands to ensure consistency
        for band_name in band_names:
            band_values = []
            for i, row_dict in enumerate(df[feature_name]):
                if not isinstance(row_dict, dict):
                    logger.warning(
                        f"Row {i} of {feature_name} is not a dict. Using NaNs."
                    )
                    band_values.append(np.full(len(self.channel_names), np.nan))
                    continue

                if band_name in row_dict:
                    val = row_dict[band_name]
                    if isinstance(val, list):
                        val = np.array(val)
                    band_values.append(val)
                else:
                    logger.warning(
                        f"Band {band_name} missing in {feature_name} at row {i}"
                    )
                    band_values.append(np.full(len(self.channel_names), np.nan))

            # Store as list of arrays/values
            df[f"{feature_name}_{band_name}"] = band_values

        return df

    def _extract_banded_matrix_features(
        self, df: pd.DataFrame, feature_name: str, band_names: list[str]
    ) -> pd.DataFrame:
        """Extract individual frequency bands from banded matrix features.

        This method handles banded matrix features (cohere, zcohere, imcoh, zimcoh)
        which are stored as dicts with band names as keys mapping to 2D matrices.

        Note: Simple matrix features (pcorr, zpcorr) should NOT be processed by this
        method - they are single 2D matrices without frequency band structure.

        Args:
            df: DataFrame containing the banded matrix feature
            feature_name: Name of the banded matrix feature column
            band_names: List of band names to extract

        Returns:
            DataFrame with new columns for each band (feature_name_bandname format)
        """
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        if feature_name not in df.columns:
            return df

        # Check first element to determine storage format
        first_element = df[feature_name].iloc[0]

        if isinstance(first_element, dict):
            for band_name in band_names:
                band_matrices = []
                for matrix_dict in df[feature_name]:
                    if isinstance(matrix_dict, dict) and band_name in matrix_dict:
                        matrix = matrix_dict[band_name]
                        # Convert list to numpy array if needed (legacy format)
                        if isinstance(matrix, list):
                            matrix = np.array(matrix)

                        if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                            band_matrices.append(matrix)
                        else:
                            logger.warning(
                                f"Expected 2D matrix for {feature_name}[{band_name}], "
                                f"got {type(matrix)} with shape {getattr(matrix, 'shape', 'N/A')}"
                            )
                            band_matrices.append(
                                np.full(
                                    (len(self.channel_names), len(self.channel_names)),
                                    np.nan,
                                )
                            )
                    else:
                        logger.warning(
                            f"Missing band {band_name} in {feature_name} dictionary"
                        )
                        band_matrices.append(
                            np.full(
                                (len(self.channel_names), len(self.channel_names)),
                                np.nan,
                            )
                        )

                df[f"{feature_name}_{band_name}"] = band_matrices

        elif isinstance(first_element, (np.ndarray, list)):
            if isinstance(first_element, list):
                first_element = np.array(first_element)

            if first_element.ndim == 3:
                # 3D Array format: (Bands, Ch, Ch)
                # Verify band count matches
                if first_element.shape[0] != len(band_names):
                    raise ValueError(
                        f"Matrix feature {feature_name} has {first_element.shape[0]} bands, "
                        f"but {len(band_names)} were expected ({band_names})."
                    )

                for i, band_name in enumerate(band_names):
                    band_matrices = []
                    for matrix_3d in df[feature_name]:
                        if isinstance(matrix_3d, list):
                            matrix_3d = np.array(matrix_3d)

                        if isinstance(matrix_3d, np.ndarray) and matrix_3d.ndim == 3:
                            if matrix_3d.shape[0] == len(band_names):
                                band_matrices.append(matrix_3d[i, :, :])
                            else:
                                raise ValueError(
                                    f"Band count mismatch for {feature_name}: "
                                    f"array has {matrix_3d.shape[0]} bands, expected {len(band_names)}."
                                )
                        else:
                            raise ValueError(
                                f"Expected 3D matrix for {feature_name}, "
                                f"got {type(matrix_3d)} with shape {getattr(matrix_3d, 'shape', 'N/A')}"
                            )

                    df[f"{feature_name}_{band_name}"] = band_matrices

            elif first_element.ndim == 2:
                raise ValueError(
                    f"Matrix feature {feature_name} is stored as a 2D array, but is defined as a "
                    f"banded feature. Expected a dictionary with band keys or a 3D array (Bands, Ch, Ch). "
                    f"If this feature should not have bands, add it to SIMPLE_MATRIX_FEATURES in constants."
                )
            else:
                raise ValueError(
                    f"Matrix feature {feature_name} has wrong dimensionality: {first_element.ndim}D. "
                    f"Expected 3D (Bands, Ch, Ch) or dict."
                )

        else:
            raise ValueError(
                f"Banded matrix feature {feature_name} has unexpected format: {type(first_element)}. "
                f"Expected dict with band keys or 3D array. If this is a simple matrix feature (pcorr, zpcorr), "
                f"it should not be processed by this method."
            )

        return df

    def _average_across_channels(
        self, df: pd.DataFrame, features: list[str]
    ) -> pd.DataFrame:
        """Average features across channels to produce scalar values.

        This method operates on *expanded* feature columns (e.g.
        ``cohere_delta``, ``psdband_theta``) that have already been unpacked
        from their dict-stored representation by
        :meth:`_extract_band_features` / :meth:`_extract_banded_matrix_features`.
        Because expanded names do not exist in :data:`constants.FEATURE_TYPES`,
        dispatch is based on array dimensionality rather than
        :func:`classify_feature`.

        Handles two types of features:
        - Vector features (1D arrays): Average across channels
        - Matrix features (2D arrays): Average upper triangle (excluding diagonal)

        Args:
            df: DataFrame with features as columns
            features: List of feature column names to average

        Returns:
            DataFrame with averaged features replacing original arrays
        """
        for feature in features:
            if feature not in df.columns:
                continue

            first_element = df[feature].iloc[0]

            if isinstance(first_element, (np.ndarray, list)):
                if isinstance(first_element, list):
                    first_element = np.array(first_element)

                if first_element.ndim == 1:
                    # Vector features: Mean across channels
                    try:
                        feature_arrays = extract_linear_array(df[feature])
                        feature_avg = np.nanmean(feature_arrays, axis=1)
                    except ValueError as e:
                        raise ValueError(
                            f"Feature {feature} has inconsistent channel counts across windows. "
                            f"All windows must have the same number of channels. "
                            f"This likely indicates data corruption during feature extraction. "
                            f"Original error: {e}"
                        ) from e

                    df[feature] = feature_avg

                elif first_element.ndim == 2:
                    # Matrix features: Mean of upper triangle
                    feature_avg = []
                    for matrix in df[feature].values:
                        if isinstance(matrix, list):
                            matrix = np.array(matrix)

                        # Validate matrix shape
                        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                            logging.warning(
                                f"Expected 2D matrix for {feature}, "
                                f"got {type(matrix)} with ndim {getattr(matrix, 'ndim', 'N/A')}"
                            )
                            feature_avg.append(np.nan)
                            continue

                        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                            # Can't get upper triangle (excluding diag) from 1x1 or smaller
                            feature_avg.append(
                                np.nanmean(matrix) if matrix.size > 0 else np.nan
                            )
                            continue

                        upper_tri_indices = np.triu_indices_from(matrix, k=1)
                        upper_tri_values = matrix[upper_tri_indices]

                        if len(upper_tri_values) == 0:
                            avg_val = np.nanmean(matrix) if matrix.size > 0 else np.nan
                        else:
                            avg_val = np.nanmean(upper_tri_values)

                        feature_avg.append(avg_val)

                    df[feature] = feature_avg

            elif isinstance(first_element, (int, float, np.number)):
                pass

        return df

    def get_filter_logrms_range(self, df: pd.DataFrame = None, z_range=3, **kwargs):
        """Filter windows based on log(rms).

        Args:
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            z_range (float, optional): The z-score range to filter by. Values outside this range will be set to NaN.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        result = df.copy() if df is not None else self.result.copy()
        z_range = abs(z_range)
        np_rms = np.array(result["rms"].tolist())
        np_logrms = np.log(np_rms)
        del np_rms
        np_logrmsz = zscore(np_logrms, axis=0, nan_policy="omit")
        np_logrms[(np_logrmsz > z_range) | (np_logrmsz < -z_range)] = np.nan

        out = np.full(np_logrms.shape, True)
        out[(np_logrmsz > z_range) | (np_logrmsz < -z_range)] = False
        return out

    def get_filter_high_rms(self, df: pd.DataFrame = None, max_rms=500, **kwargs):
        """Filter windows based on rms.

        Args:
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            max_rms (float, optional): The maximum rms value to filter by. Values above this will be set to NaN.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        result = df.copy() if df is not None else self.result.copy()
        np_rms = np.array(result["rms"].tolist())
        np_rmsnan = np_rms.copy()
        # Convert to float to allow NaN assignment for integer arrays
        if np_rmsnan.dtype.kind in ("i", "u"):  # integer types
            np_rmsnan = np_rmsnan.astype(float)
        np_rmsnan[np_rms > max_rms] = np.nan
        result["rms"] = np_rmsnan.tolist()

        out = np.full(np_rms.shape, True)
        out[np_rms > max_rms] = False
        return out

    def get_filter_low_rms(self, df: pd.DataFrame = None, min_rms=30, **kwargs):
        """Filter windows based on rms.

        Args:
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            min_rms (float, optional): The minimum rms value to filter by. Values below this will be set to NaN.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        result = df.copy() if df is not None else self.result.copy()
        np_rms = np.array(result["rms"].tolist())
        np_rmsnan = np_rms.copy()
        np_rmsnan[np_rms < min_rms] = np.nan
        result["rms"] = np_rmsnan.tolist()

        out = np.full(np_rms.shape, True)
        out[np_rms < min_rms] = False
        return out

    def get_filter_high_beta(
        self, df: pd.DataFrame = None, max_beta_prop=0.4, **kwargs
    ):
        """Filter windows based on beta power.

        Args:
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            max_beta_prop (float, optional): The maximum beta power to filter by. Values above this will be set to NaN. Defaults to 0.4.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        result = df.copy() if df is not None else self.result.copy()
        if "psdfrac" in result.columns:
            df_psdfrac = pd.DataFrame(result["psdfrac"].tolist())
            np_prop = np.array(df_psdfrac["beta"].tolist())
        elif "psdband" in result.columns and "psdtotal" in result.columns:
            df_psdband = pd.DataFrame(result["psdband"].tolist())
            np_beta = np.array(df_psdband["beta"].tolist())
            np_total = np.array(result["psdtotal"].tolist())
            np_prop = np_beta / np_total
        else:
            raise ValueError(
                "psdfrac or psdband+psdtotal required for beta power filtering"
            )

        out = np.full(np_prop.shape, True)
        out[np_prop > max_beta_prop] = False
        out = np.broadcast_to(np.all(out, axis=-1)[:, np.newaxis], out.shape)
        return out

    def get_filter_reject_channels(
        self,
        df: pd.DataFrame = None,
        bad_channels: list[str] = None,
        use_abbrevs: bool = None,
        save_bad_channels: Literal["overwrite", "union", None] = "union",
        **kwargs,
    ):
        """Filter channels to reject.

        Args:
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            bad_channels (list[str]): List of channels to reject. Can be either full channel names or abbreviations.
                The method will automatically detect which format is being used. If None, no filtering is performed.
            use_abbrevs (bool, optional): Override automatic detection. If True, channels are assumed to be channel abbreviations. If False, channels are assumed to be channel names.
                If None, channels are parsed to abbreviations and matched against self.channel_abbrevs.
            save_bad_channels (Literal["overwrite", "union", None], optional): How to save bad channels to self.bad_channels_dict.
                "overwrite": Replace self.bad_channels_dict completely with bad channels applied to all sessions.
                "union": Merge bad channels with existing self.bad_channels_dict for all sessions.
                None: Don't save to self.bad_channels_dict. Defaults to "union".
                Note: When using "overwrite" mode, the bad_channels parameter and bad_channels_dict parameter
                may conflict and overwrite each other's bad channel definitions if both are provided.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        n_samples = len(self.result)
        n_channels = len(self.channel_names)
        mask = np.ones((n_samples, n_channels), dtype=bool)

        if bad_channels is None:
            return mask

        channel_targets = (
            self.channel_abbrevs
            if use_abbrevs or use_abbrevs is None
            else self.channel_names
        )  # Match to appropriate target
        if use_abbrevs is None:  # Match channels as abbreviations
            bad_channels = [
                core.parse_chname_to_abbrev(
                    ch, assume_from_number=self.assume_from_number
                )
                for ch in bad_channels
            ]

        # Match channels to channel_targets
        for ch in bad_channels:
            if ch in channel_targets:
                mask[:, channel_targets.index(ch)] = False
            else:
                warnings.warn(f"Channel {ch} not found in {channel_targets}")

        # Save bad channels to self.bad_channels_dict if requested
        if save_bad_channels is not None:
            # Get all unique animal days from the result
            animaldays = self.result["animalday"].unique()

            # Convert bad channels to the format used in bad_channels_dict (original channel names)
            channels_to_save = (
                bad_channels.copy()
                if use_abbrevs is False
                else [
                    core.parse_chname_to_abbrev(
                        ch, assume_from_number=self.assume_from_number
                    )
                    for ch in bad_channels
                ]
            )

            if save_bad_channels == "overwrite":
                # Replace entire dict with bad channels applied to all sessions
                self.bad_channels_dict = {
                    animalday: channels_to_save.copy() for animalday in animaldays
                }
            elif save_bad_channels == "union":
                # Merge with existing bad channels for all sessions
                updated_dict = self.bad_channels_dict.copy()
                for animalday in animaldays:
                    if animalday in updated_dict:
                        # Union of existing and new channels (sorted for deterministic order)
                        updated_dict[animalday] = sorted(
                            set(updated_dict[animalday]) | set(channels_to_save)
                        )
                    else:
                        updated_dict[animalday] = channels_to_save.copy()
                self.bad_channels_dict = updated_dict

        return mask

    def get_filter_reject_channels_by_recording_session(
        self,
        df: pd.DataFrame = None,
        bad_channels_dict: dict[str, list[str]] = None,
        use_abbrevs: bool = None,
        save_bad_channels: Literal["overwrite", "union", None] = "union",
        **kwargs,
    ):
        """Filter channels to reject for each recording session

        Args:
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            bad_channels_dict (dict[str, list[str]]): Dictionary of list of channels to reject for each recording session.
                Can be either full channel names or abbreviations. The method will automatically detect which format is being used.
                If None, the method will use the bad_channels_dict passed to the constructor.
            use_abbrevs (bool, optional): Override automatic detection. If True, channels are assumed to be channel abbreviations. If False, channels are assumed to be channel names.
                If None, channels are parsed to abbreviations and matched against self.channel_abbrevs.
            save_bad_channels (Literal["overwrite", "union", None], optional): How to save bad channels to self.bad_channels_dict.
                "overwrite": Replace self.bad_channels_dict completely with bad_channels_dict.
                "union": Merge bad_channels_dict with existing self.bad_channels_dict per session.
                None: Don't save to self.bad_channels_dict. Defaults to "union".
                Note: When using "overwrite" mode, the bad_channels parameter and bad_channels_dict parameter
                may conflict and overwrite each other's bad channel definitions if both are provided.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        if bad_channels_dict is None:
            bad_channels_dict = self.bad_channels_dict.copy()

        n_samples = len(self.result)
        n_channels = len(self.channel_names)
        mask = np.ones((n_samples, n_channels), dtype=bool)

        # Group by animalday to apply filters per recording session
        for animalday, group in self.result.groupby("animalday"):
            if bad_channels_dict:
                if animalday not in bad_channels_dict:
                    raise ValueError(
                        f"No bad channels specified for recording session {animalday}. Check that all days are present in bad_channels_dict"
                    )
                bad_channels = bad_channels_dict[animalday]
            else:
                bad_channels = []

            channel_targets = (
                self.channel_abbrevs
                if use_abbrevs or use_abbrevs is None
                else self.channel_names
            )
            if use_abbrevs is None:
                bad_channels = [
                    core.parse_chname_to_abbrev(
                        ch, assume_from_number=self.assume_from_number
                    )
                    for ch in bad_channels
                ]

            # Get indices for this recording session
            session_indices = group.index

            # Apply channel filtering for this session
            for ch in bad_channels:
                if ch in channel_targets:
                    ch_idx = channel_targets.index(ch)
                    mask[session_indices, ch_idx] = False
                else:
                    logging.warning(
                        f"Channel {ch} not found in {channel_targets} for session {animalday}"
                    )

        # Save bad channels to self.bad_channels_dict if requested
        if save_bad_channels is not None and bad_channels_dict is not None:
            if save_bad_channels == "overwrite":
                self.bad_channels_dict = bad_channels_dict.copy()
            elif save_bad_channels == "union":
                # Merge with existing bad channels per session
                updated_dict = self.bad_channels_dict.copy()
                for animalday, channels in bad_channels_dict.items():
                    if animalday in updated_dict:
                        # Union of existing and new channels (sorted for deterministic order)
                        updated_dict[animalday] = sorted(
                            set(updated_dict[animalday]) | set(channels)
                        )
                    else:
                        updated_dict[animalday] = channels.copy()
                self.bad_channels_dict = updated_dict

        return mask

    def get_filter_morphological_smoothing(
        self, filter_mask: np.ndarray, smoothing_seconds: float, **kwargs
    ) -> np.ndarray:
        """Apply morphological smoothing to a filter mask.

        Args:
            filter_mask (np.ndarray): Input boolean mask of shape (n_windows, n_channels)
            smoothing_seconds (float): Time window in seconds for morphological operations

        Returns:
            np.ndarray: Smoothed boolean mask
        """
        if "duration" not in self.result.columns:
            raise ValueError(
                "Cannot calculate window duration - 'duration' column missing"
            )

        window_duration = self.result["duration"].median()
        structure_size = max(1, int(smoothing_seconds / window_duration))

        if structure_size <= 1:
            return filter_mask

        smoothed_mask = filter_mask.copy()
        for ch_idx in range(filter_mask.shape[1]):
            channel_mask = filter_mask[:, ch_idx]
            # Opening removes small isolated artifacts
            channel_mask = binary_opening(
                channel_mask, structure=np.ones(structure_size)
            )
            # Closing fills small gaps in valid data
            channel_mask = binary_closing(
                channel_mask, structure=np.ones(structure_size)
            )
            smoothed_mask[:, ch_idx] = channel_mask

        return smoothed_mask

    def filter_morphological_smoothing(
        self, smoothing_seconds: float
    ) -> "WindowAnalysisResult":
        """Apply morphological smoothing to all data.

        Args:
            smoothing_seconds (float): Time window in seconds for morphological operations

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        # Start with all-True mask and smooth it
        base_mask = np.ones((len(self.result), len(self.channel_names)), dtype=bool)
        smoothed_mask = self.get_filter_morphological_smoothing(
            base_mask, smoothing_seconds
        )
        return self._create_filtered_copy(smoothed_mask)

    def filter_all(
        self,
        df: pd.DataFrame = None,
        inplace=True,
        # bad_channels: list[str] = None,
        min_valid_channels=3,
        filters: list[callable] = None,
        morphological_smoothing_seconds: float = None,
        # save_bad_channels: Literal["overwrite", "union", None] = "union",
        **kwargs,
    ):
        """Apply a list of filters to the data. Filtering should be performed before aggregation.

        Args:
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            inplace (bool, optional): If True, modify the result in place. Defaults to True.
            bad_channels (list[str], optional): List of channels to reject. Defaults to None.
            min_valid_channels (int, optional): Minimum number of valid channels required per window. Defaults to 3.
            filters (list[callable], optional): List of filter functions to apply. Each function should return a boolean mask.
                If None, uses default filters: [get_filter_logrms_range, get_filter_high_rms, get_filter_low_rms, get_filter_high_beta].
                Defaults to None.
            morphological_smoothing_seconds (float, optional): If provided, apply morphological opening/closing to smooth the filter mask.
                This removes isolated false positives/negatives along the time axis for each channel independently.
                The value specifies the time window in seconds for the morphological operations. Defaults to None.
            save_bad_channels (Literal["overwrite", "union", None], optional): How to save bad channels to self.bad_channels_dict.
                This parameter is passed to the filtering functions. Defaults to "union".
                Note: When using "overwrite" mode, the bad_channels parameter and bad_channels_dict parameter
                may conflict and overwrite each other's bad channel definitions if both are provided.
            **kwargs: Additional keyword arguments to pass to filter functions.

        Returns:
            WindowAnalysisResult: Filtered result
        """
        if filters is None:
            # TODO refactor these into standalone functions, which take in a war as the first parameter, then pass
            # filt_bool = filt(self, df, **kwargs) as needed
            filters = [
                self.get_filter_logrms_range,
                self.get_filter_high_rms,
                self.get_filter_low_rms,
                self.get_filter_high_beta,
                self.get_filter_reject_channels_by_recording_session,
                self.get_filter_reject_channels,
            ]

        filt_bools = []
        # Apply each filter function
        for filter_function in filters:
            filt_bool = filter_function(df, **kwargs)
            filt_bools.append(filt_bool)
            logging.info(
                f"{filter_function.__name__}:\tfiltered {filt_bool.size - np.count_nonzero(filt_bool)}/{filt_bool.size}"
            )

        # Apply all filters
        filt_bool_all = np.prod(np.stack(filt_bools, axis=-1), axis=-1).astype(bool)
        logging.debug(
            f"filt_bool_all.shape: {filt_bool_all.shape}"
        )  # (windows, channels)

        # Apply morphological smoothing if requested
        if morphological_smoothing_seconds is not None:
            if "duration" not in self.result.columns:
                raise ValueError(
                    "Cannot calculate window duration - 'duration' column missing from result dataframe"
                )
            window_duration = self.result["duration"].median()

            # Calculate number of windows for the smoothing
            structure_size = max(
                1, int(morphological_smoothing_seconds / window_duration)
            )

            if structure_size > 1:
                logging.info(
                    f"Applying morphological smoothing with {structure_size} windows ({morphological_smoothing_seconds}s / {window_duration}s per window)"
                )
                # Apply channel-wise temporal smoothing (each channel processed independently)
                # This avoids spatial assumptions while smoothing temporal artifacts
                for ch_idx in range(filt_bool_all.shape[1]):
                    channel_mask = filt_bool_all[:, ch_idx]
                    # Opening removes small isolated artifacts
                    channel_mask = binary_opening(
                        channel_mask, structure=np.ones(structure_size)
                    )
                    # Closing fills small gaps in valid data
                    channel_mask = binary_closing(
                        channel_mask, structure=np.ones(structure_size)
                    )
                    filt_bool_all[:, ch_idx] = channel_mask
            else:
                logging.info(
                    "Skipping morphological smoothing - structure size would be 1 (no effect)"
                )

        # Filter windows based on number of valid channels
        valid_channels_per_window = np.sum(filt_bool_all, axis=1)  # axis 1 = channel
        window_mask = (
            valid_channels_per_window >= min_valid_channels
        )  # True if window has enough valid channels
        filt_bool_all = (
            filt_bool_all & window_mask[:, np.newaxis]
        )  # Apply window mask to all channels

        filtered_result = self._apply_filter(filt_bool_all)
        if inplace:
            del self.result
            self.result = filtered_result
        return WindowAnalysisResult._from_existing(self, filtered_result)

    def _create_filtered_copy(
        self, filter_mask: np.ndarray, filter_name: str = None
    ) -> "WindowAnalysisResult":
        """Create a new WindowAnalysisResult with the filter applied.

        Args:
            filter_mask (np.ndarray): Boolean mask of shape (n_windows, n_channels)
            filter_name (str, optional): Name of the filter for logging. Defaults to None.

        Returns:
            WindowAnalysisResult: New instance with filter applied
        """
        if filter_name is not None:
            logging.info(
                f"{filter_name}: filtered {filter_mask.size - np.count_nonzero(filter_mask)}/{filter_mask.size}"
            )
        filtered_result = self._apply_filter(filter_mask)
        return WindowAnalysisResult._from_existing(self, filtered_result)

    def filter_logrms_range(self, z_range: float = 3) -> "WindowAnalysisResult":
        """Filter based on log(rms) z-score range.

        Args:
            z_range (float): Z-score range threshold. Defaults to 3.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_logrms_range(z_range=z_range)
        return self._create_filtered_copy(mask, filter_name="logrms_range")

    def filter_high_rms(self, max_rms: float = 500) -> "WindowAnalysisResult":
        """Filter out windows with RMS above threshold.

        Args:
            max_rms (float): Maximum RMS threshold. Defaults to 500.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_high_rms(max_rms=max_rms)
        return self._create_filtered_copy(mask, filter_name="high_rms")

    def filter_low_rms(self, min_rms: float = 50) -> "WindowAnalysisResult":
        """Filter out windows with RMS below threshold.

        Args:
            min_rms (float): Minimum RMS threshold. Defaults to 50.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_low_rms(min_rms=min_rms)
        return self._create_filtered_copy(mask, filter_name="low_rms")

    def filter_high_beta(self, max_beta_prop: float = 0.4) -> "WindowAnalysisResult":
        """Filter out windows with high beta power.

        Args:
            max_beta_prop (float): Maximum beta power proportion. Defaults to 0.4.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_high_beta(max_beta_prop=max_beta_prop)
        return self._create_filtered_copy(mask, filter_name="high_beta")

    def filter_reject_channels(
        self, bad_channels: list[str], use_abbrevs: bool = None
    ) -> "WindowAnalysisResult":
        """Filter out specified bad channels.

        Args:
            bad_channels (list[str]): List of channel names to reject
            use_abbrevs (bool, optional): Whether to use abbreviations. Defaults to None.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_reject_channels(
            bad_channels=bad_channels, use_abbrevs=use_abbrevs
        )
        return self._create_filtered_copy(mask, filter_name="reject_channels")

    def filter_reject_channels_by_session(
        self, bad_channels_dict: dict[str, list[str]] = None, use_abbrevs: bool = None
    ) -> "WindowAnalysisResult":
        """Filter out bad channels by recording session.

        Args:
            bad_channels_dict (dict[str, list[str]], optional): Dictionary mapping recording session
                identifiers to lists of bad channel names to reject. Session identifiers are in the
                format "{animal_id} {genotype} {day}" (e.g., "A10 WT Apr-01-2023"). Channel names
                can be either full names (e.g., "Left Auditory") or abbreviations (e.g., "LAud").
                If None, uses the bad_channels_dict from the constructor. Defaults to None.
            use_abbrevs (bool, optional): Override automatic channel name format detection. If True,
                channels are assumed to be abbreviations. If False, channels are assumed to be full
                names. If None, automatically detects format and converts to abbreviations for matching.
                Defaults to None.

        Returns:
            WindowAnalysisResult: New filtered instance with bad channels masked as NaN for their
                respective recording sessions

        Examples:
            Filter specific channels per session using abbreviations:
            >>> bad_channels = {
            ...     "A10 WT Apr-01-2023": ["LAud", "RMot"],  # Session 1: reject left auditory, right motor
            ...     "A10 WT Apr-02-2023": ["LVis"]           # Session 2: reject left visual only
            ... }
            >>> filtered_war = war.filter_reject_channels_by_session(bad_channels, use_abbrevs=True)

            Filter using full channel names:
            >>> bad_channels = {
            ...     "A12 KO May-15-2023": ["Left Motor", "Right Barrel"],
            ...     "A12 KO May-16-2023": ["Left Auditory", "Left Visual", "Right Motor"]
            ... }
            >>> filtered_war = war.filter_reject_channels_by_session(bad_channels, use_abbrevs=False)

            Auto-detect channel format (recommended):
            >>> bad_channels = {
            ...     "A15 WT Jun-10-2023": ["LMot", "RBar"],  # Will auto-detect as abbreviations
            ...     "A15 WT Jun-11-2023": ["LAud"]
            ... }
            >>> filtered_war = war.filter_reject_channels_by_session(bad_channels)

        Note:
            - Session identifiers must exactly match the "animalday" values in the result DataFrame
            - Available channel abbreviations: LAud, RAud, LVis, RVis, LHip, RHip, LBar, RBar, LMot, RMot
            - Channel names are case-insensitive and support various formats (e.g., "left aud", "Left Auditory")
            - If a session identifier is not found in bad_channels_dict, a warning is logged but processing continues
            - If a channel name is not recognized, a warning is logged but other channels are still processed
        """
        mask = self.get_filter_reject_channels_by_recording_session(
            bad_channels_dict=bad_channels_dict, use_abbrevs=use_abbrevs
        )
        return self._create_filtered_copy(mask, filter_name="reject_channels_by_session")

    def apply_filters(
        self,
        filter_config: dict = None,
        min_valid_channels: int = 3,
        morphological_smoothing_seconds: float = None,
    ) -> "WindowAnalysisResult":
        """Apply multiple filters using configuration.

        Args:
            filter_config (dict, optional): Dictionary of filter names and parameters.
                Available filters: 'logrms_range', 'high_rms', 'low_rms', 'high_beta',
                'reject_channels', 'reject_channels_by_session', 'morphological_smoothing'
            min_valid_channels (int): Minimum valid channels per window. Defaults to 3.
            morphological_smoothing_seconds (float, optional): Temporal smoothing window (deprecated, use config instead)

        Returns:
            WindowAnalysisResult: New filtered instance

        Examples:
            >>> config = {
            ...     'logrms_range': {'z_range': 3},
            ...     'high_rms': {'max_rms': 500},
            ...     'reject_channels': {'bad_channels': ['LMot', 'RMot']},
            ...     'morphological_smoothing': {'smoothing_seconds': 8.0}
            ... }
            >>> filtered_war = war.apply_filters(config)
        """
        if filter_config is None:
            filter_config = {
                "logrms_range": {"z_range": 3},
                "high_rms": {"max_rms": 500},
                "low_rms": {"min_rms": 50},
                "high_beta": {"max_beta_prop": 0.4},
                "reject_channels_by_session": {},
            }

        filter_methods = {
            "logrms_range": self.get_filter_logrms_range,
            "high_rms": self.get_filter_high_rms,
            "low_rms": self.get_filter_low_rms,
            "high_beta": self.get_filter_high_beta,
            "reject_channels": self.get_filter_reject_channels,
            "reject_channels_by_session": self.get_filter_reject_channels_by_recording_session,
        }

        filt_bools = []
        morphological_params = None

        for filter_name, filter_params in filter_config.items():
            if filter_name == "morphological_smoothing":
                morphological_params = filter_params
                continue

            if filter_name not in filter_methods:
                raise ValueError(
                    f"Unknown filter: {filter_name}. Available: {list(filter_methods.keys()) + ['morphological_smoothing']}"
                )

            filter_func = filter_methods[filter_name]
            filt_bool = filter_func(**filter_params)
            filt_bools.append(filt_bool)
            logging.info(
                f"{filter_name}: filtered {filt_bool.size - np.count_nonzero(filt_bool)}/{filt_bool.size}"
            )

        # Combine all filter masks
        if filt_bools:
            filt_bool_all = np.prod(np.stack(filt_bools, axis=-1), axis=-1).astype(bool)
        else:
            filt_bool_all = np.ones(
                (len(self.result), len(self.channel_names)), dtype=bool
            )

        # Apply morphological smoothing if requested (either from config or parameter)
        if morphological_params or morphological_smoothing_seconds is not None:
            if morphological_params:
                smoothing_seconds = morphological_params["smoothing_seconds"]
            else:
                smoothing_seconds = morphological_smoothing_seconds

            filt_bool_all = self.get_filter_morphological_smoothing(
                filt_bool_all, smoothing_seconds
            )
            logging.info(f"Applied morphological smoothing: {smoothing_seconds}s")

        # Filter windows based on minimum valid channels
        valid_channels_per_window = np.sum(filt_bool_all, axis=1)
        window_mask = valid_channels_per_window >= min_valid_channels
        filt_bool_all = filt_bool_all & window_mask[:, np.newaxis]

        return self._create_filtered_copy(filt_bool_all)

    def _apply_filter(self, filter_tfs: np.ndarray):
        result = self.result.copy()
        filter_tfs = np.array(filter_tfs, dtype=bool)  # (M fragments, N channels)
        for feat in constants.FEATURES:
            if feat not in result.columns:
                logging.debug(f"Skipping {feat} because it is not in result")
                continue
            logging.debug(f"Filtering {feat}")
            ftype = constants.classify_feature(feat)

            if ftype is constants.FeatureType.LINEAR:
                vals = extract_linear_array(result[feat]).astype(float, copy=False)
                vals[~filter_tfs] = np.nan
                result[feat] = vals.tolist()

            elif ftype is constants.FeatureType.LINEAR_2D:
                vals = extract_linear_array(result[feat]).astype(float, copy=False)
                mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], vals.shape)
                vals[~mask] = np.nan
                result[feat] = vals.tolist()

            elif ftype is constants.FeatureType.HIST:
                # FIXME The sampling rates have changed between computation passes so WARs have different shapes.
                # Add a check for same sampling frequency, other war-relevant properties etc.
                # The logging lines below should be removed at some point, but I'll keep it this way for now
                logging.debug(
                    f"set([np.asarray(x[0]).shape for x in result[feat].tolist()]) = {list(set([np.asarray(x[0]).shape for x in result[feat].tolist()]))}"
                )
                logging.debug(
                    f"set([np.asarray(x[1]).shape for x in result[feat].tolist()]) = {list(set([np.asarray(x[1]).shape for x in result[feat].tolist()]))}"
                )
                coords, vals = extract_hist_data(result[feat])
                vals = vals.astype(float, copy=False)
                # vals is canonical (W, C, F); filter_tfs is (W, C)
                mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], vals.shape)
                vals[~mask] = np.nan
                # Repack as (F, C) per cell to preserve per-cell storage format
                outs = [(c, vals[i].T) for i, c in enumerate(coords)]
                result[feat] = outs

            elif ftype is constants.FeatureType.BAND:
                band_vals, band_keys = extract_band_dict(result[feat])
                band_vals = band_vals.astype(float, copy=False)
                # band_vals is canonical (W, C, B); index band on last axis
                for bi, colname in enumerate(band_keys):
                    v = band_vals[:, :, bi]  # (W, C)
                    v[~filter_tfs] = np.nan
                    band_vals[:, :, bi] = v
                result[feat] = repack_band_dict(band_vals, band_keys)

            elif ftype is constants.FeatureType.BANDED_MATRIX:
                band_vals, band_keys = extract_band_dict(result[feat])
                band_vals = band_vals.astype(float, copy=False)
                # band_vals is canonical (W, C, C, B); index band on last axis
                shape = band_vals[:, :, :, 0].shape  # (W, C, C)
                mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], shape)
                for bi, colname in enumerate(band_keys):
                    v = band_vals[:, :, :, bi]  # (W, C, C)
                    v[~mask] = np.nan
                    v[~mask.transpose(0, 2, 1)] = np.nan
                    band_vals[:, :, :, bi] = v
                result[feat] = repack_band_dict(band_vals, band_keys)

            elif ftype is constants.FeatureType.SIMPLE_MATRIX:
                vals = extract_linear_array(result[feat]).astype(float, copy=False)
                mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], vals.shape)
                vals[~mask] = np.nan
                vals[~mask.transpose(0, 2, 1)] = np.nan
                result[feat] = vals.tolist()

            else:
                raise ValueError(
                    f"Unsupported FeatureType {ftype} for filtering: {feat}"
                )
        return result

    def save_pickle_and_json(
        self,
        folder: str | Path,
        make_folder=True,
        filename: str = None,
        slugify_filename=False,
        save_abbrevs_as_chnames=False,
    ):
        """Archive window analysis result into the folder specified, as a parquet and json file.

        The result DataFrame is saved as a Parquet file (stable across pandas
        versions).  A pickle copy is also written for backward compatibility
        with older workflows.

        Args:
            folder (str | Path): Destination folder to save results to
            make_folder (bool, optional): If True, create the folder if it doesn't exist. Defaults to True.
            filename (str, optional): Name of the file to save. Defaults to "war".
            slugify_filename (bool, optional): If True, slugify the filename (replace special characters). Defaults to False.
            save_abbrevs_as_chnames (bool, optional): If True, save the channel abbreviations as the channel names in the json file. Defaults to False.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        folder = Path(folder)
        if make_folder:
            folder.mkdir(parents=True, exist_ok=True)

        filename = "war" if filename is None else filename
        filename = slugify(filename) if slugify_filename else filename

        filepath = str(folder / filename)

        # Write pickle for backward compatibility with existing workflows
        self.result.to_pickle(filepath + ".pkl")
        logging.info(f"Saved WAR to {filepath + '.pkl'}")

        # Write parquet as the primary stable format.
        # Object-like columns (lists/dicts/ndarrays) are JSON-encoded per-cell.
        # The list of encoded columns is stored in the parquet schema metadata
        # so they can be decoded on load.
        pq_df, encoded_cols = self._encode_df_for_parquet(self.result)
        table = pa.Table.from_pandas(pq_df)
        neurodent_meta = json.dumps({"encoded_columns": encoded_cols}).encode()
        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, b"neurodent": neurodent_meta}
        table = table.replace_schema_metadata(merged_meta)
        pq.write_table(table, filepath + ".parquet")
        logging.info(f"Saved WAR to {filepath + '.parquet'}")

        json_dict = {
            "animal_id": self.animal_id,
            "genotype": self.genotype,
            "sex": self.sex,
            "channel_names": (
                self.channel_abbrevs if save_abbrevs_as_chnames else self.channel_names
            ),
            "assume_from_number": (
                False if save_abbrevs_as_chnames else self.assume_from_number
            ),
            "bad_channels_dict": self.bad_channels_dict,
            "suppress_short_interval_error": self.suppress_short_interval_error,
            "lof_scores_dict": self.lof_scores_dict.copy(),
        }

        with open(filepath + ".json", "w") as f:
            json.dump(json_dict, f, indent=2)
            logging.info(f"Saved WAR to {filepath + '.json'}")

    class _NumpyEncoder(json.JSONEncoder):
        """JSON encoder that handles numpy types transparently.

        The stdlib encoder already recurses into lists and dicts, so we only
        need to override *default* for types it cannot handle natively.
        """

        def default(self, o: Any) -> Any:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    @staticmethod
    def _encode_df_for_parquet(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Return a copy of *df* where complex/object columns have been
        JSON-encoded as strings so they can be written to Parquet safely.

        Returns:
            (encoded_df, encoded_columns) — the modified DataFrame and the
            list of column names that were encoded.
        """
        df_copy = df.copy()
        encoded_cols: list[str] = []
        for col in df_copy.columns:
            ser = df_copy[col]
            needs_encoding = False
            if ser.dtype == object:
                sample = ser.dropna().head(20)
                for v in sample:
                    if not isinstance(v, (str, int, float, bool, type(None))):
                        needs_encoding = True
                        break

            if needs_encoding:
                encoded_cols.append(col)
                df_copy[col] = ser.apply(
                    lambda x: json.dumps(x, cls=WindowAnalysisResult._NumpyEncoder, ensure_ascii=False)
                )

        return df_copy, encoded_cols

    @staticmethod
    def _decode_df_from_parquet(df: pd.DataFrame, encoded_cols: list[str]) -> pd.DataFrame:
        """Decode JSON-encoded columns back into Python objects.

        Values are returned as plain Python types (lists, dicts, scalars) —
        the same representation that ``json.loads`` produces.  Consuming code
        (e.g. ``_apply_filter``) already wraps values with ``np.array()`` /
        ``np.asarray()`` where needed, so no eager numpy conversion is done
        here.  This avoids over-converting list-based features and keeps the
        per-cell cost to a single ``json.loads`` call.
        """
        df_copy = df.copy()
        for col in encoded_cols:
            if col not in df_copy.columns:
                continue
            # Some parquet engines may already return Python objects for nulls;
            # only attempt json.loads on actual string values.
            def _try_load(v):
                if isinstance(v, str):
                    try:
                        return json.loads(v)
                    except json.JSONDecodeError:
                        return v
                return v

            df_copy[col] = df_copy[col].apply(_try_load)

        return df_copy

    def get_bad_channels_by_lof_threshold(self, lof_threshold: float) -> dict:
        """Apply LOF threshold directly to stored scores to get bad channels.

        Args:
            lof_threshold (float): Threshold for determining bad channels.

        Returns:
            dict: Dictionary mapping animal days to lists of bad channel names.
        """
        if not hasattr(self, "lof_scores_dict") or not self.lof_scores_dict:
            raise ValueError(
                "LOF scores not available in this WAR. Compute LOF scores first."
            )

        bad_channels_dict = {}
        for animalday, lof_data in self.lof_scores_dict.items():
            if "lof_scores" in lof_data and "channel_names" in lof_data:
                scores = np.array(lof_data["lof_scores"])
                channel_names = lof_data["channel_names"]

                is_inlier = scores < lof_threshold
                bad_channels = [channel_names[i] for i in np.where(~is_inlier)[0]]
                bad_channels_dict[animalday] = bad_channels
            else:
                raise ValueError(f"LOF scores not available for {animalday}")

        return bad_channels_dict

    def get_lof_scores(self) -> dict:
        """Get LOF scores from this WAR.

        Returns:
            dict: Dictionary mapping animal days to LOF score dictionaries.
        """
        if not hasattr(self, "lof_scores_dict") or not self.lof_scores_dict:
            raise ValueError(
                "LOF scores not available in this WAR. Compute LOF scores first."
            )

        result = {}
        for animalday, lof_data in self.lof_scores_dict.items():
            if "lof_scores" in lof_data and "channel_names" in lof_data:
                scores = lof_data["lof_scores"]
                channel_names = lof_data["channel_names"]
                result[animalday] = dict(zip(channel_names, scores))
            else:
                raise ValueError(f"LOF scores not available for {animalday}")

        return result

    def evaluate_lof_threshold_binary(
        self,
        ground_truth_bad_channels: dict = None,
        threshold: float = None,
        evaluation_channels: list[str] = None,
    ) -> tuple:
        """Evaluate single threshold against ground truth for binary classification.

        Args:
            ground_truth_bad_channels: Dict mapping animal-day to bad channel sets.
                                     If None, uses self.bad_channels_dict as ground truth.
            threshold: LOF threshold to test
            evaluation_channels: Subset of channels to include in evaluation. If none, uses all channels.

        Returns:
            tuple: (y_true_list, y_pred_list) for sklearn.metrics.f1_score
                   Each element represents one channel from one animal-day
        """
        if not hasattr(self, "lof_scores_dict") or not self.lof_scores_dict:
            raise ValueError(
                "LOF scores not available in this WAR. Run compute_bad_channels() first."
            )

        if threshold is None:
            raise ValueError("threshold parameter is required")

        # Use self.bad_channels_dict as default ground truth
        if ground_truth_bad_channels is None:
            if hasattr(self, "bad_channels_dict") and self.bad_channels_dict:
                ground_truth_bad_channels = {}

                # Filter bad_channels_dict to only include keys that exist in lof_scores_dict
                lof_keys = set(self.lof_scores_dict.keys())
                bad_channels_keys = set(self.bad_channels_dict.keys())

                missing_keys = bad_channels_keys - lof_keys
                if missing_keys:
                    raise ValueError(
                        f"bad_channels_dict contains keys not found in lof_scores_dict: {missing_keys}. "
                        f"Available LOF keys: {sorted(lof_keys)}"
                    )

                # Only use bad channel keys that have corresponding LOF data
                ground_truth_bad_channels = {
                    key: value
                    for key, value in self.bad_channels_dict.items()
                    if key in lof_keys
                }

                logging.info(
                    f"Using filtered bad_channels_dict as ground truth with {len(ground_truth_bad_channels)} animal-day sessions"
                )
            else:
                raise ValueError(
                    "No ground truth provided and self.bad_channels_dict is empty."
                )

        # Get all channels if no subset specified
        if evaluation_channels is None:
            evaluation_channels = self.channel_names

        y_true_list = []
        y_pred_list = []

        # Debug: Log what we're working with
        logging.debug(
            f"evaluate_lof_threshold_binary: evaluation_channels = {evaluation_channels}"
        )
        logging.debug(
            f"evaluate_lof_threshold_binary: ground_truth_bad_channels keys = {list(ground_truth_bad_channels.keys())}"
        )
        logging.debug(
            f"evaluate_lof_threshold_binary: lof_scores_dict keys = {list(self.lof_scores_dict.keys())}"
        )

        # Iterate through each animal-day and evaluate channels
        for animalday, lof_data in self.lof_scores_dict.items():
            if "lof_scores" not in lof_data or "channel_names" not in lof_data:
                raise ValueError(
                    f"Invalid LOF data for {animalday}: missing required fields 'lof_scores' or 'channel_names'"
                )

            scores = np.array(lof_data["lof_scores"])
            channel_names = lof_data["channel_names"]

            # Validate data integrity before processing
            # NOTE address this issue since this should not be happening in the first place
            # if len(scores) == 0:
            #     logging.warning(
            #         f"Skipping {animalday}: No LOF scores available. "
            #         f"This session will be excluded from LOF accuracy evaluation."
            #     )
            #     continue

            # if len(scores) != len(channel_names):
            #     logging.error(
            #         f"Skipping {animalday}: LOF scores ({len(scores)}) and "
            #         f"channels ({len(channel_names)}) length mismatch. "
            #         f"This indicates a data integrity issue - the animalday may have been "
            #         f"improperly mapped during LOF score collection."
            #     )
            #     continue

            # Get ground truth bad channels for this animal-day
            animalday_bad_channels = ground_truth_bad_channels.get(animalday, set())

            # Debug: Log details for this animal-day
            logging.debug(f"Processing {animalday}: channel_names = {channel_names}")
            logging.debug(
                f"Processing {animalday}: animalday_bad_channels = {animalday_bad_channels}"
            )
            logging.debug(f"Processing {animalday}: scores shape = {scores.shape}")

            # Evaluate each channel in the evaluation subset
            channels_processed = 0
            for i, channel in enumerate(channel_names):
                if (
                    channel in evaluation_channels
                    or parse_chname_to_abbrev(channel, strict_matching=False)
                    in evaluation_channels
                ):
                    channels_processed += 1

                    # Ground truth: 1 if channel is marked as bad, 0 otherwise
                    is_bad_channel = (
                        channel in animalday_bad_channels
                        or parse_chname_to_abbrev(channel, strict_matching=False)
                        in animalday_bad_channels
                    )
                    # if is_bad_channel and channel not in animalday_bad_channels:
                    #     logging.debug(f"Mapped full channel '{channel}' -> '{parse_chname_to_abbrev(channel, strict_matching=False)}' found in bad channels")

                    y_true = 1 if is_bad_channel else 0
                    # Prediction: 1 if LOF score > threshold, 0 otherwise
                    y_pred = 1 if scores[i] > threshold else 0

                    y_true_list.append(y_true)
                    y_pred_list.append(y_pred)

                    logging.debug(
                        f"Channel {channel}: y_true={y_true}, y_pred={y_pred} (score={scores[i]:.3f}, threshold={threshold})"
                    )

                    # Extra debugging for the alignment issue
                    if y_true == 1:
                        logging.info(
                            f"TRUE POSITIVE CANDIDATE: {channel} mapped to bad channel in: {animalday_bad_channels}"
                        )
                    if y_pred == 1:
                        logging.info(
                            f"LOF PREDICTION: {channel} has score {scores[i]:.3f} > threshold {threshold}"
                        )

            logging.debug(f"Processed {channels_processed} channels for {animalday}")

        return y_true_list, y_pred_list

    @classmethod
    def load_pickle_and_json(cls, folder_path=None, pickle_name=None, json_name=None):
        """Load WindowAnalysisResult from folder

        Args:
            folder_path (str, optional): Path of folder containing .pkl and .json files. Defaults to None.
            pickle_name (str, optional): Name of the pickle file. Can be just the filename (e.g. "war.pkl")
                or a path relative to folder_path (e.g. "subdir/war.pkl"). If None and folder_path is provided,
                expects exactly one .pkl file in folder_path. Defaults to None.
            json_name (str, optional): Name of the JSON file. Can be just the filename (e.g. "war.json")
                or a path relative to folder_path (e.g. "subdir/war.json"). If None and folder_path is provided,
                expects exactly one .json file in folder_path. Defaults to None.

        Raises:
            ValueError: folder_path does not exist
            ValueError: Expected exactly one pickle and one json file in folder_path (when pickle_name/json_name not specified)
            FileNotFoundError: Specified pickle_name or json_name not found

        Returns:
            result: WindowAnalysisResult object
        """
        if folder_path is not None:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise ValueError(f"Folder path {folder_path} does not exist")

            if pickle_name is not None:
                # Handle pickle_name as either absolute path or relative to folder_path
                pickle_path = Path(pickle_name)
                if pickle_path.is_absolute():
                    df_pickle_path = pickle_path
                else:
                    df_pickle_path = folder_path / pickle_name

                if not df_pickle_path.exists():
                    raise FileNotFoundError(f"Pickle file not found: {df_pickle_path}")
            else:
                pkl_files = list(folder_path.glob("*.pkl"))
                if len(pkl_files) != 1:
                    raise ValueError(
                        f"Expected exactly one pickle file in {folder_path}, found {len(pkl_files)}"
                    )
                df_pickle_path = pkl_files[0]

            if json_name is not None:
                # Handle json_name as either absolute path or relative to folder_path
                json_path = Path(json_name)
                if json_path.is_absolute():
                    json_path = json_path
                else:
                    json_path = folder_path / json_name

                if not json_path.exists():
                    raise FileNotFoundError(f"JSON file not found: {json_path}")
            else:
                json_files = list(folder_path.glob("*.json"))
                if len(json_files) != 1:
                    raise ValueError(
                        f"Expected exactly one json file in {folder_path}, found {len(json_files)}"
                    )
                json_path = json_files[0]
        else:
            if pickle_name is None or json_name is None:
                raise ValueError(
                    "Either folder_path must be provided, or both pickle_name and json_name must be provided as absolute paths"
                )

            df_pickle_path = Path(pickle_name)
            json_path = Path(json_name)

            if not df_pickle_path.exists():
                raise FileNotFoundError(f"Pickle file not found: {df_pickle_path}")
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Prefer Parquet if available (parquet is more stable across pandas versions)
        parquet_path = df_pickle_path.with_suffix(".parquet")
        data: pd.DataFrame
        if parquet_path.exists():
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(parquet_path)
                # Encoded-column list is stored in schema metadata
                encoded_cols: list[str] = []
                schema_meta = table.schema.metadata or {}
                if b"neurodent" in schema_meta:
                    nd_meta = json.loads(schema_meta[b"neurodent"])
                    encoded_cols = nd_meta.get("encoded_columns", [])
                else:
                    # Fallback: try legacy .parquet.meta.json sidecar file
                    legacy_meta_path = parquet_path.parent / (
                        parquet_path.name + ".meta.json"
                    )
                    if legacy_meta_path.exists():
                        with open(legacy_meta_path, "r") as mf:
                            pq_meta = json.load(mf)
                        encoded_cols = pq_meta.get("encoded_columns", [])

                data = table.to_pandas()
                data = cls._decode_df_from_parquet(data, encoded_cols)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Failed to load parquet WAR ({parquet_path}): {e}, falling back to pickle"
                )
                with open(df_pickle_path, "rb") as f:
                    data = pd.read_pickle(f)
        else:
            with open(df_pickle_path, "rb") as f:
                data = pd.read_pickle(f)
        with open(json_path, "r") as f:
            metadata = json.load(f)
        return cls(data, **metadata)

    def aggregate_time_windows(
        self, groupby: list[str] | str = ["animalday", "isday"]
    ) -> None:
        """Aggregate time windows into a single data point per groupby by averaging features. This reduces the number of rows in the result.

        Args:
            groupby (list[str] | str, optional): Columns to group by. Defaults to ['animalday', 'isday'], which groups by animalday (recording session) and isday (day/night).

        Raises:
            ValueError: groupby must be from ['animalday', 'isday']
            ValueError: Columns in groupby not found in result
            ValueError: Columns in groupby are not constant in groups
        """
        if isinstance(groupby, str):
            groupby = [groupby]
        if not all(col in ["animalday", "isday"] for col in groupby):
            raise ValueError(
                f"groupby must be from ['animalday', 'isday']. Got {groupby}"
            )
        if not all(col in self.result.columns for col in groupby):
            raise ValueError(
                f"Columns {groupby} not found in result. Columns: {self.result.columns.tolist()}"
            )

        features = [f for f in constants.FEATURES if f in self.result.columns]
        logging.debug(f"Aggregating {features}")
        result_grouped = self.result.groupby(groupby)

        agg_dict = {}

        if "animalday" not in groupby:
            agg_dict["animalday"] = lambda df: None
        if "isday" not in groupby:
            agg_dict["isday"] = lambda df: None

        special_agg_cols = {"animalday", "isday", "duration", "endfile", "timestamp"}
        constant_cols = [
            col
            for col in self._nonfeature_columns
            if col not in groupby and col not in special_agg_cols
        ]
        for col in constant_cols:
            if col in self.result.columns:
                is_constant = result_grouped[col].nunique() == 1
                if not is_constant.all():
                    non_constant_groups = is_constant[~is_constant].index.tolist()
                    raise ValueError(
                        f"Column {col} is not constant in groups: {non_constant_groups}"
                    )
                agg_dict[col] = lambda df, col=col: df[col].iloc[0]

        if "duration" in self.result.columns:
            agg_dict["duration"] = lambda df: np.sum(df["duration"])

        if "endfile" in self.result.columns:
            agg_dict["endfile"] = lambda df: df["endfile"].iloc[-1]

        if "timestamp" in self.result.columns:
            agg_dict["timestamp"] = lambda df: df["timestamp"].iloc[0]

        for feat in features:
            agg_dict[feat] = lambda df, feat=feat: self._average_feature(
                df, feat, "duration"
            )

        aggregated_df = result_grouped.apply(
            lambda df: pd.Series(
                {
                    col: agg_dict[col](df)
                    for col in self.result.columns
                    if col not in groupby
                }
            )
        )

        self.result = aggregated_df.reset_index(
            drop=False
        )  # Keep animalday/isday as a column

        self.suppress_short_interval_error = True
        logging.info("Setting suppress_short_interval_error to True")
        self._update_instance_vars()

    def add_unique_hash(self, nbytes: int | None = None):
        """Adds a hex hash to the animal ID to ensure uniqueness. This prevents collisions when, for example, multiple animals in ExperimentPlotter have the same animal ID.

        Args:
            nbytes (int, optional): Number of bytes to generate. This is passed directly to secrets.token_hex(). Defaults to None, which generates 16 hex characters (8 bytes).
        """
        import secrets

        hash_suffix = secrets.token_hex(nbytes)
        new_animal_id = f"{self.animal_id}_{hash_suffix}"

        if "animal" in self.result.columns:
            self.result["animal"] = new_animal_id
        if "animalday" in self.result.columns:
            self.result["animalday"] = self.result["animalday"].str.replace(
                self.animal_id, new_animal_id
            )
        self.animal_id = new_animal_id

        self._update_instance_vars()


# Note: WindowAnalysisResult was moved to window_analysis_result.py; helpers
# like bin_spike_times/_bin_spike_df live there now. This module keeps a
# compatibility import so external code (and pickles) referencing
# neurodent.visualization.results.WindowAnalysisResult continue to work.
