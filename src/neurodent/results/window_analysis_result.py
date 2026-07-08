"""Windowed feature analysis results.

``WindowAnalysisResult`` wraps the windowed feature DataFrame produced by
:meth:`neurodent.analysis.animal_analyzer.AnimalAnalyzer.compute_windowed_analysis`.
Its behavior is composed from mixins in ``results/war_*.py``; this module keeps the
base methods, the class definition, and the module-level feature/spike helpers that
external code imports from this path.
"""

from __future__ import annotations

import copy
import logging
import warnings

import numpy as np
import pandas as pd

from neurodent import constants
from neurodent.core.utils import resolve_channel, slugify
from .feature_utils import average_feature
from .feature_handlers import handler_for


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
    # Convert fragment durations to bin edges
    bin_edges = np.cumsum([0] + fragment_durations)

    # Use numpy's histogram function to count spikes in each bin
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


from .war_features import WARFeatureMixin
from .war_filtering import WARFilterMixin
from .war_lof import WARLofMixin
from .war_serialization import WARSerializationMixin
from .war_spikes import WARSpikeMixin


class WindowAnalysisResult(
    WARFeatureMixin,
    WARFilterMixin,
    WARLofMixin,
    WARSerializationMixin,
    WARSpikeMixin,
):
    """
    Wrapper for output of windowed analysis. Has useful functions like group-wise and global averaging, filtering, and saving

    Args:
        result (pd.DataFrame): Result comes from AnimalAnalyzer.compute_windowed_analysis()
        animal_id (str, optional): Identifier for the animal where result was computed from. Defaults to None.
        genotype (str, optional): Genotype of animal. Defaults to None.
        channel_names (list[str], optional): The recording's channel labels (raw names as
            they appear in the data). Defaults to None.
        bad_channels_dict (dict[str, list[str]], optional): Dictionary of channels to reject for each recording session. Defaults to {}.
        suppress_short_interval_error (bool, optional): If True, suppress ValueError for short intervals between timestamps. Useful for aggregated WARs with large window sizes. Defaults to False.

    Attributes:
        result (pd.DataFrame): DataFrame containing the windowed analysis results.
        animal_id (str): Identifier for the animal.
        genotype (str): Genotype of the animal.
        channel_names (list[str]): The *current working* channel labels — the raw names at
            construction, or the canonical abbreviations after :meth:`reorder_and_pad_channels`
            is run with ``use_abbrevs=True``.
        channel_abbrevs (list[str]): The canonical channel abbreviations, always derived from
            ``channel_names`` via :func:`~neurodent.core.resolve_channel` (exact lookup).
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
        bad_channels_dict: dict[str, list[str]] = {},
        suppress_short_interval_error=False,
        lof_scores_dict: dict[str, dict] = {},
    ) -> None:
        self.result = result
        self.animal_id = animal_id
        self.genotype = genotype
        self.sex = sex
        self.channel_names = channel_names
        self.bad_channels_dict = bad_channels_dict.copy()
        self.suppress_short_interval_error = suppress_short_interval_error
        self.lof_scores_dict = lof_scores_dict

        self._update_instance_vars()

        # Single source of truth: re-enrich sex/genotype from the active config
        # (constants.ANIMAL_METADATA) so every WAR construction — generation, disk
        # load, copy — yields metadata consistent with the config.
        self._enrich_metadata_from_constants()

        logging.info(f"Channel names: \t{self.channel_names}")
        logging.info(f"Channel abbreviations: \t{self.channel_abbrevs}")

    def _enrich_metadata_from_constants(self) -> None:
        """Overwrite ``sex``/``genotype`` from ``constants.ANIMAL_METADATA``.

        Makes the dataset config (loaded into ``constants.ANIMAL_METADATA`` by
        ``apply_samples_config``) the single source of truth for per-animal
        metadata, applied identically to ``sex`` and ``genotype`` at every WAR
        construction. Updates BOTH the object attributes AND the per-row
        ``result["sex"]``/``result["genotype"]`` columns (downstream renderers read
        the columns, not the attributes).

        Guarded for portability: if the animal is absent from
        ``constants.ANIMAL_METADATA`` (e.g. a standalone load without
        ``apply_samples_config``), the baked values are left untouched. A metadata
        field that is ``None`` does not overwrite a baked value.

        Note: the ``ANIMAL_METADATA`` key and the WAR's canonical attribute/column are
        both ``"genotype"``.
        """
        animal_id = self.animal_id
        if animal_id is None or animal_id not in constants.ANIMAL_METADATA:
            return
        meta = constants.ANIMAL_METADATA[animal_id]
        for attr, new_val in (("genotype", meta.get("genotype")), ("sex", meta.get("sex"))):
            if new_val is None:
                continue  # don't overwrite a baked value with None
            old_val = getattr(self, attr, None)
            if old_val != new_val:
                logging.info(f"Re-enriched {animal_id}: {attr} {old_val!r} -> {new_val!r}")
            setattr(self, attr, new_val)
            if isinstance(self.result, pd.DataFrame):
                self.result[attr] = new_val

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

                if pct_short > 1.0 and not self.suppress_short_interval_error:
                    # Build a diagnostic showing the first few overlapping pairs
                    # so the user can identify which sessions have bad timestamps.
                    short_positions = np.flatnonzero(short_intervals.to_numpy())[:5]
                    diag_lines = []
                    has_animalday = "animalday" in self.result.columns
                    for pos in short_positions:
                        # Offset by 1 to map from sliced short_intervals (which
                        # dropped the first NaT row) back to original DataFrame
                        # positions. Without this, pos=0 wraps to iloc[-1].
                        actual_pos = pos + 1
                        prev_row = self.result.iloc[actual_pos - 1]
                        curr_row = self.result.iloc[actual_pos]
                        gap = timestamp_diffs.iloc[actual_pos]
                        prev_ad = f" ({prev_row['animalday']})" if has_animalday else ""
                        curr_ad = f" ({curr_row['animalday']})" if has_animalday else ""
                        diag_lines.append(
                            f"  {prev_row['timestamp']}{prev_ad} -> "
                            f"{curr_row['timestamp']}{curr_ad}: gap={gap}"
                        )
                    diag = "\n".join(diag_lines)
                    raise ValueError(
                        f"{warning_msg}\n"
                        f"First overlapping pairs (of {n_short} total):\n{diag}\n"
                        f"Hint: if using datetimes_are_start=False, the backward "
                        f"computation assumes contiguous files. Large gaps between "
                        f"files in a session will push computed start times too far "
                        f"back, overlapping adjacent sessions. Consider providing "
                        f"per-file timestamps or set suppress_short_interval_error=True "
                        f"to downgrade this to a warning."
                    )
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
                resolve_channel(x)
                for x in self.channel_names
            ]
        except (ValueError, KeyError) as e:
            raise type(e)(
                f"{e}\n\nChannel names in data: {self.channel_names}"
            ) from e

    def reorder_and_pad_channels(
        self, target_channels: list[str] | None = None, use_abbrevs: bool = True, inplace: bool = True
    ) -> pd.DataFrame:
        """Reorder and pad channels to match a target channel list.

        This method ensures that the data has a consistent channel order and structure
        by reordering existing channels and padding missing channels with NaNs. Channels
        present in the data but **absent from** ``target_channels`` are dropped; a warning
        names them so a montage gap can never silently discard data.

        Args:
            target_channels (list[str], optional): List of target channel names to match.
                Defaults to :data:`neurodent.constants.CHANNEL_ABBREVS` (the canonical
                channel list) when omitted.
            use_abbrevs (bool, optional): If True, target channel names are read as channel abbreviations instead of channel names. Defaults to True.
            inplace (bool, optional): If True, modify the result in place. Defaults to True.
        Returns:
            pd.DataFrame: DataFrame with reordered and padded channels
        """
        if target_channels is None:
            target_channels = list(constants.CHANNEL_ABBREVS)

        duplicates = [ch for ch in target_channels if target_channels.count(ch) > 1]
        if duplicates:
            raise ValueError(
                f"Target channels must be unique. Found duplicates: {duplicates}"
            )

        if inplace:
            result = self.result
        else:
            result = self.result.copy()

        channel_map = {ch: i for i, ch in enumerate(target_channels)}
        channel_names = self.channel_names if not use_abbrevs else self.channel_abbrevs

        valid_channels = [ch for ch in channel_names if ch in channel_map]
        if not valid_channels:
            warnings.warn(
                f"None of the channel names {channel_names} were found in target channels {target_channels}. Is use_abbrevs correctly set?"
            )
        else:
            dropped = [ch for ch in channel_names if ch not in channel_map]
            if dropped:
                warnings.warn(
                    f"Standardization dropping channels not in the target montage: {dropped}. "
                    f"Target channels: {target_channels}. Add them to the channel config "
                    f"(CHANNEL_MAP / `channels`) if this data should be kept."
                )

        for feature in self._feature_columns:
            handler = handler_for(feature)
            result[feature] = handler.reorder_pad(
                result[feature], channel_map, list(channel_names), target_channels
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

    def select_channels(
        self,
        channels: list[str],
        use_abbrevs: bool = True,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """Subset and reorder the WAR's channels to *channels*.

        Every name in *channels* must be present in the WAR's current
        channel list; missing names raise.  Source channels not in
        *channels* are dropped.  Args mirror
        :meth:`reorder_and_pad_channels` — use that one if you want
        NaN-padding for missing target channels.

        Raises:
            ValueError: if any name in *channels* is not present.
        """
        available = self.channel_abbrevs if use_abbrevs else self.channel_names
        missing = [c for c in channels if c not in available]
        if missing:
            raise ValueError(
                f"Requested channels not present in WAR (use "
                f"reorder_and_pad_channels for NaN-padding behaviour): "
                f"{missing}. Available: {list(available)}"
            )
        return self.reorder_and_pad_channels(
            channels, use_abbrevs=use_abbrevs, inplace=inplace
        )

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

    @property
    def path_safe_animal_id(self) -> str:
        """Slugified :attr:`animal_id` for filesystem paths.

        Use this property whenever building a ``Path`` or filename component
        from the animal id.  ``animal_id`` itself stays in its display form
        (which may contain ``/``, ``;``, spaces) for logs and plot labels;
        ``slugify`` is applied here so callers don't have to remember.
        """
        return slugify(self.animal_id)

    @property
    def path_safe_animaldays(self) -> list[str]:
        """Slugified :attr:`animaldays` for filesystem paths."""
        return [slugify(ad) for ad in self.animaldays]

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
            agg_dict[feat] = lambda df, feat=feat: average_feature(
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
