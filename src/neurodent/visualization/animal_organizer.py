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
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .frequency_domain_results import FrequencyDomainSpikeAnalysisResult
    import mne  # type: ignore

import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask import delayed
from scipy.stats import zscore
from scipy.ndimage import binary_opening, binary_closing
from tqdm import tqdm

from .. import constants, core
from ..core import FragmentAnalyzer, get_temp_directory
from ..core.frequency_domain_spike_detection import FrequencyDomainSpikeDetector
from ..core.utils import abbreviate_channel_names, filepath_to_index, parse_chname_to_abbrev, slugify
from .feature_utils import extract_linear_array, extract_band_dict, repack_band_dict, extract_hist_data
from .feature_parser import AnimalFeatureParser


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


class AnimalOrganizer(AnimalFeatureParser):
    """
    Organizes and analyzes recording data from a single animal across multiple sessions.

    AnimalOrganizer uses flexible pattern-based file discovery to locate recording files,
    groups them by session, and creates LongRecordingOrganizer instances for each session.

    Args:
        pattern (str | list[str]): File pattern(s) for discovering recording files.
            - Single pattern: "/path/{animal}/{session}/{index}.rhd"
            - Multiple patterns: ["/path/{animal}/{session}/data.bin", "/path/{animal}/{session}/meta.csv"]

            Placeholders:
                {animal}: Animal ID (e.g., "A10")
                {session}: Session identifier (e.g., "2025-01-24" or "day1")
                {index}: File index within a session (e.g., "1", "2", "3")

            Examples:
                - "/data/{animal}/{session}/{index}.rhd"
                - "/data/{animal}-{session}-{index}.edf"
                - "/data/{session}/\\*/{animal}-{index}.rhd"
                - "/data/\\*\\*/{animal}-{session}-{index}.rhd"
                - "/data/{animal}/{index}.edf"  (no session - will use "unknown")

        animal_id (str | None, optional): Animal ID to filter discovered files.
            If provided, only files matching this animal ID will be included.
        skip_sessions (list[str], optional): Glob patterns for sessions to exclude.
            Uses fnmatch-style wildcards (``*``, ``?``, ``[seq]``).
            E.g. ``["*bad*", "corrupted_*"]``. Defaults to [].
        truncate (bool | int, optional): If True, truncate to first 10 sessions.
            If an integer, truncate to first n sessions. Defaults to False.
        assume_from_number (bool, optional): Whether to parse channel names as numbers
            (used for analysis, not discovery). Defaults to False.
        lro_kwargs (dict, optional): Keyword arguments passed to each LongRecordingOrganizer
            instance. Common options include 'mode', 'extract_func', 'manual_datetimes'.
            Defaults to {}.
        normalize_session (callable | None, optional): A function that transforms session
            keys before grouping. For example, to merge split-day folders like
            "2023-01-15", "2023-01-15(1)", "2023-01-15(2)" into one session, pass
            ``lambda s: re.sub(r"\(\d+\)$", "", s)``. Defaults to None (no normalization).

    Attributes:
        pattern (str | list[str]): The file pattern(s) used for discovery.
        animal_id (str | None): The ID of the animal being analyzed.
        unique_animaldays (list[str]): List of unique session identifiers (format: "{animal}_{session}").
        animaldays (list[str]): Alias for unique_animaldays.
        genotype (str): Genotype of the animal (from ANIMAL_METADATA if available).
        sex (str): Sex of the animal (from ANIMAL_METADATA if available).
        long_recordings (list[LongRecordingOrganizer]): LRO instances, one per session.
        long_analyzers (list[LongRecordingAnalyzer]): Analysis instances, one per session.
        features_df (pd.DataFrame): Aggregated feature DataFrame across all sessions.
        features_avg_df (pd.DataFrame): Average features across sessions.
    """

    def _init_containers(self):
        """Initialize all output containers and processing lists.

        This method centralizes initialization to ensure consistency between
        standard __init__ and factory methods like from_lros().
        """
        # Processing lists
        self.long_analyzers: list[core.LongRecordingAnalyzer] = []

        # Output containers
        self.bad_channels_dict = {}
        self.features_df = pd.DataFrame()
        self.features_avg_df = pd.DataFrame()

        # Result objects
        self.spike_analysis_results = None
        self.frequency_domain_spike_analysis_results = None
        self.window_analysis_result = None

    def __init__(
        self,
        pattern: str | list[str],
        animal_id: str | None = None,
        skip_sessions: list[str] | None = None,
        truncate: bool | int = False,
        assume_from_number: bool = False,
        lro_kwargs: dict | None = None,
        normalize_session: Optional[Callable[[str], str]] = None,
    ) -> None:
        skip_sessions = [] if skip_sessions is None else skip_sessions
        lro_kwargs = {} if lro_kwargs is None else lro_kwargs
        self.pattern = pattern
        self.animal_id = animal_id
        self.assume_from_number = assume_from_number
        self.animal_file_match_pattern = [animal_id] if animal_id else []
        self.day_sep = None
        self.read_mode = "pattern"  # Legacy compat; new pattern-based discovery
        self._normalize_session = normalize_session

        # Warn if pattern(s) don't contain placeholders — metadata extraction won't work
        patterns = [pattern] if isinstance(pattern, (str, Path)) else pattern
        for p in patterns:
            if not re.search(r"\{\w+\}", str(p)):
                warnings.warn(
                    f"Pattern has no placeholders (e.g., '{{animal}}', '{{session}}'). "
                    f"Metadata extraction will be limited. Got: '{p}'",
                    UserWarning,
                    stacklevel=2,
                )

        from neurodent.core.discovery import FileDiscoverer

        self.discoverer = FileDiscoverer(pattern)

        filter_kwargs = {}
        if animal_id is not None:
            filter_kwargs["animal"] = animal_id

        discovered_items = self.discoverer.discover(**filter_kwargs)

        self._animalday_folder_groups = {}
        processed_animaldays = []

        for item in discovered_items:
            # All items are now DiscoveredFile objects with unified interface
            session = item.metadata.get("session", "unknown")
            animal_val = item.metadata.get("animal", animal_id if animal_id else "unknown")
            path_val = item  # Pass the entire DiscoveredFile object

            # Optionally normalize session keys (e.g., strip "(N)" suffixes)
            if self._normalize_session is not None:
                session = self._normalize_session(session)

            if any(fnmatch.fnmatch(session, pat) for pat in skip_sessions):
                continue

            if session not in self._animalday_folder_groups:
                self._animalday_folder_groups[session] = []
                processed_animaldays.append(f"{animal_val}_{session}")

            if path_val:
                self._animalday_folder_groups[session].append(path_val)

        if not self._animalday_folder_groups:
            raise ValueError(f"No items discovered for pattern: {pattern}")

        if truncate:
            from neurodent import core

            truncate = core.utils.parse_truncate(truncate)
            warnings.warn(
                f"AnimalOrganizer will be truncated to the first {truncate} sessions"
            )
            truncated_keys = list(self._animalday_folder_groups.keys())[:truncate]
            self._animalday_folder_groups = {
                k: self._animalday_folder_groups[k] for k in truncated_keys
            }
            processed_animaldays = processed_animaldays[:truncate]

        self.unique_animaldays = processed_animaldays
        self.animaldays = processed_animaldays

        from neurodent import constants

        self.genotype = (
            constants.ANIMAL_METADATA.get(self.animal_id, {}).get("gene", "Unknown")
            if self.animal_id
            else "Unknown"
        )

        self.sex = (
            constants.ANIMAL_METADATA.get(self.animal_id, {}).get("sex", "Unknown")
            if self.animal_id
            else "Unknown"
        )

        self._init_containers()

        if "manual_datetimes" in lro_kwargs:
            import logging

            logging.info("Processing manual_datetimes configuration")
            base_lro_kwargs = lro_kwargs.copy()
            from datetime import datetime

            base_lro_kwargs["manual_datetimes"] = datetime(2000, 1, 1, 0, 0, 0)

            self._processed_timestamps = self._process_manual_datetimes(
                lro_kwargs["manual_datetimes"],
                self._animalday_folder_groups,
                base_lro_kwargs,
            )
            lro_kwargs = base_lro_kwargs
        else:
            self._processed_timestamps = None

        from neurodent import core

        self.long_recordings: list[core.LongRecordingOrganizer] = []
        self._create_long_recordings(lro_kwargs)

        # Set and validate channel_names across all LROs
        self.channel_names = self._validate_channel_names(self.long_recordings)

    def _get_item_name(self, item):
        """Helper to get a representative name for an item which could be a string, Path, list of strings, or DiscoveredFile."""
        from ..core.discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            paths = item.get_path_list()
            if len(paths) > 1:
                return Path(paths[0]).name + "..."
            return Path(paths[0]).name if paths else "unknown"
        if isinstance(item, (list, tuple)):
            return Path(item[0]).name
        return Path(item).name

    def _is_item_file(self, item):
        """Helper to check if an item represents a file(s) rather than a directory."""
        from ..core.discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            paths = item.get_path_list()
            return Path(paths[0]).is_file() if paths else False
        if isinstance(item, (list, tuple)):
            return Path(item[0]).is_file()
        return Path(item).is_file()

    @staticmethod
    def _get_context_path(item) -> Path:
        """Return a single Path from an item (str, Path, list, or DiscoveredFile)."""
        from ..core.discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            return Path(item.get_path_list()[0])
        if isinstance(item, (list, tuple)):
            return Path(item[0])
        return Path(item)

    def _resolve_timestamp_input(self, input_spec, folder_path: Path):
        """
        Recursively resolve any timestamp input type to concrete datetime(s).

        Args:
            input_spec: datetime, str, List[datetime], or Callable returning either
            folder_path: Path to folder for function execution context

        Returns:
            Union[datetime, List[datetime]]: Resolved timestamp(s)

        Raises:
            TypeError: If input_spec is not a supported type
            Exception: If user function fails (wrapped with context)
        """
        if isinstance(input_spec, datetime):
            return input_spec.replace(tzinfo=None)

        elif isinstance(input_spec, str):
            dt = dateutil.parser.parse(input_spec)
            return dt.replace(tzinfo=None)

        elif isinstance(input_spec, list):
            # Validate that all items are datetime objects
            if not all(isinstance(dt, datetime) for dt in input_spec):
                raise TypeError(
                    f"All items in timestamp list must be datetime objects, got: {[type(dt) for dt in input_spec]}"
                )
            return input_spec

        elif callable(input_spec):
            try:
                logging.debug(
                    f"Executing user timestamp function on folder: {folder_path}"
                )
                result = input_spec(folder_path)
                # Recursively process the result (functions can return datetime or list)
                return self._resolve_timestamp_input(result, folder_path)
            except Exception as e:
                logging.error(
                    f"User timestamp function failed on folder '{folder_path}': {e}"
                )
                raise Exception(
                    f"User timestamp function failed on folder '{folder_path}': {e}"
                ) from e

        else:
            raise TypeError(
                f"Invalid timestamp input type: {type(input_spec)}. Expected: datetime, List[datetime], or Callable"
            )

    def _find_folder_by_name(
        self, folder_name: str, animalday_to_folders: dict
    ) -> Path:
        """Find folder path by name in the animalday groups."""
        for animalday, folders in animalday_to_folders.items():
            for folder in folders:
                if Path(folder).name == folder_name:
                    return Path(folder)

        available_names = []
        for folders in animalday_to_folders.values():
            available_names.extend([Path(f).name for f in folders])

        raise ValueError(
            f"Folder name '{folder_name}' not found. Available folders: {available_names}"
        )

    def _get_folders_for_animal(
        self, animal_id: str, animalday_to_folders: dict
    ) -> list:
        """Find all folder paths belonging to a specific animal ID."""
        matching_folders = []
        for animalday, folders in animalday_to_folders.items():
            if animalday.startswith(animal_id):
                matching_folders.extend(folders)
        return matching_folders

    def _compute_global_timeline(
        self,
        base_datetime,
        animalday_to_items: dict,
        base_lro_kwargs: dict,
        original_manual_datetimes=None,
    ) -> dict:
        total_items = sum(len(items) for items in animalday_to_items.values())
        total_animaldays = len(animalday_to_items)

        logging.info(
            f"Computing continuous timeline for {total_animaldays} animaldays ({total_items} total items) "
            f"starting at {base_datetime}"
        )

        from ..core.discovery import _natural_sort_key

        ordered_items = []
        if original_manual_datetimes is not None:
            if isinstance(original_manual_datetimes, list):
                for animalday in sorted(animalday_to_items.keys(), key=_natural_sort_key):
                    items = animalday_to_items[animalday]
                    ordered_items.extend(items)
            else:
                for animalday in sorted(animalday_to_items.keys(), key=_natural_sort_key):
                    items = animalday_to_items[animalday]
                    sorted_items = sorted(items, key=lambda f: _natural_sort_key(self._get_item_name(f)))
                    ordered_items.extend(sorted_items)
        else:
            for animalday in sorted(animalday_to_items.keys(), key=_natural_sort_key):
                items = animalday_to_items[animalday]
                if len(items) > 1:
                    item_lro_pairs = []
                    for item in items:
                        try:
                            temp_lro = core.LongRecordingOrganizer(
                                item, **base_lro_kwargs
                            )
                            item_lro_pairs.append((item, temp_lro))
                        except (FileNotFoundError, OSError, ValueError, ImportError, AttributeError, TypeError) as e:
                            logging.warning(
                                f"Failed to create temp LRO for duration estimation in {self._get_item_name(item)}: {e}"
                            )
                            item_lro_pairs.append((item, None))

                    sorted_pairs = self._sort_lros_by_median_time(item_lro_pairs)
                    ordered_items.extend([item for item, _ in sorted_pairs])
                else:
                    ordered_items.extend(items)

        item_durations = {}
        logging.info(
            f"Ordered items for timeline: {[self._get_item_name(f) for f in ordered_items]}"
        )

        if original_manual_datetimes is not None:
            if isinstance(original_manual_datetimes, list):
                if len(original_manual_datetimes) != len(ordered_items):
                    raise ValueError(
                        f"manual_datetimes list length ({len(original_manual_datetimes)}) "
                        f"does not match number of items ({len(ordered_items)})."
                    )

                item_timestamps = []
                for i, (item, ts) in enumerate(
                    zip(ordered_items, original_manual_datetimes)
                ):
                    try:
                        context_path = self._get_context_path(item)
                        resolved_ts = self._resolve_timestamp_input(ts, context_path)
                        item_timestamps.append((item, resolved_ts))
                    except Exception as e:
                        raise ValueError(
                            f"Failed to parse timestamp at index {i} for item {self._get_item_name(item)}: {e}"
                        ) from e

            elif isinstance(original_manual_datetimes, (str, type(base_datetime))):
                try:
                    if isinstance(original_manual_datetimes, str):
                        first_item = ordered_items[0] if ordered_items else "."
                        context_path = self._get_context_path(first_item)
                        resolved_ts = self._resolve_timestamp_input(
                            original_manual_datetimes, context_path
                        )
                    else:
                        resolved_ts = original_manual_datetimes

                    item_timestamps = [(item, resolved_ts) for item in ordered_items]
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse scalar manual_datetimes: {e}"
                    ) from e
            else:
                item_timestamps = []
                for item in ordered_items:
                    context_path = self._get_context_path(item)
                    resolved_ts = self._resolve_timestamp_input(
                        original_manual_datetimes, context_path
                    )
                    item_timestamps.append((item, resolved_ts))

            for item, timestamp in item_timestamps:
                _lro_kwargs = base_lro_kwargs.copy()
                if self._is_item_file(item) and _lro_kwargs.get("mode") == "mne":
                    _lro_kwargs["input_type"] = "file"
                _lro_kwargs["manual_datetimes"] = timestamp

                try:
                    temp_lro = core.LongRecordingOrganizer(item, **_lro_kwargs)
                    duration = (
                        temp_lro.LongRecording.get_duration()
                        if hasattr(temp_lro, "LongRecording") and temp_lro.LongRecording
                        else 0.0
                    )
                    item_durations[item] = duration
                    logging.info(
                        f"Item {self._get_item_name(item)}: duration = {duration:.1f}s (loaded with manual timestamp)"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load item {self._get_item_name(item)} for duration estimation: {e}"
                    ) from e

        else:
            for item in ordered_items:
                _lro_kwargs = base_lro_kwargs.copy()
                if self._is_item_file(item) and _lro_kwargs.get("mode") == "mne":
                    _lro_kwargs["input_type"] = "file"

                try:
                    temp_lro = core.LongRecordingOrganizer(item, **_lro_kwargs)
                    duration = (
                        temp_lro.LongRecording.get_duration()
                        if hasattr(temp_lro, "LongRecording") and temp_lro.LongRecording
                        else 0.0
                    )
                    item_durations[item] = duration
                    logging.info(
                        f"Item {self._get_item_name(item)}: estimated duration = {duration:.1f}s"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load item {self._get_item_name(item)} for duration estimation: {e}"
                    ) from e

        datetimes_are_start = base_lro_kwargs.get("datetimes_are_start", True)
        result = {}

        if datetimes_are_start:
            current_start_time = base_datetime
            for item in ordered_items:
                item_name = self._get_item_name(item)
                result[item_name] = current_start_time
                current_start_time = current_start_time + timedelta(
                    seconds=item_durations[item]
                )
        else:
            current_end_time = base_datetime
            for item in reversed(ordered_items):
                item_name = self._get_item_name(item)
                duration = item_durations[item]
                start_time = current_end_time - timedelta(seconds=duration)
                result[item_name] = start_time
                current_end_time = start_time

        total_duration = sum(item_durations.values())
        logging.info(
            f"Timeline computed: {len(result)} items, total duration {total_duration:.1f}s"
        )
        return result

    def _process_manual_datetimes(
        self, manual_datetimes, animalday_to_items: dict, base_lro_kwargs: dict
    ) -> dict:
        if isinstance(manual_datetimes, dict):
            animal_items = []
            for items in animalday_to_items.values():
                animal_items.extend(items)

            item_names = {self._get_item_name(f) for f in animal_items}
            has_item_keys = any(k in item_names for k in manual_datetimes.keys())
            session_keys = set(animalday_to_items.keys())
            has_session_keys = any(k in session_keys for k in manual_datetimes.keys())

            if has_item_keys:
                logging.info(
                    f"manual_datetimes keys match items for {self.animal_id}. Treating as item mapping."
                )
                if not animal_items:
                    raise ValueError(
                        f"Manual timestamps provided for '{self.animal_id}' but no items found."
                    )
                missing = [
                    name for name in item_names
                    if name not in manual_datetimes
                ]
                if missing:
                    raise ValueError(
                        f"Missing entries in manual_datetimes for items: {missing}."
                    )
                out = {}
                for item in animal_items:
                    fname = self._get_item_name(item)
                    context_path = self._get_context_path(item)
                    out[fname] = self._resolve_timestamp_input(
                        manual_datetimes[fname], context_path
                    )
                return out

            elif has_session_keys:
                logging.info(
                    f"manual_datetimes keys match sessions for {self.animal_id}. "
                    "Computing per-session timelines."
                )
                out = {}
                missing_sessions = [
                    k for k in animalday_to_items
                    if k not in manual_datetimes
                ]
                if missing_sessions:
                    raise ValueError(
                        f"Missing entries in manual_datetimes for sessions: {missing_sessions}."
                    )
                for sess_key, sess_items in animalday_to_items.items():
                    sess_ts = manual_datetimes[sess_key]
                    context_path = self._get_context_path(sess_items[0])
                    resolved_dt = self._resolve_timestamp_input(
                        sess_ts, context_path
                    )
                    sess_item_dict = {
                        self._get_item_name(f): [f] for f in sess_items
                    }
                    sess_timeline = self._compute_global_timeline(
                        resolved_dt,
                        sess_item_dict,
                        base_lro_kwargs,
                        original_manual_datetimes=sess_ts,
                    )
                    out.update(sess_timeline)
                return out

            else:
                raise ValueError(
                    f"manual_datetimes dictionary keys don't match any item names or "
                    f"session names for '{self.animal_id}'. "
                    f"Keys: {list(manual_datetimes.keys())}"
                )

        elif isinstance(manual_datetimes, (datetime, str)):
            start_dt = manual_datetimes
            if isinstance(start_dt, str):
                first_item = (
                    list(animalday_to_items.values())[0][0]
                    if animalday_to_items
                    else "."
                )
                context_path = self._get_context_path(first_item)
                start_dt = self._resolve_timestamp_input(manual_datetimes, context_path)

            from pandas import Timestamp

            if isinstance(start_dt, datetime) or isinstance(start_dt, Timestamp):
                logging.info(
                    f"Processing global manual datetimes starting at {start_dt}"
                )
                return self._compute_global_timeline(
                    start_dt,
                    animalday_to_items,
                    base_lro_kwargs,
                    original_manual_datetimes=manual_datetimes,
                )
            warnings.warn(
                "String timestamp resolved to non-scalar. Falling back to default processing."
            )

        else:
            logging.info("Processing manual datetimes input for all items")
            out = {}
            for animalday, items in animalday_to_items.items():
                for item in items:
                    item_name = self._get_item_name(item)
                    context_path = self._get_context_path(item)
                    out[item_name] = self._resolve_timestamp_input(
                        manual_datetimes, context_path
                    )
            return out

    def _create_long_recordings(self, lro_kwargs: dict):
        """Create LongRecordingOrganizer instances for each unique animalday."""
        self.long_recordings: list[core.LongRecordingOrganizer] = []
        for animalday, items in self._animalday_folder_groups.items():
            kwargs = lro_kwargs.copy()
            if getattr(self, "_processed_timestamps", None) is not None:
                # _processed_timestamps is keyed by item name, not animalday
                if len(items) == 1:
                    item_name = self._get_item_name(items[0])
                    if item_name in self._processed_timestamps:
                        kwargs["manual_datetimes"] = self._processed_timestamps[item_name]
                        kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                        logging.debug(
                            f"Using processed timestamp for {item_name}: {kwargs['manual_datetimes']}"
                        )
                else:
                    # For multi-item animaldays, collect per-item timestamps as a list
                    item_timestamps = []
                    for item in items:
                        item_name = self._get_item_name(item)
                        if item_name in self._processed_timestamps:
                            item_timestamps.append(self._processed_timestamps[item_name])
                    if item_timestamps:
                        kwargs["manual_datetimes"] = item_timestamps
                        kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                        logging.debug(
                            f"Using processed timestamps for {animalday}: {item_timestamps}"
                        )

            if len(items) == 1:
                item_to_pass = items[0]
                kw = kwargs.copy()
                if self._is_item_file(item_to_pass) and isinstance(
                    item_to_pass, (list, tuple)
                ):
                    # LRO handles lists of files directly, but we pass input_type='files'? Wait, LRO handles it natively now
                    pass
                lro = core.LongRecordingOrganizer(item_to_pass, **kw)
            else:
                logging.info(
                    f"Creating individual LROs for {len(items)} items for {animalday}"
                )
                item_lro_pairs = []
                for item in items:
                    individual_kwargs = kwargs.copy()
                    # Distribute per-item timestamp so each LRO gets its own
                    if getattr(self, "_processed_timestamps", None) is not None:
                        item_name = self._get_item_name(item)
                        if item_name in self._processed_timestamps:
                            individual_kwargs["manual_datetimes"] = (
                                self._processed_timestamps[item_name]
                            )
                            individual_kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                    individual_lro = core.LongRecordingOrganizer(
                        item, **individual_kwargs
                    )
                    item_lro_pairs.append((item, individual_lro))

                sorted_folder_lro_pairs = self._sort_lros_by_median_time(item_lro_pairs)

                logging.info("LRO merge order for overlapping animalday:")
                for i, (item, lro) in enumerate(sorted_folder_lro_pairs):
                    item_name = self._get_item_name(item)
                    try:
                        duration = (
                            lro.LongRecording.get_duration()
                            if hasattr(lro, "LongRecording") and lro.LongRecording
                            else 0
                        )
                        duration_str = f"{float(duration):.1f}s"
                    except (TypeError, ValueError):
                        duration_str = "mock"
                    logging.info(f"  {i + 1}. {item_name} (duration: {duration_str})")

                merged_lro = sorted_folder_lro_pairs[0][1]
                logging.info(
                    f"Base LRO: {self._get_item_name(sorted_folder_lro_pairs[0][0])}"
                )

                for i, (item, lro) in enumerate(sorted_folder_lro_pairs[1:], 1):
                    item_name = self._get_item_name(item)
                    logging.info(f"Merging LRO {i}: {item_name} into base LRO")
                    merged_lro.merge(lro)

                lro = merged_lro
                logging.info(
                    f"Successfully merged {len(sorted_folder_lro_pairs)} LROs for {animalday}"
                )

            self.long_recordings.append(lro)

        self._log_timeline_summary()

        if len(self.long_recordings) != len(self.unique_animaldays):
            error_msg = (
                f"Mismatch: Created {len(self.long_recordings)} LROs "
                f"but found {len(self.unique_animaldays)} unique animaldays. "
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def _log_timeline_summary(self):
        """Log timeline summary for debugging purposes."""

        lines = ["AnimalOrganizer Timeline Summary:"]

        if not getattr(self, "long_recordings", []):
            lines.append("No LongRecordings created")
        else:
            for i, lro in enumerate(self.long_recordings):
                try:
                    start_time = self._get_lro_start_time(lro)
                    end_time = self._get_lro_end_time(lro)
                    duration = (
                        lro.LongRecording.get_duration()
                        if hasattr(lro, "LongRecording") and lro.LongRecording
                        else 0
                    )
                    n_files = (
                        len(lro.file_durations)
                        if hasattr(lro, "file_durations") and lro.file_durations
                        else 1
                    )

                    if hasattr(lro, "data_files") and lro.data_files:
                        name = Path(lro.data_files[0]).name + "..."
                    elif hasattr(lro, "item") and lro.item:
                        name = self._get_item_name(lro.item)
                    else:
                        name = "unknown"

                    lines.append(
                        f"LRO {i}: {start_time} -> {end_time} "
                        f"(duration: {duration:.1f}s, items: {n_files}, item: {name})"
                    )
                except (AttributeError, TypeError, IndexError, ValueError) as e:
                    lines.append(f"Failed to get timeline info for LRO {i}: {e}")

        logging.info("\n".join(lines))

    def _get_lro_start_time(self, lro):
        """Get the start time of an LRO."""
        if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes:
            if hasattr(lro, "file_durations") and lro.file_durations:
                try:
                    first_end = next(
                        dt for dt in lro.file_end_datetimes if dt is not None
                    )
                    first_duration = lro.file_durations[0]
                    from datetime import timedelta

                    return first_end - timedelta(seconds=first_duration)
                except StopIteration:
                    pass
        return "unknown"

    def _get_lro_end_time(self, lro):
        """Get the end time of an LRO."""
        if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes:
            end_times = [dt for dt in lro.file_end_datetimes if dt is not None]
            if end_times:
                return max(end_times)
        return "unknown"

    def get_timeline_summary(self):
        """
        Get timeline summary as a DataFrame for user inspection and debugging.
        """
        if not getattr(self, "long_recordings", []):
            import pandas as pd

            return pd.DataFrame()

        timeline_data = []
        for i, lro in enumerate(self.long_recordings):
            try:
                start_time = self._get_lro_start_time(lro)
                end_time = self._get_lro_end_time(lro)
                duration = (
                    lro.LongRecording.get_duration()
                    if hasattr(lro, "LongRecording") and lro.LongRecording
                    else 0
                )
                n_files = (
                    len(lro.file_durations)
                    if hasattr(lro, "file_durations") and lro.file_durations
                    else 1
                )
                folder_path = lro.display_name

                timeline_data.append(
                    {
                        "lro_index": i,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_s": duration,
                        "n_files": n_files,
                        "folder_path": folder_path,
                        "folder_name": (
                            Path(str(folder_path)).name
                            if folder_path != "unknown"
                            else "unknown"
                        ),
                        "animalday": getattr(lro, "labels", {}).get(
                            "animalday", "unknown"
                        ),
                    }
                )
            except (AttributeError, TypeError, ValueError) as e:
                import logging

                logging.warning(f"Failed to get timeline metrics for LRO {i}: {e}")

        import pandas as pd

        return pd.DataFrame(timeline_data)

    @staticmethod
    def _sort_lros_by_median_time_static(lro_pairs):
        """Sort LROs by median timestamp of their constituent recordings.

        Static version that can be called from classmethods.

        Args:
            lro_pairs (list): List of (identifier, lro) tuples where identifier
                can be folder path or any string.

        Returns:
            list: Sorted (identifier, lro) tuples in temporal order based on median timestamp

        Note:
            Extracts file_end_datetimes from each LRO, calculates median timestamp,
            and sorts LROs by this median. Falls back to identifier ordering if
            timestamps unavailable.
        """
        if len(lro_pairs) <= 1:
            return lro_pairs

        lro_times = []

        for identifier, lro in lro_pairs:
            try:
                # Get median timestamp from constituent recordings
                if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes:
                    try:
                        valid_timestamps = [
                            ts for ts in lro.file_end_datetimes if ts is not None
                        ]
                    except TypeError:
                        valid_timestamps = []

                    if valid_timestamps:
                        # Sort and get median
                        valid_timestamps.sort()
                        n = len(valid_timestamps)

                        if n % 2 == 1:
                            median_timestamp = valid_timestamps[n // 2]
                        else:
                            mid1 = valid_timestamps[n // 2 - 1]
                            mid2 = valid_timestamps[n // 2]
                            median_timestamp = mid1 + (mid2 - mid1) / 2

                        median_time_seconds = median_timestamp.timestamp()
                        logging.debug(
                            f"LRO {identifier}: {n} recordings, "
                            f"median timestamp: {median_timestamp}"
                        )
                    else:
                        raise ValueError(f"No valid timestamps in LRO {identifier}")
                else:
                    raise ValueError(f"No file_end_datetimes in LRO {identifier}")

            except ValueError as e:
                logging.warning(
                    f"Could not determine timestamp for LRO {identifier}: {e}. "
                    f"Using fallback ordering."
                )
                # Use a very large timestamp to sort to end
                median_time_seconds = float("inf")

            lro_times.append((median_time_seconds, identifier, lro))

        # Sort by timestamp
        lro_times.sort(key=lambda x: x[0])

        # Return as (identifier, lro) tuples
        return [(identifier, lro) for _, identifier, lro in lro_times]

    def _sort_lros_by_median_time(self, folder_lro_pairs):
        """Sort LROs by median timestamp of their constituent recordings.

        Instance method wrapper around static version for backward compatibility.

        Args:
            folder_lro_pairs (list): List of (folder_path, lro) tuples

        Returns:
            list: Sorted (folder_path, lro) tuples in temporal order based on median timestamp

        Note:
            Extracts file_end_datetimes from each LRO (timestamps from LastEdit fields in metadata CSV files),
            calculates the median timestamp of constituent recordings within each LRO, and sorts LROs
            by this median timestamp. This ensures proper temporal ordering based on actual recording
            content rather than folder naming conventions. Falls back to folder modification time if
            no valid timestamps are available.
        """
        # Call static version for sorting logic
        sorted_folder_lro_pairs = self._sort_lros_by_median_time_static(
            folder_lro_pairs
        )

        # Add detailed logging (only in instance method)
        if len(folder_lro_pairs) > 1:
            from datetime import datetime

            logging.info("LRO temporal sorting details:")
            for i, (folder, lro) in enumerate(sorted_folder_lro_pairs):
                folder_name = self._get_item_name(folder)

                # Get median time for logging
                try:
                    if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes:
                        valid_timestamps = [
                            ts for ts in lro.file_end_datetimes if ts is not None
                        ]
                        if valid_timestamps:
                            valid_timestamps.sort()
                            n = len(valid_timestamps)
                            if n % 2 == 1:
                                median_timestamp = valid_timestamps[n // 2]
                            else:
                                mid1 = valid_timestamps[n // 2 - 1]
                                mid2 = valid_timestamps[n // 2]
                                median_timestamp = mid1 + (mid2 - mid1) / 2
                            median_time_str = median_timestamp.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        else:
                            median_time_str = "no timestamps"
                    else:
                        median_time_str = "no timestamps"
                except (AttributeError, TypeError, ValueError):
                    median_time_str = "error"

                # Handle mock objects gracefully for duration
                try:
                    duration = (
                        lro.LongRecording.get_duration()
                        if hasattr(lro, "LongRecording") and lro.LongRecording
                        else 0
                    )
                    duration_str = f"{float(duration):.1f}s"
                except (TypeError, ValueError):
                    duration_str = "mock"

                # Show number of recordings in LRO
                try:
                    n_recordings = (
                        len(lro.file_end_datetimes)
                        if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes
                        else 0
                    )
                except (TypeError, AttributeError):
                    n_recordings = "unknown"

                logging.info(
                    f"  {i + 1}. {folder_name}: median_timestamp={median_time_str}, {n_recordings} recordings, duration={duration_str}"
                )

            # Summary line for quick reference
            folder_names = [self._get_item_name(f) for f, _ in sorted_folder_lro_pairs]
            logging.info(f"Final sort order: {folder_names}")

        return sorted_folder_lro_pairs

    def convert_colbins_to_rowbins(
        self, overwrite=False, multiprocess_mode: Literal["dask", "serial"] = "serial"
    ):
        for lrec in tqdm(
            self.long_recordings, desc="Converting column bins to row bins"
        ):
            lrec.convert_colbins_to_rowbins(
                overwrite=overwrite, multiprocess_mode=multiprocess_mode
            )

    def convert_rowbins_to_rec(
        self, multiprocess_mode: Literal["dask", "serial"] = "serial"
    ):
        for lrec in tqdm(self.long_recordings, desc="Converting row bins to recs"):
            lrec.convert_rowbins_to_rec(multiprocess_mode=multiprocess_mode)

    def cleanup_rec(self):
        for lrec in self.long_recordings:
            lrec.cleanup_rec()

    def _iter_valid_recordings(self):
        """Yield (index, lrec) pairs, skipping recordings with zero samples.

        This centralizes empty-recording validation so that compute_bad_channels,
        compute_windowed_analysis, and compute_frequency_domain_spike_analysis
        all share the same guard.
        """
        for i, lrec in enumerate(self.long_recordings):
            if (
                hasattr(lrec, "LongRecording")
                and lrec.LongRecording is not None
                and lrec.LongRecording.get_total_samples() == 0
            ):
                logging.warning(
                    f"Skipping recording {i} ({lrec.display_name}): 0 total samples"
                )
                continue
            yield i, lrec

    def _validate_sampling_rates(self):
        """Validate that all valid recordings share the same sampling rate.

        Inconsistent sampling rates across recordings lead to PSD arrays with
        different frequency-axis lengths, which causes downstream failures in
        ``_apply_filter`` and other operations that stack arrays across windows.

        Raises:
            ValueError: If recordings have different sampling rates.
        """
        sfreqs: dict[str, float] = {}
        for _i, lrec in self._iter_valid_recordings():
            long_rec = getattr(lrec, "LongRecording", None)
            if long_rec is None:
                logging.warning(
                    f"Skipping recording {_i} ({getattr(lrec, 'display_name', 'unknown')}): "
                    "LongRecording is None"
                )
                continue
            if not hasattr(long_rec, "get_sampling_frequency"):
                raise ValueError(
                    f"LongRecording for recording "
                    f"{getattr(lrec, 'display_name', f'index {_i}')!r} does not define "
                    "get_sampling_frequency()."
                )
            sf = long_rec.get_sampling_frequency()
            sfreqs[lrec.display_name] = sf

        if not sfreqs:
            return

        unique_rates = set(sfreqs.values())
        if len(unique_rates) > 1:
            details = ", ".join(
                f"{name}: {rate} Hz" for name, rate in sfreqs.items()
            )
            raise ValueError(
                f"All recordings must have the same sampling rate to produce "
                f"consistent feature shapes (e.g. PSD). "
                f"Found {len(unique_rates)} different rates: {details}"
            )

    def compute_bad_channels(
        self, lof_threshold: float = None, force_recompute: bool = False,
        lof_chunk_duration_s: float = 60,
    ):
        """Compute bad channels using LOF analysis for all recordings.

        Args:
            lof_threshold (float, optional): Threshold for determining bad channels from LOF scores.
                                           If None, only computes/loads scores without setting bad_channel_names.
            force_recompute (bool): Whether to recompute LOF scores even if they exist.
            lof_chunk_duration_s (float): Duration in seconds of each chunk used
                for the pairwise-distance computation in LOF.  Defaults to 60.
        """
        logging.info(
            f"Computing bad channels for {len(self.long_recordings)} recordings with threshold={lof_threshold}"
        )
        for i, lrec in self._iter_valid_recordings():
            logging.debug(
                f"Computing bad channels for recording {i}: {self.animaldays[i]}"
            )
            lrec.compute_bad_channels(
                lof_threshold=lof_threshold, force_recompute=force_recompute,
                lof_chunk_duration_s=lof_chunk_duration_s,
            )
            logging.debug(
                f"Recording {i} LOF scores computed: {hasattr(lrec, 'lof_scores') and lrec.lof_scores is not None}"
            )

        # Update bad channels dict if threshold was applied
        if lof_threshold is not None:
            self.bad_channels_dict = {
                animalday: lrec.bad_channel_names
                for animalday, lrec in zip(self.animaldays, self.long_recordings)
            }

    def apply_lof_threshold(self, lof_threshold: float):
        """Apply threshold to existing LOF scores to determine bad channels for all recordings.

        Args:
            lof_threshold (float): Threshold for determining bad channels.
        """
        for lrec in self.long_recordings:
            lrec.apply_lof_threshold(lof_threshold)

        self.bad_channels_dict = {
            animalday: lrec.bad_channel_names
            for animalday, lrec in zip(self.animaldays, self.long_recordings)
        }

    def get_all_lof_scores(self) -> dict:
        """Get LOF scores for all recordings.

        Returns:
            dict: Dictionary mapping animal days to LOF score dictionaries.
        """
        return {
            animalday: lrec.get_lof_scores()
            for animalday, lrec in zip(self.animaldays, self.long_recordings)
        }

    def compute_windowed_analysis(
        self,
        features: list[str],
        exclude: list[str] = [],
        window_s=5,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        suppress_short_interval_error=False,
        apply_notch_filter=True,
        chunk_duration_s: Optional[float] = 3600,
        **kwargs,
    ) -> "WindowAnalysisResult":
        """Computes windowed analysis of animal recordings. The data is divided into windows (time bins), then features are extracted from each window. The result is
        formatted to a Dataframe and wrapped into a WindowAnalysisResult object.

        Args:
            features (list[str]): List of features to compute. See individual ``compute_...()`` functions for output format
            exclude (list[str], optional): List of features to ignore. Will override the features parameter. Defaults to [].
            window_s (int, optional): Length of each window in seconds. Note that some features break with very short window times. Defaults to 5.
            suppress_short_interval_error (bool, optional): If True, suppress ValueError for short intervals between timestamps in resulting WindowAnalysisResult. Useful for aggregated WARs. Defaults to False.
            apply_notch_filter (bool, optional): Whether to apply notch filtering to remove line noise. Uses constants.LINE_FREQ. Defaults to True.
            chunk_duration_s (float, optional): Duration in seconds of data to hold
                in memory at once during the Dask processing path.  Internally
                converted to a number of fragments via
                ``int(chunk_duration_s / window_s)``.  When ``None``,
                all fragments are loaded into a single NumPy array before being
                written to the intermediate zarr store — the original behavior,
                which maximizes throughput but requires enough RAM to hold the
                entire recording at once.  When set to a positive value, only the
                corresponding number of fragments are buffered at a time, streaming
                them to zarr incrementally; use a small value (e.g. 250) on
                memory-constrained machines and a larger value (e.g. 2500+) on
                high-memory nodes for maximum throughput.  Only has an effect when
                ``multiprocess_mode="dask"``.  Defaults to 3600.

        Raises:
            AttributeError: If a feature's ``compute_...()`` function was not implemented, this error will be raised.

        Returns:
            WindowAnalysisResult: A WindowAnalysisResult object containing extracted features for all recordings
        """
        features = _sanitize_feature_request(features, exclude)

        self._validate_sampling_rates()

        dataframes = []
        for _i, lrec in self._iter_valid_recordings():
            logging.info(f"Computing windowed analysis for {lrec.display_name}")
            lan = core.LongRecordingAnalyzer(
                lrec, fragment_len_s=window_s, apply_notch_filter=apply_notch_filter
            )
            if lan.n_fragments == 0:
                logging.warning(
                    f"No fragments found for {lrec.display_name}. Skipping."
                )
                continue

            logging.debug(f"Processing {lan.n_fragments} fragments")
            miniters = int(lan.n_fragments / 100)
            match multiprocess_mode:
                case "dask":
                    # The last fragment is not included because it makes the dask array ragged
                    logging.debug("Converting LongRecording to numpy array")

                    n_fragments_war = max(lan.n_fragments - 1, 1)
                    first_fragment = lan.get_fragment_np(0)

                    if chunk_duration_s is not None:
                        # Convert seconds → number of fragments
                        n_frag_per_chunk = max(1, int(chunk_duration_s / window_s))
                        # Streaming path: stream fragments to zarr in batches,
                        # keeping only `n_frag_per_chunk` fragments in RAM at a time.
                        tmppath = core.utils.stream_fragments_to_zarr(
                            lan.get_fragment_np,
                            n_fragments_war,
                            first_fragment.shape,
                            first_fragment.dtype,
                            n_frag_per_chunk,
                        )
                    else:
                        # Default path: allocate the full array then write to zarr in
                        # one shot.  Maximises throughput on high-memory systems.
                        np_fragments = np.empty(
                            (n_fragments_war,) + first_fragment.shape,
                            dtype=first_fragment.dtype,
                        )
                        logging.debug(f"np_fragments.shape: {np_fragments.shape}")
                        for idx in range(n_fragments_war):
                            np_fragments[idx] = lan.get_fragment_np(idx)

                        # Cache fragments to zarr
                        tmppath, _ = core.utils.cache_fragments_to_zarr(
                            np_fragments, n_fragments_war
                        )
                        del np_fragments

                    logging.debug("Processing metadata serially")
                    metadatas = [
                        self._process_fragment_metadata(idx, lan, window_s)
                        for idx in range(n_fragments_war)
                    ]
                    meta_df = pd.DataFrame(metadatas)

                    logging.debug("Processing features in parallel")
                    np_fragments_reconstruct = da.from_zarr(
                        tmppath, chunks=("auto", -1, -1)
                    )
                    logging.debug(f"Dask array shape: {np_fragments_reconstruct.shape}")
                    logging.debug(
                        f"Dask array chunks: {np_fragments_reconstruct.chunks}"
                    )

                    # Create delayed tasks for each fragment using efficient dependency resolution
                    feature_values = [
                        delayed(FragmentAnalyzer.process_fragment_with_dependencies)(
                            np_fragments_reconstruct[idx], lan.f_s, features, kwargs
                        )
                        for idx in range(n_fragments_war)
                    ]

                    # Compute features in parallel
                    feature_values = dask.compute(*feature_values)

                    # Clean up temp directory after processing
                    logging.debug("Cleaning up temp directory")
                    try:
                        import shutil

                        shutil.rmtree(tmppath)
                    except (OSError, FileNotFoundError) as e:
                        logging.warning(
                            f"Failed to remove temporary directory {tmppath}: {e}"
                        )

                    logging.debug("Combining metadata and feature values")
                    feat_df = pd.DataFrame(feature_values)
                    lan_df = pd.concat([meta_df, feat_df], axis=1)

                case _:
                    logging.debug("Processing serially")
                    lan_df = []
                    for idx in tqdm(
                        range(lan.n_fragments),
                        desc="Processing rows",
                        miniters=miniters,
                    ):
                        lan_df.append(
                            self._process_fragment_serial(
                                idx, features, lan, window_s, kwargs
                            )
                        )

            lan_df = pd.DataFrame(lan_df)

            logging.debug("Validating timestamps")
            core.validate_timestamps(lan_df["timestamp"].tolist())
            lan_df = lan_df.sort_values("timestamp").reset_index(drop=True)

            self.long_analyzers.append(lan)
            dataframes.append(lan_df)

        self.features_df = pd.concat(dataframes)
        self.features_df = self.features_df

        # Collect LOF scores from long recordings
        lof_scores_dict = {}
        missing_lof_animaldays = []
        for animalday, lrec in zip(self.animaldays, self.long_recordings):
            logging.debug(
                f"Checking LOF scores for {animalday}: has_attr={hasattr(lrec, 'lof_scores')}, "
                f"is_not_none={getattr(lrec, 'lof_scores', None) is not None}"
            )
            if hasattr(lrec, "lof_scores") and lrec.lof_scores is not None:
                lof_scores_dict[animalday] = {
                    "lof_scores": lrec.lof_scores.tolist(),
                    "channel_names": lrec.channel_names,
                }
                logging.info(
                    f"Added LOF scores for {animalday}: {len(lrec.lof_scores)} channels"
                )
            else:
                missing_lof_animaldays.append(animalday)
                logging.warning(
                    f"Missing LOF scores for {animalday}! LOF computation may have failed or "
                    f"compute_bad_channels() was not called for this LRO."
                )

        logging.info(f"Total LOF scores collected: {len(lof_scores_dict)} animal days")

        # Warn loudly if any animaldays are missing LOF scores
        if missing_lof_animaldays:
            warning_msg = (
                f"WARNING: {len(missing_lof_animaldays)} animalday(s) are missing LOF scores: {missing_lof_animaldays}. "
                f"Expected {len(self.animaldays)} but got {len(lof_scores_dict)}. "
                f"These sessions will be auto-populated with empty placeholders and excluded from LOF-based analysis."
            )
            logging.warning(warning_msg)
            warnings.warn(warning_msg)

        from .results import WindowAnalysisResult
        self.window_analysis_result = WindowAnalysisResult(
            self.features_df,
            self.animal_id,
            self.genotype,
            self.sex,
            self.channel_names,
            self.assume_from_number,
            self.bad_channels_dict,
            suppress_short_interval_error,
            lof_scores_dict,
        )

        return self.window_analysis_result

    def compute_frequency_domain_spike_analysis(
        self,
        detection_params: dict = None,
        chunk_duration_s: float = 3600,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
    ):
        """
        Compute frequency-domain spike detection on all long recordings.

        Args:
            detection_params (dict, optional): Detection parameters. Uses defaults if None.
            chunk_duration_s (float): Duration in seconds of each
                processing chunk.  Defaults to 3600 (1 hour).  The full
                recording is always analysed; this parameter controls peak RAM
                by processing in overlapping chunks.  ``None`` loads the full
                recording at once (fastest).
            multiprocess_mode (Literal["dask", "serial"]): Processing mode

        Returns:
            list[FrequencyDomainSpikeAnalysisResult]: Results for each recording session

        Raises:
            ImportError: If SpikeInterface is not available
        """
        # Import here to avoid circular imports
        from .frequency_domain_results import FrequencyDomainSpikeAnalysisResult

        fdsar_list = []

        logging.info(
            f"Running frequency-domain spike detection on {len(self.long_recordings)} recordings"
        )
        logging.info(f"Detection parameters: {detection_params}")

        for i, lrec in self._iter_valid_recordings():
            rec = lrec.LongRecording

            try:
                # Run frequency domain spike detection
                spike_indices_per_channel, mne_raw_with_annotations = (
                    FrequencyDomainSpikeDetector.detect_spikes_recording(
                        rec,
                        detection_params=detection_params,
                        chunk_duration_s=chunk_duration_s,
                        multiprocess_mode=multiprocess_mode,
                    )
                )

                # Create FrequencyDomainSpikeAnalysisResult
                fdsar = FrequencyDomainSpikeAnalysisResult.from_detection_results(
                    spike_indices_per_channel=spike_indices_per_channel,
                    mne_raw_with_annotations=mne_raw_with_annotations,
                    detection_params=detection_params or {},
                    animal_id=self.animal_id,
                    genotype=self.genotype,
                    animal_day=self.animaldays[i],
                    bin_folder_name=(
                        getattr(self, "base_folder_names", [None] * len(self.long_recordings))[i]
                        if hasattr(self, "base_folder_names")
                        else None
                    ),
                    metadata=self.long_recordings[i].meta,
                    assume_from_number=self.assume_from_number,
                )

                fdsar_list.append(fdsar)

                # Log results
                total_spikes = sum(len(spikes) for spikes in spike_indices_per_channel)
                logging.info(
                    f"Recording {i + 1}/{len(self.long_recordings)}: Detected {total_spikes} spikes across {len(spike_indices_per_channel)} channels"
                )

            except Exception as e:
                logging.error(f"Error processing recording {i + 1}/{len(self.long_recordings)}: {e}")
                raise

        # Store results for later access
        self.frequency_domain_spike_analysis_results = fdsar_list

        logging.info(
            f"Completed frequency-domain spike detection. Total recordings processed: {len(fdsar_list)}"
        )
        return fdsar_list

    def _process_fragment_serial(
        self, idx, features, lan: core.LongRecordingAnalyzer, window_s, kwargs: dict
    ):
        row = self._process_fragment_metadata(idx, lan, window_s)
        row.update(self._process_fragment_features(idx, features, lan, kwargs))
        return row

    def _process_fragment_metadata(
        self, idx, lan: core.LongRecordingAnalyzer, window_s
    ):
        row = {}

        # Build session labels from LRO's DiscoveredFile metadata
        from neurodent.core.discovery import DiscoveredFile
        from neurodent import constants

        lro = lan.LongRecording
        item = getattr(lro, "item", None)

        animal = self.animal_id or "unknown"
        genotype = self.genotype or "Unknown"
        sex = self.sex or "Unknown"
        session = None

        if isinstance(item, DiscoveredFile) and item.metadata:
            meta = item.metadata
            animal = meta.get("animal", animal)
            session = meta.get("session")
            genotype = constants.ANIMAL_METADATA.get(animal, {}).get("gene", genotype)
            sex = constants.ANIMAL_METADATA.get(animal, {}).get("sex", sex)

        if session is None:
            try:
                session = lro.get_date_string()
            except (ValueError, AttributeError):
                session = "unknown"

        row["animalday"] = f"{animal} {genotype} {session}"
        row["animal"] = animal
        row["day"] = session
        row["genotype"] = genotype
        row["sex"] = sex
        row["duration"] = lan.LongRecording.get_dur_fragment(window_s, idx)
        row["endfile"] = lan.get_file_end(idx)

        frag_dt = lan.LongRecording.get_datetime_fragment(window_s, idx)
        row["timestamp"] = frag_dt
        row["isday"] = core.utils.is_day(frag_dt)

        return row

    def _process_fragment_features(
        self, idx, features, lan: core.LongRecordingAnalyzer, kwargs: dict
    ):
        row = {}
        for feat in features:
            func = getattr(lan, f"compute_{feat}")
            if callable(func):
                row[feat] = func(idx, **kwargs)
            else:
                raise AttributeError(f"Invalid function {func}")
        return row

    @classmethod
    def from_lros(
        cls,
        lros: list[core.LongRecordingOrganizer],
        animal_id: str,
        genotype: str = "Unknown",
        sex: str = "Unknown",
        assume_from_number: bool = False,
    ) -> "AnimalOrganizer":
        """
        Create an AnimalOrganizer from an existing list of LongRecordingOrganizer objects.

        This factory method bypasses the normal folder discovery logic and creates
        an AnimalOrganizer directly from pre-existing LROs. If multiple LROs share
        the same date, they will be automatically merged into a single LRO per unique date,
        matching the behavior of the normal __init__ path.

        Args:
            lros (list[LongRecordingOrganizer]): List of LRO instances to wrap.
            animal_id (str): Animal identifier for this organizer.
            genotype (str, optional): Genotype string. Defaults to "Unknown".
            sex (str, optional): Sex string (e.g. "Male", "Female"). Defaults to "Unknown".
            assume_from_number (bool, optional): Whether to assume channel aliases
                from numbers. Defaults to False.

        Returns:
            AnimalOrganizer: A new AnimalOrganizer instance wrapping the provided LROs
                (with duplicates merged).

        Raises:
            ValueError: If lros is empty, channel names are inconsistent, or LROs
                with the same date cannot be merged due to incompatible metadata.

        Note:
            Multiple LROs with the same date will be automatically merged in temporal
            order (sorted by median timestamp). This ensures proper handling of
            multi-session recordings consolidated via generate_wars.py.

        Example:
            >>> # After splitting a multi-animal recording across multiple sessions
            >>> all_lros = []
            >>> for session_ao in session_aos:
            ...     splits = session_ao.split({"AnimalA": ["Ch0", "Ch1"]})
            ...     all_lros.append(splits["AnimalA"])
            >>> # from_lros automatically merges LROs with same date
            >>> child_ao = AnimalOrganizer.from_lros(all_lros, animal_id="AnimalA")
        """
        if not lros:
            raise ValueError("Cannot create AnimalOrganizer from empty LRO list")

        # Create instance without calling __init__
        ao = object.__new__(cls)

        # Core attributes
        ao.anim_id = animal_id
        ao.animal_id = animal_id
        ao.genotype = genotype
        ao.sex = sex
        ao.assume_from_number = assume_from_number

        # Step 1: Group LROs by date
        date_to_lros = {}  # dict[str, list[tuple[int, LRO]]]

        for i, lro in enumerate(lros):
            try:
                date_str = lro.get_date_string()
            except ValueError as e:
                raise ValueError(
                    f"Could not determine date for LRO at index {i} (item: {lro.display_name}). "
                    f"Ensure LRO has valid timestamps via metadata or manual_datetimes. Error: {e}"
                )

            if date_str not in date_to_lros:
                date_to_lros[date_str] = []
            date_to_lros[date_str].append((i, lro))

        # Step 2: Merge LROs with duplicate dates
        merged_lros = []
        merged_animaldays = []

        for date_str in sorted(date_to_lros.keys()):  # Sort for deterministic ordering
            lro_group = date_to_lros[date_str]
            animalday = f"{animal_id} {genotype} {date_str}"

            if len(lro_group) == 1:
                # Single LRO for this date - use as-is
                _, lro = lro_group[0]
                merged_lros.append(lro)
                merged_animaldays.append(animalday)
                logging.info(f"Using single LRO for {animalday}")
            else:
                # Multiple LROs for same date - merge them
                logging.info(
                    f"Found {len(lro_group)} LROs for {animalday}. "
                    f"Merging into single LRO (mimicking normal __init__ behavior)."
                )

                # Sort by median time (same logic as normal __init__)
                lro_pairs = [(f"lro_{idx}", lro) for idx, lro in lro_group]
                sorted_pairs = cls._sort_lros_by_median_time_static(lro_pairs)

                # Merge all LROs into the first one (in temporal order)
                base_lro = sorted_pairs[0][1]
                original_idx = lro_group[0][0]
                logging.info(f"Base LRO: index {original_idx}")

                for i, (_, lro) in enumerate(sorted_pairs[1:], 1):
                    try:
                        logging.info(f"Merging LRO {i} into base LRO for {animalday}")
                        base_lro.merge(lro)
                    except ValueError as e:
                        # Provide detailed error for incompatible LROs
                        raise ValueError(
                            f"Cannot merge LROs for {animalday}: {e}\n"
                            f"All LROs with the same date must have compatible metadata "
                            f"(same channels, sampling rate, etc.)."
                        ) from e

                merged_lros.append(base_lro)
                merged_animaldays.append(animalday)
                logging.info(
                    f"Successfully merged {len(lro_group)} LROs for {animalday}"
                )

        # Step 3: Set merged LROs and animaldays
        ao.long_recordings = merged_lros
        ao.unique_animaldays = merged_animaldays
        ao.animaldays = (
            merged_animaldays.copy()
        )  # Create separate list for compatibility

        # Step 4: Validate and reconcile channel names across all merged LROs
        ao.channel_names = cls._validate_channel_names(merged_lros)

        # Step 5: CRITICAL VALIDATION - ensure no duplicates after merge
        if len(ao.long_recordings) != len(set(ao.unique_animaldays)):
            duplicate_dates = [
                date
                for date in ao.unique_animaldays
                if ao.unique_animaldays.count(date) > 1
            ]
            raise ValueError(
                f"CRITICAL ERROR: Duplicate animaldays detected after merge! "
                f"This indicates a logic error in the merge process. "
                f"Duplicates: {set(duplicate_dates)}\n"
                f"Expected {len(set(ao.unique_animaldays))} unique dates, "
                f"but got {len(ao.long_recordings)} LROs."
            )

        logging.info(
            f"Validated: {len(ao.long_recordings)} LROs match "
            f"{len(ao.unique_animaldays)} unique animaldays (no duplicates)"
        )

        # Step 6: Initialize default attributes for factory-created instances
        cls._init_factory_defaults(ao, animal_id, merged_lros)

        logging.info(
            f"Created AnimalOrganizer from {len(lros)} input LROs "
            f"(merged into {len(merged_lros)} unique dates) for animal '{animal_id}'"
        )

        return ao

    @staticmethod
    def _validate_channel_names(lros: list[core.LongRecordingOrganizer]) -> list[str]:
        """
        Validate that all LROs have consistent channel names.

        Compares abbreviated channel names (via ``parse_chname_to_abbrev``)
        so that cosmetic variants like ``L Barrel`` vs ``L Barrel Ctx`` are
        treated as equivalent. If raw names differ but abbreviations match,
        the mismatched LRO's channel names are renamed to match the reference
        LRO's raw names for downstream consistency.

        If channel names are the same but in different order, the first LRO's
        order is used as reference.

        Args:
            lros: List of LROs to validate.

        Returns:
            list[str]: The canonical channel names (from the first LRO).

        Raises:
            ValueError: If LROs have different abbreviated channel sets.
        """
        if not lros:
            return []

        first_names = lros[0].channel_names if lros[0].channel_names else []
        if not first_names:
            return []

        reference_abbrevs = abbreviate_channel_names(first_names, strict_matching=False)
        reference_set = set(reference_abbrevs)
        # Map abbreviation -> canonical raw name from first LRO
        abbrev_to_raw = dict(zip(reference_abbrevs, first_names))

        for i, lro in enumerate(lros[1:], start=1):
            current_names = lro.channel_names if lro.channel_names else []
            current_abbrevs = abbreviate_channel_names(current_names, strict_matching=False)
            current_set = set(current_abbrevs)

            if current_set != reference_set:
                missing = reference_set - current_set
                extra = current_set - reference_set
                raise ValueError(
                    f"LRO {i} has inconsistent channel names. "
                    f"Abbreviated missing: {missing}, Extra: {extra}"
                )

            # If raw names differ but abbreviations match, rename to reference
            if set(current_names) != set(first_names):
                renamed = [abbrev_to_raw[a] for a in current_abbrevs]
                logging.warning(
                    f"LRO {i} has variant channel names "
                    f"({current_names} vs {first_names}), "
                    f"renaming to reference names: {renamed}"
                )
                lro.channel_names = renamed

            # If same channels but different order, log a warning
            if lro.channel_names != first_names:
                logging.warning(
                    f"LRO {i} has channels in different order, using reference order"
                )

        return first_names

    @staticmethod
    def _init_factory_defaults(
        ao: "AnimalOrganizer", animal_id: str, lros: list[core.LongRecordingOrganizer]
    ) -> None:
        """
        Initialize attribute values for factory-created instances.

        Derives values from the provided LROs where possible instead of
        leaving attributes empty.

        Args:
            ao: The AnimalOrganizer instance to initialize.
            animal_id: The animal identifier.
            lros: The LROs to derive metadata from.
        """
        # Standard attributes
        ao.animal_file_match_pattern = [animal_id]
        ao.day_sep = None
        ao.read_mode = "base"

        # Internal cache - not derivable, but private
        ao._animalday_dicts = []
        ao._animalday_folder_groups = {}
        ao._processed_timestamps = None

        ao._init_containers()

    def split(
        self,
        groups: dict[str, list[str]],
        persist_base: Union[str, Path] = None,
        format: Literal["zarr", "binary"] = "zarr",
    ) -> dict[str, "AnimalOrganizer"]:
        """
        Split this multi-animal AnimalOrganizer into per-animal AnimalOrganizers.

        For each group (animal), this method:
        1. Iterates over all LROs in this AnimalOrganizer
        2. Calls LRO.split() on each to extract the specified channels
        3. Optionally persists each split LRO to disk
        4. Creates a new AnimalOrganizer for each group

        This enables processing of joint-animal recordings where multiple animals
        are recorded on different channels of the same files.

        Args:
            groups (dict[str, list[str]]): Dictionary mapping group names (animal IDs)
                to lists of channel names. Example:
                {"AnimalA": ["Ch0", "Ch1", "Ch2", "Ch3"],
                 "AnimalB": ["Ch4", "Ch5", "Ch6", "Ch7"]}
            persist_base (Union[str, Path], optional): Base directory for persisting
                split recordings. If None, LROs remain in-memory. Structure:
                persist_base/
                    AnimalA/
                        day1.zarr
                        day2.zarr
                    AnimalB/
                        ...
            format (Literal["zarr", "binary"], optional): Format for persisted
                recordings. Defaults to "zarr".

        Returns:
            dict[str, AnimalOrganizer]: Dictionary mapping group names to new
                AnimalOrganizer instances.

        Raises:
            ValueError: If requested channels are not found in recordings.

        Example:
            >>> ao = AnimalOrganizer("/path/to/joint_data", "combined")
            >>> splits = ao.split(
            ...     groups={"MouseA": ["Ch0", "Ch1"], "MouseB": ["Ch2", "Ch3"]},
            ...     persist_base="/output/split_data",
            ... )
            >>> war_a = splits["MouseA"].compute_windowed_analysis(["all"])
            >>> war_b = splits["MouseB"].compute_windowed_analysis(["all"])
        """
        if not self.long_recordings:
            raise ValueError("No recordings loaded to split")

        if persist_base is not None:
            persist_base = Path(persist_base)
            persist_base.mkdir(parents=True, exist_ok=True)

        result = {}

        for group_name, channels in groups.items():
            logging.info(
                f"Splitting group '{group_name}' with {len(channels)} channels "
                f"across {len(self.long_recordings)} days"
            )

            child_lros = []
            for i, lro in enumerate(self.long_recordings):
                # Split this day's LRO
                day_splits = lro.split({group_name: channels})
                child_lro = day_splits[group_name]

                # Persist if requested
                if persist_base is not None:
                    # Determine day folder name
                    day_name = lro.display_name or f"day{i}"

                    output_dir = persist_base / group_name / day_name
                    child_lro.persist(output_dir, format=format)
                    logging.debug(f"Persisted {group_name}/{day_name} to {output_dir}")

                child_lros.append(child_lro)

            # Create AnimalOrganizer from the split LROs
            child_ao = AnimalOrganizer.from_lros(
                lros=child_lros,
                animal_id=group_name,
                genotype=self.genotype,
                sex=self.sex,
                assume_from_number=self.assume_from_number,
            )

            result[group_name] = child_ao
            logging.info(
                f"Created AnimalOrganizer for '{group_name}' with "
                f"{len(child_lros)} days, {len(channels)} channels"
            )

        return result




__all__ = ["AnimalOrganizer"]
