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
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .frequency_domain_results import FrequencyDomainSpikeAnalysisResult

import dask
import dask.array as da
import mne
import numpy as np
import pandas as pd
from dask import delayed
from django.utils.text import slugify
from scipy.stats import zscore
from scipy.ndimage import binary_opening, binary_closing
from tqdm import tqdm


from .. import constants, core
from ..core import FragmentAnalyzer, get_temp_directory
from ..core.frequency_domain_spike_detection import FrequencyDomainSpikeDetector
from ..core.utils import filepath_to_index, parse_chname_to_abbrev


class AnimalFeatureParser:
    # REVIEW make this a utility function and refactor across codebase?
    def _average_feature(
        self, df: pd.DataFrame, colname: str, weightsname: str | None = "duration"
    ):
        column = df[colname]
        if weightsname is None or weightsname not in df.columns:
            weights = np.ones(column.size)
        else:
            weights = df[weightsname]
        colitem = column.iloc[0]
        weights = np.asarray(weights)

        match colname:  # NOTE refactor this to use constants
            case (
                "rms"
                | "ampvar"
                | "psdtotal"
                | "pcorr"
                | "zpcorr"
                | "nspike"
                | "logrms"
                | "logampvar"
                | "logpsdtotal"
                | "lognspike"
                | "psdslope"
            ):
                col_agg = np.array(column.tolist())
                avg = core.nanaverage(col_agg, axis=0, weights=weights)

            case (
                "cohere"
                | "zcohere"
                | "imcoh"
                | "zimcoh"
                | "psdband"
                | "psdfrac"
                | "logpsdband"
                | "logpsdfrac"
            ):
                keys = colitem.keys()
                avg = {}
                for k in keys:
                    v = np.array([d[k] for d in column])
                    avg[k] = core.nanaverage(v, axis=0, weights=weights)

            case "psd":
                coords = colitem[0]
                values = np.array([x[1] for x in column])
                avg = (coords, core.nanaverage(values, axis=0, weights=weights))

            case _:
                raise TypeError(f"Unrecognized type in column {colname}: {colitem}")

        return avg


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
        skip_sessions: list[str] = [],
        truncate: bool | int = False,
        assume_from_number: bool = False,
        lro_kwargs: dict = {},
        normalize_session: Optional[Callable[[str], str]] = None,
    ) -> None:
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

    def compute_bad_channels(
        self, lof_threshold: float = None, force_recompute: bool = False
    ):
        """Compute bad channels using LOF analysis for all recordings.

        Args:
            lof_threshold (float, optional): Threshold for determining bad channels from LOF scores.
                                           If None, only computes/loads scores without setting bad_channel_names.
            force_recompute (bool): Whether to recompute LOF scores even if they exist.
        """
        logging.info(
            f"Computing bad channels for {len(self.long_recordings)} recordings with threshold={lof_threshold}"
        )
        for i, lrec in self._iter_valid_recordings():
            logging.debug(
                f"Computing bad channels for recording {i}: {self.animaldays[i]}"
            )
            lrec.compute_bad_channels(
                lof_threshold=lof_threshold, force_recompute=force_recompute
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

        Raises:
            AttributeError: If a feature's ``compute_...()`` function was not implemented, this error will be raised.

        Returns:
            WindowAnalysisResult: A WindowAnalysisResult object containing extracted features for all recordings
        """
        features = _sanitize_feature_request(features, exclude)

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

        self.window_analysis_result = WindowAnalysisResult(
            self.features_df,
            self.animal_id,
            self.genotype,
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
        max_length: int = None,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
    ):
        """
        Compute frequency-domain spike detection on all long recordings.

        Args:
            detection_params (dict, optional): Detection parameters. Uses defaults if None.
            max_length (int, optional): Maximum length in samples to analyze per recording
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
                        max_length=max_length,
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
        session = None

        if isinstance(item, DiscoveredFile) and item.metadata:
            meta = item.metadata
            animal = meta.get("animal", animal)
            session = meta.get("session")
            genotype = constants.ANIMAL_METADATA.get(animal, {}).get("gene", genotype)

        if session is None:
            try:
                session = lro.get_date_string()
            except (ValueError, AttributeError):
                session = "unknown"

        row["animalday"] = f"{animal} {genotype} {session}"
        row["animal"] = animal
        row["day"] = session
        row["genotype"] = genotype
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

        def _abbreviate(names: list[str]) -> list[str]:
            result = []
            for n in names:
                try:
                    result.append(core.parse_chname_to_abbrev(n, strict_matching=False))
                except ValueError:
                    result.append(n)  # Fall back to raw name if parsing fails
            return result

        reference_abbrevs = _abbreviate(first_names)
        reference_set = set(reference_abbrevs)
        # Map abbreviation -> canonical raw name from first LRO
        abbrev_to_raw = dict(zip(reference_abbrevs, first_names))

        for i, lro in enumerate(lros[1:], start=1):
            current_names = lro.channel_names if lro.channel_names else []
            current_abbrevs = _abbreviate(current_names)
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
                assume_from_number=self.assume_from_number,
            )

            result[group_name] = child_ao
            logging.info(
                f"Created AnimalOrganizer for '{group_name}' with "
                f"{len(child_lros)} days, {len(channels)} channels"
            )

        return result


def _sanitize_feature_request(features: list[str], exclude: list[str] = []):
    """
    Sanitizes a list of requested features for WindowAnalysisResult

    Args:
        features (list[str]): List of features to include. If "all", include all features in constants.FEATURES except for exclude.
        exclude (list[str], optional): List of features to exclude. Defaults to [].

    Returns:
        list[str]: Sanitized list of features.
    """
    if isinstance(features, str):
        features = [features]
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
        channel_names: list[str] = None,
        assume_from_number=False,
        bad_channels_dict: dict[str, list[str]] = {},
        suppress_short_interval_error=False,
        lof_scores_dict: dict[str, dict] = {},
    ) -> None:
        self.result = result
        self.animal_id = animal_id
        self.genotype = genotype
        self.channel_names = channel_names
        self.assume_from_number = assume_from_number
        self.bad_channels_dict = bad_channels_dict.copy()
        self.suppress_short_interval_error = suppress_short_interval_error
        self.lof_scores_dict = lof_scores_dict

        self.__update_instance_vars()

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
            channel_names=(
                self.channel_names.copy() if self.channel_names is not None else None
            ),
            assume_from_number=self.assume_from_number,
            bad_channels_dict=copy.deepcopy(self.bad_channels_dict),
            suppress_short_interval_error=self.suppress_short_interval_error,
            lof_scores_dict=copy.deepcopy(self.lof_scores_dict),
        )

    def __update_instance_vars(self):
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
            match feature:
                case _ if (
                    feature in constants.LINEAR_FEATURES + constants.BAND_FEATURES
                ):
                    if feature in constants.BAND_FEATURES:
                        df_bands = pd.DataFrame(result[feature].tolist())
                        vals = np.array(df_bands.values.tolist())
                        vals = vals.transpose((0, 2, 1))
                        keys = df_bands.keys()
                    else:
                        vals = np.array(result[feature].tolist())

                    new_vals = np.full(
                        (vals.shape[0], len(target_channels), *vals.shape[2:]), np.nan
                    )  # dubious

                    for i, ch in enumerate(channel_names):
                        if ch in channel_map:
                            new_vals[:, channel_map[ch]] = vals[:, i]

                    if feature in constants.BAND_FEATURES:
                        new_vals = new_vals.transpose((0, 2, 1))
                        result[feature] = [dict(zip(keys, vals)) for vals in new_vals]
                    else:
                        result[feature] = [list(x) for x in new_vals]

                case _ if feature in constants.MATRIX_FEATURES:
                    if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
                        df_bands = pd.DataFrame(result[feature].tolist())
                        vals = np.array(df_bands.values.tolist())
                        keys = df_bands.keys()
                    else:
                        vals = np.array(result[feature].tolist())

                    logging.debug(f"vals.shape: {vals.shape}")
                    new_shape = list(vals.shape[:-2]) + [
                        len(target_channels),
                        len(target_channels),
                    ]
                    new_vals = np.full(new_shape, np.nan)

                    # Map original channels to target channels
                    for i, ch1 in enumerate(channel_names):
                        if ch1 in channel_map:
                            for j, ch2 in enumerate(channel_names):
                                if ch2 in channel_map:
                                    new_vals[
                                        ..., channel_map[ch1], channel_map[ch2]
                                    ] = vals[..., i, j]

                    if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
                        result[feature] = [dict(zip(keys, vals)) for vals in new_vals]
                    else:
                        result[feature] = [list(x) for x in new_vals]

                case _ if feature in constants.HIST_FEATURES:
                    coords = np.array([x[0] for x in result[feature].tolist()])
                    vals = np.array([x[1] for x in result[feature].tolist()])
                    new_vals = np.full(
                        (*vals.shape[0:-1], len(target_channels)), np.nan
                    )

                    for i, ch in enumerate(channel_names):
                        if ch in channel_map:
                            new_vals[:, ..., channel_map[ch]] = vals[:, ..., i]

                    result[feature] = [
                        (coords[i], new_vals[i]) for i in range(len(coords))
                    ]

                case _:
                    raise ValueError(f"Invalid feature: {feature}")

        if inplace:
            self.result = result

            logging.debug(f"Old channel names: {self.channel_names}")
            self.channel_names = target_channels
            logging.debug(f"New channel names: {self.channel_names}")

            logging.debug(f"Old channel abbreviations: {self.channel_abbrevs}")
            self.__update_instance_vars()
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
        info.append(
            f"channel_names: {', '.join(self.channel_names) if self.channel_names else 'None'}"
        )

        return "\n".join(info)

    def get_result(
        self, features: list[str], exclude: list[str] = [], allow_missing=False
    ):
        """Get windowed analysis result dataframe, with helpful filters

        Args:
            features (list[str]): List of features to get from result
            exclude (list[str], optional): List of features to exclude from result; will override the features parameter. Defaults to [].
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
        features: list[str],
        exclude: list[str] = [],
        df: pd.DataFrame = None,
        groupby="animalday",
    ):
        """Group result and average within groups. Preserves data structure and shape for each feature.

        Args:
            features (list[str]): List of features to get from result
            exclude (list[str], optional): List of features to exclude from result. Will override the features parameter. Defaults to [].
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
        features: list[str],
        exclude: list[str] = [],
        df: pd.DataFrame = None,
        groupby="animalday",
    ):
        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df
        return result_win.groupby(groupby), result_win.columns

    def get_grouprows_result(
        self,
        features: list[str],
        exclude: list[str] = [],
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
        features: list[str],
        exclude: list[str] = [],
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
            features (list[str]): List of feature names to extract and average.
                Can include any combination of linear, band, or matrix features.
            exclude (list[str], optional): List of features to exclude. Defaults to [].
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

        Handles two types of features:
        - Vector features (1D arrays): Average across channels
        - Matrix features (2D arrays): Average upper triangle (excluding diagonal)

        Args:
            df: DataFrame with features as columns
            features: List of feature column names to average

        Returns:
            DataFrame with averaged features replacing original arrays
        """
        import numpy as np

        for feature in features:
            if feature not in df.columns:
                continue

            first_element = df[feature].iloc[0]

            if isinstance(first_element, (np.ndarray, list)):
                if isinstance(first_element, list):
                    first_element = np.array(first_element)

                if first_element.ndim == 1:
                    # Vector features: Mean across channels
                    # We use a robust approach to handle potential list formats or shape drifts
                    feature_values = df[feature].values

                    # Check if we can use vectorized approach (faster)
                    try:
                        # This will fail efficiently if shapes don't match
                        feature_arrays = np.vstack(feature_values)
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
                    import logging

                    logger = logging.getLogger(__name__)

                    feature_avg = []
                    for matrix in df[feature].values:
                        if isinstance(matrix, list):
                            matrix = np.array(matrix)

                        # Validate matrix shape
                        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                            logger.warning(
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
        return WindowAnalysisResult(
            filtered_result,
            self.animal_id,
            self.genotype,
            self.channel_names,
            self.assume_from_number,
            self.bad_channels_dict.copy(),
            self.suppress_short_interval_error,
            self.lof_scores_dict.copy(),
        )

    def _create_filtered_copy(self, filter_mask: np.ndarray) -> "WindowAnalysisResult":
        """Create a new WindowAnalysisResult with the filter applied.

        Args:
            filter_mask (np.ndarray): Boolean mask of shape (n_windows, n_channels)

        Returns:
            WindowAnalysisResult: New instance with filter applied
        """
        filtered_result = self._apply_filter(filter_mask)
        return WindowAnalysisResult(
            filtered_result,
            self.animal_id,
            self.genotype,
            self.channel_names,
            self.assume_from_number,
            self.bad_channels_dict.copy(),
            self.suppress_short_interval_error,
            self.lof_scores_dict.copy(),
        )

    def filter_logrms_range(self, z_range: float = 3) -> "WindowAnalysisResult":
        """Filter based on log(rms) z-score range.

        Args:
            z_range (float): Z-score range threshold. Defaults to 3.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_logrms_range(z_range=z_range)
        return self._create_filtered_copy(mask)

    def filter_high_rms(self, max_rms: float = 500) -> "WindowAnalysisResult":
        """Filter out windows with RMS above threshold.

        Args:
            max_rms (float): Maximum RMS threshold. Defaults to 500.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_high_rms(max_rms=max_rms)
        return self._create_filtered_copy(mask)

    def filter_low_rms(self, min_rms: float = 50) -> "WindowAnalysisResult":
        """Filter out windows with RMS below threshold.

        Args:
            min_rms (float): Minimum RMS threshold. Defaults to 50.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_low_rms(min_rms=min_rms)
        return self._create_filtered_copy(mask)

    def filter_high_beta(self, max_beta_prop: float = 0.4) -> "WindowAnalysisResult":
        """Filter out windows with high beta power.

        Args:
            max_beta_prop (float): Maximum beta power proportion. Defaults to 0.4.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_high_beta(max_beta_prop=max_beta_prop)
        return self._create_filtered_copy(mask)

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
        return self._create_filtered_copy(mask)

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
        return self._create_filtered_copy(mask)

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
            match feat:  # NOTE refactor this to use constants
                case (
                    "rms"
                    | "ampvar"
                    | "psdtotal"
                    | "nspike"
                    | "logrms"
                    | "logampvar"
                    | "logpsdtotal"
                    | "lognspike"
                ):
                    vals = np.array(result[feat].tolist())
                    # Convert to float to allow NaN assignment for integer features
                    if vals.dtype.kind in ("i", "u"):  # integer types
                        vals = vals.astype(float)
                    vals[~filter_tfs] = np.nan
                    result[feat] = vals.tolist()
                case "psd":
                    # FIXME The sampling rates have changed between computation passes so WARs have different shapes.
                    # Add a check for same sampling frequency, other war-relevant properties etc.
                    # The logging lines below should be removed at some point, but I'll keep it this way for now
                    logging.info(
                        f"set([x[0].shape for x in result[feat].tolist()]) = {list(set([x[0].shape for x in result[feat].tolist()]))}"
                    )
                    logging.info(
                        f"set([x[1].shape for x in result[feat].tolist()]) = {list(set([x[1].shape for x in result[feat].tolist()]))}"
                    )
                    coords = np.array([x[0] for x in result[feat].tolist()])
                    vals = np.array([x[1] for x in result[feat].tolist()])
                    mask = np.broadcast_to(filter_tfs[:, np.newaxis, :], vals.shape)
                    vals[~mask] = np.nan
                    outs = [(c, vals[i, :, :]) for i, c in enumerate(coords)]
                    result[feat] = outs
                case "psdband" | "psdfrac" | "logpsdband" | "logpsdfrac":
                    vals = pd.DataFrame(result[feat].tolist())
                    for colname in vals.columns:
                        v = np.array(vals[colname].tolist())
                        v[~filter_tfs] = np.nan
                        vals[colname] = v.tolist()
                    result[feat] = vals.to_dict("records")
                case "psdslope":
                    vals = np.array(result[feat].tolist())
                    mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], vals.shape)
                    vals[~mask] = np.nan
                    # vals = [list(map(tuple, x)) for x in vals.tolist()]
                    result[feat] = vals.tolist()
                case "cohere" | "zcohere" | "imcoh" | "zimcoh":
                    vals = pd.DataFrame(result[feat].tolist())
                    shape = np.array(vals.iloc[:, 0].tolist()).shape
                    mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], shape)
                    for colname in vals.columns:
                        v = np.array(vals[colname].tolist())
                        v[~mask] = np.nan
                        v[~mask.transpose(0, 2, 1)] = np.nan
                        vals[colname] = v.tolist()
                    result[feat] = vals.to_dict("records")
                case "pcorr" | "zpcorr":
                    vals = np.array(result[feat].tolist())
                    mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], vals.shape)
                    vals[~mask] = np.nan
                    vals[~mask.transpose(0, 2, 1)] = np.nan
                    result[feat] = vals.tolist()
                case _:
                    raise ValueError(f"Unknown feature to filter {feat}")
        return result

    def save_pickle_and_json(
        self,
        folder: str | Path,
        make_folder=True,
        filename: str = None,
        slugify_filename=False,
        save_abbrevs_as_chnames=False,
    ):
        """Archive window analysis result into the folder specified, as a pickle and json file.

        Args:
            folder (str | Path): Destination folder to save results to
            make_folder (bool, optional): If True, create the folder if it doesn't exist. Defaults to True.
            filename (str, optional): Name of the file to save. Defaults to "war".
            slugify_filename (bool, optional): If True, slugify the filename (replace special characters). Defaults to False.
            save_abbrevs_as_chnames (bool, optional): If True, save the channel abbreviations as the channel names in the json file. Defaults to False.
        """
        folder = Path(folder)
        if make_folder:
            folder.mkdir(parents=True, exist_ok=True)

        filename = "war" if filename is None else filename
        filename = slugify(filename) if slugify_filename else filename

        filepath = str(folder / filename)

        self.result.to_pickle(filepath + ".pkl")
        logging.info(f"Saved WAR to {filepath + '.pkl'}")

        json_dict = {
            "animal_id": self.animal_id,
            "genotype": self.genotype,
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

        constant_cols = ["animal", "day", "genotype"]
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
        self.__update_instance_vars()

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

        self.__update_instance_vars()


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
