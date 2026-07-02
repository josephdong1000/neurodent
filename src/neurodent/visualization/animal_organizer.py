"""Multi-session recording organization for a single animal.

``AnimalOrganizer`` discovers and loads recording files, groups them into
sessions, manages ``LongRecordingOrganizer`` instances and their timeline, and
orchestrates windowed feature analysis and spike detection — producing
:class:`~neurodent.visualization.window_analysis_result.WindowAnalysisResult`
objects.

Split out of the former monolithic ``results.py`` (issue #134).
"""

import copy
import fnmatch
import logging
import re
import time
import warnings
import dateutil.parser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import pandas as pd
from tqdm import tqdm

from .. import constants, core
from ..core.utils import resolve_channels, resolve_channel
from .pipeline import AnalysisPipeline


class AnimalOrganizer:
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
        skip_sessions: list[str] = [],
        truncate: bool | int = False,
        lro_kwargs: dict = {},
        normalize_session: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.pattern = pattern
        self.animal_id = animal_id
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
            constants.ANIMAL_METADATA.get(self.animal_id, {}).get("genotype", "Unknown")
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

    def _get_item_key(self, item):
        """Return a unique key for the item, suitable for dict lookups across sessions.

        Unlike _get_item_name() which returns only the filename (e.g. 'file-0.bin'),
        this returns the full path, ensuring items with the same filename in different
        session directories get distinct keys.
        """
        from ..core.discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            paths = item.get_path_list()
            return str(paths[0]) if paths else "unknown"
        if isinstance(item, (list, tuple)):
            return str(item[0])
        return str(item)

    @staticmethod
    def _validate_timestamp_ordering(timestamp_dict):
        """Validate that computed timestamps have no duplicates.

        Catches key collisions (duplicate keys silently overwriting produce
        equal timestamps for different items) and misconfigured manual_datetimes
        that assign the same start time to multiple items.

        Only checks datetime values; lists/functions are passed through unvalidated.
        """
        datetime_items = {
            k: v for k, v in timestamp_dict.items()
            if isinstance(v, datetime)
        }
        if len(datetime_items) < 2:
            return
        sorted_items = sorted(datetime_items.items(), key=lambda x: x[1])
        for i in range(1, len(sorted_items)):
            prev_key, prev_ts = sorted_items[i - 1]
            curr_key, curr_ts = sorted_items[i]
            if curr_ts <= prev_ts:
                raise ValueError(
                    f"Timestamp collision: {prev_key} ({prev_ts}) and "
                    f"{curr_key} ({curr_ts}) have equal or overlapping timestamps. "
                    f"This may indicate duplicate files mapping to the same "
                    f"{{index}} in the file pattern, or a manual_datetimes "
                    f"configuration that assigns the same start time to "
                    f"multiple items."
                )

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
            # Resolve each element so JSON string lists (e.g. a per-session list
            # of ISO start times) parse, while datetime objects pass through.
            resolved = []
            for el in input_spec:
                if isinstance(el, datetime):
                    resolved.append(el.replace(tzinfo=None))
                elif isinstance(el, str):
                    # dateutil raises ValueError on an unparseable string.
                    resolved.append(dateutil.parser.parse(el).replace(tzinfo=None))
                else:
                    raise TypeError(
                        "All items in timestamp list must be datetime objects or "
                        f"parseable date strings, got: {[type(dt) for dt in input_spec]}"
                    )
            return resolved

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

    def _items_have_index(self, items):
        """Check if items carry {index} metadata."""
        return (
            items
            and hasattr(items[0], "metadata")
            and "index" in getattr(items[0], "metadata", {})
        )

    def _session_sort_key(self, items):
        """Return sort-key function: use {index} metadata if available, else filename."""
        from ..core.discovery import _natural_sort_key

        if self._items_have_index(items):
            return lambda f: _natural_sort_key(f.metadata["index"])
        return lambda f: _natural_sort_key(self._get_item_name(f))

    def _compute_global_timeline(
        self,
        base_datetime,
        animalday_to_items: dict,
        base_lro_kwargs: dict,
        original_manual_datetimes=None,
    ) -> tuple[dict, datetime]:
        """Compute per-item timestamps anchored at *base_datetime*.

        Returns a tuple of ``(timeline, end_dt)`` where:

        - ``timeline`` maps each item's key to its computed start datetime.
        - ``end_dt`` is the end datetime of the last file in the chain
          (equals ``base_datetime + sum(durations)`` when datetimes are
          start times, or ``base_datetime`` when datetimes are end times).
          ``end_dt`` is what dict-with-null forward cumulation in
          :meth:`_process_manual_datetimes` uses to chain successive
          sessions.
        """
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
                    sorted_items = sorted(items, key=self._session_sort_key(items))
                    ordered_items.extend(sorted_items)
            else:
                for animalday in sorted(animalday_to_items.keys(), key=_natural_sort_key):
                    items = animalday_to_items[animalday]
                    sorted_items = sorted(items, key=self._session_sort_key(items))
                    ordered_items.extend(sorted_items)
        else:
            for animalday in sorted(animalday_to_items.keys(), key=_natural_sort_key):
                items = animalday_to_items[animalday]
                if self._items_have_index(items):
                    sorted_items = sorted(items, key=self._session_sort_key(items))
                    ordered_items.extend(sorted_items)
                elif len(items) > 1:
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

        # Filter out zero-duration items (empty/corrupt files) from the
        # timeline.  These items remain in _animalday_folder_groups and will
        # be handled by _filter_zero_sample_lros() during LRO creation.
        zero_items = [
            item for item in ordered_items if item_durations.get(item, 0) == 0
        ]
        if zero_items:
            for item in zero_items:
                logging.warning(
                    f"Skipping zero-duration item '{self._get_item_name(item)}' "
                    f"from timeline computation (empty or corrupt file)"
                )
            ordered_items = [
                item for item in ordered_items if item not in zero_items
            ]
            for item in zero_items:
                item_durations.pop(item, None)

        datetimes_are_start = base_lro_kwargs.get("datetimes_are_start", True)
        result = {}

        if datetimes_are_start:
            current_start_time = base_datetime
            for item in ordered_items:
                item_key = self._get_item_key(item)
                result[item_key] = current_start_time
                current_start_time = current_start_time + timedelta(
                    seconds=item_durations[item]
                )
            # End of last file (exclusive) — used by dict-with-null forward cumulation
            # in _process_manual_datetimes to chain successive sessions.
            end_dt = current_start_time
        else:
            current_end_time = base_datetime
            for item in reversed(ordered_items):
                item_key = self._get_item_key(item)
                duration = item_durations[item]
                start_time = current_end_time - timedelta(seconds=duration)
                result[item_key] = start_time
                current_end_time = start_time
            # base_datetime IS the end of the last file when datetimes are end times.
            end_dt = base_datetime

        # Validate monotonicity of computed timestamps in item order
        for i in range(1, len(ordered_items)):
            prev_key = self._get_item_key(ordered_items[i - 1])
            curr_key = self._get_item_key(ordered_items[i])
            if result[curr_key] < result[prev_key]:
                raise ValueError(
                    f"Timeline computation produced non-monotonic timestamps: "
                    f"{prev_key} ({result[prev_key]}) > {curr_key} ({result[curr_key]})"
                )

        total_duration = sum(item_durations.values())
        logging.info(
            f"Timeline computed: {len(result)} items, total duration {total_duration:.1f}s"
        )
        return result, end_dt

    def _assign_session_list(
        self, sess_key, sess_ts, sess_items, base_lro_kwargs: dict
    ) -> tuple[dict, datetime]:
        """Assign explicit per-file start datetimes to one session's items.

        Used by :meth:`_process_manual_datetimes` when a session's value in the
        ``manual_datetime`` dict is a list (one start per file, no cumulation) —
        e.g. to encode an internal gap from a missing file.

        Args:
            sess_key: The session key (for error messages).
            sess_ts: List of per-file start times (datetime or ISO string), one
                per item in the session, in ``{index}`` / natural-sort order.
            sess_items: The session's discovered items.
            base_lro_kwargs: Kwargs for loading an item to estimate its duration.

        Returns:
            tuple[dict, datetime]: ``(timeline, anchor_end_dt)`` where ``timeline``
            maps each item key to its explicit start datetime, and ``anchor_end_dt``
            is the end of the last file (last start + its duration) — used by a
            following ``null`` session to cumulate forward.

        Raises:
            ValueError: If the list length != number of items in the session.
        """
        # Order items the same way the merge/timeline does (by {index} when
        # present). The list is positional, so this MUST match the merge order
        # in _create_long_recordings.
        sorted_items = sorted(sess_items, key=self._session_sort_key(sess_items))
        if len(sess_ts) != len(sorted_items):
            raise ValueError(
                f"manual_datetime list for session '{sess_key}' has "
                f"{len(sess_ts)} entries but the session has "
                f"{len(sorted_items)} item(s): "
                f"{[self._get_item_name(f) for f in sorted_items]}. "
                f"Provide exactly one datetime per file, in index order."
            )

        context_path = self._get_context_path(sorted_items[0])
        resolved = self._resolve_timestamp_input(list(sess_ts), context_path)

        timeline = {}
        for item, start_dt in zip(sorted_items, resolved):
            timeline[self._get_item_key(item)] = start_dt

        # anchor_end_dt = last file's explicit start + its duration, so a
        # following null session cumulates from the true end of this session.
        last_item = sorted_items[-1]
        last_start = resolved[-1]
        _kw = base_lro_kwargs.copy()
        _kw["manual_datetimes"] = last_start
        temp_lro = core.LongRecordingOrganizer(last_item, **_kw)
        duration = (
            temp_lro.LongRecording.get_duration()
            if hasattr(temp_lro, "LongRecording") and temp_lro.LongRecording
            else 0.0
        )
        anchor_end_dt = last_start + timedelta(seconds=duration)
        return timeline, anchor_end_dt

    def _process_manual_datetimes(
        self, manual_datetimes, animalday_to_items: dict, base_lro_kwargs: dict
    ) -> dict:
        """Resolve ``manual_datetimes`` into a per-item ``{item_key: start_datetime}`` map.

        Supported forms (per animal):

        - **dict keyed by session** — each value is one of:
          a scalar start (``datetime``/ISO string) anchoring that session;
          ``null`` to cumulate forward from the previous session's end (contiguous);
          or a **list** of one start per file in that session (explicit, no
          cumulation — use to encode an internal gap from a missing file).
          Scalar/list values act as resets; nulls before the first explicit
          anchor are backfilled by walking backward.
        - **dict keyed by item/filename** — value per file (scalar or per-file list).
        - **flat list** — one datetime per discovered file across all sessions.
        - **single datetime/string** — global start for the whole animal.

        Lists and nulls require ``datetimes_are_start=True``.
        """
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
                    out[self._get_item_key(item)] = self._resolve_timestamp_input(
                        manual_datetimes[fname], context_path
                    )
                self._validate_timestamp_ordering(out)
                return out

            elif has_session_keys:
                logging.info(
                    f"manual_datetimes keys match sessions for {self.animal_id}. "
                    "Computing per-session timelines."
                )
                # Dict-form session ordering uses *dict insertion order*
                # (Python 3.7+ contract), NOT _natural_sort_key on
                # animalday_to_items.  Dict keys are the single canonical
                # chronological source — see plan: "Allow null in
                # manual_datetime dict (cumulate forward from prior anchor)".
                missing_sessions = [
                    k for k in animalday_to_items
                    if k not in manual_datetimes
                ]
                if missing_sessions:
                    raise ValueError(
                        f"Missing entries in manual_datetimes for sessions: "
                        f"{missing_sessions}. Every discovered session must "
                        f"be in the manual_datetime dict (use null to cumulate "
                        f"forward from the previous anchor)."
                    )
                extra_keys = [
                    k for k in manual_datetimes
                    if k not in animalday_to_items
                ]
                if extra_keys:
                    raise ValueError(
                        f"manual_datetime has keys not in discovered sessions "
                        f"for '{self.animal_id}': {extra_keys}. Discovered "
                        f"sessions: {list(animalday_to_items.keys())}."
                    )
                # Reject null in non-start-time mode: forward cumulation only
                # makes sense when timestamps are interpreted as starts.
                datetimes_are_start = base_lro_kwargs.get(
                    "datetimes_are_start", True
                )
                if not datetimes_are_start and any(
                    v is None for v in manual_datetimes.values()
                ):
                    raise ValueError(
                        f"manual_datetime contains null values for "
                        f"'{self.animal_id}', but datetimes_are_start is "
                        f"False. Null (cumulate forward) is only supported "
                        f"when timestamps are interpreted as start times."
                    )
                # Reject per-session lists (explicit per-file starts) in
                # non-start-time mode — they only make sense as start times.
                list_sessions = [
                    k for k, v in manual_datetimes.items() if isinstance(v, list)
                ]
                if not datetimes_are_start and list_sessions:
                    raise ValueError(
                        f"manual_datetime contains list values for "
                        f"'{self.animal_id}' (sessions {list_sessions}), but "
                        f"datetimes_are_start is False. Per-file explicit start "
                        f"lists are only supported when timestamps are "
                        f"interpreted as start times."
                    )

                # Find the first explicit anchor — used to backfill any
                # null sessions BEFORE it (working backward, assuming
                # contiguous recording).  Sessions at and after the first
                # explicit anchor use forward cumulation (current behaviour);
                # subsequent explicit anchors still act as resets.
                ordered_keys = list(manual_datetimes.keys())
                first_explicit_idx = next(
                    (i for i, v in enumerate(manual_datetimes.values())
                     if v is not None),
                    None,
                )
                if first_explicit_idx is None:
                    raise ValueError(
                        f"manual_datetime for '{self.animal_id}' has no "
                        f"explicit anchor — every session is null. At least "
                        f"one session must have a known datetime."
                    )

                out = {}

                # Backfill prefix nulls (sessions before first_explicit_idx).
                # Walk backward from the first explicit anchor's start time.
                if first_explicit_idx > 0:
                    anchor_key = ordered_keys[first_explicit_idx]
                    anchor_items = animalday_to_items[anchor_key]
                    anchor_context = self._get_context_path(anchor_items[0])
                    anchor_resolved = self._resolve_timestamp_input(
                        manual_datetimes[anchor_key], anchor_context
                    )
                    # A list anchor (explicit per-file starts) resolves to a
                    # list; the session's start is its earliest file start.
                    next_session_start = (
                        min(anchor_resolved)
                        if isinstance(anchor_resolved, list)
                        else anchor_resolved
                    )
                    # Iterate prefix sessions in REVERSE so each session's
                    # end_dt equals the next (later) session's start_dt —
                    # this is the contiguous-recording assumption made
                    # explicit.
                    backward_kwargs = dict(base_lro_kwargs)
                    backward_kwargs["datetimes_are_start"] = False
                    for prefix_idx in range(first_explicit_idx - 1, -1, -1):
                        prefix_key = ordered_keys[prefix_idx]
                        prefix_items = animalday_to_items[prefix_key]
                        sess_item_dict = {
                            self._get_item_key(f): [f] for f in prefix_items
                        }
                        # next_session_start is treated as the END of this
                        # prefix session's last file (contiguous).
                        sess_timeline, _ = self._compute_global_timeline(
                            next_session_start,
                            sess_item_dict,
                            backward_kwargs,
                            original_manual_datetimes=next_session_start,
                        )
                        out.update(sess_timeline)
                        # This session's start = min of computed file starts.
                        next_session_start = min(sess_timeline.values())

                # Forward cumulation from the first explicit anchor onward.
                # Subsequent explicit anchors reset the running chain (no
                # silent reconciliation — user's explicit values are
                # authoritative).
                anchor_end_dt = None
                for idx in range(first_explicit_idx, len(ordered_keys)):
                    sess_key = ordered_keys[idx]
                    sess_ts = manual_datetimes[sess_key]
                    sess_items = animalday_to_items[sess_key]

                    # A list value gives explicit per-file start times for this
                    # session (no cumulation) — e.g. to encode an internal gap
                    # from a missing file.  Handled self-contained; bypasses
                    # _compute_global_timeline's anchor-cumulation.
                    if isinstance(sess_ts, list):
                        sess_timeline, anchor_end_dt = self._assign_session_list(
                            sess_key, sess_ts, sess_items, base_lro_kwargs
                        )
                        out.update(sess_timeline)
                        continue

                    if sess_ts is None:
                        # anchor_end_dt is guaranteed non-None here because
                        # idx >= first_explicit_idx and the very first
                        # iteration sets it.
                        resolved_dt = anchor_end_dt
                        sess_input = anchor_end_dt
                    else:
                        context_path = self._get_context_path(sess_items[0])
                        resolved_dt = self._resolve_timestamp_input(
                            sess_ts, context_path
                        )
                        sess_input = sess_ts
                    sess_item_dict = {
                        self._get_item_key(f): [f] for f in sess_items
                    }
                    sess_timeline, anchor_end_dt = self._compute_global_timeline(
                        resolved_dt,
                        sess_item_dict,
                        base_lro_kwargs,
                        original_manual_datetimes=sess_input,
                    )
                    out.update(sess_timeline)
                self._validate_timestamp_ordering(out)
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
                timeline, _end_dt = self._compute_global_timeline(
                    start_dt,
                    animalday_to_items,
                    base_lro_kwargs,
                    original_manual_datetimes=manual_datetimes,
                )
                return timeline
            warnings.warn(
                "String timestamp resolved to non-scalar. Falling back to default processing."
            )

        else:
            logging.info("Processing manual datetimes input for all items")
            out = {}
            for animalday, items in animalday_to_items.items():
                for item in items:
                    context_path = self._get_context_path(item)
                    out[self._get_item_key(item)] = self._resolve_timestamp_input(
                        manual_datetimes, context_path
                    )
            self._validate_timestamp_ordering(out)
            return out

    def _create_long_recordings(self, lro_kwargs: dict):
        """Create LongRecordingOrganizer instances for each unique animalday."""
        self.long_recordings: list[core.LongRecordingOrganizer] = []
        skipped_animaldays: list[str] = []
        for animalday, items in self._animalday_folder_groups.items():
            kwargs = lro_kwargs.copy()
            if getattr(self, "_processed_timestamps", None) is not None:
                # _processed_timestamps is keyed by full item path, not animalday
                if len(items) == 1:
                    item_key = self._get_item_key(items[0])
                    if item_key in self._processed_timestamps:
                        kwargs["manual_datetimes"] = self._processed_timestamps[item_key]
                        kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                        logging.debug(
                            f"Using processed timestamp for {item_key}: {kwargs['manual_datetimes']}"
                        )
                else:
                    # For multi-item animaldays, collect per-item timestamps as a list
                    item_timestamps = []
                    for item in items:
                        item_key = self._get_item_key(item)
                        if item_key in self._processed_timestamps:
                            item_timestamps.append(self._processed_timestamps[item_key])
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
                    # Remove session-level timestamp list; replaced per-item
                    # below.  Items missing from _processed_timestamps (e.g.,
                    # zero-byte files skipped from timeline) must not inherit
                    # the session list, which would cause a length mismatch.
                    individual_kwargs.pop("manual_datetimes", None)
                    individual_kwargs.pop("datetimes_are_start", None)
                    # Distribute per-item timestamp so each LRO gets its own
                    if getattr(self, "_processed_timestamps", None) is not None:
                        item_key = self._get_item_key(item)
                        if item_key in self._processed_timestamps:
                            individual_kwargs["manual_datetimes"] = (
                                self._processed_timestamps[item_key]
                            )
                            individual_kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                    individual_lro = core.LongRecordingOrganizer(
                        item, **individual_kwargs
                    )
                    item_lro_pairs.append((item, individual_lro))

                # Filter out 0-sample LROs (from failed/empty file pairs) before
                # merging. A 0-sample base LRO causes merge metadata failures, and
                # downstream _iter_valid_recordings() cannot recover from that.
                valid_pairs, skipped_names = self._filter_zero_sample_lros(
                    item_lro_pairs, self._get_item_name
                )
                if skipped_names:
                    logging.warning(
                        f"Skipping {len(skipped_names)} 0-sample LRO(s) for "
                        f"'{animalday}' before merge: {skipped_names}"
                    )
                if not valid_pairs:
                    logging.error(
                        f"Skipping animalday '{animalday}' entirely: all {len(item_lro_pairs)} "
                        f"file(s) produced 0-sample LROs. Each file may have been corrupt, "
                        f"empty, or failed during loading (check earlier warnings above for "
                        f"root causes per file). Skipped files: {skipped_names}"
                    )
                    skipped_animaldays.append(animalday)
                    continue
                item_lro_pairs = valid_pairs

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

        if skipped_animaldays:
            self.unique_animaldays = [
                ad for ad in self.unique_animaldays if ad not in skipped_animaldays
            ]
            self.animaldays = self.unique_animaldays

        if not self.long_recordings:
            raise RuntimeError(
                f"No recordings were loaded for this animal. "
                f"All {len(skipped_animaldays)} animalday(s) were skipped because every "
                f"file produced a 0-sample LRO. This usually indicates a misconfiguration "
                f"(wrong file pattern, wrong data root, or corrupt data). "
                f"Skipped animaldays: {skipped_animaldays}. "
                f"Check the warnings above for per-file root causes."
            )

        # It is possible for long_recordings to contain only 0-sample placeholder LROs.
        # In that case, _iter_valid_recordings() will yield nothing and downstream analysis
        # (e.g. concatenating results) will fail with a less informative error.
        # Guard against this by raising early if there are no valid (nonzero-sample) LROs.
        valid_long_recordings = list(self._iter_valid_recordings())
        if not valid_long_recordings:
            raise RuntimeError(
                "No valid (nonzero-sample) recordings were loaded for this animal. "
                "One or more LongRecordingOrganizer instances were created, but all of "
                "them contain 0 samples. This usually indicates a misconfiguration "
                "(wrong file pattern, wrong data root, or corrupt data). "
                "Check the warnings above for per-file root causes."
            )
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

    @staticmethod
    def _filter_zero_sample_lros(lro_pairs, get_name):
        """Remove 0-sample LROs from *lro_pairs* before a merge loop.

        A 0-sample LRO used as the **base** of a merge causes
        ``si.concatenate_recordings`` to fail or produce corrupt metadata.
        This helper removes such LROs up-front so every caller's merge loop
        starts from a valid base.

        This is intentionally separate from the ``merge()`` check in
        ``LongRecordingOrganizer``, which only guards against a 0-sample
        *other_lro* being merged in.  Together the two checks cover all cases:
        base=0-sample (this helper) and other_lro=0-sample (``merge()``).

        Args:
            lro_pairs: Iterable of ``(key, lro)`` pairs.  *key* is whatever
                the caller uses to name the LRO (item path, string tag, …).
            get_name: Callable ``(key) -> str`` used to produce a human-readable
                name for warning messages.

        Returns:
            ``(valid_pairs, skipped_names)`` where *valid_pairs* is a list of
            ``(key, lro)`` pairs with 0-sample entries removed and
            *skipped_names* is a list of names of the removed LROs.
        """
        valid_pairs = []
        skipped_names = []
        for key, lro in lro_pairs:
            try:
                if (
                    hasattr(lro, "LongRecording")
                    and lro.LongRecording is not None
                    and lro.LongRecording.get_total_samples() == 0
                ):
                    skipped_names.append(get_name(key))
                    continue
            except (TypeError, AttributeError):
                pass  # Non-SI or mock — keep it
            valid_pairs.append((key, lro))
        return valid_pairs, skipped_names

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
        """Delegates to :meth:`AnalysisPipeline.compute_bad_channels`."""
        return AnalysisPipeline(self).compute_bad_channels(
            lof_threshold=lof_threshold, force_recompute=force_recompute,
            lof_chunk_duration_s=lof_chunk_duration_s,
        )

    def apply_lof_threshold(self, lof_threshold: float):
        """Delegates to :meth:`AnalysisPipeline.apply_lof_threshold`."""
        return AnalysisPipeline(self).apply_lof_threshold(lof_threshold)

    def get_all_lof_scores(self) -> dict:
        """Delegates to :meth:`AnalysisPipeline.get_all_lof_scores`."""
        return AnalysisPipeline(self).get_all_lof_scores()

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
        """Delegates to :meth:`AnalysisPipeline.compute_windowed_analysis`."""
        return AnalysisPipeline(self).compute_windowed_analysis(
            features, exclude=exclude, window_s=window_s,
            multiprocess_mode=multiprocess_mode,
            suppress_short_interval_error=suppress_short_interval_error,
            apply_notch_filter=apply_notch_filter,
            chunk_duration_s=chunk_duration_s, **kwargs,
        )

    def compute_frequency_domain_spike_analysis(
        self,
        detection_params: dict = None,
        chunk_duration_s: float = 3600,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
    ):
        """Delegates to :meth:`AnalysisPipeline.compute_frequency_domain_spike_analysis`."""
        return AnalysisPipeline(self).compute_frequency_domain_spike_analysis(
            detection_params=detection_params,
            chunk_duration_s=chunk_duration_s,
            multiprocess_mode=multiprocess_mode,
        )

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
            genotype = constants.ANIMAL_METADATA.get(animal, {}).get("genotype", genotype)
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

                # Filter out 0-sample LROs before the merge loop.  A 0-sample
                # base LRO makes si.concatenate_recordings fail; using the same
                # helper as _create_long_recordings keeps the two code paths
                # consistent.
                lro_pairs = [(f"lro_{idx}", lro) for idx, lro in lro_group]
                valid_pairs, skipped_names = cls._filter_zero_sample_lros(
                    lro_pairs, lambda k: k
                )
                if skipped_names:
                    logging.warning(
                        f"Skipping {len(skipped_names)} 0-sample LRO(s) for "
                        f"'{animalday}' before merge: {skipped_names}"
                    )
                if not valid_pairs:
                    logging.warning(
                        f"All {len(lro_group)} LRO(s) for '{animalday}' are "
                        f"0-sample; skipping this date."
                    )
                    continue
                lro_pairs = valid_pairs

                # Sort by median time (same logic as normal __init__)
                sorted_pairs = cls._sort_lros_by_median_time_static(lro_pairs)

                # Merge all LROs into the first one (in temporal order)
                base_lro = sorted_pairs[0][1]
                base_tag = sorted_pairs[0][0]
                logging.info(f"Base LRO: {base_tag}")

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

        # Step 3: Ensure at least one date produced a valid (non-empty) merged LRO.
        # If every date group was filtered out as 0-sample, merged_lros is empty
        # and downstream methods (e.g. compute_windowed_analysis) would fail with
        # less informative errors.  Raise early, consistent with the normal
        # __init__ path which raises when nothing is loadable.
        if not merged_lros:
            raise ValueError(
                f"No non-empty local recording objects (LROs) could be loaded. "
                f"All date groups were 0-sample. Cannot construct an "
                f"AnimalOrganizer with no recordings."
            )

        # Step 4: Set merged LROs and animaldays
        ao.long_recordings = merged_lros
        ao.unique_animaldays = merged_animaldays
        ao.animaldays = (
            merged_animaldays.copy()
        )  # Create separate list for compatibility

        # Step 5: Validate and reconcile channel names across all merged LROs
        ao.channel_names = cls._validate_channel_names(merged_lros)

        # Step 6: CRITICAL VALIDATION - ensure no duplicates after merge
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

        # Step 7: Initialize default attributes for factory-created instances
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

        Compares abbreviated channel names (via ``resolve_channel``)
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

        reference_abbrevs = resolve_channels(first_names)
        reference_set = set(reference_abbrevs)
        # Map abbreviation -> canonical raw name from first LRO
        abbrev_to_raw = dict(zip(reference_abbrevs, first_names))

        for i, lro in enumerate(lros[1:], start=1):
            current_names = lro.channel_names if lro.channel_names else []
            current_abbrevs = resolve_channels(current_names)
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
        output_base: Union[str, Path] = None,
        format: Literal["zarr", "binary"] = "zarr",
        overwrite: bool = False,
        persist_base: Union[str, Path] = None,
    ) -> dict[str, "AnimalOrganizer"]:
        """
        Split this multi-animal AnimalOrganizer into per-animal AnimalOrganizers.

        For each group (animal), this method:
        1. Iterates over all LROs in this AnimalOrganizer
        2. Calls LRO.split() on each to extract the specified channels
        3. Optionally saves each split LRO to disk
        4. Creates a new AnimalOrganizer for each group

        This enables processing of joint-animal recordings where multiple animals
        are recorded on different channels of the same files.

        Args:
            groups (dict[str, list[str]]): Dictionary mapping group names (animal IDs)
                to lists of channel names. Example:
                {"AnimalA": ["Ch0", "Ch1", "Ch2", "Ch3"],
                 "AnimalB": ["Ch4", "Ch5", "Ch6", "Ch7"]}
            output_base (Union[str, Path], optional): Base directory for saving
                split recordings. If None, LROs remain in-memory. Structure:
                output_base/
                    AnimalA/
                        day1.zarr
                        day2.zarr
                    AnimalB/
                        ...
            format (Literal["zarr", "binary"], optional): Format for saved
                recordings. Defaults to "zarr".
            overwrite (bool, optional): Passed to
                :meth:`LongRecordingOrganizer.save_recording`; if True, replace an
                existing (recognized) recording folder. Defaults to False.
            persist_base (Union[str, Path], optional): Deprecated alias for
                ``output_base``. If provided (not None), it is used as ``output_base``
                and emits a :class:`DeprecationWarning`.

        Returns:
            dict[str, AnimalOrganizer]: Dictionary mapping group names to new
                AnimalOrganizer instances.

        Raises:
            ValueError: If requested channels are not found in recordings.

        Example:
            >>> ao = AnimalOrganizer("/path/to/joint_data", "combined")
            >>> splits = ao.split(
            ...     groups={"MouseA": ["Ch0", "Ch1"], "MouseB": ["Ch2", "Ch3"]},
            ...     output_base="/output/split_data",
            ... )
            >>> war_a = splits["MouseA"].compute_windowed_analysis(["all"])
            >>> war_b = splits["MouseB"].compute_windowed_analysis(["all"])
        """
        if not self.long_recordings:
            raise ValueError("No recordings loaded to split")

        if persist_base is not None:
            warnings.warn(
                "The 'persist_base' argument of AnimalOrganizer.split() is deprecated; "
                "use 'output_base'.",
                DeprecationWarning,
                stacklevel=2,
            )
            if output_base is None:
                output_base = persist_base

        if output_base is not None:
            output_base = Path(output_base)
            output_base.mkdir(parents=True, exist_ok=True)

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

                # Save to disk if requested
                if output_base is not None:
                    # Determine day folder name
                    day_name = lro.display_name or f"day{i}"

                    output_dir = output_base / group_name / day_name
                    child_lro.save_recording(output_dir, format=format, overwrite=overwrite)
                    logging.debug(f"Saved {group_name}/{day_name} to {output_dir}")

                child_lros.append(child_lro)

            # Create AnimalOrganizer from the split LROs
            child_ao = AnimalOrganizer.from_lros(
                lros=child_lros,
                animal_id=group_name,
                genotype=self.genotype,
                sex=self.sex,
            )

            result[group_name] = child_ao
            logging.info(
                f"Created AnimalOrganizer for '{group_name}' with "
                f"{len(child_lros)} days, {len(channels)} channels"
            )

        return result


