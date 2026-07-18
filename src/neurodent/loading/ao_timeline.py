"""Timestamp resolution, global timeline construction, and timeline summaries.

Mixin for :class:`~neurodent.loading.animal_organizer.AnimalOrganizer`.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import dateutil.parser
import pandas as pd

from . import long_recording_organizer as _lro


class AoTimelineMixin:
    """Mixin: see module docstring."""

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
                raise RuntimeError(
                    f"User timestamp function failed on folder '{folder_path}'"
                ) from e

        else:
            raise TypeError(
                f"Invalid timestamp input type: {type(input_spec)}. Expected: datetime, List[datetime], or Callable"
            )

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

        from .discovery import _natural_sort_key

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
                            temp_lro = _lro.LongRecordingOrganizer(
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
                    temp_lro = _lro.LongRecordingOrganizer(item, **_lro_kwargs)
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
                    temp_lro = _lro.LongRecordingOrganizer(item, **_lro_kwargs)
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
        temp_lro = _lro.LongRecordingOrganizer(last_item, **_kw)
        duration = (
            temp_lro.LongRecording.get_duration()
            if hasattr(temp_lro, "LongRecording") and temp_lro.LongRecording
            else 0.0
        )
        anchor_end_dt = last_start + timedelta(seconds=duration)
        return timeline, anchor_end_dt

    def _process_manual_datetimes(
        self, manual_datetimes, animalday_to_items: dict, base_lro_kwargs: dict,
        validate_only: bool = False,
    ) -> dict:
        """Resolve ``manual_datetimes`` into a per-item ``{item_key: start_datetime}`` map.

        When ``validate_only=True`` the key/shape validation runs exactly as normal (raising
        identically on any session/item/length mismatch), but the function returns ``{}`` right
        after — skipping the timeline computation, which reads every file's duration. This is
        what makes the dry-run's single checkpoint cheap while staying byte-for-byte the same
        validation the real load performs.

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
                if validate_only:
                    return {}
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

                if validate_only:
                    return {}

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

            if validate_only:
                return {}

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
