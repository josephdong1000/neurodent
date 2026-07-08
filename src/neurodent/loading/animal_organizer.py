"""Multi-session recording organization for a single animal.

``AnimalOrganizer`` discovers and loads recording files for a single animal,
groups them into sessions, and manages ``LongRecordingOrganizer`` instances and
their timeline. Analysis (LOF, windowed features, spike detection) is a separate
stage: pass a loaded organizer to
:class:`~neurodent.analysis.animal_analyzer.AnimalAnalyzer`.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from neurodent import constants
from . import long_recording_organizer as _lro
from neurodent.core.utils import parse_truncate

from .ao_discovery import AoDiscoveryMixin
from .ao_timeline import AoTimelineMixin
from .ao_build import AoBuildMixin
from .ao_validation import AoValidationMixin


class AnimalOrganizer(
    AoDiscoveryMixin,
    AoTimelineMixin,
    AoBuildMixin,
    AoValidationMixin,
):
    """
    Organizes and loads recording data from a single animal across multiple sessions.

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
    """

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

        from .discovery import FileDiscoverer

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

            truncate = parse_truncate(truncate)
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

        if "manual_datetimes" in lro_kwargs:
            logging.info("Processing manual_datetimes configuration")
            base_lro_kwargs = lro_kwargs.copy()
            base_lro_kwargs["manual_datetimes"] = datetime(2000, 1, 1, 0, 0, 0)

            self._processed_timestamps = self._process_manual_datetimes(
                lro_kwargs["manual_datetimes"],
                self._animalday_folder_groups,
                base_lro_kwargs,
            )
            lro_kwargs = base_lro_kwargs
        else:
            self._processed_timestamps = None


        self.long_recordings: list[_lro.LongRecordingOrganizer] = []
        self._create_long_recordings(lro_kwargs)

        # Set and validate channel_names across all LROs
        self.channel_names = self._validate_channel_names(self.long_recordings)
