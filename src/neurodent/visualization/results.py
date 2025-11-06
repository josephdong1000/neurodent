import copy
import glob
import json
import logging
import os
import re
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Union

import dask
import dask.array as da
import mne
import numpy as np
import pandas as pd
import spikeinterface as si
from dask import delayed
from django.utils.text import slugify
from scipy.stats import zscore
from scipy.ndimage import binary_opening, binary_closing
from tqdm import tqdm


from .. import constants, core
from ..core import FragmentAnalyzer, get_temp_directory
from ..core.analyze_sort import MOUNTAINSORT_AVAILABLE
from ..core.frequency_domain_spike_detection import FrequencyDomainSpikeDetector
from ..core.utils import parse_chname_to_abbrev


class AnimalFeatureParser:
    # REVIEW make this a utility function and refactor across codebase?
    def _average_feature(self, df: pd.DataFrame, colname: str, weightsname: str | None = "duration"):
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

            case "cohere" | "zcohere" | "imcoh" | "zimcoh" | "psdband" | "psdfrac" | "logpsdband" | "logpsdfrac":
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
    def __init__(
        self,
        base_folder_path,
        anim_id: str,
        day_sep: str | None = None,
        mode: Literal["nest", "concat", "base", "noday"] = "concat",
        assume_from_number=False,
        skip_days: list[str] = [],
        truncate: bool | int = False,
        lro_kwargs: dict = {},
    ) -> None:
        """
        AnimalOrganizer is used to organize data from a single animal into a format that can be used for analysis.
        It is used to organize data from a single animal into a format that can be used for analysis.

        Args:
            base_folder_path (str): The path to the base folder of the animal data.
            anim_id (str): The ID of the animal. This should correspond to only one animal.
            day_sep (str, optional): Separator for day in folder name. Set to None or empty string to get all folders. Defaults to None.
            mode (Literal["nest", "concat", "base", "noday"], optional): The mode of the AnimalOrganizer. Defaults to "concat".
                File structure patterns (where * indicates search location):
                "nest": base_folder_path / animal_id / *date_format* (looks for folders/files within animal_id subdirectories)
                "concat": base_folder_path / *animal_id*date_format* (looks for folders/files with animal_id+date in name at base level)
                "base": base_folder_path / * (looks for folders/files directly in base_folder_path)
                "noday": base_folder_path / *animal_id* (same as concat but expects single unique match, no date filtering)
            assume_from_number (bool, optional): Whether to assume the animal ID is a number. Defaults to False.
            skip_days (list[str], optional): The days to skip. Defaults to [].
            truncate (bool|int, optional): Whether to truncate the data. Defaults to False.
            lro_kwargs (dict, optional): Keyword arguments for LongRecordingOrganizer. Defaults to {}.
        """

        self.base_folder_path = Path(base_folder_path)
        self.anim_id = anim_id
        self.animal_param = [anim_id]
        self.day_sep = day_sep
        self.read_mode = mode
        self.assume_from_number = assume_from_number

        match mode:
            case "nest":
                self.bin_folder_pattern = self.base_folder_path / f"*{self.anim_id}*" / "*"
            case "concat" | "noday":
                self.bin_folder_pattern = self.base_folder_path / f"*{self.anim_id}*"
                # self.bin_folder_pat = self.base_folder_path / f"*{self.anim_id}*{self.date_format}*"
            case "base":
                self.bin_folder_pattern = self.base_folder_path
            # case 'noday':
            #     self.bin_folder_pat = self.base_folder_path / f"*{self.anim_id}*"
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        self._bin_folders = glob.glob(str(self.bin_folder_pattern))

        # Filter to only include directories (LongRecordingOrganizer expects folder paths)
        before_filter_count = len(self._bin_folders)
        self._bin_folders = [x for x in self._bin_folders if Path(x).is_dir()]
        after_filter_count = len(self._bin_folders)

        if before_filter_count > after_filter_count:
            filtered_count = before_filter_count - after_filter_count
            logging.info(f"Filtered out {filtered_count} non-directory items (files) from glob results")

        # if mode != 'noday':
        #     self.__bin_folders = [x for x in self.__bin_folders if datetime.strptime(Path(x).name, self.date_format)]
        truncate = core.utils.parse_truncate(truncate)
        if truncate:
            warnings.warn(f"AnimalOrganizer will be truncated to the first {truncate} LongRecordings")
            self._bin_folders = self._bin_folders[:truncate]
        self._bin_folders = [x for x in self._bin_folders if not any(y in x for y in skip_days)]
        self.bin_folder_names = [Path(x).name for x in self._bin_folders]
        logging.info(f"bin_folder_pattern: {self.bin_folder_pattern}")
        logging.info(f"self._bin_folders: {self._bin_folders}")
        logging.info(f"self.bin_folder_names: {self.bin_folder_names}")

        if mode == "noday" and len(self._bin_folders) > 1:
            raise ValueError(f"Animal ID '{self.anim_id}' is not unique, found: {', '.join(self._bin_folders)}")
        elif len(self._bin_folders) == 0:
            raise ValueError(f"No directories found for animal ID {self.anim_id} (pattern: {self.bin_folder_pattern})")

        self._animalday_dicts = [
            core.parse_path_to_animalday(e, animal_param=self.animal_param, day_sep=self.day_sep, mode=self.read_mode)
            for e in self._bin_folders
        ]

        # Group folders by parsed animalday to handle overlapping days
        animalday_to_folders = {}
        for folder, animalday_dict in zip(self._bin_folders, self._animalday_dicts):
            animalday = animalday_dict["animalday"]
            if animalday not in animalday_to_folders:
                animalday_to_folders[animalday] = []
            animalday_to_folders[animalday].append(folder)

        # Store grouping info
        self._animalday_folder_groups = animalday_to_folders
        self.unique_animaldays = list(animalday_to_folders.keys())

        # Log merging operations for overlapping days
        overlapping_days = 0
        for animalday, folders in animalday_to_folders.items():
            if len(folders) > 1:
                overlapping_days += 1
                logging.info(f"Merging {len(folders)} folders for {animalday}: {[Path(f).name for f in folders]}")

        if overlapping_days > 0:
            logging.info(f"Found {overlapping_days} animaldays with overlapping folders")

        # Update animaldays to reflect unique days (not total folders)
        self.animaldays = self.unique_animaldays
        logging.info(f"self.animaldays (unique): {self.animaldays}")

        genotypes = [x["genotype"] for x in self._animalday_dicts]
        if len(set(genotypes)) > 1:
            warnings.warn(f"Inconsistent genotypes in {genotypes}")
        self.genotype = genotypes[0]
        logging.info(f"self.genotype: {self.genotype}")

        self.long_analyzers: list[core.LongRecordingAnalyzer] = []
        logging.debug(f"Creating {len(self.unique_animaldays)} LongRecordings (one per unique animalday)")

        # Process manual_datetimes if provided in lro_kwargs
        if "manual_datetimes" in lro_kwargs:
            logging.info("Processing manual_datetimes configuration")
            base_lro_kwargs = lro_kwargs.copy()
            base_lro_kwargs["manual_datetimes"] = datetime(2000, 1, 1, 0, 0, 0)

            self._processed_timestamps = self._process_all_timestamps(
                lro_kwargs["manual_datetimes"], self._animalday_folder_groups, base_lro_kwargs
            )
            # Remove from lro_kwargs since we'll handle it manually
            lro_kwargs = base_lro_kwargs
        else:
            self._processed_timestamps = None

        # Create LongRecordingOrganizer instances
        self._create_long_recordings(lro_kwargs)

    def _resolve_timestamp_input(self, input_spec, folder_path: Path):
        """
        Recursively resolve any timestamp input type to concrete datetime(s).

        Args:
            input_spec: datetime, List[datetime], or Callable returning either
            folder_path: Path to folder for function execution context

        Returns:
            Union[datetime, List[datetime]]: Resolved timestamp(s)

        Raises:
            TypeError: If input_spec is not a supported type
            Exception: If user function fails (wrapped with context)
        """
        if isinstance(input_spec, datetime):
            return input_spec

        elif isinstance(input_spec, list):
            # Validate that all items are datetime objects
            if not all(isinstance(dt, datetime) for dt in input_spec):
                raise TypeError(
                    f"All items in timestamp list must be datetime objects, got: {[type(dt) for dt in input_spec]}"
                )
            return input_spec

        elif callable(input_spec):
            try:
                logging.debug(f"Executing user timestamp function on folder: {folder_path}")
                result = input_spec(folder_path)
                # Recursively process the result (functions can return datetime or list)
                return self._resolve_timestamp_input(result, folder_path)
            except Exception as e:
                logging.error(f"User timestamp function failed on folder '{folder_path}': {e}")
                raise Exception(f"User timestamp function failed on folder '{folder_path}': {e}") from e

        else:
            raise TypeError(
                f"Invalid timestamp input type: {type(input_spec)}. Expected: datetime, List[datetime], or Callable"
            )

    def _find_folder_by_name(self, folder_name: str, animalday_to_folders: dict) -> Path:
        """Find folder path by name in the animalday groups."""
        for animalday, folders in animalday_to_folders.items():
            for folder in folders:
                if Path(folder).name == folder_name:
                    return Path(folder)

        available_names = []
        for folders in animalday_to_folders.values():
            available_names.extend([Path(f).name for f in folders])

        raise ValueError(f"Folder name '{folder_name}' not found. Available folders: {available_names}")

    def _compute_global_timeline(
        self, base_datetime: datetime, animalday_to_folders: dict, base_lro_kwargs: dict
    ) -> dict:
        """
        Compute contiguous timeline for all folders starting from base_datetime.

        This uses a two-pass approach:
        1. Create temporary LROs to determine durations
        2. Compute continuous start times based on cumulative durations
        3. Return timeline mapping for final LRO creation

        Args:
            base_datetime: Starting datetime for the timeline
            animalday_to_folders: Mapping of animalday -> list of folder paths
            base_lro_kwargs: Base kwargs for LRO construction (without manual_datetimes)

        Returns:
            dict: Mapping of folder_name -> start_datetime for continuous timeline
        """
        total_folders = sum(len(folders) for folders in animalday_to_folders.values())
        total_animaldays = len(animalday_to_folders)

        logging.info(
            f"Computing continuous timeline for {total_animaldays} animaldays ({total_folders} total folders) "
            f"starting at {base_datetime}"
        )

        # Step 1: Create temporary LROs to determine durations
        # We need to create LROs in the order they will appear in the final timeline
        ordered_folders = []
        for animalday in sorted(animalday_to_folders.keys()):
            folders = animalday_to_folders[animalday]
            if len(folders) > 1:
                # For overlapping folders, we need to sort them by temporal order
                # Create temp LROs to get timing info for sorting
                folder_lro_pairs = []
                for folder in folders:
                    try:
                        temp_lro = core.LongRecordingOrganizer(folder, **base_lro_kwargs)
                        folder_lro_pairs.append((folder, temp_lro))
                    except Exception as e:
                        logging.warning(f"Failed to create temp LRO for duration estimation in {folder}: {e}")
                        # Use folder order as fallback
                        folder_lro_pairs.append((folder, None))

                # Sort by median time if possible
                sorted_pairs = self._sort_lros_by_median_time(folder_lro_pairs)
                ordered_folders.extend([folder for folder, _ in sorted_pairs])
            else:
                ordered_folders.extend(folders)

        # Step 2: Estimate total duration for each folder
        folder_durations = {}

        for folder in ordered_folders:
            # Create temporary LRO to get duration
            temp_lro = core.LongRecordingOrganizer(folder, **base_lro_kwargs)
            duration = (
                temp_lro.LongRecording.get_duration()
                if hasattr(temp_lro, "LongRecording") and temp_lro.LongRecording
                else 0.0
            )
            folder_durations[folder] = duration
            logging.debug(f"Folder {Path(folder).name}: estimated duration = {duration:.1f}s")

        # Step 3: Compute continuous start times
        result = {}
        current_start_time = base_datetime

        for folder in ordered_folders:
            folder_name = Path(folder).name
            result[folder_name] = current_start_time

            # Move to next start time (current start + duration)
            duration = folder_durations[folder]
            current_start_time = current_start_time + timedelta(seconds=duration)

            logging.debug(f"Timeline: {folder_name} starts at {result[folder_name]}, duration {duration:.1f}s")

        total_timeline_duration = sum(folder_durations.values())
        logging.info(
            f"Continuous timeline computed: {len(result)} folders, total duration {total_timeline_duration:.1f}s"
        )

        return result

    def _process_all_timestamps(self, manual_datetimes, animalday_to_folders: dict, base_lro_kwargs: dict) -> dict:
        """
        Process the top-level manual_datetimes input and return folder_name -> resolved_timestamps mapping.

        Args:
            manual_datetimes: Any supported timestamp input type
            animalday_to_folders: Mapping of animalday -> list of folder paths
            base_lro_kwargs: Base kwargs for LRO construction (without manual_datetimes)

        Returns:
            dict: Mapping of folder_name -> Union[datetime, List[datetime]]
        """
        if isinstance(manual_datetimes, dict):
            # Per-folder specification
            logging.info("Processing per-folder timestamp specification")
            resolved = {}
            for folder_name, folder_spec in manual_datetimes.items():
                folder_path = self._find_folder_by_name(folder_name, animalday_to_folders)
                resolved[folder_name] = self._resolve_timestamp_input(folder_spec, folder_path)
                logging.debug(f"Resolved timestamps for {folder_name}: {resolved[folder_name]}")
            return resolved

        elif isinstance(manual_datetimes, datetime):
            # Global timeline - compute contiguous spacing
            logging.info(f"Processing global timeline starting at {manual_datetimes}")
            return self._compute_global_timeline(manual_datetimes, animalday_to_folders, base_lro_kwargs)

        else:
            # Function or list at top level - apply to all folders
            logging.info("Processing timestamp input for all folders")
            resolved = {}
            for animalday, folders in animalday_to_folders.items():
                for folder in folders:
                    folder_name = Path(folder).name
                    resolved[folder_name] = self._resolve_timestamp_input(manual_datetimes, Path(folder))
                    logging.debug(f"Resolved timestamps for {folder_name}: {resolved[folder_name]}")
            return resolved

    def _get_lro_kwargs_for_folder(self, folder_path: str, base_lro_kwargs: dict) -> dict:
        """
        Get the appropriate lro_kwargs for a specific folder, including processed timestamps if available.

        Args:
            folder_path: Path to the folder
            base_lro_kwargs: Base kwargs to extend

        Returns:
            dict: lro_kwargs with manual_datetimes added if available
        """
        if self._processed_timestamps is None:
            return base_lro_kwargs

        folder_name = Path(folder_path).name
        if folder_name in self._processed_timestamps:
            # Add the processed timestamps for this folder
            kwargs = base_lro_kwargs.copy()
            kwargs["manual_datetimes"] = self._processed_timestamps[folder_name]
            logging.debug(f"Using processed timestamps for folder {folder_name}: {kwargs['manual_datetimes']}")
            return kwargs
        else:
            # No processed timestamps for this folder - use base kwargs
            logging.debug(f"No processed timestamps for folder {folder_name}, using base kwargs")
            return base_lro_kwargs

    def _log_timeline_summary(self):
        """Log timeline summary for debugging purposes."""

        lines = ["AnimalOrganizer Timeline Summary:"]

        if not self.long_recordings:
            lines.append("No LongRecordings created")
        else:
            for i, lro in enumerate(self.long_recordings):
                try:
                    start_time = self._get_lro_start_time(lro)
                    end_time = self._get_lro_end_time(lro)
                    duration = (
                        lro.LongRecording.get_duration() if hasattr(lro, "LongRecording") and lro.LongRecording else 0
                    )
                    n_files = len(lro.file_durations) if hasattr(lro, "file_durations") and lro.file_durations else 1
                    folder_path = getattr(lro, "base_folder_path", "unknown")

                    lines.append(
                        f"LRO {i}: {start_time} → {end_time} "
                        f"(duration: {duration:.1f}s, files: {n_files}, folder: {Path(folder_path).name})"
                    )
                except Exception as e:
                    lines.append(f"Failed to get timeline info for LRO {i}: {e}")

        logging.info("\n".join(lines))

    def _get_lro_start_time(self, lro):
        """Get the start time of an LRO."""
        if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes:
            if hasattr(lro, "file_durations") and lro.file_durations:
                # Calculate start time from first end time and duration
                first_end = next(dt for dt in lro.file_end_datetimes if dt is not None)
                first_duration = lro.file_durations[0]
                return first_end - timedelta(seconds=first_duration)
        return "unknown"

    def _get_lro_end_time(self, lro):
        """Get the end time of an LRO."""
        if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes:
            # Get the last non-None end time
            end_times = [dt for dt in lro.file_end_datetimes if dt is not None]
            if end_times:
                return max(end_times)
        return "unknown"

    def get_timeline_summary(self) -> pd.DataFrame:
        """
        Get timeline summary as a DataFrame for user inspection and debugging.

        Returns:
            pd.DataFrame: Timeline information with columns:
                - lro_index: Index of the LRO
                - start_time: Start datetime of the LRO
                - end_time: End datetime of the LRO
                - duration_s: Duration in seconds
                - n_files: Number of files in the LRO
                - folder_path: Base folder path
                - animalday: Parsed animalday identifier
        """
        if not self.long_recordings:
            return pd.DataFrame()

        timeline_data = []
        for i, lro in enumerate(self.long_recordings):
            try:
                start_time = self._get_lro_start_time(lro)
                end_time = self._get_lro_end_time(lro)
                duration = (
                    lro.LongRecording.get_duration() if hasattr(lro, "LongRecording") and lro.LongRecording else 0
                )
                n_files = len(lro.file_durations) if hasattr(lro, "file_durations") and lro.file_durations else 1
                folder_path = getattr(lro, "base_folder_path", "unknown")

                timeline_data.append(
                    {
                        "lro_index": i,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_s": duration,
                        "n_files": n_files,
                        "folder_path": str(folder_path),
                        "folder_name": Path(folder_path).name if folder_path != "unknown" else "unknown",
                        "animalday": getattr(
                            lro, "_animalday", "unknown"
                        ),  # This might not exist, but useful if it does
                    }
                )
            except Exception as e:
                # Include failed LROs in the summary for debugging
                timeline_data.append(
                    {
                        "lro_index": i,
                        "start_time": "error",
                        "end_time": "error",
                        "duration_s": 0,
                        "n_files": 0,
                        "folder_path": "error",
                        "folder_name": "error",
                        "animalday": "error",
                        "error": str(e),
                    }
                )

        return pd.DataFrame(timeline_data)

    def _create_long_recordings(self, lro_kwargs: dict):
        """Create LongRecordingOrganizer instances for each unique animalday."""
        # Create one LRO per unique animalday (not per folder)
        self.long_recordings: list[core.LongRecordingOrganizer] = []
        for animalday, folders in self._animalday_folder_groups.items():
            if len(folders) == 1:
                # Single folder - use processed timestamps if available
                folder_kwargs = self._get_lro_kwargs_for_folder(folders[0], lro_kwargs)
                lro = core.LongRecordingOrganizer(folders[0], **folder_kwargs)
            else:
                # Multiple folders - create individual LROs then sort and merge
                logging.info(f"Creating individual LROs for {len(folders)} folders for {animalday}")

                # Create individual LROs first, each with their own processed timestamps
                folder_lro_pairs = []
                for folder in folders:
                    folder_kwargs = self._get_lro_kwargs_for_folder(folder, lro_kwargs)
                    individual_lro = core.LongRecordingOrganizer(folder, **folder_kwargs)
                    folder_lro_pairs.append((folder, individual_lro))

                # Sort by median time using constructed LROs
                sorted_folder_lro_pairs = self._sort_lros_by_median_time(folder_lro_pairs)

                # Debug logging to show the order of LROs being merged
                logging.info("LRO merge order for overlapping animalday:")
                for i, (folder, lro) in enumerate(sorted_folder_lro_pairs):
                    folder_name = Path(folder).name
                    # Handle mock objects gracefully
                    try:
                        duration = (
                            lro.LongRecording.get_duration()
                            if hasattr(lro, "LongRecording") and lro.LongRecording
                            else 0
                        )
                        duration_str = f"{float(duration):.1f}s"
                    except (TypeError, ValueError):
                        duration_str = "mock"
                    logging.info(f"  {i + 1}. {folder_name} (duration: {duration_str})")

                # Merge all LROs into the first one (in temporal order)
                merged_lro = sorted_folder_lro_pairs[0][1]  # Get the LRO from first tuple
                logging.info(f"Base LRO: {Path(sorted_folder_lro_pairs[0][0]).name}")

                for i, (folder, lro) in enumerate(sorted_folder_lro_pairs[1:], 1):
                    folder_name = Path(folder).name
                    logging.info(f"Merging LRO {i}: {folder_name} into base LRO")
                    merged_lro.merge(lro)

                lro = merged_lro
                logging.info(f"Successfully merged {len(sorted_folder_lro_pairs)} LROs for {animalday}")

            self.long_recordings.append(lro)

        # Log timeline summary for debugging
        self._log_timeline_summary()

        channel_names = [x.channel_names for x in self.long_recordings]
        if len(set([" ".join(x) for x in channel_names])) > 1:
            warnings.warn(f"Inconsistent channel names in long_recordings: {channel_names}")
        self.channel_names = channel_names[0]
        self.bad_channels_dict = {}

        animal_ids = [x["animal"] for x in self._animalday_dicts]
        if len(set(animal_ids)) > 1:
            warnings.warn(f"Inconsistent animal IDs in {animal_ids}")
        self.animal_id = animal_ids[0]

        self.features_df: pd.DataFrame = pd.DataFrame()
        self.features_avg_df: pd.DataFrame = pd.DataFrame()

    def _sort_lros_by_median_time(self, folder_lro_pairs):
        """Sort LROs by median timestamp of their constituent recordings.

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
        if len(folder_lro_pairs) <= 1:
            return folder_lro_pairs

        folder_lro_times = []

        for folder_path, lro in folder_lro_pairs:
            try:
                # Get median timestamp from constituent recordings within the LRO
                if hasattr(lro, "file_end_datetimes") and lro.file_end_datetimes:
                    try:
                        valid_timestamps = [ts for ts in lro.file_end_datetimes if ts is not None]
                    except TypeError:
                        valid_timestamps = []

                    if valid_timestamps:
                        # Sort timestamps and get the median
                        valid_timestamps.sort()
                        n_timestamps = len(valid_timestamps)

                        if n_timestamps % 2 == 1:
                            # Odd number of timestamps - take middle one
                            median_timestamp = valid_timestamps[n_timestamps // 2]
                        else:
                            # Even number of timestamps - take average of two middle ones
                            mid1 = valid_timestamps[n_timestamps // 2 - 1]
                            mid2 = valid_timestamps[n_timestamps // 2]
                            median_timestamp = mid1 + (mid2 - mid1) / 2

                        # Convert to seconds since epoch for sorting
                        median_time_seconds = median_timestamp.timestamp()
                        logging.debug(
                            f"LRO {Path(folder_path).name}: {n_timestamps} recordings, median timestamp: {median_timestamp}"
                        )
                    else:
                        raise ValueError(f"No file_end_datetimes available in LRO {Path(folder_path).name}, cannot determine temporal order")
                else:
                    raise ValueError(f"No file_end_datetimes available in LRO {Path(folder_path).name}, cannot determine temporal order")

                folder_lro_times.append((folder_path, lro, median_time_seconds))

            except Exception as e:
                logging.warning(f"Could not extract timing from {folder_path}: {e}")
                raise

        # Sort by median time
        sorted_folder_lro_times = sorted(folder_lro_times, key=lambda x: x[2])
        sorted_folder_lro_pairs = [(folder, lro) for folder, lro, _ in sorted_folder_lro_times]

        # Log the sorting for debugging
        if len(folder_lro_pairs) > 1:
            logging.info("LRO temporal sorting details:")
            for i, (folder, lro, median_time_seconds) in enumerate(sorted_folder_lro_times):
                folder_name = Path(folder).name

                # Convert back to datetime for readable logging
                try:
                    from datetime import datetime

                    median_datetime = datetime.fromtimestamp(median_time_seconds)
                    median_time_str = median_datetime.strftime("%Y-%m-%d %H:%M:%S")
                except (TypeError, ValueError, OSError):
                    median_time_str = f"{median_time_seconds:.1f}s"

                # Handle mock objects gracefully for duration
                try:
                    duration = (
                        lro.LongRecording.get_duration() if hasattr(lro, "LongRecording") and lro.LongRecording else 0
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
            folder_names = [Path(f).name for f, _, _ in sorted_folder_lro_times]
            median_times = []
            for _, _, median_time_seconds in sorted_folder_lro_times:
                median_datetime = datetime.fromtimestamp(median_time_seconds)
                median_times.append(median_datetime.strftime("%H:%M:%S"))

            logging.info(f"Final sort order: {list(zip(folder_names, median_times))}")

        return sorted_folder_lro_pairs

    def convert_colbins_to_rowbins(self, overwrite=False, multiprocess_mode: Literal["dask", "serial"] = "serial"):
        for lrec in tqdm(self.long_recordings, desc="Converting column bins to row bins"):
            lrec.convert_colbins_to_rowbins(overwrite=overwrite, multiprocess_mode=multiprocess_mode)

    def convert_rowbins_to_rec(self, multiprocess_mode: Literal["dask", "serial"] = "serial"):
        for lrec in tqdm(self.long_recordings, desc="Converting row bins to recs"):
            lrec.convert_rowbins_to_rec(multiprocess_mode=multiprocess_mode)

    def cleanup_rec(self):
        for lrec in self.long_recordings:
            lrec.cleanup_rec()

    def compute_bad_channels(self, lof_threshold: float = None, force_recompute: bool = False):
        """Compute bad channels using LOF analysis for all recordings.

        Args:
            lof_threshold (float, optional): Threshold for determining bad channels from LOF scores.
                                           If None, only computes/loads scores without setting bad_channel_names.
            force_recompute (bool): Whether to recompute LOF scores even if they exist.
        """
        logging.info(
            f"Computing bad channels for {len(self.long_recordings)} recordings with threshold={lof_threshold}"
        )
        for i, lrec in enumerate(self.long_recordings):
            logging.debug(f"Computing bad channels for recording {i}: {self.animaldays[i]}")
            lrec.compute_bad_channels(lof_threshold=lof_threshold, force_recompute=force_recompute)
            logging.debug(
                f"Recording {i} LOF scores computed: {hasattr(lrec, 'lof_scores') and lrec.lof_scores is not None}"
            )

        # Update bad channels dict if threshold was applied
        if lof_threshold is not None:
            self.bad_channels_dict = {
                animalday: lrec.bad_channel_names for animalday, lrec in zip(self.animaldays, self.long_recordings)
            }

    def apply_lof_threshold(self, lof_threshold: float):
        """Apply threshold to existing LOF scores to determine bad channels for all recordings.

        Args:
            lof_threshold (float): Threshold for determining bad channels.
        """
        for lrec in self.long_recordings:
            lrec.apply_lof_threshold(lof_threshold)

        self.bad_channels_dict = {
            animalday: lrec.bad_channel_names for animalday, lrec in zip(self.animaldays, self.long_recordings)
        }

    def get_all_lof_scores(self) -> dict:
        """Get LOF scores for all recordings.

        Returns:
            dict: Dictionary mapping animal days to LOF score dictionaries.
        """
        return {animalday: lrec.get_lof_scores() for animalday, lrec in zip(self.animaldays, self.long_recordings)}

    def compute_windowed_analysis(
        self,
        features: list[str],
        exclude: list[str] = [],
        window_s=4,
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
            window_s (int, optional): Length of each window in seconds. Note that some features break with very short window times. Defaults to 4.
            suppress_short_interval_error (bool, optional): If True, suppress ValueError for short intervals between timestamps in resulting WindowAnalysisResult. Useful for aggregated WARs. Defaults to False.
            apply_notch_filter (bool, optional): Whether to apply notch filtering to remove line noise. Uses constants.LINE_FREQ. Defaults to True.

        Raises:
            AttributeError: If a feature's ``compute_...()`` function was not implemented, this error will be raised.

        Returns:
            WindowAnalysisResult: A WindowAnalysisResult object containing extracted features for all recordings
        """
        features = _sanitize_feature_request(features, exclude)

        dataframes = []
        for lrec in self.long_recordings:  # Iterate over all long recordings
            logging.info(f"Computing windowed analysis for {lrec.base_folder_path}")
            lan = core.LongRecordingAnalyzer(lrec, fragment_len_s=window_s, apply_notch_filter=apply_notch_filter)
            if lan.n_fragments == 0:
                logging.warning(f"No fragments found for {lrec.base_folder_path}. Skipping.")
                continue

            logging.debug(f"Processing {lan.n_fragments} fragments")
            miniters = int(lan.n_fragments / 100)
            match multiprocess_mode:
                case "dask":
                    # The last fragment is not included because it makes the dask array ragged
                    logging.debug("Converting LongRecording to numpy array")

                    n_fragments_war = max(lan.n_fragments - 1, 1)
                    first_fragment = lan.get_fragment_np(0)
                    np_fragments = np.empty((n_fragments_war,) + first_fragment.shape, dtype=first_fragment.dtype)
                    logging.debug(f"np_fragments.shape: {np_fragments.shape}")
                    for idx in range(n_fragments_war):
                        np_fragments[idx] = lan.get_fragment_np(idx)

                    # Cache fragments to zarr
                    tmppath, _ = core.utils.cache_fragments_to_zarr(np_fragments, n_fragments_war)
                    del np_fragments

                    logging.debug("Processing metadata serially")
                    metadatas = [self._process_fragment_metadata(idx, lan, window_s) for idx in range(n_fragments_war)]
                    meta_df = pd.DataFrame(metadatas)

                    logging.debug("Processing features in parallel")
                    np_fragments_reconstruct = da.from_zarr(tmppath, chunks=("auto", -1, -1))
                    logging.debug(f"Dask array shape: {np_fragments_reconstruct.shape}")
                    logging.debug(f"Dask array chunks: {np_fragments_reconstruct.chunks}")

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
                        logging.warning(f"Failed to remove temporary directory {tmppath}: {e}")

                    logging.debug("Combining metadata and feature values")
                    feat_df = pd.DataFrame(feature_values)
                    lan_df = pd.concat([meta_df, feat_df], axis=1)

                case _:
                    logging.debug("Processing serially")
                    lan_df = []
                    for idx in tqdm(range(lan.n_fragments), desc="Processing rows", miniters=miniters):
                        lan_df.append(self._process_fragment_serial(idx, features, lan, window_s, kwargs))

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
                logging.info(f"Added LOF scores for {animalday}: {len(lrec.lof_scores)} channels")

        logging.info(f"Total LOF scores collected: {len(lof_scores_dict)} animal days")

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

    def compute_spike_analysis(self, multiprocess_mode: Literal["dask", "serial"] = "serial") -> list["SpikeAnalysisResult"]:
        """Compute spike sorting on all long recordings and return a list of SpikeAnalysisResult objects

        Args:
            multiprocess_mode (Literal['dask', 'serial']): Whether to use Dask for parallel processing. Defaults to 'serial'.

        Returns:
            list[SpikeAnalysisResult]: List of SpikeAnalysisResult objects. Each SpikeAnalysisResult object corresponds
                to a LongRecording object, typically a different day or recording session.

        Raises:
            ImportError: If mountainsort5 is not available.
        """
        # Check if mountainsort5 is available
        if not MOUNTAINSORT_AVAILABLE:
            raise ImportError("Spike analysis requires mountainsort5. Install it with: pip install mountainsort5")
        sars = []
        lrec_sorts = []
        lrec_recs = []
        recs = [lrec.LongRecording for lrec in self.long_recordings]
        logging.info(f"Sorting {len(recs)} recordings")
        for rec in recs:
            if rec.get_total_samples() == 0:
                logging.warning(f"Skipping {rec.__str__()} because it has no samples")
                sortings, recordings = [], []
            else:
                sortings, recordings = core.MountainSortAnalyzer.sort_recording(
                    rec, multiprocess_mode=multiprocess_mode
                )
            lrec_sorts.append(sortings)
            lrec_recs.append(recordings)

        if multiprocess_mode == "dask":
            lrec_sorts = dask.compute(*lrec_sorts)

        lrec_sas = [
            [
                si.create_sorting_analyzer(sorting, recording, sparse=False)
                for sorting, recording in zip(sortings, recordings)
            ]
            for sortings, recordings in zip(lrec_sorts, lrec_recs)
        ]
        sars = [
            SpikeAnalysisResult(
                result_sas=sas,
                result_mne=None,
                animal_id=self.animal_id,
                genotype=self.genotype,
                animal_day=self.animaldays[i],
                bin_folder_name=self.bin_folder_names[i],
                metadata=self.long_recordings[i].meta,
                channel_names=self.channel_names,
                assume_from_number=self.assume_from_number,
            )
            for i, sas in enumerate(lrec_sas)
        ]

        self.spike_analysis_results = sars
        return self.spike_analysis_results

    def compute_frequency_domain_spike_analysis(
        self,
        detection_params: dict = None,
        max_length: int = None,
        multiprocess_mode: Literal["dask", "serial"] = "serial"
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
        recs = [lrec.LongRecording for lrec in self.long_recordings]

        logging.info(f"Running frequency-domain spike detection on {len(recs)} recordings")
        logging.info(f"Detection parameters: {detection_params}")

        for i, rec in enumerate(recs):
            if rec.get_total_samples() == 0:
                logging.warning(f"Skipping {rec} because it has no samples")
                continue

            try:
                # Run frequency domain spike detection
                spike_indices_per_channel, mne_raw_with_annotations = (
                    FrequencyDomainSpikeDetector.detect_spikes_recording(
                        rec,
                        detection_params=detection_params,
                        max_length=max_length,
                        multiprocess_mode=multiprocess_mode
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
                    bin_folder_name=self.bin_folder_names[i],
                    metadata=self.long_recordings[i].meta,
                    assume_from_number=self.assume_from_number,
                )

                fdsar_list.append(fdsar)

                # Log results
                total_spikes = sum(len(spikes) for spikes in spike_indices_per_channel)
                logging.info(f"Recording {i+1}/{len(recs)}: Detected {total_spikes} spikes across {len(spike_indices_per_channel)} channels")

            except Exception as e:
                logging.error(f"Error processing recording {i+1}/{len(recs)}: {e}")
                raise

        # Store results for later access
        self.frequency_domain_spike_analysis_results = fdsar_list

        logging.info(f"Completed frequency-domain spike detection. Total recordings processed: {len(fdsar_list)}")
        return fdsar_list

    def _process_fragment_serial(self, idx, features, lan: core.LongRecordingAnalyzer, window_s, kwargs: dict):
        row = self._process_fragment_metadata(idx, lan, window_s)
        row.update(self._process_fragment_features(idx, features, lan, kwargs))
        return row

    def _process_fragment_metadata(self, idx, lan: core.LongRecordingAnalyzer, window_s):
        row = {}

        lan_folder = lan.LongRecording.base_folder_path
        animalday_dict = core.parse_path_to_animalday(
            lan_folder, animal_param=self.animal_param, day_sep=self.day_sep, mode=self.read_mode
        )
        row["animalday"] = animalday_dict["animalday"]
        row["animal"] = animalday_dict["animal"]
        row["day"] = animalday_dict["day"]
        row["genotype"] = animalday_dict["genotype"]
        row["duration"] = lan.LongRecording.get_dur_fragment(window_s, idx)
        row["endfile"] = lan.get_file_end(idx)

        frag_dt = lan.LongRecording.get_datetime_fragment(window_s, idx)
        row["timestamp"] = frag_dt
        row["isday"] = core.utils.is_day(frag_dt)

        return row

    def _process_fragment_features(self, idx, features, lan: core.LongRecordingAnalyzer, kwargs: dict):
        row = {}
        for feat in features:
            func = getattr(lan, f"compute_{feat}")
            if callable(func):
                row[feat] = func(idx, **kwargs)
            else:
                raise AttributeError(f"Invalid function {func}")
        return row


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
        """
        Args:
            result (pd.DataFrame): Result comes from AnimalOrganizer.compute_windowed_analysis()
            animal_id (str, optional): Identifier for the animal where result was computed from. Defaults to None.
            genotype (str, optional): Genotype of animal. Defaults to None.
            channel_names (list[str], optional): List of channel names. Defaults to None.
            assume_channels (bool, optional): If true, assumes channel names according to AnimalFeatureParser.DEFAULT_CHNUM_TO_NAME. Defaults to False.
            bad_channels_dict (dict[str, list[str]], optional): Dictionary of channels to reject for each recording session. Defaults to {}.
            suppress_short_interval_error (bool, optional): If True, suppress ValueError for short intervals between timestamps. Useful for aggregated WARs with large window sizes. Defaults to False.
        """
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
            channel_names=self.channel_names.copy() if self.channel_names is not None else None,
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
                warnings.warn("Timestamps are not sorted. Sorting result DataFrame by timestamp.")
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

                if pct_short > 1.0 and not self.suppress_short_interval_error:  # More than 1% of intervals are short
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

        self._feature_columns = [x for x in self.result.columns if x in constants.FEATURES]
        self._nonfeature_columns = [x for x in self.result.columns if x not in constants.FEATURES]
        self.animaldays = self.result.loc[:, "animalday"].unique()

        self.channel_abbrevs = [
            core.parse_chname_to_abbrev(x, assume_from_number=self.assume_from_number) for x in self.channel_names
        ]

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
            raise ValueError(f"Target channels must be unique. Found duplicates: {duplicates}")

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
                case _ if feature in constants.LINEAR_FEATURES + constants.BAND_FEATURES:
                    if feature in constants.BAND_FEATURES:
                        df_bands = pd.DataFrame(result[feature].tolist())
                        vals = np.array(df_bands.values.tolist())
                        vals = vals.transpose((0, 2, 1))
                        keys = df_bands.keys()
                    else:
                        vals = np.array(result[feature].tolist())

                    new_vals = np.full((vals.shape[0], len(target_channels), *vals.shape[2:]), np.nan)  # dubious

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
                    new_shape = list(vals.shape[:-2]) + [len(target_channels), len(target_channels)]
                    new_vals = np.full(new_shape, np.nan)

                    # Map original channels to target channels
                    for i, ch1 in enumerate(channel_names):
                        if ch1 in channel_map:
                            for j, ch2 in enumerate(channel_names):
                                if ch2 in channel_map:
                                    new_vals[..., channel_map[ch1], channel_map[ch2]] = vals[..., i, j]

                    if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
                        result[feature] = [dict(zip(keys, vals)) for vals in new_vals]
                    else:
                        result[feature] = [list(x) for x in new_vals]

                case _ if feature in constants.HIST_FEATURES:
                    coords = np.array([x[0] for x in result[feature].tolist()])
                    vals = np.array([x[1] for x in result[feature].tolist()])
                    new_vals = np.full((*vals.shape[0:-1], len(target_channels)), np.nan)

                    for i, ch in enumerate(channel_names):
                        if ch in channel_map:
                            new_vals[:, ..., channel_map[ch]] = vals[:, ..., i]

                    result[feature] = [(coords[i], new_vals[i]) for i in range(len(coords))]

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
        sars: list[Union["SpikeAnalysisResult", "FrequencyDomainSpikeAnalysisResult"]],
        read_mode: Literal["sa", "mne"] = "sa",
        inplace=True
    ):
        """
        Integrate spike analysis results into WAR by adding nspike/lognspike features.

        This method extracts spike timing information from spike detection results and bins
        them according to the WAR's time windows, adding spike count features to each row.

        Args:
            sars: List of SpikeAnalysisResult or FrequencyDomainSpikeAnalysisResult objects.
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
                            spike_times.extend(sa.sorting.get_unit_spike_train(unit_id=unit).tolist())
                        spike_times = np.array(spike_times) / sa.sorting.get_sampling_frequency()
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
        spike_counts = grouped.apply(lambda x: _bin_spike_df(x, spikes_channel=spike_counts[x.name]))
        spike_counts: pd.Series = spike_counts.explode()

        if spike_counts.size != self.result.shape[0]:
            logging.warning(f"Spike counts size {spike_counts.size} does not match result size {self.result.shape[0]}")

        result = self.result.copy()
        result["nspike"] = spike_counts.tolist()
        result["lognspike"] = list(core.log_transform(np.stack(result["nspike"].tolist(), axis=0)))
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
        info.append(f"channel_names: {', '.join(self.channel_names) if self.channel_names else 'None'}")

        return "\n".join(info)

    def get_result(self, features: list[str], exclude: list[str] = [], allow_missing=False):
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
        self, features: list[str], exclude: list[str] = [], df: pd.DataFrame = None, groupby="animalday"
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
        result_grouped, result_validcols = self.__get_groups(features=features, exclude=exclude, df=df, groupby=groupby)
        features = _sanitize_feature_request(features, exclude)

        avg_results = []
        for f in features:
            if f in result_validcols:
                avg_result_col = result_grouped.apply(self._average_feature, f, "duration", include_groups=False)
                avg_result_col.name = f
                avg_results.append(avg_result_col)
            else:
                logging.warning(f"{f} not calculated, skipping")

        return pd.concat(avg_results, axis=1)

    def __get_groups(self, features: list[str], exclude: list[str] = [], df: pd.DataFrame = None, groupby="animalday"):
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

    def get_filter_high_beta(self, df: pd.DataFrame = None, max_beta_prop=0.4, **kwargs):
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
            raise ValueError("psdfrac or psdband+psdtotal required for beta power filtering")

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
            self.channel_abbrevs if use_abbrevs or use_abbrevs is None else self.channel_names
        )  # Match to appropriate target
        if use_abbrevs is None:  # Match channels as abbreviations
            bad_channels = [
                core.parse_chname_to_abbrev(ch, assume_from_number=self.assume_from_number) for ch in bad_channels
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
                    core.parse_chname_to_abbrev(ch, assume_from_number=self.assume_from_number) for ch in bad_channels
                ]
            )

            if save_bad_channels == "overwrite":
                # Replace entire dict with bad channels applied to all sessions
                self.bad_channels_dict = {animalday: channels_to_save.copy() for animalday in animaldays}
            elif save_bad_channels == "union":
                # Merge with existing bad channels for all sessions
                updated_dict = self.bad_channels_dict.copy()
                for animalday in animaldays:
                    if animalday in updated_dict:
                        # Union of existing and new channels
                        updated_dict[animalday] = list(set(updated_dict[animalday]) | set(channels_to_save))
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

            channel_targets = self.channel_abbrevs if use_abbrevs or use_abbrevs is None else self.channel_names
            if use_abbrevs is None:
                bad_channels = [
                    core.parse_chname_to_abbrev(ch, assume_from_number=self.assume_from_number) for ch in bad_channels
                ]

            # Get indices for this recording session
            session_indices = group.index

            # Apply channel filtering for this session
            for ch in bad_channels:
                if ch in channel_targets:
                    ch_idx = channel_targets.index(ch)
                    mask[session_indices, ch_idx] = False
                else:
                    logging.warning(f"Channel {ch} not found in {channel_targets} for session {animalday}")

        # Save bad channels to self.bad_channels_dict if requested
        if save_bad_channels is not None and bad_channels_dict is not None:
            if save_bad_channels == "overwrite":
                self.bad_channels_dict = bad_channels_dict.copy()
            elif save_bad_channels == "union":
                # Merge with existing bad channels per session
                updated_dict = self.bad_channels_dict.copy()
                for animalday, channels in bad_channels_dict.items():
                    if animalday in updated_dict:
                        # Union of existing and new channels
                        updated_dict[animalday] = list(set(updated_dict[animalday]) | set(channels))
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
            raise ValueError("Cannot calculate window duration - 'duration' column missing")

        window_duration = self.result["duration"].median()
        structure_size = max(1, int(smoothing_seconds / window_duration))

        if structure_size <= 1:
            return filter_mask

        smoothed_mask = filter_mask.copy()
        for ch_idx in range(filter_mask.shape[1]):
            channel_mask = filter_mask[:, ch_idx]
            # Opening removes small isolated artifacts
            channel_mask = binary_opening(channel_mask, structure=np.ones(structure_size))
            # Closing fills small gaps in valid data
            channel_mask = binary_closing(channel_mask, structure=np.ones(structure_size))
            smoothed_mask[:, ch_idx] = channel_mask

        return smoothed_mask

    def filter_morphological_smoothing(self, smoothing_seconds: float) -> "WindowAnalysisResult":
        """Apply morphological smoothing to all data.

        Args:
            smoothing_seconds (float): Time window in seconds for morphological operations

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        # Start with all-True mask and smooth it
        base_mask = np.ones((len(self.result), len(self.channel_names)), dtype=bool)
        smoothed_mask = self.get_filter_morphological_smoothing(base_mask, smoothing_seconds)
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
        logging.debug(f"filt_bool_all.shape: {filt_bool_all.shape}")  # (windows, channels)

        # Apply morphological smoothing if requested
        if morphological_smoothing_seconds is not None:
            if "duration" not in self.result.columns:
                raise ValueError("Cannot calculate window duration - 'duration' column missing from result dataframe")
            window_duration = self.result["duration"].median()

            # Calculate number of windows for the smoothing
            structure_size = max(1, int(morphological_smoothing_seconds / window_duration))

            if structure_size > 1:
                logging.info(
                    f"Applying morphological smoothing with {structure_size} windows ({morphological_smoothing_seconds}s / {window_duration}s per window)"
                )
                # Apply channel-wise temporal smoothing (each channel processed independently)
                # This avoids spatial assumptions while smoothing temporal artifacts
                for ch_idx in range(filt_bool_all.shape[1]):
                    channel_mask = filt_bool_all[:, ch_idx]
                    # Opening removes small isolated artifacts
                    channel_mask = binary_opening(channel_mask, structure=np.ones(structure_size))
                    # Closing fills small gaps in valid data
                    channel_mask = binary_closing(channel_mask, structure=np.ones(structure_size))
                    filt_bool_all[:, ch_idx] = channel_mask
            else:
                logging.info("Skipping morphological smoothing - structure size would be 1 (no effect)")

        # Filter windows based on number of valid channels
        valid_channels_per_window = np.sum(filt_bool_all, axis=1)  # axis 1 = channel
        window_mask = valid_channels_per_window >= min_valid_channels  # True if window has enough valid channels
        filt_bool_all = filt_bool_all & window_mask[:, np.newaxis]  # Apply window mask to all channels

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

    def filter_reject_channels(self, bad_channels: list[str], use_abbrevs: bool = None) -> "WindowAnalysisResult":
        """Filter out specified bad channels.

        Args:
            bad_channels (list[str]): List of channel names to reject
            use_abbrevs (bool, optional): Whether to use abbreviations. Defaults to None.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_reject_channels(bad_channels=bad_channels, use_abbrevs=use_abbrevs)
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
        self, filter_config: dict = None, min_valid_channels: int = 3, morphological_smoothing_seconds: float = None
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
            logging.info(f"{filter_name}: filtered {filt_bool.size - np.count_nonzero(filt_bool)}/{filt_bool.size}")

        # Combine all filter masks
        if filt_bools:
            filt_bool_all = np.prod(np.stack(filt_bools, axis=-1), axis=-1).astype(bool)
        else:
            filt_bool_all = np.ones((len(self.result), len(self.channel_names)), dtype=bool)

        # Apply morphological smoothing if requested (either from config or parameter)
        if morphological_params or morphological_smoothing_seconds is not None:
            if morphological_params:
                smoothing_seconds = morphological_params["smoothing_seconds"]
            else:
                smoothing_seconds = morphological_smoothing_seconds

            filt_bool_all = self.get_filter_morphological_smoothing(filt_bool_all, smoothing_seconds)
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
                logging.info(f"Skipping {feat} because it is not in result")
                continue
            logging.info(f"Filtering {feat}")
            match feat:  # NOTE refactor this to use constants
                case "rms" | "ampvar" | "psdtotal" | "nspike" | "logrms" | "logampvar" | "logpsdtotal" | "lognspike":
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
            "channel_names": self.channel_abbrevs if save_abbrevs_as_chnames else self.channel_names,
            "assume_from_number": False if save_abbrevs_as_chnames else self.assume_from_number,
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
            raise ValueError("LOF scores not available in this WAR. Compute LOF scores first.")

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
            raise ValueError("LOF scores not available in this WAR. Compute LOF scores first.")

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
        self, ground_truth_bad_channels: dict = None, threshold: float = None, evaluation_channels: list[str] = None
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
            raise ValueError("LOF scores not available in this WAR. Run compute_bad_channels() first.")

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
                    key: value for key, value in self.bad_channels_dict.items() if key in lof_keys
                }

                logging.info(
                    f"Using filtered bad_channels_dict as ground truth with {len(ground_truth_bad_channels)} animal-day sessions"
                )
            else:
                raise ValueError("No ground truth provided and self.bad_channels_dict is empty.")

        # Get all channels if no subset specified
        if evaluation_channels is None:
            evaluation_channels = self.channel_names

        y_true_list = []
        y_pred_list = []

        # Debug: Log what we're working with
        logging.debug(f"evaluate_lof_threshold_binary: evaluation_channels = {evaluation_channels}")
        logging.debug(
            f"evaluate_lof_threshold_binary: ground_truth_bad_channels keys = {list(ground_truth_bad_channels.keys())}"
        )
        logging.debug(f"evaluate_lof_threshold_binary: lof_scores_dict keys = {list(self.lof_scores_dict.keys())}")

        # Iterate through each animal-day and evaluate channels
        for animalday, lof_data in self.lof_scores_dict.items():
            if "lof_scores" not in lof_data or "channel_names" not in lof_data:
                raise ValueError(
                    f"Invalid LOF data for {animalday}: missing required fields 'lof_scores' or 'channel_names'"
                )

            scores = np.array(lof_data["lof_scores"])
            channel_names = lof_data["channel_names"]

            # Get ground truth bad channels for this animal-day
            animalday_bad_channels = ground_truth_bad_channels.get(animalday, set())

            # Debug: Log details for this animal-day
            logging.debug(f"Processing {animalday}: channel_names = {channel_names}")
            logging.debug(f"Processing {animalday}: animalday_bad_channels = {animalday_bad_channels}")
            logging.debug(f"Processing {animalday}: scores shape = {scores.shape}")

            # Evaluate each channel in the evaluation subset
            channels_processed = 0
            for i, channel in enumerate(channel_names):
                if (
                    channel in evaluation_channels
                    or parse_chname_to_abbrev(channel, strict_matching=False) in evaluation_channels
                ):
                    channels_processed += 1

                    # Ground truth: 1 if channel is marked as bad, 0 otherwise
                    is_bad_channel = (
                        channel in animalday_bad_channels
                        or parse_chname_to_abbrev(channel, strict_matching=False) in animalday_bad_channels
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
                        logging.info(f"LOF PREDICTION: {channel} has score {scores[i]:.3f} > threshold {threshold}")

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
                    raise ValueError(f"Expected exactly one pickle file in {folder_path}, found {len(pkl_files)}")
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
                    raise ValueError(f"Expected exactly one json file in {folder_path}, found {len(json_files)}")
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

    def aggregate_time_windows(self, groupby: list[str] | str = ["animalday", "isday"]) -> None:
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
            raise ValueError(f"groupby must be from ['animalday', 'isday']. Got {groupby}")
        if not all(col in self.result.columns for col in groupby):
            raise ValueError(f"Columns {groupby} not found in result. Columns: {self.result.columns.tolist()}")

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
                    raise ValueError(f"Column {col} is not constant in groups: {non_constant_groups}")
                agg_dict[col] = lambda df, col=col: df[col].iloc[0]

        if "duration" in self.result.columns:
            agg_dict["duration"] = lambda df: np.sum(df["duration"])

        if "endfile" in self.result.columns:
            agg_dict["endfile"] = lambda df: df["endfile"].iloc[-1]

        if "timestamp" in self.result.columns:
            agg_dict["timestamp"] = lambda df: df["timestamp"].iloc[0]

        for feat in features:
            agg_dict[feat] = lambda df, feat=feat: self._average_feature(df, feat, "duration")

        aggregated_df = result_grouped.apply(
            lambda df: pd.Series({col: agg_dict[col](df) for col in self.result.columns if col not in groupby})
        )

        self.result = aggregated_df.reset_index(drop=False)  # Keep animalday/isday as a column

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
            self.result["animalday"] = self.result["animalday"].str.replace(self.animal_id, new_animal_id)
        self.animal_id = new_animal_id

        self.__update_instance_vars()


def bin_spike_times(spike_times: list[float], fragment_durations: list[float]) -> list[int]:
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


class SpikeAnalysisResult(AnimalFeatureParser):
    def __init__(
        self,
        result_sas: list[si.SortingAnalyzer],
        result_mne: mne.io.RawArray = None,
        animal_id: str = None,
        genotype: str = None,
        animal_day: str = None,
        bin_folder_name: str = None,
        metadata: core.DDFBinaryMetadata = None,
        channel_names: list[str] = None,
        assume_from_number=False,
    ) -> None:
        """
        Args:
            result (list[si.SortingAnalyzer]): Result comes from AnimalOrganizer.compute_spike_analysis(). Each SortingAnalyzer is a single channel.
            animal_id (str, optional): Identifier for the animal where result was computed from. Defaults to None.
            genotype (str, optional): Genotype of animal. Defaults to None.
            channel_names (list[str], optional): List of channel names. Defaults to None.
            assume_channels (bool, optional): If true, assumes channel names according to AnimalFeatureParser.DEFAULT_CHNUM_TO_NAME. Defaults to False.
        """
        self.result_sas = result_sas
        self.result_mne = result_mne
        if (result_mne is None) == (result_sas is None):
            raise ValueError("Exactly one of result_sas or result_mne must be provided")
        self.animal_id = animal_id
        self.genotype = genotype
        self.animal_day = animal_day
        self.bin_folder_name = bin_folder_name
        self.metadata = metadata
        self.channel_names = channel_names
        self.assume_from_number = assume_from_number
        self.channel_abbrevs = [
            core.parse_chname_to_abbrev(x, assume_from_number=assume_from_number) for x in self.channel_names
        ]

        logging.info(f"Channel names: \t{self.channel_names}")
        logging.info(f"Channel abbreviations: \t{self.channel_abbrevs}")

    def convert_to_mne(self, chunk_len: float = 60, save_raw=True) -> mne.io.RawArray:
        if self.result_mne is None:
            result_mne = SpikeAnalysisResult.convert_sas_to_mne(self.result_sas, chunk_len)
            if save_raw:
                self.result_mne = result_mne
            else:
                return result_mne
        return self.result_mne

    def save_fif_and_json(
        self,
        folder: str | Path,
        convert_to_mne=True,
        make_folder=True,
        slugify_filebase=True,
        save_abbrevs_as_chnames=False,
        overwrite=False,
    ):
        """Archive spike analysis result into the folder specified, as a fif and json file.

        Args:
            folder (str | Path): Destination folder to save results to
            convert_to_mne (bool, optional): If True, convert the SortingAnalyzers to a MNE RawArray if self.result_mne is None. Defaults to True.
            make_folder (bool, optional): If True, create the folder if it doesn't exist. Defaults to True.
            slugify_filebase (bool, optional): If True, slugify the filebase (replace special characters). Defaults to True.
            save_abbrevs_as_chnames (bool, optional): If True, save the channel abbreviations as the channel names in the json file. Defaults to False.
            overwrite (bool, optional): If True, overwrite the existing files. Defaults to False.
        """
        if self.result_mne is None:
            if convert_to_mne:
                result_mne = self.convert_to_mne(save_raw=True)
                if result_mne is None:
                    warnings.warn("No SortingAnalyzers found, skipping saving")
                    return
            else:
                raise ValueError("No MNE RawArray found, and convert_to_mne is False. Run convert_to_mne() first.")
        else:
            result_mne = self.result_mne

        folder = Path(folder)
        if make_folder:
            folder.mkdir(parents=True, exist_ok=True)

        if slugify_filebase:
            filebase = folder / slugify(f"{self.animal_id}-{self.genotype}-{self.animal_day}")
        else:
            filebase = folder / f"{self.animal_id}-{self.genotype}-{self.animal_day}"
        filebase = str(filebase)

        if not overwrite:
            if filebase + ".json" in folder.glob("*.json"):
                raise FileExistsError(f"File {filebase}.json already exists")
            if filebase + ".fif" in folder.glob("*.fif"):
                raise FileExistsError(f"File {filebase}.fif already exists")
        else:
            for f in folder.glob("*"):
                f.unlink()
        result_mne.save(filebase + "-raw.fif", overwrite=overwrite)
        del result_mne

        json_dict = {
            "animal_id": self.animal_id,
            "genotype": self.genotype,
            "animal_day": self.animal_day,
            "bin_folder_name": self.bin_folder_name,
            "metadata": self.metadata.metadata_path,
            "channel_names": self.channel_abbrevs if save_abbrevs_as_chnames else self.channel_names,
            "assume_from_number": False if save_abbrevs_as_chnames else self.assume_from_number,
        }
        with open(filebase + ".json", "w") as f:
            json.dump(json_dict, f, indent=2)

    @classmethod
    def load_fif_and_json(cls, folder: str | Path):
        folder = Path(folder)
        if not folder.exists():
            raise ValueError(f"Folder {folder} does not exist")

        fif_files = list(folder.glob("*.fif"))  # there may be more than 1 fif file
        json_files = list(folder.glob("*.json"))

        if len(json_files) != 1:
            raise ValueError(f"Expected exactly one json file in {folder}")

        fif_path = fif_files[0]
        json_path = json_files[0]

        with open(json_path, "r") as f:
            data = json.load(f)
        # data['metadata'] = core.DDFBinaryMetadata(data['metadata'])
        data["result_mne"] = mne.io.read_raw_fif(fif_path)
        data["result_sas"] = None
        return cls(**data)

    @staticmethod
    def convert_sas_to_mne(sas: list[si.SortingAnalyzer], chunk_len: float = 60) -> mne.io.RawArray:
        """Convert a list of SortingAnalyzers to a MNE RawArray.

        Args:
            sas (list[si.SortingAnalyzer]): The list of SortingAnalyzers to convert
            chunk_len (float, optional): The length of the chunks to use for the conversion. Defaults to 60.

        Returns:
            mne.io.RawArray: The converted RawArray, with spikes labeled as annotations
        """
        if len(sas) == 0:
            return None

        # Check that all SortingAnalyzers have the same sampling frequency
        sfreqs = [sa.recording.get_sampling_frequency() for sa in sas]
        if not all(sf == sfreqs[0] for sf in sfreqs):
            raise ValueError(f"All SortingAnalyzers must have the same sampling frequency. Got frequencies: {sfreqs}")

        # Preallocate data array
        total_frames = int(sas[0].recording.get_duration() * sfreqs[0])
        n_channels = len(sas)
        data = np.empty((n_channels, total_frames))
        logging.debug(f"Data shape: {data.shape}")

        # Fill data array one channel at a time
        for i, sa in enumerate(sas):
            logging.debug(f"Converting channel {i + 1} of {n_channels}")
            data[i, :] = SpikeAnalysisResult.convert_sa_to_np(sa, chunk_len)

        channel_names = [str(sa.recording.get_channel_ids().item()) for sa in sas]
        logging.debug(f"Channel names: {channel_names}")
        sfreq = sfreqs[0]

        # Extract spike times for each unit and create annotations
        onset = []
        description = []
        for sa in sas:
            for unit_id in sa.sorting.get_unit_ids():
                spike_train = sa.sorting.get_unit_spike_train(unit_id)
                # Convert to seconds and filter to recording duration
                spike_times = spike_train / sa.sorting.get_sampling_frequency()
                mask = spike_times < sa.recording.get_duration()
                spike_times = spike_times[mask]

                # Create annotation for each spike
                onset.extend(spike_times)
                description.extend(
                    [sa.recording.get_channel_ids().item()] * len(spike_times)
                )  # collapse all units into 1 spike train
        annotations = mne.Annotations(onset, duration=0, description=description)

        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data=data, info=info)
        raw = raw.set_annotations(annotations)
        return raw

    @staticmethod
    def convert_sa_to_np(sa: si.SortingAnalyzer, chunk_len: float = 60) -> np.ndarray:
        """Convert a SortingAnalyzer to an MNE RawArray.

        Args:
            sa (si.SortingAnalyzer): The SortingAnalyzer to convert. Must have only 1 channel.
            chunk_len (float, optional): The length of the chunks to use for the conversion. Defaults to 60.
        Returns:
            np.ndarray: The converted traces
        """
        # Check that SortingAnalyzer only has 1 channel
        if len(sa.recording.get_channel_ids()) != 1:
            raise ValueError(
                f"Expected SortingAnalyzer to have 1 channel, but got {len(sa.recording.get_channel_ids())} channels"
            )

        rec = sa.recording
        logging.debug(f"Recording info: {rec}")

        # Calculate total number of frames and chunks
        total_frames = int(rec.get_duration() * rec.get_sampling_frequency())
        frames_per_chunk = round(chunk_len * rec.get_sampling_frequency())
        n_chunks = total_frames // frames_per_chunk

        traces = np.empty(total_frames)

        for j in range(n_chunks):
            start_frame = j * frames_per_chunk
            if j == n_chunks - 1:
                end_frame = total_frames
            else:
                end_frame = (j + 1) * frames_per_chunk
            traces[start_frame:end_frame] = rec.get_traces(
                start_frame=start_frame, end_frame=end_frame, return_scaled=True
            ).flatten()
        traces *= 1e-6  # convert from uV to V
        return traces
