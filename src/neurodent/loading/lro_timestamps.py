"""File timestamp validation, manual-datetime handling, and date accessors.

Mixin for :class:`~neurodent.loading.long_recording_organizer.LongRecordingOrganizer`.
"""

import logging
import warnings
from datetime import datetime, timedelta

from neurodent.core.utils import TimestampMapper


class LroTimestampsMixin:
    """Mixin: see module docstring."""

    def get_datetime_fragment(self, fragment_len_s, fragment_idx):
        """
        Get the datetime for a specific fragment using the timestamp mapper.

        Args:
            fragment_len_s (float): Length of each fragment in seconds
            fragment_idx (int): Index of the fragment to get datetime for

        Returns:
            datetime: The datetime corresponding to the start of the fragment

        Raises:
            ValueError: If timestamp mapper is not initialized (only available in 'bin' mode)
        """
        return TimestampMapper(
            self.file_end_datetimes, self.file_durations
        ).get_fragment_timestamp(fragment_idx, fragment_len_s)

    def _validate_manual_time_params(self):
        """Validate that manual time parameters are correctly specified."""
        if self.manual_datetimes is not None:
            if not isinstance(self.manual_datetimes, (datetime, list, tuple)):
                raise ValueError(
                    "manual_datetimes must be a datetime object or list of datetime objects"
                )

    def _validate_timestamps_for_mode(self, mode: str, expected_n_files: int = None):
        """Validate that manual timestamps are provided when required for specific modes.

        Args:
            mode (str): The processing mode ('si', 'mne', or 'bin')
            expected_n_files (int, optional): Expected number of files for validation

        Raises:
            ValueError: If timestamps are required but not provided or if count mismatch
        """
        if mode in ["si", "mne"]:
            if self.manual_datetimes is None:
                import logging

                logging.warning(
                    f"manual_datetimes must be provided for {mode} mode when no CSV metadata is available, falling back to file creation times if possible"
                )

            # If list provided and expected files known, validate length
            if expected_n_files is not None and isinstance(self.manual_datetimes, list):
                if len(self.manual_datetimes) != expected_n_files:
                    raise ValueError(
                        f"manual_datetimes length ({len(self.manual_datetimes)}) must match "
                        f"number of input files ({expected_n_files}) for {mode} mode"
                    )

    def _compute_manual_file_datetimes(
        self, n_files: int, durations: list[float]
    ) -> list[datetime]:
        """Compute file end datetimes based on manual time specifications.

        Args:
            n_files (int): Number of files
            durations (list[float]): Duration of each file in seconds

        Returns:
            list[datetime]: End datetime for each file

        Raises:
            ValueError: If manual_datetimes length doesn't match number of files
        """
        if self.manual_datetimes is None:
            return None

        if isinstance(self.manual_datetimes, list):
            # List of times provided - one per file
            if len(self.manual_datetimes) != n_files:
                raise ValueError(
                    f"manual_datetimes length ({len(self.manual_datetimes)}) must match number of files ({n_files})"
                )

            # Convert start times to end times or vice versa
            if self.datetimes_are_start:
                # Convert start times to end times
                file_end_datetimes = [
                    start_time + timedelta(seconds=duration)
                    for start_time, duration in zip(self.manual_datetimes, durations)
                ]
            else:
                # Use as end times directly
                file_end_datetimes = list(self.manual_datetimes)

            # Check contiguity (warn instead of error)
            self._validate_file_contiguity(file_end_datetimes, durations)

            return file_end_datetimes

        else:
            # Single datetime provided - global start or end time
            if self.datetimes_are_start:
                # Global start time - compute cumulative end times
                current_time = self.manual_datetimes
                file_end_datetimes = []
                for duration in durations:
                    current_time += timedelta(seconds=duration)
                    file_end_datetimes.append(current_time)
                return file_end_datetimes
            else:
                # Global end time - work backwards
                total_duration = sum(durations)
                start_time = self.manual_datetimes - timedelta(seconds=total_duration)
                current_time = start_time
                file_end_datetimes = []
                for duration in durations:
                    current_time += timedelta(seconds=duration)
                    file_end_datetimes.append(current_time)
                return file_end_datetimes

    def _validate_file_contiguity(
        self, file_end_datetimes: list[datetime], durations: list[float]
    ):
        """Check that files are contiguous in time and warn if they're not.

        Args:
            file_end_datetimes (list[datetime]): End datetime for each file
            durations (list[float]): Duration of each file in seconds
        """
        if len(file_end_datetimes) <= 1:
            return  # Single file or no files - nothing to check

        tolerance_seconds = 1.0  # Allow 1 second tolerance for rounding errors

        for i in range(len(file_end_datetimes) - 1):
            # Start time of next file should equal end time of current file
            current_end = file_end_datetimes[i]
            next_start = file_end_datetimes[i + 1] - timedelta(seconds=durations[i + 1])

            gap_seconds = (next_start - current_end).total_seconds()
            if gap_seconds > tolerance_seconds:
                warnings.warn(
                    f"Files may not be contiguous: gap of {gap_seconds:.2f}s between "
                    f"file {i} (ends {current_end}) and file {i + 1} (starts {next_start}). "
                    f"Tolerance is {tolerance_seconds}s."
                )
            elif gap_seconds < -tolerance_seconds:
                warnings.warn(
                    f"Files may overlap: negative gap of {gap_seconds:.2f}s between "
                    f"file {i} (ends {current_end}) and file {i + 1} (starts {next_start}). "
                    f"Tolerance is {tolerance_seconds}s."
                )

    def finalize_file_timestamps(self):
        """Finalize file timestamps using manual times if provided, otherwise validate CSV times."""
        logging.info("Finalizing file timestamps")
        if not hasattr(self, "file_durations") or not self.file_durations:
            return  # No file durations available yet

        manual_file_datetimes = self._compute_manual_file_datetimes(
            len(self.file_durations), self.file_durations
        )

        if manual_file_datetimes is not None:
            self.file_end_datetimes = manual_file_datetimes
            logging.info(
                f"Using manual timestamps: {len(manual_file_datetimes)} file end times specified"
            )
        else:
            # Check if CSV times are sufficient (only for bin mode)
            if hasattr(self, "file_end_datetimes") and self.file_end_datetimes:
                if all(x is None for x in self.file_end_datetimes):
                    raise ValueError(
                        "No dates found in any metadata object and no manual times specified!"
                    )
                logging.info("Using CSV metadata timestamps")
            else:
                # For si/mne modes, manual timestamps are ideally required
                logging.warning(
                    "manual_datetimes must be provided when no CSV metadata is available! Falling back to file creation times if possible."
                )

    def get_date_string(self) -> str:
        """
        Get the string representation of the recording date (Start Time).

        Returns:
            str: Date string in format "%b-%d-%Y" (e.g. "Jan-21-2022").

        Raises:
            ValueError: If no timestamps are available in the recording.
        """
        if not hasattr(self, "file_end_datetimes") or not self.file_end_datetimes:
            raise ValueError("Cannot determine date: No file timestamps available.")

        # Find first valid timestamp
        first_valid_idx = next(
            (i for i, x in enumerate(self.file_end_datetimes) if x is not None), None
        )

        if first_valid_idx is None:
            raise ValueError("Cannot determine date: All file timestamps are None.")

        end_time = self.file_end_datetimes[first_valid_idx]
        duration = self.file_durations[first_valid_idx]

        start_time = end_time - timedelta(seconds=duration)
        return start_time.strftime("%b-%d-%Y")
