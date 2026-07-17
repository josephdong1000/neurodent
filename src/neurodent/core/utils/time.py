"""Day/night classification, timestamp mapping and validation."""

import warnings

from datetime import datetime, timedelta

import numpy as np


def is_day(dt: datetime, sunrise=6, sunset=18):
    """
    Check if a datetime object is during the day.

    Args:
        dt (datetime): Datetime object to check
        sunrise (int, optional): Sunrise hour (0-23). Defaults to 6.
        sunset (int, optional): Sunset hour (0-23). Defaults to 18.

    Returns:
        bool: True if the datetime is during the day, False otherwise

    Raises:
        TypeError: If dt is not a datetime object
    """
    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime object, got {type(dt).__name__}")
    return sunrise <= dt.hour < sunset


class TimestampMapper:
    """
    Map each fragment to its source file's timestamp.

    This class provides functionality to map data fragments back to their original
    file timestamps when data has been concatenated from multiple files with
    different recording times.

    Attributes:
        file_end_datetimes (list[datetime]): The end datetimes of each source file.
        file_durations (list[float]): The durations of each source file in seconds.
        file_start_datetimes (list[datetime]): Computed start datetimes of each file.
        cumulative_durations (np.ndarray): Cumulative sum of file durations.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> # Set up files with known end times and durations
        >>> end_times = [datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 13, 0)]
        >>> durations = [3600.0, 1800.0]  # 1 hour, 30 minutes
        >>> mapper = TimestampMapper(end_times, durations)
        >>>
        >>> # Get timestamp for fragment at index 2 with 60s fragments
        >>> timestamp = mapper.get_fragment_timestamp(2, 60.0)
        >>> print(timestamp)
        2023-01-01 11:02:00
    """

    def __init__(self, file_end_datetimes: list[datetime], file_durations: list[float]):
        """
        Initialize the TimestampMapper.

        Args:
            file_end_datetimes (list[datetime]): The end datetimes of each file.
            file_durations (list[float]): The durations of each file in seconds.

        Raises:
            ValueError: If the lengths of file_end_datetimes and file_durations don't match.
        """
        if len(file_end_datetimes) != len(file_durations):
            raise ValueError("file_end_datetimes and file_durations must have the same length")

        self.file_end_datetimes = file_end_datetimes
        self.file_durations = file_durations

        self.file_start_datetimes = [
            file_end_datetime - timedelta(seconds=file_duration)
            for file_end_datetime, file_duration in zip(self.file_end_datetimes, self.file_durations)
        ]
        self.cumulative_durations = np.cumsum(self.file_durations)

    def get_fragment_timestamp(self, fragment_idx: int, fragment_len_s: float) -> datetime:
        """
        Get the timestamp for a specific fragment based on its index and length.

        Args:
            fragment_idx (int): The index of the fragment (0-based).
            fragment_len_s (float): The length of each fragment in seconds.

        Returns:
            datetime: The timestamp corresponding to the start of the specified fragment.

        Examples:
            >>> # Get timestamp for the 5th fragment (index 4) with 30-second fragments
            >>> timestamp = mapper.get_fragment_timestamp(4, 30.0)
            >>> # This returns the timestamp 2 minutes into the first file
        """
        # Find which file this fragment belongs to
        fragment_start_time = fragment_idx * fragment_len_s
        file_idx = np.searchsorted(self.cumulative_durations, fragment_start_time)
        file_idx = min(file_idx, len(self.cumulative_durations) - 1)

        offset_in_file = fragment_start_time - self.cumulative_durations[file_idx]  # Negative

        # Return actual timestamp + offset
        return self.file_end_datetimes[file_idx] + timedelta(seconds=offset_in_file)


def validate_timestamps(timestamps: list[datetime], gap_threshold_seconds: float = 60) -> list[datetime]:
    """
    Validate that timestamps are in chronological order and check for large gaps.

    Args:
        timestamps (list[datetime]): List of timestamps to validate
        gap_threshold_seconds (float, optional): Threshold in seconds for warning about large gaps. Defaults to 60.

    Returns:
        list[datetime]: The validated timestamps in chronological order

    Raises:
        ValueError: If no valid timestamps are provided
    """
    if not timestamps:
        raise ValueError("No timestamps provided for validation")

    valid_timestamps = [ts for ts in timestamps if ts is not None]
    if len(valid_timestamps) < len(timestamps):
        warnings.warn(f"Found {len(timestamps) - len(valid_timestamps)} None timestamps that were filtered out")

    if not valid_timestamps:
        raise ValueError("No valid timestamps found (all were None)")

    # Check chronological order
    sorted_timestamps = sorted(valid_timestamps)
    if valid_timestamps != sorted_timestamps:
        warnings.warn("Timestamps are not in chronological order. This may cause issues with the data.")

    # Check for large gaps between consecutive timestamps
    for i in range(1, len(valid_timestamps)):
        gap = valid_timestamps[i] - valid_timestamps[i - 1]
        gap_seconds = gap.total_seconds()

        if gap_seconds > gap_threshold_seconds:
            warnings.warn(
                f"Large gap detected between timestamps: {gap} exceeds threshold of {gap_threshold_seconds} seconds"
            )

    return valid_timestamps
