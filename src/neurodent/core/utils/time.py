"""Time/date parsing, timestamp mapping and validation."""

import logging
import re
import warnings

from datetime import datetime, timedelta
from typing import Literal, Optional

import dateutil.parser
import numpy as np
from dateutil.parser import ParserError

from neurodent import constants


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


def parse_str_to_day(
    string: str,
    sep: str = None,
    parse_params: dict = None,
    parse_mode: Literal["full", "split", "window", "all"] = "split",
    date_patterns: list[tuple[str, str]] = None,
) -> datetime:
    """
    DEPRECATED: Use FileDiscoverer with {session} placeholder in pattern instead.

    Parses the filename of a binfolder to get the day.

    Args:
        string (str): String to parse.
        sep (str, optional): Separator to split string by. If None, split by whitespace. Defaults to None.
        parse_params (dict, optional): Parameters to pass to dateutil.parser.parse. Defaults to {'fuzzy':True}.
        parse_mode (Literal["full", "split", "window", "all"], optional): Mode for parsing the string. Defaults to "split".
            "full": Try parsing the entire cleaned string only
            "split": Try parsing individual tokens only
            "window": Try parsing sliding windows of tokens (2-4 tokens) only
            "all": Use all three approaches in the order "full", "split", "window
        date_patterns (list[tuple[str, str]], optional): List of (regex_pattern, strptime_format) tuples
            to try before falling back to token-based parsing. This allows users to specify
            exact formats to handle ambiguous cases like MM/DD/YYYY vs DD/MM/YYYY.
            Only used in "split" and "all" modes. Defaults to None (no regex patterns).

    Returns:
        datetime: Datetime object corresponding to the day of the binfolder.

    Raises:
        ValueError: If no valid date token is found in the string.
        TypeError: If date_patterns is not a list of tuples.

    Examples:
        >>> # Handle ambiguous date formats with explicit patterns
        >>> patterns = [(r'(19\\d{2}|20\\d{2})-(\\d{1,2})-(\\d{1,2})', '%Y-%m-%d')]
        >>> parse_str_to_day('2001_2023-07-04_data', date_patterns=patterns)
        datetime.datetime(2023, 7, 4, 0, 0)

        >>> # European format pattern
        >>> patterns = [(r'(\\d{1,2})/(\\d{1,2})/(19\\d{2}|20\\d{2})', '%d/%m/%Y')]
        >>> parse_str_to_day('04/07/2023_data', date_patterns=patterns)
        datetime.datetime(2023, 7, 4, 0, 0)  # July 4th, not April 7th

    Note:
        When date_patterns is provided, users have full control over date interpretation.
        Without date_patterns, the function falls back to token-based parsing which may
        be ambiguous for formats like MM/DD/YYYY vs DD/MM/YYYY.
    """
    warnings.warn(
        "parse_str_to_day is deprecated. Use FileDiscoverer with {session} placeholder in pattern instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if parse_params is None:
        parse_params = {"fuzzy": True}
    elif not isinstance(parse_params, dict):
        raise TypeError("parse_params must be a dictionary")

    # Validate date_patterns
    if date_patterns is not None:
        if not isinstance(date_patterns, list):
            raise TypeError("date_patterns must be a list of sequences")
        for i, pattern_seq in enumerate(date_patterns):
            if not isinstance(pattern_seq, (tuple, list)) or len(pattern_seq) != 2:
                raise TypeError(f"date_patterns[{i}] must be a sequence of (regex_pattern, strptime_format)")
            if not isinstance(pattern_seq[0], str) or not isinstance(pattern_seq[1], str):
                raise TypeError(f"date_patterns[{i}] must contain string elements")

    # Validate parse_mode
    valid_modes = ["full", "split", "window", "all"]
    if parse_mode not in valid_modes:
        raise ValueError(f"Invalid parse_mode: {parse_mode}. Must be one of {valid_modes}")

    clean_str = _clean_str_for_date(string)

    # Only use user-provided regex patterns for "split" and "all" modes
    if date_patterns and parse_mode in ["split", "all"]:
        date_result = _try_user_regex_patterns(clean_str, date_patterns)
        if date_result is not None:
            return date_result
        else:
            # Warn when patterns are provided but none match, falling back to token parsing
            warnings.warn(
                f"No user-provided date patterns matched '{clean_str}'. "
                f"Falling back to token-based parsing which may be ambiguous.",
                UserWarning,
            )

    # Fallback to original token-based approach
    # Try parsing based on the specified mode
    if parse_mode in ["full", "all"]:
        # Pass 1: Try parsing the entire cleaned string
        try:
            date = dateutil.parser.parse(clean_str, default=constants.DEFAULT_DAY, **parse_params)
            if date.year > 1980:
                return date
        except ParserError:
            pass

    if parse_mode in ["split", "all"]:
        # Pass 2: Try individual tokens
        tokens = clean_str.split(sep)
        if len(tokens) == 1:
            warnings.warn("Only 1 string token found. Did you mean to use a different separator or parse_mode='all'?")
        for token in tokens:
            try:
                # logging.debug(f'token: {token}')
                date = dateutil.parser.parse(token, default=constants.DEFAULT_DAY, **parse_params)
                if date.year <= 1980:
                    continue
                return date.replace(tzinfo=None)
            except ParserError:
                continue

    if parse_mode in ["window", "all"]:
        # Pass 3: Try sliding window of tokens
        tokens = clean_str.split(sep)
        if len(tokens) == 1:
            warnings.warn("Only 1 string token found. Did you mean to use a different separator or parse_mode='all'?")
        for window_size in range(2, min(5, len(tokens) + 1)):
            for i in range(len(tokens) - window_size + 1):
                grouped = " ".join(tokens[i : i + window_size])
                try:
                    date = dateutil.parser.parse(grouped, default=constants.DEFAULT_DAY, **parse_params)
                    if date.year <= 1980:
                        continue
                    return date.replace(tzinfo=None)
                except ParserError:
                    continue

    raise ValueError(f"No valid date token found in string: {string}")


def _try_user_regex_patterns(clean_str: str, date_patterns: list[tuple[str, str]]) -> Optional[datetime]:
    """
    Try user-provided regex patterns to find complete date patterns.

    Args:
        clean_str (str): Cleaned string to search for date patterns
        date_patterns (list[tuple[str, str]]): List of (regex_pattern, strptime_format) tuples

    Returns:
        Optional[datetime]: Parsed datetime if a pattern matches, None otherwise
    """
    successful_matches = []

    for pattern_idx, (pattern, date_format) in enumerate(date_patterns):
        try:
            regex_matches = re.finditer(pattern, clean_str, re.IGNORECASE)
            for match in regex_matches:
                date_str = match.group().strip()
                # Check if date string can be parsed and meets year criteria
                try:
                    date = datetime.strptime(date_str, date_format)
                    if date.year > 1980:
                        successful_matches.append((date_str, pattern_idx))
                except (ValueError, TypeError):
                    continue
        except re.error as e:
            logging.warning(f"Invalid regex pattern '{pattern}': {e}")

    # Check for multiple successful matches and warn
    if len(successful_matches) > 1:
        match_details = [f"pattern {idx}: '{match}'" for match, idx in successful_matches]
        warnings.warn(
            f"Multiple date patterns matched in '{clean_str}': {', '.join(match_details)}. "
            f"Using first match: '{successful_matches[0][0]}'",
            UserWarning,
        )

    # Return first successful match
    if successful_matches:
        first_match_str, first_pattern_idx = successful_matches[0]
        pattern, date_format = date_patterns[first_pattern_idx]
        try:
            return datetime.strptime(first_match_str, date_format)
        except (ValueError, TypeError):
            return None

    return None


def _clean_str_for_date(string: str):
    """
    Clean a string by removing common non-date tokens and patterns.

    Args:
        string (str): Input string containing date

    Returns:
        str: Cleaned string with non-date tokens removed
    """
    patterns = constants.DATEPARSER_PATTERNS_TO_REMOVE
    combined_pattern = "|".join(patterns)
    cleaned = re.sub(combined_pattern, " ", string, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    return cleaned


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
