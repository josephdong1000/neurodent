import contextlib
import csv
import itertools
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import unicodedata
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import dateutil.parser
import numpy as np
import pandas as pd
from dateutil.parser import ParserError
from sklearn.neighbors import KDTree
import zarr

try:
    import mne
    from mne.io.constants import FIFF
except Exception:  # pragma: no cover - optional at import time for tests not using MNE
    mne = None
    FIFF = None

from .. import constants


def convert_units_to_multiplier(current_units: str, target_units: str = "µV") -> float:
    """
    Convert between different voltage units and return the multiplication factor.

    This function calculates the conversion factor needed to transform values
    from one voltage unit to another (e.g., from mV to µV).

    Args:
        current_units (str): The current unit of the values. Must be one of: 'µV', 'mV', 'V', 'nV'.
        target_units (str, optional): The target unit to convert to. Defaults to 'µV'.
            Must be one of: 'µV', 'mV', 'V', 'nV'.

    Returns:
        float: The multiplication factor to convert from current_units to target_units.
            To convert values, multiply your data by this factor.

    Raises:
        AssertionError: If current_units or target_units are not supported.

    Examples:
        >>> convert_units_to_multiplier("mV", "µV")
        1000.0
        >>> convert_units_to_multiplier("V", "mV")
        1000.0
        >>> convert_units_to_multiplier("µV", "V")
        1e-06
    """
    units_to_mult = {"µV": 1e-6, "mV": 1e-3, "V": 1, "nV": 1e-9}

    assert current_units in units_to_mult.keys(), f"No valid current unit called '{current_units}' found"
    assert target_units in units_to_mult.keys(), f"No valid target unit called '{target_units}' found"

    return units_to_mult[current_units] / units_to_mult[target_units]


def extract_mne_unit_info(raw_info: dict) -> tuple[str | None, float | None]:
    """Extract unit information from MNE Raw info object.

    Args:
        raw_info (dict): MNE Raw.info object containing channel information

    Returns:
        tuple[str | None, float | None]: (unit_name, mult_to_uV) where unit_name
                                        is the consistent unit across all channels
                                        and mult_to_uV is the conversion factor to µV

    Raises:
        ValueError: If channel units are inconsistent across channels
    """
    if mne is None or FIFF is None:
        logging.warning("MNE not available, cannot extract unit information")
        return None, None

    if "chs" not in raw_info or not raw_info["chs"]:
        logging.warning("No channel information found in MNE Raw.info, using default units")
        return None, None

    # Extract unit information from all channels
    channel_units = []
    unit_muls = []

    for ch_info in raw_info["chs"]:
        ch_name = ch_info.get("ch_name", "unknown")
        unit = ch_info.get("unit", None)
        unit_mul = ch_info.get("unit_mul", None)

        if unit is not None and unit_mul is not None:
            channel_units.append((ch_name, unit, unit_mul))
            unit_muls.append(unit_mul)

    if not channel_units:
        logging.warning("No unit information found in any channels, using default units")
        return None, None

    # Check for consistency in unit values
    unique_units = set(unit for _, unit, _ in channel_units)
    unique_unit_muls = set(unit_muls)

    if len(unique_units) > 1:
        unit_details = [(ch, unit, mul) for ch, unit, mul in channel_units]
        raise ValueError(
            f"Inconsistent units across channels. Found different unit values: {unique_units}. "
            f"Channel details: {unit_details}"
        )

    if len(unique_unit_muls) > 1:
        unit_details = [(ch, unit, mul) for ch, unit, mul in channel_units]
        raise ValueError(
            f"Inconsistent unit multipliers across channels. Found different unit_mul values: {unique_unit_muls}. "
            f"Channel details: {unit_details}"
        )

    # Get the consistent unit values
    unit_code = list(unique_units)[0]
    unit_mul = list(unique_unit_muls)[0]

    # Convert MNE unit codes to string representation using FIFF constants
    # Based on MNE FIFF constants documentation
    unit_str = None
    if hasattr(FIFF, "FIFF_UNIT_V") and unit_code == FIFF.FIFF_UNIT_V:
        unit_str = "V"
    elif hasattr(FIFF, "FIFF_UNIT_T") and unit_code == FIFF.FIFF_UNIT_T:
        unit_str = "T"  # Tesla - MEG magnetometer
    elif hasattr(FIFF, "FIFF_UNIT_T_M") and unit_code == FIFF.FIFF_UNIT_T_M:
        unit_str = "T/m"  # Tesla/meter - MEG gradiometer
    else:
        logging.warning(f"Unknown MNE unit code {unit_code}, using default units")
        return None, None

    # Convert unit multipliers using FIFF constants
    multiplier = None
    if hasattr(FIFF, "FIFF_UNITM_NONE") and unit_mul == FIFF.FIFF_UNITM_NONE:
        multiplier = 1.0
    elif hasattr(FIFF, "FIFF_UNITM_MU") and unit_mul == FIFF.FIFF_UNITM_MU:
        multiplier = 1e-6  # micro
    elif hasattr(FIFF, "FIFF_UNITM_M") and unit_mul == FIFF.FIFF_UNITM_M:
        multiplier = 1e-3  # milli
    elif hasattr(FIFF, "FIFF_UNITM_N") and unit_mul == FIFF.FIFF_UNITM_N:
        multiplier = 1e-9  # nano
    elif hasattr(FIFF, "FIFF_UNITM_P") and unit_mul == FIFF.FIFF_UNITM_P:
        multiplier = 1e-12  # pico
    elif hasattr(FIFF, "FIFF_UNITM_F") and unit_mul == FIFF.FIFF_UNITM_F:
        multiplier = 1e-15  # femto
    else:
        # Fallback to numerical interpretation if FIFF constants not available
        mul_mapping = {
            0: 1.0,  # FIFF_UNITM_NONE
            -3: 1e-3,  # FIFF_UNITM_M (milli)
            -6: 1e-6,  # FIFF_UNITM_MU (micro)
            -9: 1e-9,  # FIFF_UNITM_N (nano)
            -12: 1e-12,  # FIFF_UNITM_P (pico)
            -15: 1e-15,  # FIFF_UNITM_F (femto)
        }
        multiplier = mul_mapping.get(unit_mul)
        if multiplier is None:
            logging.warning(f"Unknown MNE unit multiplier {unit_mul}, using default units")
            return None, None

    # For EEG data (voltage units), compute the final unit and conversion factor
    if unit_str == "V":
        # Apply the MNE multiplier to get the actual unit
        if multiplier == 1e-6:
            final_unit = "µV"
        elif multiplier == 1e-3:
            final_unit = "mV"
        elif multiplier == 1.0:
            final_unit = "V"
        elif multiplier == 1e-9:
            final_unit = "nV"
        else:
            logging.warning(f"Unusual voltage unit multiplier {multiplier}, treating as V")
            final_unit = "V"

        # Convert to µV multiplier using existing utility
        try:
            mult_to_uV = convert_units_to_multiplier(final_unit, "µV")
            logging.info(f"Extracted MNE units: {final_unit} -> mult_to_uV = {mult_to_uV}")
            return final_unit, mult_to_uV
        except (ValueError, AssertionError) as e:
            logging.warning(f"Failed to convert units {final_unit}: {e}")
            return None, None
    else:
        # Non-voltage units (MEG, etc.) - don't convert to µV
        logging.info(f"Non-voltage units detected: {unit_str}, not converting to µV")
        return None, None


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


def convert_colpath_to_rowpath(
    rowdir_path: str | Path, col_path: str | Path, gzip: bool = True, aspath: bool = True
) -> str | Path:
    """
    Convert a ColMajor file path to its corresponding RowMajor file path.

    This function transforms file paths from column-major format to row-major format,
    which is used when converting between different data storage layouts in NeuRodent.

    Args:
        rowdir_path (str | Path): Directory path where the RowMajor file should be located.
        col_path (str | Path): Path to the ColMajor file to be converted. Must contain 'ColMajor' in the path.
        gzip (bool, optional): If True, append '.npy.gz' extension. If False, append '.bin'. Defaults to True.
        aspath (bool, optional): If True, return as Path object. If False, return as string. Defaults to True.

    Returns:
        str | Path: The converted RowMajor file path, either as string or Path object based on aspath parameter.

    Raises:
        ValueError: If 'ColMajor' is not found in col_path.

    Examples:
        >>> convert_colpath_to_rowpath("/data/row/", "/data/col/file_ColMajor_001.bin")
        PosixPath('/data/row/file_RowMajor_001.npy.gz')
        >>> convert_colpath_to_rowpath("/data/row/", "/data/col/file_ColMajor_001.bin", gzip=False)
        PosixPath('/data/row/file_RowMajor_001.bin')
        >>> convert_colpath_to_rowpath("/data/row/", "/data/col/file_ColMajor_001.bin", aspath=False)
        '/data/row/file_RowMajor_001.npy.gz'
    """
    # TODO it would make more sense to not have a rowdir_path aparameter, since this is outside the scope of the function
    if "ColMajor" not in col_path:
        raise ValueError(f"Expected 'ColMajor' in col_path: {col_path}")

    out = Path(rowdir_path) / f"{get_file_stem(Path(col_path).name).replace('ColMajor', 'RowMajor')}"
    if gzip:
        out = str(out) + ".npy.gz"
    else:
        out = str(out) + ".bin"
    return Path(out) if aspath else out


def filepath_to_index(filepath) -> int:
    """
    Extract the index number from a filepath.

    This function extracts the last number found in a filepath after removing common suffixes
    and file extensions. For example, from "/path/to/data_ColMajor_001.bin" it returns 1.

    Args:
        filepath (str | Path): Path to the file to extract index from.

    Returns:
        int: The extracted index number, or 0 if no number is found in the filename.

    Examples:
        >>> filepath_to_index("/path/to/data_ColMajor_001.bin")
        1
        >>> filepath_to_index("/path/to/data_2023_015_ColMajor.bin")
        15
        >>> filepath_to_index("/path/to/data_Meta_010.json")
        10
    """
    fpath = str(filepath)
    for suffix in ["_RowMajor", "_ColMajor", "_Meta"]:
        fpath = fpath.replace(suffix, "")

    # Remove only the actual file extension, not dots within the filename
    path_obj = Path(fpath)
    if path_obj.suffix:
        fpath = str(path_obj.with_suffix(""))

    fname = Path(fpath).name
    fname = re.split(r"\D+", fname)
    fname = list(filter(None, fname))
    if not fname:
        return 0
    return int(fname[-1])


def parse_truncate(truncate: int | bool) -> int:
    """
    Parse the truncate parameter to determine how many characters to truncate.

    If truncate is a boolean, returns 10 if True and 0 if False.
    If truncate is an integer, returns that integer value directly.

    Args:
        truncate (int | bool): If bool, True=10 chars and False=0 chars.
                              If int, specifies exact number of chars.

    Returns:
        int: Number of characters to truncate (0 means no truncation)

    Raises:
        ValueError: If truncate is not a boolean or integer
    """
    if isinstance(truncate, bool):
        return 10 if truncate else 0
    elif isinstance(truncate, int):
        return truncate
    else:
        raise ValueError(f"Invalid truncate value: {truncate}")


def get_feature_label(feature_name: str) -> str:
    """
    Convert a feature column name to a human-readable label.
    
    Handles:
    - Base features: "rms" -> "RMS"
    - Banded features: "logpsdband_delta" -> "Log Band Power - Delta"
    - Baseline-subtracted: "logrms_nobase" -> "Log(RMS) - Baseline"
    
    Args:
        feature_name: Column name (e.g., "logpsdband_delta_nobase")
    
    Returns:
        Human-readable label. Falls back to the original name if not found.
    
    Examples:
        >>> get_feature_label("logpsdband_delta")
        'Log Band Power (Delta)'
        >>> get_feature_label("alphadelta")
        'Alpha/Delta Ratio'
        >>> get_feature_label("logrms_nobase")
        'Log(RMS) - Baseline'
    """
    # Check for _nobase suffix
    is_baseline_subtracted = feature_name.endswith("_nobase")
    if is_baseline_subtracted:
        feature_name = feature_name[:-7]  # Remove "_nobase"
    
    # Check for band suffix (only for banded/matrix features)
    band_name = None
    base_feature = feature_name
    
    # Only check for band suffix if the base is a known banded feature
    for band in constants.BAND_NAMES:
        if feature_name.endswith(f"_{band}"):
            potential_base = feature_name[: -(len(band) + 1)]
            if potential_base in constants.BAND_FEATURES or potential_base in constants.MATRIX_FEATURES:
                base_feature = potential_base
                band_name = band
                break
    
    # Look up base label
    base_label = constants.FEATURE_LABELS.get(base_feature, feature_name)
    
    # Build final label
    label = base_label
    if band_name:
        label = f"{label} ({band_name.capitalize()})"
    if is_baseline_subtracted:
        label = f"{label} - Baseline"
    
    return label


def nanaverage(A: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute weighted average of an array, ignoring NaN values.

    This function computes a weighted average along the specified axis while
    properly handling NaN values by masking them out of the calculation.

    Args:
        A (np.ndarray): Input array containing the values to average.
        weights (np.ndarray): Array of weights corresponding to the values in A.
            Must be broadcastable with A along the specified axis.
        axis (int, optional): Axis along which to compute the average. Defaults to -1 (last axis).

    Returns:
        np.ndarray: Weighted average with NaN values properly handled. If all values
            along an axis are NaN, the result will be NaN for that position.

    Examples:
        >>> import numpy as np
        >>> A = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        >>> weights = np.array([1, 2, 1])
        >>> nanaverage(A, weights, axis=1)
        array([1.66666667, 5.        ])

    Note:
        Be careful with zero or negative weights as they may produce unexpected results.
        The function uses numpy's masked array functionality for robust NaN handling.
    """
    masked = np.ma.masked_array(A, np.isnan(A))
    avg = np.ma.average(masked, axis=axis, weights=weights)

    # Handle case where np.ma.average returns a scalar instead of masked array
    if np.ma.is_masked(avg):
        return avg.filled(np.nan)
    else:
        # avg is a scalar or regular array, convert to array and handle NaN
        result = np.asarray(avg)
        return np.where(np.isfinite(result), result, np.nan)


def parse_str_to_animal(string: str, animal_param: tuple[int, str] | str | list[str] = (0, None)) -> str:
    """
    DEPRECATED: Use FileDiscoverer with {animal} placeholder in pattern instead.

    Parses the filename of a binfolder to get the animal id.

    Args:
        string (str): String to parse.
        animal_param: Parameter specifying how to parse the animal ID:
            tuple[int, str]: (index, separator) for simple split and index. Not recommended for inconsistent naming conventions.
            str: regex pattern to extract ID. Most general use case. If multiple matches are found, returns the first match.
            list[str]: list of possible animal IDs to match against. Returns first match in list order, case-sensitive, ignoring empty strings.

    Returns:
        str: Animal id.

    Examples:
        # Tuple format: (index, separator)
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", (1, "_"))
        'A10'
        >>> parse_str_to_animal("A10_WT_recording.bin", (0, "_"))
        'A10'

        # Regex pattern format
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", r"A\\\\d+")
        'A10'
        >>> parse_str_to_animal("subject_123_data.bin", r"\\\\d+")
        '123'

        # List format: possible IDs to match
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", ["A10", "A11", "A12"])
        'A10'
        >>> parse_str_to_animal("WT_A10_data.bin", ["B15", "C20"])  # No match
        ValueError: No matching ID found in WT_A10_data.bin from possible IDs: ['B15', 'C20']
    """
    warnings.warn(
        "parse_str_to_animal is deprecated. Use FileDiscoverer with {animal} placeholder in pattern instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if isinstance(animal_param, tuple):
        index, sep = animal_param
        animid = string.split(sep)
        return animid[index]
    elif isinstance(animal_param, str):
        pattern = animal_param
        match = re.search(pattern, string)
        if match:
            return match.group()
        raise ValueError(f"No match found for pattern {pattern} in string {string}")
    elif isinstance(animal_param, list):
        possible_ids = animal_param
        for id in possible_ids:
            # Skip empty or whitespace-only strings
            if id and id.strip() and id in string:
                return id
        raise ValueError(f"No matching ID found in {string} from possible IDs: {possible_ids}")
    else:
        raise ValueError(f"Invalid animal_param type: {type(animal_param)}")


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


def resolve_channel(channel_name: str) -> str:
    """
    Resolve a raw channel name to its canonical channel abbreviation by **exact lookup**.

    Resolution is explicit and never inferred: (1) the (stripped) name is already a
    canonical abbreviation (:data:`neurodent.constants.CHANNEL_ABBREVS`); (2) it is an
    exact key in :data:`neurodent.constants.CHANNEL_ABBREV_BY_RAW` (the per-dataset
    ``raw name -> abbrev`` map). Anything else **raises loudly** — there is no fuzzy,
    substring, or number-based guessing.

    Args:
        channel_name (str): Raw channel name from the data.

    Returns:
        str: Canonical channel abbreviation.

    Raises:
        ValueError: When the name is not in the configured channel map. Configure the exact
            raw name under its abbreviation (``channels`` in the samples config, or
            :func:`neurodent.set_channel_map`).

    Examples:
        >>> resolve_channel("LMot")          # already canonical
        'LMot'
        >>> resolve_channel("L Motor Ctx")   # configured raw name -> abbrev
        'LMot'
    """
    raw = channel_name.strip()
    if raw in constants.CHANNEL_ABBREVS:
        return raw
    if raw in constants.CHANNEL_ABBREV_BY_RAW:
        return constants.CHANNEL_ABBREV_BY_RAW[raw]
    raise ValueError(
        f"Channel {raw!r} is not in the configured channel map. "
        f"Canonical labels: {constants.CHANNEL_ABBREVS}; configured raw names: "
        f"{sorted(constants.CHANNEL_ABBREV_BY_RAW)}. "
        f"Add the exact raw name under its abbreviation in the samples config "
        f"(channels) or via neurodent.set_channel_map()."
    )


def resolve_channels(names: list[str]) -> list[str]:
    """Abbreviate a list of raw channel names via exact lookup.

    Unmappable names are **warned about loudly** (and kept as-is so callers comparing
    channel sets still get a value) rather than silently swallowed.

    Args:
        names: List of raw channel name strings.

    Returns:
        List of canonical abbreviations (same length as input); an unmappable entry is
        returned unchanged after a warning.
    """
    result = []
    for name in names:
        try:
            result.append(resolve_channel(name))
        except (ValueError, KeyError, AttributeError) as e:
            warnings.warn(
                f"Channel name {name!r} could not be mapped to a canonical abbreviation: {e}",
                UserWarning,
                stacklevel=2,
            )
            result.append(name)
    return result


def normalize_value_from_aliases(
    value: str,
    alias_dict: dict[str, list[str]],
) -> str | None:
    """Normalize a value to its canonical form using a value map.

    Performs **exact** matching: the value must equal one of the accepted spellings
    listed for a canonical label. Used for normalizing standalone configuration values
    against an exact ``_MAP`` (e.g. :data:`~neurodent.constants.SEX_MAP`,
    :data:`~neurodent.constants.GENOTYPE_MAP`).

    Args:
        value: The raw value to normalize (e.g., ``"M"``, ``"female"``).
        alias_dict: Dictionary of ``{canonical_key: [accepted spellings]}``.

    Returns:
        The canonical key if *value* matches any spelling, or ``None`` if no match.
    """
    for canonical_key, aliases in alias_dict.items():
        if value in aliases:
            return canonical_key
    return None


def set_temp_directory(path: str | Path) -> None:
    """
    Set the temporary directory for NeuRodent operations.

    This function configures the temporary directory used by NeuRodent for intermediate
    files and operations. The directory will be created if it doesn't exist.

    Args:
        path (str | Path): Path to the temporary directory. Will be created if it doesn't exist.

    Examples:
        >>> set_temp_directory("/tmp/neurodent_temp")
        >>> set_temp_directory(Path.home() / "neurodent_workspace" / "temp")

    Note:
        This function modifies the TMPDIR environment variable, which affects
        the behavior of other temporary file operations in the process.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(path)
    logging.info(f"Temporary directory set to {path}")


def get_temp_directory() -> Path:
    """
    Get the current temporary directory used by NeuRodent.

    Returns:
        Path: Path object representing the current temporary directory.

    Examples:
        >>> temp_dir = get_temp_directory()
        >>> print(f"Current temp directory: {temp_dir}")
        Current temp directory: /tmp/neurodent_temp

    Raises:
        KeyError: If TMPDIR environment variable is not set.
    """
    return Path(os.environ["TMPDIR"])


def safe_unlink(path: Union[str, Path]) -> None:
    """Delete a file if it exists, ignoring a missing file.

    Used for self-healing cache deletion: a corrupt cache file is removed so it
    can be regenerated, and a concurrently-removed file is not an error.

    Args:
        path: Path to the file to delete.
    """
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except (OSError, PermissionError) as e:
        logging.warning(f"Failed to delete {path}: {e}")


def is_si_recording_folder(path: Union[str, Path]) -> bool:
    """Return True if ``path`` looks like a SpikeInterface recording output folder.

    Recognizes the two formats written by :meth:`LongRecording.save` as well as
    folders written by NeuRodent's own :meth:`LongRecordingOrganizer.save_recording`.
    This is a safety gate so destructive overwrites only ever target folders we
    actually produced — never an arbitrary user directory.

    A folder qualifies when it is a directory and any of the following hold:

    - **Zarr**: the folder ends in ``.zarr`` and contains zarr group metadata
      (``.zattrs``, ``.zmetadata``, or ``zarr.json``).
    - **Binary**: the folder contains SpikeInterface's recognition marker
      ``si_folder.json`` (or ``binary.json``).
    - **NeuRodent**: the folder contains our own sidecar
      (:data:`~neurodent.constants.NEURODENT_SIDECAR_NAME`).

    Args:
        path: Path to inspect.

    Returns:
        bool: True if ``path`` is a recognized recording output folder.
    """
    p = Path(path)
    if not p.is_dir():
        return False

    # NeuRodent sidecar — recognizes a folder we wrote even across SI versions.
    if (p / constants.NEURODENT_SIDECAR_NAME).exists():
        return True

    # Zarr folder: suffix + zarr group metadata.
    if p.suffix == ".zarr" and (
        (p / ".zattrs").exists()
        or (p / ".zmetadata").exists()
        or (p / "zarr.json").exists()
    ):
        return True

    # Binary folder: SpikeInterface's own recognition markers.
    if (p / "si_folder.json").exists() or (p / "binary.json").exists():
        return True

    return False


def safe_rmtree(path: Union[str, Path], *, require_marker: bool = True) -> None:
    """Recursively delete a directory tree, refusing unrecognized targets.

    A guarded counterpart to :func:`safe_unlink` for directories. By default it
    will only delete a directory that :func:`is_si_recording_folder` recognizes,
    so a mistyped or malicious path can never wipe an arbitrary data directory.

    Args:
        path: Directory to remove.
        require_marker: When True (default), raise :class:`ValueError` unless the
            target is a recognized SpikeInterface/NeuRodent recording folder.

    Raises:
        ValueError: If ``require_marker`` is True and the target is not a
            recognized recording folder.
    """
    p = Path(path)
    if not p.exists():
        return
    if require_marker and not is_si_recording_folder(p):
        raise ValueError(
            f"Refusing to delete {p}: it does not look like a SpikeInterface "
            "recording output folder. Delete it manually if you are sure."
        )
    try:
        shutil.rmtree(p)
    except FileNotFoundError:
        pass
    except (OSError, PermissionError) as e:
        logging.warning(f"Failed to remove {p}: {e}")


@contextlib.contextmanager
def atomic_output_path(final_path: Union[str, Path]):
    """Context manager yielding a temporary sibling path for an atomic write.

    The caller writes to the yielded temporary path. On clean exit the temp file
    is atomically moved into place with :func:`os.replace`; on exception the temp
    file is removed and the original error re-raised. Because the temp file lives
    in the same directory as ``final_path`` (same filesystem), ``os.replace`` is
    atomic, so a crash mid-write can never leave a partial file at ``final_path``.

    Args:
        final_path: The destination path the content should end up at.

    Yields:
        Path: A temporary path in the same directory to write to.

    Examples:
        >>> with atomic_output_path("out.bin") as tmp:  # doctest: +SKIP
        ...     data.tofile(tmp)
    """
    final_path = Path(final_path)
    tmp_path = final_path.with_name(f"{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        yield tmp_path
    except BaseException:
        safe_unlink(tmp_path)
        raise
    else:
        os.replace(tmp_path, final_path)


def atomic_write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    """Atomically write ``obj`` to ``path`` as JSON.

    Serializes to a temporary sibling file and atomically renames it into place,
    so an interrupted write never leaves a partial/corrupt JSON file at ``path``.

    Args:
        path: Destination JSON file path.
        obj: JSON-serializable object to write.
        indent: Indentation passed to :func:`json.dump`.
    """
    with atomic_output_path(path) as tmp:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=indent)


def cache_fragments_to_zarr(
    np_fragments: np.ndarray,
    n_fragments: int,
    tmpdir: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> tuple[str, "zarr.Array"]:
    """
    Cache numpy fragments array to zarr format for efficient memory management.

    This function converts a numpy array of recording fragments to a zarr array stored
    in a temporary location. This allows better memory management and garbage collection
    by avoiding keeping large numpy arrays in memory for extended periods.

    Args:
        np_fragments (np.ndarray): Numpy array of shape (n_fragments, n_samples, n_channels)
            containing the recording fragments to cache.
        n_fragments (int): Number of fragments to cache (allows for subset caching).
        tmpdir (str, optional): Directory path for temporary zarr storage. If None,
            uses get_temp_directory(). Defaults to None.
        chunk_size (int, optional): Number of fragments per zarr chunk along the first
            axis. Controls the read/write granularity when accessing the zarr array.
            Smaller values reduce memory overhead per chunk; larger values improve
            sequential throughput. When None, defaults to ``min(100, n_fragments)``.

    Returns:
        tuple[str, zarr.Array]: A tuple containing:
            - str: Path to the temporary zarr file
            - zarr.Array: The zarr array object for accessing cached data

    Raises:
        ImportError: If zarr is not available
        ValueError: If ``chunk_size`` is not None and is less than 1
    """
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(
            f"chunk_size must be >= 1, got {chunk_size!r}. "
            "Pass None to use the default chunk size."
        )

    try:
        import zarr
    except ImportError:
        raise ImportError("zarr package is required for fragment caching")

    if tmpdir is None:
        tmpdir = get_temp_directory()

    # Generate unique temporary path
    tmppath = os.path.join(tmpdir, f"temp_{os.urandom(24).hex()}.zarr")

    logging.debug(f"Caching numpy array with zarr in {tmppath}")

    # Create Zarr array with optimal settings for fragment-wise access
    if chunk_size is None:
        chunk_size = min(100, n_fragments)  # Cap at 100 fragments per chunk
    else:
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer or None")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 when provided")
        chunk_size = min(chunk_size, n_fragments)
    zarr_array = zarr.open(
        tmppath,
        mode="w",
        shape=np_fragments.shape,
        chunks=(
            chunk_size,
            -1,  # No chunking along timestamp dimension
            -1,  # No chunking along channel dimension
        ),
        dtype=np_fragments.dtype,
        compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.SHUFFLE),  # Fast compression
    )
    zarr_array[:n_fragments] = np_fragments[:n_fragments]

    # Log debug properties of the zarr array
    total_memory_bytes = zarr_array.nbytes
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    total_memory_gb = total_memory_mb / 1024

    logging.debug(f"  - Total memory footprint: {total_memory_mb:.2f} MB, {total_memory_gb:.3f} GB")
    logging.debug(f"  - Zarr array shape: {zarr_array.shape}")
    logging.debug(f"  - Zarr array chunks: {zarr_array.chunks}")

    return tmppath, zarr_array


def stream_fragments_to_zarr(
    get_fragment_fn: Callable[[int], np.ndarray],
    n_fragments: int,
    fragment_shape: tuple,
    fragment_dtype: np.dtype,
    chunk_size: int,
    tmpdir: Optional[str] = None,
) -> str:
    """Stream recording fragments to a zarr store in memory-bounded batches.

    Unlike :func:`cache_fragments_to_zarr`, this function never holds more than
    ``chunk_size`` fragments in RAM at once.  It calls ``get_fragment_fn`` one
    batch at a time, writes each batch to the zarr store, and immediately frees
    the batch buffer — so peak RAM is proportional to ``chunk_size`` rather than
    ``n_fragments``.

    Args:
        get_fragment_fn (Callable[[int], np.ndarray]): A callable that accepts a
            fragment index (0-based) and returns the corresponding fragment as a
            NumPy array of shape ``fragment_shape``.
        n_fragments (int): Total number of fragments to stream.
        fragment_shape (tuple): Shape of a single fragment (e.g. ``(n_samples,
            n_channels)``).
        fragment_dtype (np.dtype): Data-type of the fragment arrays.
        chunk_size (int): Number of fragments to buffer per batch.  Must be >= 1.
            Larger values improve sequential write throughput; smaller values
            reduce peak RAM.
        tmpdir (str, optional): Directory for the temporary zarr file.  If
            ``None``, uses :func:`get_temp_directory`.

    Returns:
        str: Path to the temporary zarr file on disk.

    Raises:
        ValueError: If ``chunk_size`` < 1.
        ImportError: If zarr is not available.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    if tmpdir is None:
        tmpdir = get_temp_directory()

    tmppath = os.path.join(tmpdir, f"temp_{os.urandom(24).hex()}.zarr")
    batch = min(chunk_size, n_fragments)

    logging.debug(
        f"Streaming {n_fragments} fragments to zarr in batches of {batch} "
        f"(path: {tmppath})"
    )

    zarr_array = zarr.open(
        tmppath,
        mode="w",
        shape=(n_fragments,) + fragment_shape,
        chunks=(batch, -1, -1),
        dtype=fragment_dtype,
        compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
    )

    for batch_start in range(0, n_fragments, batch):
        batch_end = min(batch_start + batch, n_fragments)
        batch_len = batch_end - batch_start
        np_batch = np.empty((batch_len,) + fragment_shape, dtype=fragment_dtype)
        for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
            np_batch[local_idx] = get_fragment_fn(global_idx)
        zarr_array[batch_start:batch_end] = np_batch
        del np_batch

    logging.debug(f"  - Zarr array shape: {zarr_array.shape}")
    logging.debug(f"  - Zarr array chunks: {zarr_array.chunks}")
    del zarr_array

    return tmppath


def stream_recording_to_zarr(
    recording,
    n_fragments: int,
    n_samples_per_frag: int,
    n_frag_per_chunk: int,
    tmpdir: Optional[str] = None,
) -> str:
    """Stream a SpikeInterface recording to a zarr store in memory-bounded batches.

    Reads chunk-sized slices from ``recording.get_traces()``, reshapes each
    chunk to ``(n_frags_in_chunk, n_samples_per_frag, n_channels)``, and
    writes it to a zarr store.  Peak RAM is proportional to ``n_frag_per_chunk``
    rather than ``n_fragments``.

    Args:
        recording: A SpikeInterface ``BaseRecording`` object (may be a lazy
            wrapper such as a ``NotchFilterRecording``).
        n_fragments (int): Total number of fragments to stream.
        n_samples_per_frag (int): Number of samples in each fragment.
        n_frag_per_chunk (int): Number of fragments to buffer per batch.
            Must be >= 1.  Larger values improve sequential write throughput;
            smaller values reduce peak RAM.
        tmpdir (str, optional): Directory for the temporary zarr file.  If
            ``None``, uses :func:`get_temp_directory`.

    Returns:
        str: Path to the temporary zarr file on disk.

    Raises:
        ValueError: If ``n_frag_per_chunk`` < 1.
        ImportError: If zarr is not available.
    """
    if n_frag_per_chunk < 1:
        raise ValueError(f"n_frag_per_chunk must be >= 1, got {n_frag_per_chunk}")

    try:
        import zarr
    except ImportError:
        raise ImportError("zarr package is required for fragment caching")

    if tmpdir is None:
        tmpdir = get_temp_directory()

    n_channels = recording.get_num_channels()
    fragment_dtype = recording.get_dtype()
    tmppath = os.path.join(tmpdir, f"temp_{os.urandom(24).hex()}.zarr")
    # Cap the batch size so we never request more fragments than exist
    batch_size = min(n_frag_per_chunk, n_fragments)

    logging.debug(
        f"Streaming recording ({n_fragments} fragments × {n_samples_per_frag} samples) "
        f"to zarr in batches of {batch_size} (path: {tmppath})"
    )

    zarr_array = zarr.open(
        tmppath,
        mode="w",
        shape=(n_fragments, n_samples_per_frag, n_channels),
        chunks=(batch_size, -1, -1),
        dtype=fragment_dtype,
        compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
    )

    for batch_start in range(0, n_fragments, batch_size):
        batch_end = min(batch_start + batch_size, n_fragments)
        start_sample = batch_start * n_samples_per_frag
        end_sample = batch_end * n_samples_per_frag
        chunk_traces = recording.get_traces(
            start_frame=start_sample, end_frame=end_sample, return_scaled=True
        )
        chunk_fragments = chunk_traces.reshape(
            batch_end - batch_start, n_samples_per_frag, n_channels
        )
        zarr_array[batch_start:batch_end] = chunk_fragments
        # Explicitly free each batch to keep peak RAM bounded to batch_size fragments
        del chunk_traces, chunk_fragments

    logging.debug(f"  - Zarr array shape: {zarr_array.shape}")
    logging.debug(f"  - Zarr array chunks: {zarr_array.chunks}")
    del zarr_array

    return tmppath


def chunked_channel_distance_matrix(
    get_traces_fn: Callable[[int, int], np.ndarray],
    n_channels: int,
    n_samples: int,
    chunk_samples: int,
) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between channels in chunks.

    Instead of loading the full ``(n_samples, n_channels)`` trace matrix at
    once, this function reads ``chunk_samples`` frames at a time and
    accumulates squared distances using the identity

        ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 * c_i · c_j

    so that peak RAM is proportional to ``chunk_samples * n_channels``
    rather than ``n_samples * n_channels``.

    Args:
        get_traces_fn: ``fn(start_frame, end_frame) -> np.ndarray`` with
            shape ``(frames, n_channels)``.  Typically
            ``recording.get_traces(start_frame=..., end_frame=...,
            return_scaled=True)``.
        n_channels: Number of channels.
        n_samples: Total number of samples in the recording.
        chunk_samples: Number of samples to read per chunk.

    Returns:
        np.ndarray: Symmetric ``(n_channels, n_channels)`` Euclidean
        distance matrix.
    """
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")

    sq_dist_accum = np.zeros((n_channels, n_channels), dtype=np.float64)

    for start in range(0, n_samples, chunk_samples):
        end = min(start + chunk_samples, n_samples)
        chunk = get_traces_fn(start, end)  # (chunk_len, n_channels)
        chunk_t = chunk.T  # (n_channels, chunk_len)

        # ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 * c_i · c_j
        norms_sq = np.sum(chunk_t ** 2, axis=1)  # (n_channels,)
        gram = chunk_t @ chunk_t.T  # (n_channels, n_channels)
        sq_dist_accum += norms_sq[:, None] + norms_sq[None, :] - 2 * gram
        del chunk, chunk_t, norms_sq, gram

    # Clamp tiny negatives from floating-point arithmetic
    np.maximum(sq_dist_accum, 0, out=sq_dist_accum)
    distance_matrix = np.sqrt(sq_dist_accum)
    del sq_dist_accum
    return distance_matrix


def get_file_stem(filepath: Union[str, Path]) -> str:
    """Get the true stem for files, handling double extensions like .npy.gz."""
    filepath = Path(filepath)
    name = filepath.name

    return name.split(".")[0]


def _get_groupby_keys(df: pd.DataFrame, groupby: str | list[str]):
    """
    Get the unique values of the groupby variable.
    """
    return list(df.groupby(groupby).groups.keys())


def _get_pairwise_combinations(x: list):
    """
    Get all pairwise combinations of a list.
    """
    return list(itertools.combinations(x, 2))


class _HiddenPrints:
    """
    Context manager to suppress print output during code execution.

    This class provides a way to temporarily suppress print statements and other
    stdout output, which is useful when calling functions that produce unwanted
    console output.

    Args:
        silence (bool, optional): Whether to actually suppress output. Defaults to True.
            If False, acts as a no-op context manager.

    Examples:
        >>> with _HiddenPrints():
        ...     print("This won't be displayed")
        ...     some_noisy_function()
        >>> print("This will be displayed")
        This will be displayed

        >>> with _HiddenPrints(silence=False):
        ...     print("This will be displayed")
        This will be displayed
    """

    def __init__(self, silence: bool = True) -> None:
        self.silence = silence

    def __enter__(self):
        if self.silence:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silence:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def nanmean_series_of_np(x: pd.Series, axis: int = 0) -> np.ndarray:
    """
    Efficiently compute NaN-aware mean of a pandas Series containing numpy arrays.

    This function is optimized for computing the mean across a Series where each element
    is a numpy array. It uses different strategies based on the size of the Series
    for optimal performance.

    Args:
        x (pd.Series): Series containing numpy arrays as elements.
        axis (int, optional): Axis along which to compute the mean. Defaults to 0.
            - axis=0: Mean across the Series elements (most common)
            - axis=1: Mean within each array element

    Returns:
        np.ndarray: Array containing the computed means with NaN values properly handled.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a Series of numpy arrays
        >>> arrays = [np.array([1.0, 2.0, np.nan]),
        ...           np.array([4.0, np.nan, 6.0]),
        ...           np.array([7.0, 8.0, 9.0])]
        >>> series = pd.Series(arrays)
        >>> nanmean_series_of_np(series)
        array([4. , 5. , 7.5])

    Performance Notes:
        - For Series with more than 1000 elements containing numpy arrays,
          uses `np.stack()` for better performance
        - Falls back to list conversion for smaller Series or mixed types
        - Handles shape mismatches gracefully by falling back to the slower method
    """
    # logging.debug(f"Unique shapes in x: {set(np.shape(item) for item in x)}")

    if len(x) > 1000:
        try:
            if isinstance(x.iloc[0], np.ndarray):
                xmean: np.ndarray = np.nanmean(np.stack(x.values, axis=0), axis=axis)
                return xmean
        except (ValueError, TypeError):
            pass

    xmean: np.ndarray = np.nanmean(np.array(list(x)), axis=axis)
    return xmean


def log_transform(rec: np.ndarray, **kwargs) -> np.ndarray:
    """Log transform the signal

    Args:
        rec (np.ndarray): The signal to log transform.

    Returns:
        np.ndarray: ln(rec + 1)
    """
    if rec is not None:
        return np.log(rec + 1)
    else:
        return None


def sort_dataframe_by_plot_order(df: pd.DataFrame, df_sort_order: Optional[dict] = None) -> pd.DataFrame:
    """
    Sort DataFrame columns according to predefined orders.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort
    df_sort_order : dict
        Dictionary mapping column names to the order of the values in the column.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame

    Raises
    ------
    ValueError
        If df_sort_order is not a valid dictionary or contains invalid categories
    """
    if df_sort_order is None:
        df_sort_order = constants.DF_SORT_ORDER.copy()
    elif not isinstance(df_sort_order, dict):
        raise ValueError("df_sort_order must be a dictionary")

    if df.empty:
        return df.copy()

    for col, categories in df_sort_order.items():
        if not isinstance(categories, (list, tuple)):
            raise ValueError(f"Categories for column '{col}' must be a list or tuple")

    columns_to_sort = [col for col in df.columns if col in df_sort_order]
    df_sorted = df.copy()

    if not columns_to_sort:
        return df_sorted

    for col in columns_to_sort:
        categories = df_sort_order[col]

        # Check for values not in predefined categories
        unique_values = set(df_sorted[col].dropna().unique())
        missing_values = unique_values - set(categories)

        if missing_values:
            raise ValueError(
                f"Column '{col}' contains values not in sort order dictionary: {missing_values}. Add them to plot_order in ExperimentPlotter init."
            )

        # Filter categories to only include those that exist in the DataFrame
        existing_categories = [cat for cat in categories if cat in unique_values]

        df_sorted[col] = pd.Categorical(df_sorted[col], categories=existing_categories, ordered=True)

    df_sorted = df_sorted.sort_values(columns_to_sort)
    # REVIEW since "sex" is not inherently part of the pipeline (add ad-hoc), this could be a feature worth sorting
    # But this might mean rewriting the data loading pipeline, file-reading, etc.
    # Maybe a dictionary corresponding to animal/id -> sex would be good enough, instead of reading it in from filenames
    # which would be difficult since name conventions are not standardized

    return df_sorted


class Natural_Neighbor(object):
    """
    Natural Neighbor algorithm implementation for finding natural neighbors in a dataset.

    This class implements the Natural Neighbor algorithm which finds mutual neighbors
    in a dataset by iteratively expanding the neighborhood radius until convergence.
    """

    def __init__(self):
        """
        Initialize the Natural Neighbor algorithm.

        Attributes:
            nan_edges (dict): Graph of mutual neighbors
            nan_num (dict): Number of natural neighbors for each instance
            repeat (dict): Data structure that counts repetitions of the count method
            target (list): Set of classes
            data (list): Set of instances
            knn (dict): Structure that stores neighbors of each instance
        """
        self.nan_edges = {}  # Graph of mutual neighbors
        self.nan_num = {}  # Number of natural neighbors for each instance
        self.repeat = {}  # Data structure that counts repetitions of the count method
        self.target = []  # Set of classes
        self.data = []  # Set of instances
        self.knn = {}  # Structure that stores neighbors of each instance

    def load(self, filename):
        """
        Load dataset from a CSV file, separating attributes and classes.

        Args:
            filename (str): Path to the CSV file containing the dataset
        """
        aux = []
        with open(filename, "r") as dataset:
            data = list(csv.reader(dataset))
            for inst in data:
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                aux.append(row)
        self.data = np.array(aux)

    def read(self, data: np.ndarray):
        """
        Load data directly from a numpy array.

        Args:
            data (np.ndarray): Input data array
        """
        self.data = data
        self._distance_matrix = None

    def read_distance_matrix(self, distance_matrix: np.ndarray):
        """
        Load a precomputed distance matrix for neighbor search.

        When a distance matrix is provided, :meth:`algorithm` uses
        argsort-based neighbor lookup instead of a KDTree, avoiding the
        need to hold the raw high-dimensional data in memory.

        Args:
            distance_matrix (np.ndarray): Symmetric (n, n) distance matrix.
        """
        self._distance_matrix = distance_matrix
        # Set data length so existing helpers (asserts, count, etc.) work
        self.data = np.empty((distance_matrix.shape[0], 0))

    def asserts(self):
        """
        Initialize data structures for the algorithm.

        Sets up the necessary data structures including:
        - nan_edges as an empty set
        - knn, nan_num, and repeat dictionaries for each instance
        """
        self.nan_edges = set()
        for j in range(len(self.data)):
            self.knn[j] = set()
            self.nan_num[j] = 0
            self.repeat[j] = 0

    def count(self):
        """
        Count the number of instances that have no natural neighbors.

        Returns:
            int: Number of instances with zero natural neighbors
        """
        nan_zeros = 0
        for x in self.nan_num:
            if self.nan_num[x] == 0:
                nan_zeros += 1
        return nan_zeros

    def findKNN(self, inst, r, tree):
        """
        Find the indices of the k nearest neighbors.

        Args:
            inst: Instance to find neighbors for
            r (int): Radius/parameter for neighbor search
            tree: KDTree object for efficient neighbor search

        Returns:
            np.ndarray: Array of neighbor indices (excluding the instance itself)
        """
        _, ind = tree.query([inst], r + 1)
        return np.delete(ind[0], 0)

    def _findKNN_precomputed(self, i, r):
        """
        Find the r nearest neighbors of point *i* using the precomputed
        distance matrix (argsort-based, exact).

        Args:
            i (int): Index of the query point.
            r (int): Number of neighbors to return.

        Returns:
            np.ndarray: Array of neighbor indices (excluding point *i*).
        """
        dists = self._distance_matrix[i]
        # argsort gives indices sorted by distance; index 0 is self (distance 0)
        sorted_idx = np.argsort(dists)
        # Skip self at position 0 (distance to self is always 0)
        return sorted_idx[1 : r + 1]

    def algorithm(self):
        """
        Execute the Natural Neighbor algorithm.

        The algorithm iteratively expands the neighborhood radius until convergence,
        finding mutual neighbors between instances.

        When a precomputed distance matrix is available (see
        :meth:`read_distance_matrix`), neighbor lookup is performed via
        argsort instead of a KDTree, which avoids holding the raw
        high-dimensional data in memory.

        Returns:
            int: The final radius value when convergence is reached
        """
        use_precomputed = (
            hasattr(self, "_distance_matrix")
            and self._distance_matrix is not None
        )

        if not use_precomputed:
            # Initialize KDTree for efficient neighbor search
            tree = KDTree(self.data)
        else:
            tree = None  # not used in precomputed path

        self.asserts()
        flag = 0
        r = 1

        n_points = len(self.data)
        max_r = n_points - 1  # r + 1 must not exceed n_points
        while flag == 0:
            for i in range(n_points):
                if use_precomputed:
                    knn = self._findKNN_precomputed(i, r)
                else:
                    knn = self.findKNN(self.data[i], r, tree)
                n = knn[-1]
                self.knn[i].add(n)
                if i in self.knn[n] and (i, n) not in self.nan_edges:
                    self.nan_edges.add((i, n))
                    self.nan_edges.add((n, i))
                    self.nan_num[i] += 1
                    self.nan_num[n] += 1

            cnt = self.count()
            rep = self.repeat[cnt]
            self.repeat[cnt] += 1
            if cnt == 0 or rep >= math.sqrt(r - rep) or r >= max_r:
                flag = 1
            else:
                r += 1
        return r


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


def should_use_cached_file(
    cache_path: Union[str, Path],
    source_paths: list[Union[str, Path]],
    use_cached: Literal["auto", "always", "never", "error"] = "auto",
) -> bool:
    """
    Determine whether to use a cached intermediate file based on caching policy and file timestamps.

    Args:
        cache_path: Path to the cached intermediate file
        source_paths: List of source file paths that the cache depends on
        use_cached: Caching policy
            - "auto": Use cached if exists and newer than all sources (default)
            - "always": Always use cached if it exists
            - "never": Never use cached (always regenerate)
            - "error": Raise error if cached doesn't exist

    Returns:
        bool: True if cached file should be used, False if it should be regenerated

    Raises:
        FileNotFoundError: When use_cached="error" and cache doesn't exist
        ValueError: For invalid use_cached values
    """
    cache_path = Path(cache_path)
    source_paths = [Path(p) for p in source_paths]

    if use_cached == "never":
        return False
    elif use_cached == "error":
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file required but not found: {cache_path}")
        return True
    elif use_cached == "always":
        return cache_path.exists()
    elif use_cached == "auto":
        if not cache_path.exists():
            return False

        # Check if cache is newer than all source files
        cache_mtime = cache_path.stat().st_mtime

        for source_path in source_paths:
            if not source_path.exists():
                continue  # Skip missing source files
            if source_path.stat().st_mtime > cache_mtime:
                logging.info(f"Cache {cache_path.name} is older than {source_path.name}, regenerating")
                return False

        logging.info(f"Using cached intermediate file: {cache_path.name}")
        return True
    else:
        raise ValueError(f"Invalid use_cached value: {use_cached}")


def get_cache_status_message(cache_path: Union[str, Path], use_cached: bool) -> str:
    """Generate a descriptive message about cache usage for logging."""
    cache_path = Path(cache_path)

    if use_cached:
        return f"Using cached intermediate: {cache_path.name}"
    else:
        return f"Regenerating intermediate: {cache_path.name}"


def should_use_cache_unified(
    cache_path: Union[str, Path],
    source_paths: list[Union[str, Path]],
    cache_policy: Literal["auto", "always", "force_regenerate"],
) -> bool:
    """Unified cache decision logic for all intermediate files.

    Args:
        cache_path: Path to the cache file
        source_paths: List of source file paths to check timestamps against
        cache_policy: Caching policy:
            - "auto": Use cache if exists and newer than sources, regenerate with logging if missing/invalid
            - "always": Use cache if exists, raise error if missing/invalid
            - "force_regenerate": Always regenerate and overwrite existing cache

    Returns:
        bool: True if cache should be used, False if should regenerate

    Raises:
        ValueError: If cache_policy is invalid
    """
    if cache_policy == "force_regenerate":
        return False
    elif cache_policy == "always":
        return Path(cache_path).exists()
    elif cache_policy == "auto":
        return should_use_cached_file(cache_path, source_paths, "auto")
    else:
        raise ValueError(f"Invalid cache_policy: {cache_policy}. Must be one of: auto, always, force_regenerate")


def rename_mne_channels(mne_obj):
    """Rename an MNE object's channels in place to canonical abbreviations.

    Applies :func:`resolve_channel` (exact lookup) to every entry of
    ``mne_obj.info['ch_names']``. Format-agnostic — works on any MNE object
    whose raw channel names are declared in :data:`~neurodent.constants.CHANNEL_MAP`.

    Args:
        mne_obj: An MNE object exposing ``info['ch_names']`` (e.g. a ``RawArray``).

    Returns:
        The same ``mne_obj``, with channel names replaced by their canonical abbreviations.
    """
    for i in range(len(mne_obj.info['ch_names'])):
        mne_obj.info['ch_names'][i] = resolve_channel(mne_obj.info['ch_names'][i])
    return mne_obj


def slugify(value, allow_unicode=False):
    """Convert a string to a URL-friendly slug.

    Converts to ASCII (unless *allow_unicode* is ``True``), lowercases,
    removes non-alphanumeric characters (except hyphens and underscores),
    and converts spaces and repeated dashes to single dashes.

    Drop-in replacement for ``django.utils.text.slugify`` using only the
    standard library.

    **Path-safety convention.**
        This is the canonical helper for converting display-friendly identifiers
        (animal IDs, animaldays, genotype strings) into filesystem-safe path
        components.  Any code that constructs a ``Path`` or filename from one of
        these strings **must** route the value through ``slugify(...)`` directly,
        or through one of the ``path_safe_*`` accessors on
        :class:`~neurodent.visualization.WindowAnalysisResult`,
        :class:`~neurodent.visualization.streaming.LazyWindowAnalysisResult`, or
        :class:`~neurodent.visualization.FrequencyDomainSpikeAnalysisResult`.

        Display strings — which may contain ``/``, ``;``, parens, spaces, etc.
        (e.g. the real arxrosa genotype ``Arx(F/y); Rosa(+/wt)``) — are correct
        domain notation and stay unchanged on the public attributes
        (``animal_id``, ``animaldays``, ``genotype``).  They're the source of
        truth for what humans see in logs and plot labels.  Only the
        ``path_safe_*`` accessors return the slugified form.

    Args:
        value: The string to slugify.
        allow_unicode: If ``True``, keep Unicode characters instead of
            transliterating to ASCII.

    Returns:
        str: A URL-safe slug string.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")