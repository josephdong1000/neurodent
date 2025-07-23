import csv
import itertools
import logging
import math
import os
import platform
import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import dateutil.parser
import numpy as np
import pandas as pd
from dateutil.parser import ParserError
from sklearn.neighbors import KDTree

from .. import constants


def convert_units_to_multiplier(current_units, target_units="µV"):
    units_to_mult = {"µV": 1e-6, "mV": 1e-3, "V": 1, "nV": 1e-9}

    assert current_units in units_to_mult.keys(), f"No valid current unit called '{current_units}' found"
    assert target_units in units_to_mult.keys(), f"No valid target unit called '{target_units}' found"

    return units_to_mult[current_units] / units_to_mult[target_units]


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


def convert_colpath_to_rowpath(rowdir_path, col_path, gzip=True, aspath=True):
    # TODO it would make more sense to not have a rowdir_path aparameter, since this is outside the scope of the function
    if not 'ColMajor' in col_path:
        raise ValueError(f"Expected 'ColMajor' in col_path: {col_path}")

    out = Path(rowdir_path) / f"{Path(col_path).stem.replace('ColMajor', 'RowMajor')}"
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
        int: The extracted index number.

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
        fpath = str(path_obj.with_suffix(''))
    
    fname = Path(fpath).name
    fname = re.split(r"\D+", fname)
    fname = list(filter(None, fname))
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


def nanaverage(A, weights, axis=-1):
    """
    Average of an array, ignoring NaNs.
    """
    masked = np.ma.masked_array(A, np.isnan(A))
    avg = np.ma.average(masked, axis=axis, weights=weights)
    return avg.filled(np.nan)


def parse_path_to_animalday(
    filepath: str | Path,
    animal_param: tuple[int, str] | str | list[str] = (0, None),
    day_sep: str | None = None,
    mode: Literal["nest", "concat", "base", "noday"] = "concat",
    **day_parse_kwargs,
):
    """
    Parses the filename of a binfolder to get the animalday identifier (animal id, genotype, and day).

    Args:
        filepath (str | Path): Filepath of the binfolder.
        animal_param (tuple[int, str] | str | list[str], optional): Parameter specifying how to parse the animal ID:
            tuple[int, str]: (index, separator) for simple split and index
            str: regex pattern to extract ID
            list[str]: list of possible animal IDs to match against
        day_sep (str, optional): Separator for day in filename. Defaults to None.
        mode (Literal['nest', 'concat', 'base', 'noday'], optional): Mode to parse the filename. Defaults to 'concat'.
            'nest': Extracts genotype/animal from parent directory name and date from filename
                   e.g. "/WT_A10/recording_2023-04-01.*" 
            'concat': Extracts all info from filename, expects genotype_animal_date format
                     e.g. "/WT_A10_2023-04-01.*"
            'base': Same as concat
            'noday': Extracts only genotype and animal ID, uses default date
                    e.g. "/WT_A10_recording.*"
        **day_parse_kwargs: Additional keyword arguments to pass to parse_str_to_day function.
                           Common options include parse_params dict for dateutil.parser.parse.

    Returns:
        dict[str, str]: Dictionary with keys "animal", "genotype", "day", and "animalday" (concatenated).
            Example: {"animal": "A10", "genotype": "WT", "day": "Apr-01-2023", "animalday": "A10 WT Apr-01-2023"}

    Raises:
        ValueError: If mode is invalid or required components cannot be extracted
        TypeError: If filepath is not str or Path
    """
    filepath = Path(filepath)
    match mode:
        case "nest":
            geno = parse_str_to_genotype(filepath.parent.name)
            animid = parse_str_to_animal(filepath.parent.name, animal_param=animal_param)
            day = parse_str_to_day(filepath.name, sep=day_sep, **day_parse_kwargs).strftime("%b-%d-%Y")
        case "concat" | "base":
            geno = parse_str_to_genotype(filepath.name)
            animid = parse_str_to_animal(filepath.name, animal_param=animal_param)
            day = parse_str_to_day(filepath.name, sep=day_sep, **day_parse_kwargs).strftime("%b-%d-%Y")
        case "noday":
            geno = parse_str_to_genotype(filepath.name)
            animid = parse_str_to_animal(filepath.name, animal_param=animal_param)
            day = constants.DEFAULT_DAY.strftime("%b-%d-%Y")
        case _:
            raise ValueError(f"Invalid mode: {mode}")
    return {
        "animal": animid,
        "genotype": geno,
        "day": day,
        "animalday": f"{animid} {geno} {day}",
    }


def parse_str_to_genotype(string: str, strict_matching: bool = False) -> str:
    """
    Parses the filename of a binfolder to get the genotype.

    Args:
        string (str): String to parse.
        strict_matching (bool, optional): If True, ensures the input matches exactly one genotype.
            If False, allows overlapping matches and uses longest. Defaults to False for 
            backward compatibility.

    Returns:
        str: Genotype.
        
    Raises:
        ValueError: When string cannot be parsed or contains ambiguous matches in strict mode.
        
    Examples:
        >>> parse_str_to_genotype("WT_A10_data")
        'WT'
        >>> parse_str_to_genotype("WT_KO_comparison", strict_matching=True)  # Would raise error
        ValueError: Ambiguous match...
        >>> parse_str_to_genotype("WT_KO_comparison", strict_matching=False)  # Uses longest match
        'WT'  # or 'KO' depending on which alias is longer
    """
    return _get_key_from_match_values(string, constants.GENOTYPE_ALIASES, strict_matching)


def parse_str_to_animal(string: str, animal_param: tuple[int, str] | str | list[str] = (0, None)) -> str:
    """
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
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", r"A\\d+")
        'A10'
        >>> parse_str_to_animal("subject_123_data.bin", r"\\d+")
        '123'
        
        # List format: possible IDs to match
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", ["A10", "A11", "A12"])
        'A10'
        >>> parse_str_to_animal("WT_A10_data.bin", ["B15", "C20"])  # No match
        ValueError: No matching ID found in WT_A10_data.bin from possible IDs: ['B15', 'C20']
    """
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


def parse_str_to_day(string: str, sep: str = None, parse_params: dict = None, parse_mode: Literal["full", "split", "window", "all"] = "split") -> datetime:
    """
    Parses the filename of a binfolder to get the day.

    Args:
        string (str): String to parse.
        sep (str, optional): Separator to split string by. If None, split by whitespace. Defaults to None.
        parse_params (dict, optional): Parameters to pass to dateutil.parser.parse. Defaults to {'fuzzy':True}.
        parse_mode (Literal["full", "split", "window", "all"], optional): Mode for parsing the string. Defaults to "split".
            "full": Try parsing the entire cleaned string only
            "split": Try parsing individual tokens only
            "window": Try parsing sliding windows of tokens (2-4 tokens) only
            "all": Use all three approaches in sequence
    Returns:
        datetime: Datetime object corresponding to the day of the binfolder.

    Raises:
        ValueError: If no valid date token is found in the string.
        
    Note:
        The function is designed to be conservative to avoid false positives.
        Some complex date formats may parse with the default year (2000) instead
        of the actual year in the string, which is acceptable behavior for
        maintaining safety against false positives.
    """
    if parse_params is None:
        parse_params = {"fuzzy": True}
    
    # Validate parse_mode
    valid_modes = ["full", "split", "window", "all"]
    if parse_mode not in valid_modes:
        raise ValueError(f"Invalid parse_mode: {parse_mode}. Must be one of {valid_modes}")
    
    clean_str = _clean_str_for_date(string)
    # logging.debug(f'raw str: {string}, clean_str: {clean_str}')

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
        for token in tokens:
            try:
                # logging.debug(f'token: {token}')
                date = dateutil.parser.parse(token, default=constants.DEFAULT_DAY, **parse_params)
                if date.year <= 1980:
                    continue
                return date
            except ParserError:
                continue
    
    if parse_mode in ["window", "all"]:
        # Pass 3: Try sliding window of tokens
        tokens = clean_str.split(sep)
        for window_size in range(2, min(5, len(tokens) + 1)):
            for i in range(len(tokens) - window_size + 1):
                grouped = " ".join(tokens[i:i+window_size])
                try:
                    date = dateutil.parser.parse(grouped, default=constants.DEFAULT_DAY, **parse_params)
                    if date.year <= 1980:
                        continue
                    return date
                except ParserError:
                    continue

    raise ValueError(f"No valid date token found in string: {string}")


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


def parse_chname_to_abbrev(channel_name: str, assume_from_number=False, strict_matching=True) -> str:
    """
    Parses the channel name to get the abbreviation.

    Args:
        channel_name (str): Name of the channel.
        assume_from_number (bool, optional): If True, assume the abbreviation based on the last number 
            in the channel name when normal parsing fails. Defaults to False.
        strict_matching (bool, optional): If True, ensures the input matches exactly one L/R alias and 
            one channel alias. If False, allows multiple matches and uses longest. Defaults to True.

    Returns:
        str: Abbreviation of the channel name.
        
    Raises:
        ValueError: When channel_name cannot be parsed or contains ambiguous matches in strict mode.
        KeyError: When assume_from_number=True but the detected number is not a valid channel ID.
        
    Examples:
        >>> parse_chname_to_abbrev("left Aud")
        'LAud'
        >>> parse_chname_to_abbrev("Right VIS")
        'RVis'
        >>> parse_chname_to_abbrev("channel_9", assume_from_number=True)
        'LAud'
        >>> parse_chname_to_abbrev("LRAud", strict_matching=False)  # Would work in non-strict mode
        'LAud'  # Uses longest L/R match
    """
    if channel_name in constants.DEFAULT_ID_TO_NAME.values():
        logging.debug(f"{channel_name} is already an abbreviation")
        return channel_name
    
    try:
        lr = _get_key_from_match_values(channel_name, constants.LR_ALIASES, strict_matching)
        chname = _get_key_from_match_values(channel_name, constants.CHNAME_ALIASES, strict_matching)
    except ValueError as e:
        if assume_from_number:
            logging.warning(f"{channel_name} does not match name aliases. Assuming alias from number in channel name.")
            nums = re.findall(r"\d+", channel_name)
            
            if not nums:
                raise ValueError(f"Expected to find a number in channel name '{channel_name}' when assume_from_number=True, but no numbers were found.")
            
            num = int(nums[-1])
            if num not in constants.DEFAULT_ID_TO_NAME:
                available_ids = sorted(constants.DEFAULT_ID_TO_NAME.keys())
                raise KeyError(f"Channel number {num} found in '{channel_name}' is not a valid channel ID. Available channel IDs: {available_ids}")
            
            return constants.DEFAULT_ID_TO_NAME[num]
        else:
            raise e
    
    return lr + chname


def _get_key_from_match_values(input_string: str, alias_dict: dict, strict_matching: bool = True):
    """
    Find the best matching key from alias dictionary.
    
    Args:
        input_string (str): String to search in
        alias_dict (dict): Dictionary of {key: [aliases]} to match against
        strict_matching (bool): If True, ensures only one alias matches across all keys
        
    Returns:
        str: The key with the best matching alias
        
    Raises:
        ValueError: When no matches found or multiple matches in strict mode
    """
    matches = [
        (key, candidate, len(candidate))
        for key, aliases in alias_dict.items()
        for candidate in aliases
        if candidate in input_string
    ]
    
    if not matches:
        alias_examples = {key: aliases[:2] for key, aliases in alias_dict.items()}  # Show first 2 aliases per key
        raise ValueError(f"{input_string} does not have any matching values. Available aliases (examples): {alias_examples}")
    
    if strict_matching:
        # Check if multiple different keys match
        matching_keys = set(match[0] for match in matches)
        if len(matching_keys) > 1:
            matched_aliases = {key: [alias for k, alias, _ in matches if k == key] for key in matching_keys}
            raise ValueError(f"Ambiguous match in '{input_string}'. Multiple alias types matched: {matched_aliases}. Use strict_matching=False to allow ambiguous matches.")
    
    # Return the key with the longest matching alias
    best_match_key, _, _ = max(matches, key=lambda x: x[2])
    return best_match_key


def set_temp_directory(path):
    """Set the temporary directory for PyEEG operations.

    Args:
        path (str or Path): Path to the temporary directory. Will be created if it doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(path)
    logging.info(f"Temporary directory set to {path}")


def get_temp_directory() -> Path:
    """
    Returns the temporary directory.
    """
    return Path(os.environ["TMPDIR"])


# def _get_groupby_keys(df: pd.DataFrame, groupby: str | list[str]):
#     """
#     Get the unique values of the groupby variable.
#     """
#     return list(df.groupby(groupby).groups.keys())


# def _get_pairwise_combinations(x: list):
#     """
#     Get all pairwise combinations of a list.
#     """
#     return list(itertools.combinations(x, 2))


class _HiddenPrints:
    def __init__(self, silence=True) -> None:
        self.silence = silence

    def __enter__(self):
        if self.silence:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silence:
            sys.stdout.close()
            sys.stdout = self._original_stdout



def nanmean_series_of_np(x: pd.Series, axis: int = 0):
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
    return np.log(rec + 1)


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
            raise ValueError(f"Column '{col}' contains values not in sort order dictionary: {missing_values}")

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

    def algorithm(self):
        """
        Execute the Natural Neighbor algorithm.

        The algorithm iteratively expands the neighborhood radius until convergence,
        finding mutual neighbors between instances.

        Returns:
            int: The final radius value when convergence is reached
        """
        # Initialize KDTree for efficient neighbor search
        tree = KDTree(self.data)
        self.asserts()
        flag = 0
        r = 1

        while flag == 0:
            for i in range(len(self.data)):
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
            if cnt == 0 or rep >= math.sqrt(r - rep):
                flag = 1
            else:
                r += 1
        return r

class TimestampMapper:
    """
    Map each fragment to its source file's timestamp.
    """

    def __init__(self, file_end_datetimes: list[datetime], file_durations: list[float]):
        """
        Initialize the TimestampMapper.

        Args:
            file_end_datetimes (list[datetime]): The end datetimes of each file.
            file_durations (list[float]): The durations of each file.
        """
        self.file_end_datetimes = file_end_datetimes
        self.file_durations = file_durations

        self.file_start_datetimes = [
            file_end_datetime - timedelta(seconds=file_duration)
            for file_end_datetime, file_duration in zip(self.file_end_datetimes, self.file_durations)
        ]
        self.cumulative_durations = np.cumsum(self.file_durations)

    def get_fragment_timestamp(self, fragment_idx, fragment_len_s) -> datetime:
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