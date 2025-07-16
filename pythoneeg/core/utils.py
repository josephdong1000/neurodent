import csv
import itertools
import logging
import math
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import dateutil.parser
import numpy as np
import pandas as pd
from dateutil.parser import ParserError
from sklearn.neighbors import KDTree

from .. import constants


def convert_path(inputPath):
    # Convert path string to match the os
    system = platform.system()
    home = str(Path.home())


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
    Returns:
        dict[str, str]: Dictionary with keys "animal", "genotype", "day", and "animalday" (concatenated).
    """
    filepath = Path(filepath)
    match mode:
        case "nest":
            geno = parse_str_to_genotype(filepath.parent.name)
            animid = parse_str_to_animal(filepath.parent.name, animal_param=animal_param)
            day = parse_str_to_day(filepath.name, sep=day_sep).strftime("%b-%d-%Y")
        case "concat" | "base":
            geno = parse_str_to_genotype(filepath.name)
            animid = parse_str_to_animal(filepath.name, animal_param=animal_param)
            day = parse_str_to_day(filepath.name, sep=day_sep).strftime("%b-%d-%Y")
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


def parse_str_to_genotype(string: str) -> str:
    """
    Parses the filename of a binfolder to get the genotype.

    Args:
        string (str): String to parse.

    Returns:
        str: Genotype.
    """
    return __get_key_from_match_values(string, constants.GENOTYPE_ALIASES)


def parse_str_to_animal(string: str, animal_param: tuple[int, str] | str | list[str] = (0, None)) -> str:
    """
    Parses the filename of a binfolder to get the animal id.

    Args:
        string (str): String to parse.
        animal_param: Parameter specifying how to parse the animal ID:
            tuple[int, str]: (index, separator) for simple split and index
            str: regex pattern to extract ID
            list[str]: list of possible animal IDs to match against

    Returns:
        str: Animal id.
    """
    match animal_param:
        case (index, sep):
            # 1. split, and get by index. Simple to use, but sometimes inconsistent
            animid = string.split(sep)
            return animid[index]
        case str() as pattern:
            # 2. use regex to pull info. Most general, but hard to use for some users
            match = re.search(pattern, string)
            if match:
                return match.group()
            raise ValueError(f"No match found for pattern {pattern} in string {string}")
        case list() as possible_ids:
            # 3. match to list. Simple to understand, but tedious to use
            for id in possible_ids:
                if id in string:
                    return id
            raise ValueError(f"No matching ID found in {string} from possible IDs: {possible_ids}")
        case _:
            raise ValueError(f"Invalid animal_param type: {type(animal_param)}")


def parse_str_to_day(string: str, sep: str = None, parse_params: dict = {"fuzzy": True}) -> datetime:
    """
    Parses the filename of a binfolder to get the day.

    Args:
        string (str): String to parse.
        sep (str, optional): Separator to split string by. If None, split by whitespace. Defaults to None.
        parse_params (dict, optional): Parameters to pass to dateutil.parser.parse. Defaults to {'fuzzy':True}.

    Returns:
        datetime: Datetime object corresponding to the day of the binfolder.

    Raises:
        ValueError: If no valid date token is found in the string.
    """
    clean_str = _clean_str_for_date(string)
    # logging.debug(f'raw str: {string}, clean_str: {clean_str}')

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

    raise ValueError(f"No valid date token found in string: {string}")


def _clean_str_for_date(string: str):
    """
    Clean a string by removing common non-date tokens and patterns.

    Args:
        string (str): Input string containing date

    Returns:
        str: Cleaned string with non-date tokens removed
    """

    # Create one pattern that matches any of the above
    patterns = constants.DATEPARSER_PATTERNS_TO_REMOVE
    combined_pattern = "|".join(patterns)

    # Remove all matching patterns, replace with space
    cleaned = re.sub(combined_pattern, " ", string, flags=re.IGNORECASE)

    # Clean up extra whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned


def parse_chname_to_abbrev(channel_name: str, assume_from_number=False) -> str:
    """
    Parses the channel name to get the abbreviation.

    Args:
        channel_name (str): Name of the channel.
        assume_from_number (bool, optional): If True, assume the abbreviation based on the last number in the channel name. Defaults to False.

    Returns:
        str: Abbreviation of the channel name.
    """
    # REVIEW maybe DEFAULT_ID_TO_NAME is not the right place to get default abbreviations
    if channel_name in constants.DEFAULT_ID_TO_NAME.values():
        logging.debug(f"{channel_name} is already an abbreviation")
        return channel_name
    try:
        lr = __get_key_from_match_values(channel_name, constants.LR_ALIASES)
        chname = __get_key_from_match_values(channel_name, constants.CHNAME_ALIASES)
    except ValueError as e:
        if assume_from_number:
            logging.warning(f"{channel_name} does not match name aliases. Assuming alias from number in channel name.")
            nums = re.findall(r"\d+", channel_name)
            num = int(nums[-1])
            return constants.DEFAULT_ID_TO_NAME[num]
        else:
            raise e
    return lr + chname


def __get_key_from_match_values(searchonvals: str, dictionary: dict):
    for k, v in dictionary.items():
        if any([x in searchonvals for x in v]):
            return k
    raise ValueError(f"{searchonvals} does not have any matching values. Available values: {dictionary}")


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


class _CustomNamedTemporaryFile:
    """
    This custom implementation is needed because of the following limitation of tempfile.NamedTemporaryFile:

    > Whether the name can be used to open the file a second time, while the named temporary file is still open,
    > varies across platforms (it can be so used on Unix; it cannot on Windows NT or later).
    """

    def __init__(self, mode="wb", delete=True):
        self._mode = mode
        self._delete = delete

    def __enter__(self):
        # Generate a random temporary file name
        file_name = os.path.join(get_temp_directory(), os.urandom(24).hex())
        # Ensure the file is created
        open(file_name, "x").close()
        # Open the file in the given mode
        self._tempFile = open(file_name, self._mode)
        return self._tempFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tempFile.close()
        if self._delete:
            os.remove(self._tempFile.name)


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


def clean_channel_name(name):
    # Get the parts after the last '/'
    if "/" in name:
        name = name.split("/")[-1]

    # Split by spaces
    parts = name.split()

    # For channels with Ctx at the end (L Aud Ctx, R Vis Ctx)
    if parts[-1] == "Ctx" and len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{parts[-1]}"

    # For other channels (L Hipp, R Barrel, etc.)
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"

    # Fallback for anything else
    return name


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


def sort_dataframe_by_plot_order(df: pd.DataFrame, df_sort_order: dict = constants.DF_SORT_ORDER) -> pd.DataFrame:
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
    if not isinstance(df_sort_order, dict):
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
