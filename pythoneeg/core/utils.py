# Standard library imports
import os
import sys
import tempfile
import re
from datetime import datetime
import dateutil.parser
from dateutil.parser import ParserError
from pathlib import Path
import platform
import logging
import copy
import itertools
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from .. import constants

def convert_path(inputPath):
    # Convert path string to match the os
    system = platform.system()
    home = str(Path.home())

    
def convert_units_to_multiplier(current_units, target_units='µV'):
    units_to_mult = {'µV' : 1e-6,
                     'mV' : 1e-3,
                     'V' : 1,
                     'nV' : 1e-9}

    assert current_units in units_to_mult.keys(), f"No valid current unit called '{current_units}' found"
    assert target_units in units_to_mult.keys(), f"No valid target unit called '{target_units}' found"

    return units_to_mult[current_units] / units_to_mult[target_units]


def is_day(dt: datetime, sunrise=6, sunset=18):
    return sunrise <= dt.hour < sunset


def convert_colpath_to_rowpath(rowdir_path, col_path, gzip=True, aspath=True):
    out = Path(rowdir_path) / f'{Path(col_path).stem.replace("ColMajor", "RowMajor")}'
    if gzip:
        out = str(out) + ".npy.gz"
    else:
        out = str(out) + ".bin"
    return Path(out) if aspath else out


def filepath_to_index(filepath) -> int:
    fpath = str(filepath)
    for suffix in ['_RowMajor', '_ColMajor', '_Meta']:
        fpath = fpath.replace(suffix, '')
    fpath = fpath.removesuffix(''.join(Path(fpath).suffixes))
    fname = Path(fpath).name
    fname = re.split(r'\D+', fname)
    fname = list(filter(None, fname))
    return int(fname[-1])

def nanaverage(A, weights, axis=-1):
    """
    Average of an array, ignoring NaNs.
    """
    masked = np.ma.masked_array(A, np.isnan(A))
    avg = np.ma.average(masked, axis=axis, weights=weights)
    return avg.filled(np.nan)


def parse_path_to_animalday(filepath:str|Path, 
                            animal_param:tuple[int, str]|str|list[str] = (0, None),
                            day_sep:str|None = None,
                            mode:Literal['nest', 'concat', 'base', 'noday']='concat'):
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
        case 'nest':
            geno = parse_str_to_genotype(filepath.parent.name)
            animid = parse_str_to_animal(filepath.parent.name, animal_param=animal_param)
            day = parse_str_to_day(filepath.name, sep=day_sep).strftime("%b-%d-%Y")
        case 'concat' | 'base':
            geno = parse_str_to_genotype(filepath.name)
            animid = parse_str_to_animal(filepath.name, animal_param=animal_param)
            day = parse_str_to_day(filepath.name, sep=day_sep).strftime("%b-%d-%Y")
        case 'noday':
            geno = parse_str_to_genotype(filepath.name)
            animid = parse_str_to_animal(filepath.name, animal_param=animal_param)
            day = constants.DEFAULT_DAY.strftime("%b-%d-%Y")
        case _:
            raise ValueError(f"Invalid mode: {mode}")
    return {'animal': animid, 'genotype': geno, 'day': day, 'animalday': f"{animid} {geno} {day}"}

def parse_str_to_genotype(string:str) -> str:
    """
    Parses the filename of a binfolder to get the genotype.

    Args:
        string (str): String to parse.

    Returns:
        str: Genotype.
    """
    return __get_key_from_match_values(string, constants.GENOTYPE_ALIASES)

def parse_str_to_animal(string:str, animal_param:tuple[int, str]|str|list[str] = (0, None)) -> str:
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
            # 3. match to list. Simple to understand, but tedious for users
            for id in possible_ids:
                if id in string:
                    return id
            raise ValueError(f"No matching ID found in {string} from possible IDs: {possible_ids}")
        case _:
            raise ValueError(f"Invalid animal_param type: {type(animal_param)}")

def parse_str_to_day(string:str, sep:str=None, parse_params:dict={'fuzzy':True}) -> datetime:
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

def _clean_str_for_date(string:str):
    """
    Clean a string by removing common non-date tokens and patterns.
    
    Args:
        string (str): Input string containing date
        
    Returns:
        str: Cleaned string with non-date tokens removed
    """
    
    # Create one pattern that matches any of the above
    patterns = constants.DATEPARSER_PATTERNS_TO_REMOVE
    combined_pattern = '|'.join(patterns)
    
    # Remove all matching patterns, replace with space
    cleaned = re.sub(combined_pattern, ' ', string, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def parse_chname_to_abbrev(channel_name:str, assume_from_number=False) -> str:
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
            nums = re.findall(r'\d+', channel_name)
            num = int(nums[-1])
            return constants.DEFAULT_ID_TO_NAME[num]
        else:
            raise e
    return lr + chname

def __get_key_from_match_values(searchonvals:str, dictionary:dict):
    for k,v in dictionary.items():
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
    os.environ['TMPDIR'] = str(path)
    logging.info(f"Temporary directory set to {path}")


def get_temp_directory() -> Path:
    """
    Returns the temporary directory.
    """
    return Path(os.environ['TMPDIR'])


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
    def __init__(self, mode='wb', delete=True):
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
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silence:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def clean_channel_name(name):
    # Get the parts after the last '/'
    if '/' in name:
        name = name.split('/')[-1]
    
    # Split by spaces
    parts = name.split()
    
    # For channels with Ctx at the end (L Aud Ctx, R Vis Ctx)
    if parts[-1] == 'Ctx' and len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
    
    # For other channels (L Hipp, R Barrel, etc.)
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    
    # Fallback for anything else
    return name

def nanmean_series_of_np(x: pd.Series, axis: int = 0):
    logging.debug(f'Unique shapes in x: {set(np.shape(item) for item in x)}')
    xmean: np.ndarray = np.nanmean(np.array(list(x)), axis=axis)
    return xmean