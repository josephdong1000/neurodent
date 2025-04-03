# Standard library imports
import os
import sys
import tempfile
import re
from datetime import datetime
from pathlib import Path
import platform
import logging
import copy
import itertools

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

def _sanitize_feature_request(features: list[str], exclude: list[str]=[]):
    """
    Sanitizes a list of requested features.

    Args:
        features (list[str]): List of features to include. If "all", include all features in constants.FEATURES.
        exclude (list[str], optional): List of features to exclude. Defaults to [].

    Returns:
        list[str]: Sanitized list of features.
    """
    if features == ["all"]:
        feat = copy.deepcopy(constants.FEATURES)
    elif not features:
        raise ValueError("Features cannot be empty")
    else:
        assert all(f in constants.FEATURES for f in features), f"Available features are: {constants.FEATURES}"
        feat = copy.deepcopy(features)
    if exclude is not None:
        for e in exclude:
            try:
                feat.remove(e)
            except ValueError:
                pass
    return feat

def parse_path_to_animalday(filepath:str|Path, id_index:int=0, delimiter:str=' ', date_pattern=None):
    """
    Parses the filename of a binfolder to get the animalday identifier (animal id, genotype, and day).

    Args:
        filepath (str | Path): Filepath of the binfolder.
        id_index (int, optional): Index of the animal id in the filename. Defaults to 0, i.e. the first element in the filename.
        delimiter (str, optional): Delimiter in the filename. Defaults to ' '.
        date_pattern (str, optional): Date pattern in the filename. If None, the date pattern is assumed to be "MM DD YYYY".

    Returns:
        str: Animal day in the format "animal-id genotype day".
    """
    # not tested if this handles alternative __readmodes yet
    geno = parse_path_to_genotype(filepath)

    animid = parse_path_to_animal(filepath, id_index, delimiter)
    
    day = parse_path_to_day(filepath, date_pattern=date_pattern).strftime("%b-%d-%Y")

    return f"{animid} {geno} {day}"

def parse_path_to_genotype(filepath:str|Path):
    """
    Parses the filename of a binfolder to get the genotype.

    Args:
        filepath (str | Path): Filepath of the binfolder.

    Returns:
        str: Genotype.
    """
    name = Path(filepath).name
    return __get_key_from_match_values(name, constants.GENOTYPE_ALIASES)

def parse_path_to_animal(filepath:str|Path, id_index:int=0, delimiter:str=' '):
    """
    Parses the filename of a binfolder to get the animal id.

    Args:
        filepath (str | Path): Filepath of the binfolder.
        id_index (int, optional): Index of the animal id in the filename. Defaults to 0, i.e. the first element in the filename.
        delimiter (str, optional): Delimiter in the filename. Defaults to ' '.

    Returns:
        str: Animal id.
    """
    animid = Path(filepath).name.split(delimiter)
    animid = animid[id_index]
    return animid

def parse_path_to_day(filepath:str|Path, date_pattern=None) -> datetime:
    """
    Parses the filename of a binfolder to get the day.

    Args:
        filepath (str | Path): Filepath of the binfolder.
        date_pattern (str, optional): Date pattern in the filename. If None, the date pattern is assumed to be "MM DD YYYY".

    Returns:
        datetime: Day of the binfolder.
    """
    date_pattern = r'(\d{2})\D*(\d{2})\D*(\d{4})' if date_pattern is None else date_pattern
    match = re.search(date_pattern, Path(filepath).name)
    if match:
        month, day, year = match.groups()
        month = int(month)
        day = int(day)
        year = int(year)
    else:
        month, day, year = (1, 1, 1)
    return datetime(year=year, month=month, day=day)

def parse_chname_to_abbrev(channel_name:str, assume_from_number=False):
    """
    Parses the channel name to get the abbreviation.

    Args:
        channel_name (str): Name of the channel.
        assume_from_number (bool, optional): If True, assume the abbreviation based on the last number in the channel name. Defaults to False.

    Returns:
        str: Abbreviation of the channel name.
    """
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