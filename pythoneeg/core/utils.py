# Standard library imports
import re
from datetime import datetime
from pathlib import Path


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
