"""Implementation of AnimalFeatureParser moved out of results.py.

This module contains the implementation extracted from the large
`results.py` file so it can be maintained independently.
"""
import numpy as np
import pandas as pd

from .. import constants
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for type checking and static analysis only; these modules may
    # import heavy optional dependencies (mne) and should not be required at
    # runtime during minimal imports.
    from .. import core  # type: ignore
else:
    # Import core lazily at runtime inside functions that need it to avoid
    # optional-heavy import failures during package-level imports.
    core = None
from .feature_utils import extract_linear_array, extract_band_dict, extract_hist_data


class AnimalFeatureParser:
    def _average_feature(
        self, df: pd.DataFrame, colname: str, weightsname: str | None = "duration"
    ):
        column = df[colname]
        if weightsname is None or weightsname not in df.columns:
            weights = np.ones(column.size)
        else:
            weights = df[weightsname]
        weights = np.asarray(weights)

        ftype = constants.classify_feature(colname)
        if ftype in (constants.FeatureType.LINEAR, constants.FeatureType.LINEAR_2D, constants.FeatureType.SIMPLE_MATRIX):
            col_agg = extract_linear_array(column)
            # Import core lazily to avoid heavy optional deps at module import time
            from .. import core as _core
            avg = _core.nanaverage(col_agg, axis=0, weights=weights)

        elif ftype.is_dict_stored:
            vals, keys = extract_band_dict(column)
            from .. import core as _core
            avg_vals = _core.nanaverage(vals, axis=0, weights=weights)
            # vals is canonical (W, C, B) for BAND or (W, C, C, B) for BANDED_MATRIX.
            # avg_vals after axis=0 is (C, B) or (C, C, B).
            # Bands are always on the last axis — use [..., i] to slice per band.
            avg = {keys[i]: avg_vals[..., i] for i in range(len(keys))}

        elif ftype is constants.FeatureType.HIST:
            coords, values = extract_hist_data(column)
            # values is canonical (W, C, F); average over windows → (C, F).
            # Transpose to (F, C) to preserve per-cell storage format.
            from .. import core as _core
            avg = (coords[0], _core.nanaverage(values, axis=0, weights=weights).T)

        else:
            raise TypeError(
                f"Unsupported FeatureType {ftype} for averaging column {colname}"
            )

        return avg

__all__ = ["AnimalFeatureParser"]
