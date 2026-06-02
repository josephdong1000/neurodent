import copy

import numpy as np
import pandas as pd

from .. import constants, core
from .feature_utils import extract_band_dict, extract_hist_data, extract_linear_array


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
            avg = core.nanaverage(col_agg, axis=0, weights=weights)

        elif ftype.is_dict_stored:
            vals, keys = extract_band_dict(column)
            avg_vals = core.nanaverage(vals, axis=0, weights=weights)
            # vals is canonical (W, C, B) for BAND or (W, C, C, B) for BANDED_MATRIX.
            # avg_vals after axis=0 is (C, B) or (C, C, B).
            # Bands are always on the last axis — use [..., i] to slice per band.
            avg = {keys[i]: avg_vals[..., i] for i in range(len(keys))}

        elif ftype is constants.FeatureType.HIST:
            coords, values = extract_hist_data(column)
            # values is canonical (W, C, F); average over windows → (C, F).
            # Transpose to (F, C) to preserve per-cell storage format.
            avg = (coords[0], core.nanaverage(values, axis=0, weights=weights).T)

        else:
            raise TypeError(
                f"Unsupported FeatureType {ftype} for averaging column {colname}"
            )

        return avg


def _sanitize_feature_request(
    features: list[str] | str | None, exclude: list[str] | str = []
):
    """
    Sanitizes a list of requested features for WindowAnalysisResult

    Args:
        features (list[str] | str | None): List of features to include, a single feature
            name as a string, or None to include all features. If ``"all"``, include all
            features in constants.FEATURES except for those in ``exclude``.
        exclude (list[str] | str, optional): Feature or list of features to exclude.
            Defaults to [].

    Returns:
        list[str]: Sanitized list of features.
    """
    if features is None:
        features = ["all"]
    if isinstance(features, str):
        features = [features]
    if isinstance(exclude, str):
        exclude = [exclude]
    if features == ["all"]:
        feat = copy.deepcopy(constants.FEATURES)
    elif not features:
        raise ValueError("Features cannot be empty")
    else:
        if not all(f in constants.FEATURES for f in features):
            raise ValueError(f"Available features are: {constants.FEATURES}")
        feat = copy.deepcopy(features)
    if exclude is not None:
        for e in exclude:
            try:
                feat.remove(e)
            except ValueError:
                pass
    return feat


