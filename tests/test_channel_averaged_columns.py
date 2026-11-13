"""Every feature's channel-averaged columns are the ones channel_averaged_columns names.

The zeitgeber extraction used to name columns by its own rule, which kept psdslope whole, so
psdslope_slope and psdslope_intercept never reached the zeitgeber outputs while
get_channel_averaged_result made them. One function now names them for every caller.
"""

from unittest.mock import patch

import numpy as np
import pytest

from neurodent import constants
from neurodent.core.utils import channel_averaged_columns
from neurodent.results import WindowAnalysisResult

AVERAGEABLE = [f for f in constants.FEATURES if f not in constants.HIST_FEATURES]
TEN = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]

# The ways real WARs differ: channel count, days, a single window, a dead channel, and a
# channel order other than the canonical one.
SHAPES = {
    "three channels": ("a1", "A1", "WT", "Male", {}),
    "ten channels, two days": ("a2", "A2", "KO", "Female", {"channels": TEN, "days": 2}),
    "two channels, one window": ("a3", "A3", "WT", "Male", {"channels": ["RMot", "LMot"], "windows": 1}),
    "dead channel": ("a4", "A4", "KO", "Female", {"dead_channel": 1}),
}


class TestNames:
    def test_linear_and_simple_matrix_features_keep_their_name(self):
        for feature in constants.LINEAR_FEATURES + constants.SIMPLE_MATRIX_FEATURES:
            assert channel_averaged_columns(feature) == [feature]

    def test_band_and_banded_matrix_features_give_one_column_per_band(self):
        for feature in constants.BAND_FEATURES + constants.BANDED_MATRIX_FEATURES:
            assert channel_averaged_columns(feature) == [
                f"{feature}_{band}" for band in ["delta", "theta", "alpha", "beta", "gamma"]
            ]

    def test_psdslope_gives_one_column_per_component(self):
        assert channel_averaged_columns("psdslope") == ["psdslope_slope", "psdslope_intercept"]

    def test_psd_has_no_channel_average(self):
        with pytest.raises(ValueError, match="spectrum"):
            channel_averaged_columns("psd")

    def test_an_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            channel_averaged_columns("logpsband")

    def test_a_component_feature_without_labels_raises(self):
        with patch.object(constants, "COMPONENT_LABELS", {}):
            with pytest.raises(ValueError, match="COMPONENT_LABELS"):
                channel_averaged_columns("psdslope")


def _feature_columns(war, df):
    return [c for c in df.columns if c not in war._nonfeature_columns]


@pytest.mark.parametrize("shape", list(SHAPES))
def test_each_feature_gives_exactly_its_named_columns(create_war_files, shape):
    ((parquet, _),) = create_war_files([SHAPES[shape]])
    war = WindowAnalysisResult.load_parquet_and_json(parquet.parent)

    for feature in AVERAGEABLE:
        df = war.get_channel_averaged_result(features=[feature])
        columns = _feature_columns(war, df)
        assert columns == channel_averaged_columns(feature), feature
        assert len(df) == len(war.result), feature
        for column in columns:
            assert all(isinstance(v, (float, np.floating)) for v in df[column]), (feature, column)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("request_all", [None, "all", ["all"]])
def test_every_feature_at_once_leaves_psd_out(create_war_files, shape, request_all):
    """psd used to come through unaveraged, as the spectrum tuples the WAR stores."""
    ((parquet, _),) = create_war_files([SHAPES[shape]])
    war = WindowAnalysisResult.load_parquet_and_json(parquet.parent)

    df = war.get_channel_averaged_result(features=request_all)

    expected = {c for feature in AVERAGEABLE for c in channel_averaged_columns(feature)}
    assert set(_feature_columns(war, df)) == expected
    assert "psd" not in df.columns


def test_naming_psd_raises(create_war_files):
    ((parquet, _),) = create_war_files([SHAPES["three channels"]])
    war = WindowAnalysisResult.load_parquet_and_json(parquet.parent)

    with pytest.raises(ValueError, match="spectrum"):
        war.get_channel_averaged_result(features=["logrms", "psd"])


def test_a_feature_the_war_lacks_is_left_out(create_war_files):
    ((parquet, _),) = create_war_files([("a5", "A5", "WT", "Female", {"drop": ["psdslope", "zcohere"]})])
    war = WindowAnalysisResult.load_parquet_and_json(parquet.parent)

    df = war.get_channel_averaged_result(features=["logrms", "psdslope", "zcohere"])

    assert _feature_columns(war, df) == ["logrms"]
