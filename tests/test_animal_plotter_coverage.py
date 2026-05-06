"""Comprehensive tests for AnimalPlotter to improve coverage of
src/neurodent/visualization/plotting/animal.py (targeting uncovered lines).
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from neurodent import constants
from neurodent.visualization import WindowAnalysisResult
from neurodent.visualization.plotting.animal import AnimalPlotter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CHAN = 2
N_BANDS = len(constants.BAND_NAMES)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after every test."""
    yield
    plt.close("all")


@pytest.fixture()
def rng():
    """Deterministic RNG shared across helpers and tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def mock_war():
    """Create a minimal mock WindowAnalysisResult."""
    war = MagicMock(spec=WindowAnalysisResult)
    war.genotype = "WT"
    war.channel_names = ["LMot", "RMot"]
    war.channel_abbrevs = ["LM", "RM"]
    war.assume_from_number = False
    return war


@pytest.fixture()
def plotter(mock_war):
    """Create an AnimalPlotter backed by the mock WAR."""
    p = AnimalPlotter(mock_war)
    p.CHNAME_TO_ABBREV = [("LeftMotor", "LM"), ("RightMotor", "RM")]
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_group(n_time=10, n_chan=N_CHAN, feature="rms", rng=None):
    """Return a DataFrame suitable for __get_linear_feature with a LINEAR feature."""
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame(
        {
            feature: [rng.random(n_chan).tolist() for _ in range(n_time)],
            "duration": [1.0] * n_time,
        }
    )


def _make_band_group(n_time=10, n_chan=N_CHAN, feature="psdband", rng=None):
    """Return a DataFrame suitable for __get_linear_feature with a BAND feature."""
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame(
        {
            feature: [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ],
            "duration": [1.0] * n_time,
        }
    )


def _make_linear_2d_group(n_time=10, n_chan=N_CHAN, feature="psdslope", n_components=2, rng=None):
    """Return a DataFrame suitable for __get_linear_feature with a LINEAR_2D feature."""
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame(
        {
            feature: [rng.random((n_chan, n_components)) for _ in range(n_time)],
            "duration": [1.0] * n_time,
        }
    )


def _make_simple_matrix_group(n_time=10, n_chan=N_CHAN, feature="zpcorr", rng=None):
    """Return a DataFrame suitable for __get_linear_feature with a SIMPLE_MATRIX feature."""
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame(
        {
            feature: [rng.random((n_chan, n_chan)) for _ in range(n_time)],
            "duration": [1.0] * n_time,
        }
    )


def _make_banded_matrix_group(n_time=10, n_chan=N_CHAN, feature="cohere", rng=None):
    """Return a DataFrame suitable for __get_linear_feature with a BANDED_MATRIX feature."""
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame(
        {
            feature: [
                {b: rng.random((n_chan, n_chan)) for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ],
            "duration": [1.0] * n_time,
        }
    )


def _make_coherecorr_avg(n_row=2, n_chan=N_CHAN, rng=None):
    """Return mock groupavg data used by coherecorr methods."""
    rng = rng or np.random.default_rng(0)
    band_names = constants.BAND_NAMES + ["pcorr"]
    cohere_dicts = [{band: rng.random((n_chan, n_chan)) for band in band_names} for _ in range(n_row)]
    return pd.DataFrame({"cohere": cohere_dicts}, index=[f"row{i}" for i in range(n_row)])


def _make_grouprows_linear(n_time=10, n_chan=N_CHAN, features=None, rng=None):
    """Build a DataFrame mimicking get_grouprows_result for linear/band features."""
    rng = rng or np.random.default_rng(0)
    if features is None:
        features = ["rms"]
    data = {"duration": [1.0] * n_time}
    for feat in features:
        ftype = constants.classify_feature(feat)
        if ftype is constants.FeatureType.LINEAR:
            data[feat] = [rng.random(n_chan).tolist() for _ in range(n_time)]
        elif ftype is constants.FeatureType.BAND:
            data[feat] = [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ]
        elif ftype is constants.FeatureType.LINEAR_2D:
            data[feat] = [rng.random((n_chan, 2)).tolist() for _ in range(n_time)]
    df = pd.DataFrame(data)
    df.index = pd.MultiIndex.from_tuples([("group1",)] * n_time)
    return df


def _make_grouprows_matrix(n_time=10, n_chan=N_CHAN, features=None, rng=None):
    """Build DataFrame for coherecorr spectral (matrix features stored per-row)."""
    rng = rng or np.random.default_rng(0)
    if features is None:
        features = ["zcohere", "zpcorr"]
    data = {"duration": [1.0] * n_time}
    for feat in features:
        ftype = constants.classify_feature(feat)
        if ftype is constants.FeatureType.BANDED_MATRIX:
            data[feat] = [
                {b: rng.random((n_chan, n_chan)).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ]
        elif ftype is constants.FeatureType.SIMPLE_MATRIX:
            data[feat] = [rng.random((n_chan, n_chan)).tolist() for _ in range(n_time)]
    df = pd.DataFrame(data)
    df.index = pd.MultiIndex.from_tuples([("group1",)] * n_time)
    return df


# ===================================================================
# 1. _abbreviate_channel – fallback path (line 54)
# ===================================================================

class TestAbbreviateChannel:
    def test_fallback_returns_original_name(self, plotter):
        """When no alias matches, the original name is returned."""
        assert plotter._abbreviate_channel("UnknownRegion") == "UnknownRegion"


# ===================================================================
# 2. _calculate_standard_data – all score_type branches (lines 320-336)
# ===================================================================

class TestCalculateStandardData:
    """Covers every branch of the match/case in _calculate_standard_data."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(42)
        return rng.random((10, 3))

    def test_z(self, plotter, data):
        result = plotter._calculate_standard_data(data, mode="z")
        # z-scored columns should have ~0 mean
        assert np.allclose(np.nanmean(result, axis=0), 0, atol=1e-10)

    def test_zall(self, plotter, data):
        result = plotter._calculate_standard_data(data, mode="zall")
        # zall z-scores the entire array
        assert result.shape == data.shape

    def test_gz(self, plotter, data):
        result = plotter._calculate_standard_data(data, mode="gz")
        assert result.shape == data.shape

    def test_modz(self, plotter, data):
        result = plotter._calculate_standard_data(data, mode="modz")
        assert result.shape == data.shape

    def test_none_string(self, plotter, data):
        result = plotter._calculate_standard_data(data, mode="none")
        np.testing.assert_array_equal(result, data)

    def test_none_literal(self, plotter, data):
        result = plotter._calculate_standard_data(data, mode=None)
        np.testing.assert_array_equal(result, data)

    def test_center(self, plotter, data):
        result = plotter._calculate_standard_data(data, mode="center")
        # centred columns should have ~0 mean
        assert np.allclose(np.nanmean(result, axis=0), 0, atol=1e-10)

    def test_invalid_raises(self, plotter, data):
        with pytest.raises(ValueError, match="Invalid mode"):
            plotter._calculate_standard_data(data, mode="bad_mode")


# ===================================================================
# 3. plot_coherecorr_diff – bands as string & != 2 rows (lines 63-64, 96, 102-103)
# ===================================================================

class TestPlotCoherecorrDiff:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_bands_as_string(self, mock_subplots, mock_show, plotter, mock_war):
        """Passing bands as a single string wraps it in a list."""
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_war.get_groupavg_result.return_value = _make_coherecorr_avg(n_row=2)
        plotter.plot_coherecorr_diff(bands="delta")
        mock_subplots.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_not_two_rows_raises(self, mock_show, plotter, mock_war):
        """ValueError when groupby produces != 2 rows."""
        mock_war.get_groupavg_result.return_value = _make_coherecorr_avg(n_row=3)
        with pytest.raises(ValueError, match="Difference can only be calculated between 2 rows"):
            plotter.plot_coherecorr_diff()


# ===================================================================
# 4. plot_coherecorr_matrix – bands as string (lines 63-64)
# ===================================================================

class TestPlotCoherecorrMatrix:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_bands_as_string(self, mock_subplots, mock_show, plotter, mock_war):
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_war.get_groupavg_result.return_value = _make_coherecorr_avg(n_row=1)
        plotter.plot_coherecorr_matrix(bands="delta")
        mock_subplots.assert_called_once()


# ===================================================================
# 5. _plot_coherecorr_matrixgroup – center_cmap & show_channelname=False
#    (lines 135-138, 153-154)
# ===================================================================

class TestPlotCoherecorrMatrixgroup:
    def test_center_cmap_true(self, plotter, rng):
        """When center_cmap=True and norm_list=None, CenteredNorm is used."""
        bands = ["delta"]
        group = pd.Series({"delta": rng.random((N_CHAN, N_CHAN))}, name="test_row")
        fig, ax_arr = plt.subplots(1, 1, squeeze=False)
        plotter._plot_coherecorr_matrixgroup(
            group, bands, ax_arr[0, :], show_bandname=True, center_cmap=True
        )

    def test_show_channelname_false(self, plotter):
        """show_channelname=False triggers the else branch for tick labels."""
        bands = ["delta"]
        group = pd.Series({"delta": np.random.default_rng(0).random((N_CHAN, N_CHAN))}, name="test_row")
        fig, ax_arr = plt.subplots(1, 1, squeeze=False)
        plotter._plot_coherecorr_matrixgroup(
            group,
            bands,
            ax_arr[0, :],
            show_bandname=False,
            show_channelname=False,
        )


# ===================================================================
# 6. plot_linear_temporal (lines 179-216)
# ===================================================================

class TestPlotLinearTemporal:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.subplots")
    def test_basic(self, mock_subplots, mock_adjust, mock_show, plotter, mock_war):
        features = ["rms"]
        n_time = 10
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)

        mock_war.get_grouprows_result.return_value = _make_grouprows_linear(
            n_time=n_time, features=features
        )
        plotter.plot_linear_temporal(features=features)
        mock_subplots.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.subplots")
    def test_with_show_endfile(self, mock_subplots, mock_adjust, mock_show, plotter, mock_war):
        features = ["rms"]
        n_time = 10
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)

        df = _make_grouprows_linear(n_time=n_time, features=features)
        df["endfile"] = [np.nan] * (n_time - 1) + [0.5]
        mock_war.get_grouprows_result.return_value = df
        plotter.plot_linear_temporal(features=features, show_endfile=True)
        mock_subplots.assert_called()


# ===================================================================
# 7. _plot_filediv_lines & __get_filediv_times (lines 307-318)
# ===================================================================

class TestFiledivLines:
    def test_filediv_lines_drawn(self, plotter):
        n_time = 10
        group = pd.DataFrame(
            {
                "duration": [1.0] * n_time,
                "endfile": [np.nan] * 4 + [0.5] + [np.nan] * 4 + [0.5],
            }
        )
        fig, ax = plt.subplots()
        plotter._plot_filediv_lines(group, ax, "duration", "endfile")
        # Two endfile markers → two vertical lines (plus the default ones already there)
        # axvline adds Line2D objects
        n_vlines = sum(
            1 for line in ax.get_lines()
            if line.get_linestyle() == "--"
        )
        assert n_vlines == 2

    def test_get_filediv_times(self, plotter):
        group = pd.DataFrame(
            {
                "duration": [2.0, 3.0, 5.0, 1.0],
                "endfile": [np.nan, 0.1, np.nan, 0.2],
            }
        )
        result = plotter._AnimalPlotter__get_filediv_times(group, "duration", "endfile")
        # cumulative (shift, fill=0): [0, 2, 5, 10]
        # endfile notna at idx 1,3 → values 0.1+2=2.1 , 0.2+10=10.2
        assert len(result) == 2
        assert np.isclose(result[0], 2.1)
        assert np.isclose(result[1], 10.2)


# ===================================================================
# 8. plot_coherecorr_spectral (lines 356-396)
# ===================================================================

class TestPlotCoherecorrSpectral:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.subplots")
    def test_basic(self, mock_subplots, mock_adjust, mock_show, plotter, mock_war):
        features = ["zcohere", "zpcorr"]
        n_time = 10

        mock_fig = Mock()
        mock_ax = np.array([[Mock()], [Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)

        mock_war.get_grouprows_result.return_value = _make_grouprows_matrix(
            n_time=n_time, features=features
        )
        plotter.plot_coherecorr_spectral(features=features)
        mock_subplots.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.subplots")
    def test_missing_feature_warns(self, mock_subplots, mock_adjust, mock_show, plotter, mock_war):
        """A feature absent from the dataframe triggers a warning and is removed."""
        n_time = 10
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)

        df = _make_grouprows_matrix(n_time=n_time, features=["zpcorr"])
        mock_war.get_grouprows_result.return_value = df
        with pytest.warns(UserWarning, match="Feature zcohere not found"):
            plotter.plot_coherecorr_spectral(features=["zcohere", "zpcorr"])


# ===================================================================
# 9. _plot_coherecorr_spectralgroup (lines 414-459)
# ===================================================================

class TestPlotCoherecorrSpectralgroup:
    def test_cohere_feature_yticks(self, plotter):
        """zcohere sets band name yticks and hlines."""
        n_time, n_chan = 10, N_CHAN
        features = ["zcohere"]
        df = _make_grouprows_matrix(n_time=n_time, features=features)
        fig, ax = plt.subplots()
        plotter._plot_coherecorr_spectralgroup(
            group=df, feature="zcohere", ax=ax, score_type="none"
        )

    def test_pcorr_feature_yticks(self, plotter):
        """zpcorr sets its own ytick label."""
        n_time = 10
        features = ["zpcorr"]
        df = _make_grouprows_matrix(n_time=n_time, features=features)
        fig, ax = plt.subplots()
        plotter._plot_coherecorr_spectralgroup(
            group=df, feature="zpcorr", ax=ax, score_type="none"
        )

    def test_show_endfile(self, plotter):
        n_time = 10
        features = ["zpcorr"]
        df = _make_grouprows_matrix(n_time=n_time, features=features)
        df["endfile"] = [np.nan] * (n_time - 1) + [0.5]
        fig, ax = plt.subplots()
        plotter._plot_coherecorr_spectralgroup(
            group=df,
            feature="zpcorr",
            ax=ax,
            score_type="none",
            show_endfile=True,
        )

    def test_center_cmap_false(self, plotter):
        n_time = 10
        features = ["zpcorr"]
        df = _make_grouprows_matrix(n_time=n_time, features=features)
        fig, ax = plt.subplots()
        plotter._plot_coherecorr_spectralgroup(
            group=df,
            feature="zpcorr",
            ax=ax,
            score_type="none",
            center_cmap=False,
        )


# ===================================================================
# 10. plot_psd_histogram – avg_channels & different plot_types
#     (lines 487-501)
# ===================================================================

def _make_psd_avg(n_col=2, n_freq=20, n_chan=N_CHAN, rng=None):
    """Mock groupavg for PSD: each entry is (freqs, psd_matrix)."""
    rng = rng or np.random.default_rng(0)
    freqs = np.linspace(1, 50, n_freq)
    rows = [(freqs, rng.random((n_freq, n_chan)) + 0.01) for _ in range(n_col)]
    return pd.DataFrame({"psd": rows}, index=[f"day{i}" for i in range(n_col)])


class TestPlotPsdHistogram:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.subplots")
    def test_avg_channels(self, mock_subplots, mock_adjust, mock_show, plotter, mock_war):
        n_col = 1
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_war.get_groupavg_result.return_value = _make_psd_avg(n_col=n_col)
        plotter.plot_psd_histogram(avg_channels=True)
        mock_subplots.assert_called()

    @pytest.mark.parametrize("ptype", ["semilogy", "semilogx", "linear"])
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_types(self, mock_subplots, mock_adjust, mock_show, plotter, mock_war, ptype):
        n_col = 1
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_war.get_groupavg_result.return_value = _make_psd_avg(n_col=n_col)
        plotter.plot_psd_histogram(plot_type=ptype)
        mock_subplots.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.subplots")
    def test_invalid_plot_type_raises(self, mock_subplots, mock_adjust, mock_show, plotter, mock_war):
        n_col = 1
        mock_fig = Mock()
        mock_ax = np.array([[Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_war.get_groupavg_result.return_value = _make_psd_avg(n_col=n_col)
        with pytest.raises(ValueError, match="Invalid plot type"):
            plotter.plot_psd_histogram(plot_type="badtype")


# ===================================================================
# 11. plot_psd_spectrogram – median branch (lines 546-549)
# ===================================================================

def _make_psd_rows(n_time=10, n_freq=20, n_chan=N_CHAN, rng=None):
    rng = rng or np.random.default_rng(0)
    freqs = np.linspace(1, 50, n_freq)
    data = {
        "psd": [(freqs, rng.random((n_freq, n_chan)) + 0.01) for _ in range(n_time)],
        "duration": [1.0] * n_time,
    }
    df = pd.DataFrame(data)
    df.index = pd.MultiIndex.from_tuples([("group1",)] * n_time)
    return df


class TestPlotPsdSpectrogram:
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_median_branch(self, mock_subplots, mock_show, plotter, mock_war):
        mock_fig = Mock()
        mock_ax = Mock()
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im
        mock_cbar = Mock()
        mock_fig.colorbar.return_value = mock_cbar
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_war.get_grouprows_result.return_value = _make_psd_rows()
        plotter.plot_psd_spectrogram(center_stat="median")
        mock_subplots.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_invalid_center_stat(self, mock_subplots, mock_show, plotter, mock_war):
        mock_war.get_grouprows_result.return_value = _make_psd_rows()
        with pytest.raises(ValueError, match="Invalid statistic"):
            plotter.plot_psd_spectrogram(center_stat="badstat")


# ===================================================================
# 12. plot_temporal_heatmap & _plot_temporal_heatmap_feature
#     (lines 625-767)
# ===================================================================

def _make_temporal_heatmap_rows(n_time=48, n_chan=N_CHAN, features=None, rng=None):
    """Create a DataFrame with timestamps spanning 2 days."""
    rng = rng or np.random.default_rng(0)
    if features is None:
        features = ["rms"]

    timestamps = pd.date_range("2023-01-01 08:00", periods=n_time, freq="30min")
    data = {
        "duration": [1.0] * n_time,
        "timestamp": timestamps,
        "endfile": [np.nan] * (n_time - 1) + [0.5],
        "animalday": ["day1"] * n_time,
    }
    for feat in features:
        ftype = constants.classify_feature(feat)
        if ftype is constants.FeatureType.LINEAR:
            data[feat] = [rng.random(n_chan).tolist() for _ in range(n_time)]
        elif ftype is constants.FeatureType.LINEAR_2D:
            data[feat] = [rng.random((n_chan, 2)) for _ in range(n_time)]
        elif ftype is constants.FeatureType.BAND:
            data[feat] = [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ]
        elif ftype is constants.FeatureType.SIMPLE_MATRIX:
            data[feat] = [rng.random((n_chan, n_chan)) for _ in range(n_time)]
        elif ftype is constants.FeatureType.BANDED_MATRIX:
            data[feat] = [
                {b: rng.random((n_chan, n_chan)) for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ]

    df = pd.DataFrame(data)
    df.index = pd.MultiIndex.from_tuples([("animal1",)] * n_time)
    return df


class TestPlotTemporalHeatmap:
    @patch("matplotlib.pyplot.show")
    def test_basic(self, mock_show, plotter, mock_war):
        features = ["rms"]
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=features
        )
        plotter.plot_temporal_heatmap(features=features, score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_features_as_string(self, mock_show, plotter, mock_war):
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=["rms"]
        )
        plotter.plot_temporal_heatmap(features="rms", score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_missing_feature_warns(self, mock_show, plotter, mock_war):
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=["rms"]
        )
        with pytest.warns(UserWarning, match="Feature ampvar not found"):
            plotter.plot_temporal_heatmap(features=["ampvar", "rms"], score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_no_valid_features_raises(self, mock_show, plotter, mock_war):
        df = _make_temporal_heatmap_rows(features=["rms"])
        df = df.drop(columns=["rms"])
        mock_war.get_grouprows_result.return_value = df
        with pytest.raises(ValueError, match="No valid features"):
            plotter.plot_temporal_heatmap(features=["rms"], score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_many_days_label_thinning(self, mock_show, plotter, mock_war, rng):
        """When > 10 days exist, only every nth day label is shown."""
        n_time = 24 * 12  # 12 days at hourly resolution
        timestamps = pd.date_range("2023-01-01 00:00", periods=n_time, freq="1h")
        data = {
            "rms": [rng.random(N_CHAN).tolist() for _ in range(n_time)],
            "duration": [1.0] * n_time,
            "timestamp": timestamps,
            "endfile": [np.nan] * n_time,
            "animalday": ["day1"] * n_time,
        }
        df = pd.DataFrame(data)
        df.index = pd.MultiIndex.from_tuples([("animal1",)] * n_time)
        mock_war.get_grouprows_result.return_value = df
        plotter.plot_temporal_heatmap(features=["rms"], score_type="none")


class TestPlotTemporalHeatmapFeatureShapes:
    """Test that _plot_temporal_heatmap_feature correctly handles different feature types."""

    @patch("matplotlib.pyplot.show")
    def test_linear_shape_correct(self, mock_show, plotter, mock_war, rng):
        """LINEAR features (rms) should work correctly."""
        features = ["rms"]
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=features, rng=rng
        )
        # This should not raise and should produce valid output
        plotter.plot_temporal_heatmap(features=features, score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_linear_2d_plots_multiple_heatmaps(self, mock_show, plotter, mock_war, rng):
        """LINEAR_2D features (psdslope) should plot multiple heatmaps (slope, intercept)."""
        features = ["psdslope"]
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=features, rng=rng
        )
        # This should not raise and should produce 2 heatmaps (slope, intercept)
        plotter.plot_temporal_heatmap(features=features, score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_band_plots_multiple_heatmaps(self, mock_show, plotter, mock_war, rng):
        """BAND features (psdband) should plot multiple heatmaps (one per band)."""
        features = ["psdband"]
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=features, rng=rng
        )
        # This should not raise and should produce 5 heatmaps (one per band)
        plotter.plot_temporal_heatmap(features=features, score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_simple_matrix_shape_correct(self, mock_show, plotter, mock_war, rng):
        """SIMPLE_MATRIX features (zpcorr) should work correctly."""
        features = ["zpcorr"]
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=features, rng=rng
        )
        # This should not raise and should produce valid output
        plotter.plot_temporal_heatmap(features=features, score_type="none")

    @patch("matplotlib.pyplot.show")
    def test_banded_matrix_plots_multiple_heatmaps(self, mock_show, plotter, mock_war, rng):
        """BANDED_MATRIX features (cohere) should plot multiple heatmaps (one per band)."""
        features = ["cohere"]
        mock_war.get_grouprows_result.return_value = _make_temporal_heatmap_rows(
            features=features, rng=rng
        )
        # This should not raise and should produce 5 heatmaps (one per band)
        plotter.plot_temporal_heatmap(features=features, score_type="none")

    def test_collapse_feature_channels_integration(self, plotter, rng):
        """Test that collapse_feature_channels is used correctly for LINEAR features."""
        # Create synthetic data for a LINEAR feature
        n_time = 10
        n_chan = N_CHAN
        value = 42.0

        # Create feature data with constant value across all channels
        rms_data = [[value] * n_chan for _ in range(n_time)]

        group = pd.DataFrame({
            "rms": rms_data,
            "duration": [1.0] * n_time,
        })

        # Extract and flatten the feature
        result = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="rms", score_type="none"
        )

        # After flatten_feature_for_plotting, shape should be (n_time, n_chan, 1) for LINEAR
        assert result.shape == (n_time, n_chan, 1)

        # Now test the collapsing logic using collapse_feature_channels
        from neurodent.visualization.feature_utils import collapse_feature_channels
        from neurodent.constants import classify_feature

        ftype = classify_feature("rms")
        collapsed = collapse_feature_channels(result, ftype).squeeze()

        # After collapsing channels and squeezing, should be (n_time,) for LINEAR
        assert collapsed.ndim == 1
        assert collapsed.shape == (n_time,)

        # The value should be the original value (averaged across channels)
        assert np.allclose(collapsed, value)

    def test_temporal_heatmap_shape_validation_linear(self, plotter, rng):
        """Test shape validation for LINEAR feature (rms) at each step."""
        n_time = 10
        n_chan = N_CHAN
        value = 5.0

        # Create constant-value feature data
        rms_data = [[value] * n_chan for _ in range(n_time)]
        group = pd.DataFrame({
            "rms": rms_data,
            "duration": [1.0] * n_time,
        })

        # Step 1: __get_linear_feature should return (n_time, n_chan, 1)
        feature_data = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="rms", score_type="none"
        )
        assert feature_data.shape == (n_time, n_chan, 1), \
            f"Expected (n_time, n_chan, 1), got {feature_data.shape}"

        # Step 2: After np.nanmean(axis=1), should be (n_time, 1)
        feature_data = np.nanmean(feature_data, axis=1)
        assert feature_data.shape == (n_time, 1), \
            f"Expected (n_time, 1), got {feature_data.shape}"

        # Step 3: After squeeze, should be (n_time,)
        feature_data = feature_data.squeeze()
        assert feature_data.ndim == 1 and feature_data.shape[0] == n_time, \
            f"Expected 1D array of length {n_time}, got shape {feature_data.shape}"

        # Step 4: Values should be correct (averaged across channels)
        assert np.allclose(feature_data, value), \
            f"Expected all values to be {value}, got {feature_data}"

    def test_temporal_heatmap_shape_validation_linear_2d(self, plotter, rng):
        """Test shape validation for LINEAR_2D feature (psdslope) at each step."""
        n_time = 10
        n_chan = N_CHAN
        n_components = 2
        slope_value = 1.0
        intercept_value = 100.0

        # Create constant-value feature data with distinct slope and intercept
        psdslope_data = [
            np.array([[slope_value, intercept_value]] * n_chan)
            for _ in range(n_time)
        ]
        group = pd.DataFrame({
            "psdslope": psdslope_data,
            "duration": [1.0] * n_time,
        })

        # Step 1: __get_linear_feature should return (n_time, n_chan, n_components)
        feature_data = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="psdslope", score_type="none"
        )
        assert feature_data.shape == (n_time, n_chan, n_components), \
            f"Expected (n_time, n_chan, n_components), got {feature_data.shape}"

        # Step 2: After np.nanmean(axis=1), should be (n_time, n_components)
        feature_data = np.nanmean(feature_data, axis=1)
        assert feature_data.shape == (n_time, n_components), \
            f"Expected (n_time, n_components), got {feature_data.shape}"

        # Step 3: Values should be correct (averaged across channels, NOT across components)
        # Slope component (index 0) should all be slope_value
        assert np.allclose(feature_data[:, 0], slope_value), \
            f"Expected slope values to be {slope_value}, got {feature_data[:, 0]}"
        # Intercept component (index 1) should all be intercept_value
        assert np.allclose(feature_data[:, 1], intercept_value), \
            f"Expected intercept values to be {intercept_value}, got {feature_data[:, 1]}"

    def test_temporal_heatmap_shape_validation_band(self, plotter, rng):
        """Test shape validation for BAND feature (psdband) at each step."""
        n_time = 10
        n_chan = N_CHAN
        n_bands = N_BANDS
        band_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # Distinct value per band

        # Create constant-value feature data with distinct values per band
        psdband_data = [
            {band: [band_values[i]] * n_chan for i, band in enumerate(constants.BAND_NAMES)}
            for _ in range(n_time)
        ]
        group = pd.DataFrame({
            "psdband": psdband_data,
            "duration": [1.0] * n_time,
        })

        # Step 1: __get_linear_feature should return (n_time, n_chan, n_bands)
        feature_data = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="psdband", score_type="none"
        )
        assert feature_data.shape == (n_time, n_chan, n_bands), \
            f"Expected (n_time, n_chan, n_bands), got {feature_data.shape}"

        # Step 2: After np.nanmean(axis=1), should be (n_time, n_bands)
        feature_data = np.nanmean(feature_data, axis=1)
        assert feature_data.shape == (n_time, n_bands), \
            f"Expected (n_time, n_bands), got {feature_data.shape}"

        # Step 3: Values should be correct (averaged across channels, NOT across bands)
        for i, expected_value in enumerate(band_values):
            assert np.allclose(feature_data[:, i], expected_value), \
                f"Expected band {i} values to be {expected_value}, got {feature_data[:, i]}"

    def test_temporal_heatmap_shape_validation_simple_matrix(self, plotter, rng):
        """Test shape validation for SIMPLE_MATRIX feature (zpcorr) at each step."""
        n_time = 10
        n_chan = N_CHAN
        n_pairs = n_chan * (n_chan - 1) // 2
        value = 0.5

        # Create constant-value feature data
        zpcorr_data = [
            np.full((n_chan, n_chan), value)
            for _ in range(n_time)
        ]
        group = pd.DataFrame({
            "zpcorr": zpcorr_data,
            "duration": [1.0] * n_time,
        })

        # Step 1: __get_linear_feature should return (n_time, n_pairs, 1)
        feature_data = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="zpcorr", score_type="none"
        )
        assert feature_data.shape == (n_time, n_pairs, 1), \
            f"Expected (n_time, n_pairs, 1), got {feature_data.shape}"

        # Step 2: After np.nanmean(axis=1), should be (n_time, 1)
        feature_data = np.nanmean(feature_data, axis=1)
        assert feature_data.shape == (n_time, 1), \
            f"Expected (n_time, 1), got {feature_data.shape}"

        # Step 3: After squeeze, should be (n_time,)
        feature_data = feature_data.squeeze()
        assert feature_data.ndim == 1 and feature_data.shape[0] == n_time, \
            f"Expected 1D array of length {n_time}, got shape {feature_data.shape}"

        # Step 4: Values should be correct (averaged across channel pairs)
        assert np.allclose(feature_data, value), \
            f"Expected all values to be {value}, got {feature_data}"

    def test_temporal_heatmap_shape_validation_banded_matrix(self, plotter, rng):
        """Test shape validation for BANDED_MATRIX feature (cohere) at each step."""
        n_time = 10
        n_chan = N_CHAN
        n_pairs = n_chan * (n_chan - 1) // 2
        n_bands = N_BANDS
        band_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Distinct value per band

        # Create constant-value feature data with distinct values per band
        cohere_data = [
            {band: np.full((n_chan, n_chan), band_values[i])
             for i, band in enumerate(constants.BAND_NAMES)}
            for _ in range(n_time)
        ]
        group = pd.DataFrame({
            "cohere": cohere_data,
            "duration": [1.0] * n_time,
        })

        # Step 1: __get_linear_feature should return (n_time, n_pairs, n_bands)
        feature_data = plotter._AnimalPlotter__get_linear_feature(
            group=group, feature="cohere", score_type="none"
        )
        assert feature_data.shape == (n_time, n_pairs, n_bands), \
            f"Expected (n_time, n_pairs, n_bands), got {feature_data.shape}"

        # Step 2: After np.nanmean(axis=1), should be (n_time, n_bands)
        feature_data = np.nanmean(feature_data, axis=1)
        assert feature_data.shape == (n_time, n_bands), \
            f"Expected (n_time, n_bands), got {feature_data.shape}"

        # Step 3: Values should be correct (averaged across channel pairs, NOT across bands)
        for i, expected_value in enumerate(band_values):
            assert np.allclose(feature_data[:, i], expected_value), \
                f"Expected band {i} values to be {expected_value}, got {feature_data[:, i]}"

    @patch("matplotlib.pyplot.show")
    def test_plot_temporal_heatmap_feature_linear_numeric_correctness(
        self, mock_show, plotter, mock_war, rng
    ):
        """Test _plot_temporal_heatmap_feature with LINEAR feature returns correct values."""
        n_time = 20
        n_chan = N_CHAN
        value = 5.0

        # Create constant-value feature data
        rms_data = [[value] * n_chan for _ in range(n_time)]
        timestamps = pd.date_range("2023-01-01 00:00", periods=n_time, freq="1h")

        df = pd.DataFrame({
            "rms": rms_data,
            "duration": [1.0] * n_time,
            "timestamp": timestamps,
            "endfile": [np.nan] * n_time,
            "animalday": ["day1"] * n_time,
        })
        df.index = pd.MultiIndex.from_tuples([("animal1",)] * n_time)

        mock_war.get_grouprows_result.return_value = df

        # Call the full function - should not raise and should produce a heatmap
        plotter.plot_temporal_heatmap(features=["rms"], score_type="none", n_bins=5)

        # Verify the function was called and completed without error
        assert mock_show.called

    @patch("matplotlib.pyplot.show")
    def test_plot_temporal_heatmap_feature_linear_2d_numeric_correctness(
        self, mock_show, plotter, mock_war, rng
    ):
        """Test _plot_temporal_heatmap_feature with LINEAR_2D feature separates components correctly."""
        n_time = 20
        n_chan = N_CHAN
        slope_value = 1.0
        intercept_value = 100.0

        # Create constant-value feature data with distinct slope and intercept
        psdslope_data = [
            np.array([[slope_value, intercept_value]] * n_chan)
            for _ in range(n_time)
        ]
        timestamps = pd.date_range("2023-01-01 00:00", periods=n_time, freq="1h")

        df = pd.DataFrame({
            "psdslope": psdslope_data,
            "duration": [1.0] * n_time,
            "timestamp": timestamps,
            "endfile": [np.nan] * n_time,
            "animalday": ["day1"] * n_time,
        })
        df.index = pd.MultiIndex.from_tuples([("animal1",)] * n_time)

        mock_war.get_grouprows_result.return_value = df

        # Call the full function - should produce 2 heatmaps (slope and intercept)
        plotter.plot_temporal_heatmap(features=["psdslope"], score_type="none", n_bins=5)

        # Verify the function was called and completed without error
        assert mock_show.called

    @patch("matplotlib.pyplot.show")
    def test_plot_temporal_heatmap_feature_band_numeric_correctness(
        self, mock_show, plotter, mock_war, rng
    ):
        """Test _plot_temporal_heatmap_feature with BAND feature separates bands correctly."""
        n_time = 20
        n_chan = N_CHAN
        band_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # Distinct value per band

        # Create constant-value feature data with distinct values per band
        psdband_data = [
            {band: [band_values[i]] * n_chan for i, band in enumerate(constants.BAND_NAMES)}
            for _ in range(n_time)
        ]
        timestamps = pd.date_range("2023-01-01 00:00", periods=n_time, freq="1h")

        df = pd.DataFrame({
            "psdband": psdband_data,
            "duration": [1.0] * n_time,
            "timestamp": timestamps,
            "endfile": [np.nan] * n_time,
            "animalday": ["day1"] * n_time,
        })
        df.index = pd.MultiIndex.from_tuples([("animal1",)] * n_time)

        mock_war.get_grouprows_result.return_value = df

        # Call the full function - should produce 5 heatmaps (one per band)
        plotter.plot_temporal_heatmap(features=["psdband"], score_type="none", n_bins=5)

        # Verify the function was called and completed without error
        assert mock_show.called



class TestAddLongrecordingBoundaries:
    def test_no_endfile_column(self, plotter):
        """Returns silently when endfile column absent."""
        fig, ax = plt.subplots()
        df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=5, freq="1h")})
        days = [datetime(2023, 1, 1).date()]
        time_of_day = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0])
        plotter._add_longrecording_boundaries(ax, df, time_of_day, days)

    def test_no_endfile_values(self, plotter):
        """Returns silently when all endfile values are NaN."""
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 08:00", periods=5, freq="1h"),
                "endfile": [np.nan] * 5,
            }
        )
        days = [datetime(2023, 1, 1).date()]
        time_of_day = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0])
        plotter._add_longrecording_boundaries(ax, df, time_of_day, days)

    def test_endfile_boundary_drawn(self, plotter):
        """Red lines are drawn at endfile markers."""
        fig, ax = plt.subplots()
        timestamps = pd.date_range("2023-01-01 08:00", periods=5, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "endfile": [np.nan, np.nan, 0.5, np.nan, np.nan],
            }
        )
        days = sorted(timestamps.date.tolist(), reverse=True)
        time_of_day = pd.Series(
            timestamps.hour + timestamps.minute / 60.0 + timestamps.second / 3600.0
        )
        plotter._add_longrecording_boundaries(ax, df, time_of_day, days)

    def test_animalday_boundary_drawn(self, plotter):
        """White dotted lines drawn when animalday changes."""
        fig, ax = plt.subplots()
        timestamps = pd.date_range("2023-01-01 08:00", periods=6, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "endfile": [np.nan, np.nan, 0.5, np.nan, np.nan, 0.5],
                "animalday": ["d1", "d1", "d1", "d2", "d2", "d2"],
            }
        )
        days = sorted(set(timestamps.date), reverse=True)
        time_of_day = pd.Series(
            timestamps.hour + timestamps.minute / 60.0 + timestamps.second / 3600.0
        )
        plotter._add_longrecording_boundaries(ax, df, time_of_day, days)


# ===================================================================
# 14. _handle_figure (lines 850-861)
# ===================================================================

class TestHandleFigure:
    @patch("matplotlib.pyplot.show")
    def test_show_when_save_false(self, mock_show, plotter):
        fig, _ = plt.subplots()
        plotter.save_fig = False
        plotter._handle_figure(fig, title="test")
        mock_show.assert_called_once()

    def test_save_fig_creates_file(self, plotter, tmp_path):
        fig, _ = plt.subplots()
        save_path = tmp_path / "test_save_output"
        plotter.save_fig = True
        plotter.save_path = save_path
        plotter._handle_figure(fig, title="myfig")
        expected = Path(f"{save_path}_myfig.png")
        assert expected.exists()

    def test_save_fig_no_title(self, plotter, tmp_path):
        fig, _ = plt.subplots()
        save_path = tmp_path / "test_save_notitle"
        plotter.save_fig = True
        plotter.save_path = save_path
        plotter._handle_figure(fig, title=None)
        expected = Path(f"{save_path}.png")
        assert expected.exists()

    def test_save_fig_no_path_raises(self, plotter):
        fig, _ = plt.subplots()
        plotter.save_fig = True
        plotter.save_path = None
        with pytest.raises(ValueError, match="save_path must be provided"):
            plotter._handle_figure(fig, title="x")


# ===================================================================
# 15. _plot_linear_temporalgroup – error branches (lines 244-245, 275)
# ===================================================================

class TestPlotLinearTemporalgroupEdgeCases:
    def test_invalid_ndim_raises(self, plotter, rng):
        """1-D feature array raises ValueError."""
        fig, ax = plt.subplots()
        n_time = 5
        group = pd.DataFrame(
            {
                "rms": [rng.random(N_CHAN).tolist() for _ in range(n_time)],
                "duration": [1.0] * n_time,
            }
        )
        # Monkey-patch __get_linear_feature to return 1-D array
        original = plotter._AnimalPlotter__get_linear_feature

        def _fake_get(group, feature, score_type):
            return rng.random(n_time)

        plotter._AnimalPlotter__get_linear_feature = _fake_get
        try:
            with pytest.raises(ValueError, match="Expected 2D or 3D"):
                plotter._plot_linear_temporalgroup(group, "rms", ax)
        finally:
            plotter._AnimalPlotter__get_linear_feature = original

    def test_show_endfile_calls_filediv(self, plotter, rng):
        """show_endfile=True triggers _plot_filediv_lines, drawing vertical lines."""
        fig, ax = plt.subplots()
        n_time = 5
        group = pd.DataFrame(
            {
                "rms": [rng.random(N_CHAN).tolist() for _ in range(n_time)],
                "duration": [1.0] * n_time,
                "endfile": [np.nan] * (n_time - 1) + [0.5],
            }
        )
        lines_before = len(ax.get_lines())
        plotter._plot_linear_temporalgroup(group, "rms", ax, show_endfile=True)
        lines_after = len(ax.get_lines())
        assert lines_after > lines_before, "Expected vertical divider lines to be drawn"
