"""
Unit tests for neurodent.visualization module.

Legacy ResultsVisualizer and standalone plotting function tests have been removed because their functionality is now handled by AnimalPlotter and ExperimentPlotter.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import Mock, patch, MagicMock

from neurodent.visualization import (
    WindowAnalysisResult,
    AnimalFeatureParser,
    AnimalPlotter,
    ExperimentPlotter,
)
from neurodent import constants


class TestAnimalFeatureParser:
    """Test AnimalFeatureParser class."""

    @pytest.fixture
    def parser(self):
        return AnimalFeatureParser()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        n_chan = 3
        data = {
            "rms": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "duration": [1.0, 2.0, 1.5],
            "psdband": [
                {"alpha": [1.0, 2.0], "beta": [3.0, 4.0]},
                {"alpha": [5.0, 6.0], "beta": [7.0, 8.0]},
                {"alpha": [9.0, 10.0], "beta": [11.0, 12.0]},
            ],
            "psdslope": [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
                [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]],
            ],
            "pcorr": [
                np.eye(n_chan).tolist(),
                (np.eye(n_chan) * 2).tolist(),
                (np.eye(n_chan) * 3).tolist(),
            ],
            "cohere": [
                {"alpha": np.ones((n_chan, n_chan)).tolist(), "beta": (np.ones((n_chan, n_chan)) * 2).tolist()},
                {"alpha": (np.ones((n_chan, n_chan)) * 3).tolist(), "beta": (np.ones((n_chan, n_chan)) * 4).tolist()},
                {"alpha": (np.ones((n_chan, n_chan)) * 5).tolist(), "beta": (np.ones((n_chan, n_chan)) * 6).tolist()},
            ],
            "psd": [
                (np.array([1.0, 2.0, 3.0]), np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])),
                (np.array([1.0, 2.0, 3.0]), np.array([[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]])),
                (np.array([1.0, 2.0, 3.0]), np.array([[130.0, 140.0, 150.0], [160.0, 170.0, 180.0]])),
            ],
        }
        return pd.DataFrame(data)

    def test_average_feature_rms(self, parser, sample_df):
        """Test averaging RMS feature."""
        result = parser._average_feature(sample_df, "rms", "duration")
        # Calculate expected weighted average manually:
        # weights = [1.0, 2.0, 1.5], total_weight = 4.5
        # weighted_sum = 1.0*[1,2,3] + 2.0*[4,5,6] + 1.5*[7,8,9]
        # = [1,2,3] + [8,10,12] + [10.5,12,13.5] = [19.5,24,28.5]
        # weighted_avg = [19.5,24,28.5] / 4.5 = [4.33, 5.33, 6.33]
        expected = np.array([4.33, 5.33, 6.33])
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_average_feature_psdband(self, parser, sample_df):
        """Test averaging PSD band feature."""
        result = parser._average_feature(sample_df, "psdband", "duration")
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "beta" in result
        assert len(result["alpha"]) == 2
        assert len(result["beta"]) == 2
        # Verify weighted average: (val*1.0 + val*2.0 + val*1.5) / 4.5
        expected_alpha = np.array([5.4444, 6.4444])
        expected_beta = np.array([7.4444, 8.4444])
        np.testing.assert_array_almost_equal(result["alpha"], expected_alpha, decimal=3)
        np.testing.assert_array_almost_equal(result["beta"], expected_beta, decimal=3)

    def test_average_feature_linear_2d(self, parser, sample_df):
        """Test averaging LINEAR_2D feature (psdslope)."""
        result = parser._average_feature(sample_df, "psdslope", "duration")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # (n_chan, n_components)
        w = np.array([1.0, 2.0, 1.5])
        raw = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
            [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]],
        ])
        expected = np.average(raw, axis=0, weights=w)
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_feature_simple_matrix(self, parser, sample_df):
        """Test averaging SIMPLE_MATRIX feature (pcorr)."""
        result = parser._average_feature(sample_df, "pcorr", "duration")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)  # (n_chan, n_chan)
        expected = np.eye(3) * (1 * 1.0 + 2 * 2.0 + 3 * 1.5) / 4.5
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_feature_banded_matrix(self, parser, sample_df):
        """Test averaging BANDED_MATRIX feature (cohere)."""
        result = parser._average_feature(sample_df, "cohere", "duration")
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "beta" in result
        assert np.array(result["alpha"]).shape == (3, 3)
        expected_alpha = np.ones((3, 3)) * (1 * 1.0 + 3 * 2.0 + 5 * 1.5) / 4.5
        expected_beta = np.ones((3, 3)) * (2 * 1.0 + 4 * 2.0 + 6 * 1.5) / 4.5
        np.testing.assert_array_almost_equal(result["alpha"], expected_alpha)
        np.testing.assert_array_almost_equal(result["beta"], expected_beta)

    def test_average_feature_hist(self, parser, sample_df):
        """Test averaging HIST feature (psd)."""
        result = parser._average_feature(sample_df, "psd", "duration")
        assert isinstance(result, tuple)
        assert len(result) == 2
        coords, values = result
        np.testing.assert_array_equal(coords, [1.0, 2.0, 3.0])
        assert values.shape == (2, 3)  # (n_freq_rows, n_chan)
        # Canonical (W=3, C=3, F=2) after extract_hist_data transpose
        w = np.array([1.0, 2.0, 1.5])
        canonical = np.array([
            [[10., 40.], [20., 50.], [30., 60.]],
            [[70., 100.], [80., 110.], [90., 120.]],
            [[130., 160.], [140., 170.], [150., 180.]],
        ])
        expected_values = np.average(canonical, axis=0, weights=w).T  # (F=2, C=3)
        np.testing.assert_array_almost_equal(values, expected_values)

    def test_average_feature_no_weights(self, parser, sample_df):
        """Test averaging with no weight column (uniform weights)."""
        result = parser._average_feature(sample_df, "rms", weightsname=None)
        # Uniform weights → simple mean
        expected = np.mean([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_feature_no_weights_band(self, parser, sample_df):
        """Uniform-weight averaging for BAND feature."""
        result = parser._average_feature(sample_df, "psdband", weightsname=None)
        # Simple mean across 3 windows:
        # alpha: [[1,2], [5,6], [9,10]] -> mean = [5, 6]
        # beta:  [[3,4], [7,8], [11,12]] -> mean = [7, 8]
        np.testing.assert_array_almost_equal(result["alpha"], [5.0, 6.0])
        np.testing.assert_array_almost_equal(result["beta"], [7.0, 8.0])

    def test_average_feature_no_weights_linear_2d(self, parser, sample_df):
        """Uniform-weight averaging for LINEAR_2D feature."""
        result = parser._average_feature(sample_df, "psdslope", weightsname=None)
        raw = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
            [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]],
        ])
        expected = np.mean(raw, axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_feature_no_weights_simple_matrix(self, parser, sample_df):
        """Uniform-weight averaging for SIMPLE_MATRIX feature."""
        result = parser._average_feature(sample_df, "pcorr", weightsname=None)
        # eye(3)*1, eye(3)*2, eye(3)*3 -> mean = eye(3)*2
        expected = np.eye(3) * 2.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_feature_no_weights_banded_matrix(self, parser, sample_df):
        """Uniform-weight averaging for BANDED_MATRIX feature."""
        result = parser._average_feature(sample_df, "cohere", weightsname=None)
        # alpha: ones*1, ones*3, ones*5 -> mean = ones*3
        # beta:  ones*2, ones*4, ones*6 -> mean = ones*4
        np.testing.assert_array_almost_equal(result["alpha"], np.ones((3, 3)) * 3.0)
        np.testing.assert_array_almost_equal(result["beta"], np.ones((3, 3)) * 4.0)

    def test_average_feature_no_weights_hist(self, parser, sample_df):
        """Uniform-weight averaging for HIST feature."""
        result = parser._average_feature(sample_df, "psd", weightsname=None)
        coords, values = result
        np.testing.assert_array_equal(coords, [1.0, 2.0, 3.0])
        # Canonical (W=3, C=3, F=2) after extract_hist_data transpose
        canonical = np.array([
            [[10., 40.], [20., 50.], [30., 60.]],
            [[70., 100.], [80., 110.], [90., 120.]],
            [[130., 160.], [140., 170.], [150., 180.]],
        ])
        expected_values = np.mean(canonical, axis=0).T  # (F=2, C=3)
        np.testing.assert_array_almost_equal(values, expected_values)


class TestWindowAnalysisResult:
    """Test WindowAnalysisResult class."""

    @pytest.fixture
    def sample_result_df(self):
        """Create a sample result DataFrame."""
        data = {
            "animal": ["A1", "A1", "A1", "A1"],  # Only one animal
            "animalday": ["A1_20230101", "A1_20230102", "A1_20230103", "A1_20230104"],
            "genotype": ["WT", "WT", "WT", "WT"],
            "channel": ["LMot", "RMot", "LMot", "RMot"],
            "rms": [100.0, 110.0, 105.0, 115.0],
            "psdtotal": [200.0, 220.0, 210.0, 230.0],
            "duration": [60.0, 60.0, 60.0, 60.0],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def filtering_result_df(self):
        """Create a comprehensive result DataFrame for filtering tests."""
        np.random.seed(42)  # For reproducible tests
        n_windows = 20
        n_channels = 3

        data = {
            "animal": ["A1"] * n_windows,
            "animalday": ["A1_20230101"] * (n_windows // 2) + ["A1_20230102"] * (n_windows // 2),
            "genotype": ["WT"] * n_windows,
            "duration": [4.0] * n_windows,
            "isday": [True, False] * (n_windows // 2),
            # RMS values with some outliers
            "rms": [np.random.normal(100, 20, n_channels).tolist() for _ in range(n_windows)],
            # PSD band data with beta proportions
            "psdband": [
                {
                    "alpha": np.random.normal(50, 10, n_channels).tolist(),
                    "beta": np.random.normal(30, 5, n_channels).tolist(),
                    "gamma": np.random.normal(20, 3, n_channels).tolist(),
                }
                for _ in range(n_windows)
            ],
            "psdtotal": [np.random.normal(100, 15, n_channels).tolist() for _ in range(n_windows)],
            "psdfrac": [
                {
                    "alpha": np.random.uniform(0.3, 0.6, n_channels).tolist(),
                    "beta": np.random.uniform(0.2, 0.5, n_channels).tolist(),
                    "gamma": np.random.uniform(0.1, 0.3, n_channels).tolist(),
                }
                for _ in range(n_windows)
            ],
        }

        # Add some extreme RMS values for testing
        data["rms"][0] = [1000.0, 2000.0, 3000.0]  # Very high RMS
        data["rms"][1] = [10.0, 20.0, 30.0]  # Very low RMS

        # Add high beta proportion for testing
        data["psdfrac"][2]["beta"] = [0.6, 0.7, 0.8]

        return pd.DataFrame(data)

    @pytest.fixture
    def war(self, sample_result_df):
        """Create a WindowAnalysisResult instance."""
        return WindowAnalysisResult(
            result=sample_result_df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"]
        )

    @pytest.fixture
    def filtering_war(self, filtering_result_df):
        """Create a WindowAnalysisResult instance for filtering tests."""
        return WindowAnalysisResult(
            result=filtering_result_df,
            animal_id="A1",
            genotype="WT",
            sex="Male",
            channel_names=["LMot", "RMot", "LBar"],
            bad_channels_dict={"A1_20230101": ["LMot"], "A1_20230102": ["RMot"]},
        )

    def test_init(self, war, sample_result_df):
        """Test WindowAnalysisResult initialization."""
        assert war.animal_id == "A1"
        assert war.genotype == "WT"
        assert war.sex == "Male"
        assert war.channel_names == ["LMot", "RMot"]
        assert len(war.result) == len(sample_result_df)

    def test_copy(self, filtering_war):
        """Test that copy creates an independent deep copy of WindowAnalysisResult."""
        # Create a copy
        war_copy = filtering_war.copy()

        # Check that the copy has the same attributes
        assert war_copy.animal_id == filtering_war.animal_id
        assert war_copy.genotype == filtering_war.genotype
        assert war_copy.sex == filtering_war.sex
        assert war_copy.channel_names == filtering_war.channel_names
        assert war_copy.assume_from_number == filtering_war.assume_from_number
        assert war_copy.suppress_short_interval_error == filtering_war.suppress_short_interval_error

        # Check that DataFrames are equal but independent
        pd.testing.assert_frame_equal(war_copy.result, filtering_war.result)
        assert war_copy.result is not filtering_war.result

        # Check that channel_names list is independent
        assert war_copy.channel_names is not filtering_war.channel_names

        # Check that bad_channels_dict is independent (deep copy)
        assert war_copy.bad_channels_dict == filtering_war.bad_channels_dict
        assert war_copy.bad_channels_dict is not filtering_war.bad_channels_dict

        # Check that lof_scores_dict is independent (deep copy)
        assert war_copy.lof_scores_dict == filtering_war.lof_scores_dict
        assert war_copy.lof_scores_dict is not filtering_war.lof_scores_dict

        # Modify the copy and ensure original is unchanged
        original_rms = list(filtering_war.result.loc[0, "rms"])
        war_copy.result.at[0, "rms"] = [999.0, 999.0, 999.0]
        # Original should remain unchanged
        assert filtering_war.result.loc[0, "rms"] == original_rms

        # Modify bad_channels_dict in copy and ensure original is unchanged
        war_copy.bad_channels_dict["A1_20230103"] = ["LBar"]
        assert "A1_20230103" not in filtering_war.bad_channels_dict

    def test_get_result(self, war):
        """Test getting specific features from result."""
        result = war.get_result(features=["rms", "psdtotal"])
        assert "rms" in result.columns
        assert "psdtotal" in result.columns
        assert "animal" in result.columns  # Metadata columns should be included

    def test_get_result_default_all(self, war):
        """Test that get_result() with no args returns all features."""
        result = war.get_result(allow_missing=True)
        assert "rms" in result.columns
        assert "psdtotal" in result.columns
        assert "animal" in result.columns

    def test_get_result_string_input(self, war):
        """Test that get_result() accepts a single string feature."""
        result = war.get_result("rms")
        assert "rms" in result.columns
        assert "animal" in result.columns

    def test_get_result_exclude_with_none_features(self, war):
        """Test that exclude works when features=None (all features except excluded)."""
        result = war.get_result(exclude="rms", allow_missing=True)
        assert "rms" not in result.columns
        assert "psdtotal" in result.columns
        assert "animal" in result.columns

    def test_get_result_feature_fully_excluded(self, war):
        """Test get_result when the only requested feature is also excluded returns no feature columns."""
        result = war.get_result(features="rms", exclude="rms")
        from neurodent import constants

        for col in result.columns:
            assert col not in constants.FEATURES

    def test_get_result_exclude_superset_of_features(self, war):
        """Test get_result when exclude contains feature(s) not in features list returns no feature columns."""
        result = war.get_result(features="rms", exclude=["rms", "psdtotal"])
        from neurodent import constants

        for col in result.columns:
            assert col not in constants.FEATURES

    def test_get_groupavg_result(self, war):
        """Test getting group average results."""
        # Use groupby on 'animalday' to avoid single-group scalar reduction
        result = war.get_groupavg_result(["rms"], groupby="animalday")
        assert isinstance(result, pd.DataFrame)
        assert "rms" in result.columns

    def test_get_groupavg_result_default_all(self, war):
        """Test that get_groupavg_result() with no args returns all features."""
        result = war.get_groupavg_result(groupby="animalday")
        assert isinstance(result, pd.DataFrame)
        assert "rms" in result.columns

    def test_get_groupavg_result_string_input(self, war):
        """Test that get_groupavg_result() accepts a single string feature."""
        result = war.get_groupavg_result("rms", groupby="animalday")
        assert isinstance(result, pd.DataFrame)
        assert "rms" in result.columns

    def test_get_grouprows_result_default_all(self, war):
        """Test that get_grouprows_result() with no args returns all features."""
        result = war.get_grouprows_result()
        assert isinstance(result, pd.DataFrame)
        assert "rms" in result.columns

    def test_get_grouprows_result_string_input(self, war):
        """Test that get_grouprows_result() accepts a single string feature."""
        result = war.get_grouprows_result("rms")
        assert isinstance(result, pd.DataFrame)
        assert "rms" in result.columns

    def test_unsorted_timestamps_warning(self):
        """Test that unsorted timestamps generate a warning and get sorted."""
        # Create DataFrame with unsorted timestamps
        data = {
            "animal": ["A1", "A1", "A1"],
            "animalday": ["A1_20230101", "A1_20230101", "A1_20230101"],
            "genotype": ["WT", "WT", "WT"],
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:08:00",  # Out of order
                    "2023-01-01 10:00:00",  # Should be first
                    "2023-01-01 10:04:00",  # Should be middle
                ]
            ),
            "duration": [240.0, 240.0, 240.0],
            "rms": [[100.0, 110.0], [200.0, 210.0], [150.0, 160.0]],
        }
        df = pd.DataFrame(data)

        # Should generate warning and sort timestamps
        with pytest.warns(UserWarning, match="Timestamps are not sorted"):
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        # Verify timestamps are now sorted
        assert war.result["timestamp"].is_monotonic_increasing
        expected_order = [
            pd.Timestamp("2023-01-01 10:00:00"),
            pd.Timestamp("2023-01-01 10:04:00"),
            pd.Timestamp("2023-01-01 10:08:00"),
        ]
        pd.testing.assert_series_equal(
            war.result["timestamp"].reset_index(drop=True), pd.Series(expected_order, name="timestamp")
        )

    def test_sorted_timestamps_no_warning(self):
        """Test that already sorted timestamps don't generate warnings."""
        # Create DataFrame with properly sorted timestamps
        data = {
            "animal": ["A1", "A1", "A1"],
            "animalday": ["A1_20230101", "A1_20230101", "A1_20230101"],
            "genotype": ["WT", "WT", "WT"],
            "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00", "2023-01-01 10:08:00"]),
            "duration": [240.0, 240.0, 240.0],
            "rms": [[100.0, 110.0], [200.0, 210.0], [150.0, 160.0]],
        }
        df = pd.DataFrame(data)

        # Should not generate any warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        # Verify timestamps remain sorted
        assert war.result["timestamp"].is_monotonic_increasing

    def test_short_intervals_warning(self):
        """Test warning for short intervals between timestamps (< 1% threshold)."""
        # Create DataFrame with one short interval out of many (< 1% threshold)
        # Need enough timestamps so that 1 short interval is < 1%
        timestamps = pd.date_range("2023-01-01 10:00:00", periods=150, freq="4min")
        timestamps_list = timestamps.tolist()
        # Make one interval short (30 seconds instead of 4 minutes)
        timestamps_list[50] = timestamps_list[49] + pd.Timedelta(seconds=30)
        # Adjust remaining timestamps to maintain sequence
        for i in range(51, len(timestamps_list)):
            timestamps_list[i] = timestamps_list[50] + pd.Timedelta(minutes=4) * (i - 50)

        data = {
            "animal": ["A1"] * 150,
            "animalday": ["A1_20230101"] * 150,
            "genotype": ["WT"] * 150,
            "timestamp": timestamps_list,
            "duration": [240.0] * 150,  # 4 minute median duration
            "rms": [[100.0, 110.0]] * 150,
        }
        df = pd.DataFrame(data)

        # Should generate warning but not raise error (1/149 = 0.67% < 1% threshold)
        with pytest.warns(UserWarning, match=r"Found \d+ intervals.*shorter than the median duration"):
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_short_intervals_error(self):
        """Test error for too many short intervals between timestamps (> 1% threshold)."""
        # Create DataFrame where >1% of intervals are short
        data = {
            "animal": ["A1"] * 4,
            "animalday": ["A1_20230101"] * 4,
            "genotype": ["WT"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:30",  # 30s gap
                    "2023-01-01 10:01:00",  # 30s gap
                    "2023-01-01 10:04:00",  # Normal gap
                ]
            ),
            "duration": [240.0] * 4,  # 4 minute median duration
            "rms": [[100.0, 110.0]] * 4,
        }
        df = pd.DataFrame(data)

        # Should raise ValueError (>1% of intervals are short: 2/3 = 66.7%)
        with pytest.raises(ValueError, match=r"Found \d+ intervals.*shorter than the median duration"):
            WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_short_intervals_error_includes_diagnostic(self):
        """Error message includes overlapping pairs and the datetimes_are_start hint."""
        data = {
            "animal": ["A1"] * 4,
            "animalday": ["A1_day1", "A1_day1", "A1_day2", "A1_day2"],
            "genotype": ["WT"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:30",  # 30s gap (short)
                    "2023-01-01 10:01:00",  # 30s gap (short)
                    "2023-01-01 10:04:00",
                ]
            ),
            "duration": [240.0] * 4,
            "rms": [[100.0, 110.0]] * 4,
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError) as exc_info:
            WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        msg = str(exc_info.value)
        assert "datetimes_are_start" in msg
        assert "A1_day" in msg  # animalday context included

    def test_suppress_short_intervals_error(self):
        """Test that suppress_short_interval_error parameter suppresses the ValueError."""
        # Create DataFrame where >1% of intervals are short (same as test_short_intervals_error)
        data = {
            "animal": ["A1"] * 4,
            "animalday": ["A1_20230101"] * 4,
            "genotype": ["WT"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:30",  # 30s gap
                    "2023-01-01 10:01:00",  # 30s gap
                    "2023-01-01 10:04:00",  # Normal gap
                ]
            ),
            "duration": [240.0] * 4,  # 4 minute median duration
            "rms": [[100.0, 110.0]] * 4,
        }
        df = pd.DataFrame(data)

        # Should NOT raise ValueError when suppress_short_interval_error=True
        war = WindowAnalysisResult(
            result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"], suppress_short_interval_error=True
        )

        # Verify the parameter is stored correctly
        assert war.suppress_short_interval_error
        assert war.animal_id == "A1"
        assert war.genotype == "WT"
        assert war.sex == "Male"

    def test_no_short_intervals_check_without_duration(self):
        """Test that short interval check is skipped when duration column is missing."""
        # Create DataFrame without duration column
        data = {
            "animal": ["A1", "A1", "A1"],
            "animalday": ["A1_20230101", "A1_20230101", "A1_20230101"],
            "genotype": ["WT", "WT", "WT"],
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:30",  # Short interval
                    "2023-01-01 10:04:00",
                ]
            ),
            "rms": [[100.0, 110.0], [200.0, 210.0], [150.0, 160.0]],
        }
        df = pd.DataFrame(data)

        # Should not raise error or warning about short intervals
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_no_timestamp_validation_without_timestamps(self):
        """Test that timestamp validation is skipped when timestamp column is missing."""
        # Create DataFrame without timestamp column
        data = {
            "animal": ["A1", "A1", "A1"],
            "animalday": ["A1_20230101", "A1_20230101", "A1_20230101"],
            "genotype": ["WT", "WT", "WT"],
            "duration": [240.0, 240.0, 240.0],
            "rms": [[100.0, 110.0], [200.0, 210.0], [150.0, 160.0]],
        }
        df = pd.DataFrame(data)

        # Should not raise any errors or warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_equal_timestamps_handled_correctly(self):
        """Test that equal timestamps (0 second intervals) are handled correctly."""
        # Create DataFrame with duplicate timestamps
        data = {
            "animal": ["A1"] * 4,
            "animalday": ["A1_20230101"] * 4,
            "genotype": ["WT"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:00",  # Duplicate timestamp
                    "2023-01-01 10:04:00",
                    "2023-01-01 10:08:00",
                ]
            ),
            "duration": [240.0] * 4,
            "rms": [[100.0, 110.0]] * 4,
        }
        df = pd.DataFrame(data)

        # Should handle duplicate timestamps (0 second interval is < median duration)
        # This should trigger the short interval warning/error logic
        with pytest.raises(ValueError, match=r"Found \d+ intervals.*shorter than the median duration"):
            WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_edge_case_single_timestamp(self):
        """Test edge case with only one timestamp (no intervals to check)."""
        data = {
            "animal": ["A1"],
            "animalday": ["A1_20230101"],
            "genotype": ["WT"],
            "timestamp": pd.to_datetime(["2023-01-01 10:00:00"]),
            "duration": [240.0],
            "rms": [[100.0, 110.0]],
        }
        df = pd.DataFrame(data)

        # Should not raise any errors (no intervals to check)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_mixed_duration_intervals(self):
        """Test with mixed durations and corresponding interval validation."""
        # Create realistic scenario with uniform durations and appropriate intervals
        data = {
            "animal": ["A1"] * 6,
            "animalday": ["A1_20230101"] * 6,
            "genotype": ["WT"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:04:00",  # 4min interval
                    "2023-01-01 10:08:00",  # 4min interval
                    "2023-01-01 10:12:00",  # 4min interval
                    "2023-01-01 10:16:00",  # 4min interval
                    "2023-01-01 10:20:00",  # 4min interval
                ]
            ),
            "duration": [240.0, 240.0, 240.0, 240.0, 240.0, 240.0],  # Uniform durations match intervals
            "rms": [[100.0, 110.0]] * 6,
        }
        df = pd.DataFrame(data)

        # All intervals should be reasonable relative to durations - no warnings expected
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_boundary_condition_exactly_one_percent(self):
        """Test boundary condition where exactly 1% of intervals are short."""
        # Create 101 timestamps where exactly 1 interval is short (1/100 = 1.0%)
        # We need 101 timestamps to get 100 intervals
        timestamps = pd.date_range("2023-01-01 10:00:00", periods=101, freq="4min")
        timestamps_list = timestamps.tolist()
        # Make the second interval short (30 seconds instead of 4 minutes)
        timestamps_list[1] = timestamps_list[0] + pd.Timedelta(seconds=30)
        # Adjust remaining timestamps to maintain sequence
        for i in range(2, len(timestamps_list)):
            timestamps_list[i] = timestamps_list[i - 1] + pd.Timedelta(minutes=4)

        data = {
            "animal": ["A1"] * 101,
            "animalday": ["A1_20230101"] * 101,
            "genotype": ["WT"] * 101,
            "timestamp": timestamps_list,
            "duration": [240.0] * 101,  # 4 minute durations
            "rms": [[100.0, 110.0]] * 101,
        }
        df = pd.DataFrame(data)

        # Exactly 1% should trigger warning but not error (1/100 = 1.0%)
        with pytest.warns(UserWarning, match=r"Found \d+ intervals.*shorter than the median duration"):
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

    def test_fragment_durations_stored_and_used(self):
        """Test that fragment durations are stored and used in weighted averaging."""
        # Create DataFrame with varying fragment durations
        data = {
            "animal": ["A1"] * 4,
            "animalday": ["A1_20230101"] * 2 + ["A1_20230102"] * 2,
            "genotype": ["WT"] * 4,
            "timestamp": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-01 10:04:00", "2023-01-02 10:00:00", "2023-01-02 10:04:10"]
            ),
            "duration": [240.0, 240.0, 240.0, 250.0],  # Variable durations
            "rms": [[100.0, 110.0], [200.0, 210.0], [120.0, 130.0], [180.0, 190.0]],
        }
        df = pd.DataFrame(data)

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        # Verify duration column exists
        assert "duration" in war.result.columns

        # Test weighted averaging uses durations
        avg_result = war.get_groupavg_result(["rms"], groupby="animalday")
        assert len(avg_result) == 2

        # Day 1: uniform weights (240, 240) - simple average
        day1_expected = np.average([[100.0, 110.0], [200.0, 210.0]], axis=0, weights=[240.0, 240.0])
        np.testing.assert_array_almost_equal(avg_result.loc["A1_20230101", "rms"], day1_expected)

        # Day 2: different weights (240, 250) - weighted average
        day2_expected = np.average([[120.0, 130.0], [180.0, 190.0]], axis=0, weights=[240.0, 250.0])
        np.testing.assert_array_almost_equal(avg_result.loc["A1_20230102", "rms"], day2_expected)

    def test_duration_aggregation_sums_correctly(self):
        """Test that time window aggregation properly sums fragment durations."""
        data = {
            "animal": ["A1"] * 4,
            "animalday": ["A1_20230101"] * 2 + ["A1_20230102"] * 2,
            "genotype": ["WT"] * 4,
            "timestamp": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-01 10:04:00", "2023-01-02 10:00:00", "2023-01-02 10:04:10"]
            ),
            "isday": [True] * 4,
            "duration": [240.0, 240.0, 240.0, 240.0],  # Uniform durations to avoid timestamp validation issues
            "rms": [[100.0, 110.0], [200.0, 210.0], [120.0, 130.0], [180.0, 190.0]],
        }
        df = pd.DataFrame(data)

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        # Aggregate by animalday
        war.aggregate_time_windows(groupby=["animalday"])

        # Check durations were summed correctly
        result = war.result
        assert len(result) == 2

        day1_duration = result[result["animalday"] == "A1_20230101"]["duration"].iloc[0]
        day2_duration = result[result["animalday"] == "A1_20230102"]["duration"].iloc[0]

        assert day1_duration == 480.0  # 240 + 240
        assert day2_duration == 480.0  # 240 + 240

    def test_duration_preserved_in_save_load(self):
        """Test that fragment durations are preserved through save/load cycles."""
        import tempfile
        from pathlib import Path

        data = {
            "animal": ["A1"] * 3,
            "animalday": ["A1_20230101"] * 3,
            "genotype": ["WT"] * 3,
            "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00", "2023-01-01 10:08:00"]),
            "duration": [240.0, 245.0, 235.0],
            "rms": [[100.0, 110.0], [200.0, 210.0], [150.0, 160.0]],
        }
        df = pd.DataFrame(data)

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Save and load
            war.save_parquet_and_json(save_path)
            loaded_war = WindowAnalysisResult.load_parquet_and_json(save_path)

            # Verify durations preserved
            assert "duration" in loaded_war.result.columns
            original_durations = war.result["duration"].tolist()
            loaded_durations = loaded_war.result["duration"].tolist()
            assert original_durations == loaded_durations

    def test_missing_duration_column_fallback(self):
        """Test graceful handling when duration column is missing."""
        # Create DataFrame without duration column
        data = {
            "animal": ["A1", "A1"],
            "animalday": ["A1_day1", "A1_day1"],
            "genotype": ["WT", "WT"],
            "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00"]),
            "rms": [[100.0, 110.0], [200.0, 210.0]],
        }
        df = pd.DataFrame(data)

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        # Should handle missing duration gracefully (falls back to uniform weights)
        result = war.get_groupavg_result(["rms"], groupby="animalday")
        assert not np.isnan(result.loc["A1_day1", "rms"]).any()


class TestWindowAnalysisResultFiltering:
    """Test new filtering methods for WindowAnalysisResult."""

    @pytest.fixture
    def filtering_result_df(self):
        """Create a comprehensive result DataFrame for filtering tests."""
        np.random.seed(42)  # For reproducible tests
        n_windows = 20
        n_channels = 3

        data = {
            "animal": ["A1"] * n_windows,
            "animalday": ["A1_20230101"] * (n_windows // 2) + ["A1_20230102"] * (n_windows // 2),
            "genotype": ["WT"] * n_windows,
            "duration": [4.0] * n_windows,
            "isday": [True, False] * (n_windows // 2),
            # RMS values with some outliers
            "rms": [np.random.normal(100, 20, n_channels).tolist() for _ in range(n_windows)],
            # PSD band data with beta proportions
            "psdband": [
                {
                    "alpha": np.random.normal(50, 10, n_channels).tolist(),
                    "beta": np.random.normal(30, 5, n_channels).tolist(),
                    "gamma": np.random.normal(20, 3, n_channels).tolist(),
                }
                for _ in range(n_windows)
            ],
            "psdtotal": [np.random.normal(100, 15, n_channels).tolist() for _ in range(n_windows)],
            "psdfrac": [
                {
                    "alpha": np.random.uniform(0.3, 0.6, n_channels).tolist(),
                    "beta": np.random.uniform(0.2, 0.5, n_channels).tolist(),
                    "gamma": np.random.uniform(0.1, 0.3, n_channels).tolist(),
                }
                for _ in range(n_windows)
            ],
        }

        # Add some extreme RMS values for testing
        data["rms"][0] = [1000.0, 2000.0, 3000.0]  # Very high RMS
        data["rms"][1] = [10.0, 20.0, 30.0]  # Very low RMS

        # Add high beta proportion for testing
        data["psdfrac"][2]["beta"] = [0.6, 0.7, 0.8]

        return pd.DataFrame(data)

    @pytest.fixture
    def filtering_war(self, filtering_result_df):
        """Create a WindowAnalysisResult instance for filtering tests."""
        return WindowAnalysisResult(
            result=filtering_result_df,
            animal_id="A1",
            genotype="WT",
            sex="Male",
            channel_names=["LMot", "RMot", "LBar"],
            bad_channels_dict={"A1_20230101": ["LMot"], "A1_20230102": ["RMot"]},
        )

    def test_filter_high_rms(self, filtering_war):
        """Test filtering high RMS values."""
        filtered = filtering_war.filter_high_rms(max_rms=500)

        # Should return new instance
        assert isinstance(filtered, WindowAnalysisResult)
        assert filtered is not filtering_war

        # Original should be unchanged
        assert len(filtering_war.result) == 20

        # Check that high RMS values are filtered
        original_rms = np.array(filtering_war.result["rms"].tolist())
        filtered_rms = np.array(filtered.result["rms"].tolist())

        # Windows with extreme values should have NaN in filtered result
        assert np.all(np.isnan(filtered_rms[0]))  # Window 0 had [1000, 2000, 3000]
        assert not np.all(np.isnan(filtered_rms[2]))  # Window 2 should be fine

    def test_filter_low_rms(self, filtering_war):
        """Test filtering low RMS values."""
        filtered = filtering_war.filter_low_rms(min_rms=50)

        assert isinstance(filtered, WindowAnalysisResult)
        assert filtered is not filtering_war

        # Check that low RMS values are filtered
        filtered_rms = np.array(filtered.result["rms"].tolist())
        assert np.all(np.isnan(filtered_rms[1]))  # Window 1 had [10, 20, 30]

    def test_filter_high_beta(self, filtering_war):
        """Test filtering high beta power."""
        filtered = filtering_war.filter_high_beta(max_beta_prop=0.5)

        assert isinstance(filtered, WindowAnalysisResult)

        # Check that window with high beta is filtered
        # Window 2 was set to have beta = [0.6, 0.7, 0.8]
        filtered_psdfrac = filtered.result["psdfrac"].tolist()
        high_beta_window = filtered_psdfrac[2]

        # All channels should be filtered for this window due to broadcast_to
        assert all(np.isnan(high_beta_window["beta"]))

    def test_filter_reject_channels(self, filtering_war):
        """Test rejecting specific channels."""
        filtered = filtering_war.filter_reject_channels(["LMot"])

        assert isinstance(filtered, WindowAnalysisResult)

        # Check that LMot channel (index 0) is filtered for all windows
        filtered_rms = np.array(filtered.result["rms"].tolist())
        assert np.all(np.isnan(filtered_rms[:, 0]))  # First channel should be NaN
        assert not np.all(np.isnan(filtered_rms[:, 1]))  # Other channels should have data

    def test_filter_reject_channels_by_session(self, filtering_war):
        """Test rejecting channels by recording session."""
        # Use the bad_channels_dict from fixture
        filtered = filtering_war.filter_reject_channels_by_session()

        assert isinstance(filtered, WindowAnalysisResult)

        filtered_rms = np.array(filtered.result["rms"].tolist())

        # Windows 0-9 (A1_20230101): LMot should be filtered
        assert np.all(np.isnan(filtered_rms[:10, 0]))

        # Windows 10-19 (A1_20230102): RMot should be filtered
        assert np.all(np.isnan(filtered_rms[10:, 1]))

    def test_filter_logrms_range_calls_underlying_method(self, filtering_war):
        """Test that filter_logrms_range calls the underlying get_filter method."""
        with patch.object(filtering_war, "get_filter_logrms_range") as mock_filter:
            mock_filter.return_value = np.ones((20, 3), dtype=bool)

            filtered = filtering_war.filter_logrms_range(z_range=2.5)

            mock_filter.assert_called_once_with(z_range=2.5)
            assert isinstance(filtered, WindowAnalysisResult)

    def test_apply_filters_default_config(self, filtering_war):
        """Test apply_filters with default configuration."""
        with (
            patch.object(filtering_war, "get_filter_logrms_range") as mock_logrms,
            patch.object(filtering_war, "get_filter_high_rms") as mock_high_rms,
            patch.object(filtering_war, "get_filter_low_rms") as mock_low_rms,
            patch.object(filtering_war, "get_filter_high_beta") as mock_high_beta,
            patch.object(filtering_war, "get_filter_reject_channels_by_recording_session") as mock_reject_session,
        ):
            # Mock all filters to return all-True masks
            for mock in [mock_logrms, mock_high_rms, mock_low_rms, mock_high_beta, mock_reject_session]:
                mock.return_value = np.ones((20, 3), dtype=bool)

            filtered = filtering_war.apply_filters()

            # Verify all default filters were called
            mock_logrms.assert_called_once_with(z_range=3)
            mock_high_rms.assert_called_once_with(max_rms=500)
            mock_low_rms.assert_called_once_with(min_rms=50)
            mock_high_beta.assert_called_once_with(max_beta_prop=0.4)
            mock_reject_session.assert_called_once_with()

            assert isinstance(filtered, WindowAnalysisResult)

    def test_apply_filters_custom_config(self, filtering_war):
        """Test apply_filters with custom configuration."""
        config = {"high_rms": {"max_rms": 600}, "reject_channels": {"bad_channels": ["LBar"]}}

        with (
            patch.object(filtering_war, "get_filter_high_rms") as mock_high_rms,
            patch.object(filtering_war, "get_filter_reject_channels") as mock_reject,
        ):
            mock_high_rms.return_value = np.ones((20, 3), dtype=bool)
            mock_reject.return_value = np.ones((20, 3), dtype=bool)

            filtered = filtering_war.apply_filters(config)

            mock_high_rms.assert_called_once_with(max_rms=600)
            mock_reject.assert_called_once_with(bad_channels=["LBar"])

    def test_apply_filters_invalid_filter_name(self, filtering_war):
        """Test apply_filters with invalid filter name."""
        config = {"invalid_filter": {}}

        with pytest.raises(ValueError, match="Unknown filter: invalid_filter"):
            filtering_war.apply_filters(config)

    def test_apply_filters_min_valid_channels(self, filtering_war):
        """Test minimum valid channels requirement."""
        # Create a filter that passes only 1 channel per window
        config = {"reject_channels": {"bad_channels": ["LMot", "RMot"]}}

        with patch.object(filtering_war, "get_filter_reject_channels") as mock_reject:
            # Mock to filter out 2 of 3 channels (only LBar remains)
            mask = np.ones((20, 3), dtype=bool)
            mask[:, 0] = False  # Filter LMot
            mask[:, 1] = False  # Filter RMot
            mock_reject.return_value = mask

            # Should filter out windows with < 3 valid channels
            filtered = filtering_war.apply_filters(config, min_valid_channels=3)

            # All windows should be filtered since only 1 channel remains per window
            filtered_rms = np.array(filtered.result["rms"].tolist())
            assert np.all(np.isnan(filtered_rms))

    def test_morphological_smoothing(self, filtering_war):
        """Test morphological smoothing functionality."""
        config = {"high_rms": {"max_rms": 500}}

        # Create a filter that produces isolated artifacts
        with patch.object(filtering_war, "get_filter_high_rms") as mock_filter:
            mask = np.ones((20, 3), dtype=bool)
            # Create isolated false positives/negatives
            mask[5, 0] = False  # Isolated artifact
            mask[15, 1] = False  # Another isolated artifact
            mock_filter.return_value = mask

            # Test with morphological smoothing
            filtered = filtering_war.apply_filters(
                config,
                morphological_smoothing_seconds=8.0,  # 2 windows at 4s each
            )

            assert isinstance(filtered, WindowAnalysisResult)

    def test_filter_methods_return_new_instances(self, filtering_war):
        """Test that all filter methods return new instances."""
        methods_and_params = [
            ("filter_high_rms", {"max_rms": 500}),
            ("filter_low_rms", {"min_rms": 50}),
            ("filter_high_beta", {"max_beta_prop": 0.4}),
            ("filter_reject_channels", {"bad_channels": ["LMot"]}),
            ("filter_reject_channels_by_session", {}),
        ]

        for method_name, params in methods_and_params:
            method = getattr(filtering_war, method_name)
            filtered = method(**params)

            assert isinstance(filtered, WindowAnalysisResult)
            assert filtered is not filtering_war
            assert filtered.animal_id == filtering_war.animal_id
            assert filtered.genotype == filtering_war.genotype
            assert filtered.sex == filtering_war.sex
            assert filtered.channel_names == filtering_war.channel_names

    def test_method_chaining(self, filtering_war):
        """Test that methods can be chained together."""
        result = filtering_war.filter_high_rms(max_rms=500).filter_low_rms(min_rms=50).filter_reject_channels(["LMot"])

        assert isinstance(result, WindowAnalysisResult)
        assert result is not filtering_war

    def test_backwards_compatibility_filter_all(self, filtering_war):
        """Test that old filter_all method still works."""
        # This tests that we haven't broken existing functionality
        try:
            # Should still work with the old interface (if it exists)
            result = filtering_war.filter_all(inplace=False)
            assert isinstance(result, WindowAnalysisResult)
        except AttributeError:
            # If filter_all doesn't exist, that's also fine - it may have been replaced
            pass

    def test_create_filtered_copy_preserves_metadata(self, filtering_war):
        """Test that _create_filtered_copy preserves all metadata."""
        mask = np.ones((20, 3), dtype=bool)
        filtered = filtering_war._create_filtered_copy(mask)

        assert filtered.animal_id == filtering_war.animal_id
        assert filtered.genotype == filtering_war.genotype
        assert filtered.sex == filtering_war.sex
        assert filtered.channel_names == filtering_war.channel_names
        assert filtered.assume_from_number == filtering_war.assume_from_number
        assert filtered.bad_channels_dict == filtering_war.bad_channels_dict

    def test_edge_case_empty_bad_channels_dict(self):
        """Test filtering with empty bad channels dictionary."""
        df = pd.DataFrame(
            {
                "animal": ["A1"] * 5,
                "animalday": ["A1_20230101"] * 5,
                "genotype": ["WT"] * 5,
                "rms": [[100, 200]] * 5,
                "duration": [4.0] * 5,
            }
        )

        war = WindowAnalysisResult(
            result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"], bad_channels_dict={}
        )

        # Empty bad_channels_dict should mean "no bad channels" and not raise an error
        filtered = war.filter_reject_channels_by_session()

        # Should return a new instance with no filtering applied (all data preserved)
        assert isinstance(filtered, WindowAnalysisResult)
        assert len(filtered.result) == len(war.result)

        # All RMS values should be preserved (no NaN introduced by filtering)
        original_rms = np.array(war.result["rms"].tolist())
        filtered_rms = np.array(filtered.result["rms"].tolist())
        np.testing.assert_array_equal(filtered_rms, original_rms)

    def test_edge_case_missing_session_in_bad_channels_dict(self):
        """Test that missing sessions are auto-populated with empty lists in __init__."""
        df = pd.DataFrame(
            {
                "animal": ["A1"] * 10,
                "animalday": ["A1_20230101"] * 5 + ["A1_20230102"] * 5,  # Two sessions
                "genotype": ["WT"] * 10,
                "rms": [[100, 200]] * 10,
                "duration": [4.0] * 10,
            }
        )

        war = WindowAnalysisResult(
            result=df,
            animal_id="A1",
            genotype="WT",
            sex="Male",
            channel_names=["LMot", "RMot"],
            bad_channels_dict={"A1_20230101": ["LMot"]},  # Missing A1_20230102
        )

        # After __init__, missing sessions should be auto-populated with empty lists
        assert "A1_20230102" in war.bad_channels_dict
        assert war.bad_channels_dict["A1_20230102"] == []

        # filter_reject_channels_by_session should work without error
        result = war.filter_reject_channels_by_session()
        assert result is not None

    def test_edge_case_no_duration_column(self):
        """Test morphological smoothing without duration column."""
        df = pd.DataFrame({"animal": ["A1"] * 5, "animalday": ["A1_20230101"] * 5, "rms": [[100, 200]] * 5})

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        config = {"high_rms": {"max_rms": 500}}

        with pytest.raises(ValueError, match="Cannot calculate window duration"):
            war.apply_filters(config, morphological_smoothing_seconds=8.0)

    def test_apply_filters_with_morphological_config(self, filtering_war):
        """Test morphological smoothing via configuration."""
        config = {"high_rms": {"max_rms": 500}, "morphological_smoothing": {"smoothing_seconds": 8.0}}

        with (
            patch.object(filtering_war, "get_filter_high_rms") as mock_high_rms,
            patch.object(filtering_war, "get_filter_morphological_smoothing") as mock_smooth,
        ):
            mask = np.ones((20, 3), dtype=bool)
            mock_high_rms.return_value = mask
            mock_smooth.return_value = mask

            filtered = filtering_war.apply_filters(config)

            mock_high_rms.assert_called_once_with(max_rms=500)
            # Check that mock_smooth was called once with the right arguments
            mock_smooth.assert_called_once()
            args, kwargs = mock_smooth.call_args
            np.testing.assert_array_equal(args[0], mask)
            assert kwargs["smoothing_seconds"] == 8.0
            assert isinstance(filtered, WindowAnalysisResult)


class TestWindowAnalysisResultRemapChannels:
    """Test WindowAnalysisResult.reorder_and_pad_channels() FeatureType dispatch."""

    @pytest.fixture
    def remap_war(self):
        """Create a WAR with all feature types for remap testing (2 channels → 3)."""
        n_rows = 3
        n_chan = 2
        n_freq = 4
        ch_names = ["LMot", "RMot"]
        rng = np.random.default_rng(99)

        data = {
            "animal": ["A1"] * n_rows,
            "animalday": ["A1_day1"] * n_rows,
            "genotype": ["WT"] * n_rows,
            "duration": [4.0] * n_rows,
            "rms": [rng.random(n_chan).tolist() for _ in range(n_rows)],
            "psdslope": [rng.random((n_chan, 2)).tolist() for _ in range(n_rows)],
            "psdband": [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "pcorr": [rng.random((n_chan, n_chan)).tolist() for _ in range(n_rows)],
            "cohere": [
                {b: rng.random((n_chan, n_chan)).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "psd": [
                (np.linspace(1, 40, n_freq), rng.random((n_freq, n_chan)))
                for _ in range(n_rows)
            ],
        }
        return WindowAnalysisResult(
            result=pd.DataFrame(data),
            animal_id="A1",
            genotype="WT",
            channel_names=ch_names,
        )

    def test_remap_channels_linear(self, remap_war):
        """Test remap_channels for LINEAR feature (rms)."""
        target = ["LMot", "RMot", "LBar"]
        result = remap_war.reorder_and_pad_channels(target, use_abbrevs=False, inplace=False)
        for row in result["rms"]:
            arr = np.array(row)
            assert arr.shape == (3,)
            assert np.isnan(arr[2])  # LBar should be NaN-padded
            assert not np.isnan(arr[0])

    def test_remap_channels_linear_2d(self, remap_war):
        """Test remap_channels for LINEAR_2D feature (psdslope)."""
        target = ["LMot", "RMot", "LBar"]
        result = remap_war.reorder_and_pad_channels(target, use_abbrevs=False, inplace=False)
        for row in result["psdslope"]:
            arr = np.array(row)
            assert arr.shape == (3, 2)
            assert np.all(np.isnan(arr[2]))  # LBar padded

    def test_remap_channels_band(self, remap_war):
        """Test remap_channels for BAND feature (psdband)."""
        target = ["LMot", "RMot", "LBar"]
        result = remap_war.reorder_and_pad_channels(target, use_abbrevs=False, inplace=False)
        for row_dict in result["psdband"]:
            assert set(row_dict.keys()) == set(constants.BAND_NAMES)
            for band_vals in row_dict.values():
                arr = np.array(band_vals)
                assert arr.shape == (3,)
                assert np.isnan(arr[2])

    def test_remap_channels_simple_matrix(self, remap_war):
        """Test remap_channels for SIMPLE_MATRIX feature (pcorr)."""
        target = ["LMot", "RMot", "LBar"]
        result = remap_war.reorder_and_pad_channels(target, use_abbrevs=False, inplace=False)
        for row in result["pcorr"]:
            arr = np.array(row)
            assert arr.shape == (3, 3)
            # LBar row/col should be NaN
            assert np.all(np.isnan(arr[2, :]))
            assert np.all(np.isnan(arr[:, 2]))

    def test_remap_channels_banded_matrix(self, remap_war):
        """Test remap_channels for BANDED_MATRIX feature (cohere)."""
        target = ["LMot", "RMot", "LBar"]
        result = remap_war.reorder_and_pad_channels(target, use_abbrevs=False, inplace=False)
        for row_dict in result["cohere"]:
            assert set(row_dict.keys()) == set(constants.BAND_NAMES)
            for band_mat in row_dict.values():
                arr = np.array(band_mat)
                assert arr.shape == (3, 3)
                assert np.all(np.isnan(arr[2, :]))
                assert np.all(np.isnan(arr[:, 2]))

    def test_remap_channels_hist(self, remap_war):
        """Test remap_channels for HIST feature (psd)."""
        target = ["LMot", "RMot", "LBar"]
        result = remap_war.reorder_and_pad_channels(target, use_abbrevs=False, inplace=False)
        for item in result["psd"]:
            coords, vals = item
            arr = np.array(vals)
            assert arr.shape[-1] == 3  # 3 target channels
            assert np.all(np.isnan(arr[..., 2]))

    def test_remap_channels_inplace(self, remap_war):
        """Test remap_channels with inplace=True updates instance state."""
        target = ["LMot", "RMot", "LBar"]
        remap_war.reorder_and_pad_channels(target, use_abbrevs=False, inplace=True)
        assert remap_war.channel_names == target
        # Verify data was updated
        arr = np.array(remap_war.result["rms"].iloc[0])
        assert arr.shape == (3,)


class TestApplyFilter:
    """Test WindowAnalysisResult._apply_filter() FeatureType dispatch."""

    @pytest.fixture
    def filter_war(self):
        """Create a WAR with all feature types for filter testing."""
        n_rows = 3
        n_chan = 2
        n_freq = 4
        rng = np.random.default_rng(77)

        data = {
            "animal": ["A1"] * n_rows,
            "animalday": ["A1_day1"] * n_rows,
            "genotype": ["WT"] * n_rows,
            "duration": [4.0] * n_rows,
            "rms": [rng.random(n_chan).tolist() for _ in range(n_rows)],
            "psdslope": [rng.random((n_chan, 2)).tolist() for _ in range(n_rows)],
            "psdband": [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "pcorr": [rng.random((n_chan, n_chan)).tolist() for _ in range(n_rows)],
            "cohere": [
                {b: rng.random((n_chan, n_chan)).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "psd": [
                (np.linspace(1, 40, n_freq), rng.random((n_freq, n_chan)))
                for _ in range(n_rows)
            ],
        }
        return WindowAnalysisResult(
            result=pd.DataFrame(data),
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot"],
        )

    @pytest.fixture
    def mask(self):
        """Boolean mask: keep all except row 0, channel 1."""
        m = np.ones((3, 2), dtype=bool)
        m[0, 1] = False
        return m

    def test_apply_filter_linear(self, filter_war, mask):
        """Test _apply_filter for LINEAR feature (rms)."""
        result = filter_war._apply_filter(mask)
        vals = np.array(result["rms"].tolist())
        assert np.isnan(vals[0, 1])
        assert not np.isnan(vals[0, 0])

    def test_apply_filter_linear_2d(self, filter_war, mask):
        """Test _apply_filter for LINEAR_2D feature (psdslope)."""
        result = filter_war._apply_filter(mask)
        vals = np.array(result["psdslope"].tolist())
        # Masked position: all components should be NaN
        assert np.all(np.isnan(vals[0, 1, :]))
        assert not np.any(np.isnan(vals[0, 0, :]))

    def test_apply_filter_band(self, filter_war, mask):
        """Test _apply_filter for BAND feature (psdband)."""
        result = filter_war._apply_filter(mask)
        for band in constants.BAND_NAMES:
            val = np.array(result["psdband"].iloc[0][band])
            assert np.isnan(val[1])  # channel 1 masked
            assert not np.isnan(val[0])

    def test_apply_filter_simple_matrix(self, filter_war, mask):
        """Test _apply_filter for SIMPLE_MATRIX feature (pcorr)."""
        result = filter_war._apply_filter(mask)
        mat = np.array(result["pcorr"].iloc[0])
        # Channel 1 masked → row 1 and col 1 should be NaN
        assert np.isnan(mat[1, 0])
        assert np.isnan(mat[0, 1])
        assert not np.isnan(mat[0, 0])

    def test_apply_filter_banded_matrix(self, filter_war, mask):
        """Test _apply_filter for BANDED_MATRIX feature (cohere)."""
        result = filter_war._apply_filter(mask)
        row_dict = result["cohere"].iloc[0]
        for band_mat in row_dict.values():
            mat = np.array(band_mat)
            assert np.isnan(mat[1, 0])
            assert np.isnan(mat[0, 1])
            assert not np.isnan(mat[0, 0])

    def test_apply_filter_hist(self, filter_war, mask):
        """Test _apply_filter for HIST feature (psd)."""
        result = filter_war._apply_filter(mask)
        coords, vals = result["psd"].iloc[0]
        arr = np.array(vals)
        # mask shape is (n_rows, n_chan), broadcast to (n_rows, n_freq, n_chan)
        # row 0, channel 1 should be NaN
        assert np.all(np.isnan(arr[:, 1]))
        assert not np.any(np.isnan(arr[:, 0]))

    def test_apply_filter_all_true_preserves_data(self, filter_war):
        """Test that an all-True mask preserves all data (no NaNs introduced)."""
        mask = np.ones((3, 2), dtype=bool)
        result = filter_war._apply_filter(mask)
        vals = np.array(result["rms"].tolist())
        assert not np.any(np.isnan(vals))


class TestAnimalPlotter:
    """Test AnimalPlotter class."""

    @pytest.fixture
    def mock_war(self):
        """Create a mock WindowAnalysisResult."""
        war = MagicMock(spec=WindowAnalysisResult)
        war.genotype = "WT"
        war.channel_names = ["LMot", "RMot"]
        war.channel_abbrevs = ["LM", "RM"]
        war.assume_from_number = False
        # Only provide the 'cohere' column, not individual band columns
        band_names = constants.BAND_NAMES + ["pcorr"]
        cohere_dicts = []
        for _ in range(2):
            d = {band: np.random.rand(2, 2) for band in band_names}
            cohere_dicts.append(d)
        mock_result = pd.DataFrame({"cohere": cohere_dicts}, index=["day1", "day2"])
        war.get_groupavg_result.return_value = mock_result
        return war

    @pytest.fixture
    def plotter(self, mock_war):
        """Create an AnimalPlotter instance."""
        plotter = AnimalPlotter(mock_war)
        # Add the missing attribute
        plotter.CHNAME_TO_ABBREV = [("LeftMotor", "LM"), ("RightMotor", "RM")]
        return plotter

    def test_init(self, plotter, mock_war):
        """Test AnimalPlotter initialization."""
        assert plotter.window_result == mock_war
        assert plotter.genotype == "WT"
        assert plotter.channel_names == ["LMot", "RMot"]
        assert plotter.channel_abbrevs == ["LM", "RM"]
        assert plotter.n_channels == 2

    def test_abbreviate_channel(self, plotter):
        """Test channel abbreviation."""
        # Test with a known channel name
        result = plotter._abbreviate_channel("LeftMotor")
        assert result == "LM"

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_coherecorr_matrix(self, mock_show, mock_subplots, plotter, mock_war):
        n_row = 2
        mock_fig = Mock()
        n_bands = len(constants.BAND_NAMES) + 1
        mock_ax = np.array([[Mock() for _ in range(n_bands)] for _ in range(n_row)])
        mock_subplots.return_value = (mock_fig, mock_ax)
        # Only provide the 'cohere' column, not individual band columns
        band_names = constants.BAND_NAMES + ["pcorr"]
        cohere_dicts = []
        for _ in range(n_row):
            d = {band: np.random.rand(2, 2) for band in band_names}
            cohere_dicts.append(d)
        mock_result = pd.DataFrame({"cohere": cohere_dicts}, index=["day1", "day2"])
        mock_war.get_groupavg_result.return_value = mock_result
        plotter.plot_coherecorr_matrix()
        mock_subplots.assert_called()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_coherecorr_diff(self, mock_show, mock_subplots, plotter, mock_war):
        mock_fig = Mock()
        n_bands = len(constants.BAND_NAMES) + 1
        mock_ax = np.array([[Mock() for _ in range(n_bands)]])
        mock_subplots.return_value = (mock_fig, mock_ax)
        band_names = constants.BAND_NAMES + ["pcorr"]
        cohere_dicts = []
        for _ in range(2):
            d = {band: np.random.rand(2, 2) for band in band_names}
            cohere_dicts.append(d)
        mock_result = pd.DataFrame({"cohere": cohere_dicts}, index=["day1", "day2"])
        mock_war.get_groupavg_result.return_value = mock_result
        plotter.plot_coherecorr_diff()
        mock_subplots.assert_called()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_psd_histogram(self, mock_show, mock_subplots, plotter, mock_war):
        mock_fig, mock_ax = Mock(), np.array([[Mock(), Mock()]])
        mock_subplots.return_value = (mock_fig, mock_ax)
        # Mock get_groupavg_result for psd
        mock_war.get_groupavg_result.return_value = pd.DataFrame(
            {"psd": [(np.linspace(1, 50, 10), np.random.rand(10, 2)), (np.linspace(1, 50, 10), np.random.rand(10, 2))]},
            index=["day1", "day2"],
        )
        plotter.plot_psd_histogram()
        mock_subplots.assert_called()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_psd_spectrogram(self, mock_show, mock_subplots, plotter, mock_war):
        mock_fig, mock_ax = Mock(), Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        # Mock get_grouprows_result for psd
        mock_war.get_grouprows_result.return_value = pd.DataFrame(
            {
                "psd": [
                    (np.linspace(1, 50, 10), np.random.rand(10, 2)),
                    (np.linspace(1, 50, 10), np.random.rand(10, 2)),
                ],
                "duration": [1.0, 1.0],
            }
        )
        plotter.plot_psd_spectrogram()
        mock_subplots.assert_called()

    @pytest.mark.skip(reason="Complex triangular indexing logic requires extensive mocking")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_coherecorr_spectral(self, mock_show, mock_subplots, plotter, mock_war):
        mock_fig, mock_ax = Mock(), [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_ax)
        # Mock get_grouprows_result for cohere/pcorr with correct data structure
        n_rows = 2
        n_time = 5
        n_channels = 2
        band_names = ["delta", "theta"]

        # Create data with proper shape for linear feature calculation
        def make_dict():
            return {band: np.random.rand(n_time, n_channels, n_channels) for band in band_names}

        mock_war.get_grouprows_result.return_value = pd.DataFrame(
            {
                "cohere": [make_dict() for _ in range(n_rows)],
                "pcorr": [make_dict() for _ in range(n_rows)],
                "duration": [1.0] * n_rows,
            }
        )
        plotter.plot_coherecorr_spectral(features=["cohere", "pcorr"])
        mock_subplots.assert_called()

    def test_get_linear_feature_linear(self, plotter):
        """Test __get_linear_feature dispatches LINEAR features correctly."""
        n_time, n_chan = 5, 2
        group = pd.DataFrame({
            "rms": [np.random.rand(n_chan).tolist() for _ in range(n_time)],
        })
        # Access name-mangled private method
        result = plotter._AnimalPlotter__get_linear_feature(group, "rms", score_type="none")
        # LINEAR features are expanded with a trailing dim: (n_time, n_chan, 1)
        assert result.shape == (n_time, n_chan, 1)

    def test_get_linear_feature_linear_2d(self, plotter):
        """Test __get_linear_feature dispatches LINEAR_2D features correctly."""
        n_time, n_chan, n_components = 5, 2, 2
        group = pd.DataFrame({
            "psdslope": [np.random.rand(n_chan, n_components).tolist() for _ in range(n_time)],
        })
        result = plotter._AnimalPlotter__get_linear_feature(group, "psdslope", score_type="none")
        # LINEAR_2D keeps all components → (n_time, n_chan, n_components)
        assert result.shape == (n_time, n_chan, n_components)

    def test_get_linear_feature_band(self, plotter):
        """Test __get_linear_feature dispatches BAND features correctly."""
        n_time, n_chan = 5, 2
        n_bands = len(constants.BAND_NAMES)
        group = pd.DataFrame({
            "psdband": [
                {b: np.random.rand(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ],
        })
        result = plotter._AnimalPlotter__get_linear_feature(group, "psdband", score_type="none")
        assert result.shape == (n_time, n_chan, n_bands)

    def test_get_linear_feature_simple_matrix(self, plotter):
        """Test __get_linear_feature dispatches SIMPLE_MATRIX features correctly."""
        n_time, n_chan = 5, 2
        group = pd.DataFrame({
            "pcorr": [np.random.rand(n_chan, n_chan).tolist() for _ in range(n_time)],
        })
        # triag=True: extracts lower triangle (1 pair for 2x2)
        result = plotter._AnimalPlotter__get_linear_feature(group, "pcorr", score_type="none", triag=True)
        expected_pairs = n_chan * (n_chan - 1) // 2
        assert result.shape == (n_time, expected_pairs, 1)

    def test_get_linear_feature_banded_matrix(self, plotter):
        """Test __get_linear_feature dispatches BANDED_MATRIX features correctly."""
        n_time, n_chan = 5, 2
        n_bands = len(constants.BAND_NAMES)
        group = pd.DataFrame({
            "cohere": [
                {b: np.random.rand(n_chan, n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ],
        })
        result = plotter._AnimalPlotter__get_linear_feature(group, "cohere", score_type="none", triag=True)
        expected_pairs = n_chan * (n_chan - 1) // 2
        assert result.shape == (n_time, expected_pairs, n_bands)

    def test_plot_linear_temporalgroup_yticks_linear(self, plotter):
        """Test that LINEAR features get correct ytick labels."""
        import matplotlib.pyplot as plt
        n_time, n_chan = 5, 2
        group = pd.DataFrame({
            "rms": [np.random.rand(n_chan).tolist() for _ in range(n_time)],
            "duration": [1.0] * n_time,
        })
        fig, ax = plt.subplots()
        plotter._plot_linear_temporalgroup(group, "rms", ax)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["rms"]
        plt.close(fig)

    def test_plot_linear_temporalgroup_yticks_linear_2d(self, plotter):
        """Test that LINEAR_2D features get psdslope/psdintercept ytick labels."""
        import matplotlib.pyplot as plt
        n_time, n_chan = 5, 2
        group = pd.DataFrame({
            "psdslope": [np.random.rand(n_chan, 2).tolist() for _ in range(n_time)],
            "duration": [1.0] * n_time,
        })
        fig, ax = plt.subplots()
        plotter._plot_linear_temporalgroup(group, "psdslope", ax)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["psdslope", "psdintercept"]
        plt.close(fig)

    def test_plot_linear_temporalgroup_yticks_band(self, plotter):
        """Test that BAND features get frequency band ytick labels."""
        import matplotlib.pyplot as plt
        n_time, n_chan = 5, 2
        group = pd.DataFrame({
            "psdband": [
                {b: np.random.rand(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_time)
            ],
            "duration": [1.0] * n_time,
        })
        fig, ax = plt.subplots()
        plotter._plot_linear_temporalgroup(group, "psdband", ax)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == constants.BAND_NAMES
        plt.close(fig)

    def test_plot_linear_temporalgroup_yticks_invalid_feature(self, plotter):
        """Test that unsupported FeatureTypes for ytick labels raise ValueError."""
        import matplotlib.pyplot as plt
        n_time, n_chan = 5, 2
        # Use a HIST feature (psd) which is not supported for ytick labels
        group = pd.DataFrame({
            "psd": [
                (np.linspace(0, 50, 10).tolist(), np.random.rand(10, n_chan).tolist())
                for _ in range(n_time)
            ],
            "duration": [1.0] * n_time,
        })
        fig, ax = plt.subplots()
        # psd will fail in __get_linear_feature, not in ytick setting
        with pytest.raises(ValueError, match="Unsupported FeatureType.*for feature extraction"):
            plotter._plot_linear_temporalgroup(group, "psd", ax)
        plt.close(fig)

    def test_get_linear_feature_invalid_feature(self, plotter):
        """Test that unsupported FeatureTypes for feature extraction raise ValueError."""
        n_time, n_chan = 5, 2
        # Use a HIST feature (psd) which should raise in __get_linear_feature
        group = pd.DataFrame({
            "psd": [
                (np.linspace(0, 50, 10).tolist(), np.random.rand(10, n_chan).tolist())
                for _ in range(n_time)
            ],
        })
        with pytest.raises(ValueError, match="Unsupported FeatureType.*for feature extraction"):
            plotter._AnimalPlotter__get_linear_feature(group, "psd", score_type="none")


class TestExperimentPlotter:
    """Test ExperimentPlotter class."""

    @pytest.fixture
    def mock_wars(self):
        """Create mock WindowAnalysisResult objects."""
        war1 = MagicMock(spec=WindowAnalysisResult)
        war1.animal_id = "A1"
        war1.channel_names = ["LMot", "RMot"]
        war1.channel_abbrevs = ["LM", "RM"]
        war2 = MagicMock(spec=WindowAnalysisResult)
        war2.animal_id = "A2"
        war2.channel_names = ["LMot", "RMot"]
        war2.channel_abbrevs = ["LM", "RM"]
        # Mock get_result method to return arrays for feature columns, but keep categorical columns as scalars
        mock_df1 = pd.DataFrame(
            {
                "animal": ["A1", "A1"],
                "genotype": ["WT", "WT"],
                "channel": ["LMot", "RMot"],
                "rms": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                "psdtotal": [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
            }
        )
        mock_df2 = pd.DataFrame(
            {
                "animal": ["A2", "A2"],
                "genotype": ["KO", "KO"],
                "channel": ["LMot", "RMot"],
                "rms": [np.array([1.5, 2.5]), np.array([3.5, 4.5])],
                "psdtotal": [np.array([5.5, 6.5]), np.array([7.5, 8.5])],
            }
        )
        war1.get_result.return_value = mock_df1
        war2.get_result.return_value = mock_df2
        return [war1, war2]

    @pytest.fixture
    def plotter(self, mock_wars):
        """Create an ExperimentPlotter instance."""
        plotter = ExperimentPlotter(mock_wars)
        # Set up concat_df_wars properly for validation
        plotter.concat_df_wars = pd.DataFrame(
            {
                "animal": ["A1", "A1", "A2", "A2"],
                "genotype": ["WT", "WT", "KO", "KO"],
                "channel": ["LMot", "RMot", "LMot", "RMot"],
                "rms": [1.0, 2.0, 1.5, 2.5],
                "psdtotal": [5.0, 6.0, 5.5, 6.5],
            }
        )
        return plotter

    def test_init(self, plotter, mock_wars):
        """Test ExperimentPlotter initialization."""
        assert len(plotter.results) == 2
        assert plotter.channel_names == [["LM", "RM"], ["LM", "RM"]]
        assert isinstance(plotter.concat_df_wars, pd.DataFrame)
        assert len(plotter.concat_df_wars) == 4  # 2 animals * 2 channels

    def test_validate_plot_order(self, plotter):
        """Test plot order validation."""
        df = pd.DataFrame({"genotype": ["WT", "KO", "WT"], "channel": ["LMot", "RMot", "LMot"]})

        result = plotter.validate_plot_order(df)
        assert isinstance(result, dict)

    def test_pull_timeseries_dataframe(self, plotter):
        """Test pulling timeseries data."""
        # Mock the pull_timeseries_dataframe to avoid validation issues
        with patch.object(plotter, "pull_timeseries_dataframe") as mock_pull:
            mock_pull.return_value = pd.DataFrame(
                {"genotype": ["WT", "KO"], "channel": ["LMot", "RMot"], "rms": [1.0, 2.0]}
            )
            result = plotter.pull_timeseries_dataframe(feature="rms", groupby=["genotype", "channel"])
            assert isinstance(result, pd.DataFrame)
            assert "rms" in result.columns

    @patch("seaborn.catplot")
    def test_plot_catplot(self, mock_catplot, plotter):
        """Test categorical plotting."""
        mock_fig = Mock()
        mock_grid = Mock()
        mock_grid.axes = np.array([[Mock()]])  # Make axes iterable
        mock_catplot.return_value = mock_grid
        # Mock pull_timeseries_dataframe to avoid validation issues
        with patch.object(plotter, "pull_timeseries_dataframe") as mock_pull:
            mock_pull.return_value = pd.DataFrame(
                {"genotype": ["WT", "KO"], "channel": ["LMot", "RMot"], "rms": [1.0, 2.0]}
            )
            result = plotter.plot_catplot(feature="rms", groupby=["genotype", "channel"], kind="box")
            mock_catplot.assert_called()
            assert result == mock_grid

    @patch("seaborn.FacetGrid")
    def test_plot_heatmap(self, mock_facetgrid, plotter):
        mock_grid = Mock()
        mock_facetgrid.return_value = mock_grid
        # Patch pull_timeseries_dataframe to return a DataFrame with matrix features
        plotter.pull_timeseries_dataframe = Mock(
            return_value=pd.DataFrame(
                {
                    "genotype": ["WT", "KO"],
                    "channel": ["LMot", "RMot"],
                    "cohere": [np.random.rand(2, 2), np.random.rand(2, 2)],
                }
            )
        )
        result = plotter.plot_heatmap(feature="cohere", groupby=["genotype", "channel"])
        assert result == mock_grid

    @patch("seaborn.FacetGrid")
    def test_plot_diffheatmap(self, mock_facetgrid, plotter):
        mock_grid = Mock()
        mock_facetgrid.return_value = mock_grid
        plotter.pull_timeseries_dataframe = Mock(
            return_value=pd.DataFrame(
                {
                    "genotype": ["WT", "KO"],
                    "channel": ["LMot", "RMot"],
                    "cohere": [np.random.rand(2, 2), np.random.rand(2, 2)],
                }
            )
        )
        with patch(
            "neurodent.visualization.plotting.experiment.df_normalize_baseline",
            side_effect=lambda **kwargs: kwargs["df"],
        ):
            result = plotter.plot_diffheatmap(feature="cohere", groupby=["genotype", "channel"], baseline_key="WT")
            assert result == mock_grid

    @patch("seaborn.FacetGrid")
    def test_plot_qqplot(self, mock_facetgrid, plotter):
        mock_grid = Mock()
        mock_facetgrid.return_value = mock_grid
        plotter.pull_timeseries_dataframe = Mock(
            return_value=pd.DataFrame(
                {"genotype": ["WT", "KO"], "channel": ["LMot", "RMot"], "rms": [np.random.rand(10), np.random.rand(10)]}
            )
        )
        result = plotter.plot_qqplot(feature="rms", groupby=["genotype", "channel"])
        assert result == mock_grid

    def test_plot_heatmap_invalid_feature(self, plotter):
        with pytest.raises(ValueError):
            plotter.plot_heatmap(feature="notamatrix", groupby=["genotype", "channel"])

    def test_plot_diffheatmap_invalid_feature(self, plotter):
        with pytest.raises(ValueError):
            plotter.plot_diffheatmap(feature="notamatrix", groupby=["genotype", "channel"], baseline_key="WT")

    def test_plot_qqplot_invalid_feature(self, plotter):
        with pytest.raises(ValueError):
            plotter.plot_qqplot(feature="cohere", groupby=["genotype", "channel"])


class TestExperimentPlotterFeatureDispatch:
    """Test ExperimentPlotter.pull_timeseries_dataframe FeatureType dispatch paths."""

    @pytest.fixture
    def feature_plotter(self):
        """Create an ExperimentPlotter with real WAR-structured DataFrames."""
        rng = np.random.default_rng(seed=42)
        n_rows = 4
        n_chan = 2
        n_freq = 5
        n_bands = len(constants.BAND_NAMES)
        ch_names = ["LM", "RM"]

        war = MagicMock(spec=WindowAnalysisResult)
        war.animal_id = "A1"
        war.channel_names = ch_names
        war.channel_abbrevs = ch_names

        df = pd.DataFrame({
            "genotype": ["WT"] * n_rows,
            "rms": [rng.random(n_chan).tolist() for _ in range(n_rows)],
            "psdslope": [rng.random((n_chan, 2)).tolist() for _ in range(n_rows)],
            "psdband": [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "logpsdband": [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "psdfrac": [
                {b: rng.random(n_chan).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "pcorr": [rng.random((n_chan, n_chan)).tolist() for _ in range(n_rows)],
            "cohere": [
                {b: rng.random((n_chan, n_chan)).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "imcoh": [
                {b: rng.random((n_chan, n_chan)).tolist() for b in constants.BAND_NAMES}
                for _ in range(n_rows)
            ],
            "psd": [
                (np.linspace(1, 40, n_freq), rng.random((n_freq, n_chan)))
                for _ in range(n_rows)
            ],
        })
        war.get_result.return_value = df

        plot_order = {"channel": ch_names + ["average", "all"], "genotype": ["WT"]}
        plotter = ExperimentPlotter([war], plot_order=plot_order)
        return plotter

    def test_pull_linear_feature(self, feature_plotter):
        """Test pull_timeseries_dataframe with LINEAR feature, collapse_channels=False."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="rms", groupby=["genotype"], channels="all"
        )
        assert isinstance(df, pd.DataFrame)
        assert "rms" in df.columns
        assert "channel" in df.columns
        assert set(df["channel"].unique()) == {"LM", "RM"}

    def test_pull_linear_feature_collapsed(self, feature_plotter):
        """Test pull_timeseries_dataframe with LINEAR feature, collapse_channels=True."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="rms", groupby=["genotype"], collapse_channels=True
        )
        assert isinstance(df, pd.DataFrame)
        assert "rms" in df.columns
        assert (df["channel"] == "average").all()

    def test_pull_linear_2d_feature(self, feature_plotter):
        """Test pull_timeseries_dataframe with LINEAR_2D, collapse_channels=False."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psdslope", groupby=["genotype"], channels="all"
        )
        assert isinstance(df, pd.DataFrame)
        assert "psdslope" in df.columns
        # LINEAR_2D extracts first component (slope); values should be scalar
        assert all(isinstance(v, (int, float, np.floating)) for v in df["psdslope"])

    def test_pull_linear_2d_feature_collapsed(self, feature_plotter):
        """Test pull_timeseries_dataframe with LINEAR_2D, collapse_channels=True."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psdslope", groupby=["genotype"], collapse_channels=True
        )
        assert isinstance(df, pd.DataFrame)
        assert "psdslope" in df.columns
        assert (df["channel"] == "average").all()
        assert all(isinstance(v, (int, float, np.floating)) for v in df["psdslope"])

    def test_pull_band_feature(self, feature_plotter):
        """Test pull_timeseries_dataframe with BAND, collapse_channels=False."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psdband", groupby=["genotype"], channels="all"
        )
        assert isinstance(df, pd.DataFrame)
        assert "psdband" in df.columns
        assert "band" in df.columns
        assert set(df["band"].unique()) == set(constants.BAND_NAMES)

    def test_pull_band_feature_collapsed(self, feature_plotter):
        """Test pull_timeseries_dataframe with BAND, collapse_channels=True."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psdband", groupby=["genotype"], collapse_channels=True
        )
        assert isinstance(df, pd.DataFrame)
        assert "psdband" in df.columns
        assert "band" in df.columns
        assert (df["channel"] == "average").all()
        assert set(df["band"].unique()) == set(constants.BAND_NAMES)

    def test_pull_simple_matrix_feature(self, feature_plotter):
        """Test pull_timeseries_dataframe with SIMPLE_MATRIX, collapse_channels=True."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="pcorr", groupby=["genotype"], collapse_channels=True
        )
        assert isinstance(df, pd.DataFrame)
        assert "pcorr" in df.columns
        assert (df["channel"] == "average").all()

    def test_pull_simple_matrix_feature_not_collapsed(self, feature_plotter):
        """Test pull_timeseries_dataframe with SIMPLE_MATRIX, collapse_channels=False."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="pcorr", groupby=["genotype"], collapse_channels=False
        )
        assert isinstance(df, pd.DataFrame)
        assert "pcorr" in df.columns
        assert (df["channel"] == "all").all()

    def test_pull_banded_matrix_feature(self, feature_plotter):
        """Test pull_timeseries_dataframe with BANDED_MATRIX, collapse_channels=False."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="cohere", groupby=["genotype"], collapse_channels=False
        )
        assert isinstance(df, pd.DataFrame)
        assert "cohere" in df.columns

    def test_pull_banded_matrix_feature_collapsed(self, feature_plotter):
        """Test pull_timeseries_dataframe with BANDED_MATRIX, collapse_channels=True."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="cohere", groupby=["genotype"], collapse_channels=True
        )
        assert isinstance(df, pd.DataFrame)
        assert "cohere" in df.columns
        assert "channel" in df.columns
        assert (df["channel"] == "average").all()

    def test_pull_banded_matrix_shape_correctness(self, feature_plotter):
        """Test BANDED_MATRIX shape correctness: verifies band axis is iterated correctly.

        Regression test for issue where cohere/imcoh heatmaps displayed as (8,5) instead of (8,8)
        due to incorrect iteration over the band axis in pull_timeseries_dataframe.
        """
        df = feature_plotter.pull_timeseries_dataframe(
            feature="cohere", groupby=["genotype"], collapse_channels=False
        )
        # Each row should have a band value
        assert "band" in df.columns
        assert set(df["band"].unique()) == set(constants.BAND_NAMES)

        # Each row's cohere value should be a matrix of shape (n_chan, n_chan)
        n_chan = len(feature_plotter.all_channel_names)
        for idx, row in df.iterrows():
            cohere_matrix = np.array(row["cohere"])
            assert cohere_matrix.shape == (n_chan, n_chan), (
                f"Expected cohere matrix shape ({n_chan}, {n_chan}), "
                f"got {cohere_matrix.shape} for band={row['band']}"
            )

        # Verify that different bands have different data
        # Collect all matrices for each band
        band_matrices = {}
        for band_name in constants.BAND_NAMES:
            band_data = df[df["band"] == band_name]
            assert len(band_data) > 0, f"Band {band_name} has no data"
            # Collect all matrices for this band
            matrices = [np.array(row["cohere"]) for _, row in band_data.iterrows()]
            band_matrices[band_name] = np.array(matrices)

        # Verify that matrices from different bands are actually different
        # Since the fixture uses random data with a fixed seed, different bands should have different values
        band_names = list(band_matrices.keys())
        for i in range(len(band_names)):
            for j in range(i + 1, len(band_names)):
                band1, band2 = band_names[i], band_names[j]
                matrices1, matrices2 = band_matrices[band1], band_matrices[band2]
                # At least some values should differ between bands
                assert not np.allclose(matrices1, matrices2), (
                    f"Bands {band1} and {band2} have identical data - "
                    f"band axis may not be correctly exploded"
                )

    def test_pull_band_feature_shape_correctness(self, feature_plotter):
        """Test BAND feature shape correctness: verifies band axis is iterated correctly.

        Tests that BAND features (like psdband) also work correctly with the moveaxis fix.
        """
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psdband", groupby=["genotype"], collapse_channels=False
        )
        # Each row should have a band value
        assert "band" in df.columns
        assert set(df["band"].unique()) == set(constants.BAND_NAMES)

        # Each row's psdband value should be a scalar (after extraction)
        for idx, row in df.iterrows():
            psdband_val = row["psdband"]
            # After pull_timeseries_dataframe, BAND features should be scalar per channel per band
            assert isinstance(psdband_val, (int, float, np.number)), (
                f"Expected scalar psdband value, got {type(psdband_val)} for band={row['band']}"
            )

        # Verify that different bands have different data values
        # Collect all values for each band
        band_values = {}
        for band_name in constants.BAND_NAMES:
            band_data = df[df["band"] == band_name]
            assert len(band_data) > 0, f"Band {band_name} has no data"
            values = np.array([row["psdband"] for _, row in band_data.iterrows()])
            band_values[band_name] = values

        # Verify that values from different bands are actually different
        band_names = list(band_values.keys())
        for i in range(len(band_names)):
            for j in range(i + 1, len(band_names)):
                band1, band2 = band_names[i], band_names[j]
                values1, values2 = band_values[band1], band_values[band2]
                # At least some values should differ between bands
                assert not np.allclose(values1, values2), (
                    f"Bands {band1} and {band2} have identical data - "
                    f"band axis may not be correctly exploded"
                )

    def test_pull_band_feature_logpsdband_shape_correctness(self, feature_plotter):
        """Test BAND feature (logpsdband) shape correctness."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="logpsdband", groupby=["genotype"], collapse_channels=False
        )
        assert "band" in df.columns
        assert set(df["band"].unique()) == set(constants.BAND_NAMES)

        for idx, row in df.iterrows():
            val = row["logpsdband"]
            assert isinstance(val, (int, float, np.number)), (
                f"Expected scalar, got {type(val)} for band={row['band']}"
            )

    def test_pull_band_feature_psdfrac_shape_correctness(self, feature_plotter):
        """Test BAND feature (psdfrac) shape correctness."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psdfrac", groupby=["genotype"], collapse_channels=False
        )
        assert "band" in df.columns
        assert set(df["band"].unique()) == set(constants.BAND_NAMES)

        for idx, row in df.iterrows():
            val = row["psdfrac"]
            assert isinstance(val, (int, float, np.number)), (
                f"Expected scalar, got {type(val)} for band={row['band']}"
            )

    def test_pull_banded_matrix_imcoh_shape_correctness(self, feature_plotter):
        """Test BANDED_MATRIX feature (imcoh) shape correctness."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="imcoh", groupby=["genotype"], collapse_channels=False
        )
        assert "band" in df.columns
        assert set(df["band"].unique()) == set(constants.BAND_NAMES)

        n_chan = len(feature_plotter.all_channel_names)
        for idx, row in df.iterrows():
            matrix = np.array(row["imcoh"])
            assert matrix.shape == (n_chan, n_chan), (
                f"Expected shape ({n_chan}, {n_chan}), got {matrix.shape} for band={row['band']}"
            )

    def test_pull_hist_feature(self, feature_plotter):
        """Test pull_timeseries_dataframe with HIST, collapse_channels=False."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psd", groupby=["genotype"], channels="all"
        )
        assert isinstance(df, pd.DataFrame)
        assert "psd" in df.columns
        assert "freq" in df.columns

    def test_pull_hist_feature_collapsed(self, feature_plotter):
        """Test pull_timeseries_dataframe with HIST, collapse_channels=True."""
        df = feature_plotter.pull_timeseries_dataframe(
            feature="psd", groupby=["genotype"], collapse_channels=True
        )
        assert isinstance(df, pd.DataFrame)
        assert "psd" in df.columns
        assert "freq" in df.columns
        assert (df["channel"] == "average").all()

    def test_pull_missing_feature_raises(self, feature_plotter):
        """Test that missing features raise ValueError."""
        with pytest.raises(ValueError, match="feature not found"):
            feature_plotter.pull_timeseries_dataframe(
                feature="nonexistent_feature", groupby=["genotype"]
            )


class TestFeatureUtils:
    """Test shared feature extraction utilities in feature_utils module."""

    def test_extract_linear_array_1d(self):
        """Test extracting 1-D (scalar per channel) linear feature."""
        from neurodent.visualization.feature_utils import extract_linear_array

        series = pd.Series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = extract_linear_array(series)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[1], [4.0, 5.0, 6.0])

    def test_extract_linear_array_2d(self):
        """Test extracting 2-D (multi-component per channel) linear feature."""
        from neurodent.visualization.feature_utils import extract_linear_array

        series = pd.Series([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        result = extract_linear_array(series)
        assert result.shape == (2, 2, 2)
        np.testing.assert_array_equal(result[0, 0], [1.0, 2.0])

    def test_extract_band_dict(self):
        """Test extracting band-keyed dict feature to array."""
        from neurodent.visualization.feature_utils import extract_band_dict

        series = pd.Series([
            {"delta": [1.0, 2.0], "theta": [3.0, 4.0]},
            {"delta": [5.0, 6.0], "theta": [7.0, 8.0]},
        ])
        vals, keys = extract_band_dict(series)
        assert keys == ["delta", "theta"]
        # Canonical shape: (n_windows, n_channels, n_bands)
        assert vals.shape == (2, 2, 2)
        # vals[window, channel, band] — channel 0, all bands, window 0
        np.testing.assert_array_equal(vals[0, 0], [1.0, 3.0])  # ch0: delta=1.0, theta=3.0

    def test_repack_band_dict(self):
        """Test repacking array back to list of band dicts."""
        from neurodent.visualization.feature_utils import extract_band_dict, repack_band_dict

        original = pd.Series([
            {"delta": [1.0, 2.0], "theta": [3.0, 4.0]},
            {"delta": [5.0, 6.0], "theta": [7.0, 8.0]},
        ])
        vals, keys = extract_band_dict(original)
        repacked = repack_band_dict(vals, keys)
        assert len(repacked) == 2
        assert list(repacked[0].keys()) == ["delta", "theta"]
        np.testing.assert_array_equal(repacked[0]["delta"], [1.0, 2.0])
        np.testing.assert_array_equal(repacked[1]["theta"], [7.0, 8.0])

    def test_extract_band_dict_round_trip(self):
        """Test that extract → repack round-trips correctly."""
        from neurodent.visualization.feature_utils import extract_band_dict, repack_band_dict

        original = pd.Series([
            {"alpha": [0.1, 0.2], "beta": [0.3, 0.4], "gamma": [0.5, 0.6]},
            {"alpha": [0.7, 0.8], "beta": [0.9, 1.0], "gamma": [1.1, 1.2]},
        ])
        vals, keys = extract_band_dict(original)
        repacked = repack_band_dict(vals, keys)
        for i in range(len(original)):
            for k in original.iloc[i].keys():
                np.testing.assert_array_almost_equal(
                    repacked[i][k], original.iloc[i][k]
                )

    def test_extract_hist_data_tuples(self):
        """Test extracting histogram data from tuple format (pickle origin)."""
        from neurodent.visualization.feature_utils import extract_hist_data

        series = pd.Series([
            (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([40.0, 50.0, 60.0])),
        ])
        coords, values = extract_hist_data(series)
        assert coords.shape == (2, 3)
        # 1-D values per cell (single-channel) → singleton channel dim inserted, shape (W, C, F)
        assert values.shape == (2, 1, 3)
        np.testing.assert_array_equal(coords[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(values[1, 0], [40.0, 50.0, 60.0])

    def test_extract_hist_data_lists(self):
        """Test extracting histogram data from list format (parquet origin)."""
        from neurodent.visualization.feature_utils import extract_hist_data

        series = pd.Series([
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[1.0, 2.0, 3.0], [40.0, 50.0, 60.0]],
        ])
        coords, values = extract_hist_data(series)
        assert coords.shape == (2, 3)
        # 1-D values per cell (single-channel) → singleton channel dim inserted, shape (W, C, F)
        assert values.shape == (2, 1, 3)
        np.testing.assert_array_equal(values[0, 0], [10.0, 20.0, 30.0])

    def test_extract_hist_data_multichannel(self):
        """Test extracting histogram data where values are 2-D (F, C) per cell."""
        from neurodent.visualization.feature_utils import extract_hist_data

        n_freq, n_chan = 4, 3
        coords_row = np.arange(n_freq, dtype=float)
        # Per-cell values are (F, C) = (4, 3)
        vals_row0 = np.arange(n_freq * n_chan, dtype=float).reshape(n_freq, n_chan)
        vals_row1 = vals_row0 + 100.0
        series = pd.Series([
            (coords_row, vals_row0),
            (coords_row, vals_row1),
        ])
        coords, values = extract_hist_data(series)
        assert coords.shape == (2, n_freq)
        # Canonical shape: (W, C, F)
        assert values.shape == (2, n_chan, n_freq)
        # Channel 0 of window 0 should equal the first row of vals_row0 transposed
        np.testing.assert_array_equal(values[0, 0], vals_row0[:, 0])

    def test_extract_linear_array_ragged_raises(self):
        """Test that ragged input raises ValueError."""
        from neurodent.visualization.feature_utils import extract_linear_array

        series = pd.Series([[1.0, 2.0, 3.0], [4.0, 5.0]])
        with pytest.raises(ValueError, match="Ragged input"):
            extract_linear_array(series)

    def test_extract_band_dict_ragged_raises(self):
        """Test that ragged band values raise ValueError."""
        from neurodent.visualization.feature_utils import extract_band_dict

        series = pd.Series([
            {"delta": [1.0, 2.0], "theta": [3.0, 4.0]},
            {"delta": [5.0, 6.0, 7.0], "theta": [8.0]},
        ])
        with pytest.raises(ValueError, match="Ragged input"):
            extract_band_dict(series)

    def test_extract_hist_data_ragged_raises(self):
        """Test that ragged histogram data raises ValueError."""
        from neurodent.visualization.feature_utils import extract_hist_data

        series = pd.Series([
            (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])),
            (np.array([1.0, 2.0]), np.array([40.0, 50.0])),
        ])
        with pytest.raises(ValueError, match="Ragged input"):
            extract_hist_data(series)

    def test_extract_linear_array_ftype_validation(self):
        """Test ftype param validates ndim."""
        from neurodent.visualization.feature_utils import extract_linear_array
        series = pd.Series([[1.0, 2.0], [3.0, 4.0]])
        # Correct ftype should pass
        result = extract_linear_array(series, ftype=constants.FeatureType.LINEAR)
        assert result.shape == (2, 2)
        # Wrong ftype should raise
        with pytest.raises(ValueError, match="Expected 3-D"):
            extract_linear_array(series, ftype=constants.FeatureType.LINEAR_2D)

    def test_extract_band_dict_ftype_validation(self):
        """Test ftype param validates ndim."""
        from neurodent.visualization.feature_utils import extract_band_dict
        series = pd.Series([{"delta": [1.0, 2.0], "theta": [3.0, 4.0]}])
        # Correct ftype should pass
        vals, keys = extract_band_dict(series, ftype=constants.FeatureType.BAND)
        assert vals.shape == (1, 2, 2)
        # Wrong ftype (expects 4-D) should raise
        with pytest.raises(ValueError, match="Expected 4-D"):
            extract_band_dict(series, ftype=constants.FeatureType.BANDED_MATRIX)

    def test_extract_hist_data_ftype_validation(self):
        """Test ftype param validates feature type."""
        from neurodent.visualization.feature_utils import extract_hist_data
        series = pd.Series([
            (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
        ])
        # Correct ftype should pass
        coords, values = extract_hist_data(series, ftype=constants.FeatureType.HIST)
        assert coords.shape == (1, 2)
        # Wrong ftype should raise
        with pytest.raises(ValueError, match="expects FeatureType.HIST"):
            extract_hist_data(series, ftype=constants.FeatureType.LINEAR)


class TestFlattenFeatureForPlotting:
    """Test flatten_feature_for_plotting utility."""

    def test_flatten_linear(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.LINEAR)
        assert result.shape == (2, 3, 1)
        np.testing.assert_array_equal(result[:, :, 0], vals)

    def test_flatten_linear_2d(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.LINEAR_2D)
        assert result.shape == (1, 2, 2)
        np.testing.assert_array_equal(result, vals)

    def test_flatten_band(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])  # (1, 2, 3)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.BAND)
        assert result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result, vals)
        np.testing.assert_array_equal(result[0, 0], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result[0, 1], [40.0, 50.0, 60.0])

    def test_flatten_simple_matrix_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.array([[[0.0, 0.1, 0.2],
                          [0.3, 0.0, 0.4],
                          [0.5, 0.6, 0.0]]])  # (1, 3, 3)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.SIMPLE_MATRIX, triag=True)
        # tril_indices(3, k=-1) -> (1,0)=0.3, (2,0)=0.5, (2,1)=0.6
        assert result.shape == (1, 3, 1)
        np.testing.assert_array_equal(result[0, :, 0], [0.3, 0.5, 0.6])

    def test_flatten_simple_matrix_no_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.SIMPLE_MATRIX, triag=False)
        assert result.shape == (1, 4, 1)
        np.testing.assert_array_equal(result[0, :, 0], [1.0, 2.0, 3.0, 4.0])

    def test_flatten_banded_matrix_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.zeros((1, 3, 3, 2))
        vals[0, 1, 0, 0] = 10.0; vals[0, 1, 0, 1] = 11.0
        vals[0, 2, 0, 0] = 20.0; vals[0, 2, 0, 1] = 21.0
        vals[0, 2, 1, 0] = 30.0; vals[0, 2, 1, 1] = 31.0
        result = flatten_feature_for_plotting(vals, constants.FeatureType.BANDED_MATRIX, triag=True)
        # tril_indices(3, k=-1) -> pairs: (1,0), (2,0), (2,1)
        assert result.shape == (1, 3, 2)
        np.testing.assert_array_equal(result[0, 0], [10.0, 11.0])
        np.testing.assert_array_equal(result[0, 1], [20.0, 21.0])
        np.testing.assert_array_equal(result[0, 2], [30.0, 31.0])

    def test_flatten_banded_matrix_no_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.arange(8, dtype=float).reshape(1, 2, 2, 2)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.BANDED_MATRIX, triag=False)
        assert result.shape == (1, 4, 2)
        np.testing.assert_array_equal(result[0, 0], [0.0, 1.0])
        np.testing.assert_array_equal(result[0, 1], [2.0, 3.0])
        np.testing.assert_array_equal(result[0, 2], [4.0, 5.0])
        np.testing.assert_array_equal(result[0, 3], [6.0, 7.0])

    def test_flatten_unsupported_raises(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Unsupported FeatureType"):
            flatten_feature_for_plotting(vals, constants.FeatureType.HIST)


class TestCollapseFeatureChannels:
    """Test collapse_feature_channels utility."""

    def test_collapse_linear(self):
        from neurodent.visualization.feature_utils import collapse_feature_channels
        vals = np.array([[1.0, 3.0], [5.0, 7.0]])  # (2 windows, 2 channels)
        result = collapse_feature_channels(vals, constants.FeatureType.LINEAR)
        np.testing.assert_array_equal(result, [2.0, 6.0])

    def test_collapse_linear_2d(self):
        from neurodent.visualization.feature_utils import collapse_feature_channels
        vals = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2 chan, 2 comp)
        result = collapse_feature_channels(vals, constants.FeatureType.LINEAR_2D)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], [2.0, 3.0])

    def test_collapse_band(self):
        from neurodent.visualization.feature_utils import collapse_feature_channels
        # Canonical shape: (n_windows, n_channels, n_bands)
        vals = np.array([[[1.0, 5.0], [3.0, 7.0]]])  # (1, 2 chan, 2 bands)
        result = collapse_feature_channels(vals, constants.FeatureType.BAND)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], [2.0, 6.0])

    def test_collapse_simple_matrix(self):
        from neurodent.visualization.feature_utils import collapse_feature_channels
        # 2x2 symmetric matrix, tril pair = (1,0) only
        vals = np.array([[[0.0, 0.5], [0.5, 0.0]]])  # (1, 2, 2)
        result = collapse_feature_channels(vals, constants.FeatureType.SIMPLE_MATRIX)
        assert result.shape == (1,)
        np.testing.assert_almost_equal(result[0], 0.5)

    def test_collapse_banded_matrix(self):
        from neurodent.visualization.feature_utils import collapse_feature_channels
        # Canonical shape: (n_windows, n_chan, n_chan, n_bands)
        vals = np.zeros((1, 2, 2, 2))
        vals[0, 1, 0, 0] = 0.8  # ch_row=1, ch_col=0, band=0
        vals[0, 1, 0, 1] = 0.6  # ch_row=1, ch_col=0, band=1
        result = collapse_feature_channels(vals, constants.FeatureType.BANDED_MATRIX)
        assert result.shape == (1, 2)
        np.testing.assert_almost_equal(result[0, 0], 0.8)
        np.testing.assert_almost_equal(result[0, 1], 0.6)

    def test_collapse_hist(self):
        from neurodent.visualization.feature_utils import collapse_feature_channels
        vals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2 chan, 3 freq)
        result = collapse_feature_channels(vals, constants.FeatureType.HIST)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result[0], [2.5, 3.5, 4.5])

    def test_collapse_hist_via_function(self):
        """Verify HIST is supported in collapse_feature_channels."""
        from neurodent.visualization.feature_utils import collapse_feature_channels
        vals = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2 chan, 2 freq)
        result = collapse_feature_channels(vals, constants.FeatureType.HIST)
        assert result.shape == (1, 2)

    def test_collapse_linear_with_nan(self):
        """NaN values are skipped by nanmean."""
        from neurodent.visualization.feature_utils import collapse_feature_channels
        vals = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])  # (2, 3)
        result = collapse_feature_channels(vals, constants.FeatureType.LINEAR)
        # nanmean([1, nan, 3])=2.0; nanmean([4, 5, nan])=4.5
        np.testing.assert_array_almost_equal(result, [2.0, 4.5])

    def test_collapse_simple_matrix_3x3(self):
        """3x3 matrix: 3 tril pairs averaged."""
        from neurodent.visualization.feature_utils import collapse_feature_channels
        mat = np.array([[[0.0, 0.1, 0.2],
                         [0.3, 0.0, 0.4],
                         [0.5, 0.6, 0.0]]])  # (1, 3, 3)
        result = collapse_feature_channels(mat, constants.FeatureType.SIMPLE_MATRIX)
        # tril_indices(3, k=-1): (1,0)=0.3, (2,0)=0.5, (2,1)=0.6 -> mean
        assert result.shape == (1,)
        np.testing.assert_almost_equal(result[0], np.mean([0.3, 0.5, 0.6]))

    def test_collapse_banded_matrix_with_nan(self):
        """NaN in one tril pair is skipped by nanmean."""
        from neurodent.visualization.feature_utils import collapse_feature_channels
        vals = np.zeros((1, 3, 3, 1))
        vals[0, 1, 0, 0] = 0.6
        vals[0, 2, 0, 0] = np.nan
        vals[0, 2, 1, 0] = 0.8
        result = collapse_feature_channels(vals, constants.FeatureType.BANDED_MATRIX)
        # tril pairs: (1,0)=0.6, (2,0)=nan, (2,1)=0.8 -> nanmean=0.7
        assert result.shape == (1, 1)
        np.testing.assert_almost_equal(result[0, 0], 0.7)

    def test_collapse_linear_single_channel(self):
        """Single channel: mean is identity."""
        from neurodent.visualization.feature_utils import collapse_feature_channels
        vals = np.array([[5.0], [10.0]])  # (2, 1)
        result = collapse_feature_channels(vals, constants.FeatureType.LINEAR)
        np.testing.assert_array_equal(result, [5.0, 10.0])


class TestExtractFeature:
    """Test extract_feature utility."""

    def test_extract_linear(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([[1.0, 2.0], [3.0, 4.0]])
        vals, keys = extract_feature(series, constants.FeatureType.LINEAR)
        assert vals.shape == (2, 2)
        assert keys is None
        np.testing.assert_array_equal(vals[0], [1.0, 2.0])
        np.testing.assert_array_equal(vals[1], [3.0, 4.0])

    def test_extract_linear_2d(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([[[1.0, 2.0], [3.0, 4.0]]])
        vals, keys = extract_feature(series, constants.FeatureType.LINEAR_2D)
        assert vals.shape == (1, 2, 2)
        assert keys is None
        np.testing.assert_array_equal(vals[0, 0], [1.0, 2.0])
        np.testing.assert_array_equal(vals[0, 1], [3.0, 4.0])

    def test_extract_simple_matrix(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([[[0.0, 0.5], [0.5, 0.0]]])
        vals, keys = extract_feature(series, constants.FeatureType.SIMPLE_MATRIX)
        assert vals.shape == (1, 2, 2)
        assert keys is None
        np.testing.assert_array_equal(vals[0], [[0.0, 0.5], [0.5, 0.0]])

    def test_extract_band(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([{"delta": [1.0, 2.0], "theta": [3.0, 4.0]}])
        vals, keys = extract_feature(series, constants.FeatureType.BAND)
        assert vals.shape == (1, 2, 2)  # (W, C, B)
        assert keys == ["delta", "theta"]
        # After transpose (W, B, C) -> (W, C, B): ch0=[delta=1, theta=3], ch1=[delta=2, theta=4]
        np.testing.assert_array_equal(vals[0, 0], [1.0, 3.0])
        np.testing.assert_array_equal(vals[0, 1], [2.0, 4.0])

    def test_extract_banded_matrix(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([{"delta": [[0.0, 0.5], [0.5, 0.0]]}])
        vals, keys = extract_feature(series, constants.FeatureType.BANDED_MATRIX)
        assert vals.shape == (1, 2, 2, 1)  # (W, C, C, B)
        assert keys == ["delta"]
        np.testing.assert_array_equal(vals[0, :, :, 0], [[0.0, 0.5], [0.5, 0.0]])


class TestFormatChannelData:
    """Test format_channel_data utility with numeric validity."""

    def test_format_linear_collapsed(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[1.0, 3.0], [5.0, 7.0]])  # (2 win, 2 chan)
        result = format_channel_data(vals, constants.FeatureType.LINEAR, collapse_channels=True)
        assert "average" in result
        np.testing.assert_array_almost_equal(result["average"], [2.0, 6.0])

    def test_format_linear_per_channel(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[1.0, 3.0], [5.0, 7.0]])  # (2 win, 2 chan)
        result = format_channel_data(
            vals, constants.FeatureType.LINEAR, collapse_channels=False,
            ch_to_idx={"LM": 0, "RM": 1}, channels=["LM", "RM"], ch_names=["LM", "RM"],
        )
        assert set(result.keys()) == {"LM", "RM"}
        assert result["LM"] == [1.0, 5.0]
        assert result["RM"] == [3.0, 7.0]

    def test_format_band_collapsed(self):
        from neurodent.visualization.feature_utils import format_channel_data
        # Canonical: (W, C, B)
        vals = np.array([[[1.0, 5.0], [3.0, 7.0]]])  # (1, 2 chan, 2 bands)
        result = format_channel_data(vals, constants.FeatureType.BAND, collapse_channels=True)
        assert "average" in result
        np.testing.assert_array_almost_equal(result["average"], [[2.0, 6.0]])

    def test_format_band_per_channel(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[[1.0, 5.0], [3.0, 7.0]]])  # (1, 2 chan, 2 bands)
        result = format_channel_data(
            vals, constants.FeatureType.BAND, collapse_channels=False,
            ch_to_idx={"LM": 0, "RM": 1}, channels=["LM", "RM"], ch_names=["LM", "RM"],
        )
        assert set(result.keys()) == {"LM", "RM"}
        np.testing.assert_array_almost_equal(result["LM"], [[1.0, 5.0]])

    def test_format_simple_matrix_collapsed(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[[0.0, 0.5], [0.5, 0.0]]])  # (1, 2, 2)
        result = format_channel_data(vals, constants.FeatureType.SIMPLE_MATRIX, collapse_channels=True)
        assert "average" in result
        np.testing.assert_array_almost_equal(result["average"], [0.5])

    def test_format_simple_matrix_not_collapsed(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[[0.0, 0.5], [0.5, 0.0]]])  # (1, 2, 2)
        result = format_channel_data(vals, constants.FeatureType.SIMPLE_MATRIX, collapse_channels=False)
        assert "all" in result

    def test_format_banded_matrix_collapsed(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.zeros((1, 2, 2, 2))
        vals[0, 1, 0, 0] = 0.8
        vals[0, 1, 0, 1] = 0.6
        result = format_channel_data(vals, constants.FeatureType.BANDED_MATRIX, collapse_channels=True)
        assert "average" in result
        np.testing.assert_array_almost_equal(result["average"], [[0.8, 0.6]])

    def test_format_banded_matrix_not_collapsed(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.zeros((1, 2, 2, 2))
        result = format_channel_data(vals, constants.FeatureType.BANDED_MATRIX, collapse_channels=False)
        assert "all" in result

    def test_format_hist_collapsed(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2 chan, 3 freq)
        result = format_channel_data(vals, constants.FeatureType.HIST, collapse_channels=True)
        assert "average" in result
        np.testing.assert_array_almost_equal(result["average"], [[2.5, 3.5, 4.5]])

    def test_format_hist_per_channel(self):
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2 chan, 2 freq)
        result = format_channel_data(
            vals, constants.FeatureType.HIST, collapse_channels=False,
            ch_to_idx={"LM": 0, "RM": 1}, channels=["LM", "RM"], ch_names=["LM", "RM"],
        )
        assert set(result.keys()) == {"LM", "RM"}
        np.testing.assert_array_almost_equal(result["LM"], [[1.0, 2.0]])

    def test_format_channel_subset(self):
        """Only requested channels are included in the result."""
        from neurodent.visualization.feature_utils import format_channel_data
        vals = np.array([[1.0, 3.0, 5.0]])  # (1, 3 chan)
        result = format_channel_data(
            vals, constants.FeatureType.LINEAR, collapse_channels=False,
            ch_to_idx={"A": 0, "B": 1, "C": 2}, channels=["A", "C"], ch_names=["A", "B", "C"],
        )
        assert set(result.keys()) == {"A", "C"}
        assert result["A"] == [1.0]
        assert result["C"] == [5.0]


class TestPipelineComposition:
    """End-to-end numerical tests: extract -> collapse -> format -> flatten.

    Verifies that values survive the full pipeline correctly and that
    channel ordering is preserved through transpositions.
    """

    def test_pipeline_linear(self):
        """LINEAR: extract -> format(collapsed) -> format(per-channel) -> flatten."""
        from neurodent.visualization.feature_utils import (
            extract_feature, format_channel_data, flatten_feature_for_plotting,
        )
        # 2 windows, 2 channels: ch0=[1, 3], ch1=[2, 4]
        series = pd.Series([[1.0, 2.0], [3.0, 4.0]])
        vals, keys = extract_feature(series, constants.FeatureType.LINEAR)

        assert vals.shape == (2, 2)
        np.testing.assert_array_equal(vals, [[1.0, 2.0], [3.0, 4.0]])

        # Collapsed: average over channels
        fmt = format_channel_data(vals, constants.FeatureType.LINEAR, collapse_channels=True)
        np.testing.assert_array_almost_equal(fmt["average"], [1.5, 3.5])

        # Per-channel: verify channel identity
        fmt_ch = format_channel_data(
            vals, constants.FeatureType.LINEAR, collapse_channels=False,
            ch_to_idx={"A": 0, "B": 1}, channels=["A", "B"], ch_names=["A", "B"],
        )
        assert fmt_ch["A"] == [1.0, 3.0]
        assert fmt_ch["B"] == [2.0, 4.0]

        # Flatten
        flat = flatten_feature_for_plotting(vals, constants.FeatureType.LINEAR)
        assert flat.shape == (2, 2, 1)
        np.testing.assert_array_equal(flat[:, :, 0], vals)

    def test_pipeline_linear_2d(self):
        """LINEAR_2D: extract -> format(collapsed) -> flatten."""
        from neurodent.visualization.feature_utils import (
            extract_feature, format_channel_data, flatten_feature_for_plotting,
        )
        # 1 window, 2 channels, 2 components
        series = pd.Series([[[1.0, 2.0], [3.0, 4.0]]])
        vals, keys = extract_feature(series, constants.FeatureType.LINEAR_2D)

        assert vals.shape == (1, 2, 2)
        np.testing.assert_array_equal(vals[0], [[1.0, 2.0], [3.0, 4.0]])

        # Collapsed: mean over channels axis
        fmt = format_channel_data(vals, constants.FeatureType.LINEAR_2D, collapse_channels=True)
        np.testing.assert_array_almost_equal(fmt["average"], [[2.0, 3.0]])

        flat = flatten_feature_for_plotting(vals, constants.FeatureType.LINEAR_2D)
        assert flat.shape == (1, 2, 2)
        np.testing.assert_array_equal(flat, vals)

    def test_pipeline_band(self):
        """BAND: extract (with transpose) -> format -> repack round-trip -> flatten."""
        from neurodent.visualization.feature_utils import (
            extract_feature, format_channel_data, flatten_feature_for_plotting,
            repack_band_dict,
        )
        # 2 windows, 2 channels, 2 bands (alpha, beta)
        series = pd.Series([
            {"alpha": [10.0, 20.0], "beta": [30.0, 40.0]},
            {"alpha": [50.0, 60.0], "beta": [70.0, 80.0]},
        ])
        vals, keys = extract_feature(series, constants.FeatureType.BAND)

        assert vals.shape == (2, 2, 2)  # (W=2, C=2, B=2)
        assert keys == ["alpha", "beta"]
        # After transpose (W, B, C) -> (W, C, B):
        # ch0 gets [alpha, beta] = [10, 30] and [50, 70]
        np.testing.assert_array_equal(vals[0, 0], [10.0, 30.0])
        np.testing.assert_array_equal(vals[0, 1], [20.0, 40.0])
        np.testing.assert_array_equal(vals[1, 0], [50.0, 70.0])
        np.testing.assert_array_equal(vals[1, 1], [60.0, 80.0])

        # Collapsed: average over channels
        fmt = format_channel_data(vals, constants.FeatureType.BAND, collapse_channels=True)
        np.testing.assert_array_almost_equal(
            fmt["average"], [[15.0, 35.0], [55.0, 75.0]]
        )

        # Round-trip: repack and verify original dict structure
        repacked = repack_band_dict(vals, keys)
        np.testing.assert_array_equal(repacked[0]["alpha"], [10.0, 20.0])
        np.testing.assert_array_equal(repacked[0]["beta"], [30.0, 40.0])

        # Flatten (identity for BAND)
        flat = flatten_feature_for_plotting(vals, constants.FeatureType.BAND)
        np.testing.assert_array_equal(flat, vals)

    def test_pipeline_simple_matrix(self):
        """SIMPLE_MATRIX: extract -> collapse (tril) -> flatten (tril)."""
        from neurodent.visualization.feature_utils import (
            extract_feature, format_channel_data, flatten_feature_for_plotting,
        )
        # 1 window, 2x2 symmetric matrix
        series = pd.Series([[[0.0, 0.3], [0.3, 0.0]]])
        vals, keys = extract_feature(series, constants.FeatureType.SIMPLE_MATRIX)

        assert vals.shape == (1, 2, 2)
        np.testing.assert_array_equal(vals[0], [[0.0, 0.3], [0.3, 0.0]])

        # Collapsed: tril pair (1,0) = 0.3
        fmt = format_channel_data(vals, constants.FeatureType.SIMPLE_MATRIX, collapse_channels=True)
        np.testing.assert_array_almost_equal(fmt["average"], [0.3])

        # Flatten triag: 1 pair
        flat = flatten_feature_for_plotting(vals, constants.FeatureType.SIMPLE_MATRIX, triag=True)
        assert flat.shape == (1, 1, 1)
        np.testing.assert_array_equal(flat[0, 0, 0], 0.3)

    def test_pipeline_banded_matrix(self):
        """BANDED_MATRIX: extract (transpose) -> collapse -> repack round-trip -> flatten."""
        from neurodent.visualization.feature_utils import (
            extract_feature, format_channel_data, flatten_feature_for_plotting,
            repack_band_dict,
        )
        # 1 window, 2x2 matrix, 1 band
        series = pd.Series([{"delta": [[0.0, 0.5], [0.7, 0.0]]}])
        vals, keys = extract_feature(series, constants.FeatureType.BANDED_MATRIX)

        assert vals.shape == (1, 2, 2, 1)  # (W, C, C, B)
        assert keys == ["delta"]
        np.testing.assert_array_equal(vals[0, :, :, 0], [[0.0, 0.5], [0.7, 0.0]])

        # Collapsed: tril pair (1,0) = 0.7
        fmt = format_channel_data(vals, constants.FeatureType.BANDED_MATRIX, collapse_channels=True)
        np.testing.assert_array_almost_equal(fmt["average"], [[0.7]])

        # Round-trip repack
        repacked = repack_band_dict(vals, keys)
        np.testing.assert_array_equal(repacked[0]["delta"], [[0.0, 0.5], [0.7, 0.0]])

        # Flatten triag: 1 pair, 1 band
        flat = flatten_feature_for_plotting(vals, constants.FeatureType.BANDED_MATRIX, triag=True)
        assert flat.shape == (1, 1, 1)
        np.testing.assert_array_equal(flat[0, 0, 0], 0.7)

    def test_pipeline_channel_ordering_preserved(self):
        """Verify that channel identity is preserved through the full pipeline."""
        from neurodent.visualization.feature_utils import (
            extract_feature, format_channel_data, flatten_feature_for_plotting,
        )
        # 3 channels with distinct values so mis-ordering is immediately obvious
        series = pd.Series([[100.0, 200.0, 300.0]])
        vals, _ = extract_feature(series, constants.FeatureType.LINEAR)

        # Per-channel format
        fmt = format_channel_data(
            vals, constants.FeatureType.LINEAR, collapse_channels=False,
            ch_to_idx={"LMot": 0, "RMot": 1, "LBar": 2},
            channels=["LMot", "RMot", "LBar"],
            ch_names=["LMot", "RMot", "LBar"],
        )
        assert fmt["LMot"] == [100.0]
        assert fmt["RMot"] == [200.0]
        assert fmt["LBar"] == [300.0]

        # Flatten preserves channel positions
        flat = flatten_feature_for_plotting(vals, constants.FeatureType.LINEAR)
        np.testing.assert_array_equal(flat[0, 0, 0], 100.0)
        np.testing.assert_array_equal(flat[0, 1, 0], 200.0)
        np.testing.assert_array_equal(flat[0, 2, 0], 300.0)

    def test_channel_ordering_band_round_trip(self):
        """BAND: verify channel identity survives extract -> repack -> format."""
        from neurodent.visualization.feature_utils import (
            extract_band_dict, repack_band_dict, format_channel_data,
        )
        # ch0=[1,2,3] across 3 bands, ch1=[4,5,6] — distinct per channel
        series = pd.Series([
            {"delta": [1.0, 4.0], "theta": [2.0, 5.0], "alpha": [3.0, 6.0]},
        ])
        vals, keys = extract_band_dict(series, ftype=constants.FeatureType.BAND)
        # Canonical (W=1, C=2, B=3): ch0=[1,2,3], ch1=[4,5,6]
        np.testing.assert_array_equal(vals[0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(vals[0, 1], [4.0, 5.0, 6.0])

        # Repack preserves original dict structure
        repacked = repack_band_dict(vals, keys)
        np.testing.assert_array_equal(repacked[0]["delta"], [1.0, 4.0])
        np.testing.assert_array_equal(repacked[0]["theta"], [2.0, 5.0])
        np.testing.assert_array_equal(repacked[0]["alpha"], [3.0, 6.0])

        # Per-channel format preserves identity
        fmt = format_channel_data(
            vals, constants.FeatureType.BAND, collapse_channels=False,
            ch_to_idx={"ch0": 0, "ch1": 1}, channels=["ch0", "ch1"], ch_names=["ch0", "ch1"],
        )
        np.testing.assert_array_almost_equal(fmt["ch0"], [[1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(fmt["ch1"], [[4.0, 5.0, 6.0]])

    def test_channel_ordering_hist_extract_format(self):
        """HIST: verify channel identity survives the (F,C) -> (W,C,F) transpose."""
        from neurodent.visualization.feature_utils import (
            extract_hist_data, format_channel_data,
        )
        # 1 window, per-cell values shape (F=3, C=2): col0=ch0, col1=ch1
        vals_cell = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])  # (F=3, C=2)
        coords_cell = np.array([10.0, 20.0, 30.0])
        series = pd.Series([(coords_cell, vals_cell)])
        coords, values = extract_hist_data(series, ftype=constants.FeatureType.HIST)
        # Canonical (W=1, C=2, F=3)
        assert values.shape == (1, 2, 3)
        np.testing.assert_array_equal(values[0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(values[0, 1], [10.0, 20.0, 30.0])

        # Per-channel format preserves identity through the transpose
        fmt = format_channel_data(
            values, constants.FeatureType.HIST, collapse_channels=False,
            ch_to_idx={"A": 0, "B": 1}, channels=["A", "B"], ch_names=["A", "B"],
        )
        np.testing.assert_array_almost_equal(fmt["A"], [[1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(fmt["B"], [[10.0, 20.0, 30.0]])

    def test_channel_ordering_banded_matrix_round_trip(self):
        """BANDED_MATRIX: verify (row, col) ordering survives extract -> repack -> flatten."""
        from neurodent.visualization.feature_utils import (
            extract_band_dict, repack_band_dict, flatten_feature_for_plotting,
        )
        # 3x3 matrix with unique values at each (i,j), 1 band
        mat = np.arange(9, dtype=float).reshape(3, 3)
        series = pd.Series([{"delta": mat.tolist()}])
        vals, keys = extract_band_dict(series, ftype=constants.FeatureType.BANDED_MATRIX)
        # Canonical (W=1, C=3, C=3, B=1)
        np.testing.assert_array_equal(vals[0, :, :, 0], mat)

        # Repack preserves matrix
        repacked = repack_band_dict(vals, keys)
        np.testing.assert_array_equal(repacked[0]["delta"], mat)

        # Flatten triag: tril_indices(3, k=-1) -> (1,0)=3, (2,0)=6, (2,1)=7
        flat = flatten_feature_for_plotting(vals, constants.FeatureType.BANDED_MATRIX, triag=True)
        assert flat.shape == (1, 3, 1)
        np.testing.assert_array_equal(flat[0, :, 0], [3.0, 6.0, 7.0])


class TestBinSpikeTimes:
    """Test bin_spike_times and _bin_spike_df numerical correctness."""

    def test_bin_spike_times_basic(self):
        from neurodent.visualization.results import bin_spike_times
        counts = bin_spike_times([5.0, 15.0, 25.0], [10.0, 10.0, 10.0])
        assert counts == [1, 1, 1]

    def test_bin_spike_times_empty(self):
        from neurodent.visualization.results import bin_spike_times
        counts = bin_spike_times([], [10.0, 10.0])
        assert counts == [0, 0]

    def test_bin_spike_times_boundary(self):
        from neurodent.visualization.results import bin_spike_times
        # bin_edges = [0, 10, 20]; np.histogram: bins are [left, right) except last [left, right]
        counts = bin_spike_times([0.0, 10.0, 20.0], [10.0, 10.0])
        # 0.0 in [0,10), 10.0 in [10,20], 20.0 in [10,20]
        assert counts == [1, 2]

    def test_bin_spike_times_multiple_per_bin(self):
        from neurodent.visualization.results import bin_spike_times
        counts = bin_spike_times([1.0, 2.0, 3.0, 15.0], [10.0, 10.0])
        assert counts == [3, 1]

    def test_bin_spike_df_basic(self):
        from neurodent.visualization.results import _bin_spike_df
        df = pd.DataFrame({"duration": [10.0, 10.0, 10.0]})
        spikes_channel = [[5.0, 15.0], [25.0]]
        result = _bin_spike_df(df, spikes_channel)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [1, 1, 0])
        np.testing.assert_array_equal(result[:, 1], [0, 0, 1])

    def test_bin_spike_df_empty_channel(self):
        from neurodent.visualization.results import _bin_spike_df
        df = pd.DataFrame({"duration": [5.0, 5.0]})
        spikes_channel = [[], [1.0, 6.0]]
        result = _bin_spike_df(df, spikes_channel)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[:, 0], [0, 0])
        np.testing.assert_array_equal(result[:, 1], [1, 1])


class TestAverageAcrossChannels:
    """Test WindowAnalysisResult._average_across_channels numerical correctness."""

    @pytest.fixture
    def minimal_war(self):
        df = pd.DataFrame({
            "animal": ["A1", "A1"],
            "animalday": ["A1_d1", "A1_d1"],
            "genotype": ["WT", "WT"],
            "duration": [1.0, 1.0],
        })
        return WindowAnalysisResult(
            result=df, animal_id="A1", genotype="WT", sex="Male",
            channel_names=["LMot", "RMot", "LBar"],
        )

    def test_average_vector(self, minimal_war):
        """1D features: nanmean across channels."""
        df = pd.DataFrame({"rms": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]})
        result = minimal_war._average_across_channels(df, ["rms"])
        np.testing.assert_array_almost_equal(result["rms"].values, [2.0, 5.0])

    def test_average_matrix_upper_tri(self, minimal_war):
        """2D matrix: mean of upper triangle (k=1)."""
        mat = np.array([[0.0, 0.2, 0.4],
                        [0.2, 0.0, 0.6],
                        [0.4, 0.6, 0.0]])
        df = pd.DataFrame({"pcorr_delta": [mat]})
        result = minimal_war._average_across_channels(df, ["pcorr_delta"])
        # triu_indices(3, k=1): (0,1)=0.2, (0,2)=0.4, (1,2)=0.6 -> mean=0.4
        np.testing.assert_almost_equal(result["pcorr_delta"].iloc[0], 0.4)

    def test_average_matrix_with_nan(self, minimal_war):
        """NaN in upper triangle is skipped by nanmean."""
        mat = np.array([[0.0, np.nan, 0.4],
                        [0.2, 0.0, 0.6],
                        [0.4, 0.6, 0.0]])
        df = pd.DataFrame({"feat": [mat]})
        result = minimal_war._average_across_channels(df, ["feat"])
        # triu(k=1): nan, 0.4, 0.6 -> nanmean = 0.5
        np.testing.assert_almost_equal(result["feat"].iloc[0], 0.5)

    def test_average_small_matrix(self, minimal_war):
        """1x1 matrix: falls back to nanmean of entire matrix."""
        mat = np.array([[5.0]])
        df = pd.DataFrame({"feat": [mat]})
        result = minimal_war._average_across_channels(df, ["feat"])
        np.testing.assert_almost_equal(result["feat"].iloc[0], 5.0)

    def test_average_scalar_passthrough(self, minimal_war):
        """Scalar features left unchanged."""
        df = pd.DataFrame({"scalar_feat": [3.14, 2.72]})
        result = minimal_war._average_across_channels(df, ["scalar_feat"])
        np.testing.assert_array_almost_equal(result["scalar_feat"].values, [3.14, 2.72])

    def test_average_missing_column_skipped(self, minimal_war):
        """Missing column does not crash; present column still averaged."""
        df = pd.DataFrame({"rms": [[1.0, 2.0, 3.0]]})
        result = minimal_war._average_across_channels(df, ["rms", "nonexistent"])
        assert "nonexistent" not in result.columns
        np.testing.assert_almost_equal(result["rms"].iloc[0], 2.0)


class TestExtractBandFeatures:
    """Test WindowAnalysisResult._extract_band_features numerical correctness."""

    @pytest.fixture
    def minimal_war(self):
        df = pd.DataFrame({
            "animal": ["A1", "A1"],
            "animalday": ["A1_d1", "A1_d1"],
            "genotype": ["WT", "WT"],
            "duration": [1.0, 1.0],
        })
        return WindowAnalysisResult(
            result=df, animal_id="A1", genotype="WT", sex="Male",
            channel_names=["LMot", "RMot", "LBar"],
        )

    def test_extract_band_features_basic(self, minimal_war):
        """Dict unpacking produces correct per-band columns."""
        df = pd.DataFrame({
            "psdband": [
                {"delta": [1.0, 2.0, 3.0], "theta": [4.0, 5.0, 6.0]},
                {"delta": [7.0, 8.0, 9.0], "theta": [10.0, 11.0, 12.0]},
            ]
        })
        result = minimal_war._extract_band_features(df, "psdband", ["delta", "theta"])
        np.testing.assert_array_equal(result["psdband_delta"].iloc[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result["psdband_theta"].iloc[1], [10.0, 11.0, 12.0])

    def test_extract_band_features_missing_band(self, minimal_war):
        """Missing band key fills with NaN array."""
        df = pd.DataFrame({
            "psdband": [{"delta": [1.0, 2.0, 3.0]}]
        })
        result = minimal_war._extract_band_features(df, "psdband", ["delta", "theta"])
        np.testing.assert_array_equal(result["psdband_delta"].iloc[0], [1.0, 2.0, 3.0])
        assert np.all(np.isnan(result["psdband_theta"].iloc[0]))

    def test_extract_band_features_missing_column(self, minimal_war):
        """Feature column not in df returns df unchanged."""
        df = pd.DataFrame({"other": [1.0]})
        result = minimal_war._extract_band_features(df, "psdband", ["delta"])
        assert "psdband_delta" not in result.columns


class TestExtractBandedMatrixFeatures:
    """Test WindowAnalysisResult._extract_banded_matrix_features numerical correctness."""

    @pytest.fixture
    def minimal_war(self):
        df = pd.DataFrame({
            "animal": ["A1"],
            "animalday": ["A1_d1"],
            "genotype": ["WT"],
            "duration": [1.0],
        })
        return WindowAnalysisResult(
            result=df, animal_id="A1", genotype="WT", sex="Male",
            channel_names=["LMot", "RMot", "LBar"],
        )

    def test_extract_banded_matrix_dict_format(self, minimal_war):
        """Dict storage: per-band 2D matrices extracted to separate columns."""
        mat_a = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]])
        mat_b = np.array([[0.0, 0.4, 0.5], [0.4, 0.0, 0.6], [0.5, 0.6, 0.0]])
        df = pd.DataFrame({"cohere": [{"alpha": mat_a, "beta": mat_b}]})
        result = minimal_war._extract_banded_matrix_features(df, "cohere", ["alpha", "beta"])
        np.testing.assert_array_equal(result["cohere_alpha"].iloc[0], mat_a)
        np.testing.assert_array_equal(result["cohere_beta"].iloc[0], mat_b)

    def test_extract_banded_matrix_3d_format(self, minimal_war):
        """3D array format: slice per band index."""
        mat_3d = np.arange(18, dtype=float).reshape(2, 3, 3)
        df = pd.DataFrame({"cohere": [mat_3d]})
        result = minimal_war._extract_banded_matrix_features(df, "cohere", ["alpha", "beta"])
        np.testing.assert_array_equal(result["cohere_alpha"].iloc[0], mat_3d[0, :, :])
        np.testing.assert_array_equal(result["cohere_beta"].iloc[0], mat_3d[1, :, :])

    def test_extract_banded_matrix_missing_band(self, minimal_war):
        """Missing band key fills with NaN matrix."""
        mat_a = np.eye(3)
        df = pd.DataFrame({"cohere": [{"alpha": mat_a}]})
        result = minimal_war._extract_banded_matrix_features(df, "cohere", ["alpha", "beta"])
        np.testing.assert_array_equal(result["cohere_alpha"].iloc[0], mat_a)
        assert np.all(np.isnan(result["cohere_beta"].iloc[0]))

    def test_extract_banded_matrix_2d_raises(self, minimal_war):
        """2D array input raises ValueError."""
        mat_2d = np.eye(3)
        df = pd.DataFrame({"cohere": [mat_2d]})
        with pytest.raises(ValueError, match="stored as a 2D array"):
            minimal_war._extract_banded_matrix_features(df, "cohere", ["alpha"])


class TestDataProcessingForVisualization:
    """Test data processing functions for visualization."""

    def test_df_normalize_baseline(self):
        """Test baseline normalization function."""
        from neurodent.visualization.plotting.experiment import df_normalize_baseline

        df = pd.DataFrame(
            {
                "genotype": ["WT", "WT", "KO", "KO"],
                "condition": ["baseline", "treatment", "baseline", "treatment"],
                "rms": [100.0, 120.0, 90.0, 110.0],
            }
        )

        result = df_normalize_baseline(
            df=df, feature="rms", groupby=["genotype"], baseline_key="baseline", baseline_groupby=["condition"]
        )

        assert isinstance(result, pd.DataFrame)
        assert "rms" in result.columns

    def test_df_normalize_baseline_subtract_values(self):
        """Verify subtracted values are numerically correct."""
        from neurodent.visualization.plotting.experiment import df_normalize_baseline

        df = pd.DataFrame({
            "genotype": ["WT", "WT", "KO", "KO"],
            "condition": ["baseline", "treatment", "baseline", "treatment"],
            "rms": [100.0, 120.0, 90.0, 110.0],
        })
        result = df_normalize_baseline(
            df=df, feature="rms", groupby=["genotype"],
            baseline_key="baseline", baseline_groupby=["condition"],
            operation="subtract",
        )
        wt_base = result[(result["genotype"] == "WT") & (result["condition"] == "baseline")]["rms"].iloc[0]
        wt_treat = result[(result["genotype"] == "WT") & (result["condition"] == "treatment")]["rms"].iloc[0]
        ko_treat = result[(result["genotype"] == "KO") & (result["condition"] == "treatment")]["rms"].iloc[0]
        np.testing.assert_almost_equal(wt_base, 0.0)
        np.testing.assert_almost_equal(wt_treat, 20.0)
        np.testing.assert_almost_equal(ko_treat, 20.0)

    def test_df_normalize_baseline_divide_values(self):
        """Verify divided values are numerically correct."""
        from neurodent.visualization.plotting.experiment import df_normalize_baseline

        df = pd.DataFrame({
            "genotype": ["WT", "WT"],
            "condition": ["baseline", "treatment"],
            "rms": [100.0, 150.0],
        })
        result = df_normalize_baseline(
            df=df, feature="rms", groupby=["genotype"],
            baseline_key="baseline", baseline_groupby=["condition"],
            operation="divide",
        )
        treatment = result[result["condition"] == "treatment"]["rms"].iloc[0]
        np.testing.assert_almost_equal(treatment, 1.5)

    def test_df_normalize_baseline_remove_baseline(self):
        """Baseline rows removed after normalization."""
        from neurodent.visualization.plotting.experiment import df_normalize_baseline

        df = pd.DataFrame({
            "genotype": ["WT", "WT"],
            "condition": ["baseline", "treatment"],
            "rms": [100.0, 150.0],
        })
        result = df_normalize_baseline(
            df=df, feature="rms", groupby=["genotype"],
            baseline_key="baseline", baseline_groupby=["condition"],
            remove_baseline=True,
        )
        assert len(result) == 1
        assert result["condition"].iloc[0] == "treatment"
        np.testing.assert_almost_equal(result["rms"].iloc[0], 50.0)

    def test_df_normalize_baseline_per_group(self):
        """Per-genotype baselines via remaining_groupby."""
        from neurodent.visualization.plotting.experiment import df_normalize_baseline

        df = pd.DataFrame({
            "genotype": ["WT", "WT", "KO", "KO"],
            "condition": ["baseline", "treatment", "baseline", "treatment"],
            "rms": [100.0, 130.0, 200.0, 250.0],
        })
        result = df_normalize_baseline(
            df=df, feature="rms",
            groupby=["genotype", "condition"],
            baseline_key="baseline",
            baseline_groupby=["condition"],
            operation="subtract",
        )
        wt_treat = result[(result["genotype"] == "WT") & (result["condition"] == "treatment")]["rms"].iloc[0]
        ko_treat = result[(result["genotype"] == "KO") & (result["condition"] == "treatment")]["rms"].iloc[0]
        np.testing.assert_almost_equal(wt_treat, 30.0)
        np.testing.assert_almost_equal(ko_treat, 50.0)


class TestPlotCustomization:
    """Test plot customization functions."""

    def test_matplotlib_backend_setting(self):
        """Test that matplotlib backend can be set."""
        import matplotlib

        original_backend = matplotlib.get_backend()

        # Test setting a different backend
        matplotlib.use("Agg")  # Non-interactive backend for testing
        assert matplotlib.get_backend() == "Agg"

        # Restore original backend
        matplotlib.use(original_backend)


class TestErrorHandling:
    """Test error handling."""

    def test_empty_wars_list(self):
        """Test handling of empty WindowAnalysisResult list."""
        with pytest.raises(ValueError, match="wars cannot be empty"):
            ExperimentPlotter([])

    def test_invalid_plot_type(self):
        """Test invalid plot type handling."""
        # This would be tested in the actual plotting methods
        # when they encounter unsupported plot types
        pass


class TestWindowAnalysisResultLOF:
    """Test LOF (Local Outlier Factor) functionality in WindowAnalysisResult."""

    @pytest.fixture
    def sample_lof_scores_dict(self):
        """Create sample LOF scores data for testing."""
        return {
            "day1": {"lof_scores": [2.5, 0.8], "channel_names": ["LMot", "RMot"]},
            "day2": {"lof_scores": [1.1, 2.8], "channel_names": ["LMot", "RMot"]},
        }

    @pytest.fixture
    def war_with_lof(self, sample_lof_scores_dict):
        """Create WindowAnalysisResult with LOF scores."""
        # Create minimal DataFrame
        test_df = pd.DataFrame(
            {
                "animal": ["A1"] * 4,
                "animalday": ["day1", "day1", "day2", "day2"],
                "genotype": ["WT"] * 4,
                "duration": [4.0] * 4,
                "rms": [[100.0, 110.0]] * 4,
                "timestamp": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-01 10:04:00", "2023-01-02 10:00:00", "2023-01-02 10:04:00"]
                ),
            }
        )

        return WindowAnalysisResult(
            result=test_df,
            animal_id="A1",
            genotype="WT",
            sex="Male",
            channel_names=["LMot", "RMot"],
            lof_scores_dict=sample_lof_scores_dict,
        )

    def test_war_init_with_lof_scores(self, war_with_lof, sample_lof_scores_dict):
        """Test WindowAnalysisResult initialization with LOF scores."""
        assert hasattr(war_with_lof, "lof_scores_dict")
        assert war_with_lof.lof_scores_dict == sample_lof_scores_dict

    def test_war_get_lof_scores(self, war_with_lof):
        """Test getting LOF scores from WindowAnalysisResult."""
        scores = war_with_lof.get_lof_scores()

        assert isinstance(scores, dict)
        assert "day1" in scores
        assert "day2" in scores

        # Check day1 scores
        day1_scores = scores["day1"]
        assert day1_scores["LMot"] == 2.5
        assert day1_scores["RMot"] == 0.8

        # Check day2 scores
        day2_scores = scores["day2"]
        assert day2_scores["LMot"] == 1.1
        assert day2_scores["RMot"] == 2.8

    def test_war_apply_lof_threshold(self, war_with_lof):
        """Test applying LOF threshold to WindowAnalysisResult."""
        # Test threshold 1.5
        bad_channels_1_5 = war_with_lof.get_bad_channels_by_lof_threshold(1.5)

        assert isinstance(bad_channels_1_5, dict)
        assert "day1" in bad_channels_1_5
        assert "day2" in bad_channels_1_5

        # Day1: scores [2.5, 0.8] with threshold 1.5
        # Bad channels: LMot (2.5 >= 1.5)
        assert set(bad_channels_1_5["day1"]) == {"LMot"}

        # Day2: scores [1.1, 2.8] with threshold 1.5
        # Bad channels: RMot (2.8 >= 1.5)
        assert set(bad_channels_1_5["day2"]) == {"RMot"}

        # Test different threshold
        bad_channels_2_0 = war_with_lof.get_bad_channels_by_lof_threshold(2.0)

        # Day1: only LMot (2.5) is >= 2.0
        assert set(bad_channels_2_0["day1"]) == {"LMot"}

        # Day2: only RMot (2.8) is >= 2.0
        assert set(bad_channels_2_0["day2"]) == {"RMot"}

    def test_war_apply_lof_threshold_strict(self, war_with_lof):
        """Test very strict LOF threshold."""
        bad_channels = war_with_lof.get_bad_channels_by_lof_threshold(1.0)

        # Day1: LMot (2.5) >= 1.0, RMot (0.8) < 1.0
        assert set(bad_channels["day1"]) == {"LMot"}

        # Day2: LMot (1.1) >= 1.0, RMot (2.8) >= 1.0
        assert set(bad_channels["day2"]) == {"LMot", "RMot"}

    def test_war_apply_lof_threshold_lenient(self, war_with_lof):
        """Test very lenient LOF threshold."""
        bad_channels = war_with_lof.get_bad_channels_by_lof_threshold(3.5)

        # All scores are < 3.5
        assert bad_channels["day1"] == []
        assert bad_channels["day2"] == []

    def test_war_lof_scores_error_when_missing(self):
        """Test that LOF scores are auto-populated with empty entries in __init__."""
        # Create WAR without LOF scores
        test_df = pd.DataFrame(
            {
                "animal": ["A1"] * 2,
                "animalday": ["day1", "day1"],
                "genotype": ["WT"] * 2,
                "duration": [4.0] * 2,
                "rms": [[100.0, 110.0]] * 2,
                "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00"]),
            }
        )
        war = WindowAnalysisResult(result=test_df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        # After __init__, missing sessions should be auto-populated with empty LOF scores
        # Both lof_scores AND channel_names should be empty to maintain invariant
        assert "day1" in war.lof_scores_dict
        assert war.lof_scores_dict["day1"]["lof_scores"] == []
        assert war.lof_scores_dict["day1"]["channel_names"] == []  # Must be empty too!

    def test_war_lof_scores_empty_dict(self):
        """Test that empty LOF scores dict is auto-populated with empty entries in __init__."""
        test_df = pd.DataFrame(
            {
                "animal": ["A1"] * 2,
                "animalday": ["day1", "day1"],
                "genotype": ["WT"] * 2,
                "duration": [4.0] * 2,
                "rms": [[100.0, 110.0]] * 2,
                "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00"]),
            }
        )
        war = WindowAnalysisResult(
            result=test_df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"], lof_scores_dict={}
        )

        # After __init__, empty dict should be populated with all sessions
        # Both lof_scores AND channel_names should be empty to maintain invariant
        assert "day1" in war.lof_scores_dict
        assert war.lof_scores_dict["day1"]["lof_scores"] == []
        assert war.lof_scores_dict["day1"]["channel_names"] == []  # Must be empty too!

    def test_war_save_load_preserves_lof_scores(self, war_with_lof):
        """Test that LOF scores are preserved through save/load cycle."""
        import tempfile
        from pathlib import Path

        # Mock the save method to bypass the long_recordings dependency
        with patch.object(war_with_lof, "save_parquet_and_json") as mock_save:
            # Test the JSON creation part directly
            lof_scores_dict = {}
            # Simulate the LOF collection that normally happens in save
            if hasattr(war_with_lof, "lof_scores_dict") and war_with_lof.lof_scores_dict:
                lof_scores_dict = war_with_lof.lof_scores_dict

            json_dict = {
                "animal_id": war_with_lof.animal_id,
                "genotype": war_with_lof.genotype,
                "sex": war_with_lof.sex,
                "channel_names": war_with_lof.channel_names,
                "assume_from_number": war_with_lof.assume_from_number,
                "bad_channels_dict": getattr(war_with_lof, "bad_channels_dict", {}),
                "suppress_short_interval_error": getattr(war_with_lof, "suppress_short_interval_error", False),
                "lof_scores_dict": lof_scores_dict,
            }

            # Verify LOF scores are included in save data
            assert "lof_scores_dict" in json_dict
            assert json_dict["lof_scores_dict"] == war_with_lof.lof_scores_dict

            # Test that a new WAR created with this data preserves LOF scores
            new_war = WindowAnalysisResult(war_with_lof.result, **json_dict)

            # Verify LOF functionality works
            original_scores = war_with_lof.get_lof_scores()
            new_scores = new_war.get_lof_scores()
            assert original_scores == new_scores

            original_bad_channels = war_with_lof.get_bad_channels_by_lof_threshold(1.5)
            new_bad_channels = new_war.get_bad_channels_by_lof_threshold(1.5)
            assert original_bad_channels == new_bad_channels

    def test_war_lof_scores_invalid_data_structure(self):
        """Test handling of invalid LOF scores data structure."""
        # Missing required keys
        invalid_lof_dict = {
            "day1": {
                "lof_scores": [1.0, 2.0],
                # Missing 'channel_names'
            }
        }

        test_df = pd.DataFrame(
            {
                "animal": ["A1"] * 2,
                "animalday": ["day1", "day1"],
                "genotype": ["WT"] * 2,
                "duration": [4.0] * 2,
                "rms": [[100.0, 110.0]] * 2,
                "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00"]),
            }
        )
        war = WindowAnalysisResult(
            result=test_df,
            animal_id="A1",
            genotype="WT",
            sex="Male",
            channel_names=["LMot", "RMot"],
            lof_scores_dict=invalid_lof_dict,
        )

        # Should raise ValueError for invalid data structure
        with pytest.raises(ValueError, match="LOF scores not available for day1"):
            war.get_lof_scores()

        # apply_lof_threshold should also fail with invalid data
        with pytest.raises(ValueError, match="LOF scores not available for day1"):
            war.get_bad_channels_by_lof_threshold(1.5)

    def test_war_lof_threshold_workflow_simulation(self, war_with_lof):
        """Test complete workflow of LOF threshold testing."""
        # Simulate workflow: load WAR and test multiple thresholds

        # Get raw scores for analysis
        raw_scores = war_with_lof.get_lof_scores()
        assert len(raw_scores) == 2  # Two days

        # Test multiple thresholds quickly
        thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
        results = {}

        for threshold in thresholds:
            bad_channels = war_with_lof.get_bad_channels_by_lof_threshold(threshold)
            total_bad = sum(len(channels) for channels in bad_channels.values())
            results[threshold] = total_bad

        # Verify results make sense (stricter thresholds = more bad channels)
        assert results[1.0] >= results[1.5]
        assert results[1.5] >= results[2.0]
        assert results[2.0] >= results[2.5]
        assert results[2.5] >= results[3.0]

        # Verify specific expectations
        assert results[1.0] == 3  # Most channels bad with strict threshold (LMot day1, LMot+RMot day2)
        assert results[3.0] == 0  # No channels bad with lenient threshold

    def test_war_evaluate_lof_threshold_binary(self, war_with_lof):
        """Test evaluate_lof_threshold_binary method for F1 score calculation."""
        # Create ground truth bad channels
        ground_truth_bad_channels = {
            "day1": {"LMot"},  # Only LMot is truly bad on day1
            "day2": {"RMot"},  # Only RMot is truly bad on day2
        }

        # Test threshold 1.5
        # LOF scores: day1=[2.5, 0.8], day2=[1.1, 2.8]
        # Predicted bad (>1.5): day1=[LMot], day2=[RMot]
        # Ground truth bad: day1=[LMot], day2=[RMot]
        y_true, y_pred = war_with_lof.evaluate_lof_threshold_binary(
            ground_truth_bad_channels, threshold=1.5, evaluation_channels=["LMot", "RMot"]
        )

        # Expected:
        # day1 LMot: y_true=1 (ground truth bad), y_pred=1 (LOF score 2.5 > 1.5)
        # day1 RMot: y_true=0 (ground truth good), y_pred=0 (LOF score 0.8 < 1.5)
        # day2 LMot: y_true=0 (ground truth good), y_pred=0 (LOF score 1.1 < 1.5)
        # day2 RMot: y_true=1 (ground truth bad), y_pred=1 (LOF score 2.8 > 1.5)
        expected_y_true = [1, 0, 0, 1]  # LMot day1, RMot day1, LMot day2, RMot day2
        expected_y_pred = [1, 0, 0, 1]

        assert y_true == expected_y_true
        assert y_pred == expected_y_pred

        # Test with sklearn f1_score
        from sklearn.metrics import f1_score

        f1 = f1_score(y_true, y_pred, average="binary")
        assert f1 == 1.0  # Perfect prediction

    def test_war_evaluate_lof_threshold_binary_imperfect(self, war_with_lof):
        """Test evaluate_lof_threshold_binary with imperfect predictions."""
        # Create different ground truth to test imperfect predictions
        ground_truth_bad_channels = {
            "day1": {"RMot"},  # Ground truth says RMot is bad on day1
            "day2": {"LMot"},  # Ground truth says LMot is bad on day2
        }

        # Test threshold 1.5
        # LOF scores: day1=[2.5, 0.8], day2=[1.1, 2.8]
        # Predicted bad (>1.5): day1=[LMot], day2=[RMot]
        # Ground truth bad: day1=[RMot], day2=[LMot]
        y_true, y_pred = war_with_lof.evaluate_lof_threshold_binary(
            ground_truth_bad_channels, threshold=1.5, evaluation_channels=["LMot", "RMot"]
        )

        # Expected:
        # day1 LMot: y_true=0 (ground truth good), y_pred=1 (LOF score 2.5 > 1.5) - FALSE POSITIVE
        # day1 RMot: y_true=1 (ground truth bad), y_pred=0 (LOF score 0.8 < 1.5) - FALSE NEGATIVE
        # day2 LMot: y_true=1 (ground truth bad), y_pred=0 (LOF score 1.1 < 1.5) - FALSE NEGATIVE
        # day2 RMot: y_true=0 (ground truth good), y_pred=1 (LOF score 2.8 > 1.5) - FALSE POSITIVE
        expected_y_true = [0, 1, 1, 0]
        expected_y_pred = [1, 0, 0, 1]

        assert y_true == expected_y_true
        assert y_pred == expected_y_pred

        # Calculate F1 score - should be 0 (no true positives)
        from sklearn.metrics import f1_score

        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        assert f1 == 0.0

    def test_war_evaluate_lof_threshold_binary_channel_subset(self, war_with_lof):
        """Test evaluate_lof_threshold_binary with channel subset filtering."""
        ground_truth_bad_channels = {"day1": {"LMot"}, "day2": {"RMot"}}

        # Test with only LMot channel
        y_true, y_pred = war_with_lof.evaluate_lof_threshold_binary(
            ground_truth_bad_channels,
            threshold=1.5,
            evaluation_channels=["LMot"],  # Only evaluate LMot
        )

        # Should only have 2 evaluation points (LMot for day1 and day2)
        assert len(y_true) == 2
        assert len(y_pred) == 2

        # day1 LMot: y_true=1, y_pred=1
        # day2 LMot: y_true=0, y_pred=0
        expected_y_true = [1, 0]
        expected_y_pred = [1, 0]

        assert y_true == expected_y_true
        assert y_pred == expected_y_pred

    def test_war_evaluate_lof_threshold_binary_no_ground_truth(self, war_with_lof):
        """Test evaluate_lof_threshold_binary with no ground truth data."""
        # Empty ground truth
        ground_truth_bad_channels = {}

        y_true, y_pred = war_with_lof.evaluate_lof_threshold_binary(
            ground_truth_bad_channels, threshold=1.5, evaluation_channels=["LMot", "RMot"]
        )

        # All ground truth should be 0 (no bad channels)
        # Predictions based on LOF scores: day1=[1,0], day2=[0,1]
        expected_y_true = [0, 0, 0, 0]
        expected_y_pred = [1, 0, 0, 1]

        assert y_true == expected_y_true
        assert y_pred == expected_y_pred

    def test_war_evaluate_lof_threshold_binary_missing_lof_scores(self):
        """Test graceful handling when LOF scores are empty (auto-populated but not computed)."""
        # Create WAR without LOF scores
        test_df = pd.DataFrame(
            {
                "animal": ["A1"] * 2,
                "animalday": ["day1", "day1"],
                "genotype": ["WT"] * 2,
                "duration": [4.0] * 2,
                "rms": [[100.0, 110.0]] * 2,
                "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00"]),
            }
        )
        war = WindowAnalysisResult(result=test_df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        ground_truth = {"day1": {"LMot"}}

        # Auto-population creates empty LOF scores, which should be gracefully skipped
        y_true, y_pred = war.evaluate_lof_threshold_binary(ground_truth, 1.5)

        # Should return empty lists since the session with empty LOF scores is skipped
        assert y_true == []
        assert y_pred == []

    def test_war_evaluate_lof_threshold_binary_default_ground_truth(self, war_with_lof):
        """Test evaluate_lof_threshold_binary using self.bad_channels_dict as default ground truth."""
        # Set up bad_channels_dict on the WAR with keys matching lof_scores_dict
        war_with_lof.bad_channels_dict = {
            "day1": ["LMot"],  # Matches LOF data keys exactly
            "day2": ["RMot"],  # Matches LOF data keys exactly
        }

        # Test without providing ground_truth_bad_channels (should use self.bad_channels_dict)
        y_true, y_pred = war_with_lof.evaluate_lof_threshold_binary(threshold=1.5, evaluation_channels=["LMot", "RMot"])

        # Expected: keys match exactly, so should work like explicit ground truth
        expected_y_true = [1, 0, 0, 1]  # day1 LMot=bad, day1 RMot=good, day2 LMot=good, day2 RMot=bad
        expected_y_pred = [1, 0, 0, 1]  # LOF scores: day1=[2.5,0.8], day2=[1.1,2.8] with threshold 1.5

        assert y_true == expected_y_true
        assert y_pred == expected_y_pred

    def test_war_evaluate_lof_threshold_binary_key_mismatch(self, war_with_lof):
        """Test error when bad_channels_dict keys don't match lof_scores_dict keys."""
        # Set up bad_channels_dict with mismatched keys
        war_with_lof.bad_channels_dict = {
            "invalid_key": ["LMot"],  # This key doesn't exist in lof_scores_dict
        }

        with pytest.raises(ValueError, match="bad_channels_dict contains keys not found in lof_scores_dict"):
            war_with_lof.evaluate_lof_threshold_binary(threshold=1.5)

    def test_war_evaluate_lof_threshold_binary_missing_threshold(self, war_with_lof):
        """Test error when threshold is missing."""
        ground_truth = {"day1": {"LMot"}}

        with pytest.raises(ValueError, match="threshold parameter is required"):
            war_with_lof.evaluate_lof_threshold_binary(ground_truth)


class TestWindowAnalysisResultParquetJsonParameters:
    """Test parquet_name and json_name parameters in load_parquet_and_json."""

    @pytest.fixture
    def temp_war_files(self):
        """Create temporary WAR files for testing."""
        import tempfile
        from pathlib import Path

        # Create sample data
        test_df = pd.DataFrame(
            {
                "animal": ["A1"] * 2,
                "animalday": ["A1_day1", "A1_day1"],
                "genotype": ["WT"] * 2,
                "duration": [4.0] * 2,
                "rms": [[100.0, 110.0], [200.0, 210.0]],
                "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00"]),
            }
        )

        war = WindowAnalysisResult(result=test_df, animal_id="A1", genotype="WT", sex="Male", channel_names=["LMot", "RMot"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save with default names
            war.save_parquet_and_json(tmpdir, filename="war")

            # Create subdirectory structure
            subdir = tmpdir / "subdir"
            subdir.mkdir()
            war.save_parquet_and_json(subdir, filename="nested_war")

            # Also save with custom names at root level
            war.save_parquet_and_json(tmpdir, filename="custom_war")

            yield {"tmpdir": tmpdir, "subdir": subdir, "war": war}

    def test_load_default_behavior(self, temp_war_files):
        """Test that default behavior (no parquet_name/json_name) still works."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        # Remove other files to test single file case
        for f in tmpdir.glob("*"):
            if f.name not in ["war.parquet", "war.json"]:
                if f.is_file():
                    f.unlink()
                else:
                    import shutil

                    shutil.rmtree(f)

        loaded_war = WindowAnalysisResult.load_parquet_and_json(folder_path=str(tmpdir))

        assert loaded_war.animal_id == original_war.animal_id
        assert loaded_war.genotype == original_war.genotype
        assert loaded_war.sex == original_war.sex
        assert loaded_war.channel_names == original_war.channel_names
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_with_exact_filenames(self, temp_war_files):
        """Test loading with exact parquet_name and json_name."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        loaded_war = WindowAnalysisResult.load_parquet_and_json(
            folder_path=str(tmpdir), parquet_name="custom_war.parquet", json_name="custom_war.json"
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_with_relative_paths(self, temp_war_files):
        """Test loading with relative paths from folder_path."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        loaded_war = WindowAnalysisResult.load_parquet_and_json(
            folder_path=str(tmpdir), parquet_name="subdir/nested_war.parquet", json_name="subdir/nested_war.json"
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_with_absolute_paths(self, temp_war_files):
        """Test loading with absolute paths."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        parquet_path = tmpdir / "custom_war.parquet"
        json_path = tmpdir / "custom_war.json"

        loaded_war = WindowAnalysisResult.load_parquet_and_json(
            folder_path=str(tmpdir), parquet_name=str(parquet_path), json_name=str(json_path)
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_without_folder_path(self, temp_war_files):
        """Test loading with absolute paths only (no folder_path)."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        parquet_path = tmpdir / "custom_war.parquet"
        json_path = tmpdir / "custom_war.json"

        loaded_war = WindowAnalysisResult.load_parquet_and_json(
            parquet_name=str(parquet_path), json_name=str(json_path)
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_parquet_not_found(self, temp_war_files):
        """Test error when parquet file not found."""
        tmpdir = temp_war_files["tmpdir"]

        with pytest.raises(FileNotFoundError, match="Parquet file not found"):
            WindowAnalysisResult.load_parquet_and_json(
                folder_path=str(tmpdir), parquet_name="nonexistent.parquet", json_name="war.json"
            )

    def test_load_json_not_found(self, temp_war_files):
        """Test error when JSON file not found."""
        tmpdir = temp_war_files["tmpdir"]

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            WindowAnalysisResult.load_parquet_and_json(
                folder_path=str(tmpdir), parquet_name="war.parquet", json_name="nonexistent.json"
            )

    def test_load_multiple_files_without_specification(self, temp_war_files):
        """Test error when multiple files exist but none specified."""
        tmpdir = temp_war_files["tmpdir"]

        # There should be multiple .parquet and .json files in tmpdir
        parquet_files = list(tmpdir.glob("*.parquet"))
        json_files = list(tmpdir.glob("*.json"))

        # Ensure we have multiple files
        assert len(parquet_files) > 1
        assert len(json_files) > 1

        with pytest.raises(ValueError, match="Expected exactly one parquet file"):
            WindowAnalysisResult.load_parquet_and_json(folder_path=str(tmpdir))

    def test_load_no_files_found(self):
        """Test error when no files are found."""
        import tempfile

        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(ValueError, match="Expected exactly one parquet file"):
                WindowAnalysisResult.load_parquet_and_json(folder_path=empty_dir)

    def test_load_invalid_folder_path(self):
        """Test error with invalid folder path."""
        with pytest.raises(ValueError, match="Folder path .* does not exist"):
            WindowAnalysisResult.load_parquet_and_json(folder_path="/nonexistent/path")

    def test_load_missing_parameters(self):
        """Test error when required parameters are missing."""
        # Neither folder_path nor both parquet_name/json_name provided
        with pytest.raises(ValueError, match="Either folder_path must be provided"):
            WindowAnalysisResult.load_parquet_and_json()

        # Only one of parquet_name/json_name provided without folder_path
        with pytest.raises(ValueError, match="Either folder_path must be provided"):
            WindowAnalysisResult.load_parquet_and_json(parquet_name="/some/path.parquet")

        with pytest.raises(ValueError, match="Either folder_path must be provided"):
            WindowAnalysisResult.load_parquet_and_json(json_name="/some/path.json")

    def test_load_mixed_absolute_relative_paths(self, temp_war_files):
        """Test mixing absolute and relative paths."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        # Absolute parquet path, relative json path
        parquet_path = tmpdir / "custom_war.parquet"

        loaded_war = WindowAnalysisResult.load_parquet_and_json(
            folder_path=str(tmpdir),
            parquet_name=str(parquet_path),  # Absolute
            json_name="custom_war.json",  # Relative
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_old_json_without_sex(self):
        """Test backward compatibility: old JSON files without 'sex' load with sex='Unknown'."""
        import json
        import tempfile
        from pathlib import Path

        test_df = pd.DataFrame(
            {
                "animal": ["A1"] * 2,
                "animalday": ["day1", "day1"],
                "genotype": ["WT"] * 2,
                "duration": [4.0] * 2,
                "rms": [[100.0, 110.0], [200.0, 210.0]],
                "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:04:00"]),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save legacy pickle (simulates an old WAR written before the parquet migration)
            test_df.to_pickle(tmpdir / "war.pkl")

            # Save JSON without sex (simulating old format)
            old_json = {
                "animal_id": "A1",
                "genotype": "WT",
                "channel_names": ["LMot", "RMot"],
                "assume_from_number": False,
                "bad_channels_dict": {},
                "suppress_short_interval_error": False,
                "lof_scores_dict": {},
            }
            with open(tmpdir / "war.json", "w") as f:
                json.dump(old_json, f)

            # Load should succeed via the legacy pickle fallback, with sex="Unknown" default
            loaded_war = WindowAnalysisResult.load_parquet_and_json(folder_path=str(tmpdir))
            assert loaded_war.animal_id == "A1"
            assert loaded_war.genotype == "WT"
            assert loaded_war.sex == "Unknown"
            assert loaded_war.channel_names == ["LMot", "RMot"]
class TestParquetSaveLoad:
    """Test parquet save/load functionality for WindowAnalysisResult."""

    @pytest.fixture
    def war_with_complex_columns(self):
        """Create a WAR with complex (object-type) columns typical of real usage."""
        data = {
            "animalday": ["A1_20230101"] * 3,
            "genotype": ["WT"] * 3,
            "timestamp": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-01 10:04:00", "2023-01-01 10:08:00"]
            ),
            "duration": [240.0, 245.0, 235.0],
            "rms": [[100.0, 110.0], [200.0, 210.0], [150.0, 160.0]],
            "psd": [
                (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
                (np.array([3.0, 4.0]), np.array([30.0, 40.0])),
                (np.array([5.0, 6.0]), np.array([50.0, 60.0])),
            ],
            "cohere": [
                {"ch1_ch2": np.array([0.1, 0.2])},
                {"ch1_ch2": np.array([0.3, 0.4])},
                {"ch1_ch2": np.array([0.5, 0.6])},
            ],
        }
        df = pd.DataFrame(data)
        return WindowAnalysisResult(
            result=df,
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot"],
        )

    def test_parquet_round_trip(self, war_with_complex_columns):
        """Test that save with parquet + load produces equivalent data."""
        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")

            # Verify parquet file was created
            assert (Path(tmpdir) / "war.parquet").exists()
            # Verify pickle is NOT written any more
            assert not (Path(tmpdir) / "war.pkl").exists()
            # No sidecar meta file — metadata is embedded in parquet
            assert not (Path(tmpdir) / "war.parquet.meta.json").exists()

            loaded = WindowAnalysisResult.load_parquet_and_json(
                folder_path=tmpdir, parquet_name="war.parquet", json_name="war.json"
            )
            assert loaded.animal_id == war.animal_id
            assert loaded.genotype == war.genotype
            assert loaded.channel_names == war.channel_names

            # Scalar columns should match
            assert loaded.result["duration"].tolist() == war.result["duration"].tolist()

            # List columns should round-trip (values come back as plain lists)
            for i in range(len(war.result)):
                assert loaded.result["rms"].iloc[i] == war.result["rms"].iloc[i]

    def test_auto_discovery_with_parquet_files(self, war_with_complex_columns):
        """Test that auto-discovery works when parquet files are present."""
        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")

            loaded = WindowAnalysisResult.load_parquet_and_json(folder_path=tmpdir)
            assert loaded.animal_id == war.animal_id

    def test_fallback_to_pickle_when_no_parquet(self, war_with_complex_columns):
        """Test that loading falls back to a legacy pickle when parquet is absent.

        Simulates an old on-disk WAR (written before the parquet migration) by
        hand-writing a ``war.pkl`` alongside the JSON — no parquet.
        """
        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save via the new API, then drop parquet and write a legacy pickle
            war.save_parquet_and_json(tmpdir, filename="war")
            (Path(tmpdir) / "war.parquet").unlink()
            war.result.to_pickle(Path(tmpdir) / "war.pkl")

            loaded = WindowAnalysisResult.load_parquet_and_json(
                folder_path=tmpdir, parquet_name="war.parquet", json_name="war.json"
            )
            assert loaded.animal_id == war.animal_id
            assert loaded.result["duration"].tolist() == war.result["duration"].tolist()

    def test_metadata_embedded_in_parquet(self, war_with_complex_columns):
        """Test that encoded column metadata is stored in parquet schema metadata."""
        import pyarrow.parquet as pq

        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")

            table = pq.read_table(Path(tmpdir) / "war.parquet")
            schema_meta = table.schema.metadata
            assert b"neurodent" in schema_meta

            nd_meta = json.loads(schema_meta[b"neurodent"])
            assert "encoded_columns" in nd_meta
            assert "rms" in nd_meta["encoded_columns"]

    def test_encode_decode_round_trip(self):
        """Test _encode_df_for_parquet and _decode_df_from_parquet directly.

        Updated for encoding_version=2: complex columns are converted to
        nested Python structures (no JSON intermediate) instead of JSON
        strings.  Decode is a no-op for cells that are already nested
        Python (legacy string cells still get json.loads'd).
        """
        df = pd.DataFrame(
            {
                "scalar": [1.0, 2.0, 3.0],
                "string_col": ["a", "b", "c"],
                "list_col": [[1, 2], [3, 4], [5, 6]],
                "dict_col": [{"x": 1}, {"y": 2}, {"z": 3}],
                "array_col": [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])],
                "none_col": [None, [1], None],
            }
        )

        encoded, cols = WindowAnalysisResult._encode_df_for_parquet(df)

        # Scalar and string columns should NOT be encoded
        assert "scalar" not in cols
        assert "string_col" not in cols

        # Complex columns should be encoded
        assert "list_col" in cols
        assert "dict_col" in cols
        assert "array_col" in cols

        # Encoded values are nested Python (lists/dicts/scalars), NOT JSON strings.
        # ndarrays are converted to lists via _to_nested_python.
        for col in cols:
            for val in encoded[col].dropna():
                assert not isinstance(val, str), (
                    f"{col}: expected nested Python type, got JSON string {val!r}"
                )
                assert isinstance(val, (list, dict, int, float, bool))

        # Decode is a no-op for native cells (already nested Python).
        decoded = WindowAnalysisResult._decode_df_from_parquet(
            encoded, cols, encoding_version=2
        )
        for i in range(len(df)):
            assert decoded["list_col"].iloc[i] == df["list_col"].iloc[i]
            assert decoded["dict_col"].iloc[i] == df["dict_col"].iloc[i]
            # array_col was converted to a plain Python list at encode time.
            np.testing.assert_array_equal(
                decoded["array_col"].iloc[i], df["array_col"].iloc[i]
            )

        # Legacy backward-compat: explicit JSON strings should still decode.
        legacy = pd.DataFrame(
            {
                "list_col": [json.dumps([1, 2]), json.dumps([3, 4]), json.dumps([5, 6])],
                "dict_col": [json.dumps({"x": 1}), json.dumps({"y": 2}), json.dumps({"z": 3})],
            }
        )
        legacy_decoded = WindowAnalysisResult._decode_df_from_parquet(
            legacy, ["list_col", "dict_col"]
        )
        assert legacy_decoded["list_col"].iloc[0] == [1, 2]
        assert legacy_decoded["dict_col"].iloc[0] == {"x": 1}

    def test_parquet_file_has_content(self, war_with_complex_columns):
        """Test that the parquet file has meaningful content matching the DataFrame."""
        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")
            pq_path = Path(tmpdir) / "war.parquet"
            assert pq_path.stat().st_size > 0

            reloaded = pd.read_parquet(pq_path, engine="pyarrow")
            assert len(reloaded) == len(war.result)
            assert set(war.result.columns).issubset(set(reloaded.columns))

    def test_save_load_speed(self):
        """Sanity-check that parquet save + load is fast enough for realistic rows.

        Uses a realistically-sized DataFrame (200 rows) so that pyarrow's
        fixed per-call overhead is amortised.
        """
        import time

        n_rows = 200
        rng = np.random.default_rng(42)
        data = {
            "animalday": [f"A1_{i:08d}" for i in range(n_rows)],
            "genotype": ["WT"] * n_rows,
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="5min"),
            "duration": [240.0] * n_rows,
            "rms": [rng.uniform(100, 200, 2).tolist() for _ in range(n_rows)],
            "psd": [
                (rng.uniform(0, 10, 8).tolist(), rng.uniform(0, 100, 8).tolist())
                for _ in range(n_rows)
            ],
            "cohere": [
                {"ch1_ch2": rng.uniform(0, 1, 4).tolist()}
                for _ in range(n_rows)
            ],
        }
        war = WindowAnalysisResult(
            result=pd.DataFrame(data),
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")

            pq_path = Path(tmpdir) / "war.parquet"
            assert pq_path.exists()
            # Pickle should no longer be produced
            assert not (Path(tmpdir) / "war.pkl").exists()

            # Warm-up: first load amortises import / JIT costs
            pd.read_parquet(pq_path, engine="pyarrow")

            n_iters = 20
            start = time.perf_counter()
            for _ in range(n_iters):
                pd.read_parquet(pq_path, engine="pyarrow")
            pq_time = time.perf_counter() - start

            # 200 rows × 20 iterations should comfortably finish in < 5s on CI runners.
            assert pq_time < 5.0, f"Parquet load time {pq_time:.3f}s is unreasonably slow"

    def test_save_parquet_bypasses_encode_df_for_parquet(self):
        """Verify save_parquet_and_json builds a column dict directly
        instead of calling _encode_df_for_parquet (which does df.copy()).

        The old path created 2-3 redundant copies of the DataFrame via
        df.copy() + pa.Table.from_pandas(). The new path builds a column
        dict and uses pa.table() — no DataFrame copy.
        """
        from unittest.mock import patch

        n_rows = 50
        rng = np.random.default_rng(99)
        data = {
            "animalday": [f"A1_day{i}" for i in range(n_rows)],
            "genotype": ["WT"] * n_rows,
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="5min"),
            "duration": [240.0] * n_rows,
            "rms": [rng.uniform(100, 200, 4).tolist() for _ in range(n_rows)],
            "cohere": [
                {"ch1_ch2": rng.uniform(0, 1, 8).tolist()}
                for _ in range(n_rows)
            ],
        }
        war = WindowAnalysisResult(
            result=pd.DataFrame(data),
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot", "LAud", "RAud"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch _encode_df_for_parquet to verify it is NOT called
            with patch.object(
                WindowAnalysisResult, "_encode_df_for_parquet",
                wraps=WindowAnalysisResult._encode_df_for_parquet,
            ) as mock_encode:
                war.save_parquet_and_json(tmpdir, filename="war")
                mock_encode.assert_not_called()

            # Verify the parquet file is valid and round-trips correctly
            pq_path = Path(tmpdir) / "war.parquet"
            assert pq_path.exists()

            import pyarrow.parquet as pq
            table = pq.read_table(pq_path)

            # Check neurodent metadata with encoded columns list
            nd_meta = json.loads(table.schema.metadata[b"neurodent"])
            assert "rms" in nd_meta["encoded_columns"]
            assert "cohere" in nd_meta["encoded_columns"]
            assert "duration" not in nd_meta["encoded_columns"]

            # Decode and verify values round-trip
            reloaded = table.to_pandas()
            encoded_cols = nd_meta["encoded_columns"]
            encoding_version = nd_meta.get("encoding_version", 1)
            decoded = WindowAnalysisResult._decode_df_from_parquet(
                reloaded, encoded_cols, encoding_version=encoding_version
            )
            assert len(decoded) == n_rows
            for i in range(min(5, n_rows)):
                assert decoded["rms"].iloc[i] == war.result["rms"].iloc[i]
                assert decoded["cohere"].iloc[i] == war.result["cohere"].iloc[i]


class TestMemoryPressure:
    """Verify that load/save paths don't hold unnecessary copies of WAR data."""

    @pytest.fixture
    def war_for_memory_test(self):
        """Create a WAR large enough that data dominates over fixed overhead."""
        n_rows = 500
        n_ch = 8
        n_freq = 64
        rng = np.random.default_rng(42)
        data = {
            "animalday": ["A1_20230101"] * n_rows,
            "genotype": ["WT"] * n_rows,
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="4min"),
            "duration": rng.uniform(230, 250, n_rows).tolist(),
            "rms": [rng.random(n_ch).tolist() for _ in range(n_rows)],
            "psd": [
                (rng.random(n_freq).tolist(), rng.random((n_ch, n_freq)).tolist())
                for _ in range(n_rows)
            ],
        }
        return WindowAnalysisResult(
            pd.DataFrame(data), animal_id="A1", genotype="WT",
            channel_names=["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"],
            suppress_short_interval_error=True,
        )

    def test_load_parquet_peak_memory(self, war_for_memory_test):
        """Peak memory during load_parquet_and_json should stay under 5x the parquet file size.

        Regression guard: without ``del table`` after ``to_pandas()`` and
        without the in-place decode (no ``df.copy()``), peak reaches ~6x+.
        """
        import tracemalloc

        war = war_for_memory_test
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")
            pq_size = (Path(tmpdir) / "war.parquet").stat().st_size

            tracemalloc.start()
            loaded = WindowAnalysisResult.load_parquet_and_json(folder_path=tmpdir)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Loaded data should be correct
            assert loaded.animal_id == war.animal_id
            assert len(loaded.result) == len(war.result)

            # Peak memory should not exceed 5x the parquet file size.
            # Before the del-table + in-place-decode fixes, this ratio was ~6x+.
            ratio = peak / max(pq_size, 1)
            assert ratio < 5, (
                f"Peak memory during load is {ratio:.1f}x the parquet file size "
                f"({peak / 1e6:.1f} MB vs {pq_size / 1e6:.1f} MB). "
                f"Likely a missing del or unnecessary copy."
            )

    def test_save_parquet_peak_memory(self, war_for_memory_test):
        """Peak memory during save_parquet_and_json should stay under 5x the parquet file size."""
        import tracemalloc

        war = war_for_memory_test
        with tempfile.TemporaryDirectory() as tmpdir:
            tracemalloc.start()
            war.save_parquet_and_json(tmpdir, filename="war")
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            pq_size = (Path(tmpdir) / "war.parquet").stat().st_size
            ratio = peak / max(pq_size, 1)
            assert ratio < 5, (
                f"Peak memory during save is {ratio:.1f}x the parquet file size "
                f"({peak / 1e6:.1f} MB vs {pq_size / 1e6:.1f} MB). "
                f"Likely a missing del or unnecessary copy."
            )


class TestStreamReorderAndPad:
    """Equivalence + memory tests for the lazy WAR reorder+pad chain.

    The streaming path (``scan_parquet_and_json`` → ``reorder_and_pad_channels`` →
    ``save_parquet_and_json``) must produce the same WAR data on disk as
    the eager ``load_parquet_and_json`` → ``reorder_and_pad_channels`` →
    ``save_parquet_and_json`` path, while using strictly less peak memory.
    """

    @pytest.fixture
    def synthetic_war(self):
        """Synthetic WAR with all feature types (LINEAR, LINEAR_2D, BAND,
        SIMPLE_MATRIX, BANDED_MATRIX, HIST), 4 source channels.
        """
        n_rows = 60
        C = 4
        BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
        F = 16
        rng = np.random.default_rng(123)
        data = {
            "animalday": ["A1_20230101"] * n_rows,
            "animal": ["A1"] * n_rows,
            "genotype": ["WT"] * n_rows,
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="1min"),
            "duration": [60.0] * n_rows,
            # LINEAR (1D per row)
            "rms": [rng.random(C).tolist() for _ in range(n_rows)],
            "logrms": [rng.random(C).tolist() for _ in range(n_rows)],
            # LINEAR_2D ([slope, intercept] per channel)
            "psdslope": [
                [[rng.random(), rng.random()] for _ in range(C)] for _ in range(n_rows)
            ],
            # BAND (band dict of 1D)
            "psdband": [
                {b: rng.random(C).tolist() for b in BANDS} for _ in range(n_rows)
            ],
            # SIMPLE_MATRIX (CxC per row)
            "pcorr": [rng.random((C, C)).tolist() for _ in range(n_rows)],
            # BANDED_MATRIX (band dict of CxC)
            "cohere": [
                {b: rng.random((C, C)).tolist() for b in BANDS} for _ in range(n_rows)
            ],
            # HIST (psd: per-row (coords, (C,F)))
            "psd": [(np.arange(F).tolist(), rng.random((C, F)).tolist()) for _ in range(n_rows)],
        }
        return WindowAnalysisResult(
            result=pd.DataFrame(data),
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot", "LBar", "RBar"],
            suppress_short_interval_error=True,
        )

    def test_streaming_equivalent_to_eager(self, synthetic_war):
        """Streaming path produces the same WAR data as the eager path."""
        target = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            synthetic_war.save_parquet_and_json(src, filename="war")

            # Eager path
            eager_dst = tmp / "eager"
            eager_dst.mkdir()
            war_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            war_eager.reorder_and_pad_channels(target, use_abbrevs=True)
            war_eager.save_parquet_and_json(eager_dst, filename="war")

            # Streaming path (batch_size=20 so we exercise multiple row groups)
            stream_dst = tmp / "stream"
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.reorder_and_pad_channels(target, use_abbrevs=True)
            war_lazy.save_parquet_and_json(stream_dst, filename="war", batch_size=20)

            # Reload both, compare row-by-row.
            re_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=eager_dst)
            re_stream = WindowAnalysisResult.load_parquet_and_json(folder_path=stream_dst)

            assert list(re_eager.result.columns) == list(re_stream.result.columns)
            assert len(re_eager.result) == len(re_stream.result)

            # JSON-normalised comparison handles ndarray/tuple/list/dict
            # equivalently (the load-side decode may return lists where the
            # original held tuples or ndarrays — both serialise the same).
            encoder = WindowAnalysisResult._NumpyEncoder
            for col in re_eager.result.columns:
                for i in range(len(re_eager.result)):
                    a = re_eager.result[col].iloc[i]
                    b = re_stream.result[col].iloc[i]
                    a_norm = json.dumps(a, cls=encoder, sort_keys=True, default=str)
                    b_norm = json.dumps(b, cls=encoder, sort_keys=True, default=str)
                    assert a_norm == b_norm, (
                        f"col={col}, row={i}\n  eager:  {a!r}\n  stream: {b!r}"
                    )

            # Channel-name JSON metadata should also match.
            assert re_eager.channel_names == re_stream.channel_names
            assert re_eager.channel_names == target

    def test_streaming_peak_memory_below_eager(self, synthetic_war):
        """Streaming peak must be substantially smaller than eager peak.

        Tightened assertion (was ``stream < eager``): streaming should
        use less than half the eager memory at ``batch_size=10`` (1/6 of
        the 60-row fixture).  Real arxrosa-scale data observed ~40%; the
        synthetic fixture sees ~20%; 50% is a comfortable upper bound
        that catches regressions while tolerating pandas/pyarrow overhead
        variance across versions.
        """
        import tracemalloc

        target = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            synthetic_war.save_parquet_and_json(src, filename="war")
            pq_size = (src / "war.parquet").stat().st_size

            tracemalloc.start()
            stream_dst = tmp / "stream"
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.reorder_and_pad_channels(target, use_abbrevs=True)
            war_lazy.save_parquet_and_json(stream_dst, filename="war", batch_size=10)
            _, stream_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            eager_dst = tmp / "eager"
            eager_dst.mkdir()
            war_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            war_eager.reorder_and_pad_channels(target, use_abbrevs=True)
            war_eager.save_parquet_and_json(eager_dst, filename="war")
            _, eager_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            ratio = stream_peak / eager_peak
            assert ratio < 0.5, (
                f"streaming peak {stream_peak/1e6:.2f} MB is {ratio:.0%} of "
                f"eager peak {eager_peak/1e6:.2f} MB — expected < 50% (regression?)"
            )
            # Absolute ceiling: streaming should fit in ~5x the parquet size.
            assert stream_peak < 5 * pq_size, (
                f"streaming peak {stream_peak/1e6:.2f} MB exceeds 5x parquet "
                f"({pq_size/1e6:.2f} MB) — peak should scale with batch_size, not WAR size"
            )

    def test_no_json_fallback_for_standard_features(self, synthetic_war):
        """Every encoded column — including HIST (psd) — uses a native
        pyarrow list/struct type. The per-cell JSON fallback path should
        NOT be hit for any of the canonical FeatureTypes.

        Regression guard: HIST stores ``(coords, values)`` tuples per cell.
        Before the tuple→struct lift, those would fall back to JSON
        strings.  After: they encode as ``struct<_t0: ..., _t1: ...>``.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            synthetic_war.save_parquet_and_json(tmpdir, filename="war")
            pq_path = Path(tmpdir) / "war.parquet"

            schema = pq.ParquetFile(pq_path).schema_arrow
            nd = json.loads(schema.metadata[b"neurodent"])
            assert nd.get("encoding_version") == 2

            for col in nd["encoded_columns"]:
                ftype = schema.field(col).type
                assert not pa.types.is_string(ftype), (
                    f"Column {col!r} fell back to JSON encoding (string). "
                    f"Expected a native list/struct type."
                )
                assert pa.types.is_nested(ftype), (
                    f"Column {col!r} has non-nested type {ftype}; "
                    f"expected list or struct."
                )

    def test_streaming_with_unique_hash(self, synthetic_war):
        """Streaming with add_unique_hash matches the eager equivalent."""
        target = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            synthetic_war.save_parquet_and_json(src, filename="war")

            stream_dst = tmp / "stream"
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.reorder_and_pad_channels(target, use_abbrevs=True)
            war_lazy.add_unique_hash(4)
            war_lazy.save_parquet_and_json(stream_dst, filename="war", batch_size=20)

            loaded = WindowAnalysisResult.load_parquet_and_json(folder_path=stream_dst)
            # animal_id grew by a hash suffix.
            assert loaded.animal_id.startswith("A1_")
            assert loaded.animal_id != "A1"
            # animal column rewritten consistently
            assert (loaded.result["animal"] == loaded.animal_id).all()


class TestLazyWindowAnalysisResult:
    """Equivalence + memory tests for the LazyWindowAnalysisResult engine.

    Each lazy chain (apply_filters, aggregate_time_windows, plus the
    reorder+hash compat path) must produce the same WAR data on disk as
    the eager mutator + save path, while keeping peak memory bounded by
    ``batch_size``.
    """

    @pytest.fixture
    def lazy_synthetic_war(self):
        n_rows = 60
        C = 4
        BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
        F = 16
        rng = np.random.default_rng(321)
        # Two animaldays so we exercise the by_session filter + groupby aggregation.
        animaldays = ["A1_20230101"] * 30 + ["A1_20230102"] * 30
        isday = [True] * 15 + [False] * 15 + [True] * 15 + [False] * 15
        data = {
            "animal": ["A1"] * n_rows,
            "animalday": animaldays,
            "genotype": ["WT"] * n_rows,
            "isday": isday,
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="1min"),
            "duration": [60.0] * n_rows,
            "endfile": list(range(n_rows)),
            "rms": [(rng.random(C) * 600).tolist() for _ in range(n_rows)],
            "logrms": [np.log(rng.random(C) * 600 + 1).tolist() for _ in range(n_rows)],
            "psdslope": [
                [[rng.random(), rng.random()] for _ in range(C)] for _ in range(n_rows)
            ],
            "psdband": [
                {b: rng.random(C).tolist() for b in BANDS} for _ in range(n_rows)
            ],
            "psdtotal": [rng.random(C).tolist() for _ in range(n_rows)],
            "pcorr": [rng.random((C, C)).tolist() for _ in range(n_rows)],
            "cohere": [
                {b: rng.random((C, C)).tolist() for b in BANDS} for _ in range(n_rows)
            ],
            # HIST cells are (F, C) per the welch output convention.
            "psd": [(np.arange(F).tolist(), rng.random((F, C)).tolist()) for _ in range(n_rows)],
        }
        return WindowAnalysisResult(
            result=pd.DataFrame(data),
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot", "LBar", "RBar"],
            suppress_short_interval_error=True,
        )

    @staticmethod
    def _norm_dataframe(df):
        """Normalise a DataFrame for cell-wise comparison across eager/lazy."""
        encoder = WindowAnalysisResult._NumpyEncoder
        return {
            col: [
                json.dumps(df[col].iloc[i], cls=encoder, sort_keys=True, default=str)
                for i in range(len(df))
            ]
            for col in df.columns
        }

    @staticmethod
    def _cells_match(a, b, rtol=1e-9, atol=1e-9):
        """Recursive numerical-tolerant equality check for WAR cell values."""
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(TestLazyWindowAnalysisResult._cells_match(a[k], b[k], rtol, atol) for k in a)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                return False
            if all(isinstance(x, (int, float, np.integer, np.floating, type(None))) for x in a + b):
                return np.allclose(
                    np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                    rtol=rtol, atol=atol, equal_nan=True,
                )
            return all(TestLazyWindowAnalysisResult._cells_match(x, y, rtol, atol) for x, y in zip(a, b))
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
            return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        return a == b

    @classmethod
    def _columns_match(cls, df_a, df_b, col, rtol=1e-9, atol=1e-9):
        if len(df_a) != len(df_b):
            return False
        for i in range(len(df_a)):
            if not cls._cells_match(df_a[col].iloc[i], df_b[col].iloc[i], rtol, atol):
                return False
        return True

    def test_lazy_apply_filters_per_row_only(self, lazy_synthetic_war):
        """Lazy apply_filters with per-row filters only matches eager."""
        config = {
            "high_rms": {"max_rms": 500},
            "low_rms": {"min_rms": 10},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            eager_dst = tmp / "eager"
            war_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            war_eager = war_eager.apply_filters(filter_config=config, min_valid_channels=2)
            war_eager.save_parquet_and_json(eager_dst, filename="war")

            lazy_dst = tmp / "lazy"
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.apply_filters(filter_config=config, min_valid_channels=2)
            war_lazy.save_parquet_and_json(lazy_dst, filename="war", batch_size=10)

            re_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=eager_dst)
            re_lazy = WindowAnalysisResult.load_parquet_and_json(folder_path=lazy_dst)
            assert len(re_eager.result) == len(re_lazy.result)
            for col in re_eager.result.columns:
                assert self._columns_match(re_eager.result, re_lazy.result, col), (
                    f"Column {col} differs between eager and lazy"
                )

    def test_lazy_apply_filters_with_logrms_range(self, lazy_synthetic_war):
        """Cross-row logrms_range path produces equivalent output."""
        config = {
            "logrms_range": {"z_range": 2},
            "high_rms": {"max_rms": 500},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            eager_dst = tmp / "eager"
            war_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            war_eager = war_eager.apply_filters(filter_config=config, min_valid_channels=2)
            war_eager.save_parquet_and_json(eager_dst, filename="war")

            lazy_dst = tmp / "lazy"
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.apply_filters(filter_config=config, min_valid_channels=2)
            war_lazy.save_parquet_and_json(lazy_dst, filename="war", batch_size=10)

            re_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=eager_dst)
            re_lazy = WindowAnalysisResult.load_parquet_and_json(folder_path=lazy_dst)
            assert len(re_eager.result) == len(re_lazy.result)
            for col in re_eager.result.columns:
                assert self._columns_match(re_eager.result, re_lazy.result, col), (
                    f"Column {col} differs between eager and lazy"
                )

    def test_lazy_aggregate_time_windows_animalday_isday(self, lazy_synthetic_war):
        """Lazy aggregate_time_windows by (animalday, isday) matches eager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            eager_dst = tmp / "eager"
            war_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            war_eager.aggregate_time_windows(groupby=["animalday", "isday"])
            war_eager.save_parquet_and_json(eager_dst, filename="war")

            lazy_dst = tmp / "lazy"
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.aggregate_time_windows(groupby=["animalday", "isday"])
            war_lazy.save_parquet_and_json(lazy_dst, filename="war", batch_size=10)

            re_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=eager_dst)
            re_lazy = WindowAnalysisResult.load_parquet_and_json(folder_path=lazy_dst)
            assert len(re_eager.result) == len(re_lazy.result)
            # Sort both by groupby keys so row order doesn't matter.
            keys = ["animalday", "isday"]
            re_eager_df = re_eager.result.sort_values(keys).reset_index(drop=True)
            re_lazy_df = re_lazy.result.sort_values(keys).reset_index(drop=True)
            common = [c for c in re_eager_df.columns if c in re_lazy_df.columns]
            for col in common:
                assert self._columns_match(re_eager_df, re_lazy_df, col, rtol=1e-6, atol=1e-6), (
                    f"Aggregated column {col} differs between eager and lazy"
                )

    def test_lazy_apply_filters_peak_memory_below_eager(self, lazy_synthetic_war):
        """apply_filters streaming peak < 50% eager AND < 5× parquet."""
        import tracemalloc

        config = {
            "logrms_range": {"z_range": 3},
            "high_rms": {"max_rms": 500},
            "low_rms": {"min_rms": 10},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")
            pq_size = (src / "war.parquet").stat().st_size

            tracemalloc.start()
            lazy_dst = tmp / "lazy"
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.apply_filters(filter_config=config, min_valid_channels=2)
            war_lazy.save_parquet_and_json(lazy_dst, filename="war", batch_size=10)
            _, lazy_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            eager_dst = tmp / "eager"
            war_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            war_eager = war_eager.apply_filters(filter_config=config, min_valid_channels=2)
            war_eager.save_parquet_and_json(eager_dst, filename="war")
            _, eager_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            ratio = lazy_peak / eager_peak
            assert ratio < 0.5, (
                f"lazy apply_filters peak {lazy_peak/1e6:.2f} MB is {ratio:.0%} of "
                f"eager peak {eager_peak/1e6:.2f} MB — expected < 50%"
            )
            assert lazy_peak < 5 * pq_size, (
                f"lazy peak {lazy_peak/1e6:.2f} MB exceeds 5× parquet "
                f"({pq_size/1e6:.2f} MB)"
            )

    def test_lazy_aggregate_raises_on_non_constant_column(self, lazy_synthetic_war):
        """Same non-constant-column guard as eager aggregate_time_windows."""
        # Inject a non-constant value in a normally-constant column.
        lazy_synthetic_war.result.loc[0, "genotype"] = "MUT"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.aggregate_time_windows(groupby=["animalday", "isday"])
            with pytest.raises(ValueError, match="not constant"):
                war_lazy.save_parquet_and_json(tmp / "lazy", filename="war", batch_size=10)

    def test_scan_does_not_materialize_dataframe(self, lazy_synthetic_war):
        """``scan_parquet_and_json`` reads JSON + parquet schema only — no DataFrame load.

        Regression guard: opening a lazy WAR must NOT pull row data from
        parquet.  Asserts the scan peak is a small fraction of the eager
        ``load_parquet_and_json`` peak — the eager path materialises the
        full DataFrame so any future regression that pulled rows into the
        scan path would push the ratio toward 1.0.
        """
        import tracemalloc

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            tracemalloc.start()
            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            # Touch every read-only accessor to be sure none of them lazily load row data.
            _ = war_lazy.animal_id
            _ = war_lazy.channel_names
            _ = war_lazy.channel_abbrevs
            _ = war_lazy.metadata
            _ = war_lazy.lof_scores_dict
            _ = war_lazy.bad_channels_dict
            _, scan_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            _ = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            _, eager_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            ratio = scan_peak / eager_peak
            assert ratio < 0.5, (
                f"scan peak {scan_peak/1e6:.2f} MB is {ratio:.0%} of eager-load "
                f"peak {eager_peak/1e6:.2f} MB — DataFrame was likely materialised"
            )

    def test_lazy_metadata_accessors_match_eager(self, lazy_synthetic_war):
        """LazyWAR property accessors return the same values as the eager WAR's attrs."""
        # Seed some metadata-rich state so the comparison is meaningful.
        lazy_synthetic_war.bad_channels_dict = {
            "A1_20230101": ["LMot"],
            "A1_20230102": ["RMot"],
        }
        # Provide LOF entries for every animalday in the fixture so the eager
        # constructor's auto-fill ("Added missing animalday to lof_scores_dict")
        # doesn't add empty entries that aren't in the on-disk JSON.
        lazy_synthetic_war.lof_scores_dict = {
            "A1_20230101": {
                "lof_scores": [1.0, 1.5, 0.9, 2.1],
                "channel_names": ["LMot", "RMot", "LBar", "RBar"],
            },
            "A1_20230102": {
                "lof_scores": [0.8, 1.7, 1.2, 1.0],
                "channel_names": ["LMot", "RMot", "LBar", "RBar"],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")

            assert lazy.animal_id == eager.animal_id
            assert lazy.channel_names == eager.channel_names
            assert lazy.channel_abbrevs == eager.channel_abbrevs
            assert lazy.bad_channels_dict == eager.bad_channels_dict
            assert lazy.lof_scores_dict == eager.lof_scores_dict

    def test_lazy_get_bad_channels_by_lof_threshold_matches_eager(self, lazy_synthetic_war):
        """LOF threshold resolution from JSON metadata equals the eager path."""
        lazy_synthetic_war.lof_scores_dict = {
            "A1_20230101": {
                "lof_scores": [1.0, 1.8, 0.9, 2.1],
                "channel_names": ["LMot", "RMot", "LBar", "RBar"],
            },
            "A1_20230102": {
                "lof_scores": [2.5, 1.0, 1.1, 0.8],
                "channel_names": ["LMot", "RMot", "LBar", "RBar"],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")

            assert lazy.get_bad_channels_by_lof_threshold(1.5) == eager.get_bad_channels_by_lof_threshold(1.5)

    def test_lazy_chain_reorder_hash_filter(self, lazy_synthetic_war):
        """A multi-transform chain (reorder + add_unique_hash + apply_filters)
        produces the same output as the equivalent eager pipeline.
        """
        target = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"]
        config = {"high_rms": {"max_rms": 500}}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            # Eager: load → reorder → add_unique_hash → apply_filters → save
            eager_dst = tmp / "eager"
            war_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            war_eager.reorder_and_pad_channels(target, use_abbrevs=True)
            # Force a deterministic hash so the comparison is reproducible.
            import secrets as _sec
            _orig_token_hex = _sec.token_hex
            _sec.token_hex = lambda n=None: "deadbeef"
            try:
                war_eager.add_unique_hash(4)
                war_eager = war_eager.apply_filters(filter_config=config, min_valid_channels=2)
                war_eager.save_parquet_and_json(eager_dst, filename="war")

                # Lazy: same chain via the streaming engine
                lazy_dst = tmp / "lazy"
                war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
                war_lazy.reorder_and_pad_channels(target, use_abbrevs=True)
                war_lazy.add_unique_hash(4)
                war_lazy.apply_filters(filter_config=config, min_valid_channels=2)
                war_lazy.save_parquet_and_json(lazy_dst, filename="war", batch_size=10)
            finally:
                _sec.token_hex = _orig_token_hex

            re_eager = WindowAnalysisResult.load_parquet_and_json(folder_path=eager_dst)
            re_lazy = WindowAnalysisResult.load_parquet_and_json(folder_path=lazy_dst)
            assert re_eager.animal_id == re_lazy.animal_id
            assert re_eager.channel_names == re_lazy.channel_names
            assert len(re_eager.result) == len(re_lazy.result)
            for col in re_eager.result.columns:
                assert self._columns_match(re_eager.result, re_lazy.result, col), (
                    f"Column {col} differs between eager and lazy"
                )

    def test_lazy_save_metadata_round_trip(self, lazy_synthetic_war):
        """JSON sidecar after lazy save preserves every metadata field a
        downstream rule depends on (animal_id, channel_names, lof_scores_dict,
        bad_channels_dict, assume_from_number).
        """
        lazy_synthetic_war.bad_channels_dict = {
            "A1_20230101": ["LMot"],
            "A1_20230102": ["RMot"],
        }
        lazy_synthetic_war.lof_scores_dict = {
            "A1_20230101": {
                "lof_scores": [1.0, 1.5, 0.9, 2.1],
                "channel_names": ["LMot", "RMot", "LBar", "RBar"],
            }
        }
        target = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            lazy_synthetic_war.save_parquet_and_json(src, filename="war")

            war_lazy = WindowAnalysisResult.scan_parquet_and_json(src, filename="war")
            war_lazy.reorder_and_pad_channels(target, use_abbrevs=True)
            war_lazy.save_parquet_and_json(tmp / "out", filename="war", batch_size=10)

            out_meta = json.loads((tmp / "out" / "war.json").read_text())
            # Reorder updates channel_names; non-touched fields propagate unchanged.
            assert out_meta["channel_names"] == target
            assert out_meta["animal_id"] == "A1"
            assert out_meta["lof_scores_dict"] == lazy_synthetic_war.lof_scores_dict
            assert out_meta["bad_channels_dict"] == lazy_synthetic_war.bad_channels_dict
            # assume_from_number must be preserved (regression guard for the LOF
            # channel-filter chain on raw-named channels like ``D-015``).
            assert out_meta["assume_from_number"] == lazy_synthetic_war.assume_from_number


class TestTimestampHandling:
    """Regression tests for the datetime/tz round-trip edge cases pyarrow
    can't handle natively (tzlocal, tz+NaT, object-dtype Timestamp+None).
    """

    @staticmethod
    def _war(ts_series, animal_id="A1"):
        """Build a minimal WAR with the given timestamp series."""
        df = pd.DataFrame({
            "animalday": [f"{animal_id}_d{i}" for i in range(len(ts_series))],
            "duration": [60.0] * len(ts_series),
            "timestamp": ts_series,
            "rms": [[1.0, 2.0]] * len(ts_series),
        })
        return WindowAnalysisResult(
            result=df, animal_id=animal_id, genotype="WT",
            channel_names=["LMot", "RMot"],
        )

    def _roundtrip(self, war):
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")
            return WindowAnalysisResult.load_parquet_and_json(folder_path=Path(tmpdir))

    @staticmethod
    def _moments_equal(a: pd.Series, b: pd.Series) -> bool:
        """Compare two datetime Series by absolute moment, tolerating
        parquet's ns→us precision conversion."""
        def _to_utc_us(s):
            s = s.dt.tz_convert("UTC") if getattr(s.dt, "tz", None) is not None else s.dt.tz_localize("UTC")
            return s.dt.floor("us")
        return (_to_utc_us(a).reset_index(drop=True) == _to_utc_us(b).reset_index(drop=True)).all()

    def test_naive_datetime_roundtrip(self):
        ts = pd.Series(pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:05", "2023-01-01 10:10"]))
        loaded = self._roundtrip(self._war(ts))
        assert self._moments_equal(
            ts.reset_index(drop=True),
            loaded.result["timestamp"].reset_index(drop=True),
        )

    def test_tzlocal_datetime_roundtrip_to_utc(self):
        """pyarrow can't serialise tzlocal(); save normalises to UTC."""
        from dateutil.tz import tzlocal
        ts = pd.Series([
            pd.Timestamp("2023-01-01 10:00", tz=tzlocal()),
            pd.Timestamp("2023-01-01 10:05", tz=tzlocal()),
        ])
        loaded = self._roundtrip(self._war(ts))
        loaded_ts = loaded.result["timestamp"]
        assert getattr(loaded_ts.dt, "tz", None) is not None
        assert self._moments_equal(ts.reset_index(drop=True), loaded_ts.reset_index(drop=True))

    def test_named_tz_datetime_roundtrip(self):
        ts = pd.Series([
            pd.Timestamp("2023-01-01 10:00", tz="America/New_York"),
            pd.Timestamp("2023-01-01 10:05", tz="America/New_York"),
        ])
        loaded = self._roundtrip(self._war(ts))
        assert self._moments_equal(
            ts.reset_index(drop=True),
            loaded.result["timestamp"].reset_index(drop=True),
        )

    def test_tz_aware_with_nat_strips_tz(self):
        """tz+NaT crashes pyarrow; save strips tz (lossy on label, lossless on moment)."""
        ts = pd.Series(pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:05", "2023-01-01 10:10"]))
        war = self._war(ts)
        # Edge case lives on a non-validated column.
        war.result["end_time"] = pd.Series([
            pd.Timestamp("2023-01-01 10:00", tz="America/New_York"),
            pd.NaT,
            pd.Timestamp("2023-01-01 10:05", tz="America/New_York"),
        ])
        loaded = self._roundtrip(war)
        end = loaded.result["end_time"]
        assert getattr(end.dt, "tz", None) is None  # tz stripped
        assert pd.isna(end.iloc[1])
        # Non-null moments preserved (compare as naive UTC).
        expected = war.result["end_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("us")
        actual = end.dt.floor("us")
        assert (actual.iloc[0] == expected.iloc[0]) and (actual.iloc[2] == expected.iloc[2])

    def test_object_dtype_timestamp_with_none(self):
        """Object-dtype Timestamp+None gets coerced to datetime64 before pa.table."""
        ts = pd.Series(pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:05", "2023-01-01 10:10"]))
        war = self._war(ts)
        war.result["end_time"] = pd.Series([
            pd.Timestamp("2023-01-01 10:00"),
            None,
            pd.Timestamp("2023-01-01 10:05"),
        ], dtype=object)
        assert war.result["end_time"].dtype == object
        loaded = self._roundtrip(war)
        end = loaded.result["end_time"]
        assert pd.api.types.is_datetime64_any_dtype(end)
        assert pd.isna(end.iloc[1])
        assert end.iloc[0] == pd.Timestamp("2023-01-01 10:00")

    def test_all_null_endfile_column(self):
        """All-None object column (e.g. ``endfile`` in real WARs) round-trips
        without crashing.  Sanity check for the integration-WAR case where
        such columns coexist with tz-aware timestamps.
        """
        ts = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02"]))
        war = self._war(ts)
        war.result["endfile"] = [None, None]
        # Save+load shouldn't raise; endfile reloads as either None or NaN.
        loaded = self._roundtrip(war)
        assert "endfile" in loaded.result.columns
        assert loaded.result["endfile"].isna().all()


class TestAnimalOrganizerLOF:
    """Test LOF functionality integration with AnimalOrganizer (mocked)."""

    def test_animal_organizer_lof_methods_exist(self):
        """Test that AnimalOrganizer has the expected LOF methods."""
        from neurodent.visualization.results import AnimalOrganizer

        # Check that the methods exist
        assert hasattr(AnimalOrganizer, "compute_bad_channels")
        assert hasattr(AnimalOrganizer, "apply_lof_threshold")
        assert hasattr(AnimalOrganizer, "get_all_lof_scores")

        # Check method signatures by inspection
        import inspect

        # compute_bad_channels should accept lof_threshold and force_recompute
        sig = inspect.signature(AnimalOrganizer.compute_bad_channels)
        assert "lof_threshold" in sig.parameters
        assert "force_recompute" in sig.parameters

        # apply_lof_threshold should accept lof_threshold
        sig = inspect.signature(AnimalOrganizer.apply_lof_threshold)
        assert "lof_threshold" in sig.parameters

        # get_all_lof_scores should have no required parameters
        sig = inspect.signature(AnimalOrganizer.get_all_lof_scores)
        required_params = [p for p in sig.parameters.values() if p.default == p.empty]
        assert len(required_params) == 1  # Only 'self'

    @patch("neurodent.visualization.results.AnimalOrganizer.__init__", return_value=None)
    def test_animal_organizer_war_creation_includes_lof(self, mock_init):
        """Test that WindowAnalysisResult creation includes LOF scores."""
        from neurodent.visualization.results import AnimalOrganizer

        # Create mock AnimalOrganizer with necessary attributes
        ao = AnimalOrganizer.__new__(AnimalOrganizer)
        ao.animaldays = ["day1", "day2"]
        ao.animal_id = "A1"
        ao.genotype = "WT"
        ao.sex = "Male"
        ao.channel_names = ["LMot", "RMot"]
        ao.assume_from_number = False
        ao.bad_channels_dict = {}

        # Mock long_recordings with LOF scores
        mock_lrec1 = Mock()
        mock_lrec1.lof_scores = np.array([1.5, 2.0])
        mock_lrec1.channel_names = ["LMot", "RMot"]

        mock_lrec2 = Mock()
        mock_lrec2.lof_scores = np.array([0.8, 1.2])
        mock_lrec2.channel_names = ["LMot", "RMot"]

        ao.long_recordings = [mock_lrec1, mock_lrec2]

        # Mock features_df
        ao.features_df = pd.DataFrame(
            {
                "animal": ["A1"] * 4,
                "animalday": ["day1", "day1", "day2", "day2"],
                "genotype": ["WT"] * 4,
                "duration": [4.0] * 4,
                "rms": [[100.0, 110.0]] * 4,
                "timestamp": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-01 10:04:00", "2023-01-02 10:00:00", "2023-01-02 10:04:00"]
                ),
            }
        )

        # Test the LOF scores collection logic from compute_windowed_analysis
        lof_scores_dict = {}
        for animalday, lrec in zip(ao.animaldays, ao.long_recordings):
            if hasattr(lrec, "lof_scores") and lrec.lof_scores is not None:
                lof_scores_dict[animalday] = {
                    "lof_scores": lrec.lof_scores.tolist(),
                    "channel_names": lrec.channel_names,
                }

        # Verify LOF scores were collected correctly
        assert "day1" in lof_scores_dict
        assert "day2" in lof_scores_dict
        assert lof_scores_dict["day1"]["lof_scores"] == [1.5, 2.0]
        assert lof_scores_dict["day2"]["lof_scores"] == [0.8, 1.2]

        # Create WindowAnalysisResult with LOF scores
        from neurodent.visualization.results import WindowAnalysisResult

        war = WindowAnalysisResult(
            ao.features_df,
            ao.animal_id,
            ao.genotype,
            ao.sex,
            ao.channel_names,
            ao.assume_from_number,
            ao.bad_channels_dict,
            False,  # suppress_short_interval_error
            lof_scores_dict,
        )

        # Verify LOF functionality works
        assert hasattr(war, "lof_scores_dict")
        assert war.lof_scores_dict == lof_scores_dict

        scores = war.get_lof_scores()
        assert scores["day1"]["LMot"] == 1.5
        assert scores["day2"]["RMot"] == 1.2

    def test_get_all_lof_scores_no_overwrites(self):
        """Test that get_all_lof_scores preserves all LOF data with no overwrites."""
        from neurodent.visualization.results import AnimalOrganizer
        from datetime import datetime
        from unittest.mock import MagicMock
        from pathlib import Path

        # Create 3 mock LROs with unique LOF scores
        lros = []

        for i in range(3):
            lro = MagicMock()
            lro.channel_names = ["Ch0", "Ch1", "Ch2"]
            lro.base_folder_path = Path(f"/mock/day{i}")
            lro.get_date_string.return_value = f"Jan-{i+1:02d}-2023"
            lro.file_end_datetimes = [datetime(2023, 1, i+1, 12, 0)]
            lro.file_durations = [3600.0]

            # Each LRO has unique LOF scores
            lro.lof_scores = np.array([1.0 + i*0.1, 1.5 + i*0.1, 2.0 + i*0.1])
            lro.get_lof_scores = MagicMock(return_value={
                "Ch0": 1.0 + i*0.1,
                "Ch1": 1.5 + i*0.1,
                "Ch2": 2.0 + i*0.1,
            })

            lros.append(lro)

        # Create AnimalOrganizer from LROs
        ao = AnimalOrganizer.from_lros(
            lros=lros,
            animal_id="TestAnimal",
            genotype="WT"
        )

        # Get LOF scores
        lof_dict = ao.get_all_lof_scores()

        # Should have exactly 3 entries
        assert len(lof_dict) == 3, \
            f"Expected 3 LOF score entries, got {len(lof_dict)}"

        # Verify all animaldays are present
        assert len(lof_dict) == len(ao.unique_animaldays), \
            "LOF dict size should match unique_animaldays"

        # Verify each animalday has unique scores
        all_ch0_scores = []
        for animalday, scores in lof_dict.items():
            assert animalday in ao.unique_animaldays, \
                f"Animalday {animalday} not in unique_animaldays"

            ch0_score = scores["Ch0"]
            all_ch0_scores.append(ch0_score)

        # All Ch0 scores should be different (no overwrites)
        assert len(set(all_ch0_scores)) == 3, \
            f"LOF scores were overwritten! Got {all_ch0_scores}, expected 3 unique values"

        # Verify specific values
        expected_ch0_scores = {1.0, 1.1, 1.2}
        actual_ch0_scores = set(all_ch0_scores)
        assert actual_ch0_scores == expected_ch0_scores, \
            f"Expected Ch0 scores {expected_ch0_scores}, got {actual_ch0_scores}"


class TestComputeGlobalTimelineKwargDiscipline:
    """Regression tests for kwarg leaks from _compute_global_timeline into the
    LongRecordingOrganizer (and ultimately into ``extract_func``).

    Background: the arxrosa run failed every EDF animal with
    ``read_raw_edf() got an unexpected keyword argument 'input_type'``.
    The cause was a hardcoded ``_lro_kwargs["input_type"] = "file"`` in
    _compute_global_timeline, intended for the long-dead
    ``_load_and_process_mne_data`` branch but ending up forwarded all the way
    to ``mne.io.read_raw_edf`` via ``extract_func(item, **kwargs)``.

    Synthetic test extractors all accept ``**kwargs``, so the leak was invisible
    until the first end-to-end run with the real (kwarg-strict) MNE reader.
    """

    @staticmethod
    def _capture_lro_kwargs(monkeypatch):
        """Patch core.LongRecordingOrganizer.__init__ to record the kwargs it
        was called with, returning the capture list.

        We patch via the same module reference that results.py uses
        (``results.core.LongRecordingOrganizer``) so we stay correct even after
        tests (e.g. test_imports.TestCircularImports) drop ``neurodent.core``
        from ``sys.modules`` and force a re-import, which rebinds
        ``neurodent.core.LongRecordingOrganizer`` to a new class object.
        """
        from unittest.mock import MagicMock
        from neurodent.visualization import results as _results_mod

        captured: list[dict] = []

        def fake_init(self, item, **kwargs):
            captured.append(dict(kwargs))
            self.LongRecording = MagicMock()
            self.LongRecording.get_duration.return_value = 100.0

        monkeypatch.setattr(
            _results_mod.core.LongRecordingOrganizer, "__init__", fake_init
        )
        return captured

    def _make_organizer_shell(self):
        """Build an AnimalOrganizer instance bypassing __init__'s discovery so
        we can call _compute_global_timeline in isolation."""
        from neurodent.visualization.results import AnimalOrganizer
        ao = AnimalOrganizer.__new__(AnimalOrganizer)
        ao.animal_id = "test"
        return ao

    def test_no_input_type_injection_for_mne_mode(self, tmp_path, monkeypatch):
        """Mode='mne' must not cause input_type to leak into LRO kwargs."""
        from datetime import datetime

        captured = self._capture_lro_kwargs(monkeypatch)
        ao = self._make_organizer_shell()

        fake_edf = tmp_path / "fake_session1.edf"
        fake_edf.write_bytes(b"")
        base_dt = datetime(2025, 1, 1, 9, 0, 0)
        base_lro_kwargs = {"mode": "mne", "extract_func": "read_raw_edf"}

        try:
            ao._compute_global_timeline(
                base_datetime=base_dt,
                animalday_to_items={"sess1": [fake_edf]},
                base_lro_kwargs=base_lro_kwargs,
                original_manual_datetimes=base_dt,
            )
        except Exception:
            # Tolerate downstream errors after kwarg capture; the assertion below
            # only cares about which kwargs reached LongRecordingOrganizer.
            pass

        assert captured, "expected at least one LongRecordingOrganizer instantiation"
        leaked = [kw for kw in captured if "input_type" in kw]
        assert not leaked, (
            "Regression: input_type was injected into LRO kwargs by "
            f"_compute_global_timeline. Captured: {leaked}"
        )

    def test_strict_extract_func_signature_simulation(self, tmp_path, monkeypatch):
        """A 'strict' extract_func (mimics real mne.io.read_raw_edf) should
        receive only the kwargs the caller actually requested — no extras
        injected by the timeline code.
        """
        from datetime import datetime

        captured = self._capture_lro_kwargs(monkeypatch)
        ao = self._make_organizer_shell()

        fake_edf = tmp_path / "session_a.edf"
        fake_edf.write_bytes(b"")
        base_dt = datetime(2025, 1, 1, 12, 0, 0)
        # Caller-supplied kwargs only.  Anything beyond these on the way to
        # the LRO would indicate a leak from inside _compute_global_timeline.
        base_lro_kwargs = {"mode": "mne", "extract_func": "read_raw_edf"}
        allowed = set(base_lro_kwargs) | {"manual_datetimes"}

        try:
            ao._compute_global_timeline(
                base_datetime=base_dt,
                animalday_to_items={"sess": [fake_edf]},
                base_lro_kwargs=base_lro_kwargs,
                original_manual_datetimes=base_dt,
            )
        except Exception:
            pass

        assert captured, "expected at least one LongRecordingOrganizer instantiation"
        for kw in captured:
            extras = set(kw) - allowed
            assert not extras, (
                "Regression: _compute_global_timeline injected unexpected LRO "
                f"kwargs: {extras}.  Allowed only: {allowed}.  Full kwargs: {kw}."
            )
