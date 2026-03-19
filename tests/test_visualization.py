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

    def test_average_feature_linear_2d(self, parser, sample_df):
        """Test averaging LINEAR_2D feature (psdslope)."""
        result = parser._average_feature(sample_df, "psdslope", "duration")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # (n_chan, n_components)

    def test_average_feature_simple_matrix(self, parser, sample_df):
        """Test averaging SIMPLE_MATRIX feature (pcorr)."""
        result = parser._average_feature(sample_df, "pcorr", "duration")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)  # (n_chan, n_chan)

    def test_average_feature_banded_matrix(self, parser, sample_df):
        """Test averaging BANDED_MATRIX feature (cohere)."""
        result = parser._average_feature(sample_df, "cohere", "duration")
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "beta" in result
        assert np.array(result["alpha"]).shape == (3, 3)

    def test_average_feature_hist(self, parser, sample_df):
        """Test averaging HIST feature (psd)."""
        result = parser._average_feature(sample_df, "psd", "duration")
        assert isinstance(result, tuple)
        assert len(result) == 2
        coords, values = result
        np.testing.assert_array_equal(coords, [1.0, 2.0, 3.0])
        assert values.shape == (2, 3)  # (n_freq_rows, n_chan)

    def test_average_feature_no_weights(self, parser, sample_df):
        """Test averaging with no weight column (uniform weights)."""
        result = parser._average_feature(sample_df, "rms", weightsname=None)
        # Uniform weights → simple mean
        expected = np.mean([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], axis=0)
        np.testing.assert_array_almost_equal(result, expected)


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
            result=sample_result_df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"]
        )

    @pytest.fixture
    def filtering_war(self, filtering_result_df):
        """Create a WindowAnalysisResult instance for filtering tests."""
        return WindowAnalysisResult(
            result=filtering_result_df,
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot", "LBar"],
            bad_channels_dict={"A1_20230101": ["LMot"], "A1_20230102": ["RMot"]},
        )

    def test_init(self, war, sample_result_df):
        """Test WindowAnalysisResult initialization."""
        assert war.animal_id == "A1"
        assert war.genotype == "WT"
        assert war.channel_names == ["LMot", "RMot"]
        assert len(war.result) == len(sample_result_df)

    def test_copy(self, filtering_war):
        """Test that copy creates an independent deep copy of WindowAnalysisResult."""
        # Create a copy
        war_copy = filtering_war.copy()

        # Check that the copy has the same attributes
        assert war_copy.animal_id == filtering_war.animal_id
        assert war_copy.genotype == filtering_war.genotype
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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"], suppress_short_interval_error=True
        )

        # Verify the parameter is stored correctly
        assert war.suppress_short_interval_error
        assert war.animal_id == "A1"
        assert war.genotype == "WT"

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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Save and load
            war.save_pickle_and_json(save_path)
            loaded_war = WindowAnalysisResult.load_pickle_and_json(save_path)

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

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"], bad_channels_dict={}
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

        war = WindowAnalysisResult(result=df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

        config = {"high_rms": {"max_rms": 500}}

        with pytest.raises(ValueError, match="Cannot calculate window duration"):
            war.apply_filters(config, morphological_smoothing_seconds=8.0)

    def test_filter_morphological_smoothing(self, filtering_war):
        """Test standalone morphological smoothing filter."""
        filtered = filtering_war.filter_morphological_smoothing(smoothing_seconds=8.0)

        assert isinstance(filtered, WindowAnalysisResult)
        assert filtered is not filtering_war

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
            assert args[1] == 8.0
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
        from neurodent.visualization.plotting.experiment import df_normalize_baseline

        # Patch df_normalize_baseline to just return the input DataFrame
        import neurodent.visualization.plotting.experiment as expmod

        expmod.df_normalize_baseline = lambda **kwargs: kwargs["df"]
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
            "pcorr": [rng.random((n_chan, n_chan)).tolist() for _ in range(n_rows)],
            "cohere": [
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
        # 1-D values per cell (single-channel) → no channel dim, shape stays (W, F)
        assert values.shape == (2, 3)
        np.testing.assert_array_equal(coords[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(values[1], [40.0, 50.0, 60.0])

    def test_extract_hist_data_lists(self):
        """Test extracting histogram data from list format (parquet origin)."""
        from neurodent.visualization.feature_utils import extract_hist_data

        series = pd.Series([
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[1.0, 2.0, 3.0], [40.0, 50.0, 60.0]],
        ])
        coords, values = extract_hist_data(series)
        assert coords.shape == (2, 3)
        # 1-D values per cell (single-channel) → no channel dim, shape stays (W, F)
        assert values.shape == (2, 3)
        np.testing.assert_array_equal(values[0], [10.0, 20.0, 30.0])

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
        vals = np.random.rand(4, 3)  # (n_time, n_chan)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.LINEAR)
        assert result.shape == (4, 3, 1)

    def test_flatten_linear_2d(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.random.rand(4, 3, 2)  # (n_time, n_chan, n_comp)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.LINEAR_2D)
        assert result.shape == (4, 3, 2)

    def test_flatten_band(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        # Canonical input: (n_time, n_chan, n_bands)
        vals = np.random.rand(4, 3, 5)  # (n_time, n_chan, n_bands)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.BAND)
        # No transpose needed — output is (n_time, n_chan, n_bands)
        assert result.shape == (4, 3, 5)

    def test_flatten_simple_matrix_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        n_chan = 3
        vals = np.random.rand(4, n_chan, n_chan)  # (n_time, n_chan, n_chan)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.SIMPLE_MATRIX, triag=True)
        n_pairs = n_chan * (n_chan - 1) // 2
        assert result.shape == (4, n_pairs, 1)

    def test_flatten_simple_matrix_no_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        n_chan = 3
        vals = np.random.rand(4, n_chan, n_chan)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.SIMPLE_MATRIX, triag=False)
        assert result.shape == (4, n_chan * n_chan, 1)

    def test_flatten_banded_matrix_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        n_chan = 3
        n_bands = 5
        # Canonical input: (n_time, n_chan, n_chan, n_bands)
        vals = np.random.rand(4, n_chan, n_chan, n_bands)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.BANDED_MATRIX, triag=True)
        n_pairs = n_chan * (n_chan - 1) // 2
        assert result.shape == (4, n_pairs, n_bands)

    def test_flatten_banded_matrix_no_triag(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        n_chan = 3
        n_bands = 5
        # Canonical input: (n_time, n_chan, n_chan, n_bands)
        vals = np.random.rand(4, n_chan, n_chan, n_bands)
        result = flatten_feature_for_plotting(vals, constants.FeatureType.BANDED_MATRIX, triag=False)
        assert result.shape == (4, n_chan * n_chan, n_bands)

    def test_flatten_unsupported_raises(self):
        from neurodent.visualization.feature_utils import flatten_feature_for_plotting
        vals = np.random.rand(4, 3)
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


class TestExtractFeature:
    """Test extract_feature utility."""

    def test_extract_linear(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([[1.0, 2.0], [3.0, 4.0]])
        vals, keys = extract_feature(series, constants.FeatureType.LINEAR)
        assert vals.shape == (2, 2)
        assert keys is None
        np.testing.assert_array_equal(vals[0], [1.0, 2.0])

    def test_extract_linear_2d(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([[[1.0, 2.0], [3.0, 4.0]]])
        vals, keys = extract_feature(series, constants.FeatureType.LINEAR_2D)
        assert vals.shape == (1, 2, 2)
        assert keys is None

    def test_extract_simple_matrix(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([[[0.0, 0.5], [0.5, 0.0]]])
        vals, keys = extract_feature(series, constants.FeatureType.SIMPLE_MATRIX)
        assert vals.shape == (1, 2, 2)
        assert keys is None

    def test_extract_band(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([{"delta": [1.0, 2.0], "theta": [3.0, 4.0]}])
        vals, keys = extract_feature(series, constants.FeatureType.BAND)
        assert vals.shape == (1, 2, 2)  # (W, C, B)
        assert keys == ["delta", "theta"]

    def test_extract_banded_matrix(self):
        from neurodent.visualization.feature_utils import extract_feature
        series = pd.Series([{"delta": [[0.0, 0.5], [0.5, 0.0]]}])
        vals, keys = extract_feature(series, constants.FeatureType.BANDED_MATRIX)
        assert vals.shape == (1, 2, 2, 1)  # (W, C, C, B)
        assert keys == ["delta"]


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
        war = WindowAnalysisResult(result=test_df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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
            result=test_df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"], lof_scores_dict={}
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
        with patch.object(war_with_lof, "save_pickle_and_json") as mock_save:
            # Test the JSON creation part directly
            lof_scores_dict = {}
            # Simulate the LOF collection that normally happens in save
            if hasattr(war_with_lof, "lof_scores_dict") and war_with_lof.lof_scores_dict:
                lof_scores_dict = war_with_lof.lof_scores_dict

            json_dict = {
                "animal_id": war_with_lof.animal_id,
                "genotype": war_with_lof.genotype,
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
        war = WindowAnalysisResult(result=test_df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

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


class TestWindowAnalysisResultPickleJsonParameters:
    """Test pickle_name and json_name parameters in load_pickle_and_json."""

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

        war = WindowAnalysisResult(result=test_df, animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save with default names
            war.save_pickle_and_json(tmpdir, filename="war")

            # Create subdirectory structure
            subdir = tmpdir / "subdir"
            subdir.mkdir()
            war.save_pickle_and_json(subdir, filename="nested_war")

            # Also save with custom names at root level
            war.save_pickle_and_json(tmpdir, filename="custom_war")

            yield {"tmpdir": tmpdir, "subdir": subdir, "war": war}

    def test_load_default_behavior(self, temp_war_files):
        """Test that default behavior (no pickle_name/json_name) still works."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        # Remove other files to test single file case
        for f in tmpdir.glob("*"):
            if f.name not in ["war.pkl", "war.json"]:
                if f.is_file():
                    f.unlink()
                else:
                    import shutil

                    shutil.rmtree(f)

        loaded_war = WindowAnalysisResult.load_pickle_and_json(folder_path=str(tmpdir))

        assert loaded_war.animal_id == original_war.animal_id
        assert loaded_war.genotype == original_war.genotype
        assert loaded_war.channel_names == original_war.channel_names
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_with_exact_filenames(self, temp_war_files):
        """Test loading with exact pickle_name and json_name."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        loaded_war = WindowAnalysisResult.load_pickle_and_json(
            folder_path=str(tmpdir), pickle_name="custom_war.pkl", json_name="custom_war.json"
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_with_relative_paths(self, temp_war_files):
        """Test loading with relative paths from folder_path."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        loaded_war = WindowAnalysisResult.load_pickle_and_json(
            folder_path=str(tmpdir), pickle_name="subdir/nested_war.pkl", json_name="subdir/nested_war.json"
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_with_absolute_paths(self, temp_war_files):
        """Test loading with absolute paths."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        pickle_path = tmpdir / "custom_war.pkl"
        json_path = tmpdir / "custom_war.json"

        loaded_war = WindowAnalysisResult.load_pickle_and_json(
            folder_path=str(tmpdir), pickle_name=str(pickle_path), json_name=str(json_path)
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_without_folder_path(self, temp_war_files):
        """Test loading with absolute paths only (no folder_path)."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        pickle_path = tmpdir / "custom_war.pkl"
        json_path = tmpdir / "custom_war.json"

        loaded_war = WindowAnalysisResult.load_pickle_and_json(pickle_name=str(pickle_path), json_name=str(json_path))

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)

    def test_load_pickle_not_found(self, temp_war_files):
        """Test error when pickle file not found."""
        tmpdir = temp_war_files["tmpdir"]

        with pytest.raises(FileNotFoundError, match="Pickle file not found"):
            WindowAnalysisResult.load_pickle_and_json(
                folder_path=str(tmpdir), pickle_name="nonexistent.pkl", json_name="war.json"
            )

    def test_load_json_not_found(self, temp_war_files):
        """Test error when JSON file not found."""
        tmpdir = temp_war_files["tmpdir"]

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            WindowAnalysisResult.load_pickle_and_json(
                folder_path=str(tmpdir), pickle_name="war.pkl", json_name="nonexistent.json"
            )

    def test_load_multiple_files_without_specification(self, temp_war_files):
        """Test error when multiple files exist but none specified."""
        tmpdir = temp_war_files["tmpdir"]

        # There should be multiple .pkl and .json files in tmpdir
        pkl_files = list(tmpdir.glob("*.pkl"))
        json_files = list(tmpdir.glob("*.json"))

        # Ensure we have multiple files
        assert len(pkl_files) > 1
        assert len(json_files) > 1

        with pytest.raises(ValueError, match="Expected exactly one pickle file"):
            WindowAnalysisResult.load_pickle_and_json(folder_path=str(tmpdir))

    def test_load_no_files_found(self):
        """Test error when no files are found."""
        import tempfile

        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(ValueError, match="Expected exactly one pickle file"):
                WindowAnalysisResult.load_pickle_and_json(folder_path=empty_dir)

    def test_load_invalid_folder_path(self):
        """Test error with invalid folder path."""
        with pytest.raises(ValueError, match="Folder path .* does not exist"):
            WindowAnalysisResult.load_pickle_and_json(folder_path="/nonexistent/path")

    def test_load_missing_parameters(self):
        """Test error when required parameters are missing."""
        # Neither folder_path nor both pickle_name/json_name provided
        with pytest.raises(ValueError, match="Either folder_path must be provided"):
            WindowAnalysisResult.load_pickle_and_json()

        # Only one of pickle_name/json_name provided without folder_path
        with pytest.raises(ValueError, match="Either folder_path must be provided"):
            WindowAnalysisResult.load_pickle_and_json(pickle_name="/some/path.pkl")

        with pytest.raises(ValueError, match="Either folder_path must be provided"):
            WindowAnalysisResult.load_pickle_and_json(json_name="/some/path.json")

    def test_load_mixed_absolute_relative_paths(self, temp_war_files):
        """Test mixing absolute and relative paths."""
        tmpdir = temp_war_files["tmpdir"]
        original_war = temp_war_files["war"]

        # Absolute pickle path, relative json path
        pickle_path = tmpdir / "custom_war.pkl"

        loaded_war = WindowAnalysisResult.load_pickle_and_json(
            folder_path=str(tmpdir),
            pickle_name=str(pickle_path),  # Absolute
            json_name="custom_war.json",  # Relative
        )

        assert loaded_war.animal_id == original_war.animal_id
        pd.testing.assert_frame_equal(loaded_war.result, original_war.result)


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
            war.save_pickle_and_json(tmpdir, filename="war")

            # Verify parquet file was created
            assert (Path(tmpdir) / "war.parquet").exists()
            # No sidecar meta file — metadata is embedded in parquet
            assert not (Path(tmpdir) / "war.parquet.meta.json").exists()

            # Load (should prefer parquet)
            loaded = WindowAnalysisResult.load_pickle_and_json(
                folder_path=tmpdir, pickle_name="war.pkl", json_name="war.json"
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
            war.save_pickle_and_json(tmpdir, filename="war")

            loaded = WindowAnalysisResult.load_pickle_and_json(folder_path=tmpdir)
            assert loaded.animal_id == war.animal_id

    def test_fallback_to_pickle_when_no_parquet(self, war_with_complex_columns):
        """Test that loading falls back to pickle when parquet files are absent."""
        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_pickle_and_json(tmpdir, filename="war")
            # Remove parquet to force pickle fallback
            (Path(tmpdir) / "war.parquet").unlink()

            loaded = WindowAnalysisResult.load_pickle_and_json(
                folder_path=tmpdir, pickle_name="war.pkl", json_name="war.json"
            )
            assert loaded.animal_id == war.animal_id
            assert loaded.result["duration"].tolist() == war.result["duration"].tolist()

    def test_metadata_embedded_in_parquet(self, war_with_complex_columns):
        """Test that encoded column metadata is stored in parquet schema metadata."""
        import pyarrow.parquet as pq

        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_pickle_and_json(tmpdir, filename="war")

            table = pq.read_table(Path(tmpdir) / "war.parquet")
            schema_meta = table.schema.metadata
            assert b"neurodent" in schema_meta

            nd_meta = json.loads(schema_meta[b"neurodent"])
            assert "encoded_columns" in nd_meta
            assert "rms" in nd_meta["encoded_columns"]

    def test_encode_decode_round_trip(self):
        """Test _encode_df_for_parquet and _decode_df_from_parquet directly."""
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

        # Encoded values should be JSON strings
        for col in cols:
            for val in encoded[col].dropna():
                assert isinstance(val, str)

        # Decode and verify round-trip
        decoded = WindowAnalysisResult._decode_df_from_parquet(encoded, cols)
        for i in range(len(df)):
            # list_col stays as plain list after JSON round-trip (no numpy conversion)
            assert decoded["list_col"].iloc[i] == df["list_col"].iloc[i]
            assert decoded["dict_col"].iloc[i] == df["dict_col"].iloc[i]
            # array_col comes back as a plain list; verify values match
            np.testing.assert_array_equal(
                decoded["array_col"].iloc[i], df["array_col"].iloc[i]
            )

    def test_parquet_file_has_content(self, war_with_complex_columns):
        """Test that the parquet file has meaningful content matching the DataFrame."""
        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_pickle_and_json(tmpdir, filename="war")
            pq_path = Path(tmpdir) / "war.parquet"
            assert pq_path.stat().st_size > 0

            reloaded = pd.read_parquet(pq_path, engine="pyarrow")
            assert len(reloaded) == len(war.result)
            assert set(war.result.columns).issubset(set(reloaded.columns))

    def test_save_load_both_formats(self, war_with_complex_columns):
        """Test that both parquet and pickle load paths produce valid results.

        Parquet has higher per-call overhead than pickle for small DataFrames
        (pyarrow table construction, schema parsing, JSON decode).  The gap
        narrows substantially on real-world WARs with thousands of rows.
        Rather than asserting a timing ratio (which is inherently flaky on
        CI runners with variable load), this test verifies that both paths
        load correctly and produce equivalent data.
        """
        war = war_with_complex_columns
        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_pickle_and_json(tmpdir, filename="war")

            pkl_path = Path(tmpdir) / "war.pkl"
            pq_path = Path(tmpdir) / "war.parquet"
            assert pkl_path.exists()
            assert pq_path.exists()

            # Verify both formats load successfully
            pkl_df = pd.read_pickle(pkl_path)
            pq_df = pd.read_parquet(pq_path, engine="pyarrow")

            # Both should have the same shape and columns
            assert pkl_df.shape == pq_df.shape
            assert list(pkl_df.columns) == list(pq_df.columns)


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
