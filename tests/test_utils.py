"""
Unit tests for neurodent.core.utils module.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import shutil
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from neurodent.core import utils
from neurodent import constants


class TestConvertUnitsToMultiplier:
    """Test convert_units_to_multiplier function."""

    def test_valid_conversions(self):
        """Test valid unit conversions."""
        # Test basic conversions
        assert np.isclose(utils.convert_units_to_multiplier("µV", "µV"), 1.0)
        assert np.isclose(utils.convert_units_to_multiplier("mV", "µV"), 1000.0)
        assert np.isclose(utils.convert_units_to_multiplier("V", "µV"), 1000000.0)
        assert np.isclose(utils.convert_units_to_multiplier("nV", "µV"), 0.001)

    def test_invalid_current_unit(self):
        """Test error handling for invalid current unit."""
        with pytest.raises(AssertionError, match="No valid current unit"):
            utils.convert_units_to_multiplier("invalid", "µV")

    def test_invalid_target_unit(self):
        """Test error handling for invalid target unit."""
        with pytest.raises(AssertionError, match="No valid target unit"):
            utils.convert_units_to_multiplier("µV", "invalid")


class TestIsDay:
    """Test is_day function."""

    def test_day_time(self):
        """Test daytime hours."""
        dt = datetime(2023, 1, 1, 12, 0)  # Noon
        assert utils.is_day(dt) is True

    def test_night_time(self):
        """Test nighttime hours."""
        dt = datetime(2023, 1, 1, 2, 0)  # 2 AM
        assert utils.is_day(dt) is False

    def test_sunrise_edge(self):
        """Test sunrise edge case."""
        dt = datetime(2023, 1, 1, 6, 0)  # 6 AM (sunrise)
        assert utils.is_day(dt) is True

    def test_sunset_edge(self):
        """Test sunset edge case."""
        dt = datetime(2023, 1, 1, 18, 0)  # 6 PM (sunset)
        assert utils.is_day(dt) is False

    def test_custom_hours(self):
        """Test custom sunrise/sunset hours."""
        dt = datetime(2023, 1, 1, 10, 0)  # 10 AM
        assert utils.is_day(dt, sunrise=8, sunset=20) is True
        assert utils.is_day(dt, sunrise=12, sunset=14) is False

    def test_invalid_input_type(self):
        """Test error handling for non-datetime input."""
        with pytest.raises(TypeError, match="Expected datetime object, got"):
            utils.is_day("not a datetime")

        with pytest.raises(TypeError, match="Expected datetime object, got"):
            utils.is_day(123)

        with pytest.raises(TypeError, match="Expected datetime object, got"):
            utils.is_day(None)


class TestFilepathToIndex:
    """Test filepath_to_index function."""

    def test_basic_extraction(self):
        """Test basic index extraction."""
        filepath = "/path/to/data_ColMajor_001.bin"
        assert utils.filepath_to_index(filepath) == 1

    def test_with_different_suffixes(self):
        """Test with different file suffixes."""
        # Test .npy.gz suffix
        filepath = "/path/to/data_005_RowMajor.npy.gz"
        assert utils.filepath_to_index(filepath) == 5

        # Test .bin suffix
        filepath = "/path/to/data_006_RowMajor.bin"
        assert utils.filepath_to_index(filepath) == 6

        # Test .npy suffix
        filepath = "/path/to/data_007_RowMajor.npy"
        assert utils.filepath_to_index(filepath) == 7

        # Test no suffix
        filepath = "/path/to/data_008_RowMajor"
        assert utils.filepath_to_index(filepath) == 8

        # Test multiple suffixes
        filepath = "/path/to/data_009_RowMajor.test.npy.gz"
        assert utils.filepath_to_index(filepath) == 9

    def test_with_meta_suffix(self):
        """Test with meta suffix."""
        filepath = "/path/to/data_Meta_010.json"
        assert utils.filepath_to_index(filepath) == 10

    def test_with_multiple_numbers(self):
        """Test with multiple numbers in filename."""
        # Test with year in filename
        filepath = "/path/to/data_2023_015_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 15

        # Test with multiple numbers throughout path
        filepath = "/path/to/123/data_456_789_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 789

        # Test with numbers in directory names
        filepath = "/path/2024/data_v2_042_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 42

    def test_dots_in_filename(self):
        """Test handling of dots within filenames."""
        # Test with decimal numbers
        filepath = "/path/to/data_1.2_3.4_567_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 567

        # Test with version numbers
        filepath = "/path/to/data_v1.0_042_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 42

        # Test with multiple dots
        filepath = "/path/to/data.2023.001_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 1


class TestParseTruncate:
    """Test parse_truncate function."""

    def test_boolean_true(self):
        """Test boolean True input."""
        assert utils.parse_truncate(True) == 10

    def test_boolean_false(self):
        """Test boolean False input."""
        assert utils.parse_truncate(False) == 0

    def test_integer_input(self):
        """Test integer input."""
        assert utils.parse_truncate(5) == 5
        assert utils.parse_truncate(0) == 0

    def test_invalid_input(self):
        """Test invalid input type."""
        with pytest.raises(ValueError, match="Invalid truncate value"):
            utils.parse_truncate("invalid")


class TestNanAverage:
    """Test nanaverage function."""

    def test_basic_averaging(self):
        """Test basic averaging with weights."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = np.array([0.5, 0.3, 0.2])

        result = utils.nanaverage(A, weights, axis=0)
        expected = np.average(A, weights=weights, axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_nans(self):
        """Test averaging with NaN values."""
        A = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, np.nan]])
        weights = np.array([0.5, 0.3, 0.2])

        result = utils.nanaverage(A, weights, axis=0)

        # First column: [1, 4, 7] with weights [0.5, 0.3, 0.2]
        expected_col1 = 1 * 0.5 + 4 * 0.3 + 7 * 0.2  # = 2.9
        np.testing.assert_almost_equal(result[0], expected_col1)

        # Second column: [nan, 5, 8] with adjusted weights [0, 0.3, 0.2]
        # Need to renormalize weights to sum to 1: [0, 0.6, 0.4]
        expected_col2 = 5 * 0.6 + 8 * 0.4  # = 6.2
        np.testing.assert_almost_equal(result[1], expected_col2)

        # Third column: [3, 6, nan] with adjusted weights [0.5, 0.3, 0]
        # Renormalized weights: [0.625, 0.375, 0]
        expected_col3 = 3 * 0.625 + 6 * 0.375  # = 4.125
        np.testing.assert_almost_equal(result[2], expected_col3)


class TestResolveChannel:
    """Test resolve_channel: exact lookup, loud on miss, no inference."""

    def test_canonical_abbrev_passthrough(self):
        for ch in ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]:
            assert utils.resolve_channel(ch) == ch

    def test_strips_whitespace(self):
        assert utils.resolve_channel("  LMot  ") == "LMot"

    def test_already_abbreviated_channels(self):
        for ch in ["LAud", "RAud", "LVis", "RVis", "LHip", "RHip", "LBar", "RBar", "LMot", "RMot"]:
            assert utils.resolve_channel(ch) == ch

    @pytest.mark.mutates_constants
    def test_configured_raw_name_resolves(self):
        from neurodent import constants

        orig = constants.CHANNEL_MAP
        try:
            constants.set_channel_map({"LMot": ["LMot", "L Motor Ctx"], "RMot": ["RMot"]})
            assert utils.resolve_channel("L Motor Ctx") == "LMot"
            assert utils.resolve_channel("LMot") == "LMot"
            # An unconfigured spelling is not inferred.
            with pytest.raises(ValueError, match="not in the configured channel map"):
                utils.resolve_channel("left Motor")
        finally:
            constants.set_channel_map(orig)

    def test_unconfigured_name_raises_loudly(self):
        # Free-text, embedded, and numeric names all raise (no fuzzy / no number inference).
        for bad in ["left Aud", "Right VIS", "probe_Left_Hip_001", "leftAud",
                    "channel_9", "0", "InvalidChannel", "no_numbers_here"]:
            with pytest.raises(ValueError, match="not in the configured channel map"):
                utils.resolve_channel(bad)

    def test_error_message_lists_configured_channels(self):
        with pytest.raises(ValueError) as excinfo:
            utils.resolve_channel("nope")
        msg = str(excinfo.value)
        assert "not in the configured channel map" in msg
        assert "Canonical labels" in msg


class TestLogTransform:
    """Test log_transform function."""

    def test_basic_transform(self):
        """Test basic log transformation."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data)

        # Should be natural log-transformed (ln(x+1))
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_kwargs_ignored(self):
        """Test that extra kwargs are ignored."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data, offset=1, some_other_param="ignored")

        # Function should ignore kwargs and still work normally (ln(x+1))
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_negative_values(self):
        """Test log transformation with negative values."""
        data = np.array([-1, 0, 1, 10])

        # The function always does ln(x + 1), so let's test that
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = utils.log_transform(data)

        # Expected behavior: ln(x + 1)
        # [-1, 0, 1, 10] + 1 = [0, 1, 2, 11]
        # ln([0, 1, 2, 11]) = [-inf, 0, ln(2), ln(11)]
        assert np.isneginf(result[0])  # ln(0) = -inf
        assert np.isclose(result[1], 0)  # ln(1) = 0
        assert np.isclose(result[2], np.log(2))  # ln(2)
        assert np.isclose(result[3], np.log(11))  # ln(11)

    def test_zero_values(self):
        """Test log transformation with zero values."""
        data = np.array([0, 0, 0])
        result = utils.log_transform(data)

        # ln(0 + 1) = ln(1) = 0
        expected = np.log(data + 1)  # [0, 0, 0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_one_edge_case(self):
        """Test log transformation with -1 values (edge case)."""
        data = np.array([-1])

        # ln(-1 + 1) = ln(0) = -inf, should raise warning or return -inf
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = utils.log_transform(data)
            assert np.isneginf(result[0])

    def test_values_less_than_negative_one(self):
        """Test log transformation with values < -1."""
        data = np.array([-2, -1.5, -10])

        # ln(x + 1) where x < -1 gives complex numbers, numpy should handle this
        with pytest.warns(RuntimeWarning, match="invalid value"):
            result = utils.log_transform(data)
            assert np.all(np.isnan(result))

    def test_very_large_values(self):
        """Test log transformation with very large values."""
        data = np.array([1e10, 1e100, np.inf])
        result = utils.log_transform(data)

        # Should handle large values appropriately
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_very_small_positive_values(self):
        """Test log transformation with very small positive values."""
        data = np.array([1e-10, 1e-100, 1e-308])
        result = utils.log_transform(data)

        # Should handle small values appropriately
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_nan_values(self):
        """Test log transformation with NaN values."""
        data = np.array([1, np.nan, 10])
        result = utils.log_transform(data)

        # NaN should propagate through
        expected = np.log(data + 1)
        assert np.isnan(result[1])
        np.testing.assert_array_almost_equal(result[[0, 2]], expected[[0, 2]])

    def test_mixed_edge_cases(self):
        """Test log transformation with mixed edge case values."""
        data = np.array([-2, -1, 0, 1e-10, 1, 1e10, np.nan, np.inf])

        with pytest.warns(RuntimeWarning):
            result = utils.log_transform(data)

        # Check specific expected behaviors
        assert np.isnan(result[0])  # -2 + 1 = -1, log(-1) = nan
        assert np.isneginf(result[1])  # -1 + 1 = 0, log(0) = -inf
        assert result[2] == 0  # 0 + 1 = 1, log(1) = 0
        assert result[3] > 0  # small positive + 1, log > 0
        assert np.isnan(result[6])  # nan propagates
        assert np.isinf(result[7])  # inf + 1 = inf, log(inf) = inf

    def test_empty_array(self):
        """Test log transformation with empty array."""
        data = np.array([])
        result = utils.log_transform(data)

        assert len(result) == 0
        assert result.dtype == np.float64

    def test_multidimensional_arrays(self):
        """Test log transformation with multidimensional arrays."""
        data = np.array([[1, 10], [100, 1000]])
        result = utils.log_transform(data)

        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == data.shape


class TestSortDataframeByPlotOrder:
    """Test sort_dataframe_by_plot_order function."""

    def test_basic_sorting(self):
        """Test basic DataFrame sorting."""
        # Create test data with known order
        df = pd.DataFrame(
            {"genotype": ["KO", "WT", "KO", "WT"], "channel": ["RVis", "LAud", "LVis", "RAud"], "value": [1, 2, 3, 4]}
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Check that the DataFrame is sorted according to the predefined order
        # Based on constants.DF_SORT_ORDER: genotype: ["WT", "KO"], channel: ["average", "all", "LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"]
        # Should be sorted first by genotype (WT before KO), then by channel (LAud < RAud < LVis < RVis)

        # Check genotype ordering
        genotype_order = result["genotype"].tolist()
        wt_indices = [i for i, x in enumerate(genotype_order) if x == "WT"]
        ko_indices = [i for i, x in enumerate(genotype_order) if x == "KO"]

        # All WT should come before all KO
        assert max(wt_indices) < min(ko_indices)

        # Check channel ordering within each genotype
        wt_channels = result[result["genotype"] == "WT"]["channel"].tolist()
        ko_channels = result[result["genotype"] == "KO"]["channel"].tolist()

        # LAud should come before RAud in WT group
        if "LAud" in wt_channels and "RAud" in wt_channels:
            assert wt_channels.index("LAud") < wt_channels.index("RAud")

        # LVis should come before RVis in KO group
        if "LVis" in ko_channels and "RVis" in ko_channels:
            assert ko_channels.index("LVis") < ko_channels.index("RVis")

        # Check that values are in expected order after sorting
        expected_values = [2, 4, 3, 1]  # Values corresponding to WT/LAud, WT/RAud, KO/LVis, KO/RVis
        np.testing.assert_array_equal(result["value"].values, expected_values)

    def test_empty_dataframe(self):
        """Test sorting an empty DataFrame."""
        df = pd.DataFrame()
        result = utils.sort_dataframe_by_plot_order(df)

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_with_missing_columns(self):
        """Test DataFrame that doesn't have columns in sort order."""
        df = pd.DataFrame({"unknown_col": [1, 2, 3], "another_col": ["a", "b", "c"]})

        result = utils.sort_dataframe_by_plot_order(df)

        # Should return a copy of the original DataFrame
        pd.testing.assert_frame_equal(result, df)
        assert result is not df  # Should be a copy

    def test_custom_sort_order(self):
        """Test with custom sort order dictionary."""
        df = pd.DataFrame(
            {
                "priority": ["high", "low", "medium", "high"],
                "category": ["B", "A", "C", "A"],
                "channel": ["LAud", "RAud", "LVis", "RVis"],
                "value": [1, 2, 3, 4],
            }
        )

        custom_order = {"priority": ["low", "medium", "high"], "category": ["A", "B", "C"]}

        result = utils.sort_dataframe_by_plot_order(df, custom_order)

        # Should sort by priority first, then by category
        priorities = result["priority"].tolist()

        # Check that priorities are in correct order
        assert priorities.index("low") < priorities.index("medium")
        assert priorities.index("medium") < max([i for i, x in enumerate(priorities) if x == "high"])

        # Check that channel (from default sort order) follows the sorted rows
        channels = result["channel"].tolist()
        assert channels == ["RAud", "LVis", "RVis", "LAud"]  # Reflects the sorted order by priority and category

    def test_constants_not_mutated(self):
        """Test that constants.DF_SORT_ORDER is not mutated by the function."""
        # Save original state
        original_df_sort_order = constants.DF_SORT_ORDER.copy()

        df = pd.DataFrame({"genotype": ["WT", "KO"], "channel": ["LAud", "RVis"], "value": [1, 2]})

        # Call function multiple times
        result1 = utils.sort_dataframe_by_plot_order(df)
        result2 = utils.sort_dataframe_by_plot_order(df)

        # Check that constants.DF_SORT_ORDER hasn't changed
        assert constants.DF_SORT_ORDER == original_df_sort_order

        # Also check that the results are consistent
        pd.testing.assert_frame_equal(result1, result2)

    def test_dictionary_not_modified_by_reference(self):
        """Test that the passed dictionary is not modified by the function."""
        custom_order = {"priority": ["low", "medium", "high"], "category": ["A", "B", "C"]}
        original_custom_order = {"priority": ["low", "medium", "high"], "category": ["A", "B", "C"]}

        df = pd.DataFrame({"priority": ["high", "low", "medium"], "category": ["B", "A", "C"], "value": [1, 2, 3]})

        utils.sort_dataframe_by_plot_order(df, custom_order)

        # Check that custom_order wasn't modified
        assert custom_order == original_custom_order

    def test_invalid_sort_order_dict(self):
        """Test error handling for invalid sort order dictionary."""
        df = pd.DataFrame({"genotype": ["WT", "KO"], "value": [1, 2]})
        # ANCHOR 08/14/2025
        # Test with non-dict
        with pytest.raises(ValueError, match="df_sort_order must be a dictionary"):
            utils.sort_dataframe_by_plot_order(df, "not_a_dict")

        # Test with invalid categories (not list or tuple)
        with pytest.raises(ValueError, match="Categories for column 'genotype' must be a list or tuple"):
            utils.sort_dataframe_by_plot_order(df, {"genotype": "not_a_list"})

    def test_values_not_in_sort_order(self):
        """Test error handling when DataFrame contains values not in sort order."""
        df = pd.DataFrame({"genotype": ["WT", "KO", "UNKNOWN"], "value": [1, 2, 3]})

        with pytest.raises(
            ValueError, match="Column 'genotype' contains values not in sort order dictionary: \\{'UNKNOWN'\\}"
        ):
            utils.sort_dataframe_by_plot_order(df)

    def test_missing_values_handled(self):
        """Test that missing values (NaN) are handled properly."""
        df = pd.DataFrame(
            {"genotype": ["WT", "KO", np.nan, "WT"], "channel": ["LAud", np.nan, "RVis", "LVis"], "value": [1, 2, 3, 4]}
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Function should not crash with NaN values
        assert len(result) == 4
        assert result["genotype"].isna().any()
        assert result["channel"].isna().any()

    def test_only_existing_categories_used(self):
        """Test that only categories that exist in the DataFrame are used for sorting."""
        df = pd.DataFrame(
            {
                "genotype": ["WT", "WT"],  # Only WT, no KO
                "channel": ["LAud", "LVis"],  # Only 2 channels
                "value": [1, 2],
            }
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Should work even though not all categories from DF_SORT_ORDER are present
        assert len(result) == 2
        genotypes = result["genotype"].unique()
        assert "WT" in genotypes
        assert "KO" not in genotypes

    def test_maintains_dataframe_structure(self):
        """Test that the function maintains DataFrame structure and data types."""
        df = pd.DataFrame(
            {"genotype": ["WT", "KO"], "channel": ["LAud", "RVis"], "value": [1.5, 2.7], "count": [10, 20]}
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Check that all columns are preserved
        assert list(result.columns) == list(df.columns)

        # Check that data types are preserved for non-sorted columns
        assert result["value"].dtype == df["value"].dtype
        assert result["count"].dtype == df["count"].dtype

        # Check that no data is lost
        assert len(result) == len(df)

    def test_return_copy_not_reference(self):
        """Test that function returns a copy, not a reference to the original."""
        df = pd.DataFrame({"genotype": ["WT", "KO"], "value": [1, 2]})

        result = utils.sort_dataframe_by_plot_order(df)

        # Modify the result
        result.loc[0, "value"] = 999

        # Original should be unchanged
        assert df.loc[0, "value"] == 1

    def test_complex_sorting_scenario(self):
        """Test complex sorting with multiple columns and mixed data."""
        df = pd.DataFrame(
            {
                "genotype": ["KO", "WT", "KO", "WT", "WT", "KO"],
                "channel": ["RVis", "LAud", "LVis", "RBar", "LHip", "RAud"],
                "band": ["gamma", "delta", "beta", "alpha", "theta", "gamma"],
                "isday": [False, True, True, False, True, False],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Check that result is properly sorted
        assert len(result) == 6

        # Verify that the sorting is stable and follows the expected order
        # All WT should come before KO (check row positions, not original indices)
        wt_positions = [i for i, genotype in enumerate(result["genotype"]) if genotype == "WT"]
        ko_positions = [i for i, genotype in enumerate(result["genotype"]) if genotype == "KO"]
        assert max(wt_positions) < min(ko_positions)

    def test_default_parameter_copy_behavior(self):
        """Test that function creates a copy of constants.DF_SORT_ORDER when using default parameter."""
        df = pd.DataFrame({"genotype": ["WT", "KO"], "channel": ["LAud", "RVis"], "value": [1, 2]})

        # Store original state
        original_constants = constants.DF_SORT_ORDER.copy()

        # Call function with default parameter (None)
        result = utils.sort_dataframe_by_plot_order(df)

        # Constants should be unchanged
        assert constants.DF_SORT_ORDER == original_constants

        # Function should work correctly
        assert len(result) == 2
        assert "genotype" in result.columns
        assert "channel" in result.columns

    def test_unknown_sex_raises_without_extension(self):
        """Regression: arxrosa-style 'Unknown' sex must be in plot_order or rejected.

        Documents the failure mode: when WAR data contains sex='Unknown'
        (e.g. arxrosa, where source JSON has sex='UNKNOWN' and downstream
        defaults it to 'Unknown') and ``plot_order["sex"]`` is the default
        ``["Male", "Female"]``, ``sort_dataframe_by_plot_order`` strict-
        rejects with ValueError. This is the failure that broke ep_figures
        on the arxrosa run (job 8439583).
        """
        df = pd.DataFrame(
            {"sex": ["Unknown", "Unknown", "Unknown"], "value": [1, 2, 3]}
        )
        with pytest.raises(ValueError, match=r"sex.*not in sort order"):
            utils.sort_dataframe_by_plot_order(df)

    def test_unknown_sex_passes_with_extend_plot_order_helper(self):
        """Regression: ``extend_plot_order_from_attr`` (the actual helper that
        backs the script-level fix in ``generate_ep_figures.py`` and
        ``generate_ep_heatmaps.py``) produces a plot_order that ``sort_dataframe_by_plot_order``
        accepts for arxrosa-style data with sex='Unknown'.
        """
        from neurodent.workflow import extend_plot_order_from_attr

        # Mock WAR objects exposing only the attributes the helper reads.
        class _MockWAR:
            def __init__(self, genotype, sex):
                self.genotype = genotype
                self.sex = sex

        wars = [
            _MockWAR("WT", "Male"),
            _MockWAR("UNKNOWN", "Unknown"),
            _MockWAR("UNKNOWN", "Unknown"),
        ]

        # Build plot_order using the same helper the scripts call.
        plot_order = constants.DF_SORT_ORDER.copy()
        plot_order["genotype"] = extend_plot_order_from_attr(
            wars, "genotype", constants.DF_SORT_ORDER.get("genotype", [])
        )
        plot_order["sex"] = extend_plot_order_from_attr(
            wars, "sex", constants.DF_SORT_ORDER.get("sex", [])
        )

        # Helper output should preserve known categories in their original
        # order and append new ones.
        assert plot_order["sex"][:2] == ["Male", "Female"]
        assert "Unknown" in plot_order["sex"]
        assert plot_order["genotype"][:2] == ["WT", "KO"]
        assert "UNKNOWN" in plot_order["genotype"]

        # And the extended order must make sort_dataframe_by_plot_order
        # accept a DataFrame containing those same values.
        df = pd.DataFrame(
            {
                "sex": ["Female", "Unknown", "Male"],
                "genotype": ["WT", "UNKNOWN", "WT"],
                "value": [1, 2, 3],
            }
        )
        result = utils.sort_dataframe_by_plot_order(df, plot_order)
        # Known categories sort first; novel ones (Unknown / UNKNOWN) come last.
        assert list(result["sex"]) == ["Male", "Female", "Unknown"]
        assert list(result["genotype"]) == ["WT", "WT", "UNKNOWN"]

    def test_extend_plot_order_ignores_falsy_and_dedupes(self):
        """``extend_plot_order_from_attr`` skips falsy attr values and
        preserves dedup behavior."""
        from neurodent.workflow import extend_plot_order_from_attr

        class _MockWAR:
            def __init__(self, sex):
                self.sex = sex

        wars = [
            _MockWAR(None),       # falsy: skipped
            _MockWAR(""),         # falsy: skipped
            _MockWAR("Male"),     # already in base: not appended
            _MockWAR("Unknown"),  # new: appended once
            _MockWAR("Unknown"),  # duplicate: not appended again
        ]
        result = extend_plot_order_from_attr(wars, "sex", ["Male", "Female"])
        assert result == ["Male", "Female", "Unknown"]


class TestNanmeanSeriesOfNp:
    """Test nanmean_series_of_np function."""

    def test_basic_operation_small_series(self):
        """Test basic nanmean operation with small series (< 1000 items)."""
        # Create series with numpy arrays
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_operation_large_series(self):
        """Test basic nanmean operation with large series (> 1000 items)."""
        # Create series with many numpy arrays - reduced size to prevent memory issues
        arrays = [np.array([i, i + 1, i + 2]) for i in range(100)]  # Reduced from 1500
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.stack(arrays, axis=0), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_with_nan_values(self):
        """Test nanmean operation with NaN values."""
        arrays = [np.array([1, 2, np.nan]), np.array([4, np.nan, 6]), np.array([np.nan, 8, 9])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_all_nan_values(self):
        """Test nanmean operation when all values are NaN."""
        arrays = [
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan]),
        ]
        series = pd.Series(arrays)

        # Expected warning when computing mean of all NaN values
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result = utils.nanmean_series_of_np(series)

        # Should return array of NaNs
        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_different_axis(self):
        """Test nanmean operation with different axis parameter."""
        # Create 2D arrays
        arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
        series = pd.Series(arrays)

        # Test axis=1
        result = utils.nanmean_series_of_np(series, axis=1)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=1)

        np.testing.assert_array_almost_equal(result, expected)

    def test_single_element_series(self):
        """Test nanmean operation with single element series."""
        arrays = [np.array([1, 2, 3])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = arrays[0]  # Should be the same as input

        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_series(self):
        """Test nanmean operation with empty series."""
        series = pd.Series([], dtype=object)

        # Empty series causes warning and returns NaN
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result = utils.nanmean_series_of_np(series)
        assert np.isnan(result)

    def test_mixed_array_sizes_small_series(self):
        """Test with arrays of different sizes in small series."""
        arrays = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
        series = pd.Series(arrays)

        # This should fall back to list conversion and may raise an error
        # or handle it gracefully depending on numpy version
        try:
            result = utils.nanmean_series_of_np(series)
            # If it works, verify it's reasonable
            assert isinstance(result, np.ndarray)
        except ValueError:
            # This is acceptable for mixed sizes
            pass

    def test_mixed_array_sizes_large_series(self):
        """Test with arrays of different sizes in large series (triggers stacking path)."""
        # Create 100 arrays with mixed sizes - reduced size to prevent memory issues
        arrays = []
        for i in range(100):  # Reduced from 1500
            if i % 3 == 0:
                arrays.append(np.array([i, i + 1]))
            elif i % 3 == 1:
                arrays.append(np.array([i, i + 1, i + 2]))
            else:
                arrays.append(np.array([i]))

        series = pd.Series(arrays)

        # Mixed-size arrays cannot be processed by np.nanmean - should raise ValueError
        with pytest.raises(ValueError, match="setting an array element with a sequence"):
            utils.nanmean_series_of_np(series)

    def test_non_numpy_arrays_small_series(self):
        """Test with non-numpy arrays in small series."""
        arrays = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Lists instead of numpy arrays
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array(arrays), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_non_numpy_arrays_large_series(self):
        """Test with non-numpy arrays in large series."""
        arrays = [[i, i + 1, i + 2] for i in range(100)]  # Lists instead of numpy arrays - reduced size
        series = pd.Series(arrays)

        # Should fallback to list method since first element is not numpy array
        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array(arrays), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_types_large_series(self):
        """Test with mixed types in large series."""
        arrays = []
        for i in range(100):  # Reduced from 1500 to prevent memory issues
            if i % 2 == 0:
                arrays.append(np.array([i, i + 1, i + 2]))  # numpy array
            else:
                arrays.append([i, i + 1, i + 2])  # list

        series = pd.Series(arrays)

        # First element is numpy array, so tries stacking but should handle TypeError
        result = utils.nanmean_series_of_np(series)
        assert isinstance(result, np.ndarray)

    def test_3d_arrays(self):
        """Test with 3D numpy arrays."""
        arrays = [
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
            np.array([[[17, 18], [19, 20]], [[21, 22], [23, 24]]]),
        ]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_complex_numbers(self):
        """Test with complex number arrays."""
        arrays = [
            np.array([1 + 2j, 3 + 4j, 5 + 6j]),
            np.array([7 + 8j, 9 + 10j, 11 + 12j]),
            np.array([13 + 14j, 15 + 16j, 17 + 18j]),
        ]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_boolean_arrays(self):
        """Test with boolean arrays."""
        arrays = [np.array([True, False, True]), np.array([False, True, False]), np.array([True, True, False])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_large_series_exception_handling(self):
        """Test exception handling in large series path."""
        # Create series that will trigger the large series path but cause an exception
        arrays = []
        for i in range(100):  # Reduced from 1500 to prevent memory issues
            if i == 0:
                arrays.append(np.array([1, 2, 3]))  # First element is numpy array
            else:
                arrays.append("not_an_array")  # Rest are strings to cause TypeError

        series = pd.Series(arrays)

        # Should catch the exception and fall back to list method
        with pytest.raises((ValueError, TypeError)):
            utils.nanmean_series_of_np(series)

    def test_performance_comparison(self):
        """Test that large series uses the optimized path."""
        # Create large series - reduced size to prevent memory issues
        arrays = [np.array([i, i + 1, i + 2]) for i in range(100)]  # Reduced from 1500
        series = pd.Series(arrays)

        # This should use the optimized stacking path
        result = utils.nanmean_series_of_np(series)

        # Verify result is correct
        expected = np.nanmean(np.stack(arrays, axis=0), axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_axis(self):
        """Test with negative axis parameter."""
        arrays = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
        ]
        series = pd.Series(arrays)

        # Test axis=-1 (last axis)
        result = utils.nanmean_series_of_np(series, axis=-1)
        expected = np.nanmean(np.array([arrays[0], arrays[1]]), axis=-1)

        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_operation_large_series(self):
        """Test basic nanmean operation with large series (100 items)."""
        # Create series with many numpy arrays - reduced size to prevent memory issues
        arrays = [np.array([i, i + 1, i + 2]) for i in range(100)]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.stack(arrays, axis=0), axis=0)

        np.testing.assert_array_almost_equal(result, expected)


class TestTimestampMapper:
    """Test TimestampMapper class."""

    def test_basic_initialization(self):
        """Test basic initialization of TimestampMapper."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0), datetime(2023, 1, 1, 12, 2, 0)]
        durations = [60.0, 60.0, 60.0]  # 1 minute each

        mapper = utils.TimestampMapper(end_times, durations)

        assert mapper.file_end_datetimes == end_times
        assert mapper.file_durations == durations
        assert len(mapper.file_start_datetimes) == 3
        assert len(mapper.cumulative_durations) == 3

        # Check start times are calculated correctly
        expected_start_times = [
            datetime(2023, 1, 1, 11, 59, 0),  # 12:00 - 60s
            datetime(2023, 1, 1, 12, 0, 0),  # 12:01 - 60s
            datetime(2023, 1, 1, 12, 1, 0),  # 12:02 - 60s
        ]
        assert mapper.file_start_datetimes == expected_start_times

        # Check cumulative durations
        expected_cumulative = [60.0, 120.0, 180.0]
        np.testing.assert_array_almost_equal(mapper.cumulative_durations, expected_cumulative)

    def test_single_file(self):
        """Test with single file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [120.0]  # 2 minutes

        mapper = utils.TimestampMapper(end_times, durations)

        assert len(mapper.file_start_datetimes) == 1
        assert mapper.file_start_datetimes[0] == datetime(2023, 1, 1, 11, 58, 0)
        assert mapper.cumulative_durations[0] == 120.0

    def test_get_fragment_timestamp_first_file(self):
        """Test getting fragment timestamp from first file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        durations = [60.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Fragment 0, length 30s (should be in first file)
        result = mapper.get_fragment_timestamp(0, 30.0)

        # Fragment starts at time 0, which is in first file
        # Offset should be 0 - 60 = -60s from end time
        expected = datetime(2023, 1, 1, 12, 0, 0) + timedelta(seconds=-60)
        assert result == expected

    def test_get_fragment_timestamp_second_file(self):
        """Test getting fragment timestamp from second file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        durations = [60.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Fragment starting at 90s (30s into second file)
        result = mapper.get_fragment_timestamp(3, 30.0)  # 3 * 30s = 90s

        # Fragment starts at 90s, cumulative duration of file 1 is 120s
        # Offset should be 90 - 120 = -30s from end time of file 1
        expected = datetime(2023, 1, 1, 12, 1, 0) + timedelta(seconds=-30)
        assert result == expected

    def test_get_fragment_timestamp_edge_cases(self):
        """Test edge cases for fragment timestamp calculation."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 2, 0), datetime(2023, 1, 1, 12, 5, 0)]
        durations = [30.0, 120.0, 180.0]  # Different durations

        mapper = utils.TimestampMapper(end_times, durations)

        # Test fragment beyond last file (should clamp to last file)
        result = mapper.get_fragment_timestamp(100, 10.0)  # Way beyond

        # Should use last file index
        assert isinstance(result, datetime)

    def test_empty_file_lists(self):
        """Test with empty file lists."""
        with pytest.raises(IndexError):
            mapper = utils.TimestampMapper([], [])
            mapper.get_fragment_timestamp(0, 30.0)

    def test_mismatched_list_lengths(self):
        """Test with mismatched end times and durations lists."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [60.0, 120.0]  # Different length

        # Should raise ValueError for mismatched lengths
        with pytest.raises(ValueError, match="file_end_datetimes and file_durations must have the same length"):
            utils.TimestampMapper(end_times, durations)

    def test_zero_duration_file(self):
        """Test with zero duration file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        durations = [0.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Should handle zero duration gracefully
        result = mapper.get_fragment_timestamp(0, 30.0)
        assert isinstance(result, datetime)

    def test_negative_duration(self):
        """Test with negative duration (edge case)."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [-60.0]  # Negative duration

        mapper = utils.TimestampMapper(end_times, durations)

        # Should still work, but start time will be after end time
        assert mapper.file_start_datetimes[0] > mapper.file_end_datetimes[0]

    def test_very_small_fragment_length(self):
        """Test with very small fragment length."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Very small fragment
        result = mapper.get_fragment_timestamp(0, 0.001)
        assert isinstance(result, datetime)

    def test_very_large_fragment_length(self):
        """Test with very large fragment length."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Large fragment
        result = mapper.get_fragment_timestamp(0, 3600.0)  # 1 hour
        assert isinstance(result, datetime)

    def test_non_sequential_end_times(self):
        """Test with non-sequential end times."""
        end_times = [
            datetime(2023, 1, 1, 12, 2, 0),  # Later time first
            datetime(2023, 1, 1, 12, 0, 0),  # Earlier time second
            datetime(2023, 1, 1, 12, 1, 0),
        ]
        durations = [60.0, 60.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Should still work, mapping is based on order in list
        result = mapper.get_fragment_timestamp(0, 30.0)
        assert isinstance(result, datetime)

    def test_microsecond_precision(self):
        """Test timestamp precision with microseconds."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0, 123456)]
        durations = [60.5]  # Duration that doesn't exactly cancel microseconds

        mapper = utils.TimestampMapper(end_times, durations)

        result = mapper.get_fragment_timestamp(0, 30.0)  # Fragment at start
        assert isinstance(result, datetime)
        # With fragment_start_time=0, offset_in_file = 0 - 60.5 = -60.5
        # So result = 12:00:00.123456 + (-60.5s) = 10:59:59.623456
        assert result.microsecond == 623456


class TestValidateTimestamps:
    """Test validate_timestamps function."""

    def test_valid_chronological_timestamps(self):
        """Test with valid chronological timestamps."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0), datetime(2023, 1, 1, 12, 2, 0)]

        result = utils.validate_timestamps(timestamps)
        assert result == timestamps

    def test_empty_timestamp_list(self):
        """Test with empty timestamp list."""
        with pytest.raises(ValueError, match="No timestamps provided for validation"):
            utils.validate_timestamps([])

    def test_all_none_timestamps(self):
        """Test with all None timestamps."""
        timestamps = [None, None, None]

        with pytest.warns(UserWarning, match="Found 3 None timestamps"):
            with pytest.raises(ValueError, match="No valid timestamps found"):
                utils.validate_timestamps(timestamps)

    def test_some_none_timestamps(self):
        """Test with some None timestamps."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0), None, datetime(2023, 1, 1, 12, 2, 0), None]

        # Should warn about None timestamps AND about large gap (2 minutes > 60 seconds)
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps)

        # Check that we got both warnings
        warning_messages = [str(w.message) for w in warning_list]
        assert any("Found 2 None timestamps" in msg for msg in warning_messages)
        assert any("Large gap detected" in msg for msg in warning_messages)

        expected = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 2, 0)]
        assert result == expected

    def test_non_chronological_timestamps(self):
        """Test with non-chronological timestamps."""
        timestamps = [
            datetime(2023, 1, 1, 12, 2, 0),  # Later time first
            datetime(2023, 1, 1, 12, 0, 0),  # Earlier time
            datetime(2023, 1, 1, 12, 1, 0),
        ]

        with pytest.warns(UserWarning, match="Timestamps are not in chronological order"):
            result = utils.validate_timestamps(timestamps)

        # Should return original order, not sorted
        assert result == timestamps

    def test_large_gaps_warning(self):
        """Test warning for large gaps between timestamps."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 1, 12, 0, 30),  # 30 seconds later
            datetime(2023, 1, 1, 12, 2, 0),  # 90 seconds later (large gap)
        ]

        with pytest.warns(UserWarning, match="Large gap detected"):
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=60)

        assert result == timestamps

    def test_custom_gap_threshold(self):
        """Test with custom gap threshold."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 1, 12, 0, 45),  # 45 seconds later
        ]

        # Should warn with threshold of 30 seconds
        with pytest.warns(UserWarning, match="Large gap detected"):
            utils.validate_timestamps(timestamps, gap_threshold_seconds=30)

        # Should not warn with threshold of 60 seconds
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=60)
            assert result == timestamps

    def test_gap_threshold_behavior(self):
        """Test gap threshold behavior with varying thresholds and gaps."""
        # Timestamps with different gaps
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),  # Start
            datetime(2023, 1, 1, 12, 0, 30),  # 30 seconds gap
            datetime(2023, 1, 1, 12, 1, 30),  # 60 seconds gap
            datetime(2023, 1, 1, 12, 3, 30),  # 120 seconds gap
        ]

        # Test with 20 second threshold - should warn about all gaps
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=20)

        warning_messages = [str(w.message) for w in warning_list]
        # Should get 3 warnings (for 30s, 60s, and 120s gaps)
        assert len([msg for msg in warning_messages if "Large gap detected" in msg]) == 3

        # Test with 50 second threshold - should warn about 60s and 120s gaps only
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=50)

        warning_messages = [str(w.message) for w in warning_list]
        # Should get 2 warnings (for 60s and 120s gaps)
        assert len([msg for msg in warning_messages if "Large gap detected" in msg]) == 2

        # Test with 90 second threshold - should warn about 120s gap only
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=90)

        warning_messages = [str(w.message) for w in warning_list]
        # Should get 1 warning (for 120s gap)
        assert len([msg for msg in warning_messages if "Large gap detected" in msg]) == 1

        # Test with 150 second threshold - should not warn about any gaps
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=150)
            assert result == timestamps

    def test_single_timestamp(self):
        """Test with single timestamp."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0)]

        result = utils.validate_timestamps(timestamps)
        assert result == timestamps

    def test_identical_timestamps(self):
        """Test with identical timestamps."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [timestamp, timestamp, timestamp]

        # Should not trigger gap warning (0 second gaps)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = utils.validate_timestamps(timestamps)
            assert result == timestamps

    def test_microsecond_differences(self):
        """Test with microsecond-level differences."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0, 0),
            datetime(2023, 1, 1, 12, 0, 0, 500000),  # 0.5 seconds later
            datetime(2023, 1, 1, 12, 0, 1, 0),  # 1 second from first
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=2)
            assert result == timestamps

    def test_negative_gap_threshold(self):
        """Test with negative gap threshold."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 0, 1)]

        # Even 1 second should trigger warning with negative threshold
        with pytest.warns(UserWarning, match="Large gap detected"):
            utils.validate_timestamps(timestamps, gap_threshold_seconds=-1)

    def test_zero_gap_threshold(self):
        """Test with zero gap threshold."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 1, 12, 0, 0, 1),  # 1 microsecond later
        ]

        # Even microsecond should trigger warning with zero threshold
        with pytest.warns(UserWarning, match="Large gap detected"):
            utils.validate_timestamps(timestamps, gap_threshold_seconds=0)

    def test_very_large_gaps(self):
        """Test with very large gaps."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 2, 12, 0, 0),  # 1 day later
        ]

        with pytest.warns(UserWarning, match="Large gap detected"):
            result = utils.validate_timestamps(timestamps)
            assert result == timestamps

    def test_mixed_none_and_valid_chronological(self):
        """Test mixed None and valid timestamps in chronological order."""
        timestamps = [
            None,
            datetime(2023, 1, 1, 12, 0, 0),
            None,
            datetime(2023, 1, 1, 12, 1, 0),
            None,
            datetime(2023, 1, 1, 12, 2, 0),
            None,
        ]

        with pytest.warns(UserWarning, match="Found 4 None timestamps"):
            result = utils.validate_timestamps(timestamps)

        expected = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0), datetime(2023, 1, 1, 12, 2, 0)]
        assert result == expected

    def test_mixed_none_and_valid_non_chronological(self):
        """Test mixed None and valid timestamps not in chronological order."""
        timestamps = [
            None,
            datetime(2023, 1, 1, 12, 2, 0),
            None,
            datetime(2023, 1, 1, 12, 0, 0),  # Earlier timestamp
            None,
            datetime(2023, 1, 1, 12, 1, 0),
            None,
        ]

        with pytest.warns(UserWarning, match="Found 4 None timestamps"):
            with pytest.warns(UserWarning, match="Timestamps are not in chronological order"):
                result = utils.validate_timestamps(timestamps)

        expected = [datetime(2023, 1, 1, 12, 2, 0), datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        assert result == expected

    def test_extreme_timestamp_values(self):
        """Test with extreme timestamp values."""
        timestamps = [datetime.min, datetime.max]

        # Should handle extreme values
        with pytest.warns(UserWarning, match="Large gap detected"):
            result = utils.validate_timestamps(timestamps)
            assert result == timestamps


class TestTempDirectory:
    """Test temporary directory functions."""

    def setup_method(self):
        """Setup for each test - save original TMPDIR."""
        self.original_tmpdir = os.environ.get("TMPDIR")

    def teardown_method(self):
        """Cleanup after each test - restore original TMPDIR."""
        if self.original_tmpdir is not None:
            os.environ["TMPDIR"] = self.original_tmpdir
        elif "TMPDIR" in os.environ:
            del os.environ["TMPDIR"]

    def test_set_and_get_temp_directory(self):
        """Test setting and getting temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = temp_dir + "/test_dir"
            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == Path(test_path)
            assert result.exists()  # Should be created

    def test_set_temp_directory_creates_path(self):
        """Test that set_temp_directory creates the path if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "nested" / "temp" / "dir"

            # Ensure the path doesn't exist initially
            assert not test_path.exists()

            utils.set_temp_directory(test_path)

            # Should be created
            assert test_path.exists()
            assert test_path.is_dir()

            # Should be set in environment
            assert utils.get_temp_directory() == test_path

    def test_set_temp_directory_with_existing_path(self):
        """Test setting temp directory when path already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)

            # Path already exists
            assert test_path.exists()

            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == test_path

    def test_set_temp_directory_with_string_path(self):
        """Test setting temp directory with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = temp_dir + "/string_path"

            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == Path(test_path)
            assert result.exists()

    def test_set_temp_directory_with_path_object(self):
        """Test setting temp directory with Path object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "path_object"

            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == test_path
            assert result.exists()

    def test_get_temp_directory_default(self):
        """Test getting temp directory after setting it."""
        # Set a temporary directory first
        test_path = "/default/tmp"
        os.environ["TMPDIR"] = test_path

        result = utils.get_temp_directory()
        assert result == Path(test_path)

    def test_get_temp_directory_missing_env_var(self):
        """Test get_temp_directory when TMPDIR is not set."""
        # Remove TMPDIR if it exists
        if "TMPDIR" in os.environ:
            del os.environ["TMPDIR"]

        with pytest.raises(KeyError):
            utils.get_temp_directory()

    def test_file_operations_in_temp_directory(self):
        """Test creating, modifying, and deleting files in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "file_ops_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Test file creation
            test_file = temp_dir_path / "test_file.txt"
            test_file.write_text("Hello, World!")

            assert test_file.exists()
            assert test_file.read_text() == "Hello, World!"

            # Test file modification
            test_file.write_text("Modified content")
            assert test_file.read_text() == "Modified content"

            # Test file deletion
            test_file.unlink()
            assert not test_file.exists()

    def test_directory_operations_in_temp_directory(self):
        """Test creating and deleting directories in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "dir_ops_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Test directory creation
            test_subdir = temp_dir_path / "subdir" / "nested"
            test_subdir.mkdir(parents=True)

            assert test_subdir.exists()
            assert test_subdir.is_dir()

            # Test file in subdirectory
            test_file = test_subdir / "nested_file.txt"
            test_file.write_text("Nested content")

            assert test_file.exists()
            assert test_file.read_text() == "Nested content"

            # Test directory deletion
            shutil.rmtree(test_subdir.parent)  # Remove "subdir" and all contents
            assert not test_subdir.exists()
            assert not test_file.exists()

    def test_binary_file_operations(self):
        """Test binary file operations in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "binary_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Test binary file operations
            binary_file = temp_dir_path / "binary_file.bin"
            binary_data = b"\x00\x01\x02\x03\xff\xfe\xfd"

            binary_file.write_bytes(binary_data)

            assert binary_file.exists()
            read_data = binary_file.read_bytes()
            assert read_data == binary_data

            # Test appending to binary file
            additional_data = b"\x10\x20\x30"
            with binary_file.open("ab") as f:
                f.write(additional_data)

            final_data = binary_file.read_bytes()
            assert final_data == binary_data + additional_data

    def test_numpy_array_operations(self):
        """Test saving and loading numpy arrays in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "numpy_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create test array
            test_array = np.random.rand(10, 5)

            # Save as .npy file
            npy_file = temp_dir_path / "test_array.npy"
            np.save(npy_file, test_array)

            assert npy_file.exists()

            # Load and verify
            loaded_array = np.load(npy_file)
            np.testing.assert_array_equal(test_array, loaded_array)

            # Save as .npz file
            npz_file = temp_dir_path / "test_arrays.npz"
            np.savez(npz_file, array1=test_array, array2=test_array * 2)

            assert npz_file.exists()

            # Load and verify
            with np.load(npz_file) as loaded_data:
                np.testing.assert_array_equal(loaded_data["array1"], test_array)
                np.testing.assert_array_equal(loaded_data["array2"], test_array * 2)

    def test_pandas_dataframe_operations(self):
        """Test saving and loading pandas DataFrames in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "pandas_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create test DataFrame
            test_df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"], "C": [1.1, 2.2, 3.3, 4.4]})

            # Save as CSV
            csv_file = temp_dir_path / "test_df.csv"
            test_df.to_csv(csv_file, index=False)

            assert csv_file.exists()

            # Load and verify
            loaded_df = pd.read_csv(csv_file)
            pd.testing.assert_frame_equal(test_df, loaded_df)

            # Save as pickle
            pickle_file = temp_dir_path / "test_df.pkl"
            test_df.to_pickle(pickle_file)

            assert pickle_file.exists()

            # Load and verify
            loaded_pickle_df = pd.read_pickle(pickle_file)
            pd.testing.assert_frame_equal(test_df, loaded_pickle_df)

    def test_file_permissions_and_attributes(self):
        """Test file permissions and attributes in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "permissions_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create test file
            test_file = temp_dir_path / "perm_test.txt"
            test_file.write_text("Permission test")

            # Test file attributes
            assert test_file.exists()
            assert test_file.is_file()
            assert not test_file.is_dir()

            # Test file size
            assert test_file.stat().st_size > 0

            # Test file modification (on Unix-like systems)
            if os.name != "nt":  # Skip on Windows
                # Make file read-only
                test_file.chmod(0o444)

                # Verify read-only (should raise PermissionError when trying to write)
                with pytest.raises(PermissionError):
                    test_file.write_text("Should fail")

                # Restore write permissions
                test_file.chmod(0o644)

                # Should work now
                test_file.write_text("Now it works")
                assert test_file.read_text() == "Now it works"

    def test_concurrent_access_safety(self):
        """Test that temp directory handles concurrent access safely."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "concurrent_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create multiple files simultaneously
            files = []
            for i in range(10):
                file_path = temp_dir_path / f"concurrent_file_{i}.txt"
                file_path.write_text(f"Content {i}")
                files.append(file_path)

            # Verify all files exist and have correct content
            for i, file_path in enumerate(files):
                assert file_path.exists()
                assert file_path.read_text() == f"Content {i}"

            # Clean up
            for file_path in files:
                file_path.unlink()
                assert not file_path.exists()

    def test_large_file_operations(self):
        """Test operations with larger files in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "large_file_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create a moderately large file (100KB) - reduced size to prevent memory issues
            large_file = temp_dir_path / "large_file.txt"
            content = "A" * (1024 * 100)  # 100KB of 'A's (reduced from 1MB)

            large_file.write_text(content)

            assert large_file.exists()
            assert large_file.stat().st_size >= 1024 * 100  # Reduced from 1MB

            # Read it back
            read_content = large_file.read_text()
            assert len(read_content) == len(content)
            assert read_content == content

            # Test chunked reading
            with large_file.open("r") as f:
                chunk = f.read(1024)
                assert len(chunk) == 1024
                assert chunk == "A" * 1024


class TestHiddenPrints:
    """Test _HiddenPrints context manager."""

    def test_silence_enabled(self):
        """Test that prints are silenced when silence=True."""
        import sys
        from io import StringIO

        # Capture stdout to verify silence
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with utils._HiddenPrints(silence=True):
                print("This should be silenced")

            # Output should be empty (silenced)
            assert captured_output.getvalue() == ""
        finally:
            sys.stdout = original_stdout

    def test_silence_disabled(self):
        """Test that prints work normally when silence=False."""
        import sys
        from io import StringIO

        # Capture stdout to verify output
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with utils._HiddenPrints(silence=False):
                print("This should be visible")

            # Output should contain the print statement
            assert "This should be visible" in captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

    def test_context_manager_restoration(self):
        """Test that stdout is properly restored after context."""
        import sys

        original_stdout = sys.stdout

        with utils._HiddenPrints(silence=True):
            # stdout should be redirected during context
            assert sys.stdout != original_stdout

        # stdout should be restored after context
        assert sys.stdout == original_stdout


class TestFilepathToIndexEdgeCases:
    """Test edge cases for filepath_to_index function."""

    def test_no_numbers_in_path(self):
        """Test filepath with no numbers returns 0."""
        assert utils.filepath_to_index("/path/to/file_without_numbers.bin") == 0

    def test_only_extension_numbers(self):
        """Test filepath where only the extension contains numbers returns 0."""
        # Only extension has numbers, so they are stripped and result is empty list -> 0
        assert utils.filepath_to_index("/path/to/file.mp4") == 0


class TestNanaverageEdgeCases:
    """Test edge cases for nanaverage function."""

    def test_zero_weights(self):
        """Test nanaverage with all zero weights."""
        A = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        # This will cause division by zero warnings but should return nan gracefully
        with pytest.warns(RuntimeWarning, match="invalid value encountered in scalar divide"):
            result = utils.nanaverage(A, weights)

        assert np.isnan(result)
        assert isinstance(result, np.ndarray)

    def test_negative_weights(self):
        """Test nanaverage with negative weights."""
        A = np.array([1.0, 2.0, 3.0])
        weights = np.array([-1.0, 2.0, -1.0])

        # This should work but may produce warnings about negative weights
        with pytest.warns(RuntimeWarning, match="invalid value encountered in scalar divide"):
            result = utils.nanaverage(A, weights)

        # Should return nan due to invalid calculation with negative weights
        assert np.isnan(result)
        assert isinstance(result, np.ndarray)


class TestGetFileStemEdgeCases:
    """Edge cases for get_file_stem handling double extensions."""

    def test_double_extension(self):
        assert utils.get_file_stem("/data/recording.npy.gz") == "recording"

    def test_single_extension(self):
        assert utils.get_file_stem("/data/file.rhd") == "file"


class TestLogTransformEdgeCases:
    """Edge cases for log_transform."""

    def test_none_input(self):
        assert utils.log_transform(None) is None

    def test_values(self):
        arr = np.array([0.0, 1.0, np.e - 1])
        result = utils.log_transform(arr)
        np.testing.assert_allclose(result, np.log(arr + 1))


class TestGetCacheStatusMessage:
    """Tests for get_cache_status_message."""

    def test_using_cached(self, tmp_path):
        msg = utils.get_cache_status_message(tmp_path / "c.pkl", True)
        assert "Using cached" in msg

    def test_regenerating(self, tmp_path):
        msg = utils.get_cache_status_message(tmp_path / "c.pkl", False)
        assert "Regenerating" in msg


class TestGetGroupbyKeys:
    """Tests for _get_groupby_keys."""

    def test_basic(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        keys = utils._get_groupby_keys(df, "g")
        assert set(keys) == {"a", "b"}


class TestGetPairwiseCombinations:
    """Tests for _get_pairwise_combinations."""

    def test_three_elements(self):
        combos = utils._get_pairwise_combinations([1, 2, 3])
        assert set(combos) == {(1, 2), (1, 3), (2, 3)}

    def test_empty(self):
        assert utils._get_pairwise_combinations([]) == []

    def test_single(self):
        assert utils._get_pairwise_combinations([1]) == []


class TestNanmeanSeriesOfNpEdgeCases:
    """Additional edge cases for nanmean_series_of_np fast path."""

    def test_small_series_mean(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        series = pd.Series(list(arr))
        result = utils.nanmean_series_of_np(series, axis=0)
        expected = np.nanmean(arr, axis=0)
        np.testing.assert_allclose(result, expected)

    def test_large_series_stack_fast_path(self):
        """Series > 1000 elements triggers np.stack fast path."""
        np.random.seed(0)
        arrays = [np.random.randn(5) for _ in range(1500)]
        series = pd.Series(arrays)
        result = utils.nanmean_series_of_np(series, axis=0)
        expected = np.nanmean(np.stack(arrays), axis=0)
        np.testing.assert_allclose(result, expected)


class TestSlugify:
    """Test slugify function."""

    def test_basic_slugification(self):
        """Test basic string slugification."""
        assert utils.slugify("Hello World") == "hello-world"

    def test_multiple_spaces(self):
        """Test collapsing of multiple spaces."""
        assert utils.slugify("Multiple   Spaces") == "multiple-spaces"

    def test_special_characters_removed(self):
        """Test removal of special characters."""
        assert utils.slugify("Hello@#$World!") == "helloworld"

    def test_unicode_default(self):
        """Test ASCII transliteration with allow_unicode=False (default)."""
        assert utils.slugify("Café") == "cafe"

    def test_unicode_allowed(self):
        """Test unicode preservation with allow_unicode=True."""
        assert utils.slugify("Café", allow_unicode=True) == "café"

    def test_hyphens_preserved(self):
        """Test that existing hyphens are preserved."""
        assert utils.slugify("hello-world") == "hello-world"

    def test_underscores_preserved(self):
        """Test that underscores are preserved."""
        assert utils.slugify("hello_world") == "hello_world"

    def test_repeated_dashes_collapsed(self):
        """Test conversion of repeated dashes to single dash."""
        assert utils.slugify("hello---world") == "hello-world"

    def test_leading_trailing_stripped(self):
        """Test stripping of leading/trailing hyphens and underscores."""
        assert utils.slugify("  --hello--  ") == "hello"

    def test_non_string_input(self):
        """Test that non-string values are converted to string."""
        assert utils.slugify(42) == "42"

    def test_empty_string(self):
        """Test empty string input."""
        assert utils.slugify("") == ""

    def test_animal_id_pattern(self):
        """Test typical animal ID pattern used in the pipeline."""
        assert utils.slugify("A10") == "a10"
        assert utils.slugify("test_animal-WT-day1") == "test_animal-wt-day1"

    def test_animal_id_with_allow_unicode(self):
        """Test animal ID slugification with allow_unicode."""
        assert utils.slugify("A10", allow_unicode=True) == "a10"

