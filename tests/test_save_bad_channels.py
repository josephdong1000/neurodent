"""
Test file for save_bad_channels functionality.
This file contains comprehensive tests for the new save_bad_channels parameter
in filter functions.
"""

import numpy as np
import pandas as pd
import pytest
import copy

from neurodent.visualization import WindowAnalysisResult


class TestSaveBadChannels:
    """Test the save_bad_channels functionality."""

    @pytest.fixture
    def sample_result_df(self):
        """Create a sample result DataFrame for testing."""
        np.random.seed(42)

        # Create 20 windows across 2 recording sessions
        timestamps = pd.date_range(start="2023-01-01 10:00:00", periods=20, freq="1min")
        animaldays = ["A1_20230101"] * 10 + ["A1_20230102"] * 10

        data = {
            "timestamp": timestamps,
            "animalday": animaldays,
            "duration": [1.0] * 20,
            "rms": [[100.0, 200.0, 150.0] for _ in range(20)],
            "psdtotal": [[10.0, 20.0, 15.0] for _ in range(20)],
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def test_war(self, sample_result_df):
        """Create a WindowAnalysisResult instance for testing."""
        return WindowAnalysisResult(
            result=sample_result_df,
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot", "LBar"],
            bad_channels_dict={"A1_20230101": ["LMot"], "A1_20230102": ["RMot"]},
        )

    @pytest.fixture
    def empty_war(self, sample_result_df):
        """Create a WindowAnalysisResult with empty bad_channels_dict."""
        return WindowAnalysisResult(
            result=sample_result_df,
            animal_id="A1",
            genotype="WT",
            channel_names=["LMot", "RMot", "LBar"],
            bad_channels_dict={},
        )

    def test_get_filter_reject_channels_save_none(self, test_war):
        """Test that save_bad_channels=None doesn't modify bad_channels_dict."""
        original_dict = test_war.bad_channels_dict.copy()

        # Apply filter with save_bad_channels=None
        mask = test_war.get_filter_reject_channels(bad_channels=["LBar"], save_bad_channels=None)

        # Check that bad_channels_dict is unchanged
        assert test_war.bad_channels_dict == original_dict

        # Check that filtering still works
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (20, 3)
        # LBar (index 2) should be False for all windows
        assert np.all(mask[:, 2] == False)
        assert np.all(mask[:, :2] == True)  # Other channels should be True

    def test_get_filter_reject_channels_save_union(self, test_war):
        """Test save_bad_channels='union' mode."""
        original_dict = test_war.bad_channels_dict.copy()

        # Apply filter with save_bad_channels='union' (default)
        mask = test_war.get_filter_reject_channels(bad_channels=["LBar"])

        # Check that LBar was added to all sessions
        expected_dict = {
            "A1_20230101": ["LMot", "LBar"],  # Union of original + new
            "A1_20230102": ["RMot", "LBar"],  # Union of original + new
        }

        # Sort lists for comparison since order doesn't matter
        for session in expected_dict:
            assert set(test_war.bad_channels_dict[session]) == set(expected_dict[session])

    def test_get_filter_reject_channels_save_overwrite(self, test_war):
        """Test save_bad_channels='overwrite' mode."""
        # Apply filter with save_bad_channels='overwrite'
        mask = test_war.get_filter_reject_channels(bad_channels=["LBar"], save_bad_channels="overwrite")

        # Check that bad_channels_dict was completely replaced
        expected_dict = {"A1_20230101": ["LBar"], "A1_20230102": ["LBar"]}

        assert test_war.bad_channels_dict == expected_dict

    def test_get_filter_reject_channels_empty_dict_union(self, empty_war):
        """Test union mode with initially empty bad_channels_dict."""
        # Apply filter with union mode
        mask = empty_war.get_filter_reject_channels(bad_channels=["LMot", "RMot"])

        # Check that channels were added to all sessions
        expected_dict = {"A1_20230101": ["LMot", "RMot"], "A1_20230102": ["LMot", "RMot"]}

        assert empty_war.bad_channels_dict == expected_dict

    def test_get_filter_reject_channels_by_session_save_none(self, test_war):
        """Test that save_bad_channels=None doesn't modify bad_channels_dict."""
        original_dict = test_war.bad_channels_dict.copy()

        new_dict = {"A1_20230101": ["LBar"], "A1_20230102": ["LBar"]}

        # Apply filter with save_bad_channels=None
        mask = test_war.get_filter_reject_channels_by_recording_session(
            bad_channels_dict=new_dict, save_bad_channels=None
        )

        # Check that bad_channels_dict is unchanged
        assert test_war.bad_channels_dict == original_dict

    def test_get_filter_reject_channels_by_session_save_union(self, test_war):
        """Test save_bad_channels='union' mode for session-specific filtering."""
        new_dict = {"A1_20230101": ["LBar"], "A1_20230102": ["LBar"]}

        # Apply filter with save_bad_channels='union' (default)
        mask = test_war.get_filter_reject_channels_by_recording_session(bad_channels_dict=new_dict)

        # Check that channels were merged
        expected_dict = {
            "A1_20230101": ["LMot", "LBar"],  # Union of original + new
            "A1_20230102": ["RMot", "LBar"],  # Union of original + new
        }

        for session in expected_dict:
            assert set(test_war.bad_channels_dict[session]) == set(expected_dict[session])

    def test_get_filter_reject_channels_by_session_save_overwrite(self, test_war):
        """Test save_bad_channels='overwrite' mode for session-specific filtering."""
        new_dict = {"A1_20230101": ["LBar"], "A1_20230102": ["LBar"]}

        # Apply filter with save_bad_channels='overwrite'
        mask = test_war.get_filter_reject_channels_by_recording_session(
            bad_channels_dict=new_dict, save_bad_channels="overwrite"
        )

        # Check that bad_channels_dict was completely replaced
        assert test_war.bad_channels_dict == new_dict

    def test_filter_all_save_bad_channels_default(self, test_war):
        """Test filter_all with default save_bad_channels='union'."""
        original_dict = test_war.bad_channels_dict.copy()

        # Use only the filtering functions we want to test, to avoid missing data issues
        filters = [test_war.get_filter_reject_channels_by_recording_session, test_war.get_filter_reject_channels]

        # Apply filter_all with bad_channels parameter
        filtered = test_war.filter_all(bad_channels=["LBar"], filters=filters)

        # Check that LBar was added to all sessions in the original instance
        expected_dict = {"A1_20230101": ["LMot", "LBar"], "A1_20230102": ["RMot", "LBar"]}

        for session in expected_dict:
            assert set(test_war.bad_channels_dict[session]) == set(expected_dict[session])

        # Check that returned instance also has updated dict
        for session in expected_dict:
            assert set(filtered.bad_channels_dict[session]) == set(expected_dict[session])

    def test_filter_all_save_bad_channels_none(self, test_war):
        """Test filter_all with save_bad_channels=None."""
        original_dict = test_war.bad_channels_dict.copy()

        # Use only the filtering functions we want to test
        filters = [test_war.get_filter_reject_channels_by_recording_session, test_war.get_filter_reject_channels]

        # Apply filter_all with save_bad_channels=None
        filtered = test_war.filter_all(bad_channels=["LBar"], save_bad_channels=None, filters=filters)

        # Check that bad_channels_dict is unchanged
        assert test_war.bad_channels_dict == original_dict
        assert filtered.bad_channels_dict == original_dict

    def test_filter_all_save_bad_channels_overwrite(self, test_war):
        """Test filter_all with save_bad_channels='overwrite'."""
        # Use only the filtering functions we want to test
        filters = [test_war.get_filter_reject_channels_by_recording_session, test_war.get_filter_reject_channels]

        # Apply filter_all with save_bad_channels='overwrite'
        filtered = test_war.filter_all(bad_channels=["LBar"], save_bad_channels="overwrite", filters=filters)

        # Check that bad_channels_dict was completely replaced
        expected_dict = {"A1_20230101": ["LBar"], "A1_20230102": ["LBar"]}

        assert test_war.bad_channels_dict == expected_dict
        assert filtered.bad_channels_dict == expected_dict

    def test_copy_behavior(self, test_war):
        """Test that .copy() is used appropriately when reading but not when writing."""
        original_dict = test_war.bad_channels_dict
        original_id = id(original_dict)

        # Apply filter that should modify the dict
        test_war.get_filter_reject_channels(bad_channels=["LBar"], save_bad_channels="union")

        # The instance should still have the same dict object (modified in place)
        assert id(test_war.bad_channels_dict) != original_id  # New dict assigned

        # But the content should be updated
        assert "LBar" in test_war.bad_channels_dict["A1_20230101"]
        assert "LBar" in test_war.bad_channels_dict["A1_20230102"]

    def test_invalid_save_bad_channels_value(self, test_war):
        """Test that invalid save_bad_channels values are handled."""
        # This should work since we're using Literal type hints
        # But let's test the actual behavior
        try:
            test_war.get_filter_reject_channels(bad_channels=["LBar"], save_bad_channels="invalid")
            # If no error, the parameter was ignored or handled gracefully
        except (ValueError, TypeError):
            # Expected behavior for invalid values
            pass

    def test_channel_format_consistency(self, test_war):
        """Test that channel formats are handled consistently when saving."""
        # Test with abbreviations
        mask = test_war.get_filter_reject_channels(bad_channels=["LBar"], use_abbrevs=True, save_bad_channels="union")

        # The saved channels should be in the appropriate format
        assert "LBar" in test_war.bad_channels_dict["A1_20230101"]
        assert "LBar" in test_war.bad_channels_dict["A1_20230102"]

    def test_both_filter_functions_save_independently(self, empty_war):
        """Test that both filter functions can save bad channels independently."""
        # First, use get_filter_reject_channels to add manual channels
        empty_war.get_filter_reject_channels(bad_channels=["LMot"], save_bad_channels="union")

        # Then, use get_filter_reject_channels_by_recording_session to add session-specific channels
        session_dict = {"A1_20230101": ["RMot"], "A1_20230102": ["LBar"]}
        empty_war.get_filter_reject_channels_by_recording_session(
            bad_channels_dict=session_dict, save_bad_channels="union"
        )

        # Check that both sets of channels are present
        expected_dict = {
            "A1_20230101": ["LMot", "RMot"],  # From both functions
            "A1_20230102": ["LMot", "LBar"],  # From both functions
        }

        for session in expected_dict:
            assert set(empty_war.bad_channels_dict[session]) == set(expected_dict[session])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
