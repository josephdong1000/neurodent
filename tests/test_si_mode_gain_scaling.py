"""
Test suite for SI mode gain scaling fix.

This test ensures that when using SpikeInterface mode (SI mode) for loading
recordings, the metadata is properly initialized with voltage units and
gain information.

Background:
- SpikeInterface's get_traces(return_scaled=True) returns data in microvolts
- The metadata must reflect this so downstream code knows the units
- Previously, V_units and mult_to_uV were not set, causing unit mismatches
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from neurodent import constants
from neurodent.core.core import LongRecordingOrganizer


class TestSIModeGainScaling:
    """Test that SI mode properly sets voltage units and gain metadata."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_si_mode_metadata_sets_v_units(self, temp_dir):
        """Test that SI mode sets V_units='µV' in metadata."""
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        # Mock SpikeInterface recording
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 4
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.array(['ch1', 'ch2', 'ch3', 'ch4'])
        mock_recording.get_duration.return_value = 3600.0

        mock_extract = Mock(return_value=mock_recording)

        # Call the SI mode conversion
        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="folder"
        )

        # Verify metadata has voltage units set
        assert organizer.meta is not None
        assert organizer.meta.V_units == 'µV'
        assert organizer.meta.mult_to_uV == 1.0

    def test_si_mode_metadata_sets_mult_to_uv(self, temp_dir):
        """Test that SI mode sets mult_to_uV=1.0 in metadata."""
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        # Mock SpikeInterface recording
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 2
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_recording.get_duration.return_value = 1800.0

        mock_extract = Mock(return_value=mock_recording)

        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="folder"
        )

        # Verify mult_to_uV is set to 1.0 (data already in µV from SI)
        assert organizer.meta.mult_to_uV == 1.0

    def test_si_mode_with_gain_property(self, temp_dir):
        """Test SI mode when recording has gain_to_uV property."""
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        # Mock SpikeInterface recording with gain property
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 4
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.array([0, 1, 2, 3])
        mock_recording.get_duration.return_value = 3600.0

        # Add gain_to_uV property
        mock_recording.get_property = Mock(return_value=np.array([0.195, 0.195, 0.195, 0.195]))

        mock_extract = Mock(return_value=mock_recording)

        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="folder"
        )

        # Metadata should still be set correctly
        assert organizer.meta.V_units == 'µV'
        assert organizer.meta.mult_to_uV == 1.0

    def test_si_mode_without_gain_property(self, temp_dir):
        """Test SI mode when recording doesn't have gain_to_uV property."""
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        # Mock SpikeInterface recording without gain property
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 4
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.array(['ch1', 'ch2', 'ch3', 'ch4'])
        mock_recording.get_duration.return_value = 3600.0

        # get_property raises AttributeError (no such property)
        mock_recording.get_property = Mock(side_effect=AttributeError("No such property"))

        mock_extract = Mock(return_value=mock_recording)

        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="folder"
        )

        # Should still work with fallback values
        assert organizer.meta.V_units == 'µV'
        assert organizer.meta.mult_to_uV == 1.0

    def test_si_mode_multiple_files_metadata_consistent(self, temp_dir):
        """Test that metadata is consistent when loading multiple files."""
        # Create test files
        file1 = temp_dir / "file1.rhd"
        file2 = temp_dir / "file2.rhd"
        file1.touch()
        file2.touch()

        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=[
                datetime(2023, 1, 1, 10, 0, 0),
                datetime(2023, 1, 1, 11, 0, 0)
            ],
            datetimes_are_start=True
        )

        # Mock recordings
        mock_rec1 = Mock()
        mock_rec1.get_duration.return_value = 3600.0

        mock_rec2 = Mock()
        mock_rec2.get_duration.return_value = 3600.0

        mock_extract = Mock(side_effect=[mock_rec1, mock_rec2])

        # Mock concatenated recording
        mock_concat = Mock()
        mock_concat.get_num_channels.return_value = 32
        mock_concat.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_concat.get_sampling_frequency.return_value = 1000.0
        mock_concat.get_channel_ids.return_value = np.arange(32)
        mock_concat.get_duration.return_value = 7200.0

        with patch('spikeinterface.core.concatenate_recordings', return_value=mock_concat):
            organizer.convert_file_with_si_to_recording(
                extract_func=mock_extract,
                input_type="files",
                file_pattern="*.rhd"
            )

        # Verify metadata is set correctly
        assert organizer.meta.V_units == 'µV'
        assert organizer.meta.mult_to_uV == 1.0

    def test_metadata_dict_serialization(self, temp_dir):
        """Test that metadata with V_units and mult_to_uV can be serialized."""
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 2
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_recording.get_duration.return_value = 1800.0

        mock_extract = Mock(return_value=mock_recording)

        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="folder"
        )

        # Test metadata can be converted to dict (for serialization)
        meta_dict = organizer.meta.to_dict()

        assert 'V_units' in meta_dict
        assert 'mult_to_uV' in meta_dict
        assert meta_dict['V_units'] == 'µV'
        assert meta_dict['mult_to_uV'] == 1.0

    def test_si_mode_units_match_get_traces_behavior(self, temp_dir):
        """
        Test that metadata units match SpikeInterface's get_traces(return_scaled=True) behavior.

        This is critical: get_traces(return_scaled=True) returns data in microvolts,
        so our metadata must reflect that the data is already in µV with mult_to_uV=1.0.
        """
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        # Create mock recording that simulates Intan data
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 4
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 20000.0
        mock_recording.get_channel_ids.return_value = np.arange(4)
        mock_recording.get_duration.return_value = 3600.0

        mock_extract = Mock(return_value=mock_recording)

        # Mock resampling since input rate != target rate
        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 4
        mock_resampled.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_resampled.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_resampled.get_channel_ids.return_value = np.arange(4)
        mock_resampled.get_duration.return_value = 3600.0

        with patch('spikeinterface.preprocessing.resample', return_value=mock_resampled):
            organizer.convert_file_with_si_to_recording(
                extract_func=mock_extract,
                input_type="folder"
            )

        # Critical assertions:
        # 1. Data from get_traces(return_scaled=True) is in µV
        # 2. Metadata reflects this with V_units='µV'
        # 3. mult_to_uV=1.0 means "already in microvolts, no conversion needed"
        assert organizer.meta.V_units == 'µV', \
            "V_units must be 'µV' to match get_traces(return_scaled=True) output"
        assert organizer.meta.mult_to_uV == 1.0, \
            "mult_to_uV must be 1.0 since data is already scaled to microvolts"


class TestTruncationConfig:
    """Test that truncation configuration change works correctly."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_truncate_false_loads_all_files(self, temp_dir):
        """Test that truncate=false allows all files to be loaded."""
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            truncate=False
        )

        # Create many files
        files = [f"file_{i:03d}.rhd" for i in range(50)]

        # Should return all files, not truncated
        result = organizer._truncate_file_list(files)
        assert len(result) == 50
        assert result == sorted(files)

    def test_truncate_int_limits_files(self, temp_dir):
        """Test that truncate=N limits to N files."""
        with pytest.warns(UserWarning, match="LongRecording will be truncated to the first 5 files"):
            organizer = LongRecordingOrganizer(
                temp_dir,
                mode=None,
                truncate=5
            )

        files = [f"file_{i:03d}.rhd" for i in range(50)]

        result = organizer._truncate_file_list(files)
        assert len(result) == 5
        assert result == sorted(files)[:5]

    def test_config_truncate_false_string(self, temp_dir):
        """Test that truncate=false (lowercase string from YAML) works."""
        # YAML parses 'false' as boolean False, so test with boolean
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            truncate=False
        )

        assert organizer.truncate is False
        assert organizer.n_truncate == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
