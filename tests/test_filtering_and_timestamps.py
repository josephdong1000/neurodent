"""
Tests for filtering utilities and timestamp fixes in SI/MNE modes.
"""

import os
import pytest
import warnings
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from neurodent import core


class TestTimestampFixes:
    """Test DEFAULT_DAY fixes in SI/MNE modes."""

    def test_si_mode_with_manual_timestamps_no_default_day(self):
        """Test SI mode doesn't use DEFAULT_DAY when manual timestamps provided."""
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("neurodent.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer("/fake/path", mode=None)
                organizer.manual_datetimes = datetime(2023, 1, 1, 10, 0, 0)

                # Mock extract_func and run SI conversion
                mock_extract_func = Mock()
                mock_rec = Mock()
                mock_rec.get_num_channels.return_value = 4
                mock_rec.get_sampling_frequency.return_value = 1000
                mock_rec.get_channel_ids.return_value = np.array(["ch1", "ch2", "ch3", "ch4"])
                mock_rec.get_duration.return_value = 10.0
                mock_extract_func.return_value = mock_rec

                # Use the correct SI method, not rowbins conversion
                organizer.convert_file_with_si_to_recording(
                    extract_func=mock_extract_func, input_type="file", file_pattern="*.edf"
                )

                # Should succeed without DEFAULT_DAY errors
                assert organizer.LongRecording is not None

    def test_mne_mode_with_manual_timestamps_no_default_day(self):
        """Test MNE mode doesn't use DEFAULT_DAY when manual timestamps provided."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("neurodent.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer(tmpdir_path, mode=None)
                organizer.manual_datetimes = datetime(2023, 1, 1, 10, 0, 0)

                # Mock the Path.glob method to return fake files
                with patch.object(Path, "glob", return_value=[Path(tmpdir) / "file.edf"]):
                    # Mock extract_func and run MNE conversion
                    mock_extract_func = Mock()
                    mock_raw = MagicMock()
                    mock_raw.info = {"sfreq": 1000, "ch_names": ["ch1", "ch2", "ch3", "ch4"], "nchan": 4}
                    mock_raw.preload = True  # Mock that data is preloaded
                    mock_raw.resample.return_value = mock_raw  # Mock resample returns same object
                    mock_extract_func.return_value = mock_raw

                    # Mock the MNE export and SpikeInterface read functions
                    with (
                        patch("neurodent.core.core.mne.export.export_raw") as mock_export,
                        patch("neurodent.core.core.se.read_edf") as mock_read_edf,
                    ):
                        # Mock the SpikeInterface recording
                        mock_recording = Mock()
                        mock_recording.get_num_channels.return_value = 4
                        mock_recording.get_sampling_frequency.return_value = 1000
                        mock_recording.get_duration.return_value = 10.0
                        mock_recording.get_channel_ids.return_value = np.array(["ch1", "ch2", "ch3", "ch4"])
                        mock_read_edf.return_value = mock_recording

                        # Use the correct MNE method
                        organizer.convert_file_with_mne_to_recording(
                            extract_func=mock_extract_func, input_type="file", file_pattern="*.edf"
                        )

                    # Should succeed without DEFAULT_DAY errors
                    assert organizer.LongRecording is not None


class TestNJobsSimplification:
    """Test that n_jobs handling is now simplified."""

    def test_n_jobs_defaults_to_one(self):
        """Test that n_jobs defaults to 1 in core functionality."""
        # This is now handled directly in the resampling code without complex detection
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("neurodent.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer("/fake/path", mode=None)
                # Default n_jobs should be 1, not complex detection
                assert organizer.n_jobs == 1

    def test_n_jobs_user_specified_respected(self):
        """Test that user-specified n_jobs values are respected."""
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("neurodent.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer("/fake/path", mode=None, n_jobs=4)
                assert organizer.n_jobs == 4
