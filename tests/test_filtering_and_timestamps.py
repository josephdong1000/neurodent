"""
Tests for filtering utilities and timestamp fixes in SI/MNE modes.
"""

import os
import pytest
import warnings
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from pythoneeg import core


class TestTimestampFixes:
    """Test DEFAULT_DAY fixes in SI/MNE modes."""

    def test_si_mode_with_manual_timestamps_no_default_day(self):
        """Test SI mode doesn't use DEFAULT_DAY when manual timestamps provided."""
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("pythoneeg.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
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

                with patch("spikeinterface.core.concatenate_recordings", return_value=mock_rec):
                    recording = organizer.convert_rowbins_to_rec()
                    
                    # Should succeed without DEFAULT_DAY errors
                    assert recording is not None

    def test_mne_mode_with_manual_timestamps_no_default_day(self):
        """Test MNE mode doesn't use DEFAULT_DAY when manual timestamps provided."""
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("pythoneeg.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer("/fake/path", mode=None)
                organizer.manual_datetimes = datetime(2023, 1, 1, 10, 0, 0)

                # Mock extract_func and run MNE conversion
                mock_extract_func = Mock()
                mock_raw = MagicMock()
                mock_raw.info = {"sfreq": 1000, "ch_names": ["ch1", "ch2", "ch3", "ch4"]}
                mock_raw.n_times = 10000
                mock_raw.times = np.linspace(0, 10, 10000)
                mock_raw.get_data.return_value = np.random.randn(4, 10000)
                mock_extract_func.return_value = mock_raw

                mne_raw = organizer.convert_to_mne()
                
                # Should succeed without DEFAULT_DAY errors
                assert mne_raw is not None


class TestNJobsSimplification:
    """Test that n_jobs handling is now simplified."""

    def test_n_jobs_defaults_to_one(self):
        """Test that n_jobs defaults to 1 in core functionality."""
        # This is now handled directly in the resampling code without complex detection
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("pythoneeg.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer("/fake/path", mode=None)
                # Default n_jobs should be 1, not complex detection
                assert organizer.n_jobs == 1

    def test_n_jobs_user_specified_respected(self):
        """Test that user-specified n_jobs values are respected."""
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("pythoneeg.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer("/fake/path", mode=None, n_jobs=4)
                assert organizer.n_jobs == 4