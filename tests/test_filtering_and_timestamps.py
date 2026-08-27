"""
Tests for filtering utilities and timestamp fixes in SI/MNE modes.
"""

import inspect
import os
import pytest
import warnings
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from neurodent.loading import long_recording_organizer as core
from neurodent import constants


class TestManualTimestamps:
    """Test manual timestamp handling in SI/MNE modes."""

    def test_si_mode_with_manual_timestamps(self):
        """SI mode uses manual timestamps when provided."""
        with patch("glob.glob", return_value=["/fake/file.edf"]):
            with patch("neurodent.loading.long_recording_organizer.LongRecordingOrganizer._validate_timestamps_for_mode"):
                organizer = core.LongRecordingOrganizer("/fake/path", mode=None)
                organizer.manual_datetimes = datetime(2023, 1, 1, 10, 0, 0)

                # Mock extract_func and run SI conversion
                mock_extract_func = Mock()
                mock_rec = Mock()
                mock_rec.get_num_channels.return_value = 4
                mock_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
                mock_rec.get_sampling_frequency.return_value = 1000
                mock_rec.get_channel_ids.return_value = np.array(["ch1", "ch2", "ch3", "ch4"])
                mock_rec.get_duration.return_value = 10.0
                mock_extract_func.return_value = mock_rec

                organizer.convert_file_with_si_to_recording(
                    extract_func=mock_extract_func, input_type="file", file_pattern="*.edf"
                )

                assert organizer.LongRecording is not None

    def test_mne_mode_with_manual_timestamps(self):
        """MNE mode uses manual timestamps when provided."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("neurodent.loading.long_recording_organizer.LongRecordingOrganizer._validate_timestamps_for_mode"):
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
                        # Intermediate is written atomically (temp + rename), so the
                        # mocked exporter must create the file at the path it is given.
                        patch(
                            "neurodent.loading.lro_loading.mne.export.export_raw",
                            side_effect=lambda path, *a, **k: open(path, "wb").close(),
                        ) as mock_export,
                        patch("neurodent.loading.lro_loading.se.read_edf") as mock_read_edf,
                    ):
                        # Mock the SpikeInterface recording
                        mock_recording = Mock()
                        mock_recording.get_num_channels.return_value = 4
                        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
                        mock_recording.get_sampling_frequency.return_value = 1000
                        mock_recording.get_duration.return_value = 10.0
                        mock_recording.get_channel_ids.return_value = np.array(["ch1", "ch2", "ch3", "ch4"])
                        mock_read_edf.return_value = mock_recording

                        # Use the correct MNE method
                        organizer.convert_file_with_mne_to_recording(
                            extract_func=mock_extract_func, input_type="file", file_pattern="*.edf"
                        )

                    assert organizer.LongRecording is not None


class TestRemovedNJobsKwarg:
    """``n_jobs`` was removed from the constructor; only ``save_recording`` still takes one."""

    def test_constructor_has_no_n_jobs_parameter(self):
        """The parameter is gone from the signature, so the docs page cannot advertise it."""
        params = inspect.signature(core.LongRecordingOrganizer.__init__).parameters
        assert "n_jobs" not in params

    def test_save_recording_still_takes_n_jobs(self):
        """SpikeInterface's chunked write is the one place n_jobs has an effect."""
        params = inspect.signature(core.LongRecordingOrganizer.save_recording).parameters
        assert "n_jobs" in params
