"""
Unit tests for neurodent.loading.long_recording_organizer module.
"""

import gzip
import gc
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import seuclidean

try:
    import spikeinterface.core as si
except Exception:
    si = None

from neurodent.loading.long_recording_organizer import (
    RecordingMetadata,
    LongRecordingOrganizer,
)
from neurodent import constants
from neurodent.core.utils import resolve_channels


def _export_creates_file(path, *args, **kwargs):
    """Side effect for a mocked ``mne.export.export_raw``.

    The intermediate file is written via ``atomic_output_path`` (temp sibling +
    rename), so a mocked exporter must create the file at the path it is given
    for the atomic rename to succeed.
    """
    Path(path).write_bytes(b"")


class TestLongRecordingOrganizer:
    """Test LongRecordingOrganizer class functionality."""

    def test_init_with_mode_none(self, temp_dir):
        """Test initialization with mode=None doesn't load data."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        assert organizer.meta is None
        assert organizer.channel_names is None
        assert organizer.LongRecording is None
        assert organizer.truncate is False
        assert organizer.n_truncate == 0

    def test_init_with_truncate_bool(self, temp_dir):
        """Test initialization with truncate=True."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore truncation warning
            organizer = LongRecordingOrganizer(temp_dir, mode=None, truncate=True)

        assert organizer.truncate is True
        assert organizer.n_truncate == 10  # Default truncate value

    def test_init_with_truncate_int(self, temp_dir):
        """Test initialization with truncate as integer."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            organizer = LongRecordingOrganizer(temp_dir, mode=None, truncate=5)

        assert organizer.truncate is True
        assert organizer.n_truncate == 5

    def test_time_conversion_methods(self, temp_dir):
        """Test __time_to_idx and __idx_to_time methods."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock LongRecording
        mock_recording = Mock()
        mock_recording.time_to_sample_index.return_value = 1000
        mock_recording.sample_index_to_time.return_value = 1.0
        organizer.LongRecording = mock_recording

        # Test time to index conversion
        idx = organizer._LongRecordingOrganizer__time_to_idx(1.0)
        assert idx == 1000
        mock_recording.time_to_sample_index.assert_called_once_with(1.0)

        # Test index to time conversion
        time_s = organizer._LongRecordingOrganizer__idx_to_time(1000)
        assert time_s == 1.0
        mock_recording.sample_index_to_time.assert_called_once_with(1000)

    def test_get_num_fragments(self, temp_dir):
        """Test get_num_fragments calculation."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock LongRecording
        mock_recording = Mock()
        mock_recording.time_to_sample_index.return_value = (
            1000  # 1 second = 1000 samples
        )
        mock_recording.get_num_frames.return_value = 5500  # 5.5 seconds total
        organizer.LongRecording = mock_recording

        # Should return ceil(5500 / 1000) = 6 fragments
        num_fragments = organizer.get_num_fragments(fragment_len_s=1.0)
        assert num_fragments == 6

    def test_fragidx_to_startendind(self, temp_dir):
        """Test __fragidx_to_startendind method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock LongRecording
        mock_recording = Mock()
        mock_recording.time_to_sample_index.return_value = (
            1000  # 1 second = 1000 samples
        )
        mock_recording.get_num_frames.return_value = 3500  # Total frames
        organizer.LongRecording = mock_recording

        # Test fragment 0
        start, end = organizer._LongRecordingOrganizer__fragidx_to_startendind(1.0, 0)
        assert start == 0
        assert end == 1000

        # Test fragment 1
        start, end = organizer._LongRecordingOrganizer__fragidx_to_startendind(1.0, 1)
        assert start == 1000
        assert end == 2000

        # Test last fragment (should be capped at total frames)
        start, end = organizer._LongRecordingOrganizer__fragidx_to_startendind(1.0, 3)
        assert start == 3000
        assert end == 3500  # Capped at total frames

    def test_get_fragment(self, temp_dir):
        """Test get_fragment method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock LongRecording
        mock_recording = Mock()
        mock_recording.time_to_sample_index.return_value = 1000
        mock_recording.get_num_frames.return_value = 5000
        mock_fragment = Mock()
        mock_recording.frame_slice.return_value = mock_fragment
        organizer.LongRecording = mock_recording

        fragment = organizer.get_fragment(fragment_len_s=1.0, fragment_idx=2)

        # Should call frame_slice with correct indices
        mock_recording.frame_slice.assert_called_once_with(2000, 3000)
        assert fragment == mock_fragment

    def test_get_dur_fragment(self, temp_dir):
        """Test get_dur_fragment method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock LongRecording
        mock_recording = Mock()
        mock_recording.time_to_sample_index.return_value = 1000
        mock_recording.get_num_frames.return_value = 5000
        mock_recording.sample_index_to_time.side_effect = lambda x: x / 1000.0
        organizer.LongRecording = mock_recording

        duration = organizer.get_dur_fragment(fragment_len_s=1.0, fragment_idx=1)

        # Fragment 1: indices 1000-2000, times 1.0-2.0, duration = 1.0
        assert duration == 1.0

    def test_cleanup_rec(self, temp_dir):
        """Test cleanup_rec method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Create temporary files
        temp_file1 = temp_dir / "temp1.bin"
        temp_file2 = temp_dir / "temp2.bin"
        temp_file1.touch()
        temp_file2.touch()

        organizer.LongRecording = Mock()
        organizer.temppaths = [temp_file1, temp_file2]

        # Test cleanup
        organizer.cleanup_rec()

        # Files should be deleted
        assert not temp_file1.exists()
        assert not temp_file2.exists()

    def test_cleanup_rec_no_recording(self, temp_dir):
        """Test cleanup_rec when LongRecording doesn't exist."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        organizer.temppaths = []

        # Should not raise exception - the method handles AttributeError internally
        # It uses logging.warning, not warnings.warn
        organizer.cleanup_rec()

    def test_detect_and_load_data_invalid_mode(self, temp_dir):
        """Test detect_and_load_data with invalid mode raises ValueError."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        with pytest.raises(ValueError, match="Invalid mode: invalid"):
            organizer.detect_and_load_data(mode="invalid")

    def test_get_datetime_fragment(self, temp_dir):
        """Test get_datetime_fragment method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock the required attributes
        end_time1 = datetime(2023, 1, 1, 12, 0, 0)
        end_time2 = datetime(2023, 1, 1, 13, 0, 0)
        organizer.file_end_datetimes = [end_time1, end_time2]
        organizer.file_durations = [3600.0, 3600.0]  # 1 hour each

        # Test getting datetime for fragment 0 with different fragment lengths
        expected = datetime(2023, 1, 1, 11, 0, 0)  # 1 hour before end_time1

        test_fragment_lengths = np.arange(1, 3600)
        for fragment_len_s in test_fragment_lengths:
            fragment_datetime = organizer.get_datetime_fragment(
                fragment_len_s=fragment_len_s, fragment_idx=0
            )
            assert fragment_datetime == expected, (
                f"Failed for fragment_len_s={fragment_len_s}"
            )

    def test_convert_to_mne(self, temp_dir):
        """Test convert_to_mne method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock LongRecording with test data
        n_channels = 3
        n_samples = 1000
        test_data = np.random.randn(n_samples, n_channels).astype(np.float32)

        mock_recording = Mock()
        mock_recording.get_traces.return_value = test_data
        mock_recording.get_sampling_frequency.return_value = 1000.0
        organizer.LongRecording = mock_recording
        organizer.channel_names = ["ch1", "ch2", "ch3"]

        with (
            patch("mne.create_info") as mock_create_info,
            patch("mne.io.RawArray") as mock_raw_array,
        ):
            mock_info = Mock()
            mock_create_info.return_value = mock_info
            mock_raw = Mock()
            mock_raw_array.return_value = mock_raw

            result = organizer.convert_to_mne()

            # Verify create_info was called correctly
            mock_create_info.assert_called_once_with(
                ch_names=["ch1", "ch2", "ch3"], sfreq=1000.0, ch_types="eeg"
            )

            # Verify RawArray was called with transposed data
            mock_raw_array.assert_called_once()
            call_args = mock_raw_array.call_args
            assert call_args[1]["info"] == mock_info
            # Data should be transposed from (n_samples, n_channels) to (n_channels, n_samples)
            passed_data = call_args[1]["data"]
            assert passed_data.shape == (n_channels, n_samples)

            assert result == mock_raw

    def test_compute_bad_channels(self, temp_dir):
        """Test compute_bad_channels method with chunked distance computation."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock LongRecording with chunked-read API
        n_channels = 4
        n_samples = 1000
        fs = 500.0

        # Create test data where channel 3 is an outlier
        normal_data = np.random.randn(n_samples, 3) * 10  # channels 0,1,2
        outlier_data = (
            np.random.randn(n_samples, 1) * 100
        )  # channel 3 (much larger amplitude)
        test_data = np.hstack([normal_data, outlier_data])

        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = n_channels
        mock_recording.get_total_samples.return_value = n_samples
        mock_recording.get_sampling_frequency.return_value = fs
        mock_recording.get_traces.side_effect = (
            lambda start_frame=0, end_frame=None, return_scaled=True: test_data[
                start_frame : (end_frame if end_frame is not None else n_samples)
            ]
        )
        mock_recording.__str__ = Mock(return_value="MockRecording")
        organizer.LongRecording = mock_recording
        organizer.channel_names = ["ch1", "ch2", "ch3", "ch4"]

        with (
            patch("neurodent.loading.lro_quality.Natural_Neighbor") as mock_nn_class,
            patch("neurodent.loading.lro_quality.LocalOutlierFactor") as mock_lof_class,
        ):
            # Mock Natural_Neighbor
            mock_nn = Mock()
            mock_nn.algorithm.return_value = 3
            mock_nn_class.return_value = mock_nn

            # Mock LocalOutlierFactor
            mock_lof = Mock()
            mock_lof.negative_outlier_factor_ = np.array(
                [-1.0, -1.0, -1.0, -2.0]
            )  # ch4 is outlier
            mock_lof_class.return_value = mock_lof

            # Test with default threshold
            organizer.compute_bad_channels(lof_threshold=1.5)

            # Verify Natural_Neighbor was given a distance matrix
            mock_nn.read_distance_matrix.assert_called_once()
            dist_mat = mock_nn.read_distance_matrix.call_args[0][0]
            assert dist_mat.shape == (n_channels, n_channels)
            mock_nn.algorithm.assert_called_once()

            # Verify LocalOutlierFactor was configured correctly
            mock_lof_class.assert_called_once_with(n_neighbors=3, metric="precomputed")
            mock_lof.fit.assert_called_once()

            # Channel 4 should be identified as bad (score 2.0 > threshold 1.5)
            assert organizer.bad_channel_names == ["ch4"]

    def test_compute_bad_channels_chunked_distance(self, temp_dir):
        """Test that chunked distance computation produces correct distances."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        n_channels = 3
        n_samples = 500
        fs = 100.0  # small fs so chunk_size > n_samples → single chunk

        np.random.seed(42)
        test_data = np.random.randn(n_samples, n_channels)

        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = n_channels
        mock_recording.get_total_samples.return_value = n_samples
        mock_recording.get_sampling_frequency.return_value = fs
        mock_recording.get_traces.side_effect = (
            lambda start_frame=0, end_frame=None, return_scaled=True: test_data[
                start_frame : (end_frame if end_frame is not None else n_samples)
            ]
        )
        mock_recording.__str__ = Mock(return_value="MockRecording")
        organizer.LongRecording = mock_recording
        organizer.channel_names = [f"ch{i}" for i in range(n_channels)]

        # Compute expected distance matrix from full data
        from scipy.spatial.distance import pdist, squareform
        expected_dists = squareform(pdist(test_data.T, metric="euclidean"))

        with (
            patch("neurodent.loading.lro_quality.Natural_Neighbor") as mock_nn_class,
            patch("neurodent.loading.lro_quality.LocalOutlierFactor") as mock_lof_class,
        ):
            mock_nn = Mock()
            mock_nn.algorithm.return_value = 2
            mock_nn_class.return_value = mock_nn

            mock_lof = Mock()
            mock_lof.negative_outlier_factor_ = np.array([-1.0] * n_channels)
            mock_lof_class.return_value = mock_lof

            organizer.compute_bad_channels(lof_threshold=2.0)

            # Verify the distance matrix passed to Natural_Neighbor matches expected
            dist_mat = mock_nn.read_distance_matrix.call_args[0][0]
            np.testing.assert_allclose(dist_mat, expected_dists, atol=1e-5)

    def test_extract_channel_names_prefers_channel_name_property(self):
        """Test _extract_channel_names uses channel_name property when available."""
        mock_recording = Mock()
        mock_recording.get_property_keys.return_value = ["channel_name", "gain_to_uV"]
        mock_recording.get_property.return_value = np.array(["C-009", "C-010", "C-012"])

        names = LongRecordingOrganizer._extract_channel_names(mock_recording)
        assert names == ["C-009", "C-010", "C-012"]

    def test_extract_channel_names_falls_back_to_channel_ids(self):
        """Test _extract_channel_names falls back to get_channel_ids when no channel_name property."""
        mock_recording = Mock()
        mock_recording.get_property_keys.return_value = ["gain_to_uV"]
        mock_recording.get_channel_ids.return_value = np.array(["ch1", "ch2"])

        names = LongRecordingOrganizer._extract_channel_names(mock_recording)
        assert names == ["ch1", "ch2"]

    def test_extract_channel_names_handles_mock_without_properties(self):
        """Test _extract_channel_names handles recordings without get_property_keys."""
        mock_recording = Mock()
        # Mock's get_property_keys returns a Mock (not iterable)
        mock_recording.get_channel_ids.return_value = np.array(["a", "b"])

        names = LongRecordingOrganizer._extract_channel_names(mock_recording)
        assert names == ["a", "b"]

    def test_extract_channel_names_with_integer_ids(self):
        """Test _extract_channel_names converts integer IDs to strings."""
        mock_recording = Mock()
        mock_recording.get_property_keys.return_value = []
        mock_recording.get_channel_ids.return_value = np.array([0, 1, 2])

        names = LongRecordingOrganizer._extract_channel_names(mock_recording)
        assert names == ["0", "1", "2"]

    def test_convert_file_with_si_to_recording_folder_mode(self, temp_dir):
        """Test convert_file_with_si_to_recording with folder input."""
        from datetime import datetime

        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
        )

        # Mock extract function and recording
        mock_extract = Mock()
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 4
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.array(
            ["ch1", "ch2", "ch3", "ch4"]
        )
        mock_recording.get_duration.return_value = 3600.0
        mock_extract.return_value = mock_recording

        organizer.item = str(temp_dir)
        organizer.convert_file_with_si_to_recording(extract_func=mock_extract)

        # Verify extract function was called with folder
        mock_extract.assert_called_once_with(str(temp_dir))
        assert organizer.LongRecording == mock_recording
        assert organizer.meta.n_channels == 4
        assert organizer.meta.f_s == 1000.0

    @patch("spikeinterface.preprocessing.resample")
    def test_convert_file_with_si_to_recording_file_mode(self, mock_resample, temp_dir):
        """Test convert_file_with_si_to_recording with single file input."""
        # Create test file
        test_file = temp_dir / "test.edf"
        test_file.touch()

        from datetime import datetime

        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
        )

        mock_extract = Mock()
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 2
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 500.0
        mock_recording.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_recording.get_duration.return_value = 1800.0
        mock_extract.return_value = mock_recording

        # Mock resampling since we're using mock recording
        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 2
        mock_resampled.get_sampling_frequency.return_value = (
            constants.GLOBAL_SAMPLING_RATE
        )
        mock_resampled.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_resampled.get_duration.return_value = 1800.0
        mock_resample.return_value = mock_resampled

        organizer.item = str(test_file)
        organizer.convert_file_with_si_to_recording(extract_func=mock_extract)

        # Should call extract with the found file
        mock_extract.assert_called_once_with(str(test_file))
        # Should have resampled the recording since 500.0 != 1000.0
        mock_resample.assert_called_once()
        assert organizer.LongRecording == mock_resampled

    def test_convert_file_with_si_to_recording_files_mode(self, temp_dir):
        """Test convert_file_with_si_to_recording with multiple files."""
        # Create test files
        file1 = temp_dir / "file1.edf"
        file2 = temp_dir / "file2.edf"
        file1.touch()
        file2.touch()

        from datetime import datetime

        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=[
                datetime(2023, 1, 1, 10, 0, 0),
                datetime(2023, 1, 1, 11, 0, 0),
            ],
            datetimes_are_start=True,
        )

        # Mock extract function to return different recordings
        mock_extract = Mock()
        mock_rec1 = Mock()
        mock_rec2 = Mock()
        mock_rec1.get_duration.return_value = 3600.0
        mock_rec2.get_duration.return_value = 1800.0

        def extract_side_effect(arg, **kwargs):
            if isinstance(arg, list):
                raise ValueError("cannot handle list")
            return mock_rec1 if "file1" in str(arg) else mock_rec2

        mock_extract.side_effect = extract_side_effect

        # Mock concatenate_recordings
        mock_concat_rec = Mock()
        mock_concat_rec.get_num_channels.return_value = 2
        mock_concat_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_concat_rec.get_sampling_frequency.return_value = 1000.0
        mock_concat_rec.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_concat_rec.get_duration.return_value = 5400.0

        with (
            patch(
                "spikeinterface.core.concatenate_recordings",
                return_value=mock_concat_rec,
            ),
            patch.object(organizer, "_apply_resampling", return_value=mock_concat_rec),
        ):
            organizer.item = [str(file1), str(file2)]
            organizer.convert_file_with_si_to_recording(extract_func=mock_extract)

        # Should call extract twice (once per file in serial mode)
        assert mock_extract.call_count == 2
        assert organizer.LongRecording == mock_concat_rec

    def test_convert_file_with_mne_to_recording_edf_intermediate(self, temp_dir):
        """Test convert_file_with_mne_to_recording with EDF intermediate."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        from datetime import datetime

        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
        )

        # Mock MNE raw object
        mock_raw = Mock()
        mock_info = Mock()
        mock_info.sfreq = 2000.0
        mock_info.nchan = 2
        mock_info.ch_names = ["ch1", "ch2"]  # Add channel names
        mock_info.chs = []  # Add empty chs list for extract_mne_unit_info
        mock_info.__getitem__ = Mock(side_effect=lambda key: getattr(mock_info, key))
        mock_info.__contains__ = Mock(side_effect=lambda key: hasattr(mock_info, key))
        mock_raw.info = mock_info
        mock_raw.resample.return_value = mock_raw
        mock_raw.get_data.return_value = np.random.randn(2, 3600000)

        mock_extract = Mock(return_value=mock_raw)

        # Mock SpikeInterface recording - should have original sampling rate from MNE raw
        mock_si_rec = Mock()
        mock_si_rec.get_num_channels.return_value = 2
        mock_si_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_si_rec.get_sampling_frequency.return_value = (
            2000.0  # Original MNE sampling rate
        )
        mock_si_rec.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_si_rec.get_duration.return_value = 3600.0

        # Mock resampled recording
        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 2
        mock_resampled.get_sampling_frequency.return_value = (
            constants.GLOBAL_SAMPLING_RATE
        )
        mock_resampled.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_resampled.get_duration.return_value = 3600.0

        with (
            patch("mne.export.export_raw", side_effect=_export_creates_file) as mock_export,
            patch("spikeinterface.extractors.read_edf", return_value=mock_si_rec),
            patch(
                "spikeinterface.preprocessing.resample", return_value=mock_resampled
            ) as mock_resample,
        ):
            organizer.item = str(test_file)
            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract, intermediate="edf"
            )

        # Verify raw was NOT resampled (new architecture moves resampling after intermediate file creation)
        mock_raw.resample.assert_not_called()
        # Verify SpikeInterface resampling WAS called since 2000.0 != 1000.0
        mock_resample.assert_called_once()
        mock_export.assert_called_once()
        assert organizer.LongRecording == mock_resampled

    def test_convert_file_with_mne_to_recording_bin_intermediate(self, temp_dir):
        """Test convert_file_with_mne_to_recording with binary intermediate."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        from datetime import datetime

        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
        )

        # Mock MNE raw object with test data
        n_channels = 3
        n_samples = 1000
        test_data = np.random.randn(n_channels, n_samples).astype(np.float32)

        mock_raw = Mock()
        mock_info = Mock()
        mock_info.sfreq = 1000.0
        mock_info.nchan = n_channels
        mock_info.ch_names = ["ch1", "ch2", "ch3"]  # Add channel names
        mock_info.chs = []  # Add empty chs list for extract_mne_unit_info
        mock_info.__getitem__ = Mock(side_effect=lambda key: getattr(mock_info, key))
        mock_info.__contains__ = Mock(side_effect=lambda key: hasattr(mock_info, key))
        mock_raw.info = mock_info
        mock_raw.resample.return_value = mock_raw
        mock_raw.get_data.return_value = test_data

        mock_extract = Mock(return_value=mock_raw)

        # Mock SpikeInterface recording
        mock_si_rec = Mock()
        mock_si_rec.get_num_channels.return_value = n_channels
        mock_si_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_si_rec.get_sampling_frequency.return_value = 1000.0
        mock_si_rec.get_channel_ids.return_value = np.array(["ch1", "ch2", "ch3"])
        mock_si_rec.get_duration.return_value = (
            1.0  # 1000 samples at 1000 Hz = 1 second
        )

        with patch("spikeinterface.extractors.read_binary", return_value=mock_si_rec):
            organizer.item = str(test_file)
            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract, intermediate="bin", intermediate_dir=temp_dir
            )

        # Verify binary file was created and read
        bin_file = temp_dir / "test_mne-to-rec.bin"
        assert bin_file.exists()

        # Verify data was written correctly (transposed from MNE format)
        written_data = np.fromfile(bin_file, dtype=np.float32).reshape(
            n_samples, n_channels
        )
        expected_data = (
            test_data.T
        )  # MNE data is (n_channels, n_samples), we expect (n_samples, n_channels)
        np.testing.assert_array_almost_equal(written_data, expected_data)

        assert organizer.LongRecording == mock_si_rec

    @patch("spikeinterface.preprocessing.resample")
    def test_apply_resampling_different_sampling_rate(self, mock_resample, temp_dir):
        """Test _apply_resampling method with different sampling rate."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock metadata
        organizer.meta = Mock()
        organizer.meta.update_sampling_rate = Mock()

        # Mock input recording
        mock_recording = Mock()
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 2000.0

        # Mock resampled recording
        mock_resampled = Mock()
        mock_resample.return_value = mock_resampled

        result = organizer._apply_resampling(mock_recording)

        # Verify resample was called with correct parameters
        mock_resample.assert_called_once_with(
            recording=mock_recording, resample_rate=constants.GLOBAL_SAMPLING_RATE
        )

        # Verify metadata was updated
        organizer.meta.update_sampling_rate.assert_called_once_with(
            constants.GLOBAL_SAMPLING_RATE
        )

        assert result == mock_resampled

    def test_apply_resampling_same_sampling_rate(self, temp_dir):
        """Test _apply_resampling method when no resampling needed."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock input recording at target rate
        mock_recording = Mock()
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = (
            constants.GLOBAL_SAMPLING_RATE
        )

        result = organizer._apply_resampling(mock_recording)

        # Should return original recording without modification
        assert result == mock_recording

    def test_apply_resampling_no_metadata(self, temp_dir):
        """Test _apply_resampling method when no metadata available."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        organizer.meta = None

        # Mock input recording
        mock_recording = Mock()
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 2000.0

        # Mock resampled recording
        mock_resampled = Mock()
        with patch(
            "spikeinterface.preprocessing.resample", return_value=mock_resampled
        ) as mock_resample:
            result = organizer._apply_resampling(mock_recording)

            # Verify resample was still called
            mock_resample.assert_called_once()
            assert result == mock_resampled

    def test_apply_resampling_missing_spikeinterface(self, temp_dir):
        """Test _apply_resampling method when SpikeInterface preprocessing not available."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        mock_recording = Mock()
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 2000.0

        # Mock missing preprocessing module
        with patch("neurodent.loading.lro_loading.spre", None):
            with pytest.raises(
                ImportError, match="SpikeInterface preprocessing is required"
            ):
                organizer._apply_resampling(mock_recording)

    def test_unified_resampling_metadata_consistency(self, temp_dir):
        """Test that metadata is consistently updated across resampling scenarios."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Create mock metadata with specific sampling rate
        organizer.meta = Mock()
        organizer.meta.update_sampling_rate = Mock()
        original_rate = 2000.0

        # Test that metadata update is called when resampling occurs
        mock_recording = Mock()
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = original_rate

        with patch("spikeinterface.preprocessing.resample") as mock_resample:
            mock_resampled = Mock()
            mock_resample.return_value = mock_resampled

            result = organizer._apply_resampling(mock_recording)

            # Verify metadata was updated to target rate
            organizer.meta.update_sampling_rate.assert_called_once_with(
                constants.GLOBAL_SAMPLING_RATE
            )

    def test_unified_resampling_cross_pipeline_consistency(self, temp_dir):
        """Test that all pipelines use the same resampling parameters."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Test recording with non-standard sampling rate
        test_rates = [500.0, 2000.0, 4000.0]

        for test_rate in test_rates:
            mock_recording = Mock()
            mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
            mock_recording.get_sampling_frequency.return_value = test_rate

            with patch("spikeinterface.preprocessing.resample") as mock_resample:
                mock_resampled = Mock()
                mock_resample.return_value = mock_resampled

                organizer._apply_resampling(mock_recording)

                # Verify consistent parameters across all calls
                if test_rate != constants.GLOBAL_SAMPLING_RATE:
                    mock_resample.assert_called_once_with(
                        recording=mock_recording,
                        resample_rate=constants.GLOBAL_SAMPLING_RATE,
                    )
                else:
                    mock_resample.assert_not_called()

    def test_unified_resampling_performance_parameters(self, temp_dir):
        """Test that resampling uses appropriate performance parameters."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        mock_recording = Mock()
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 2000.0

        with patch("spikeinterface.preprocessing.resample") as mock_resample:
            mock_resampled = Mock()
            mock_resample.return_value = mock_resampled

            organizer._apply_resampling(mock_recording)

            # Verify performance-oriented parameters
            mock_resample.assert_called_once_with(
                recording=mock_recording, resample_rate=constants.GLOBAL_SAMPLING_RATE
            )

    def test_unified_resampling_logging_behavior(self, temp_dir):
        """Test that resampling provides appropriate logging."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Test logging when resampling is needed
        mock_recording = Mock()
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 2000.0

        with (
            patch("spikeinterface.preprocessing.resample") as mock_resample,
            patch("neurodent.loading.lro_loading.logging") as mock_logging,
        ):
            mock_resampled = Mock()
            mock_resample.return_value = mock_resampled

            organizer._apply_resampling(mock_recording)

            # Should log the resampling operation
            mock_logging.info.assert_any_call(
                f"Resampling recording from 2000.0 Hz to {constants.GLOBAL_SAMPLING_RATE} Hz using SpikeInterface"
            )
            mock_logging.info.assert_any_call(
                f"Successfully resampled recording to {constants.GLOBAL_SAMPLING_RATE} Hz"
            )

        # Test logging when no resampling is needed
        mock_recording.get_sampling_frequency.return_value = (
            constants.GLOBAL_SAMPLING_RATE
        )

        with patch("neurodent.loading.lro_loading.logging") as mock_logging:
            organizer._apply_resampling(mock_recording)

            # Should log that no resampling is needed
            mock_logging.info.assert_called_with(
                f"Recording already at target sampling rate ({constants.GLOBAL_SAMPLING_RATE} Hz) or unable to determine, no resampling needed"
            )


class TestMNENJobsParameter:
    """Test n_jobs parameter functionality in MNE conversions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_default_n_jobs_equals_one(self, temp_dir):
        """Test that n_jobs defaults to 1 for safety."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        # Default n_jobs should be 1
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
        )

        assert organizer.n_jobs == 1

        # Mock MNE raw object
        mock_raw = Mock()
        mock_info = Mock()
        mock_info.sfreq = 2000.0
        mock_info.nchan = 2
        mock_info.ch_names = ["ch1", "ch2"]  # Add channel names
        mock_info.chs = []  # Add empty chs list for extract_mne_unit_info
        mock_info.__getitem__ = Mock(side_effect=lambda key: getattr(mock_info, key))
        mock_info.__contains__ = Mock(side_effect=lambda key: hasattr(mock_info, key))
        mock_raw.info = mock_info
        mock_raw.resample.return_value = mock_raw
        mock_raw.get_data.return_value = np.random.randn(2, 3600)

        mock_extract = Mock(return_value=mock_raw)

        # Mock SpikeInterface recording
        mock_si_rec = Mock()
        mock_si_rec.get_num_channels.return_value = 2
        mock_si_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_si_rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_si_rec.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_si_rec.get_duration.return_value = 3.6

        with (
            patch("mne.export.export_raw", side_effect=_export_creates_file),
            patch("spikeinterface.extractors.read_edf", return_value=mock_si_rec),
            patch("spikeinterface.preprocessing.resample", return_value=mock_si_rec),
        ):
            organizer.item = str(test_file)
            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract, intermediate="edf"
            )

        # MNE resample should NOT be called (new architecture uses SpikeInterface resampling)
        mock_raw.resample.assert_not_called()

    def test_explicit_n_jobs_override(self, temp_dir):
        """Test that users can override n_jobs parameter."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        # User specifies n_jobs=4
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
            n_jobs=4,
        )

        assert organizer.n_jobs == 4

        # Mock MNE raw object
        mock_raw = Mock()
        mock_info = Mock()
        mock_info.sfreq = 2000.0
        mock_info.nchan = 2
        mock_info.ch_names = ["ch1", "ch2"]  # Add channel names
        mock_info.chs = []  # Add empty chs list for extract_mne_unit_info
        mock_info.__getitem__ = Mock(side_effect=lambda key: getattr(mock_info, key))
        mock_info.__contains__ = Mock(side_effect=lambda key: hasattr(mock_info, key))
        mock_raw.info = mock_info
        mock_raw.resample.return_value = mock_raw
        mock_raw.get_data.return_value = np.random.randn(2, 3600)

        mock_extract = Mock(return_value=mock_raw)

        # Mock SpikeInterface recording
        mock_si_rec = Mock()
        mock_si_rec.get_num_channels.return_value = 2
        mock_si_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_si_rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_si_rec.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_si_rec.get_duration.return_value = 3.6

        with (
            patch("mne.export.export_raw", side_effect=_export_creates_file),
            patch("spikeinterface.extractors.read_edf", return_value=mock_si_rec),
            patch("spikeinterface.preprocessing.resample", return_value=mock_si_rec),
        ):
            organizer.item = str(test_file)
            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract, intermediate="edf"
            )

        # MNE resample should NOT be called (new architecture uses SpikeInterface resampling)
        mock_raw.resample.assert_not_called()

    def test_n_jobs_direct_method_call(self, temp_dir):
        """Test n_jobs parameter when calling convert_file_with_mne_to_recording directly."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
            n_jobs=1,  # Default
        )

        # Mock MNE raw object
        mock_raw = Mock()
        mock_info = Mock()
        mock_info.sfreq = 2000.0
        mock_info.nchan = 2
        mock_info.ch_names = ["ch1", "ch2"]  # Add channel names
        mock_info.chs = []  # Add empty chs list for extract_mne_unit_info
        mock_info.__getitem__ = Mock(side_effect=lambda key: getattr(mock_info, key))
        mock_info.__contains__ = Mock(side_effect=lambda key: hasattr(mock_info, key))
        mock_raw.info = mock_info
        mock_raw.resample.return_value = mock_raw
        mock_raw.get_data.return_value = np.random.randn(2, 1000)

        mock_extract = Mock(return_value=mock_raw)

        # Mock SpikeInterface recording
        mock_si_rec = Mock()
        mock_si_rec.get_num_channels.return_value = 2
        mock_si_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_si_rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_si_rec.get_channel_ids.return_value = np.array(["ch1", "ch2"])
        mock_si_rec.get_duration.return_value = 1.0

        with (
            patch("mne.export.export_raw", side_effect=_export_creates_file),
            patch("spikeinterface.extractors.read_edf", return_value=mock_si_rec),
            patch("spikeinterface.preprocessing.resample", return_value=mock_si_rec),
        ):
            # Call directly with override n_jobs=6
            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract,
                input_type="file",
                file_pattern="*.bdf",
                intermediate="edf",
                n_jobs=6,  # Override the instance default
            )

        # MNE resample should NOT be called (new architecture uses SpikeInterface resampling)
        mock_raw.resample.assert_not_called()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestResolveFuncPath:
    """Test ``LongRecordingOrganizer._resolve_func_path``."""

    def test_resolves_file_path(self):
        """File path to the mini-real reader resolves correctly."""
        func = LongRecordingOrganizer._resolve_func_path(
            "tests/integration/readers.py:read_bin_csv_pair"
        )
        assert callable(func)
        assert func.__name__ == "read_bin_csv_pair"

    def test_raises_on_missing_file(self):
        """Non-existent file raises an error."""
        with pytest.raises((ImportError, FileNotFoundError)):
            LongRecordingOrganizer._resolve_func_path(
                "nonexistent/path.py:some_func"
            )

    def test_raises_on_missing_attr(self):
        """Valid file but missing attribute raises AttributeError."""
        with pytest.raises(AttributeError):
            LongRecordingOrganizer._resolve_func_path(
                "tests/integration/readers.py:nonexistent_function_xyz"
            )

    def test_raises_on_bare_name(self):
        """Bare name (no colon) raises ImportError."""
        with pytest.raises(ImportError, match="file.py:func_name"):
            LongRecordingOrganizer._resolve_func_path("read_nwb_recording")


class TestExtractFuncFilePathResolution:
    """Verify that file-path resolution works silently (no warning/info)."""

    @pytest.fixture
    def lro_mode_none(self, tmp_path):
        """Create an LRO with mode=None so no data loading happens."""
        return LongRecordingOrganizer(str(tmp_path), mode=None)

    def test_si_file_path_resolves_silently(self, lro_mode_none):
        """SI mode resolves file path without logging warnings."""
        func_name = "tests/integration/readers.py:read_bin_csv_pair"
        with (
            patch("neurodent.loading.lro_loading.logging") as mock_logging,
            patch.object(lro_mode_none, "convert_file_with_si_to_recording"),
        ):
            lro_mode_none.detect_and_load_data(
                mode="si",
                extract_func=func_name,
            )
            mock_logging.warning.assert_not_called()
            mock_logging.info.assert_not_called()

    def test_mne_file_path_resolves_silently(self, lro_mode_none):
        """MNE mode resolves file path without logging warnings."""
        func_name = "tests/integration/readers.py:read_bin_csv_pair"
        with (
            patch("neurodent.loading.lro_loading.logging") as mock_logging,
            patch.object(lro_mode_none, "convert_file_with_mne_to_recording"),
        ):
            lro_mode_none.detect_and_load_data(
                mode="mne",
                extract_func=func_name,
            )
            mock_logging.warning.assert_not_called()
            mock_logging.info.assert_not_called()

    def test_si_builtin_extractor_no_warning(self, lro_mode_none):
        """SI mode does NOT warn when using a built-in SI extractor name."""
        with (
            patch("neurodent.loading.lro_loading.logging") as mock_logging,
            patch.object(lro_mode_none, "convert_file_with_si_to_recording"),
        ):
            lro_mode_none.detect_and_load_data(
                mode="si",
                extract_func="read_nwb_recording",
            )
            mock_logging.warning.assert_not_called()
            mock_logging.info.assert_not_called()


class TestZeroSampleRecordingCheck:
    """Tests for the unified 0-sample check in convert_file_with_si_to_recording.

    Ensures that a 0-duration file (valid header but no data) is detected and
    logged, so that _iter_valid_recordings() can skip it downstream.
    """

    def test_discovered_file_zero_samples_logs_warning(self, temp_dir, caplog):
        """DiscoveredFile branch logs warning for 0-sample recording."""
        import logging as _logging
        from neurodent.loading.discovery import DiscoveredFile

        df = DiscoveredFile(
            paths=("/tmp/a.bin", "/tmp/a.csv"), metadata={"session": "s1"}
        )
        organizer = LongRecordingOrganizer(None, mode=None)
        organizer.item = df
        organizer.n_truncate = 0
        organizer.truncate = False
        organizer.manual_datetimes = None
        organizer.datetimes_are_start = True

        mock_rec = Mock()
        mock_rec.get_total_samples.return_value = 0
        mock_rec.get_num_channels.return_value = 2
        mock_rec.get_channel_ids.return_value = ["ch1", "ch2"]
        mock_rec.get_sampling_frequency.return_value = 1000.0
        mock_rec.get_duration.return_value = 0.0
        mock_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_rec.get_total_duration.return_value = 0.0
        mock_rec.has_scaleable_traces.return_value = False

        extract_func = Mock(return_value=mock_rec)

        with caplog.at_level(_logging.WARNING):
            organizer.convert_file_with_si_to_recording(extract_func)

        assert organizer.LongRecording is not None
        assert any("0-sample recording" in msg for msg in caplog.messages)

    def test_single_file_zero_samples_logs_warning(self, temp_dir, caplog):
        """Single-file branch logs warning for 0-sample recording."""
        import logging as _logging

        organizer = LongRecordingOrganizer(None, mode=None)
        organizer.item = "/tmp/test.bin"
        organizer.n_truncate = 0
        organizer.truncate = False
        organizer.manual_datetimes = None
        organizer.datetimes_are_start = True

        mock_rec = Mock()
        mock_rec.get_total_samples.return_value = 0
        mock_rec.get_num_channels.return_value = 2
        mock_rec.get_channel_ids.return_value = ["ch1", "ch2"]
        mock_rec.get_sampling_frequency.return_value = 1000.0
        mock_rec.get_duration.return_value = 0.0
        mock_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_rec.get_total_duration.return_value = 0.0
        mock_rec.has_scaleable_traces.return_value = False

        extract_func = Mock(return_value=mock_rec)

        with caplog.at_level(_logging.WARNING):
            organizer.convert_file_with_si_to_recording(extract_func)

        assert organizer.LongRecording is not None
        assert any("0-sample recording" in msg for msg in caplog.messages)

    def test_nonzero_samples_no_warning(self, temp_dir, caplog):
        """No warning for recordings with samples."""
        import logging as _logging

        organizer = LongRecordingOrganizer(None, mode=None)
        organizer.item = "/tmp/test.bin"
        organizer.n_truncate = 0
        organizer.truncate = False
        organizer.manual_datetimes = None
        organizer.datetimes_are_start = True

        mock_rec = Mock()
        mock_rec.get_total_samples.return_value = 5000
        mock_rec.get_num_channels.return_value = 2
        mock_rec.get_channel_ids.return_value = ["ch1", "ch2"]
        mock_rec.get_sampling_frequency.return_value = 1000.0
        mock_rec.get_duration.return_value = 5.0
        mock_rec.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_rec.get_total_duration.return_value = 5.0
        mock_rec.has_scaleable_traces.return_value = False

        extract_func = Mock(return_value=mock_rec)

        with caplog.at_level(_logging.WARNING):
            organizer.convert_file_with_si_to_recording(extract_func)

        assert not any("0-sample recording" in msg for msg in caplog.messages)

    @pytest.mark.skipif(si is None, reason="SpikeInterface not available")
    def test_multi_file_extract_func_crash_creates_placeholder(self, temp_dir, caplog):
        """Crashing extract_func on a multi-file DiscoveredFile yields a 0-sample placeholder.

        This guards against corrupt/empty file pairs (e.g. header-only CSV) killing
        the entire animal's pipeline run instead of just skipping the bad file.
        """
        import logging as _logging
        from neurodent.loading.discovery import DiscoveredFile

        df = DiscoveredFile(
            paths=("/tmp/bad.bin", "/tmp/bad.csv"), metadata={"session": "s1"}
        )
        organizer = LongRecordingOrganizer(None, mode=None)
        organizer.item = df
        organizer.n_truncate = 0
        organizer.truncate = False
        organizer.manual_datetimes = None
        organizer.datetimes_are_start = True

        def crashing_extract_func(discovered_file, **kwargs):
            raise ValueError("CSV metadata file has no data rows (header-only): bad.csv")

        with caplog.at_level(_logging.WARNING):
            organizer.convert_file_with_si_to_recording(crashing_extract_func)

        assert organizer.LongRecording is not None
        assert any("extract_func failed" in msg for msg in caplog.messages)
        assert any("0-sample" in msg for msg in caplog.messages)

    @pytest.mark.skipif(si is None, reason="SpikeInterface not available")
    def test_multi_file_extract_func_unexpected_error_propagates(self, temp_dir):
        """Non-file-data errors (e.g. RuntimeError) are NOT swallowed."""
        from neurodent.loading.discovery import DiscoveredFile

        df = DiscoveredFile(
            paths=("/tmp/bad.bin", "/tmp/bad.csv"), metadata={"session": "s1"}
        )
        organizer = LongRecordingOrganizer(None, mode=None)
        organizer.item = df
        organizer.n_truncate = 0
        organizer.truncate = False
        organizer.manual_datetimes = None
        organizer.datetimes_are_start = True

        def buggy_extract_func(discovered_file, **kwargs):
            raise RuntimeError("unexpected internal error")

        with pytest.raises(RuntimeError, match="unexpected internal error"):
            organizer.convert_file_with_si_to_recording(buggy_extract_func)


@pytest.mark.core
@pytest.mark.spikeinterface
class TestReadBinCsvPair:
    """Tests for read_bin_csv_pair validation of malformed file pairs."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path

    def test_empty_csv_raises(self, temp_dir):
        """header-only CSV (no data rows) raises ValueError with clear message."""
        from tests.integration.readers import read_bin_csv_pair
        from neurodent.loading.discovery import DiscoveredFile

        bin_path = str(temp_dir / "test.bin")
        csv_path = str(temp_dir / "test.csv")

        np.zeros(100, dtype=np.float32).tofile(bin_path)
        with open(csv_path, "w") as f:
            f.write("Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit\n")

        df = DiscoveredFile(paths=(bin_path, csv_path), metadata={"session": "s1"})
        with pytest.raises(ValueError, match="CSV metadata file has no data rows"):
            read_bin_csv_pair(df)

    def test_empty_bin_raises(self, temp_dir):
        """Zero-byte .bin file raises ValueError with clear message."""
        from tests.integration.readers import read_bin_csv_pair
        from neurodent.loading.discovery import DiscoveredFile

        bin_path = str(temp_dir / "test.bin")
        csv_path = str(temp_dir / "test.csv")

        open(bin_path, "wb").close()  # empty file
        with open(csv_path, "w") as f:
            f.write("Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit\n")
            f.write("0,0,LMot,,1000.0,uV,float32,2022-01-01\n")

        df = DiscoveredFile(paths=(bin_path, csv_path), metadata={"session": "s1"})
        with pytest.raises(ValueError, match="Binary file is empty"):
            read_bin_csv_pair(df)

    def test_fortran_order_channels_read_correctly(self, temp_dir):
        """Column-major (Fortran-order) binary is read with correct channel assignment.

        Writes a file where each channel has a distinct constant value,
        stored in Fortran order (all samples of ch0, then ch1, etc.).
        Verifies that read_bin_csv_pair returns each channel with its
        expected value — a C-order reader would garble the channels.
        """
        from tests.integration.readers import read_bin_csv_pair
        from neurodent.loading.discovery import DiscoveredFile

        n_channels = 3
        n_samples = 200
        fs = 1000.0

        # Each channel has a distinct constant: ch0=10, ch1=20, ch2=30
        data = np.column_stack(
            [np.full(n_samples, (ch + 1) * 10.0, dtype=np.float32) for ch in range(n_channels)]
        )

        # Write in Fortran order (column-major): all ch0 samples, then ch1, then ch2
        bin_path = str(temp_dir / "test_ColMajor.bin")
        data.flatten(order="F").tofile(bin_path)

        csv_path = str(temp_dir / "test_Meta.csv")
        with open(csv_path, "w") as f:
            f.write("Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit\n")
            for ch in range(n_channels):
                f.write(f"{ch},{ch},Ch{ch},,{fs},uV,float32,2022-01-01\n")

        df = DiscoveredFile(paths=(bin_path, csv_path), metadata={"session": "s1"})
        rec = read_bin_csv_pair(df)
        traces = rec.get_traces(return_scaled=True)

        assert traces.shape == (n_samples, n_channels)
        for ch in range(n_channels):
            expected = (ch + 1) * 10.0
            np.testing.assert_allclose(
                traces[:, ch], expected,
                err_msg=f"Channel {ch} should be all {expected} but got mean={traces[:, ch].mean():.1f}",
            )


@pytest.mark.core
@pytest.mark.spikeinterface
class TestZeroSampleMerge:
    """Tests that merging a 0-sample LRO updates general metadata but filters file metadata.

    When a 0-sample LRO is merged, _update_metadata_after_merge is still called
    so that dt_end etc. are updated, but 0-duration entries are filtered out of
    file_end_datetimes/file_durations to avoid corrupting TimestampMapper.
    """

    def _make_lro(self, total_samples, channel_names, file_end_datetimes=None, file_durations=None):
        """Create a mock LRO with the given properties."""
        lro = LongRecordingOrganizer(None, mode=None)
        lro.channel_names = channel_names

        mock_rec = Mock()
        mock_rec.get_total_samples.return_value = total_samples
        lro.LongRecording = mock_rec

        lro.meta = Mock()
        lro.meta.f_s = 1000.0
        lro.meta.n_channels = len(channel_names)
        lro.meta.dt_end = datetime(2023, 1, 1, 12, 0)
        lro.item = "test_item"

        lro.file_end_datetimes = file_end_datetimes or []
        lro.file_durations = file_durations or []

        return lro

    def test_zero_sample_merge_filters_zero_duration_but_updates_dt_end(self, caplog):
        """Merging a 0-sample LRO should update dt_end but not extend file_end_datetimes/file_durations."""
        import logging as _logging

        base_lro = self._make_lro(
            total_samples=5000,
            channel_names=["ch1", "ch2"],
            file_end_datetimes=[datetime(2023, 1, 1, 12, 0)],
            file_durations=[5.0],
        )

        zero_lro = self._make_lro(
            total_samples=0,
            channel_names=["ch1", "ch2"],
            file_end_datetimes=[datetime(2023, 1, 1, 12, 5)],
            file_durations=[0.0],
        )
        zero_lro.meta.dt_end = datetime(2023, 1, 1, 12, 5)

        with caplog.at_level(_logging.WARNING):
            base_lro.merge(zero_lro)

        # dt_end SHOULD have been updated
        assert base_lro.meta.dt_end == datetime(2023, 1, 1, 12, 5)

        # file_end_datetimes/file_durations should NOT have been extended
        # (0-duration entries are filtered out)
        assert len(base_lro.file_end_datetimes) == 1, (
            f"Expected 1 file_end_datetime, got {len(base_lro.file_end_datetimes)}"
        )
        assert len(base_lro.file_durations) == 1, (
            f"Expected 1 file_duration, got {len(base_lro.file_durations)}"
        )
        assert base_lro.file_durations[0] == 5.0

        # Warning should be logged
        assert any("0 samples" in msg for msg in caplog.messages)

    def test_nonzero_sample_merge_extends_metadata(self):
        """Merging a non-zero LRO should extend file_end_datetimes and file_durations."""
        base_lro = self._make_lro(
            total_samples=5000,
            channel_names=["ch1", "ch2"],
            file_end_datetimes=[datetime(2023, 1, 1, 12, 0)],
            file_durations=[5.0],
        )

        other_lro = self._make_lro(
            total_samples=3000,
            channel_names=["ch1", "ch2"],
            file_end_datetimes=[datetime(2023, 1, 1, 12, 10)],
            file_durations=[3.0],
        )

        with patch("neurodent.loading.lro_merge.si") as mock_si:
            mock_concat = Mock()
            mock_si.concatenate_recordings.return_value = mock_concat
            base_lro.merge(other_lro)

        # Metadata SHOULD have been extended
        assert len(base_lro.file_end_datetimes) == 2
        assert len(base_lro.file_durations) == 2
        assert base_lro.file_durations == [5.0, 3.0]


class TestMergeChannelNameAbbreviation:
    """Tests that merge validation compares channel names by abbreviation."""

    def _make_lro(self, total_samples, channel_names):
        """Create a mock LRO with the given properties."""
        lro = LongRecordingOrganizer(None, mode=None)
        lro.channel_names = channel_names

        mock_rec = Mock()
        mock_rec.get_total_samples.return_value = total_samples
        lro.LongRecording = mock_rec

        lro.meta = Mock()
        lro.meta.f_s = 1000.0
        lro.meta.n_channels = len(channel_names)
        lro.meta.dt_end = datetime(2023, 1, 1, 12, 0)
        lro.item = "test_item"

        lro.file_end_datetimes = []
        lro.file_durations = []

        return lro

    @pytest.mark.mutates_constants
    def test_same_abbreviation_different_raw_names_succeeds(self):
        """Merging LROs with different raw names but same abbreviations should rename and succeed."""
        from neurodent import constants

        constants.set_channel_map({
            "LBar": ["L Barrel", "L Barrel Ctx"],
            "LMot": ["L Motor", "L Motor Ctx"],
        })
        base_lro = self._make_lro(5000, ["L Barrel", "L Motor"])
        other_lro = self._make_lro(3000, ["L Barrel Ctx", "L Motor Ctx"])
        original_other_rec = other_lro.LongRecording
        renamed_rec = Mock()
        original_other_rec.rename_channels.return_value = renamed_rec

        with patch("neurodent.loading.lro_merge.si") as mock_si:
            mock_si.concatenate_recordings.return_value = Mock()
            base_lro.merge(other_lro)

            # rename_channels should have been called with base's channel names
            original_other_rec.rename_channels.assert_called_once_with(
                new_channel_ids=["L Barrel", "L Motor"]
            )
            # The renamed rec should be passed to concatenate_recordings
            call_args = mock_si.concatenate_recordings.call_args[0][0]
            assert call_args[1] is renamed_rec

        # other_lro's channel_names should have been updated to match base
        assert other_lro.channel_names == ["L Barrel", "L Motor"]

    def test_different_abbreviations_raises(self):
        """Merging LROs with genuinely different channels should fail."""
        base_lro = self._make_lro(5000, ["L Barrel", "L Motor"])
        other_lro = self._make_lro(3000, ["L Hipp", "L Motor"])

        with pytest.raises(ValueError, match="Channel names mismatch"):
            base_lro.merge(other_lro)

    def test_unparseable_names_falls_back_to_exact_match(self):
        """When abbreviation parsing fails, fall back to exact string comparison."""
        # Same unparseable names — should succeed
        base_lro = self._make_lro(5000, ["weird_ch1", "weird_ch2"])
        other_lro = self._make_lro(3000, ["weird_ch1", "weird_ch2"])

        with patch("neurodent.loading.lro_merge.si") as mock_si:
            mock_si.concatenate_recordings.return_value = Mock()
            base_lro.merge(other_lro)

    def test_unparseable_names_different_raises(self):
        """When abbreviation parsing fails and names differ, should raise."""
        base_lro = self._make_lro(5000, ["weird_ch1", "weird_ch2"])
        other_lro = self._make_lro(3000, ["weird_ch1", "weird_ch3"])

        with pytest.raises(ValueError, match="Channel names mismatch"):
            base_lro.merge(other_lro)


class TestAbbreviateChannelNames:
    """Tests for the resolve_channels utility (exact lookup against the configured map)."""

    @pytest.mark.mutates_constants
    def test_configured_names_are_abbreviated(self):
        """Raw names configured under their abbreviation resolve exactly."""
        from neurodent import constants

        constants.set_channel_map({
            "LBar": ["LBar", "L Barrel"],
            "LMot": ["LMot", "L Motor"],
            "RHip": ["RHip", "R Hipp"],
        })
        assert resolve_channels(["L Barrel", "L Motor", "R Hipp"]) == ["LBar", "LMot", "RHip"]

    def test_unparseable_names_pass_through(self):
        """Unmappable names are returned unchanged (with a warning)."""
        names = ["weird_ch1", "weird_ch2"]
        result = resolve_channels(names)
        assert result == ["weird_ch1", "weird_ch2"]

    @pytest.mark.mutates_constants
    def test_mixed_names(self):
        """Mix of configured and unmappable names."""
        from neurodent import constants

        constants.set_channel_map({"LBar": ["LBar", "L Barrel"], "RMot": ["RMot", "R Motor"]})
        assert resolve_channels(["L Barrel", "weird_ch", "R Motor"]) == ["LBar", "weird_ch", "RMot"]

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert resolve_channels([]) == []

    @pytest.mark.mutates_constants
    def test_variant_names_same_abbreviation(self):
        """Different raw names configured under one abbreviation resolve to it."""
        from neurodent import constants

        constants.set_channel_map({"LBar": ["LBar", "L Barrel", "L Barrel Ctx"]})
        assert resolve_channels(["L Barrel"]) == resolve_channels(["L Barrel Ctx"]) == ["LBar"]
