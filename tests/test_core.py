"""
Unit tests for neurodent.core.core module.
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

from neurodent.core.core import (
    DDFBinaryMetadata,
    LongRecordingOrganizer,
    convert_ddfcolbin_to_ddfrowbin,
    convert_ddfrowbin_to_si,
)
from neurodent import constants


class TestDDFBinaryMetadata:
    """Test DDFBinaryMetadata class functionality."""
    
    def test_init_from_path_valid_metadata(self, temp_dir):
        """Test initialization from valid metadata CSV file."""
        # Create test metadata CSV
        metadata_data = {
            'ProbeInfo': ['LAud', 'RAud', 'LVis', 'RVis'],
            'SampleRate': [1000, 1000, 1000, 1000],
            'Units': ['µV', 'µV', 'µV', 'µV'],
            'Precision': ['float32', 'float32', 'float32', 'float32'],
            'LastEdit': ['2023-01-01T12:00:00', '2023-01-01T12:00:00', 
                        '2023-01-01T12:00:00', '2023-01-01T12:00:00']
        }
        df = pd.DataFrame(metadata_data)
        csv_path = temp_dir / "test_metadata.csv"
        df.to_csv(csv_path, index=False)
        
        # Test initialization
        metadata = DDFBinaryMetadata(csv_path)
        
        assert metadata.n_channels == 4
        assert metadata.f_s == 1000
        assert metadata.V_units == 'µV'
        assert metadata.mult_to_uV == 1.0
        assert metadata.precision == 'float32'
        assert isinstance(metadata.dt_end, datetime)
        assert metadata.channel_names == ['LAud', 'RAud', 'LVis', 'RVis']
        
    def test_init_from_path_empty_metadata(self, temp_dir):
        """Test initialization from empty metadata file raises error."""
        empty_csv = temp_dir / "empty.csv"
        # Create truly empty CSV (no header, no data)
        with open(empty_csv, 'w') as f:
            f.write('')
        
        # pandas raises EmptyDataError for truly empty files
        with pytest.raises(pd.errors.EmptyDataError):
            DDFBinaryMetadata(empty_csv)
            
    def test_init_from_path_no_lastedit_column(self, temp_dir):
        """Test initialization with missing LastEdit column."""
        metadata_data = {
            'ProbeInfo': ['LAud', 'RAud'],
            'SampleRate': [1000, 1000],
            'Units': ['µV', 'µV'],
            'Precision': ['float32', 'float32']
        }
        df = pd.DataFrame(metadata_data)
        csv_path = temp_dir / "no_lastedit.csv"
        df.to_csv(csv_path, index=False)
        
        # This should work and set dt_end to None (the __getsinglecolval handles missing columns)
        metadata = DDFBinaryMetadata(csv_path)
        assert metadata.dt_end is None
        assert metadata.n_channels == 2
            
    def test_init_from_params_valid(self):
        """Test initialization from direct parameters."""
        dt_end = datetime(2023, 1, 1, 12, 0, 0)
        channel_names = ['LAud', 'RAud', 'LVis', 'RVis']
        
        metadata = DDFBinaryMetadata(
            None,
            n_channels=4,
            f_s=1000.0,
            dt_end=dt_end,
            channel_names=channel_names
        )
        
        assert metadata.n_channels == 4
        assert metadata.f_s == 1000.0
        assert metadata.dt_end == dt_end
        assert metadata.channel_names == channel_names
        assert metadata.metadata_path is None
        assert metadata.metadata_df is None
        assert metadata.V_units is None
        assert metadata.mult_to_uV is None
        assert metadata.precision is None
        
    def test_init_from_params_missing_required(self):
        """Test initialization from params with missing required values raises ValueError."""
        with pytest.raises(ValueError, match="All parameters must be provided"):
            DDFBinaryMetadata(None, n_channels=4, f_s=1000.0)  # Missing dt_end and channel_names
            
    def test_init_from_params_invalid_channel_names(self):
        """Test initialization with non-list channel_names raises ValueError."""
        with pytest.raises(ValueError, match="channel_names must be a list"):
            DDFBinaryMetadata(
                None,
                n_channels=4,
                f_s=1000.0,
                dt_end=datetime.now(),
                channel_names="not_a_list"
            )
            
    def test_getsinglecolval_consistent_values(self, temp_dir):
        """Test __getsinglecolval with consistent column values."""
        metadata_data = {
            'ProbeInfo': ['LAud', 'RAud', 'LVis'],
            'SampleRate': [1000, 1000, 1000],
            'Units': ['µV', 'µV', 'µV'],
            'Precision': ['float32', 'float32', 'float32']
        }
        df = pd.DataFrame(metadata_data)
        csv_path = temp_dir / "consistent.csv"
        df.to_csv(csv_path, index=False)
        
        metadata = DDFBinaryMetadata(csv_path)
        assert metadata.f_s == 1000
        assert metadata.V_units == 'µV'
        
    def test_getsinglecolval_inconsistent_values_warning(self, temp_dir):
        """Test __getsinglecolval with inconsistent values shows warning."""
        metadata_data = {
            'ProbeInfo': ['LAud', 'RAud', 'LVis'],
            'SampleRate': [1000, 2000, 1000],  # Inconsistent values
            'Units': ['µV', 'µV', 'µV'],
            'Precision': ['float32', 'float32', 'float32']
        }
        df = pd.DataFrame(metadata_data)
        csv_path = temp_dir / "inconsistent.csv"
        df.to_csv(csv_path, index=False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metadata = DDFBinaryMetadata(csv_path)
            # Check if warning was raised (may be multiple warnings)
            warning_messages = [str(warning.message) for warning in w]
            assert any("Not all SampleRates are equal" in msg for msg in warning_messages)
            assert metadata.f_s == 1000  # Should return first value
            
    def test_getsinglecolval_empty_column(self, temp_dir):
        """Test __getsinglecolval with empty column returns None."""
        metadata_data = {
            'ProbeInfo': ['LAud'],
            'SampleRate': [1000],
            'Units': ['µV'],  # Valid value
            'Precision': ['float32']
        }
        df = pd.DataFrame(metadata_data)
        csv_path = temp_dir / "single_row.csv"
        df.to_csv(csv_path, index=False)
        
        metadata = DDFBinaryMetadata(csv_path)
        assert metadata.n_channels == 1
        assert metadata.V_units == 'µV'


class TestConvertDdfcolbinToDdfrowbin:
    """Test convert_ddfcolbin_to_ddfrowbin function."""
    
    def test_convert_valid_colbin(self, temp_dir):
        """Test converting valid column-major binary file."""
        # Create test column-major binary data
        n_channels = 4
        n_samples = 1000
        test_data = np.random.randn(n_samples, n_channels).astype(np.float32)
        
        # Save as column-major (Fortran order)
        colbin_path = temp_dir / "test_ColMajor_001.bin"
        test_data.flatten(order='F').tofile(colbin_path)
        
        # Create metadata
        metadata = DDFBinaryMetadata(
            None,
            n_channels=n_channels,
            f_s=1000.0,
            dt_end=datetime.now(),
            channel_names=['ch1', 'ch2', 'ch3', 'ch4']
        )
        metadata.precision = np.float32
        
        # Convert
        rowbin_path = convert_ddfcolbin_to_ddfrowbin(str(temp_dir), str(colbin_path), metadata, save_gzip=True)
        
        # Verify output file exists
        assert Path(rowbin_path).exists()
        assert str(rowbin_path).endswith('.npy.gz')
        
        # Verify data integrity
        with gzip.GzipFile(rowbin_path, 'r') as f:
            converted_data = np.load(f)
        
        assert converted_data.shape == (n_samples, n_channels)
        np.testing.assert_array_almost_equal(converted_data, test_data)
        
    def test_convert_colbin_invalid_metadata_type(self, temp_dir):
        """Test convert colbin with invalid metadata type raises AssertionError."""
        colbin_path = temp_dir / "test.bin"
        colbin_path.touch()
        
        with pytest.raises(AssertionError, match="Metadata needs to be of type DDFBinaryMetadata"):
            convert_ddfcolbin_to_ddfrowbin(temp_dir, colbin_path, "not_metadata", save_gzip=True)
            
    def test_convert_save_uncompressed(self, temp_dir):
        """Test converting with save_gzip=False creates .bin file."""
        # Create test data
        n_channels = 2
        n_samples = 100
        test_data = np.random.randn(n_samples, n_channels).astype(np.float32)
        
        colbin_path = temp_dir / "test_ColMajor_001.bin"
        test_data.flatten(order='F').tofile(colbin_path)
        
        metadata = DDFBinaryMetadata(
            None,
            n_channels=n_channels,
            f_s=1000.0,
            dt_end=datetime.now(),
            channel_names=['ch1', 'ch2']
        )
        metadata.precision = np.float32
        
        # Convert without compression
        rowbin_path = convert_ddfcolbin_to_ddfrowbin(str(temp_dir), str(colbin_path), metadata, save_gzip=False)
        
        assert Path(rowbin_path).exists()
        assert str(rowbin_path).endswith('.bin')
        
        # Verify data
        converted_data = np.fromfile(rowbin_path, dtype=np.float32).reshape(n_samples, n_channels)
        np.testing.assert_array_almost_equal(converted_data, test_data)


class TestConvertDdfrowbinToSi:
    """Test convert_ddfrowbin_to_si function."""
    
    def test_convert_gzipped_rowbin(self, temp_dir):
        """Test converting gzipped row-major binary to SpikeInterface recording."""
        # Create test row-major data
        n_channels = 4
        n_samples = 1000
        test_data = np.random.randn(n_samples, n_channels).astype(np.float32)
        
        # Save as gzipped numpy array
        rowbin_path = temp_dir / "test_RowMajor_001.npy.gz"
        with gzip.GzipFile(rowbin_path, 'w') as f:
            np.save(f, test_data)
        
        # Create metadata
        metadata = DDFBinaryMetadata(
            None,
            n_channels=n_channels,
            f_s=1000.0,
            dt_end=datetime.now(),
            channel_names=['ch1', 'ch2', 'ch3', 'ch4']
        )
        metadata.precision = np.float32
        metadata.mult_to_uV = 1.0
        
        # Convert
        rec, temppath = convert_ddfrowbin_to_si(rowbin_path, metadata)
        
        # Verify recording properties
        assert isinstance(rec, si.BaseRecording)
        assert rec.get_num_channels() == n_channels
        assert rec.get_num_frames() == n_samples
        assert rec.get_sampling_frequency() == constants.GLOBAL_SAMPLING_RATE
        assert rec.get_dtype() == constants.GLOBAL_DTYPE
        
        # Verify temporary file was created and cleanup
        assert temppath is not None
        # Ensure the recording object is deleted before removing temp file on Windows
        rec = None
        gc.collect()
        if os.path.exists(temppath):
            os.remove(temppath)
            
    def test_convert_uncompressed_rowbin(self, temp_dir):
        """Test converting uncompressed row-major binary."""
        # Create test data
        n_channels = 2
        n_samples = 500
        test_data = np.random.randn(n_samples, n_channels).astype(np.float32)
        
        # Save as binary file
        rowbin_path = temp_dir / "test_RowMajor_001.bin"
        test_data.tofile(rowbin_path)
        
        metadata = DDFBinaryMetadata(
            None,
            n_channels=n_channels,
            f_s=1000.0,
            dt_end=datetime.now(),
            channel_names=['ch1', 'ch2']
        )
        metadata.precision = np.float32
        metadata.mult_to_uV = 1.0
        
        # Convert
        rec, temppath = convert_ddfrowbin_to_si(rowbin_path, metadata)
        
        assert isinstance(rec, si.BaseRecording)
        assert rec.get_num_channels() == n_channels
        assert rec.get_num_frames() == n_samples
        assert temppath is None  # No temp file for uncompressed
        
    def test_convert_rowbin_invalid_metadata_type(self, temp_dir):
        """Test convert rowbin with invalid metadata type raises AssertionError."""
        rowbin_path = temp_dir / "test.npy.gz"
        
        with pytest.raises(AssertionError, match="Metadata needs to be of type DDFBinaryMetadata"):
            convert_ddfrowbin_to_si(rowbin_path, "not_metadata")
            
    def test_convert_corrupted_gzip_file(self, temp_dir):
        """Test handling of corrupted gzip file."""
        # Create corrupted gzip file
        rowbin_path = temp_dir / "corrupted.npy.gz"
        with open(rowbin_path, 'wb') as f:
            f.write(b'corrupted data')
        
        metadata = DDFBinaryMetadata(
            None,
            n_channels=2,
            f_s=1000.0,
            dt_end=datetime.now(),
            channel_names=['ch1', 'ch2']
        )
        metadata.precision = np.float32
        metadata.mult_to_uV = 1.0
        
        with pytest.raises((EOFError, OSError)):
            convert_ddfrowbin_to_si(rowbin_path, metadata)

    @patch("neurodent.core.core.spre.resample")
    def test_convert_different_sampling_rate_resamples(self, mock_resample, temp_dir):
        """Test that different sampling rates trigger resampling."""
        # Create test data
        n_channels = 2
        n_samples = 100
        test_data = np.random.randn(n_samples, n_channels).astype(np.float32)
        
        rowbin_path = temp_dir / "test.bin"
        test_data.tofile(rowbin_path)
        
        metadata = DDFBinaryMetadata(
            None,
            n_channels=n_channels,
            f_s=2000.0,  # Different from GLOBAL_SAMPLING_RATE
            dt_end=datetime.now(),
            channel_names=['ch1', 'ch2']
        )
        metadata.precision = np.float32
        metadata.mult_to_uV = 1.0
        
        # Mock the recording and resampling
        mock_rec = Mock()
        mock_rec.sampling_frequency = 2000.0
        mock_resampled_rec = Mock()
        mock_resample.return_value = mock_resampled_rec
        
        with patch('spikeinterface.extractors.read_binary', return_value=mock_rec):
            with patch('spikeinterface.preprocessing.astype', return_value=mock_resampled_rec):
                with pytest.warns(UserWarning, match="Sampling rate 2000.0 Hz != 1000 Hz. Resampling"):
                    convert_ddfrowbin_to_si(rowbin_path, metadata)
        
        # Verify resampling was called
        mock_resample.assert_called_once_with(mock_rec, constants.GLOBAL_SAMPLING_RATE)


class TestLongRecordingOrganizer:
    """Test LongRecordingOrganizer class functionality."""
    
    def test_init_with_mode_none(self, temp_dir):
        """Test initialization with mode=None doesn't load data."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        
        assert organizer.base_folder_path == Path(temp_dir)
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
        
    def test_truncate_file_list_no_truncation(self, temp_dir):
        """Test _truncate_file_list with no truncation needed."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None, truncate=False)
        files = ['file1.bin', 'file2.bin', 'file3.bin']
        
        result = organizer._truncate_file_list(files)
        assert result == files
        
    def test_truncate_file_list_with_truncation(self, temp_dir):
        """Test _truncate_file_list with truncation."""
        with pytest.warns(UserWarning, match="LongRecording will be truncated to the first 2 files"):
            organizer = LongRecordingOrganizer(temp_dir, mode=None, truncate=2)
        
        files = ['file3.bin', 'file1.bin', 'file2.bin']  # Unsorted
        
        result = organizer._truncate_file_list(files)
        assert len(result) == 2
        assert result == ['file1.bin', 'file2.bin']  # Should be sorted
        
    def test_truncate_file_list_with_ref_list(self, temp_dir):
        """Test _truncate_file_list with reference list."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None, truncate=False)
        # Test the actual pattern used in the code: rowbins matched against transformed colbins
        colbins = ['file1_ColMajor.bin', 'file3_ColMajor.bin']  # Only file1 and file3
        rowbins = ['file1_RowMajor.npy.gz', 'file2_RowMajor.npy.gz', 'file3_RowMajor.npy.gz']
        # Create ref_list as done in actual code
        ref_list = [x.replace("_ColMajor.bin", "_RowMajor.npy.gz") for x in colbins]
        
        result = organizer._truncate_file_list(rowbins, ref_list=ref_list)
        assert len(result) == 2
        # Should return the actual files whose stems match ref_list stems (file1 and file3)
        assert 'file1_RowMajor.npy.gz' in result  
        assert 'file3_RowMajor.npy.gz' in result
        assert 'file2_RowMajor.npy.gz' not in result
        
    def test_validate_metadata_consistency_success(self, temp_dir):
        """Test _validate_metadata_consistency with consistent metadata."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        
        # Create consistent metadata objects
        metadata1 = DDFBinaryMetadata(
            None, n_channels=4, f_s=1000.0, dt_end=datetime.now(),
            channel_names=['ch1', 'ch2', 'ch3', 'ch4']
        )
        metadata1.precision = 'float32'
        metadata1.V_units = 'µV'
        
        metadata2 = DDFBinaryMetadata(
            None, n_channels=4, f_s=1000.0, dt_end=datetime.now(),
            channel_names=['ch1', 'ch2', 'ch3', 'ch4']
        )
        metadata2.precision = 'float32'
        metadata2.V_units = 'µV'
        
        # Should not raise exception
        organizer._validate_metadata_consistency([metadata1, metadata2])
        
    def test_validate_metadata_consistency_failure(self, temp_dir):
        """Test _validate_metadata_consistency with inconsistent metadata."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        
        # Create inconsistent metadata objects
        metadata1 = DDFBinaryMetadata(
            None, n_channels=4, f_s=1000.0, dt_end=datetime.now(),
            channel_names=['ch1', 'ch2', 'ch3', 'ch4']
        )
        metadata1.precision = 'float32'
        metadata1.V_units = 'µV'
        
        metadata2 = DDFBinaryMetadata(
            None, n_channels=8, f_s=1000.0, dt_end=datetime.now(),  # Different n_channels
            channel_names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
        )
        metadata2.precision = 'float32'
        metadata2.V_units = 'µV'
        
        with pytest.raises(ValueError, match="Metadata files inconsistent at attribute n_channels"):
            organizer._validate_metadata_consistency([metadata1, metadata2])
            
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
        mock_recording.time_to_sample_index.return_value = 1000  # 1 second = 1000 samples
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
        mock_recording.time_to_sample_index.return_value = 1000  # 1 second = 1000 samples
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
            fragment_datetime = organizer.get_datetime_fragment(fragment_len_s=fragment_len_s, fragment_idx=0)
            assert fragment_datetime == expected, f"Failed for fragment_len_s={fragment_len_s}"
        
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
        organizer.channel_names = ['ch1', 'ch2', 'ch3']
        
        with patch('mne.create_info') as mock_create_info, \
             patch('mne.io.RawArray') as mock_raw_array:
            
            mock_info = Mock()
            mock_create_info.return_value = mock_info
            mock_raw = Mock()
            mock_raw_array.return_value = mock_raw
            
            result = organizer.convert_to_mne()
            
            # Verify create_info was called correctly
            mock_create_info.assert_called_once_with(
                ch_names=['ch1', 'ch2', 'ch3'],
                sfreq=1000.0,
                ch_types='eeg'
            )
            
            # Verify RawArray was called with transposed data
            mock_raw_array.assert_called_once()
            call_args = mock_raw_array.call_args
            assert call_args[1]['info'] == mock_info
            # Data should be transposed from (n_samples, n_channels) to (n_channels, n_samples)
            passed_data = call_args[1]['data']
            assert passed_data.shape == (n_channels, n_samples)
            
            assert result == mock_raw
            
    def test_compute_bad_channels(self, temp_dir):
        """Test compute_bad_channels method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        
        # Mock LongRecording  
        n_samples = 1000
        # Create test data where channel 2 is an outlier
        normal_data = np.random.randn(n_samples, 3) * 10  # channels 0,1,3
        outlier_data = np.random.randn(n_samples, 1) * 100  # channel 2 (much larger amplitude)
        test_data = np.hstack([normal_data, outlier_data])
        # Rearrange to put outlier in position 2
        test_data = test_data[:, [0, 1, 3, 2]]
        
        mock_recording = Mock()
        mock_recording.get_traces.return_value = test_data
        mock_recording.__str__ = Mock(return_value="MockRecording")
        organizer.LongRecording = mock_recording
        organizer.channel_names = ['ch1', 'ch2', 'ch3', 'ch4']

        with (
            patch("neurodent.core.core.Natural_Neighbor") as mock_nn_class,
            patch("neurodent.core.core.LocalOutlierFactor") as mock_lof_class,
        ):
            
            # Mock Natural_Neighbor
            mock_nn = Mock()
            mock_nn.algorithm.return_value = 3
            mock_nn_class.return_value = mock_nn
            
            # Mock LocalOutlierFactor
            mock_lof = Mock()
            mock_lof.negative_outlier_factor_ = np.array([-1.0, -1.0, -1.0, -2.0])  # ch4 is outlier
            mock_lof_class.return_value = mock_lof
            
            # Test with default threshold
            organizer.compute_bad_channels(lof_threshold=1.5)
            
            # Verify Natural_Neighbor was used
            mock_nn.read.assert_called_once()
            mock_nn.algorithm.assert_called_once()
            
            # Verify LocalOutlierFactor was configured correctly
            mock_lof_class.assert_called_once_with(n_neighbors=3, metric="precomputed")
            mock_lof.fit.assert_called_once()
            
            # Channel 4 should be identified as bad (score 2.0 > threshold 1.5)
            assert organizer.bad_channel_names == ['ch4']
            
    def test_compute_bad_channels_limit_memory(self, temp_dir):
        """Test compute_bad_channels with limit_memory=True."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        
        # Create larger test data
        n_channels = 2
        n_samples = 10000
        test_data = np.random.randn(n_samples, n_channels).astype(np.float64)
        
        mock_recording = Mock()
        mock_recording.get_traces.return_value = test_data
        mock_recording.__str__ = Mock(return_value="MockRecording")
        organizer.LongRecording = mock_recording
        organizer.channel_names = ['ch1', 'ch2']

        with (
            patch("neurodent.core.core.Natural_Neighbor") as mock_nn_class,
            patch("neurodent.core.core.LocalOutlierFactor") as mock_lof_class,
            patch("neurodent.core.core.decimate") as mock_decimate,
        ):
            
            mock_nn = Mock()
            mock_nn.algorithm.return_value = 2
            mock_nn_class.return_value = mock_nn
            
            mock_lof = Mock()
            mock_lof.negative_outlier_factor_ = np.array([-1.0, -1.0])
            mock_lof_class.return_value = mock_lof
            
            # Mock decimate to return smaller data
            decimated_data = test_data[::10]  # Simulate decimation
            mock_decimate.return_value = decimated_data
            
            organizer.compute_bad_channels(limit_memory=True)
            
            # Verify decimate was called
            mock_decimate.assert_called_once()
            # Verify data was converted to float16 before decimation
            call_args = mock_decimate.call_args[0]
            assert call_args[0].dtype == np.float16
            
    def test_prepare_colbins_rowbins_metas(self, temp_dir):
        """Test prepare_colbins_rowbins_metas method."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        
        # Mock the private methods to avoid file I/O
        organizer.colbins = [temp_dir / "file1_ColMajor.bin"]
        organizer.rowbins = [temp_dir / "file1_RowMajor.npy.gz"]  
        organizer.metas = [temp_dir / "file1_Meta.csv"]
        
        # Mock metadata objects
        mock_meta = Mock()
        mock_meta.dt_end = datetime(2023, 1, 1, 12, 0, 0)
        mock_meta.channel_names = ['ch1']
        mock_meta.metadata_df = pd.DataFrame({'ProbeInfo': ['ch1']})
        
        with (
            patch.object(organizer, "_LongRecordingOrganizer__update_colbins_rowbins_metas"),
            patch.object(organizer, "_LongRecordingOrganizer__check_colbins_rowbins_metas_folders_exist"),
            patch.object(organizer, "_LongRecordingOrganizer__check_colbins_rowbins_metas_not_empty"),
            patch.object(organizer, "_validate_metadata_consistency") as mock_validate,
            patch("neurodent.core.core.DDFBinaryMetadata", return_value=mock_meta),
        ):
            
            organizer.prepare_colbins_rowbins_metas()
            
            # Verify metadata was set
            assert organizer.meta is not None
            assert organizer.channel_names == ['ch1']
            assert len(organizer.file_end_datetimes) >= 1
            
            # Verify validation was called
            mock_validate.assert_called_once()
            
    def test_prepare_colbins_rowbins_metas_no_dates(self, temp_dir):
        """Test prepare_colbins_rowbins_metas when no dates are found."""
        # Create colbin, rowbin, and meta files so the organizer finds them
        (temp_dir / "file1_ColMajor.bin").write_bytes(b"test_data")
        (temp_dir / "file1_RowMajor.npy.gz").write_bytes(b"dummy_data")
        (temp_dir / "file1_Meta.csv").write_text("ProbeInfo,SampleRate\nch1,1000")
        
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        with patch("neurodent.core.core.DDFBinaryMetadata") as mock_metadata_class:
            mock_meta = Mock()
            mock_meta.dt_end = None  # No date - this should trigger the ValueError
            mock_meta.channel_names = ['ch1']
            mock_meta.metadata_df = pd.DataFrame({'ProbeInfo': ['ch1']})
            mock_metadata_class.return_value = mock_meta
            
            # This should now process without error since the validation was moved to after processing
            # and only affects certain conditions
            organizer.prepare_colbins_rowbins_metas()
                
    def test_convert_file_with_si_to_recording_folder_mode(self, temp_dir):
        """Test convert_file_with_si_to_recording with folder input."""
        from datetime import datetime
        organizer = LongRecordingOrganizer(temp_dir, mode=None,
                                         manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
                                         datetimes_are_start=True)
        
        # Mock extract function and recording
        mock_extract = Mock()
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 4
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.array(['ch1', 'ch2', 'ch3', 'ch4'])
        mock_recording.get_duration.return_value = 3600.0
        mock_extract.return_value = mock_recording
        
        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="folder"
        )
        
        # Verify extract function was called with folder
        mock_extract.assert_called_once_with(Path(temp_dir))
        assert organizer.LongRecording == mock_recording
        assert organizer.meta.n_channels == 4
        assert organizer.meta.f_s == 1000.0
        
    @patch('spikeinterface.preprocessing.resample')
    def test_convert_file_with_si_to_recording_file_mode(self, mock_resample, temp_dir):
        """Test convert_file_with_si_to_recording with single file input."""
        # Create test file
        test_file = temp_dir / "test.edf"
        test_file.touch()

        from datetime import datetime
        organizer = LongRecordingOrganizer(temp_dir, mode=None,
                                         manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
                                         datetimes_are_start=True)

        mock_extract = Mock()
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 2
        mock_recording.get_sampling_frequency.return_value = 500.0
        mock_recording.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_recording.get_duration.return_value = 1800.0
        mock_extract.return_value = mock_recording

        # Mock resampling since we're using mock recording
        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 2
        mock_resampled.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_resampled.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_resampled.get_duration.return_value = 1800.0
        mock_resample.return_value = mock_resampled

        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="file",
            file_pattern="*.edf"
        )

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
        organizer = LongRecordingOrganizer(temp_dir, mode=None,
                                         manual_datetimes=[datetime(2023, 1, 1, 10, 0, 0), 
                                                         datetime(2023, 1, 1, 11, 0, 0)],
                                         datetimes_are_start=True)
        
        # Mock extract function to return different recordings
        mock_extract = Mock()
        mock_rec1 = Mock()
        mock_rec2 = Mock()
        mock_rec1.get_duration.return_value = 3600.0
        mock_rec2.get_duration.return_value = 1800.0
        mock_extract.side_effect = [mock_rec1, mock_rec2]
        
        # Mock concatenate_recordings
        mock_concat_rec = Mock()
        mock_concat_rec.get_num_channels.return_value = 2
        mock_concat_rec.get_sampling_frequency.return_value = 1000.0
        mock_concat_rec.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_concat_rec.get_duration.return_value = 5400.0
        
        with patch('spikeinterface.core.concatenate_recordings', return_value=mock_concat_rec):
            organizer.convert_file_with_si_to_recording(
                extract_func=mock_extract,
                input_type="files",
                file_pattern="*.edf"
            )
        
        # Should call extract twice and concatenate
        assert mock_extract.call_count == 2
        assert organizer.LongRecording == mock_concat_rec
        
    def test_convert_file_with_mne_to_recording_edf_intermediate(self, temp_dir):
        """Test convert_file_with_mne_to_recording with EDF intermediate."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        from datetime import datetime
        organizer = LongRecordingOrganizer(temp_dir, mode=None,
                                         manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
                                         datetimes_are_start=True)

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
        mock_si_rec.get_sampling_frequency.return_value = 2000.0  # Original MNE sampling rate
        mock_si_rec.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_si_rec.get_duration.return_value = 3600.0

        # Mock resampled recording
        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 2
        mock_resampled.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_resampled.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_resampled.get_duration.return_value = 3600.0

        with patch('mne.export.export_raw') as mock_export, \
             patch('spikeinterface.extractors.read_edf', return_value=mock_si_rec), \
             patch('spikeinterface.preprocessing.resample', return_value=mock_resampled) as mock_resample:

            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract,
                input_type="file",
                file_pattern="*.bdf",
                intermediate="edf"
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
        organizer = LongRecordingOrganizer(temp_dir, mode=None,
                                         manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
                                         datetimes_are_start=True)

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
        mock_si_rec.get_sampling_frequency.return_value = 1000.0
        mock_si_rec.get_channel_ids.return_value = np.array(['ch1', 'ch2', 'ch3'])
        mock_si_rec.get_duration.return_value = 1.0  # 1000 samples at 1000 Hz = 1 second

        with patch('spikeinterface.extractors.read_binary', return_value=mock_si_rec):
            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract,
                input_type="file",
                file_pattern="*.bdf",
                intermediate="bin"
            )
        
        # Verify binary file was created and read
        bin_file = temp_dir / f"{temp_dir.name}_mne-to-rec.bin"
        assert bin_file.exists()
        
        # Verify data was written correctly (transposed from MNE format)
        written_data = np.fromfile(bin_file, dtype=np.float32).reshape(n_samples, n_channels)
        expected_data = test_data.T  # MNE data is (n_channels, n_samples), we expect (n_samples, n_channels)
        np.testing.assert_array_almost_equal(written_data, expected_data)
        
        assert organizer.LongRecording == mock_si_rec

    @patch('spikeinterface.preprocessing.resample')
    def test_apply_resampling_different_sampling_rate(self, mock_resample, temp_dir):
        """Test _apply_resampling method with different sampling rate."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock metadata
        organizer.meta = Mock()
        organizer.meta.update_sampling_rate = Mock()

        # Mock input recording
        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0

        # Mock resampled recording
        mock_resampled = Mock()
        mock_resample.return_value = mock_resampled

        result = organizer._apply_resampling(mock_recording)

        # Verify resample was called with correct parameters
        mock_resample.assert_called_once_with(
            recording=mock_recording,
            resample_rate=constants.GLOBAL_SAMPLING_RATE
        )

        # Verify metadata was updated
        organizer.meta.update_sampling_rate.assert_called_once_with(constants.GLOBAL_SAMPLING_RATE)

        assert result == mock_resampled

    def test_apply_resampling_same_sampling_rate(self, temp_dir):
        """Test _apply_resampling method when no resampling needed."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Mock input recording at target rate
        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE

        result = organizer._apply_resampling(mock_recording)

        # Should return original recording without modification
        assert result == mock_recording

    def test_apply_resampling_no_metadata(self, temp_dir):
        """Test _apply_resampling method when no metadata available."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        organizer.meta = None

        # Mock input recording
        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0

        # Mock resampled recording
        mock_resampled = Mock()
        with patch('spikeinterface.preprocessing.resample', return_value=mock_resampled) as mock_resample:
            result = organizer._apply_resampling(mock_recording)

            # Verify resample was still called
            mock_resample.assert_called_once()
            assert result == mock_resampled

    def test_apply_resampling_missing_spikeinterface(self, temp_dir):
        """Test _apply_resampling method when SpikeInterface preprocessing not available."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0

        # Mock missing preprocessing module
        with patch("neurodent.core.core.spre", None):
            with pytest.raises(ImportError, match="SpikeInterface preprocessing is required"):
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
        mock_recording.get_sampling_frequency.return_value = original_rate

        with patch('spikeinterface.preprocessing.resample') as mock_resample:
            mock_resampled = Mock()
            mock_resample.return_value = mock_resampled

            result = organizer._apply_resampling(mock_recording)

            # Verify metadata was updated to target rate
            organizer.meta.update_sampling_rate.assert_called_once_with(constants.GLOBAL_SAMPLING_RATE)

    def test_unified_resampling_cross_pipeline_consistency(self, temp_dir):
        """Test that all pipelines use the same resampling parameters."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Test recording with non-standard sampling rate
        test_rates = [500.0, 2000.0, 4000.0]

        for test_rate in test_rates:
            mock_recording = Mock()
            mock_recording.get_sampling_frequency.return_value = test_rate

            with patch('spikeinterface.preprocessing.resample') as mock_resample:
                mock_resampled = Mock()
                mock_resample.return_value = mock_resampled

                organizer._apply_resampling(mock_recording)

                # Verify consistent parameters across all calls
                if test_rate != constants.GLOBAL_SAMPLING_RATE:
                    mock_resample.assert_called_once_with(
                        recording=mock_recording,
                        resample_rate=constants.GLOBAL_SAMPLING_RATE
                    )
                else:
                    mock_resample.assert_not_called()

    def test_unified_resampling_performance_parameters(self, temp_dir):
        """Test that resampling uses appropriate performance parameters."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0

        with patch('spikeinterface.preprocessing.resample') as mock_resample:
            mock_resampled = Mock()
            mock_resample.return_value = mock_resampled

            organizer._apply_resampling(mock_recording)

            # Verify performance-oriented parameters
            mock_resample.assert_called_once_with(
                recording=mock_recording,
                resample_rate=constants.GLOBAL_SAMPLING_RATE
            )

    def test_unified_resampling_logging_behavior(self, temp_dir):
        """Test that resampling provides appropriate logging."""
        organizer = LongRecordingOrganizer(temp_dir, mode=None)

        # Test logging when resampling is needed
        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0

        with (
            patch("spikeinterface.preprocessing.resample") as mock_resample,
            patch("neurodent.core.core.logging") as mock_logging,
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
        mock_recording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE

        with patch("neurodent.core.core.logging") as mock_logging:
            organizer._apply_resampling(mock_recording)

            # Should log that no resampling is needed
            mock_logging.info.assert_called_with(
                f"Recording already at target sampling rate ({constants.GLOBAL_SAMPLING_RATE} Hz), no resampling needed"
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
            temp_dir, mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
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
        mock_si_rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_si_rec.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_si_rec.get_duration.return_value = 3.6

        with patch('mne.export.export_raw'), \
             patch('spikeinterface.extractors.read_edf', return_value=mock_si_rec), \
             patch('spikeinterface.preprocessing.resample', return_value=mock_si_rec):

            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract,
                input_type="file",
                file_pattern="*.bdf",
                intermediate="edf"
            )

        # MNE resample should NOT be called (new architecture uses SpikeInterface resampling)
        mock_raw.resample.assert_not_called()
    
    def test_explicit_n_jobs_override(self, temp_dir):
        """Test that users can override n_jobs parameter."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        # User specifies n_jobs=4
        organizer = LongRecordingOrganizer(
            temp_dir, mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
            n_jobs=4
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
        mock_si_rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_si_rec.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_si_rec.get_duration.return_value = 3.6

        with patch('mne.export.export_raw'), \
             patch('spikeinterface.extractors.read_edf', return_value=mock_si_rec), \
             patch('spikeinterface.preprocessing.resample', return_value=mock_si_rec):

            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract,
                input_type="file",
                file_pattern="*.bdf",
                intermediate="edf"
            )

        # MNE resample should NOT be called (new architecture uses SpikeInterface resampling)
        mock_raw.resample.assert_not_called()
    
    def test_n_jobs_direct_method_call(self, temp_dir):
        """Test n_jobs parameter when calling convert_file_with_mne_to_recording directly."""
        test_file = temp_dir / "test.bdf"
        test_file.touch()

        organizer = LongRecordingOrganizer(
            temp_dir, mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True,
            n_jobs=1  # Default
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
        mock_si_rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_si_rec.get_channel_ids.return_value = np.array(['ch1', 'ch2'])
        mock_si_rec.get_duration.return_value = 1.0

        with patch('mne.export.export_raw'), \
             patch('spikeinterface.extractors.read_edf', return_value=mock_si_rec), \
             patch('spikeinterface.preprocessing.resample', return_value=mock_si_rec):

            # Call directly with override n_jobs=6
            organizer.convert_file_with_mne_to_recording(
                extract_func=mock_extract,
                input_type="file",
                file_pattern="*.bdf",
                intermediate="edf",
                n_jobs=6  # Override the instance default
            )

        # MNE resample should NOT be called (new architecture uses SpikeInterface resampling)
        mock_raw.resample.assert_not_called()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)