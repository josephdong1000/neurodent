"""
Integration tests for manual timestamp functionality and SI/MNE mode requirements.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from pythoneeg.core.core import LongRecordingOrganizer


class TestManualTimestampsIntegration:
    """Test integration of manual timestamps with different modes."""

    def test_si_mode_requires_timestamps_early_validation(self):
        """Test that SI mode fails early if no manual timestamps provided."""
        with patch('glob.glob', return_value=['/fake/file.edf']):
            with pytest.raises(ValueError, match="manual_datetimes must be provided for si mode"):
                LongRecordingOrganizer(
                    "/fake/path",
                    mode="si",
                    extract_func=Mock(),
                    input_type="file",
                    file_pattern="*.edf"
                )

    def test_mne_mode_requires_timestamps_early_validation(self):
        """Test that MNE mode fails early if no manual timestamps provided."""
        with patch.object(LongRecordingOrganizer, '_validate_timestamps_for_mode') as mock_validate:
            mock_validate.side_effect = ValueError("manual_datetimes must be provided for mne mode")
            
            with pytest.raises(ValueError, match="manual_datetimes must be provided for mne mode"):
                organizer = LongRecordingOrganizer("/fake/path", mode=None)
                organizer.convert_file_with_mne_to_recording(
                    extract_func=Mock(),
                    input_type="file",
                    file_pattern="*.edf"
                )

    def test_si_mode_files_validates_count_early(self):
        """Test that SI mode with files validates timestamp count early."""
        from pathlib import Path
        mock_files = [Path('/fake/file1.edf'), Path('/fake/file2.edf')]
        
        with patch.object(Path, 'glob', return_value=mock_files):
            with pytest.raises(ValueError, match="manual_datetimes length.*must match number of input files"):
                LongRecordingOrganizer(
                    "/fake/path",
                    mode="si", 
                    extract_func=Mock(),
                    input_type="files",
                    file_pattern="*.edf",
                    manual_datetimes=[datetime(2023, 1, 1, 10, 0, 0)]  # Only 1 time for 2 files
                )

    def test_bin_mode_allows_no_timestamps_initially(self):
        """Test that bin mode doesn't require timestamps until after processing."""
        # This should not raise an error during initialization
        organizer = LongRecordingOrganizer(
            "/fake/path",
            mode=None  # Don't trigger processing
        )
        assert organizer.manual_datetimes is None

    def test_early_validation_prevents_slow_operations(self):
        """Test that early validation prevents slow extract_func calls."""
        from pathlib import Path
        extract_func = Mock()
        mock_files = [Path('/fake/file1.edf'), Path('/fake/file2.edf')]
        
        with patch.object(Path, 'glob', return_value=mock_files):
            with pytest.raises(ValueError, match="manual_datetimes must be provided"):
                LongRecordingOrganizer(
                    "/fake/path",
                    mode="si",
                    extract_func=extract_func,
                    input_type="files",
                    file_pattern="*.edf"
                )
        
        # extract_func should never have been called
        extract_func.assert_not_called()

    def test_si_mode_single_file_with_datetime(self):
        """Test that SI mode works with single datetime for single file."""
        with patch('glob.glob', return_value=['/fake/file.edf']):
            with patch.object(LongRecordingOrganizer, '_validate_timestamps_for_mode'):
                # Should not raise an error with proper timestamp
                organizer = LongRecordingOrganizer(
                    "/fake/path",
                    mode=None,  # Don't actually process
                    manual_datetimes=datetime(2023, 1, 1, 10, 0, 0)
                )
                
                # This should pass validation
                organizer._validate_timestamps_for_mode("si", 1)

    def test_mne_mode_multiple_files_with_list(self):
        """Test that MNE mode works with list of datetimes for multiple files."""
        mock_files = ['/fake/file1.edf', '/fake/file2.edf']
        datetimes = [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 11, 0, 0)]
        
        organizer = LongRecordingOrganizer(
            "/fake/path",
            mode=None,  # Don't actually process
            manual_datetimes=datetimes
        )
        
        # This should pass validation
        organizer._validate_timestamps_for_mode("mne", 2)