"""
Unit tests for manual timestamp functionality in LongRecordingOrganizer.
"""
import pytest
from datetime import datetime, timedelta

from neurodent.core.core import LongRecordingOrganizer


class TestManualTimestamps:
    """Test manual timestamp functionality in LongRecordingOrganizer."""

    def test_manual_datetimes_validation(self):
        """Test validation of manual_datetimes parameter."""
        start_time = datetime(2023, 1, 1, 10, 0, 0)
        times_list = [start_time, start_time + timedelta(hours=1)]
        
        # Test valid single datetime
        organizer = LongRecordingOrganizer(
            "/fake/path", 
            mode=None,
            manual_datetimes=start_time
        )
        assert organizer.manual_datetimes == start_time
        
        # Test valid list of datetimes
        organizer = LongRecordingOrganizer(
            "/fake/path",
            mode=None,
            manual_datetimes=times_list
        )
        assert organizer.manual_datetimes == times_list
        
        # Test invalid type
        with pytest.raises(ValueError, match="must be a datetime object or list"):
            LongRecordingOrganizer(
                "/fake/path", 
                mode=None,
                manual_datetimes="not_a_datetime"
            )

    def test_datetimes_are_start_parameter(self):
        """Test that datetimes_are_start parameter works correctly."""
        start_time = datetime(2023, 1, 1, 10, 0, 0)
        
        # Test default behavior (start times)
        organizer = LongRecordingOrganizer(
            "/fake/path",
            mode=None,
            manual_datetimes=start_time
        )
        assert organizer.datetimes_are_start is True
        
        # Test explicit end times
        organizer = LongRecordingOrganizer(
            "/fake/path",
            mode=None,
            manual_datetimes=start_time,
            datetimes_are_start=False
        )
        assert organizer.datetimes_are_start is False

    def test_compute_manual_file_datetimes_global_start(self):
        """Test computing file datetimes from global start time."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        organizer.manual_datetimes = datetime(2023, 1, 1, 10, 0, 0)
        organizer.datetimes_are_start = True
        
        durations = [3600.0, 1800.0, 900.0]  # 1hr, 30min, 15min
        result = organizer._compute_manual_file_datetimes(3, durations)
        
        expected = [
            datetime(2023, 1, 1, 11, 0, 0),   # 10:00 + 1hr
            datetime(2023, 1, 1, 11, 30, 0),  # 11:00 + 30min
            datetime(2023, 1, 1, 11, 45, 0),  # 11:30 + 15min
        ]
        assert result == expected

    def test_compute_manual_file_datetimes_global_end(self):
        """Test computing file datetimes from global end time."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        organizer.manual_datetimes = datetime(2023, 1, 1, 12, 0, 0)
        organizer.datetimes_are_start = False
        
        durations = [3600.0, 1800.0]  # 1hr, 30min (total 1.5hr)
        result = organizer._compute_manual_file_datetimes(2, durations)
        
        # Global end at 12:00, total duration 1.5hr, so start at 10:30
        expected = [
            datetime(2023, 1, 1, 11, 30, 0),  # 10:30 + 1hr
            datetime(2023, 1, 1, 12, 0, 0),   # 11:30 + 30min
        ]
        assert result == expected

    def test_compute_manual_file_datetimes_list_start_times(self):
        """Test computing file datetimes from list of start times."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        organizer.manual_datetimes = [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 11, 0, 0),
        ]
        organizer.datetimes_are_start = True
        
        durations = [3600.0, 1800.0]  # 1hr, 30min
        result = organizer._compute_manual_file_datetimes(2, durations)
        
        expected = [
            datetime(2023, 1, 1, 11, 0, 0),   # 10:00 + 1hr
            datetime(2023, 1, 1, 11, 30, 0),  # 11:00 + 30min
        ]
        assert result == expected

    def test_compute_manual_file_datetimes_list_end_times(self):
        """Test computing file datetimes from list of end times."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        organizer.manual_datetimes = [
            datetime(2023, 1, 1, 11, 0, 0),
            datetime(2023, 1, 1, 11, 30, 0),
        ]
        organizer.datetimes_are_start = False
        
        durations = [3600.0, 1800.0]  # 1hr, 30min
        result = organizer._compute_manual_file_datetimes(2, durations)
        
        expected = [
            datetime(2023, 1, 1, 11, 0, 0),   # end time directly
            datetime(2023, 1, 1, 11, 30, 0),  # end time directly
        ]
        assert result == expected

    def test_manual_datetimes_length_validation(self):
        """Test that manual_datetimes length must match number of files."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        organizer.manual_datetimes = [
            datetime(2023, 1, 1, 10, 0, 0),
        ]
        organizer.datetimes_are_start = True
        
        durations = [3600.0, 1800.0]  # 2 files, but only 1 time
        
        with pytest.raises(ValueError, match="manual_datetimes length .* must match number of files"):
            organizer._compute_manual_file_datetimes(2, durations)

    def test_validate_file_contiguity_success(self):
        """Test contiguity validation when files are properly contiguous."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        
        file_end_datetimes = [
            datetime(2023, 1, 1, 11, 0, 0),   # File 1 ends at 11:00
            datetime(2023, 1, 1, 11, 30, 0),  # File 2 ends at 11:30
        ]
        durations = [3600.0, 1800.0]  # File 1: 1hr (10:00-11:00), File 2: 30min (11:00-11:30)
        
        # Should not raise an error
        organizer._validate_file_contiguity(file_end_datetimes, durations)

    def test_validate_file_contiguity_with_tolerance(self):
        """Test contiguity validation allows small gaps within tolerance."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        
        file_end_datetimes = [
            datetime(2023, 1, 1, 11, 0, 0),   # File 1 ends at 11:00
            datetime(2023, 1, 1, 11, 30, 1),  # File 2 ends at 11:30:01 (1 second gap)
        ]
        durations = [3600.0, 1800.0]  # File 1: 1hr, File 2: 30min (should start at 11:00:01)
        
        # Should not raise an error (within 1 second tolerance)
        organizer._validate_file_contiguity(file_end_datetimes, durations)

    def test_validate_file_contiguity_warning(self):
        """Test contiguity validation warns when files have large gaps."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        
        file_end_datetimes = [
            datetime(2023, 1, 1, 11, 0, 0),   # File 1 ends at 11:00
            datetime(2023, 1, 1, 11, 35, 0),  # File 2 ends at 11:35 (should start at 11:05)
        ]
        durations = [3600.0, 1800.0]  # File 1: 1hr, File 2: 30min (5 minute gap)
        
        with pytest.warns(UserWarning, match="Files may not be contiguous.*gap of.*between"):
            organizer._validate_file_contiguity(file_end_datetimes, durations)

    def test_validate_file_contiguity_single_file(self):
        """Test contiguity validation with single file (should always pass)."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        
        file_end_datetimes = [datetime(2023, 1, 1, 11, 0, 0)]
        durations = [3600.0]
        
        # Should not raise an error
        organizer._validate_file_contiguity(file_end_datetimes, durations)

    def test_compute_manual_file_datetimes_no_manual_times(self):
        """Test that None is returned when no manual times are specified."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        organizer.manual_datetimes = None
        
        durations = [3600.0, 1800.0]
        result = organizer._compute_manual_file_datetimes(2, durations)
        
        assert result is None

    def test_finalize_file_timestamps_with_manual_start(self):
        """Test that _finalize_file_timestamps uses manual start time."""
        # Create a minimal organizer with mode=None to avoid file system dependencies
        organizer = LongRecordingOrganizer(
            "/fake/path", 
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )
        
        # Manually set file_durations as would happen in the real flow
        organizer.file_durations = [3600.0, 1800.0]
        
        # Call the method we want to test
        organizer.finalize_file_timestamps()
        
        # Verify the correct end times were computed
        expected = [
            datetime(2023, 1, 1, 11, 0, 0),   # 10:00 + 1hr
            datetime(2023, 1, 1, 11, 30, 0),  # 11:00 + 30min
        ]
        assert organizer.file_end_datetimes == expected

    def test_contiguity_validation_integrated_with_manual_times(self):
        """Test that contiguity validation warns when using manual times with gaps."""
        organizer = LongRecordingOrganizer("/fake/path", mode=None)
        organizer.manual_datetimes = [
            datetime(2023, 1, 1, 10, 0, 0),   # File 1 starts at 10:00
            datetime(2023, 1, 1, 11, 30, 0),  # File 2 starts at 11:30 (30 min gap!)
        ]
        organizer.datetimes_are_start = True
        
        durations = [3600.0, 1800.0]  # File 1: 1hr (10:00-11:00), File 2: 30min (11:30-12:00)
        
        # Should warn about contiguity (30 minute gap)
        with pytest.warns(UserWarning, match="Files may not be contiguous"):
            result = organizer._compute_manual_file_datetimes(2, durations)
            # Should still return the computed times
            expected = [
                datetime(2023, 1, 1, 11, 0, 0),   # 10:00 + 1hr
                datetime(2023, 1, 1, 12, 0, 0),   # 11:30 + 30min
            ]
            assert result == expected