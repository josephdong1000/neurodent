import datetime
from datetime import timedelta
import pytest
from neurodent.core import LongRecordingOrganizer
from neurodent import core

def test_lro_merge_preserves_timestamps():
    """
    Test that merging two LROs preserves their individual file timestamps.
    This is critical for detecting gaps in recording sessions correctly.
    """
    # Create distinct times for two recordings
    start_time_1 = datetime.datetime(2023, 1, 1, 10, 0, 0)
    duration_1 = 600.0  # 10 minutes
    end_time_1 = start_time_1 + timedelta(seconds=duration_1)
    
    # Second recording starts 1 minute after the first one ends (gap)
    gap_duration = 60.0
    start_time_2 = end_time_1 + timedelta(seconds=gap_duration)
    duration_2 = 600.0
    end_time_2 = start_time_2 + timedelta(seconds=duration_2)
    
    # Mocking LRO 1
    lro1 = LongRecordingOrganizer(item=None, recording=None)
    lro1.LongRecording = type('MockRecording', (), {
        'get_num_channels': lambda: 1, 
        'get_sampling_frequency': lambda: 1000,
        'get_duration': lambda: duration_1
    })()
    lro1.meta = type('MockMeta', (), {'n_channels': 1, 'f_s': 1000, 'dt_end': end_time_1})()
    lro1.file_end_datetimes = [end_time_1]
    lro1.file_durations = [duration_1]
    lro1.channel_names = ["Ch1"]
    lro1.labels = {}

    # Mocking LRO 2
    lro2 = LongRecordingOrganizer(item=None, recording=None)
    lro2.LongRecording = type('MockRecording', (), {
        'get_num_channels': lambda: 1, 
        'get_sampling_frequency': lambda: 1000,
        'get_duration': lambda: duration_2
    })()
    lro2.meta = type('MockMeta', (), {'n_channels': 1, 'f_s': 1000, 'dt_end': end_time_2})()
    lro2.file_end_datetimes = [end_time_2]
    lro2.file_durations = [duration_2]
    lro2.channel_names = ["Ch1"]
    lro2.labels = {}

    # Perform Merge Logic (direct call to verify metadata updates)
    lro1._update_metadata_after_merge(lro2)
    
    # Assertions
    assert len(lro1.file_end_datetimes) == 2, "Should have 2 file end timestamps"
    assert lro1.file_end_datetimes[0] == end_time_1
    assert lro1.file_end_datetimes[1] == end_time_2
    
    assert len(lro1.file_durations) == 2, "Should have 2 file durations"
    assert lro1.file_durations[0] == duration_1
    assert lro1.file_durations[1] == duration_2
    
    # Check gap detection works correctly with the updated lists
    with pytest.warns(UserWarning, match="Files may not be contiguous"):
        lro1._validate_file_contiguity(lro1.file_end_datetimes, lro1.file_durations)

def test_lro_merge_timestamps_missing_attributes():
    """Test resilience when file_durations or file_end_datetimes are missing."""
    lro1 = LongRecordingOrganizer(item=None, recording=None)
    lro1.meta = type('MockMeta', (), {'n_channels': 1, 'dt_end': None})()
    lro1.labels = {}
    
    lro2 = LongRecordingOrganizer(item=None, recording=None)
    lro2.meta = type('MockMeta', (), {'n_channels': 1, 'dt_end': None})()
    lro2.labels = {}
    
    # Neither has attributes -> no crash
    lro1._update_metadata_after_merge(lro2)
    
    # One has it
    lro1.file_durations = [10.0]
    lro1._update_metadata_after_merge(lro2)
    assert len(lro1.file_durations) == 1

def test_lro_merge_overlap_warning():
    """Test that merging overlapping files triggers a warning in check_file_gaps."""
    # File 1 ends at 10:10
    end_time_1 = datetime.datetime(2023, 1, 1, 10, 10, 0)
    duration_1 = 600.0
    
    # File 2 starts at 10:09 (1 minute overlap)
    # Start = End - Duration
    # We want Start2 = 10:09. If Duration2 = 600, End2 = 10:19.
    duration_2 = 600.0
    start_time_2 = datetime.datetime(2023, 1, 1, 10, 9, 0)
    end_time_2 = start_time_2 + timedelta(seconds=duration_2)
    
    lro1 = LongRecordingOrganizer(item=None, recording=None)
    lro1.meta = type('MockMeta', (), {'n_channels': 1})()
    lro1.labels = {}
    lro1.file_end_datetimes = [end_time_1]
    lro1.file_durations = [duration_1]

    lro2 = LongRecordingOrganizer(item=None, recording=None)
    lro2.meta = type('MockMeta', (), {'n_channels': 1})()
    lro2.labels = {}
    lro2.file_end_datetimes = [end_time_2]
    lro2.file_durations = [duration_2]
    
    # Merge
    lro1._update_metadata_after_merge(lro2)
    
    # Verify overlap warning
    with pytest.warns(UserWarning, match="Files may overlap"):
        lro1._validate_file_contiguity(lro1.file_end_datetimes, lro1.file_durations)
