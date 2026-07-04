
import pytest
from neurodent.loading import LongRecordingOrganizer
import datetime
import numpy as np
import spikeinterface.core as si

def test_split_creates_mismatch_for_multifile_recording():
    """
    Reproduces the bug where lro.split() creates a child LRO with mismatched
    file_durations (len 1) and file_end_datetimes (len N) if the parent had multiple files.
    """
    # 1. Mock a parent LRO with 2 files (segments)
    lro = LongRecordingOrganizer(item=None, recording=None)
    
    # Mock the LongRecording to have 2 segments
    class MockRecording:
        def get_channel_ids(self):
            return ["Ch1", "Ch2"]
        def get_num_channels(self):
            return 2
        def get_sampling_frequency(self):
            return 1000.0
        def get_total_duration(self):
            return 20.0
        def get_num_segments(self):
            return 2
        def get_duration(self, segment_index=None):
            return 10.0 # Each segment 10s
        def get_dtype(self):
            return np.float32
        def select_channels(self, channel_ids):
            return self # specific logic not needed for this bug, just needs to return a recording
        def rename_channels(self, new_channel_ids):
            return self

    lro.LongRecording = MockRecording()
    lro.channel_names = ["Ch1", "Ch2"]
    
    # Parent has valid metadata for 2 files
    lro.file_durations = [10.0, 10.0]
    lro.file_end_datetimes = [
        datetime.datetime(2023, 1, 1, 10, 0, 10),
        datetime.datetime(2023, 1, 1, 10, 0, 20)
    ]
    lro.labels = {}
    
    # 2. Perform Split
    # This should trigger the bug in the child LRO
    splits = lro.split(groups={"Group1": ["Ch1"]})
    child = splits["Group1"]
    
    # 3. Verify Child State
    print(f"Child file_durations len: {len(child.file_durations)}")
    print(f"Child file_end_datetimes len: {len(child.file_end_datetimes)}")
    
    # Expectation: 
    # file_durations should be [10.0, 10.0] (len 2)
    # file_end_datetimes should be [dt1, dt2] (len 2)
    
    # Actual Bug: 
    # file_durations is [20.0] (len 1)
    # file_end_datetimes is [dt1, dt2] (len 2)
    
    assert len(child.file_durations) == len(child.file_end_datetimes), \
        f"Mismatch! Durations: {len(child.file_durations)}, Times: {len(child.file_end_datetimes)}"

def test_split_with_real_multisegment_recording():
    """
    Test split() with a real SpikeInterface recording (multi-segment) to verify 
    that the fix works with actual recordings, not just mocks.
    """
    # Generate a real multi-segment recording
    rec = si.generate_recording(durations=[1.0, 1.0], num_channels=4, sampling_frequency=1000.0)
    
    # Create LRO from the recording
    lro = LongRecordingOrganizer(item=None, recording=rec)
    
    # Manually set metadata to match multifile scenario
    lro.file_durations = [1.0, 1.0]
    lro.cumulative_file_durations = [1.0, 2.0]
    lro.file_end_datetimes = [
        datetime.datetime(2023, 1, 1, 10, 0, 1),
        datetime.datetime(2023, 1, 1, 10, 0, 2)
    ]
    lro.channel_names = ["Ch0", "Ch1", "Ch2", "Ch3"]
    
    # Perform split
    splits = lro.split(groups={"GroupA": ["Ch0", "Ch1"], "GroupB": ["Ch2", "Ch3"]})
    
    # Verify Child A
    child_a = splits["GroupA"]
    assert len(child_a.file_durations) == 2
    assert len(child_a.file_end_datetimes) == 2
    assert child_a.file_durations == [1.0, 1.0]
    assert child_a.LongRecording.get_num_channels() == 2
    
    # Verify Child B
    child_b = splits["GroupB"]
    assert len(child_b.file_durations) == 2
    assert len(child_b.file_end_datetimes) == 2
