
import pytest
from neurodent.core import LongRecordingOrganizer
import datetime
import numpy as np
import spikeinterface.core as si
from unittest.mock import MagicMock

def test_split_inherits_all_critical_metadata():
    """
    Exhaustive test to ensure split() inherits ALL critical metadata required
    for downstream processing, not just timestamps.
    """
    # 1. Setup Parent LRO with rich metadata
    lro = LongRecordingOrganizer(base_folder_path=None, recording=None)
    
    # Mock Recording
    # Mock Recording (Real SI object for robustness)
    full_traces = np.zeros((30000, 2), dtype=np.float32) # 30s @ 1000Hz
    # Use real SI recording to support select_channels, get_dtype, etc.
    rec = si.NumpyRecording(full_traces, sampling_frequency=1000.0, channel_ids=["Ch1", "Ch2"])
    
    lro.LongRecording = rec
    lro.channel_names = ["Ch1", "Ch2"]
    
    # Metadata that SHOULD be inherited
    # 3 files/segments
    lro.file_durations = [10.0, 10.0, 10.0]
    lro.cumulative_file_durations = [10.0, 20.0, 30.0]
    lro.file_end_datetimes = [
        datetime.datetime(2023, 1, 1, 10, 0, 10),
        datetime.datetime(2023, 1, 1, 10, 0, 20),
        datetime.datetime(2023, 1, 1, 10, 0, 30)
    ]
    lro.manual_datetimes = [datetime.datetime(2023, 1, 1, 10, 0, 0)] # e.g. start time provided manually
    lro.datetimes_are_start = True
    lro.n_jobs = 4
    lro.bad_channel_names = ["Ch1", "Ch3"] 
    lro.n_truncate = 5
    lro.truncate = True
    
    # Rich metadata to test deepcopy
    # We need to set lro.meta to something that has attributes we can check
    class MockMeta:
        pass
    lro.meta = MockMeta()
    lro.meta.n_channels = 2
    lro.meta.channel_names = ["Ch1", "Ch2"]
    lro.meta.mult_to_uV = 0.195 # Specific value to test validation
    
    # Labels (should NOT be inherited identically, but copied)
    lro.labels = {"Session": "JointSession1"}

    # 2. Perform Split
    splits = lro.split(groups={"AnimalA": ["Ch1"], "AnimalB": ["Ch2"]})
    child = splits["AnimalA"]
    
    # 3. Assertions
    
    # A. Timestamps and Durations ( The Bug Fix )
    assert child.file_durations == lro.file_durations
    assert child.cumulative_file_durations == lro.cumulative_file_durations
    assert child.file_end_datetimes == lro.file_end_datetimes
    
    # B. Manual timing usage
    assert child.manual_datetimes == lro.manual_datetimes
    assert child.datetimes_are_start == lro.datetimes_are_start
    assert child.n_jobs == lro.n_jobs
    
    # C. Truncate
    assert child.n_truncate == lro.n_truncate
    assert child.truncate == lro.truncate
    
    # D. Metadata (Deep Copy)
    assert child.meta.mult_to_uV == lro.meta.mult_to_uV
    assert child.meta.n_channels == 1 # Updated count
    assert child.meta.channel_names == ["Ch1"] # Updated list
    assert child.meta is not lro.meta # Must be a copy
    
    # E. Bad Channels (Filtered)
    # Child A has Ch1. Parent had Ch1 and Ch3 as bad.
    # Ch3 is not in Child A (implied by split logic? No, check setup). 
    # splits is by channel mapping. AnimalA gets Ch1.
    # So bad_channel_names should contain Ch1.
    assert "Ch1" in child.bad_channel_names
    assert "Ch3" not in child.bad_channel_names # Should be filtered out or just not present
    assert len(child.bad_channel_names) == 1 
    
    # D. Labels
    assert child.labels == lro.labels
    assert child.labels is not lro.labels # Must be a copy
    
    # E. State Consistency
    assert len(child.file_durations) == len(child.file_end_datetimes)
    assert child._is_in_memory is True

def test_split_handles_empty_metadata_gracefully():
    """
    Ensure split() doesn't crash if parent LRO has partial metadata 
    (e.g. valid duration but no timestamps).
    """
    # Create a real numpy recording (10 samples, 1 channel)
    # This guarantees compatibility with the new _apply_resampling logic (get_dtype, etc.)
    full_traces = np.zeros((10, 1), dtype=np.float32)
    rec = si.NumpyRecording(full_traces, sampling_frequency=1000.0)
    
    # Create LRO from this recording
    lro = LongRecordingOrganizer(base_folder_path=None, recording=rec)
    
    # Manually populate metadata being tested
    lro.channel_names = ["Ch1"]
    lro.file_durations = [10.0]
    lro.cumulative_file_durations = [10.0]
    lro.truncate = 100
    lro.n_truncate = 5
    lro.n_jobs = 4
    lro.bad_channel_names = ["Ch1"]
    
    # Mock metadata object
    lro.meta = MagicMock()
    lro.meta.n_channels = 1
    lro.meta.channel_names = ["Ch1"]
    lro.meta.mult_to_uV = 0.5
    lro.meta.f_s = 1000.0  
    # No file_end_datetimes
    
    splits = lro.split(groups={"A": ["Ch1"]})
    child = splits["A"]
    
    # Should inherit durations
    assert child.file_durations == [10.0]
    # Should NOT have timestamps
    assert not hasattr(child, "file_end_datetimes")

