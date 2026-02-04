import datetime
from datetime import timedelta
import pytest
from neurodent.core import LongRecordingOrganizer
from neurodent.core.utils import TimestampMapper

def test_reproduce_metadata_mismatch_bug():
    """
    Reproduces the bug where merging an LRO with missing timestamps into one with timestamps
    causes file_durations to grow while file_end_datetimes does not, leading to a length mismatch.
    """
    # LRO 1: Complete healthy metadata
    duration_1 = 600.0
    end_time_1 = datetime.datetime(2023, 1, 1, 10, 0, 0)
    
    lro1 = LongRecordingOrganizer(base_folder_path=None, recording=None)
    lro1.LongRecording = type('MockRecording', (), {})() # Mock object
    lro1.meta = type('MockMeta', (), {'dt_end': end_time_1})()
    lro1.file_end_datetimes = [end_time_1]
    lro1.file_durations = [duration_1]
    lro1.labels = {}

    # LRO 2: Missing timestamps (e.g. parsing failed), but has durations
    duration_2 = 300.0
    
    lro2 = LongRecordingOrganizer(base_folder_path=None, recording=None)
    lro2.LongRecording = type('MockRecording', (), {})()
    lro2.meta = type('MockMeta', (), {'dt_end': None})()
    
    # Crucial part: LRO2 has durations but NO file_end_datetimes
    lro2.file_end_datetimes = None  # Or empty list []
    lro2.file_durations = [duration_2]
    lro2.labels = {}

    # Initial state check
    assert len(lro1.file_end_datetimes) == 1
    assert len(lro1.file_durations) == 1
    
    # Perform the merge logic
    # We call the internal method directly as that's where the logic resides
    # EXPECTED BEHAVIOR AFTER FIX: 
    # The merge should fail immediately with a descriptive error because timestamps are missing
    with pytest.raises(ValueError, match="Merge failed: .* has durations but missing 'file_end_datetimes'"):
        lro1._update_metadata_after_merge(lro2)

    # Verify state was NOT corrupted (lengths remain 1)
    assert len(lro1.file_end_datetimes) == 1
    assert len(lro1.file_durations) == 1
