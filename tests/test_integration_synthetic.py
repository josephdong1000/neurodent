"""
Integration tests using synthetic data from spikeinterface.
"""
import datetime
import numpy as np
import spikeinterface.core as si
from neurodent.core import LongRecordingOrganizer
from neurodent import constants


def test_split_inheritance_synthetic_data():
    """
    Test that split preserves critical metadata and dtype is enforced.
    
    Previously: lro.split() could inadvertently drop file_end_datetimes
    while retaining file_durations, creating an inconsistent state.
    
    This test verifies:
    1. Original LRO enforces GLOBAL_DTYPE
    2. split() preserves both file_end_datetimes and file_durations
    3. Child LRO also enforces GLOBAL_DTYPE
    """
    # 1. Create synthetic multi-file scenario using direct traces
    # LRO should convert int16 -> float32 via _apply_resampling
    traces1 = np.random.randn(2000, 2).astype(np.int16)  # 2s @ 1000Hz
    traces2 = np.random.randn(3000, 2).astype(np.int16)  # 3s @ 1000Hz
    
    rec1 = si.NumpyRecording(traces1, sampling_frequency=1000.0)
    rec2 = si.NumpyRecording(traces2, sampling_frequency=1000.0)
    
    combined_rec = si.concatenate_recordings([rec1, rec2])
    
    lro = LongRecordingOrganizer(base_folder_path=None, recording=combined_rec)
    
    # 2. Manually set metadata matching the 2-file scenario
    lro.file_durations = [2.0, 3.0]
    lro.cumulative_file_durations = [2.0, 5.0]
    
    # Set start times
    t0 = datetime.datetime(2023, 1, 1, 10, 0, 0)
    lro.file_end_datetimes = [
        t0 + datetime.timedelta(seconds=2),
        t0 + datetime.timedelta(seconds=5)
    ]
    
    # 3. Verify original LRO dtype (should be float32 from _apply_resampling)
    assert lro.LongRecording.get_dtype() == constants.GLOBAL_DTYPE
    
    # 4. Test split preserves metadata
    lro_child = lro.split(start_frame_idx=0, end_frame_idx=1000)
    
    # Verify child has consistent metadata
    assert lro_child.file_end_datetimes is not None
    assert len(lro_child.file_end_datetimes) == len(lro_child.file_durations)
    
    # Verify child also enforces GLOBAL_DTYPE
    assert lro_child.LongRecording.get_dtype() == constants.GLOBAL_DTYPE


def test_analysis_pipeline_integration():
    """
    Verify that the LRO produced by this process works with the Analysis pipeline
    (which historically had that dtype crash).
    
    NOTE: Uses single-segment recording since the focus is dtype enforcement,
    not multi-segment handling. Multi-segment tests are covered elsewhere.
    """
    from neurodent.core import LongRecordingAnalyzer
    
    # 1. Create single-segment synthetic recording with int16 dtype
    # LRO should convert this to float32 via _apply_resampling
    traces = np.random.randn(5000, 2).astype(np.int16)  # 5s @ 1000Hz
    rec = si.NumpyRecording(traces, sampling_frequency=1000.0)
    
    lro = LongRecordingOrganizer(base_folder_path=None, recording=rec)
    
    # 2. Add metadata for analysis
    lro.file_durations = [5.0]
    lro.cumulative_file_durations = [5.0]
    # Set start times
    t0 = datetime.datetime(2023, 1, 1, 10, 0, 0)
    lro.file_end_datetimes = [t0 + datetime.timedelta(seconds=5)]
    
    # 3. Create Analyzer
    # This calls get_fragment_rec -> checks assertions -> applies notch filter
    ana = LongRecordingAnalyzer(lro, fragment_len_s=0.5, apply_notch_filter=True)
    
    assert ana.n_fragments > 0
    
    # 4. Trigger actual computation (which runs notch filter on data)
    # If dtype is wrong (unsigned), this would crash/assert inside get_fragment_rec
    frag0 = ana.get_fragment_rec(0)
    assert frag0.get_dtype() == constants.GLOBAL_DTYPE
    
    # Also verify get_fragment_np works
    data_np = ana.get_fragment_np(0)
    assert data_np.dtype == constants.GLOBAL_DTYPE


def test_lro_merge_and_analysis_pipeline():
    """
    Test merging 2 LROs together and then reading values from the resultant
    LongRecordingAnalyzer.
    
    This tests the full pipeline: create 2 LROs with synthetic data, merge them,
    then verify the merged LRO works correctly with the analysis pipeline.
    """
    from neurodent.core import LongRecordingAnalyzer
    
    # 1. Create first LRO with synthetic data
    traces1 = np.random.randn(3000, 2).astype(np.int16)  # 3s @ 1000Hz
    rec1 = si.NumpyRecording(traces1, sampling_frequency=1000.0)
    lro1 = LongRecordingOrganizer(base_folder_path=None, recording=rec1)
    
    # Set metadata for LRO1
    t0 = datetime.datetime(2023, 1, 1, 10, 0, 0)
    lro1.file_durations = [3.0]
    lro1.cumulative_file_durations = [3.0]
    lro1.file_end_datetimes = [t0 + datetime.timedelta(seconds=3)]
    
    # 2. Create second LRO with synthetic data
    traces2 = np.random.randn(2000, 2).astype(np.int16)  # 2s @ 1000Hz
    rec2 = si.NumpyRecording(traces2, sampling_frequency=1000.0)
    lro2 = LongRecordingOrganizer(base_folder_path=None, recording=rec2)
    
    # Set metadata for LRO2 (continues from LRO1)
    lro2.file_durations = [2.0]
    lro2.cumulative_file_durations = [2.0]
    lro2.file_end_datetimes = [t0 + datetime.timedelta(seconds=5)]
    
    # 3. Merge the two LROs (modifies lro1 in-place)
    lro1.merge(lro2)
    lro_merged = lro1  # lro1 now contains the merged recording
    
    # 4. Verify merged LRO has correct metadata
    # The merge appends the second LRO's data to the first
    assert len(lro_merged.file_durations) == 2
    assert lro_merged.file_durations == [3.0, 2.0]
    assert lro_merged.LongRecording.get_dtype() == constants.GLOBAL_DTYPE
    # Total duration should be 5.0s (3.0 + 2.0)
    total_duration = lro_merged.LongRecording.get_duration()
    assert abs(total_duration - 5.0) < 0.01  # Allow small floating point error
    
    # 5. Create Analyzer from merged LRO
    ana = LongRecordingAnalyzer(lro_merged, fragment_len_s=0.5, apply_notch_filter=True)
    
    # 6. Verify analyzer works correctly
    assert ana.n_fragments > 0
    assert ana.n_fragments == int(np.ceil(5.0 / 0.5))  # 10 fragments for 5s recording
    
    # 7. Read values from analyzer to verify it works end-to-end
    frag0 = ana.get_fragment_rec(0)
    assert frag0.get_dtype() == constants.GLOBAL_DTYPE
    
    # Test reading from middle fragment (crosses LRO boundary at 3s)
    frag6 = ana.get_fragment_rec(6)  # Fragment 6 is at 3.0s, right at the boundary
    assert frag6.get_dtype() == constants.GLOBAL_DTYPE
    
    # Test reading numpy data
    data_np_first = ana.get_fragment_np(0)
    assert data_np_first.dtype == constants.GLOBAL_DTYPE
    assert data_np_first.shape[0] == 500  # 0.5s * 1000Hz
    
    data_np_last = ana.get_fragment_np(ana.n_fragments - 1)
    assert data_np_last.dtype == constants.GLOBAL_DTYPE
