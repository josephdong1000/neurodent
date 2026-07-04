
import numpy as np
import pytest
from unittest.mock import MagicMock
from neurodent.loading import LongRecordingOrganizer
import neurodent.constants as constants
try:
    import spikeinterface.core as si
except ImportError:
    si = None

@pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
def test_lro_enforces_global_dtype():
    """
    Test that initializing an LRO with a non-GLOBAL_DTYPE recording 
    automatically converts it to GLOBAL_DTYPE.
    """
    # 1. Create a dummy recording with int16 dtype
    # We use a generator to simulate a recording
    class MockRecording:
        def get_num_segments(self): return 1
        def get_num_channels(self): return 1
        def get_channel_ids(self): return ["0"]
        def get_sampling_frequency(self): return 1000.0
        def get_total_duration(self): return 1.0
        def get_dtype(self): return np.int16
        def get_traces(self, **kwargs): 
            return np.zeros((1000, 1), dtype=np.int16)
        
        # Method needed for validation but not testing logic directly
        def get_times(self): return np.arange(1000) / 1000.0
        
    mock_rec = MockRecording()
    
    # 2. Initialize LRO with this mock
    # NOTE: LRO expects a real SI object usually, but since we mock _apply_resampling interactions,
    # we need to ensure the mock behaves enough like an SI object OR we rely on spre working.
    # Actually, `si.BaseRecording` is expected. Let's use a real numpy recording for reliability.
    
    # Create a real numpy recording
    traces = np.zeros((1000, 2), dtype=np.int16)
    rec_int16 = si.NumpyRecording(traces, sampling_frequency=1000.0)
    
    assert rec_int16.get_dtype() == np.int16
    assert rec_int16.get_dtype() != constants.GLOBAL_DTYPE
    
    # 3. Create LRO
    lro = LongRecordingOrganizer(item=None, 
        recording=rec_int16,
        mode=None # In-memory mode
    )
    
    # 4. Verify LRO recording is now float32 (GLOBAL_DTYPE)
    actual_dtype = lro.LongRecording.get_dtype()
    expected_dtype = constants.GLOBAL_DTYPE
    
    assert actual_dtype == expected_dtype, f"LRO failed to convert dtype. Expected {expected_dtype}, got {actual_dtype}"
    
    # Verify it's still a recording object
    assert isinstance(lro.LongRecording, si.BaseRecording)

@pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
def test_lro_handles_unsigned_conversion():
    """
    Test that unsigned types (uint16) are converted to signed/float correctly.
    """
    traces = np.zeros((1000, 2), dtype=np.uint16)
    rec_uint16 = si.NumpyRecording(traces, sampling_frequency=1000.0)
    
    lro = LongRecordingOrganizer(item=None, 
        recording=rec_uint16,
        mode=None
    )
    
    assert lro.LongRecording.get_dtype() == constants.GLOBAL_DTYPE
