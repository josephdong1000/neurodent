"""
Tests for generic unsigned to signed conversion in LongRecordingOrganizer.
"""
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pytest

# Try to import spikeinterface, but mocked objects will be used mostly
try:
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre
except ImportError:
    si = None
    spre = None

import neurodent.loading.long_recording_organizer as core_module
from neurodent.loading.long_recording_organizer import LongRecordingOrganizer
from neurodent import constants

class TestUnsignedConversion(unittest.TestCase):
    """Test generic unsigned to signed conversion logic."""

    def test_unsigned_with_scaleable_traces_calls_scale_to_uv(self):
        """Test that uint16 with scaleable traces calls scale_to_uV instead of unsigned_to_signed.

        When a recording has gain_to_uV and offset_to_uV properties (has_scaleable_traces=True),
        scale_to_uV should be applied FIRST to avoid the double-offset bug from unsigned_to_signed.
        """
        with patch.object(core_module, 'spre') as mock_spre, \
             patch.object(core_module, 'constants') as mock_constants:

            mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
            mock_constants.GLOBAL_DTYPE = np.float32

            mock_recording = Mock()
            mock_recording.get_sampling_frequency.return_value = 2000.0
            mock_recording.get_dtype.return_value = 'uint16'
            mock_recording.has_scaleable_traces.return_value = True

            # After scale_to_uV, the recording is float32
            mock_scaled = Mock()
            mock_scaled.get_dtype.return_value = np.float32
            mock_scaled.get_sampling_frequency.return_value = 2000.0
            mock_spre.scale_to_uV.return_value = mock_scaled

            organizer = LongRecordingOrganizer(item=".", mode=None)
            organizer._apply_resampling(mock_recording)

            # scale_to_uV should be called, unsigned_to_signed should NOT
            mock_spre.scale_to_uV.assert_called_once_with(mock_recording)
            mock_spre.unsigned_to_signed.assert_not_called()

    def test_unsigned_without_scaleable_traces_calls_unsigned_to_signed(self):
        """Test that uint16 without scaleable traces still calls unsigned_to_signed."""
        with patch.object(core_module, 'spre') as mock_spre, \
             patch.object(core_module, 'constants') as mock_constants:

            mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
            mock_constants.GLOBAL_DTYPE = np.float32

            mock_recording = Mock()
            mock_recording.get_sampling_frequency.return_value = 2000.0
            mock_recording.get_dtype.return_value = 'uint16'
            mock_recording.has_scaleable_traces.return_value = False

            organizer = LongRecordingOrganizer(item=".", mode=None)
            organizer._apply_resampling(mock_recording)

            # unsigned_to_signed should be called, scale_to_uV should NOT
            mock_spre.unsigned_to_signed.assert_called_once_with(mock_recording)
            mock_spre.scale_to_uV.assert_not_called()

    def test_unsigned_uint32_with_scaleable_traces(self):
        """Test that uint32 with scaleable traces calls scale_to_uV."""
        with patch.object(core_module, 'spre') as mock_spre, \
             patch.object(core_module, 'constants') as mock_constants:

            mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
            mock_constants.GLOBAL_DTYPE = np.float32

            mock_recording = Mock()
            mock_recording.get_sampling_frequency.return_value = 2000.0
            mock_recording.get_dtype.return_value = np.dtype('uint32')
            mock_recording.has_scaleable_traces.return_value = True

            mock_scaled = Mock()
            mock_scaled.get_dtype.return_value = np.float32
            mock_scaled.get_sampling_frequency.return_value = 2000.0
            mock_spre.scale_to_uV.return_value = mock_scaled

            organizer = LongRecordingOrganizer(item=".", mode=None)
            organizer._apply_resampling(mock_recording)

            mock_spre.scale_to_uV.assert_called_once_with(mock_recording)
            mock_spre.unsigned_to_signed.assert_not_called()

    def test_no_conversion_for_signed_int16(self):
        """Test that int16 is NOT converted."""
        with patch.object(core_module, 'spre') as mock_spre, \
             patch.object(core_module, 'constants') as mock_constants:
        
            mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
            
            mock_recording = Mock()
            mock_recording.get_sampling_frequency.return_value = 2000.0 
            mock_recording.get_dtype.return_value = 'int16'
            
            organizer = LongRecordingOrganizer(item=".", mode=None)
            
            organizer._apply_resampling(mock_recording)
            
            # Verify NOT called
            mock_spre.unsigned_to_signed.assert_not_called()

    def test_no_conversion_for_float(self):
        """Test that float32 is NOT converted."""
        with patch.object(core_module, 'spre') as mock_spre, \
             patch.object(core_module, 'constants') as mock_constants:
            
            mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
            
            mock_recording = Mock()
            mock_recording.get_sampling_frequency.return_value = 2000.0 
            mock_recording.get_dtype.return_value = np.dtype('float32')
            
            organizer = LongRecordingOrganizer(item=".", mode=None)
            
            organizer._apply_resampling(mock_recording)
            
            # Verify NOT called
            mock_spre.unsigned_to_signed.assert_not_called()
