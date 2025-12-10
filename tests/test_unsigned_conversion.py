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

from neurodent.core.core import LongRecordingOrganizer
from neurodent import constants

class TestUnsignedConversion(unittest.TestCase):
    """Test generic unsigned to signed conversion logic."""

    @patch("neurodent.core.core.spre")
    @patch("neurodent.core.core.constants")
    def test_unsigned_to_signed_conversion_uint16(self, mock_constants, mock_spre):
        """Test that uint16 is detected and converted."""
        # Setup MOcks
        mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
        
        # Mock recording
        mock_recording = Mock()
        # Different rate to trigger resampling logic where the check is
        mock_recording.get_sampling_frequency.return_value = 2000.0 
        mock_recording.get_dtype.return_value = 'uint16'
        
        # Mock organizer
        organizer = LongRecordingOrganizer(base_folder_path=".", mode=None)
        
        # Call method
        organizer._apply_resampling(mock_recording)
        
        # Verify
        mock_spre.unsigned_to_signed.assert_called_once_with(mock_recording)
        
    @patch("neurodent.core.core.spre")
    @patch("neurodent.core.core.constants")
    def test_unsigned_to_signed_conversion_uint32(self, mock_constants, mock_spre):
        """Test that uint32 is detected and converted (generic check)."""
        mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
        
        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0 
        # using numpy dtype object instead of string
        mock_recording.get_dtype.return_value = np.dtype('uint32')
        
        organizer = LongRecordingOrganizer(base_folder_path=".", mode=None)
        
        organizer._apply_resampling(mock_recording)
        
        # Verify
        mock_spre.unsigned_to_signed.assert_called_once_with(mock_recording)

    @patch("neurodent.core.core.spre")
    @patch("neurodent.core.core.constants")
    def test_no_conversion_for_signed_int16(self, mock_constants, mock_spre):
        """Test that int16 is NOT converted."""
        mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
        
        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0 
        mock_recording.get_dtype.return_value = 'int16'
        
        organizer = LongRecordingOrganizer(base_folder_path=".", mode=None)
        
        organizer._apply_resampling(mock_recording)
        
        # Verify NOT called
        mock_spre.unsigned_to_signed.assert_not_called()

    @patch("neurodent.core.core.spre")
    @patch("neurodent.core.core.constants")
    def test_no_conversion_for_float(self, mock_constants, mock_spre):
        """Test that float32 is NOT converted."""
        mock_constants.GLOBAL_SAMPLING_RATE = 1000.0
        
        mock_recording = Mock()
        mock_recording.get_sampling_frequency.return_value = 2000.0 
        mock_recording.get_dtype.return_value = np.dtype('float32')
        
        organizer = LongRecordingOrganizer(base_folder_path=".", mode=None)
        
        organizer._apply_resampling(mock_recording)
        
        # Verify NOT called
        mock_spre.unsigned_to_signed.assert_not_called()
