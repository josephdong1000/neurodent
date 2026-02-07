
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from neurodent.core import LongRecordingOrganizer
import pytest

class TestLROSplit(unittest.TestCase):
    def setUp(self):
        # Patch dependencies
        self.si_patcher = patch('neurodent.core.core.si')
        self.mock_si = self.si_patcher.start()
        
        self.spre_patcher = patch('neurodent.core.core.spre')
        self.mock_spre = self.spre_patcher.start()
        # Make spre functions allow pass-through
        self.mock_spre.astype.side_effect = lambda rec, **kwargs: rec
        self.mock_spre.unsigned_to_signed.side_effect = lambda rec: rec
        self.mock_spre.resample.side_effect = lambda recording, **kwargs: recording
        
        # Patch LRO.__init__ to avoid real initialization logic but set attributes
        self.init_patcher = patch.object(
            LongRecordingOrganizer, 
            '__init__', 
            side_effect=self._init_side_effect, 
            return_value=None, 
            autospec=True
        )
        self.init_patcher.start()

        # Mock LongRecording methods
        self.mock_recording = MagicMock()
        self.mock_recording.get_channel_ids.return_value = ["ch1", "ch2"]
        self.mock_recording.select_channels.return_value = self.mock_recording 
        self.mock_recording.get_sampling_frequency.return_value = 1000.0
        self.mock_recording.get_dtype.return_value = "float32"
        self.mock_recording.get_total_duration.return_value = 10.0
        self.mock_recording.get_num_channels.return_value = 2
        
    def tearDown(self):
        self.si_patcher.stop()
        self.spre_patcher.stop()
        self.init_patcher.stop()

    def _init_side_effect(self, instance, *args, **kwargs):
        """Mock behavior for LRO.__init__ to set basic attributes."""
        instance.labels = kwargs.get('labels', {})
        instance.base_folder_path = kwargs.get('base_folder_path', None)
        # Default attributes usually set by init or needed by split/logic
        instance.manual_datetimes = None
        instance.datetimes_are_start = True
        instance.n_jobs = 1
        instance.n_truncate = None
        instance.truncate = False
        instance.file_end_datetimes = [1, 2] # Dummy
        instance.file_durations = [1.0]
        instance.cumulative_file_durations = [1.0]
        instance.bad_channel_names = []
        instance.temppaths = []
        instance.meta = None

    def test_split_inherits_base_folder_path(self):
        """Verify that split() children inherit parent's base_folder_path."""
        base_path = Path("/tmp/test/recording")
        
        # Instantiate parent (uses _init_side_effect)
        lro = LongRecordingOrganizer(base_folder_path=base_path)
        
        # Manually set additional state needed for split logic
        lro.LongRecording = self.mock_recording
        lro.channel_names = ["ch1", "ch2"]
        # lro.base_folder_path is already set by side_effect
        
        # Perform split
        splits = lro.split(groups={"Group1": ["ch1"]})
        
        # Assertions
        self.assertIn("Group1", splits)
        child = splits["Group1"]
        # Split children now inherit base_folder_path from parent
        self.assertEqual(child.base_folder_path, base_path)
        
    def test_split_child_properties(self):
        """Verify other properties are inherited."""
        base_path = Path("/tmp/test/recording")
        
        lro = LongRecordingOrganizer(base_folder_path=base_path)
        lro.LongRecording = self.mock_recording
        lro.channel_names = ["ch1", "ch2"]
        lro.labels = {"key": "val"} # Overwrite what init set (empty dict)
        
        # Perform split
        splits = lro.split(groups={"Group1": ["ch1"]})
        child = splits["Group1"]
        
        self.assertEqual(child.labels, {"key": "val"})
