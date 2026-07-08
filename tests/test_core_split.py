
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime
from neurodent.loading import LongRecordingOrganizer, RecordingMetadata
import pytest

class TestLROSplit(unittest.TestCase):
    def setUp(self):
        # Patch dependencies
        self.si_patcher = patch('neurodent.loading.lro_merge.si')
        self.mock_si = self.si_patcher.start()

        self.spre_patcher = patch('neurodent.loading.lro_loading.spre')
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
        """Mock behavior for LRO.__init__ to set basic attributes from constructor parameters."""
        # Core parameters
        instance.item = kwargs.get('item', args[0] if args else None)
        instance.manual_datetimes = kwargs.get('manual_datetimes', None)
        instance.datetimes_are_start = kwargs.get('datetimes_are_start', True)
        instance.n_jobs = kwargs.get('n_jobs', 1)
        instance.n_truncate = kwargs.get('truncate', None)
        instance.truncate = bool(instance.n_truncate)

        # Initialize attributes to defaults (as in real __init__)
        instance.file_durations = []
        instance.cumulative_file_durations = []
        instance.bad_channel_names = []
        instance.meta = None

        # Other defaults
        instance.temppaths = []
        instance.channel_names = None
        instance.LongRecording = kwargs.get('recording', None)
        instance._is_in_memory = False

    def test_split_inherits_manual_datetimes(self):
        """Verify that split() children inherit parent's manual_datetimes."""
        manual_dt = datetime(2023, 1, 1, 12, 0, 0)

        lro = LongRecordingOrganizer(
            item=Path("/tmp/test/recording"),
            manual_datetimes=manual_dt,
            datetimes_are_start=False
        )
        lro.LongRecording = self.mock_recording
        lro.channel_names = ["ch1", "ch2"]

        splits = lro.split(groups={"Group1": ["ch1"]})

        child = splits["Group1"]
        self.assertEqual(child.manual_datetimes, manual_dt)
        self.assertEqual(child.datetimes_are_start, False)

    def test_split_inherits_file_timestamps(self):
        """Verify that split() children inherit file_end_datetimes via post-instantiation assignment."""
        lro = LongRecordingOrganizer(item=Path("/tmp/test"))
        lro.LongRecording = self.mock_recording
        lro.channel_names = ["ch1", "ch2"]
        lro.file_end_datetimes = [datetime(2023, 1, 1, 12, 0, 0)]
        lro.file_durations = [10.0]
        lro.cumulative_file_durations = [10.0]

        splits = lro.split(groups={"Group1": ["ch1"]})
        child = splits["Group1"]

        # These should be copied via post-instantiation assignment
        self.assertEqual(child.file_end_datetimes, lro.file_end_datetimes)
        self.assertEqual(child.file_durations, lro.file_durations)
        self.assertEqual(child.cumulative_file_durations, lro.cumulative_file_durations)

    def test_split_inherits_bad_channels_filtered(self):
        """Verify that split() children inherit only relevant bad channels."""
        lro = LongRecordingOrganizer(item=Path("/tmp/test"))
        lro.LongRecording = self.mock_recording
        lro.channel_names = ["ch1", "ch2"]
        lro.bad_channel_names = ["ch1", "ch3"]  # ch3 not in this recording

        splits = lro.split(groups={"Group1": ["ch1"]})
        child = splits["Group1"]

        # Should only inherit ch1, not ch3 (not in split) or ch2 (not bad)
        self.assertEqual(child.bad_channel_names, ["ch1"])

    def test_split_inherits_metadata_with_updated_channels(self):
        """Verify that split() children inherit metadata with updated channel info."""
        meta = RecordingMetadata(
            None,
            n_channels=2,
            f_s=1000.0,
            dt_end=datetime(2023, 1, 1),
            channel_names=["ch1", "ch2"],
            V_units="µV",
            mult_to_uV=1.0
        )

        lro = LongRecordingOrganizer(item=Path("/tmp/test"))
        lro.LongRecording = self.mock_recording
        lro.channel_names = ["ch1", "ch2"]
        lro.meta = meta

        splits = lro.split(groups={"Group1": ["ch1"]})
        child = splits["Group1"]

        # Metadata should be deep copied and updated
        self.assertIsNotNone(child.meta)
        self.assertEqual(child.meta.n_channels, 1)
        self.assertEqual(child.meta.channel_names, ["ch1"])
        self.assertEqual(child.meta.f_s, 1000.0)  # Inherited
        self.assertEqual(child.meta.V_units, "µV")  # Inherited

        # Should be a deep copy, not the same object
        self.assertIsNot(child.meta, meta)

    def test_split_all_attributes_passed_via_constructor(self):
        """Ensure constructor params are passed correctly and other attributes are assigned post-instantiation."""
        manual_dt = datetime(2023, 1, 1, 12, 0, 0)
        file_end_dt = [datetime(2023, 1, 1, 13, 0, 0)]
        meta = RecordingMetadata(
            None,
            n_channels=2,
            f_s=1000.0,
            dt_end=datetime(2023, 1, 1),
            channel_names=["ch1", "ch2"]
        )

        lro = LongRecordingOrganizer(item=Path("/tmp/test"))
        lro.LongRecording = self.mock_recording
        lro.channel_names = ["ch1", "ch2"]
        lro.manual_datetimes = manual_dt
        lro.datetimes_are_start = False
        lro.n_jobs = 4
        lro.file_end_datetimes = file_end_dt
        lro.file_durations = [3600.0]
        lro.cumulative_file_durations = [3600.0]
        lro.bad_channel_names = ["ch2"]
        lro.meta = meta

        splits = lro.split(groups={"Group1": ["ch1"]})
        child = splits["Group1"]

        # Constructor params should be passed via constructor
        self.assertEqual(child.manual_datetimes, manual_dt)
        self.assertEqual(child.datetimes_are_start, False)
        self.assertEqual(child.n_jobs, 4)

        # File-level attributes should be assigned post-instantiation
        self.assertEqual(child.file_end_datetimes, file_end_dt)
        self.assertEqual(child.file_durations, [3600.0])
        self.assertEqual(child.cumulative_file_durations, [3600.0])
        self.assertEqual(child.bad_channel_names, [])  # ch2 not in Group1
        self.assertIsNotNone(child.meta)
        self.assertEqual(child.meta.n_channels, 1)

