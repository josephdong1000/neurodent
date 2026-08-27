"""
Test split, save_recording, and load_recording functionality for LongRecordingOrganizer.
"""

import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import spikeinterface.core as si

from neurodent.loading import LongRecordingOrganizer, split_recording


# Fixtures

@pytest.fixture
def dummy_long_recording(tmp_path):
    """Create a dummy LongRecordingOrganizer with 4 channels for testing."""
    duration_s = 2.0
    sampling_frequency = 1000.0
    num_channels = 4
    num_samples = int(duration_s * sampling_frequency)
    
    traces = np.random.randn(num_samples, num_channels).astype(np.float32)
    recording = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=sampling_frequency,
    )
    
    rec_folder = tmp_path / "input_recording"
    recording.save(folder=rec_folder, format="binary")
    
    lro = LongRecordingOrganizer(
        item=rec_folder,
        
        manual_datetimes=datetime(2023, 1, 1, 12, 0),
    )
    
    lro.channel_names = ["Ch0", "Ch1", "Ch2", "Ch3"]
    
    return lro


# Basic Functionality Tests

class TestSplitInMemory:
    """Test in-memory splitting functionality."""
    
    def test_split_returns_dict_of_lros(self, dummy_long_recording):
        """Test that split() returns a dictionary of LRO objects."""
        groups = {"GroupA": ["Ch0", "Ch1"], "GroupB": ["Ch2", "Ch3"]}
        splits = dummy_long_recording.split(groups)
        
        assert isinstance(splits, dict)
        assert "GroupA" in splits
        assert "GroupB" in splits
        assert isinstance(splits["GroupA"], LongRecordingOrganizer)

    def test_split_creates_in_memory_lros(self, dummy_long_recording):
        """Test that split() creates in-memory LROs."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        assert splits["GroupA"]._is_in_memory is True

    def test_split_preserves_channel_names(self, dummy_long_recording):
        """Test that channel names are preserved in split LROs."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)
        
        assert "Ch0" in splits["GroupA"].channel_names
        assert "Ch1" in splits["GroupA"].channel_names
        assert splits["GroupA"].LongRecording.get_num_channels() == 2

    def test_split_propagates_timestamps(self, dummy_long_recording):
        """Test that timestamps are propagated to child LROs."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)
        
        assert splits["GroupA"].manual_datetimes == dummy_long_recording.manual_datetimes


# Save Tests

class TestSaveRecording:
    """Test save_recording() functionality."""

    def test_save_recording_creates_zarr_folder(self, dummy_long_recording, tmp_path):
        """Test that save_recording() creates a zarr folder."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        output_dir = tmp_path / "output" / "GroupA"
        actual_path = splits["GroupA"].save_recording(output_dir, format="zarr")

        # SI appends .zarr suffix
        assert actual_path.exists()
        assert str(actual_path).endswith(".zarr")

    def test_save_recording_writes_sidecar(self, dummy_long_recording, tmp_path):
        """Test that save_recording() writes a NeuRodent metadata sidecar."""
        from neurodent import constants
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        output_dir = tmp_path / "output" / "GroupA"
        actual_path = splits["GroupA"].save_recording(output_dir, format="zarr")

        assert (actual_path / constants.NEURODENT_SIDECAR_NAME).exists()

    def test_save_recording_updates_in_memory_flag(self, dummy_long_recording, tmp_path):
        """Test that save_recording() clears the in-memory flag."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        output_dir = tmp_path / "output" / "GroupA"
        splits["GroupA"].save_recording(output_dir, format="zarr")

        assert splits["GroupA"]._is_in_memory is False

    def test_save_recording_binary_format(self, dummy_long_recording, tmp_path):
        """Test that save_recording() works with binary format."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        output_dir = tmp_path / "output" / "GroupA"
        actual_path = splits["GroupA"].save_recording(output_dir, format="binary")

        assert actual_path.exists()


# Standalone Function Tests

class TestSplitRecordingFunction:
    """Test the standalone split_recording() function."""

    def test_split_recording_persists_by_default(self, tmp_path):
        """Test that split_recording() persists results by default."""
        traces = np.random.randn(2000, 4).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        input_folder = tmp_path / "input"
        recording.save(folder=input_folder, format="binary")
        
        groups = {"AnimalA": ["0", "1"], "AnimalB": ["2", "3"]}
        output_base = tmp_path / "output"
        
        splits = split_recording(
            input_path=input_folder,
            groups=groups,
            output_base=output_base,
            
            format="zarr",
            manual_datetimes=datetime.now(),
        )
        
        assert "AnimalA" in splits
        assert "AnimalB" in splits
        # Zarr folders created with .zarr suffix
        assert (output_base / "AnimalA.zarr").exists()
        assert (output_base / "AnimalB.zarr").exists()

    def test_split_recording_no_save(self, tmp_path):
        """Test split_recording() with save=False."""
        traces = np.random.randn(2000, 4).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        input_folder = tmp_path / "input"
        recording.save(folder=input_folder, format="binary")

        groups = {"AnimalA": ["0", "1"]}

        splits = split_recording(
            input_path=input_folder,
            groups=groups,
            save=False,
            manual_datetimes=datetime.now(),
        )

        assert "AnimalA" in splits
        assert splits["AnimalA"]._is_in_memory is True

# Edge Cases and Error Handling

class TestSplitEdgeCases:
    """Test edge cases and error handling."""

    def test_split_missing_channels_raises_error(self, dummy_long_recording):
        """Test that requesting non-existent channels raises ValueError."""
        groups = {"GroupA": ["Ch0", "NonExistentChannel"]}
        
        with pytest.raises(ValueError, match="Channels not found"):
            dummy_long_recording.split(groups)

    def test_split_unused_channels_warning(self, dummy_long_recording, caplog):
        """Test that unused channels emit a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        
        groups = {"GroupA": ["Ch0", "Ch1"]}
        dummy_long_recording.split(groups)
        
        assert "channels not included in any group" in caplog.text

    def test_split_empty_groups_dict(self, dummy_long_recording):
        """Test split with empty groups dictionary."""
        groups = {}
        splits = dummy_long_recording.split(groups)
        
        assert splits == {}

    def test_save_recording_no_recording_raises_error(self, tmp_path):
        """Test that save_recording() raises error when no recording is loaded."""
        lro = LongRecordingOrganizer(item=None)

        with pytest.raises(ValueError, match="No recording to save"):
            lro.save_recording(tmp_path / "output")

    def test_split_no_recording_raises_error(self):
        """Test that split() raises error when no recording is loaded."""
        lro = LongRecordingOrganizer(item=None)
        
        with pytest.raises(ValueError, match="No recording loaded"):
            lro.split({"GroupA": ["Ch0"]})

    def test_split_si_import_error(self, monkeypatch):
        """Test that split raises ImportError when SI is not available."""
        import sys
        
        # Get the current module from sys.modules (handles test_imports.py reimporting)
        core_module = sys.modules['neurodent.loading.lro_merge']
        LRO = sys.modules['neurodent.loading.long_recording_organizer'].LongRecordingOrganizer
        
        lro = LRO(item='.', mode=None)
        lro.LongRecording = "dummy"
        
        monkeypatch.setattr(core_module, 'si', None)
        
        with pytest.raises(ImportError, match="SpikeInterface is required for split"):
            lro.split({"A": ["Ch0"]})

    def test_save_recording_si_import_error(self, monkeypatch, tmp_path):
        """Test that save_recording raises ImportError when SI is not available."""
        import sys

        # Get the current module from sys.modules (handles test_imports.py reimporting)
        core_module = sys.modules['neurodent.loading.lro_persistence']
        LRO = sys.modules['neurodent.loading.long_recording_organizer'].LongRecordingOrganizer

        lro = LRO(item='.', mode=None)
        lro.LongRecording = "dummy"

        monkeypatch.setattr(core_module, 'si', None)

        with pytest.raises(ImportError, match="SpikeInterface is required for save_recording"):
            lro.save_recording(tmp_path / "output")

# In-Memory Initialization Tests

class TestInitFromRecording:
    """Test initialization from existing recording objects."""

    def test_init_from_recording_sets_flag(self, tmp_path):
        """Test that _is_in_memory flag is set when initializing from recording."""
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        
        assert lro._is_in_memory is True

    def test_init_from_recording_sets_channel_names(self, tmp_path):
        """Test that channel_names are extracted from recording."""
        traces = np.random.randn(1000, 3).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        
        assert len(lro.channel_names) == 3
        assert all(isinstance(name, str) for name in lro.channel_names)

    def test_init_from_recording_sets_metadata(self, tmp_path):
        """Test that metadata is created from recording."""
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=500.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        
        assert lro.meta.n_channels == 2
        # LRO now resamples to GLOBAL_SAMPLING_RATE (1000.0 Hz) during initialization
        assert lro.meta.f_s == 1000.0

    def test_init_from_recording_sets_durations(self, tmp_path):
        """Test that file_durations is computed from recording."""
        duration_s = 2.0
        traces = np.random.randn(int(2000), 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        
        assert len(lro.file_durations) == 1
        assert abs(lro.file_durations[0] - duration_s) < 0.1


# Save Format / Overwrite Tests

class TestSaveRecordingFormats:
    """Test save_recording formats and overwrite semantics."""

    def test_save_recording_with_explicit_zarr_suffix(self, dummy_long_recording, tmp_path):
        """Test save_recording when output_dir already ends with .zarr."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        output_dir = tmp_path / "output" / "GroupA.zarr"
        actual_path = splits["GroupA"].save_recording(output_dir, format="zarr")

        # Should not double the .zarr suffix
        assert actual_path == output_dir
        assert actual_path.exists()

    def test_save_recording_refuses_overwrite_by_default(self, dummy_long_recording, tmp_path):
        """Saving to an existing folder without overwrite=True raises FileExistsError."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        output_dir = tmp_path / "output" / "GroupA"
        splits["GroupA"].save_recording(output_dir, format="zarr")

        with pytest.raises(FileExistsError, match="already exists"):
            splits["GroupA"].save_recording(output_dir, format="zarr")

    def test_save_recording_overwrite_existing(self, dummy_long_recording, tmp_path, caplog):
        """Test that save_recording(overwrite=True) replaces an existing folder with a warning."""
        import logging
        caplog.set_level(logging.WARNING)

        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        output_dir = tmp_path / "output" / "GroupA"

        # First save
        splits["GroupA"].save_recording(output_dir, format="zarr")

        # Second save with overwrite=True should warn and succeed
        caplog.clear()
        actual_path = splits["GroupA"].save_recording(output_dir, format="zarr", overwrite=True)

        assert "Overwriting existing recording folder" in caplog.text
        assert actual_path.exists()

    def test_save_recording_refuses_foreign_dir(self, dummy_long_recording, tmp_path):
        """save_recording(overwrite=True) must refuse to delete a non-recording directory."""
        groups = {"GroupA": ["Ch0", "Ch1"]}
        splits = dummy_long_recording.split(groups)

        # A foreign directory that is not a recognized SI/NeuRodent recording folder.
        foreign = tmp_path / "important_data"
        foreign.mkdir(parents=True)
        sentinel = foreign / "notes.txt"
        sentinel.write_text("do not delete")

        with pytest.raises(ValueError, match="does not look like"):
            splits["GroupA"].save_recording(foreign, format="binary", overwrite=True)

        # The foreign directory and its contents must survive untouched.
        assert sentinel.exists()
        assert sentinel.read_text() == "do not delete"


# Channel Name Inheritance Tests

class TestChannelNameInheritance:
    """Test channel name handling in split operations."""

    def test_split_with_renamed_channels(self, tmp_path):
        """Test split works when parent has custom channel names."""
        traces = np.random.randn(2000, 4).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        input_folder = tmp_path / "input"
        recording.save(folder=input_folder, format="binary")
        
        lro = LongRecordingOrganizer(
            item=input_folder,
            
            manual_datetimes=datetime(2023, 1, 1, 12, 0),
        )
        
        # Set custom channel names
        lro.channel_names = ["Left_Front", "Left_Back", "Right_Front", "Right_Back"]
        
        splits = lro.split({
            "Left": ["Left_Front", "Left_Back"],
            "Right": ["Right_Front", "Right_Back"],
        })
        
        assert "Left_Front" in splits["Left"].channel_names
        assert "Right_Back" in splits["Right"].channel_names

    def test_integer_channel_ids_warning(self, tmp_path, caplog):
        """Test that integer channel IDs trigger a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        input_folder = tmp_path / "input"
        recording.save(folder=input_folder, format="binary")
        
        # Loading should trigger warning about integer channel IDs
        lro = LongRecordingOrganizer(
            item=input_folder,
            
            manual_datetimes=datetime(2023, 1, 1),
        )
        
        assert "Channel IDs are integers" in caplog.text

    def test_split_with_none_channel_names(self, tmp_path):
        """Test split when channel_names is None (uses str(id) fallback)."""
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        lro.channel_names = None
        
        splits = lro.split({"A": ["0"]})
        assert splits["A"].LongRecording.get_num_channels() == 1

    def test_split_with_empty_channel_names(self, tmp_path):
        """Test split when channel_names is empty list."""
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        lro.channel_names = []
        
        splits = lro.split({"A": ["0"]})
        assert splits["A"].LongRecording.get_num_channels() == 1

    def test_split_with_mismatched_channel_names_length(self, tmp_path, caplog):
        """Test split when channel_names length doesn't match recording channels."""
        import logging
        caplog.set_level(logging.WARNING)
        
        traces = np.random.randn(1000, 4).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        lro.channel_names = ["Ch0", "Ch1"]  # Mismatched: 2 names for 4 channels
        
        splits = lro.split({"A": ["0", "1"]})
        
        assert "channel_names length mismatch" in caplog.text
        assert splits["A"].LongRecording.get_num_channels() == 2


# Boundary Condition Tests

class TestBoundaryConditions:
    """Test boundary conditions and stress cases."""

    def test_split_single_channel_group(self, dummy_long_recording):
        """Test splitting with only one channel per group."""
        groups = {"Single": ["Ch0"]}
        splits = dummy_long_recording.split(groups)
        
        assert splits["Single"].LongRecording.get_num_channels() == 1
        assert "Ch0" in splits["Single"].channel_names

    def test_split_all_channels_into_one_group(self, dummy_long_recording):
        """Test putting all channels into a single group."""
        groups = {"All": ["Ch0", "Ch1", "Ch2", "Ch3"]}
        splits = dummy_long_recording.split(groups)
        
        assert splits["All"].LongRecording.get_num_channels() == 4

    def test_split_many_groups(self, tmp_path):
        """Test splitting into many small groups."""
        traces = np.random.randn(1000, 10).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        
        lro = LongRecordingOrganizer(
            item=None,
            
            recording=recording,
        )
        lro.channel_names = [f"Ch{i}" for i in range(10)]
        
        # Create 5 groups of 2 channels each
        groups = {f"Group{i}": [f"Ch{i*2}", f"Ch{i*2+1}"] for i in range(5)}
        splits = lro.split(groups)
        
        assert len(splits) == 5
        for i in range(5):
            assert splits[f"Group{i}"].LongRecording.get_num_channels() == 2

    def test_split_recording_preserves_data_integrity(self, dummy_long_recording, tmp_path):
        """Test that split recordings contain correct data."""
        groups = {"First": ["Ch0"], "Second": ["Ch1"]}
        splits = dummy_long_recording.split(groups)
        
        # Get traces from original
        original_traces = dummy_long_recording.LongRecording.get_traces()
        
        # Get traces from splits
        first_traces = splits["First"].LongRecording.get_traces()
        second_traces = splits["Second"].LongRecording.get_traces()
        
        # Verify shapes
        assert first_traces.shape[1] == 1
        assert second_traces.shape[1] == 1
        
        # Verify data matches original channels
        np.testing.assert_array_almost_equal(first_traces[:, 0], original_traces[:, 0])
        np.testing.assert_array_almost_equal(second_traces[:, 0], original_traces[:, 1])


# Standalone Function Edge Cases

class TestSplitRecordingFunctionEdgeCases:
    """Test edge cases for the standalone split_recording() function."""

    def test_split_recording_requires_output_base_when_save(self, tmp_path):
        """Test that output_base is required when save=True."""
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        input_folder = tmp_path / "input"
        recording.save(folder=input_folder, format="binary")

        with pytest.raises(ValueError, match="output_base is required"):
            split_recording(
                input_path=input_folder,
                groups={"A": ["0"]},
                save=True,
                manual_datetimes=datetime.now(),
            )

    def test_split_recording_binary_format(self, tmp_path):
        """Test split_recording with binary format instead of zarr."""
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        input_folder = tmp_path / "input"
        recording.save(folder=input_folder, format="binary")
        
        splits = split_recording(
            input_path=input_folder,
            groups={"A": ["0"]},
            output_base=tmp_path / "output",
            format="binary",
            
            manual_datetimes=datetime.now(),
        )
        
        assert (tmp_path / "output" / "A").exists()


# Recording Data Verification Tests

class TestDataVerification:
    """Verify data integrity across split and persist operations."""

    def test_round_trip_data_integrity(self, dummy_long_recording, tmp_path):
        """Test that data survives split -> save_recording -> reload cycle."""
        original_traces = dummy_long_recording.LongRecording.get_traces()

        # Split
        splits = dummy_long_recording.split({"A": ["Ch0", "Ch1"]})

        # Save
        output_dir = tmp_path / "output" / "A"
        actual_path = splits["A"].save_recording(output_dir, format="binary")

        # Reload via load_recording
        reloaded = LongRecordingOrganizer.load_recording(actual_path)
        reloaded_traces = reloaded.LongRecording.get_traces()

        # Verify
        np.testing.assert_array_almost_equal(
            reloaded_traces[:, 0],
            original_traces[:, 0],
            decimal=5
        )


# Save/Load Round-Trip Tests

class TestSaveLoadRoundTrip:
    """Test the save_recording -> load_recording faithful round-trip."""

    def _make_child(self, dummy_long_recording):
        """Split a parent LRO and return a child with rich metadata set."""
        dummy_long_recording.file_end_datetimes = [datetime(2023, 1, 1, 14, 0)]
        dummy_long_recording.bad_channel_names = ["Ch1"]
        dummy_long_recording.labels = {"cohort": "A"}
        return dummy_long_recording.split({"A": ["Ch0", "Ch1"]})["A"]

    @pytest.mark.parametrize("fmt", ["zarr", "binary"])
    def test_round_trip_restores_metadata(self, dummy_long_recording, tmp_path, fmt):
        """save_recording -> load_recording restores LRO-level metadata faithfully."""
        child = self._make_child(dummy_long_recording)

        actual_path = child.save_recording(tmp_path / "out" / "A", format=fmt)
        back = LongRecordingOrganizer.load_recording(actual_path)

        assert back.channel_names == ["Ch0", "Ch1"]
        assert back.file_end_datetimes == child.file_end_datetimes
        assert back.bad_channel_names == child.bad_channel_names
        assert back.labels == {"cohort": "A"}
        assert back.datetimes_are_start == child.datetimes_are_start
        assert back.file_durations == child.file_durations
        assert back.meta.f_s == child.meta.f_s
        assert back.meta.V_units == child.meta.V_units
        assert back._is_in_memory is False
        assert Path(back.base_folder_path) == Path(actual_path)
        np.testing.assert_array_almost_equal(
            back.LongRecording.get_traces(),
            child.LongRecording.get_traces(),
            decimal=5,
        )

    def test_load_recording_missing_sidecar_warns(self, dummy_long_recording, tmp_path, caplog):
        """Missing sidecar -> bare reload + warning (strict=False)."""
        import logging
        from neurodent import constants

        child = self._make_child(dummy_long_recording)
        actual_path = child.save_recording(tmp_path / "out" / "A", format="binary")
        (actual_path / constants.NEURODENT_SIDECAR_NAME).unlink()

        caplog.set_level(logging.WARNING)
        back = LongRecordingOrganizer.load_recording(actual_path)

        assert "sidecar" in caplog.text.lower()
        # Bare reload still recovers the traces and folder path.
        assert back.LongRecording.get_num_channels() == 2
        assert Path(back.base_folder_path) == Path(actual_path)

    def test_load_recording_missing_sidecar_strict_raises(self, dummy_long_recording, tmp_path):
        """Missing sidecar with strict=True raises FileNotFoundError."""
        from neurodent import constants

        child = self._make_child(dummy_long_recording)
        actual_path = child.save_recording(tmp_path / "out" / "A", format="binary")
        (actual_path / constants.NEURODENT_SIDECAR_NAME).unlink()

        with pytest.raises(FileNotFoundError, match="sidecar"):
            LongRecordingOrganizer.load_recording(actual_path, strict=True)

    def test_load_recording_not_a_directory_raises(self, tmp_path):
        """load_recording on a non-directory raises FileNotFoundError."""
        missing = tmp_path / "nope"
        with pytest.raises(FileNotFoundError):
            LongRecordingOrganizer.load_recording(missing)

    def test_metadata_consistency_after_split(self, dummy_long_recording):
        """Test that metadata is consistent after splitting."""
        original_fs = dummy_long_recording.meta.f_s
        
        splits = dummy_long_recording.split({"A": ["Ch0", "Ch1"]})
        
        assert splits["A"].meta.f_s == original_fs
        assert splits["A"].meta.n_channels == 2


# File End Datetime Inheritance Tests

class TestDatetimeInheritance:
    """Test datetime propagation in split operations."""

    def test_split_inherits_file_end_datetimes(self, dummy_long_recording):
        """Test that file_end_datetimes are inherited."""
        # Set file_end_datetimes on parent
        dummy_long_recording.file_end_datetimes = [datetime(2023, 1, 1, 14, 0)]
        
        splits = dummy_long_recording.split({"A": ["Ch0"]})
        
        assert hasattr(splits["A"], "file_end_datetimes")
        assert splits["A"].file_end_datetimes == [datetime(2023, 1, 1, 14, 0)]

    def test_split_inherits_datetimes_are_start_flag(self, dummy_long_recording):
        """Test that datetimes_are_start flag is inherited."""
        dummy_long_recording.datetimes_are_start = False
        
        splits = dummy_long_recording.split({"A": ["Ch0"]})

        assert splits["A"].datetimes_are_start is False


# Default SI Extractor Tests

class TestDefaultSiExtractor:
    """Guard the default mode='si' load path against the deprecated si.load_extractor."""

    def test_si_mode_load_does_not_use_deprecated_load_extractor(self, tmp_path):
        """Loading with the default extract_func must not emit the load_extractor warning."""
        traces = np.random.randn(1000, 2).astype(np.float32)
        recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
        input_folder = tmp_path / "input"
        recording.save(folder=input_folder, format="binary")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lro = LongRecordingOrganizer(item=input_folder, mode="si")

        offending = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "load_extractor" in str(w.message)
        ]
        assert not offending, f"Unexpected load_extractor deprecation: {[str(w.message) for w in offending]}"
        assert lro.LongRecording.get_num_channels() == 2

