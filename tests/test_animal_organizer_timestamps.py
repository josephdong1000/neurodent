#!/usr/bin/env python3
"""
Test cases for AnimalOrganizer's enhanced timestamp handling functionality.

This module tests the new timestamp processing system that allows:
- Single datetime (global timeline)
- List of datetimes (per-LRO assignment)
- User-defined timestamp extraction functions
- Mixed dictionaries with functions and explicit timestamps
- Error handling for invalid inputs and failed user functions
"""

import pytest
import re
import tempfile
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from neurodent.visualization import results
from neurodent import core


class TestAnimalOrganizerTimestampHandling:
    """Test AnimalOrganizer's enhanced timestamp handling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        self.animal_id = "A123"

        # Create test folders
        self.folder1 = self.base_path / f"WT_{self.animal_id}_2023-01-15"
        self.folder2 = self.base_path / f"WT_{self.animal_id}_2023-01-16"
        self.folder3 = self.base_path / f"WT_{self.animal_id}_2023-01-17"

        for folder in [self.folder1, self.folder2, self.folder3]:
            folder.mkdir(parents=True)
            (folder / "dummy_ColMajor_001.bin").touch()
            (folder / "dummy_Meta_001.json").touch()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_mock_lro(self, folder_name="test", start_time=None):
        """Create a mock LongRecordingOrganizer for testing."""
        mock_lro = Mock()
        mock_lro.channel_names = ["LMot", "RMot", "LAud"]
        mock_lro.meta = Mock(f_s=1000, n_channels=3)
        mock_lro.base_folder_path = folder_name
        mock_lro.file_durations = [100.0]  # 100 second recording

        # Mock recording
        mock_recording = Mock()
        mock_recording.get_duration.return_value = 100.0
        mock_lro.LongRecording = mock_recording

        # Mock file_end_datetimes for timeline calculation
        if start_time:
            mock_lro.file_end_datetimes = [start_time + timedelta(seconds=100)]
        else:
            mock_lro.file_end_datetimes = [None]

        return mock_lro

    @patch("glob.glob")
    def test_single_datetime_global_timeline(self, mock_glob):
        """Test that a single datetime creates a global timeline for all LROs."""
        # Setup
        mock_glob.return_value = [
            str(self.folder1),
            str(self.folder2),
            str(self.folder3),
        ]

        with patch.object(core, "LongRecordingOrganizer") as mock_lro_class:
            mock_lro_class.return_value = self._create_mock_lro()

            global_start = datetime(2023, 1, 15, 10, 0, 0)

            # Create AnimalOrganizer with single datetime using new pattern-based discovery
            ao = results.AnimalOrganizer(
                pattern=str(self.base_path) + "/WT_{animal}_{session}",
                animal_id=self.animal_id,
                lro_kwargs={"manual_datetimes": global_start},
            )

            # Verify processing
            assert hasattr(ao, "_processed_timestamps")
            assert ao._processed_timestamps is not None
            assert len(ao._processed_timestamps) == 3  # One per folder

            # All folders should get continuous timestamps (no longer the same datetime)
            # First folder should start at global_start, subsequent folders should be offset by durations
            sorted_folders = sorted(ao._processed_timestamps.keys())
            first_folder_timestamp = ao._processed_timestamps[sorted_folders[0]]
            assert first_folder_timestamp == global_start

            # Verify timestamps are continuous (verified by the continuous timeline test above)

            # Verify LROs were created (multiple times due to two-pass approach for continuous timeline)
            assert mock_lro_class.call_count >= 3

            # The continuous timeline functionality should compute different start times for each folder
            unique_timestamps = set(ao._processed_timestamps.values())
            assert len(unique_timestamps) == 3  # All timestamps should be different

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_list_of_datetimes_per_lro_assignment(self, mock_glob, mock_lro_class):
        """Test that a list of datetimes gets assigned to LROs in order."""
        # Setup
        mock_glob.return_value = [
            str(self.folder1),
            str(self.folder2),
            str(self.folder3),
        ]
        mock_lro_class.return_value = self._create_mock_lro()

        datetime_list = [
            datetime(2023, 1, 15, 10, 0, 0),
            datetime(2023, 1, 16, 11, 0, 0),
            datetime(2023, 1, 17, 12, 0, 0),
        ]

        # Create AnimalOrganizer with datetime list
        ao = results.AnimalOrganizer(
            pattern=str(self.base_path) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            
            lro_kwargs={"manual_datetimes": datetime_list},
        )

        # Verify processing - should apply list to all folders
        assert len(ao._processed_timestamps) == 3
        for folder_name, timestamp in ao._processed_timestamps.items():
            assert timestamp == datetime_list  # Each folder gets the entire list

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_user_defined_timestamp_function(self, mock_glob, mock_lro_class):
        """Test that user-defined functions can extract timestamps from folders."""
        # Setup
        mock_glob.return_value = [
            str(self.folder1),
            str(self.folder2),
            str(self.folder3),
        ]
        mock_lro_class.return_value = self._create_mock_lro()

        def extract_timestamp_from_folder(folder_path):
            """Extract timestamp from folder name pattern."""
            folder_name = Path(folder_path).name
            if "2023-01-15" in folder_name:
                return datetime(2023, 1, 15, 9, 0, 0)
            elif "2023-01-16" in folder_name:
                return datetime(2023, 1, 16, 10, 0, 0)
            elif "2023-01-17" in folder_name:
                return datetime(2023, 1, 17, 11, 0, 0)
            return datetime(2023, 1, 1, 0, 0, 0)  # fallback

        # Create AnimalOrganizer with user function
        ao = results.AnimalOrganizer(
            pattern=str(self.base_path) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            
            lro_kwargs={"manual_datetimes": extract_timestamp_from_folder},
        )

        # Verify processing
        assert len(ao._processed_timestamps) == 3

        # Check that function was applied to each folder
        # Keys are now full paths (from _get_item_key), not just folder names
        expected_times = {
            str(self.folder1): datetime(2023, 1, 15, 9, 0, 0),
            str(self.folder2): datetime(2023, 1, 16, 10, 0, 0),
            str(self.folder3): datetime(2023, 1, 17, 11, 0, 0),
        }

        for folder_path, expected_time in expected_times.items():
            assert ao._processed_timestamps[folder_path] == expected_time

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_mixed_dictionary_specification(self, mock_glob, mock_lro_class):
        """Test dictionary with mixed function and explicit timestamp specification."""
        # Setup
        mock_glob.return_value = [
            str(self.folder1),
            str(self.folder2),
            str(self.folder3),
        ]
        mock_lro_class.return_value = self._create_mock_lro()

        def extract_for_folder2(folder_path):
            return datetime(2023, 1, 16, 14, 30, 0)

        mixed_spec = {
            f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 15, 8, 0, 0),
            f"WT_{self.animal_id}_2023-01-16": extract_for_folder2,
            f"WT_{self.animal_id}_2023-01-17": [
                datetime(2023, 1, 17, 10, 0, 0),
                datetime(2023, 1, 17, 14, 0, 0),
            ],
        }

        # Create AnimalOrganizer with mixed dictionary
        ao = results.AnimalOrganizer(
            pattern=str(self.base_path) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            
            lro_kwargs={"manual_datetimes": mixed_spec},
        )

        # Verify processing
        assert len(ao._processed_timestamps) == 3

        # Keys are now full paths (from _get_item_key), not just folder names
        # Check explicit datetime
        assert ao._processed_timestamps[str(self.folder1)] == datetime(
            2023, 1, 15, 8, 0, 0
        )

        # Check function result
        assert ao._processed_timestamps[str(self.folder2)] == datetime(
            2023, 1, 16, 14, 30, 0
        )

        # Check list
        expected_list = [
            datetime(2023, 1, 17, 10, 0, 0),
            datetime(2023, 1, 17, 14, 0, 0),
        ]
        assert (
            ao._processed_timestamps[str(self.folder3)] == expected_list
        )

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_invalid_timestamp_type_error(self, mock_glob, mock_lro_class):
        """Test that invalid timestamp types raise appropriate errors."""
        # Setup
        mock_glob.return_value = [str(self.folder1)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Test invalid type (int instead of datetime/str/list)
        with pytest.raises(TypeError) as exc_info:
            results.AnimalOrganizer(
                pattern=str(self.base_path) + "/WT_{animal}_{session}",
                animal_id=self.animal_id,
                
                lro_kwargs={"manual_datetimes": 12345},  # Int instead of datetime
            )

        assert "Invalid timestamp input type" in str(exc_info.value)

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_invalid_list_items_error(self, mock_glob, mock_lro_class):
        """Test that lists with non-datetime items raise errors."""
        # Setup
        mock_glob.return_value = [str(self.folder1)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Test invalid list items
        invalid_list = [datetime(2023, 1, 15, 10, 0, 0), "not a datetime"]

        with pytest.raises(TypeError) as exc_info:
            results.AnimalOrganizer(
                pattern=str(self.base_path) + "/WT_{animal}_{session}",
                animal_id=self.animal_id,
                
                lro_kwargs={"manual_datetimes": invalid_list},
            )

        assert "All items in timestamp list must be datetime objects" in str(
            exc_info.value
        )

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_invalid_list_items_error(self, mock_glob, mock_lro_class):
        """Test that lists with non-datetime items raise errors."""
        # Setup
        mock_glob.return_value = [str(self.folder1)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Test invalid list items
        invalid_list = [datetime(2023, 1, 15, 10, 0, 0), "not a datetime"]

        with pytest.raises(TypeError) as exc_info:
            results.AnimalOrganizer(
                pattern=str(self.base_path) + "/WT_{animal}_{session}",
                animal_id=self.animal_id,
                
                lro_kwargs={"manual_datetimes": invalid_list},
            )

        assert "All items in timestamp list must be datetime objects" in str(
            exc_info.value
        )

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_user_function_failure_error(self, mock_glob, mock_lro_class):
        """Test that user function failures are wrapped with context."""
        # Setup
        mock_glob.return_value = [str(self.folder1)]
        mock_lro_class.return_value = self._create_mock_lro()

        def failing_function(folder_path):
            raise ValueError("Simulated extraction failure")

        with pytest.raises(Exception) as exc_info:
            results.AnimalOrganizer(
                pattern=str(self.base_path) + "/WT_{animal}_{session}",
                animal_id=self.animal_id,
                
                lro_kwargs={"manual_datetimes": failing_function},
            )

        error_str = str(exc_info.value)
        assert "User timestamp function failed" in error_str
        assert "Simulated extraction failure" in error_str

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_mixed_config_is_allowed(self, mock_glob, mock_lro_class):
        """Test that dictionary with extra keys (mixed config) is allowed in fallback mode."""
        # Setup
        mock_glob.return_value = [str(self.folder1), str(self.folder2)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Dictionary with extra "Start_Animal" key - should be ignored now
        mixed_spec = {
            f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 15, 10, 0, 0),
            f"WT_{self.animal_id}_2023-01-16": datetime(2023, 1, 16, 10, 0, 0),
            "Start_Animal": datetime(2023, 1, 17, 10, 0, 0),
        }

        # Should NOT raise ValueError anymore
        ao = results.AnimalOrganizer(
            pattern=str(self.base_path) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            
            lro_kwargs={"manual_datetimes": mixed_spec},
        )

        assert len(ao._processed_timestamps) == 2

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_backward_compatibility_no_manual_datetimes(
        self, mock_glob, mock_lro_class
    ):
        """Test that AnimalOrganizer works without manual_datetimes (backward compatibility)."""
        # Setup
        mock_glob.return_value = [str(self.folder1), str(self.folder2)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Create AnimalOrganizer without manual_datetimes
        ao = results.AnimalOrganizer(
            pattern=f"{self.base_path}/WT_{{animal}}_{{session}}",
            animal_id=self.animal_id,
        )

        # Should work fine
        assert ao._processed_timestamps is None
        assert len(ao.long_recordings) == 2

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_timeline_summary_functionality(self, mock_glob, mock_lro_class):
        """Test that timeline summary functionality works correctly."""
        # Setup with specific start times
        start_times = [datetime(2023, 1, 15, 10, 0, 0), datetime(2023, 1, 16, 11, 0, 0)]

        mock_glob.return_value = [str(self.folder1), str(self.folder2)]

        # Create mocks with timing information
        def mock_lro_side_effect(*args, **kwargs):
            folder_path = args[0]
            if "2023-01-15" in folder_path:
                return self._create_mock_lro("folder1", start_times[0])
            else:
                return self._create_mock_lro("folder2", start_times[1])

        mock_lro_class.side_effect = mock_lro_side_effect

        # Create AnimalOrganizer
        ao = results.AnimalOrganizer(
            pattern=str(self.base_path) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            
            lro_kwargs={"manual_datetimes": start_times[0]},  # Single datetime
        )

        # Test timeline summary DataFrame
        timeline_df = ao.get_timeline_summary()
        assert isinstance(timeline_df, pd.DataFrame)
        assert len(timeline_df) == 2  # Two LROs

        # Check columns exist
        expected_columns = [
            "lro_index",
            "start_time",
            "end_time",
            "duration_s",
            "n_files",
            "folder_path",
            "folder_name",
        ]
        for col in expected_columns:
            assert col in timeline_df.columns

        # Check data validity
        assert timeline_df["duration_s"].iloc[0] == 100.0  # Mock duration
        assert timeline_df["n_files"].iloc[0] == 1  # Mock file count

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_recursive_function_resolution(self, mock_glob, mock_lro_class):
        """Test that functions returning functions are resolved recursively."""
        # Setup
        mock_glob.return_value = [str(self.folder1)]
        mock_lro_class.return_value = self._create_mock_lro()

        def outer_function(folder_path):
            def inner_function(folder_path):
                return datetime(2023, 1, 15, 12, 0, 0)

            return inner_function

        # Create AnimalOrganizer with nested function
        ao = results.AnimalOrganizer(
            pattern=str(self.base_path) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            
            lro_kwargs={"manual_datetimes": outer_function},
        )

        # Verify that recursive resolution worked
        # Keys are now full paths (from _get_item_key)
        assert ao._processed_timestamps[str(self.folder1)] == datetime(2023, 1, 15, 12, 0, 0)

    @patch("glob.glob")
    def test_continuous_timeline_single_datetime(self, mock_glob):
        """Test that single datetime creates continuous (non-overlapping) timeline."""
        # Setup
        mock_glob.return_value = [
            str(self.folder1),
            str(self.folder2),
            str(self.folder3),
        ]

        # Create mock LROs with specific durations
        def create_mock_lro_with_duration(duration_seconds):
            mock_lro = Mock()
            mock_lro.channel_names = ["LMot", "RMot", "LAud"]
            mock_lro.meta = Mock(f_s=1000, n_channels=3)
            mock_lro.file_durations = [duration_seconds]

            # Mock recording with specific duration
            mock_recording = Mock()
            mock_recording.get_duration.return_value = duration_seconds
            mock_lro.LongRecording = mock_recording
            mock_lro.file_end_datetimes = [None]

            return mock_lro

        # Define durations for each folder (by folder name)
        folder_durations_by_name = {
            "WT_A123_2023-01-15": 3600.0,  # 1 hour
            "WT_A123_2023-01-16": 1800.0,  # 30 minutes
            "WT_A123_2023-01-17": 7200.0,  # 2 hours
        }

        with patch.object(core, "LongRecordingOrganizer") as mock_lro_class:

            def mock_lro_side_effect(*args, **kwargs):
                item = args[0] if args else None
                item_str = str(item)
                # Match by folder name substring
                for fname, dur in folder_durations_by_name.items():
                    if fname in item_str:
                        return create_mock_lro_with_duration(dur)
                return create_mock_lro_with_duration(3600.0)

            mock_lro_class.side_effect = mock_lro_side_effect

            global_start = datetime(2023, 1, 15, 10, 0, 0)

            # Create AnimalOrganizer with single datetime
            ao = results.AnimalOrganizer(
                pattern=str(self.base_path) + "/WT_{animal}_{session}",
                animal_id=self.animal_id,
                
                lro_kwargs={"manual_datetimes": global_start},
            )

            # Verify continuous timeline
            assert len(ao._processed_timestamps) == 3

            # Convert folder paths to folder names for lookup
            folder_name_to_path = {
                Path(path).name: path
                for path in [str(self.folder1), str(self.folder2), str(self.folder3)]
            }

            # Expected timeline (continuous, non-overlapping)
            expected_timeline = {}
            current_time = global_start

            # Process folders in sorted order (by animalday, then by folder order)
            for folder_name in sorted(ao._processed_timestamps.keys()):
                expected_timeline[folder_name] = current_time
                # Look up duration by matching folder name
                duration = 3600.0  # default
                for fname, dur in folder_durations_by_name.items():
                    if fname in folder_name or folder_name in fname:
                        duration = dur
                        break
                current_time = current_time + timedelta(seconds=duration)

            # Verify continuous timeline
            for folder_name, expected_start in expected_timeline.items():
                actual_start = ao._processed_timestamps[folder_name]
                assert actual_start == expected_start, (
                    f"Folder {folder_name}: expected {expected_start}, got {actual_start}"
                )

            # Verify no temporal overlaps
            timeline_list = [
                (name, time) for name, time in ao._processed_timestamps.items()
            ]
            timeline_list.sort(key=lambda x: x[1])  # Sort by start time

            for i in range(len(timeline_list) - 1):
                current_folder, current_start = timeline_list[i]
                next_folder, next_start = timeline_list[i + 1]

                # Calculate end time of current folder using name-based lookup
                current_duration = 3600.0  # default
                for fname, dur in folder_durations_by_name.items():
                    if fname in current_folder or current_folder in fname:
                        current_duration = dur
                        break
                current_end = current_start + timedelta(seconds=current_duration)

                # Verify next folder starts exactly when current folder ends
                assert current_end == next_start, (
                    f"Gap/overlap between {current_folder} and {next_folder}: {current_folder} ends at {current_end}, {next_folder} starts at {next_start}"
                )

            logging.info(
                "✅ Continuous timeline verified: folders are sequential with no gaps or overlaps"
            )

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_overlapping_animaldays_with_timestamps(self, mock_glob, mock_lro_class):
        """Test timestamp handling with overlapping animaldays (same day, multiple folders)."""
        # Setup folders that parse to same animalday - use different base directory
        overlap_dir = self.base_path / "overlap_test"
        overlap_dir.mkdir(parents=True, exist_ok=True)

        folder_a = overlap_dir / f"WT_{self.animal_id}_2023-01-15"
        folder_b = overlap_dir / f"WT_{self.animal_id}_2023-01-15(1)"
        folder_c = overlap_dir / f"WT_{self.animal_id}_2023-01-15(2)"

        for folder in [folder_a, folder_b, folder_c]:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "dummy_ColMajor_001.bin").touch()
            (folder / "dummy_Meta_001.json").touch()

        mock_glob.return_value = [str(folder_a), str(folder_b), str(folder_c)]

        # Create different mock LROs for sorting/merging
        mock_lros = []
        expected_median_times = [
            100.0,
            50.0,
            150.0,
        ]  # Out of name order but chronological

        for i, median_time in enumerate(expected_median_times):
            mock_lro = Mock()
            mock_lro.channel_names = ["LMot", "RMot", "LAud"]
            mock_lro.meta = Mock()

            # Mock the LongRecording with timing data
            mock_recording = Mock()
            mock_recording.get_num_samples.return_value = int(median_time * 2 * 1000)
            mock_recording.get_sampling_frequency.return_value = 1000.0
            mock_lro.LongRecording = mock_recording

            # Add file_end_datetimes based on expected median times
            # Create timestamps that will result in the expected median times
            base_time = datetime(2023, 1, 15, 8, 0, 0)
            mock_lro.file_end_datetimes = [base_time + timedelta(seconds=median_time)]

            # Add merge method
            def mock_merge(other_lro):
                pass

            mock_lro.merge = mock_merge

            mock_lros.append(mock_lro)

        # Create call counter and map folders to their LROs
        call_count = 0

        def mock_lro_side_effect(*args, **kwargs):
            nonlocal call_count
            folder_path = str(args[0])
            if "2023-01-15(2)" in folder_path:
                return mock_lros[2]  # Highest median time (150.0)
            elif "2023-01-15(1)" in folder_path:
                return mock_lros[1]  # Lowest median time (50.0)
            else:  # WT_A123_2023-01-15
                return mock_lros[0]  # Middle median time (100.0)

        mock_lro_class.side_effect = mock_lro_side_effect

        # Test with per-folder timestamp specification
        folder_timestamps = {
            f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 15, 8, 0, 0),
            f"WT_{self.animal_id}_2023-01-15(1)": datetime(2023, 1, 15, 9, 0, 0),
            f"WT_{self.animal_id}_2023-01-15(2)": datetime(2023, 1, 15, 10, 0, 0),
        }

        # Create AnimalOrganizer with overlapping folders
        ao = results.AnimalOrganizer(
            pattern=str(overlap_dir) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            lro_kwargs={"manual_datetimes": folder_timestamps},
            normalize_session=lambda s: re.sub(r"\(\d+\)$", "", s),
        )

        # With normalize_session, folders with (N) suffixes are merged
        # into a single session (overlapping animalday merging)
        assert len(ao.long_recordings) == 1  # Merged into 1 session
        assert len(ao.animaldays) == 1  # 1 unique animalday

    @pytest.mark.unit
    def test_resolve_timestamp_input_unit_tests(self):
        """Unit tests for _resolve_timestamp_input method."""
        # Create AnimalOrganizer instance for testing (without full initialization)
        ao = results.AnimalOrganizer.__new__(results.AnimalOrganizer)

        test_folder = Path("/test/folder")

        # Test datetime passthrough
        test_dt = datetime(2023, 1, 15, 10, 0, 0)
        result = ao._resolve_timestamp_input(test_dt, test_folder)
        assert result == test_dt

        # Test list passthrough
        test_list = [datetime(2023, 1, 15, 10, 0, 0), datetime(2023, 1, 15, 14, 0, 0)]
        result = ao._resolve_timestamp_input(test_list, test_folder)
        assert result == test_list

        # Test function execution
        def test_function(folder_path):
            return datetime(2023, 1, 15, 12, 0, 0)

        result = ao._resolve_timestamp_input(test_function, test_folder)
        assert result == datetime(2023, 1, 15, 12, 0, 0)

        # Test invalid type
        with pytest.raises(TypeError) as exc_info:
            ao._resolve_timestamp_input(12345, test_folder)
        assert "Invalid timestamp input type" in str(exc_info.value)

        # Test invalid list items
        invalid_list = [datetime(2023, 1, 15, 10, 0, 0), "not datetime"]
        with pytest.raises(TypeError) as exc_info:
            ao._resolve_timestamp_input(invalid_list, test_folder)
        assert "All items in timestamp list must be datetime objects" in str(
            exc_info.value
        )

    @patch("glob.glob")
    def test_datetimes_are_start_end_time_support(self, mock_glob):
        """Test that datetimes_are_start=False computes timeline backwards from end time."""
        mock_glob.return_value = [
            str(self.folder1),
            str(self.folder2),
            str(self.folder3),
        ]

        def create_mock_lro_with_duration(duration_seconds):
            mock_lro = Mock()
            mock_lro.channel_names = ["LMot", "RMot", "LAud"]
            mock_lro.meta = Mock(f_s=1000, n_channels=3)
            mock_lro.file_durations = [duration_seconds]
            mock_recording = Mock()
            mock_recording.get_duration.return_value = duration_seconds
            mock_lro.LongRecording = mock_recording
            mock_lro.file_end_datetimes = [None]
            return mock_lro

        folder_durations_by_name = {
            "WT_A123_2023-01-15": 3600.0,  # 1 hour
            "WT_A123_2023-01-16": 1800.0,  # 30 minutes
            "WT_A123_2023-01-17": 7200.0,  # 2 hours
        }

        with patch.object(core, "LongRecordingOrganizer") as mock_lro_class:

            def mock_lro_side_effect(*args, **kwargs):
                item = args[0] if args else None
                item_str = str(item)
                for fname, dur in folder_durations_by_name.items():
                    if fname in item_str:
                        return create_mock_lro_with_duration(dur)
                return create_mock_lro_with_duration(3600.0)

            mock_lro_class.side_effect = mock_lro_side_effect

            # Global END time (not start)
            global_end = datetime(2023, 1, 15, 14, 30, 0)

            ao = results.AnimalOrganizer(
                pattern=str(self.base_path) + "/WT_{animal}_{session}",
                animal_id=self.animal_id,
                
                lro_kwargs={
                    "manual_datetimes": global_end,
                    "datetimes_are_start": False,
                },
            )

            assert len(ao._processed_timestamps) == 3

            # With datetimes_are_start=False, should work backwards
            # Total duration = 3600 + 1800 + 7200 = 12600s = 3.5 hours
            # So first folder should start at 14:30 - 3.5 hours = 11:00
            total_duration = sum(folder_durations_by_name.values())
            expected_first_start = global_end - timedelta(seconds=total_duration)

            sorted_folders = sorted(ao._processed_timestamps.keys())
            first_folder_start = ao._processed_timestamps[sorted_folders[0]]
            assert first_folder_start == expected_first_start, (
                f"First folder start mismatch: expected {expected_first_start}, got {first_folder_start}"
            )


    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_multi_item_animalday_passes_per_item_timestamp(
        self, mock_glob, mock_lro_class
    ):
        """Each individual LRO in a multi-item animalday gets its own timestamp.

        Regression test: previously the full list of timestamps was passed to
        every individual LRO, causing ``manual_datetimes length (N) must match
        number of input files (1)`` errors in SI mode.
        """
        # Setup: three folders that normalize into ONE animalday
        overlap_dir = self.base_path / "multi_item_test"
        overlap_dir.mkdir(parents=True, exist_ok=True)

        folder_a = overlap_dir / f"WT_{self.animal_id}_2023-01-15"
        folder_b = overlap_dir / f"WT_{self.animal_id}_2023-01-15(1)"
        folder_c = overlap_dir / f"WT_{self.animal_id}_2023-01-15(2)"

        for folder in [folder_a, folder_b, folder_c]:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "dummy_ColMajor_001.bin").touch()

        mock_glob.return_value = [str(folder_a), str(folder_b), str(folder_c)]

        # Track the kwargs each LRO receives
        lro_init_kwargs = []

        def mock_lro_side_effect(*args, **kwargs):
            lro_init_kwargs.append(kwargs.copy())
            mock_lro = Mock()
            mock_lro.channel_names = ["LMot", "RMot"]
            mock_lro.meta = Mock()
            mock_recording = Mock()
            mock_recording.get_num_samples.return_value = 100_000
            mock_recording.get_sampling_frequency.return_value = 1000.0
            mock_recording.get_duration.return_value = 100.0
            mock_lro.LongRecording = mock_recording
            base_time = datetime(2023, 1, 15, 8, 0, 0)
            mock_lro.file_end_datetimes = [base_time + timedelta(seconds=100)]
            mock_lro.merge = Mock()
            return mock_lro

        mock_lro_class.side_effect = mock_lro_side_effect

        # Single global start time — this is the scenario from the issue
        global_start = datetime(2023, 1, 15, 10, 0, 0)

        ao = results.AnimalOrganizer(
            pattern=str(overlap_dir) + "/WT_{animal}_{session}",
            animal_id=self.animal_id,
            lro_kwargs={"manual_datetimes": global_start},
            normalize_session=lambda s: re.sub(r"\(\d+\)$", "", s),
        )

        # Should produce 1 merged LRO (3 items collapsed into 1 animalday)
        assert len(ao.long_recordings) == 1

        # Each of the 3 individual LRO creations must receive a single
        # datetime, NOT a list of 3 timestamps.
        for kw in lro_init_kwargs:
            md = kw.get("manual_datetimes")
            assert not isinstance(md, list), (
                f"Individual LRO received a list of timestamps ({md!r}) instead "
                "of a single datetime; this would cause a length-mismatch error"
            )


class TestComputeGlobalTimelineNaturalSort:
    """Regression tests for natural-sort ordering in _compute_global_timeline.

    Issue: files named with numeric suffixes (e.g. MHET-0 through MHET-12) were
    ordered alphabetically ('MHET-10' < 'MHET-1_') instead of numerically
    (MHET-0, MHET-1, ..., MHET-12), causing manual timestamps to be assigned
    to the wrong recordings.
    """

    def test_natural_sort_order_with_numeric_suffixes(self):
        """Verify items are ordered numerically (0,1,...,12) not alphabetically (0,10,11,12,1,...)."""
        from neurodent.core.discovery import _natural_sort_key

        # Simulate 13 item names matching the real-world pattern from the bug report
        item_names = [f"MHET-{i}_ColMajor.bin..." for i in range(13)]

        # Alphabetical order differs from numerical: "MHET-10" < "MHET-1_"
        alphabetical_order = sorted(item_names)
        natural_order = sorted(item_names, key=_natural_sort_key)

        # Confirm the orders differ — this is the precondition for the bug
        assert alphabetical_order != natural_order, (
            "Precondition failed: alphabetical and natural orders should differ for these names"
        )

        # Build an animalday_to_items dict keyed by item name (as _compute_global_timeline receives)
        # Use simple string sentinels as "items"
        animalday_to_items = {name: [name] for name in item_names}

        # Create a minimal mock AnimalOrganizer that supports _compute_global_timeline
        with patch("neurodent.visualization.results.core.LongRecordingOrganizer") as mock_lro_class:
            mock_lro_class.side_effect = lambda item, **kwargs: _make_mock_lro(1800.0)

            ao = _make_minimal_ao()
            # Patch _get_item_name to return the item string itself (items are already strings)
            ao._get_item_name = lambda item: item

            base_dt = datetime(2025, 5, 10, 10, 0, 0)
            processed = ao._compute_global_timeline(
                base_dt,
                animalday_to_items,
                {},
                original_manual_datetimes=base_dt,
            )

        # Verify every item got a timestamp
        assert set(processed.keys()) == set(item_names)

        # Timestamps must increase in natural (numeric) order, not alphabetical order
        natural_sorted_names = sorted(item_names, key=_natural_sort_key)
        timestamps_in_natural_order = [processed[n] for n in natural_sorted_names]

        for i in range(len(timestamps_in_natural_order) - 1):
            assert timestamps_in_natural_order[i] < timestamps_in_natural_order[i + 1], (
                f"Timestamp for {natural_sorted_names[i]} ({timestamps_in_natural_order[i]}) "
                f"is not before {natural_sorted_names[i+1]} ({timestamps_in_natural_order[i+1]}). "
                "Items may be ordered alphabetically instead of numerically."
            )

        # The item after MHET-9 must be MHET-10, not MHET-1
        # (i.e. MHET-1 timestamp < MHET-10 timestamp)
        assert processed["MHET-1_ColMajor.bin..."] < processed["MHET-10_ColMajor.bin..."], (
            "MHET-1 should be assigned an earlier timestamp than MHET-10 (natural sort), "
            "but got the reverse (alphabetical sort bug)."
        )

        # The actual ordering in processed timestamps should match natural_order
        timestamps_sorted_by_value = sorted(processed.keys(), key=lambda k: processed[k])
        assert timestamps_sorted_by_value == natural_order, (
            f"Timestamps assigned in wrong order.\n"
            f"  Expected (natural): {natural_order}\n"
            f"  Got:                {timestamps_sorted_by_value}"
        )


class TestTimestampCollisionPrevention:
    """Regression tests for cross-session timestamp key collisions.

    Issue: _get_item_name() returns filename-only keys (e.g. 'file-0.bin'),
    so when two sessions contain files with the same name, the
    _processed_timestamps dict silently overwrites earlier entries.
    Fix: use _get_item_key() which returns the full path.
    """

    def test_get_item_key_returns_full_path_for_strings(self):
        ao = _make_minimal_ao()
        assert ao._get_item_key("/data/sess1/file-0.bin") == "/data/sess1/file-0.bin"
        assert ao._get_item_key("/data/sess2/file-0.bin") == "/data/sess2/file-0.bin"

    def test_get_item_key_returns_full_path_for_lists(self):
        ao = _make_minimal_ao()
        key = ao._get_item_key(["/data/sess1/file-0.bin", "/data/sess1/file-0.json"])
        assert key == "/data/sess1/file-0.bin"

    def test_get_item_key_distinct_for_same_filename(self):
        """Two items with the same filename in different directories get distinct keys."""
        ao = _make_minimal_ao()
        key1 = ao._get_item_key("/data/session_1/Cage 3 F9 Mut-0_ColMajor.bin")
        key2 = ao._get_item_key("/data/session_2/Cage 3 F9 Mut-0_ColMajor.bin")
        assert key1 != key2

    def test_get_item_name_would_collide(self):
        """Verify that _get_item_name DOES produce the same key for same-named files."""
        ao = _make_minimal_ao()
        name1 = ao._get_item_name("/data/session_1/Cage 3 F9 Mut-0_ColMajor.bin")
        name2 = ao._get_item_name("/data/session_2/Cage 3 F9 Mut-0_ColMajor.bin")
        assert name1 == name2  # This is the collision we're preventing

    def test_compute_global_timeline_uses_full_path_keys(self):
        """_compute_global_timeline result keys are full paths, not just filenames."""
        with patch("neurodent.visualization.results.core.LongRecordingOrganizer") as mock_lro_class:
            mock_lro_class.side_effect = lambda item, **kw: _make_mock_lro(100.0)

            ao = _make_minimal_ao()
            base_dt = datetime(2023, 9, 1, 10, 0, 0)

            # Two items with the same filename in different directories
            items = {
                "/data/sess1/file-0.bin": ["/data/sess1/file-0.bin"],
                "/data/sess2/file-0.bin": ["/data/sess2/file-0.bin"],
            }
            result = ao._compute_global_timeline(
                base_dt, items, {}, original_manual_datetimes=base_dt
            )

            # Keys must be full paths, not just "file-0.bin"
            assert len(result) == 2
            assert "/data/sess1/file-0.bin" in result
            assert "/data/sess2/file-0.bin" in result

    def test_session_keys_no_overwrite_across_sessions(self):
        """has_session_keys branch: items from different sessions get distinct timestamps.

        Regression test for the bug where out.update(sess_timeline) would
        overwrite session-1's file-0 timestamp with session-2's file-0 timestamp.
        """
        with patch("neurodent.visualization.results.core.LongRecordingOrganizer") as mock_lro_class:
            mock_lro_class.side_effect = lambda item, **kw: _make_mock_lro(100.0)

            ao = _make_minimal_ao()

            # Two sessions, each with one item sharing the same filename
            animalday_to_items = {
                "2022-09-03": ["/data/2022-09-03/Cage 3 F9 Mut-0_ColMajor.bin"],
                "2022-09-02": ["/data/2022-09-02/Cage 3 F9 Mut-0_ColMajor.bin"],
            }
            manual_datetimes = {
                "2022-09-03": datetime(2023, 9, 3, 9, 0, 0),
                "2022-09-02": datetime(2023, 9, 2, 15, 0, 0),
            }

            result = ao._process_manual_datetimes(
                manual_datetimes, animalday_to_items, {}
            )

            # Must have 2 distinct entries (not 1 due to collision)
            assert len(result) == 2

            # Keys must be full paths
            keys = list(result.keys())
            assert any("2022-09-03" in k for k in keys)
            assert any("2022-09-02" in k for k in keys)

            # Timestamps must be different
            timestamps = list(result.values())
            assert timestamps[0] != timestamps[1]


class TestValidateTimestampOrdering:
    """Tests for the _validate_timestamp_ordering static method."""

    def test_passes_for_distinct_ordered_timestamps(self):
        ts = {
            "/data/sess1/file-0.bin": datetime(2023, 1, 1, 10, 0, 0),
            "/data/sess2/file-0.bin": datetime(2023, 1, 2, 10, 0, 0),
        }
        results.AnimalOrganizer._validate_timestamp_ordering(ts)  # Should not raise

    def test_allows_equal_timestamps_for_zero_duration_files(self):
        """Equal timestamps are allowed (e.g., 0-duration empty files)."""
        same_time = datetime(2023, 1, 1, 10, 0, 0)
        ts = {
            "/data/sess1/file.bin": same_time,
            "/data/sess2/file.bin": same_time,
        }
        # Should not raise — 0-duration files legitimately share timestamps
        results.AnimalOrganizer._validate_timestamp_ordering(ts)

    def test_passes_for_single_item(self):
        ts = {"/data/file.bin": datetime(2023, 1, 1, 10, 0, 0)}
        results.AnimalOrganizer._validate_timestamp_ordering(ts)  # Should not raise

    def test_passes_for_empty_dict(self):
        results.AnimalOrganizer._validate_timestamp_ordering({})  # Should not raise

    def test_skips_non_datetime_values(self):
        """Non-datetime values (lists, functions) are not validated."""
        ts = {
            "/data/sess1/file.bin": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "/data/sess2/file.bin": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        }
        results.AnimalOrganizer._validate_timestamp_ordering(ts)  # Should not raise


class TestDiagnosticDisplayOffByOne:
    """Test that short-interval diagnostic display uses correct row indices."""

    def test_first_short_interval_shows_correct_rows(self):
        """When the first interval (row 0->1) is short, diagnostic should show rows 0 and 1, not row -1."""
        war = results.WindowAnalysisResult.__new__(results.WindowAnalysisResult)
        war.suppress_short_interval_error = False
        war.animal_id = "test"

        # Create a result DataFrame where the gap between row 0 and row 1 is
        # shorter than the median duration (triggering the diagnostic).
        war.result = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2023-01-01 10:00:00",
                "2023-01-01 10:00:01",  # 1s gap (short!)
                "2023-01-01 10:00:06",  # 5s gap (ok)
                "2023-01-01 10:00:11",  # 5s gap (ok)
            ]),
            "duration": [5.0, 5.0, 5.0, 5.0],
            "animal": ["test"] * 4,
        })

        # Should raise ValueError with diagnostic showing rows 0->1
        with pytest.raises(ValueError, match="Found 1 intervals") as exc_info:
            war._update_instance_vars()

        # The diagnostic should reference 10:00:00 -> 10:00:01 (not the last row)
        error_msg = str(exc_info.value)
        assert "10:00:00" in error_msg
        assert "10:00:01" in error_msg


def _make_mock_lro(duration_seconds=1800.0):
    """Return a minimal mock LRO with a fixed recording duration."""
    mock_lro = Mock()
    mock_lro.channel_names = ["ch1"]
    mock_lro.meta = Mock(f_s=1000, n_channels=1)
    mock_lro.file_durations = [duration_seconds]
    mock_recording = Mock()
    mock_recording.get_duration.return_value = duration_seconds
    mock_lro.LongRecording = mock_recording
    mock_lro.file_end_datetimes = [None]
    return mock_lro


def _make_minimal_ao():
    """Return a bare AnimalOrganizer-like object with the methods _compute_global_timeline needs."""
    ao = object.__new__(results.AnimalOrganizer)
    # Provide the minimum attributes used by _compute_global_timeline / _sort_lros_by_median_time
    ao.animal_id = "test_animal"
    return ao


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
