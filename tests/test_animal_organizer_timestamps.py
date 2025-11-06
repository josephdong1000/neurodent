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
        mock_glob.return_value = [str(self.folder1), str(self.folder2), str(self.folder3)]

        with patch.object(core, "LongRecordingOrganizer") as mock_lro_class:
            mock_lro_class.return_value = self._create_mock_lro()

            global_start = datetime(2023, 1, 15, 10, 0, 0)

            # Create AnimalOrganizer with single datetime
            ao = results.AnimalOrganizer(
                base_folder_path=str(self.base_path),
                anim_id=self.animal_id,
                mode="concat",
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
        mock_glob.return_value = [str(self.folder1), str(self.folder2), str(self.folder3)]
        mock_lro_class.return_value = self._create_mock_lro()

        datetime_list = [
            datetime(2023, 1, 15, 10, 0, 0),
            datetime(2023, 1, 16, 11, 0, 0),
            datetime(2023, 1, 17, 12, 0, 0),
        ]

        # Create AnimalOrganizer with datetime list
        ao = results.AnimalOrganizer(
            base_folder_path=str(self.base_path),
            anim_id=self.animal_id,
            mode="concat",
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
        mock_glob.return_value = [str(self.folder1), str(self.folder2), str(self.folder3)]
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
            base_folder_path=str(self.base_path),
            anim_id=self.animal_id,
            mode="concat",
            lro_kwargs={"manual_datetimes": extract_timestamp_from_folder},
        )

        # Verify processing
        assert len(ao._processed_timestamps) == 3

        # Check that function was applied to each folder
        expected_times = {
            f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 15, 9, 0, 0),
            f"WT_{self.animal_id}_2023-01-16": datetime(2023, 1, 16, 10, 0, 0),
            f"WT_{self.animal_id}_2023-01-17": datetime(2023, 1, 17, 11, 0, 0),
        }

        for folder_name, expected_time in expected_times.items():
            assert ao._processed_timestamps[folder_name] == expected_time

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_mixed_dictionary_specification(self, mock_glob, mock_lro_class):
        """Test dictionary with mixed function and explicit timestamp specification."""
        # Setup
        mock_glob.return_value = [str(self.folder1), str(self.folder2), str(self.folder3)]
        mock_lro_class.return_value = self._create_mock_lro()

        def extract_for_folder2(folder_path):
            return datetime(2023, 1, 16, 14, 30, 0)

        mixed_spec = {
            f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 15, 8, 0, 0),
            f"WT_{self.animal_id}_2023-01-16": extract_for_folder2,
            f"WT_{self.animal_id}_2023-01-17": [datetime(2023, 1, 17, 10, 0, 0), datetime(2023, 1, 17, 14, 0, 0)],
        }

        # Create AnimalOrganizer with mixed dictionary
        ao = results.AnimalOrganizer(
            base_folder_path=str(self.base_path),
            anim_id=self.animal_id,
            mode="concat",
            lro_kwargs={"manual_datetimes": mixed_spec},
        )

        # Verify processing
        assert len(ao._processed_timestamps) == 3

        # Check explicit datetime
        assert ao._processed_timestamps[f"WT_{self.animal_id}_2023-01-15"] == datetime(2023, 1, 15, 8, 0, 0)

        # Check function result
        assert ao._processed_timestamps[f"WT_{self.animal_id}_2023-01-16"] == datetime(2023, 1, 16, 14, 30, 0)

        # Check list
        expected_list = [datetime(2023, 1, 17, 10, 0, 0), datetime(2023, 1, 17, 14, 0, 0)]
        assert ao._processed_timestamps[f"WT_{self.animal_id}_2023-01-17"] == expected_list

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_invalid_timestamp_type_error(self, mock_glob, mock_lro_class):
        """Test that invalid timestamp types raise appropriate errors."""
        # Setup
        mock_glob.return_value = [str(self.folder1)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Test invalid type (string instead of datetime)
        with pytest.raises(TypeError) as exc_info:
            results.AnimalOrganizer(
                base_folder_path=str(self.base_path),
                anim_id=self.animal_id,
                mode="concat",
                lro_kwargs={"manual_datetimes": "2023-01-15 10:00:00"},  # String instead of datetime
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
                base_folder_path=str(self.base_path),
                anim_id=self.animal_id,
                mode="concat",
                lro_kwargs={"manual_datetimes": invalid_list},
            )

        assert "All items in timestamp list must be datetime objects" in str(exc_info.value)

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
                base_folder_path=str(self.base_path),
                anim_id=self.animal_id,
                mode="concat",
                lro_kwargs={"manual_datetimes": failing_function},
            )

        error_str = str(exc_info.value)
        assert "User timestamp function failed" in error_str
        assert "Simulated extraction failure" in error_str

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_missing_folder_in_dictionary_error(self, mock_glob, mock_lro_class):
        """Test that missing folders in dictionary specification raise errors."""
        # Setup
        mock_glob.return_value = [str(self.folder1), str(self.folder2)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Dictionary with nonexistent folder
        incomplete_spec = {
            f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 15, 10, 0, 0),
            f"WT_{self.animal_id}_2023-01-16": datetime(2023, 1, 16, 10, 0, 0),
            "NonexistentFolder": datetime(2023, 1, 17, 10, 0, 0),  # This folder doesn't exist
        }

        with pytest.raises(ValueError) as exc_info:
            results.AnimalOrganizer(
                base_folder_path=str(self.base_path),
                anim_id=self.animal_id,
                mode="concat",
                lro_kwargs={"manual_datetimes": incomplete_spec},
            )

        error_str = str(exc_info.value)
        assert "Folder name" in error_str and "not found" in error_str

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_backward_compatibility_no_manual_datetimes(self, mock_glob, mock_lro_class):
        """Test that AnimalOrganizer works without manual_datetimes (backward compatibility)."""
        # Setup
        mock_glob.return_value = [str(self.folder1), str(self.folder2)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Create AnimalOrganizer without manual_datetimes
        ao = results.AnimalOrganizer(base_folder_path=str(self.base_path), anim_id=self.animal_id, mode="concat")

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
            base_folder_path=str(self.base_path),
            anim_id=self.animal_id,
            mode="concat",
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
            base_folder_path=str(self.base_path),
            anim_id=self.animal_id,
            mode="concat",
            lro_kwargs={"manual_datetimes": outer_function},
        )

        # Verify that recursive resolution worked
        folder_name = f"WT_{self.animal_id}_2023-01-15"
        assert ao._processed_timestamps[folder_name] == datetime(2023, 1, 15, 12, 0, 0)

    @patch("glob.glob")
    def test_continuous_timeline_single_datetime(self, mock_glob):
        """Test that single datetime creates continuous (non-overlapping) timeline."""
        # Setup
        mock_glob.return_value = [str(self.folder1), str(self.folder2), str(self.folder3)]

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

        # Define durations for each folder (in seconds)
        folder_durations = {
            str(self.folder1): 3600.0,  # 1 hour
            str(self.folder2): 1800.0,  # 30 minutes
            str(self.folder3): 7200.0,  # 2 hours
        }

        with patch.object(core, "LongRecordingOrganizer") as mock_lro_class:

            def mock_lro_side_effect(*args, **kwargs):
                folder_path = str(args[0])
                duration = folder_durations.get(folder_path, 3600.0)
                return create_mock_lro_with_duration(duration)

            mock_lro_class.side_effect = mock_lro_side_effect

            global_start = datetime(2023, 1, 15, 10, 0, 0)

            # Create AnimalOrganizer with single datetime
            ao = results.AnimalOrganizer(
                base_folder_path=str(self.base_path),
                anim_id=self.animal_id,
                mode="concat",
                lro_kwargs={"manual_datetimes": global_start},
            )

            # Verify continuous timeline
            assert len(ao._processed_timestamps) == 3

            # Convert folder paths to folder names for lookup
            folder_name_to_path = {
                Path(path).name: path for path in [str(self.folder1), str(self.folder2), str(self.folder3)]
            }

            # Expected timeline (continuous, non-overlapping)
            expected_timeline = {}
            current_time = global_start

            # Process folders in sorted order (by animalday, then by folder order)
            for folder_name in sorted(ao._processed_timestamps.keys()):
                expected_timeline[folder_name] = current_time
                folder_path = folder_name_to_path[folder_name]
                duration = folder_durations[folder_path]
                current_time = current_time + timedelta(seconds=duration)

            # Verify continuous timeline
            for folder_name, expected_start in expected_timeline.items():
                actual_start = ao._processed_timestamps[folder_name]
                assert actual_start == expected_start, (
                    f"Folder {folder_name}: expected {expected_start}, got {actual_start}"
                )

            # Verify no temporal overlaps
            timeline_list = [(name, time) for name, time in ao._processed_timestamps.items()]
            timeline_list.sort(key=lambda x: x[1])  # Sort by start time

            for i in range(len(timeline_list) - 1):
                current_folder, current_start = timeline_list[i]
                next_folder, next_start = timeline_list[i + 1]

                # Calculate end time of current folder
                current_path = folder_name_to_path[current_folder]
                current_duration = folder_durations[current_path]
                current_end = current_start + timedelta(seconds=current_duration)

                # Verify next folder starts exactly when current folder ends
                assert current_end == next_start, (
                    f"Gap/overlap between {current_folder} and {next_folder}: {current_folder} ends at {current_end}, {next_folder} starts at {next_start}"
                )

            logging.info("✅ Continuous timeline verified: folders are sequential with no gaps or overlaps")

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
        expected_median_times = [100.0, 50.0, 150.0]  # Out of name order but chronological

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
            base_folder_path=str(overlap_dir),
            anim_id=self.animal_id,
            mode="concat",
            lro_kwargs={"manual_datetimes": folder_timestamps},
        )

        # Should have 1 LRO (merged from overlapping folders)
        assert len(ao.long_recordings) == 1
        assert len(ao.animaldays) == 1

        # Verify all folders were processed with their timestamps
        assert len(ao._processed_timestamps) == 3
        for folder_name, expected_time in folder_timestamps.items():
            assert ao._processed_timestamps[folder_name] == expected_time

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
            ao._resolve_timestamp_input("invalid", test_folder)
        assert "Invalid timestamp input type" in str(exc_info.value)

        # Test invalid list items
        invalid_list = [datetime(2023, 1, 15, 10, 0, 0), "not datetime"]
        with pytest.raises(TypeError) as exc_info:
            ao._resolve_timestamp_input(invalid_list, test_folder)
        assert "All items in timestamp list must be datetime objects" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
