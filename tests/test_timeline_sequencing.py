
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from neurodent.visualization import AnimalOrganizer

class TestTimelineSequencing:
    @pytest.fixture
    def ao(self):
        # Create a dummy AnimalOrganizer. We only need partial functionality.
        ao = MagicMock(spec=AnimalOrganizer)
        ao.animal_id = "M1"
        # Bind the real method we want to test
        ao._process_manual_datetimes = AnimalOrganizer._process_manual_datetimes.__get__(ao, AnimalOrganizer)
        # Bind the timeline method too since we want to verify it gets used
        ao._compute_global_timeline = AnimalOrganizer._compute_global_timeline.__get__(ao, AnimalOrganizer)
        # Also need to bind _sort_lros_by_median_time because _compute_global_timeline uses it
        ao._sort_lros_by_median_time = AnimalOrganizer._sort_lros_by_median_time.__get__(ao, AnimalOrganizer)
        
        # Mock dependencies
        ao._resolve_timestamp_input = MagicMock(side_effect=lambda x, y: pd.to_datetime(x))
        ao._get_folders_for_animal = MagicMock(side_effect=lambda aid, mapping: [f for folders in mapping.values() for f in folders])
        
        return ao

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_scalar_string_sequencing(self, mock_lro_cls, ao):
        """
        Test that a single string timestamp results in a sequenced timeline
        rather than broadcasting the same timestamp to all folders.
        """
        # Configure mocked LRO - use side_effect to return new instance each time
        def side_effect(*args, **kwargs):
            mock_instance = MagicMock()
            mock_instance.LongRecording.get_duration.return_value = 3600.0
            return mock_instance

        mock_lro_cls.side_effect = side_effect

        # Setup: 2 distinct folders (days)
        folders = ["/data/M1_day1", "/data/M1_day2"]
        animalday_to_folders = {
            "M1_day1": ["/data/M1_day1"],
            "M1_day2": ["/data/M1_day2"]
        }
        
        manual_datetimes = "2025-01-01 12:00:00"
        base_lro_kwargs = {"datetimes_are_start": True}
        
        
        # Execute behavior
        # We expect the pipeline to correctly sequence the days based on duration
        # when a single start timestamp is provided.
        result = ao._process_manual_datetimes(manual_datetimes, animalday_to_folders, base_lro_kwargs)
        
        # Verification
        assert "M1_day1" in result
        assert "M1_day2" in result
        
        t1 = result["M1_day1"]
        t2 = result["M1_day2"] # _compute_global_timeline sorts folders by name
        
        print(f"T1: {t1}, T2: {t2}")
        
        # Ensure they are NOT equal (Broadcasting fix verification)
        assert t1 != t2, "Timestamps should not be identical for sequential days"
        
        # Ensure they are sequenced by duration (1 hour)
        # Note: _compute_global_timeline sorts by folder name. M1_day1 < M1_day2.
        assert t2 == t1 + timedelta(seconds=3600.0), f"Expected T2 to be T1 + 1h. Got {t2} vs {t1}"

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_complex_multiday_sequencing(self, mock_lro_cls, ao):
        """
        Test a scenario mimicking the user's data: 4 consecutive days with varying durations.
        Verifies that they form a contiguous timeline without overlaps.
        """
        # Setup: 4 days (Nov 28, 29, 30, Dec 01)
        folders = ["/data/Nov28", "/data/Nov29", "/data/Nov30", "/data/Dec01"]
        animalday_to_folders = {name: [path] for name, path in zip(["Day1", "Day2", "Day3", "Day4"], folders)}
        
        # Define variable durations for each "day"
        # Day 1: 10 hours
        # Day 2: 100 hours (long recording)
        # Day 3: 5 hours
        # Day 4: 10 hours
        durations = {
            "/data/Nov28": 3600.0 * 10,
            "/data/Nov29": 3600.0 * 100,
            "/data/Nov30": 3600.0 * 5,
            "/data/Dec01": 3600.0 * 10,
        }
        
        # Configure Mock LRO to return specific durations based on input folder
        def side_effect(folder, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = durations.get(str(folder), 3600.0)
            return m
        
        mock_lro_cls.side_effect = side_effect

        manual_datetimes = "2025-11-28 12:00:00"
        base_lro_kwargs = {"datetimes_are_start": True}
        
        # Execute
        result = ao._process_manual_datetimes(manual_datetimes, animalday_to_folders, base_lro_kwargs)
        
        # Timepoints
        # Keys are folder names (basenames)
        t1 = result["Nov28"]
        t2 = result["Nov29"]
        t3 = result["Nov30"]
        t4 = result["Dec01"]
        
        print(f"T1: {t1}")
        print(f"T2: {t2}")
        print(f"T3: {t3}")
        print(f"T4: {t4}")
        
        # Verify Sequencing
        # T2 should start after T1 ends (T1 + 10h)
        assert t2 == t1 + timedelta(hours=10)
        
        # T3 should start after T2 ends (T2 + 100h)
        assert t3 == t2 + timedelta(hours=100)
        
        # T4 should start after T3 ends (T3 + 5h)
        assert t4 == t3 + timedelta(hours=5)
        
        # Verify total span
        total_span = t4 + timedelta(hours=10) - t1
        expected_span = timedelta(hours=10 + 100 + 5 + 10)
        assert total_span == expected_span

    def test_explicit_dict_mapping(self, ao):
        """
        Verify that passing an explicit dictionary mapping (legacy/power-user mode)
        still works and is NOT treated as a sequence generation request.
        """
        animalday_to_folders = {
            "Day1": ["/data/folder1"],
            "Day2": ["/data/folder2"]
        }
        
        # User explicitly says folder1 starts at 12:00 and folder2 starts at 14:00
        # Keys must match folder names for backward compatibility mode
        manual_datetimes = {
            "folder1": "2025-01-01 12:00:00",
            "folder2": "2025-01-01 14:00:00"
        }
        base_lro_kwargs = {}
        
        # Execute
        result = ao._process_manual_datetimes(manual_datetimes, animalday_to_folders, base_lro_kwargs)
        
        # Verification
        # Keys in result are folder names
        assert result["folder1"] == pd.to_datetime("2025-01-01 12:00:00")
        assert result["folder2"] == pd.to_datetime("2025-01-01 14:00:00")


