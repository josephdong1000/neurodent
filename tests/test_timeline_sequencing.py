
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from neurodent.visualization import AnimalOrganizer
from neurodent.core.discovery import DiscoveredFile

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
        # Bind helpers that _compute_global_timeline uses
        ao._get_item_name = AnimalOrganizer._get_item_name.__get__(ao, AnimalOrganizer)
        ao._is_item_file = AnimalOrganizer._is_item_file.__get__(ao, AnimalOrganizer)
        
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

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_session_keyed_dict(self, mock_lro_cls, ao):
        """
        Test that a dict keyed by session names (matching animalday_to_items keys)
        is correctly handled — each session gets its own timestamp as the start time
        and a timeline is computed for items within each session.

        This is the format used when per-animal manual_datetime is a dict in the
        unified config, e.g.:
            "manual_datetime": {
                "010822_cohort4_group2_M3_MHET_files0-12": "2022-01-08 18:55:02",
                "010822_cohort4_group2_M3_MHET_files13-21": "2022-01-08 23:25:03"
            }
        """
        # Mock LRO so _compute_global_timeline can estimate durations
        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = 3600.0
            return m

        mock_lro_cls.side_effect = side_effect

        # Each session has one item (folder), session keys match manual_datetimes keys
        animalday_to_items = {
            "010822_files0-12": ["/data/010822_files0-12"],
            "010822_files13-21": ["/data/010822_files13-21"],
        }

        manual_datetimes = {
            "010822_files0-12": "2022-01-08 18:55:02",
            "010822_files13-21": "2022-01-08 23:25:03",
        }
        base_lro_kwargs = {"datetimes_are_start": True}

        result = ao._process_manual_datetimes(
            manual_datetimes, animalday_to_items, base_lro_kwargs
        )

        # Each session's single item should get a timeline-computed timestamp
        assert "010822_files0-12" in result
        assert "010822_files13-21" in result
        # Start time matches the session's manual timestamp
        assert result["010822_files0-12"] == pd.to_datetime("2022-01-08 18:55:02")
        assert result["010822_files13-21"] == pd.to_datetime("2022-01-08 23:25:03")

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_session_keyed_dict_multi_item_sessions(self, mock_lro_cls, ao):
        """
        Test session-keyed dict when sessions contain multiple items.
        Each session's timestamp is used as the start for its items,
        and a per-session timeline is computed.
        """
        durations = {
            "/data/sess1/file_a": 1800.0,
            "/data/sess1/file_b": 1800.0,
            "/data/sess2/file_c": 3600.0,
        }

        def side_effect(folder, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = durations.get(str(folder), 3600.0)
            return m

        mock_lro_cls.side_effect = side_effect

        animalday_to_items = {
            "session1": ["/data/sess1/file_a", "/data/sess1/file_b"],
            "session2": ["/data/sess2/file_c"],
        }

        manual_datetimes = {
            "session1": "2022-01-08 10:00:00",
            "session2": "2022-01-08 14:00:00",
        }
        base_lro_kwargs = {"datetimes_are_start": True}

        result = ao._process_manual_datetimes(
            manual_datetimes, animalday_to_items, base_lro_kwargs
        )

        # session1 items: file_a at 10:00, file_b at 10:00 + 1800s = 10:30
        assert result["file_a"] == pd.to_datetime("2022-01-08 10:00:00")
        assert result["file_b"] == pd.to_datetime("2022-01-08 10:30:00")
        # session2 item: file_c at 14:00
        assert result["file_c"] == pd.to_datetime("2022-01-08 14:00:00")

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_datetimes_are_start_false_sequencing(self, mock_lro_cls, ao):
        """
        Regression test: when datetimes_are_start=False, _compute_global_timeline
        works backwards from end times to produce start times. Verify that the
        returned timestamps are correct start times and form a non-overlapping
        sequence.

        This catches a bug where _compute_global_timeline always returns start
        times, but _create_long_recordings passed them downstream with the
        original datetimes_are_start=False flag — causing LROs to misinterpret
        start times as end times and creating massive timestamp overlaps.
        """
        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = 3600.0  # 1 hour
            return m

        mock_lro_cls.side_effect = side_effect

        # Two sessions, each with one folder. End times provided.
        animalday_to_items = {
            "session1": ["/data/sess1/folder_a"],
            "session2": ["/data/sess2/folder_b"],
        }

        # These are END times (datetimes_are_start=False)
        manual_datetimes = {
            "session1": "2025-01-01 13:00:00",  # ends at 13:00
            "session2": "2025-01-01 15:00:00",  # ends at 15:00
        }
        base_lro_kwargs = {"datetimes_are_start": False}

        result = ao._process_manual_datetimes(
            manual_datetimes, animalday_to_items, base_lro_kwargs
        )

        # _compute_global_timeline should return START times computed from end times
        # session1: start = 13:00 - 1h = 12:00
        # session2: start = 15:00 - 1h = 14:00
        assert result["folder_a"] == pd.to_datetime("2025-01-01 12:00:00")
        assert result["folder_b"] == pd.to_datetime("2025-01-01 14:00:00")

        # Timestamps must be sequenced (no overlap)
        assert result["folder_b"] > result["folder_a"]

        # Gap between computed start times is preserved (12:00 → 14:00 = 2 hours)
        gap = result["folder_b"] - result["folder_a"]
        assert gap == timedelta(hours=2)

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_create_long_recordings_forces_datetimes_are_start_true(
        self, mock_lro_cls, ao
    ):
        """
        Regression test for the actual bug fix in _create_long_recordings:
        when _processed_timestamps is set, downstream LongRecordingOrganizer
        must always receive datetimes_are_start=True, even if the original
        lro_kwargs had datetimes_are_start=False (end-time inputs).

        This verifies the fix in AnimalOrganizer._create_long_recordings that
        sets kwargs["datetimes_are_start"] = True when injecting
        _processed_timestamps, since _compute_global_timeline always produces
        start times regardless of the original datetimes_are_start value.
        """
        mock_lro_instance = MagicMock()
        mock_lro_cls.return_value = mock_lro_instance

        # Bind _create_long_recordings as a real method on the mock AO
        ao._create_long_recordings = AnimalOrganizer._create_long_recordings.__get__(
            ao, AnimalOrganizer
        )

        # Pre-populate state that _create_long_recordings reads
        ao._animalday_folder_groups = {
            "session1": ["/data/sess1/folder_a"],
            "session2": ["/data/sess2/folder_b"],
        }
        ao.unique_animaldays = ["session1", "session2"]
        # _processed_timestamps holds start times (output of _compute_global_timeline)
        ao._processed_timestamps = {
            "folder_a": pd.to_datetime("2025-01-01 12:00:00"),
            "folder_b": pd.to_datetime("2025-01-01 14:00:00"),
        }

        # Simulate the bug scenario: original lro_kwargs had datetimes_are_start=False
        lro_kwargs = {"datetimes_are_start": False}

        ao._create_long_recordings(lro_kwargs)

        # Both LRO constructor calls must receive datetimes_are_start=True,
        # not the original False — because _processed_timestamps are always start times.
        assert mock_lro_cls.call_count == 2
        for call in mock_lro_cls.call_args_list:
            _, kwargs = call
            assert kwargs.get("datetimes_are_start") is True, (
                f"Expected datetimes_are_start=True but got {kwargs.get('datetimes_are_start')!r}. "
                "LongRecordingOrganizer must receive start times when _processed_timestamps is used."
            )

    def test_session_keyed_dict_missing_session_raises(self, ao):
        """
        Test that a session-keyed dict raises ValueError if some sessions
        are missing from the manual_datetimes dict.
        """
        animalday_to_items = {
            "session1": ["/data/sess1/file_a"],
            "session2": ["/data/sess2/file_b"],
        }

        # Only one session covered
        manual_datetimes = {
            "session1": "2022-01-08 10:00:00",
        }
        base_lro_kwargs = {}

        with pytest.raises(ValueError, match="Missing entries in manual_datetimes for sessions"):
            ao._process_manual_datetimes(
                manual_datetimes, animalday_to_items, base_lro_kwargs
            )

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_index_metadata_sort_scalar_datetime(self, mock_lro_cls, ao):
        """
        When items carry {index} metadata (from DiscoveredFile), sorting within
        a session should use the index value, not the filename.

        Here filenames sort differently from indices:
        - rec_A_data.bin (index="1") should come first
        - rec_B_data.bin (index="2") should come second
        - rec_C_data.bin (index="3") should come third
        """
        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = 3600.0
            return m

        mock_lro_cls.side_effect = side_effect

        # DiscoveredFile items where filename order != index order
        items = [
            DiscoveredFile(path="/data/day1/rec_C_data.bin", metadata={"animal": "M1", "session": "day1", "index": "3"}),
            DiscoveredFile(path="/data/day1/rec_A_data.bin", metadata={"animal": "M1", "session": "day1", "index": "1"}),
            DiscoveredFile(path="/data/day1/rec_B_data.bin", metadata={"animal": "M1", "session": "day1", "index": "2"}),
        ]

        animalday_to_items = {"day1": items}
        base_datetime = pd.to_datetime("2025-01-01 12:00:00")
        base_lro_kwargs = {"datetimes_are_start": True}

        result = ao._compute_global_timeline(
            base_datetime, animalday_to_items, base_lro_kwargs,
            original_manual_datetimes=base_datetime,
        )

        # Items should be ordered by index (1, 2, 3), not filename (A, B, C)
        assert result["rec_A_data.bin"] == pd.to_datetime("2025-01-01 12:00:00")  # index 1
        assert result["rec_B_data.bin"] == pd.to_datetime("2025-01-01 13:00:00")  # index 2
        assert result["rec_C_data.bin"] == pd.to_datetime("2025-01-01 14:00:00")  # index 3

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_index_metadata_sort_list_datetime(self, mock_lro_cls, ao):
        """
        When original_manual_datetimes is a list and items carry {index} metadata,
        items within each session should be sorted by index before being paired
        with the list timestamps.
        """
        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = 1800.0
            return m

        mock_lro_cls.side_effect = side_effect

        items = [
            DiscoveredFile(path="/data/day1/rec_B.bin", metadata={"animal": "M1", "session": "day1", "index": "2"}),
            DiscoveredFile(path="/data/day1/rec_A.bin", metadata={"animal": "M1", "session": "day1", "index": "1"}),
        ]

        animalday_to_items = {"day1": items}
        base_datetime = pd.to_datetime("2025-01-01 10:00:00")
        base_lro_kwargs = {"datetimes_are_start": True}
        # One timestamp per item, paired in sorted order
        list_datetimes = ["2025-01-01 10:00:00", "2025-01-01 11:00:00"]

        result = ao._compute_global_timeline(
            base_datetime, animalday_to_items, base_lro_kwargs,
            original_manual_datetimes=list_datetimes,
        )

        # index=1 (rec_A.bin) should pair with first timestamp,
        # index=2 (rec_B.bin) with second
        # Timeline uses base_datetime + cumulative durations (1800s each)
        assert result["rec_A.bin"] == pd.to_datetime("2025-01-01 10:00:00")
        assert result["rec_B.bin"] == pd.to_datetime("2025-01-01 10:30:00")

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_index_metadata_sort_no_manual_datetimes(self, mock_lro_cls, ao):
        """
        When no manual_datetimes are provided and items carry {index} metadata,
        sorting within a session should use the index value instead of falling
        back to LRO median-time sorting.
        """
        durations = {
            "/data/day1/rec_Z.bin": 1800.0,
            "/data/day1/rec_A.bin": 3600.0,
            "/data/day1/rec_M.bin": 900.0,
        }

        def side_effect(folder, **kwargs):
            import os as _os
            m = MagicMock()
            m.LongRecording.get_duration.return_value = durations.get(_os.fspath(folder), 3600.0)
            return m

        mock_lro_cls.side_effect = side_effect

        items = [
            DiscoveredFile(path="/data/day1/rec_Z.bin", metadata={"animal": "M1", "session": "day1", "index": "2"}),
            DiscoveredFile(path="/data/day1/rec_A.bin", metadata={"animal": "M1", "session": "day1", "index": "3"}),
            DiscoveredFile(path="/data/day1/rec_M.bin", metadata={"animal": "M1", "session": "day1", "index": "1"}),
        ]

        animalday_to_items = {"day1": items}
        base_datetime = pd.to_datetime("2025-01-01 08:00:00")
        base_lro_kwargs = {"datetimes_are_start": True}

        result = ao._compute_global_timeline(
            base_datetime, animalday_to_items, base_lro_kwargs,
            original_manual_datetimes=None,
        )

        # Sorted by index: rec_M (1, 900s), rec_Z (2, 1800s), rec_A (3, 3600s)
        assert result["rec_M.bin"] == pd.to_datetime("2025-01-01 08:00:00")       # index 1, duration 900s
        assert result["rec_Z.bin"] == pd.to_datetime("2025-01-01 08:15:00")       # index 2, duration 1800s
        assert result["rec_A.bin"] == pd.to_datetime("2025-01-01 08:45:00")       # index 3, duration 3600s

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_no_index_metadata_falls_back_to_filename(self, mock_lro_cls, ao):
        """
        When items don't carry {index} metadata (plain strings/paths), sorting
        should fall back to filename-based natural sort (existing behavior).
        """
        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = 3600.0
            return m

        mock_lro_cls.side_effect = side_effect

        # Plain string paths (no metadata)
        animalday_to_items = {
            "day1": ["/data/day1/file_2", "/data/day1/file_1", "/data/day1/file_10"],
        }
        base_datetime = pd.to_datetime("2025-01-01 12:00:00")
        base_lro_kwargs = {"datetimes_are_start": True}

        result = ao._compute_global_timeline(
            base_datetime, animalday_to_items, base_lro_kwargs,
            original_manual_datetimes=base_datetime,
        )

        # Natural sort: file_1 < file_2 < file_10
        assert result["file_1"] == pd.to_datetime("2025-01-01 12:00:00")
        assert result["file_2"] == pd.to_datetime("2025-01-01 13:00:00")
        assert result["file_10"] == pd.to_datetime("2025-01-01 14:00:00")

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    def test_index_metadata_sort_across_sessions(self, mock_lro_cls, ao):
        """
        Verify that index-based sorting is applied independently per session,
        and overlapping index ranges across sessions don't cause interleaving.
        """
        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = 1800.0
            return m

        mock_lro_cls.side_effect = side_effect

        animalday_to_items = {
            "session1": [
                DiscoveredFile(path="/data/s1/rec_B.bin", metadata={"animal": "M1", "session": "session1", "index": "2"}),
                DiscoveredFile(path="/data/s1/rec_A.bin", metadata={"animal": "M1", "session": "session1", "index": "1"}),
            ],
            "session2": [
                DiscoveredFile(path="/data/s2/rec_D.bin", metadata={"animal": "M1", "session": "session2", "index": "2"}),
                DiscoveredFile(path="/data/s2/rec_C.bin", metadata={"animal": "M1", "session": "session2", "index": "1"}),
            ],
        }
        base_datetime = pd.to_datetime("2025-01-01 10:00:00")
        base_lro_kwargs = {"datetimes_are_start": True}

        result = ao._compute_global_timeline(
            base_datetime, animalday_to_items, base_lro_kwargs,
            original_manual_datetimes=base_datetime,
        )

        # Session1 sorted by index: rec_A (1) then rec_B (2)
        # Session2 sorted by index: rec_C (1) then rec_D (2)
        # Sessions are processed in order: session1 then session2
        assert result["rec_A.bin"] == pd.to_datetime("2025-01-01 10:00:00")
        assert result["rec_B.bin"] == pd.to_datetime("2025-01-01 10:30:00")
        assert result["rec_C.bin"] == pd.to_datetime("2025-01-01 11:00:00")
        assert result["rec_D.bin"] == pd.to_datetime("2025-01-01 11:30:00")

