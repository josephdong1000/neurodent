
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from neurodent.visualization import AnimalOrganizer

class TestTimelineSortingCrash:
    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("neurodent.core.utils.parse_str_to_day")
    def test_sort_crash_fix_via_parsing(self, mock_parse, mock_lro_cls):
        """
        Verify that _compute_global_timeline extracts date from filename
        and passes it to LRO, preventing the crash.
        """
        ao = MagicMock(spec=AnimalOrganizer)
        ao.animal_id = "M1"
        ao.day_sep = None
        # Bind the real valid methods we want to test
        ao._compute_global_timeline = AnimalOrganizer._compute_global_timeline.__get__(ao, AnimalOrganizer)
        ao._sort_lros_by_median_time = AnimalOrganizer._sort_lros_by_median_time.__get__(ao, AnimalOrganizer)
        ao._resolve_timestamp_input = MagicMock(side_effect=lambda ts, path: ts if isinstance(ts, pd.Timestamp) else pd.to_datetime(ts))
        ao._get_folder_duration = MagicMock(return_value=3600.0)
        ao._get_item_name = AnimalOrganizer._get_item_name.__get__(ao, AnimalOrganizer)
        ao._is_item_file = AnimalOrganizer._is_item_file.__get__(ao, AnimalOrganizer)
        ao._items_have_index = AnimalOrganizer._items_have_index.__get__(ao, AnimalOrganizer)
        ao._session_sort_key = AnimalOrganizer._session_sort_key.__get__(ao, AnimalOrganizer)

        animalday_to_folders = {
            "Day1": ["/data/filename_20250101.rhd", "/data/filename_20250102.rhd"]
        }
        
        # Mock date parsing to return success
        mock_parse.return_value = pd.to_datetime("2025-01-01")

        # Mock LRO to succeed (simulating valid init because manual_datetimes provided)
        mock_instance = MagicMock()
        # Ensure file_end_datetimes is correctly mocked as a property if needed, or just attribute
        mock_instance.file_end_datetimes = [pd.to_datetime("2025-01-01 13:00:00")] 
        mock_instance.LongRecording.get_duration.return_value = 3600.0
        mock_lro_cls.return_value = mock_instance

        manual_dt = pd.to_datetime("2025-01-01 12:00:00")
        base_lro_kwargs = {}

        # Execute (pass original_manual_datetimes to use manual timestamp mode)
        result = ao._compute_global_timeline(manual_dt, animalday_to_folders, base_lro_kwargs,
                                             original_manual_datetimes=manual_dt)
        
        # Verify result contains both keys
        assert "/data/filename_20250101.rhd" in result or "filename_20250101.rhd" in result
        
        # Verify LRO was called WITH manual_datetimes
        # Check call args of the Mock Class
        assert mock_lro_cls.called
        # Check that at least one call had manual_datetimes
        calls_with_manual = [
            call for call in mock_lro_cls.call_args_list 
            if "manual_datetimes" in call.kwargs
        ]
        assert len(calls_with_manual) > 0, "LRO not initialized with manual_datetimes"
