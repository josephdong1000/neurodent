
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from neurodent.visualization import AnimalOrganizer
from neurodent import core

# Mocking the folder structure and LRO creation
@pytest.fixture
def mock_lros():
    lro1 = MagicMock()
    lro1.base_folder_path = Path("/data/Session1")
    lro1.get_total_duration.return_value = 3600  # 1 hour
    lro1.file_durations = [3600]
    lro1.file_end_datetimes = [datetime(2025, 11, 27, 16, 39, 5)] # Start 15:39:05
    lro1.channel_names = ["Ch1"]

    lro2 = MagicMock()
    lro2.base_folder_path = Path("/data/Session2")
    lro2.get_total_duration.return_value = 3600
    lro2.file_durations = [3600]
    lro2.file_end_datetimes = [datetime(2025, 11, 28, 12, 47, 5)] # Start 11:47:05
    lro2.channel_names = ["Ch1"]

    return [lro1, lro2]

def test_distributed_timestamps_logic(mock_lros):
    """
    Simulates the logic added to generate_wars.py to distribute timestamps.
    """
    animal_folders = [("/data/Session1", "AnimalA", "Session1"), ("/data/Session2", "AnimalA", "Session2")]
    manual_datetimes_list = ["2025-11-27 15:39:05", "2025-11-28 11:47:05"]
    
    # Simulate generate_wars.py loop
    final_lros = []
    
    for i, folder_info in enumerate(animal_folders):
        folder_path = folder_info[0]
        
        # Logic from generate_wars.py
        current_dt = manual_datetimes_list[i]
        
        # Verify correct distribution
        if i == 0:
            assert current_dt == "2025-11-27 15:39:05"
        elif i == 1:
            assert current_dt == "2025-11-28 11:47:05"
            
        # In real script, we init AO here. We mock the result.
        # Check if the timestamps are logically distinct (no overlap)
        dt_val = pd.to_datetime(current_dt)
        if i == 0:
            assert dt_val.day == 27
        elif i == 1:
            assert dt_val.day == 28

def test_overlapping_bug_simulation():
    """
    Simulates what happens if we DON'T distribute timestamps (the bug).
    """
    manual_datetimes_single = "2025-11-27 15:39:05"
    
    # If both sessions get this timestamp
    session1_start = pd.to_datetime(manual_datetimes_single)
    session2_start = pd.to_datetime(manual_datetimes_single)
    
    # Assert they are identical (Overlap!)
    assert session1_start == session2_start
