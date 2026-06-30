
import pytest
from pathlib import Path
from datetime import datetime
import neurodent.visualization.results as results
import neurodent.constants as constants

# Mock data based on logs
HOMO_FILES = [
    "AP3B2homo-240-M_251127_153905.nwb",
    "AP3B2homo-240-M_251127_233906.nwb",
    "AP3B2homo-240-M_251128_013907.nwb", # Should be Nov 28 (or 27 if late?)
    "AP3B2homo-240-M_251128_040907.nwb",
    "AP3B2homo-240-M_251128_113908.nwb", # Distinctly Nov 28 noon
]

MOCK_ALIASES = {
    "HOM": ["AP3B2homo-240-M"],
    "WT": ["AP3B2wt-241-M"]
}

@pytest.fixture
def mock_overlap_structure(tmp_path):
    data_dir = tmp_path / "Data"
    data_dir.mkdir()
    
    # Nest files in a subfolder matching the ID to satisfy AO glob
    anim_dir = data_dir / "AP3B2homo-240-M"
    anim_dir.mkdir()
    
    # Create the files
    for f in HOMO_FILES:
        (anim_dir / f).touch()
        
    return data_dir

@pytest.mark.skip(reason="Test uses deprecated internal methods (_bin_folders, _get_lro_kwargs_for_folder) that were removed in pattern-based refactor. Timestamp functionality is tested in test_animal_organizer_timestamps.py")
def test_ao_grouping_logic(mock_overlap_structure, monkeypatch):
    """
    Test how AO groups these specific files.
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_MAP", MOCK_ALIASES)
    # Mock LRO creation to avoid reading actual files
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    animal_id = "AP3B2homo-240-M"

    ao = results.AnimalOrganizer(
        pattern=str(mock_overlap_structure) + "/{animal}/{session}.nwb",
        animal_id=animal_id,
    )
    
    # Check grouping
    print("\n[Grouping Results]")
    for day, folders in ao._animalday_folder_groups.items():
        fnames = [Path(f).name for f in folders]
        print(f"Day {day}: {len(fnames)} files")
        for f in fnames:
            print(f"  - {f}")
            
    # Check manual timestamp processing (simulation)
    # If we pass manual_datetimes, does _process_manual_datetimes work?
    lro_kwargs = {
        "manual_datetimes": datetime(2025, 11, 27, 15, 39, 5) # Start time
    }
    
    # Manually invoke _process_manual_datetimes as it would be in __init__
    # We need to mock _resolve_timestamp_input and _compute_global_timeline slightly 
    # OR just let them run (they rely on file durations which we don't have in empty files).
    # Since files are empty, LRO creation inside _compute_global_timeline will fail.
    # We should mock _compute_global_timeline to return dummy mapping.
    
    def mock_compute_global(self, base_dt, ad_folders, kw, original_manual_datetimes=None):
        # Return dummy map + dummy end_dt (contract: (timeline_dict, end_dt))
        out = {}
        for day, folders in ad_folders.items():
            for f in folders:
                out[Path(f).name] = base_dt # Dummy
        return out, base_dt
        
    monkeypatch.setattr(results.AnimalOrganizer, "_compute_global_timeline", mock_compute_global)
    
    processed = ao._process_manual_datetimes(lro_kwargs["manual_datetimes"], ao._animalday_folder_groups, {})
    ao._processed_timestamps = processed
    
    # Check if a specific folder is found
    # Pick a file we know exists
    target_file = HOMO_FILES[0]
    
    # Test _get_lro_kwargs_for_folder
    # We need the FULL PATH that AO has.
    # In the new system, paths are in ao._animalday_folder_groups
    full_path_target = None
    for session, files in ao._animalday_folder_groups.items():
        for file_path in files:
            if target_file in str(file_path):
                full_path_target = file_path
                break
        if full_path_target:
            break

    assert full_path_target is not None, f"Could not find {target_file} in discovered files"
    
    kwargs = ao._get_lro_kwargs_for_folder(full_path_target, {})
    print(f"\nTarget File: {target_file}")
    print(f"Extracted Kwargs: {kwargs}")
    
    assert "manual_datetimes" in kwargs, "Failed to retrieve manual timestamp!"
    dt_out = kwargs["manual_datetimes"]
    assert dt_out.tzinfo is None, f"Expected Naive datetime, got {dt_out.tzinfo}"
