
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

def test_ao_grouping_logic(mock_overlap_structure, monkeypatch):
    """
    Test how AO groups these specific files.
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    # Mock LRO creation to avoid reading actual files
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    
    # Needs parsing pattern for simple date strings like 251127
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}

    ao = results.AnimalOrganizer(
        base_folder_path=mock_overlap_structure,
        animal_id="AP3B2homo-240-M",
        mode="concat",
        file_pattern="*.nwb",
        day_parse_kwargs=day_parse_kwargs
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
        # Return dummy map
        out = {}
        for day, folders in ad_folders.items():
            for f in folders:
                out[Path(f).name] = base_dt # Dummy
        return out
        
    monkeypatch.setattr(results.AnimalOrganizer, "_compute_global_timeline", mock_compute_global)
    
    processed = ao._process_manual_datetimes(lro_kwargs["manual_datetimes"], ao._animalday_folder_groups, {})
    ao._processed_timestamps = processed
    
    # Check if a specific folder is found
    # Pick a file we know exists
    target_file = HOMO_FILES[0]
    
    # Test _get_lro_kwargs_for_folder
    # We need the FULL PATH that AO has.
    # ao._bin_folders contains full paths.
    full_path_target = [f for f in ao._bin_folders if target_file in str(f)][0]
    
    kwargs = ao._get_lro_kwargs_for_folder(full_path_target, {})
    print(f"\nTarget File: {target_file}")
    print(f"Extracted Kwargs: {kwargs}")
    
    assert "manual_datetimes" in kwargs, "Failed to retrieve manual timestamp!"
    dt_out = kwargs["manual_datetimes"]
    assert dt_out.tzinfo is None, f"Expected Naive datetime, got {dt_out.tzinfo}"
