
import pytest
from pathlib import Path
import neurodent.visualization.results as results
import neurodent.constants as constants

# Mock Aliases
MOCK_ALIASES = {
    "HOMO": ["AP3B2homo-240-M"],
}

@pytest.fixture
def mock_valid_structure(tmp_path):
    """
    Creates a valid PARENT structure.
    """
    parent = tmp_path / "Data"
    parent.mkdir()
    return parent

def test_bad_date_config_masks_valid_files(mock_valid_structure, monkeypatch):
    """
    Scenario 1: Valid Animal ID, discovered by glob, but fails Date Parsing.
    Should raise SPECIFIC error, not skipped or generic 'Not Found'.
    """
    monkeypatch.setattr(constants, "GENOTYPE_ALIASES", MOCK_ALIASES)
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    parent = mock_valid_structure
    anim_id = "AP3B2homo-240-M"
    
    # 1. Create a matching subdirectory so glob finds it (AO glob is *ID*/*.nwb)
    subdir = parent / "AP3B2homo-240-M"
    subdir.mkdir(exist_ok=True)
    
    # 2. Add a file that matches ID but has NO valid date (and no defaults set)
    # utils.parse_str_to_day will raise ValueError("No valid date token found...")
    (subdir / "AP3B2homo-240-M_NoDate.nwb").touch()
    
    with pytest.raises(ValueError) as excinfo:
        results.AnimalOrganizer(
            pattern=f"{parent}/*{anim_id}*",
            animal_id=anim_id,
                                )
    
    error_msg = str(excinfo.value)
    print(f"\n[Scenario 1] Caught expected error: {error_msg}")
    
    assert "matched Animal ID/Genotype but failed parsing" in error_msg
    assert "No valid date token found" in error_msg


def test_strange_format_errors(mock_valid_structure, monkeypatch):
    """
    Scenario 2: Exhaustive checking of strange formats.
    - Garbage characters in date field?
    """
    monkeypatch.setattr(constants, "GENOTYPE_ALIASES", MOCK_ALIASES)
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    parent = mock_valid_structure
    anim_id = "AP3B2homo-240-M"
    subdir = parent / "AP3B2homo-240-M"
    subdir.mkdir(exist_ok=True)
    
    # Case A: Totally garbage date string that might confuse parser
    # "AP3B2homo-240-M_NotADateAtAll.nwb"
    # This should fail date parsing.
    (subdir / "AP3B2homo-240-M_GarbageDate.nwb").touch()
    
    with pytest.raises(ValueError) as excinfo:
        results.AnimalOrganizer(
            pattern=f"{parent}/*{anim_id}*",
            animal_id=anim_id,
                                )
    assert "matched Animal ID/Genotype but failed parsing" in str(excinfo.value)


def test_filtering_still_works(mock_valid_structure, monkeypatch):
    """
    Scenario 3: Verify that actual ID mismatches are STILL filtered (skipped/warned)
    and do NOT raise the new specific error.
    """
    monkeypatch.setattr(constants, "GENOTYPE_ALIASES", MOCK_ALIASES)
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    
    parent = mock_valid_structure
    anim_id = "AP3B2homo-240-M"
    subdir = parent / "AP3B2homo-240-M"
    subdir.mkdir(exist_ok=True)
    
    # Clean subdir
    for f in subdir.glob("*"):
        f.unlink()
        
    # File with WRONG ID (should be filtered)
    # To test filtering, we need a file that:
    # 1. Matches bin_folder GLOB: path contains *AP3B2homo-240-M*
    #    (Satisfied because it's in the 'AP3B2homo-240-M' subdir)
    # 2. FAILS ID Validation: filename does NOT contain "AP3B2homo-240-M"
    
    # "Ghost" file that accidentally lives in the folder but isn't an animal file
    (subdir / "Ghost_File_NoID_251127.nwb").touch()
    
    # Initialize AO.
    # It finds the file because parent folder matches glob.
    # It tries to parse path.
    # Parse fails because filename ("Ghost_File...") doesn't have ID.
    # It should CATCH ValueError, Log Warning, SKIP.
    # Result: bin_folders is empty.
    # Raises: "No directories found" (Generic).
    
    with pytest.raises(ValueError) as excinfo:
        results.AnimalOrganizer(
            pattern=f"{parent}/*{anim_id}*",
            animal_id=anim_id,
                                    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
        )
        
    error_msg = str(excinfo.value)
    print(f"\n[Scenario 3] Caught expected error: {error_msg}")
    
    # Should be the GENERIC error, confirming it was FILTERED (Skipped)
    # instead of crashing with a specific parse error.
    assert "No directories found" in error_msg
    assert "matched Animal ID/Genotype but failed parsing" not in error_msg
