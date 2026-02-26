
import pytest
from pathlib import Path
import neurodent.visualization.results as results
import neurodent.constants as constants
import neurodent.core.utils as utils

# Mock Aliases
MOCK_ALIASES = {
    "HOMO": ["AP3B2homo-240-M", "homo", "AP3B2homo"],
    "WT": ["AP3B2wt-241-M", "wt", "AP3B2wt"]
}

@pytest.fixture
def mock_production_structure(tmp_path):
    """
    Simulate the production structure:
    /Intan/
      PortA-AP3B2homo-240-M-PortB-AP3B2wt-241-M-standardEEG/
        PortA_... (Ghost, technically)
        AP3B2homo-240-M_Correct_HOMO.nwb
        AP3B2wt-241-M_Correct_WT.nwb
        PortC_... (Ghost, incorrect)
    """
    intan = tmp_path / "Intan"
    parent = intan / "PortA-AP3B2homo-240-M-PortB-AP3B2wt-241-M-standardEEG"
    parent.mkdir(parents=True)
    
    # Create subdirectories for animals (as AO expects them)
    # The glob pattern `base / *ID*` will match these.
    homo_dir = parent / "AP3B2homo-240-M"
    homo_dir.mkdir()
    
    wt_dir = parent / "AP3B2wt-241-M"
    wt_dir.mkdir()
    
    # 1. Correct HOMO file (Date: 251127)
    (homo_dir / "AP3B2homo-240-M_Correct_HOMO_251127.nwb").touch()
    
    # 2. Correct WT file (Date: 251127)
    (wt_dir / "AP3B2wt-241-M_Correct_WT_251127.nwb").touch()
    
    # 3. Ghost File (PortC) - If it causes issues, it must be discoverable?
    # If it's in the PARENT, AO won't find it.
    # If it's in the HOMO folder, AO will find it.
    # Let's put a Ghost file in the PARENT to verify it is IGNORED (Validation of safety).
    (parent / "PortC_251128_060908.nwb").touch()
    
    # Also put a 'Ghost' file INSIDE the HOMO folder to test filtering logic
    # if it doesn't match ID?
    (homo_dir / "PortC_Inside_251128.nwb").touch()
    
    return parent

def test_ao_directory_mode_repro(tmp_path, monkeypatch):
    """
    Regression Test: Reproduce "No directories found" for Directory Mode.
    Scenario:
      Structure: Root/Data/20230101/recording.bin
      Config: mode="nest", file_pattern=None (implies directory mode)
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    # Setup Mock
    data_dir = tmp_path / "Data"
    # Create folder named like the animal so 'nest' mode finds it?
    # Or does the folder need to be the animal ID?
    # Usually in directory mode: top-level folders ARE the animals.
    # e.g. /Data/AP3B2homo-240-M_251127/...
    
    anim_dir = data_dir / "AP3B2homo-240-M_251127"
    day_dir = anim_dir / "ignored_subdir"
    day_dir.mkdir(parents=True)
    (day_dir / "recording.bin").touch()
    
    # Mocking _create_long_recordings because we don't want real LRO instantiation
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}

    # Initialize AO
    ao = results.AnimalOrganizer(
        base_folder_path=data_dir,
        animal_id="AP3B2homo-240-M",
        mode="concat", # Directory name itself contains the ID
        file_pattern=None, # Directory mode
        day_parse_kwargs=day_parse_kwargs
    )
    
    # If successful, no error raised.
    # Check what was found
    found_folders = [Path(f).name for f in ao._bin_folders]
    assert "AP3B2homo-240-M_251127" in found_folders


def test_ao_discovery_production_setup_homo(mock_production_structure, monkeypatch):
    """
    Test Case 1a: Production Setup for HOMO.
    Config points AO to the PARENT folder.
    Expectation: Should ONLY find files with 'AP3B2homo-240-M' string.
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    parent_folder = mock_production_structure
    anim_id = "AP3B2homo-240-M"
    
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    ao = results.AnimalOrganizer(
        base_folder_path=parent_folder,
        animal_id=anim_id,
        mode="concat",
        file_pattern="*.nwb",
        day_parse_kwargs=day_parse_kwargs
    )
    
    found_files = [Path(f).name for f in ao._bin_folders]
    print(f"\n[Production HOMO Check] Found files for {anim_id}: {found_files}")
    
    # Should find Correct.nwb
    assert "AP3B2homo-240-M_Correct_HOMO_251127.nwb" in found_files
    
    # Should NOT find the sibling WT file
    assert "AP3B2wt-241-M_Correct_WT_251127.nwb" not in found_files, "HOMO AO incorrectly found WT file"
    
    # Should NOT find the Ghost file (PortC)
    # Because PortC...nwb does NOT contain 'AP3B2homo-240-M' string
    assert "PortC_251128_060908.nwb" not in found_files, "HOMO AO incorrectly found ghost file"


def test_ao_discovery_production_setup_wt(mock_production_structure, monkeypatch):
    """
    Test Case 1b: Production Setup for WT.
    Config points AO to the PARENT folder.
    Expectation: Should ONLY find files in the 'AP3B2wt-241-M' subdirectory.
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    parent_folder = mock_production_structure
    anim_id = "AP3B2wt-241-M"
    
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    ao = results.AnimalOrganizer(
        base_folder_path=parent_folder,
        animal_id=anim_id,
        mode="concat",
        file_pattern="*.nwb",
        day_parse_kwargs=day_parse_kwargs
    )
    
    found_files = [Path(f).name for f in ao._bin_folders]
    print(f"\n[Production WT Check] Found files for {anim_id}: {found_files}")
    
    assert "AP3B2wt-241-M_Correct_WT_251127.nwb" in found_files
    assert "PortC_251128_060908.nwb" not in found_files, "WT AO incorrectly found ghost file"

def test_ao_discovery_grandparent_setup(mock_production_structure, monkeypatch):
    """
    Test Case 2: Grandparent Setup.
    Initialize AO pointing to the GRANDPARENT folder (e.g. 'Intan').
    This tests if the glob is too greedy when recursing from higher up.
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    grandparent = mock_production_structure.parent
    anim_id = "AP3B2homo-240-M"
    
    # We mock _create_long_recordings to avoid LRO creation logic which requires full file access
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    # Grandparent setup with simple glob (*ID*/*.nwb) will NOT find deeply nested files
    # And it will find PortC but filter it out.
    # So it should raise ValueError: No directories found.
    with pytest.raises(ValueError, match="No directories found"):
        ao = results.AnimalOrganizer(
            base_folder_path=grandparent,
            animal_id=anim_id,
            mode="concat",
            file_pattern="*.nwb",
            day_parse_kwargs=day_parse_kwargs
        )
        
    # found_files = [Path(f).name for f in ao._bin_folders] # Unreachable
    # assert "AP3B2homo-240-M_Correct_HOMO_251127.nwb" in found_files
    # assert "PortC_251128_060908.nwb" not in found_files

def test_parse_ghost_file_behavior(monkeypatch):
    """
    Test Case 3: Verify WHY the ghost file parses as HOMO.
    Does 'PortC' inside 'PortA-AP3B2homo...' parse as HOMO?
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    # Construct a path mimicking the discovered ghost file
    # /.../PortA-AP3B2homo.../PortC...nwb
    ghost_path = Path("/mock/PortA-AP3B2homo-240-M-PortB-AP3B2wt-241-M-standardEEG/PortC_251128_060908.nwb")
    
    # Try parsing in 'concat' mode (default)
    # This usually checks the filename. 'PortC...' does NOT contain 'AP3B2homo'.
    # So 'concat' mode should FAIL to parse.
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    try:
        parsed_concat = utils.parse_path_to_animalday(ghost_path, mode="concat", **day_parse_kwargs)
        print(f"\n[Ghost Parse] Concat mode result: {parsed_concat}")
    except Exception as e:
        print(f"\n[Ghost Parse] Concat mode failed: {e}")

    # Try parsing in 'nest' mode 
    # This checks parent folder. 'PortA-AP3B2homo...' DOES contain 'AP3B2homo'.
    # So 'nest' mode should SUCCEED.
    # Expectation: Nest mode extracts the ANIMAL ID from the parent folder path.
    # Logic in utils.py verify: 
    #   geno = parse_str_to_genotype(filepath.parent.name)
    #   animid = parse_str_to_animal(filepath.parent.name, animal_param=animal_param)
    # The actual return IS the animal ID, derived from the folder name string.
    
    try:
        parsed_nest = utils.parse_path_to_animalday(ghost_path, mode="nest", **day_parse_kwargs)
        print(f"[Ghost Parse] Nest mode result: {parsed_nest}")
        
        # Verify that 'nest' mode correctly extracts the genotype and animal ID from 
        # the parent/grandparent folder path, matching the greedy behavior observed.
        
        # However, for this test, let's asserting that it matches the GENOTYPE "HOMO" at least.
        assert parsed_nest["genotype"] == "HOMO"
        # And that the animal ID *contains* the target ID (even if it's the full string, AO might match it loosely?)
        assert "AP3B2homo-240-M" in parsed_nest["animal"] 
        
    except Exception as e:
        pytest.fail(f"Nest mode parsing failed: {e}")

def test_ao_discovery_ambiguous_names(mock_production_structure, monkeypatch):
    """
    Test Case 4: Ambiguous/Overlapping Names.
    e.g. 'AP3B2homo' vs 'AP3B2homo-clone'
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    parent = mock_production_structure
    
    # Create ambiguous sibling
    # Parent/AP3B2homo-240-M-clone/
    ambiguous_dir = parent / "AP3B2homo-240-M-clone"
    ambiguous_dir.mkdir()
    (ambiguous_dir / "Ambiguous_251127.nwb").touch()
    
    # Initialize AO for original ID
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    ao = results.AnimalOrganizer(
        base_folder_path=parent,
        animal_id="AP3B2homo-240-M",
        mode="concat",
        file_pattern="*.nwb",
        day_parse_kwargs=day_parse_kwargs
    )
    
    found_files = [Path(f).name for f in ao._bin_folders]
    print(f"\n[Ambiguous Check] Found files: {found_files}")
    
    # Should find Correct.nwb
    assert "AP3B2homo-240-M_Correct_HOMO_251127.nwb" in found_files
    
    # Should NOT find Ambiguous.nwb? 
    # If glob pattern is *ID*, it matches ID-clone.
    # This detects if glob is too greedy.
    assert "Ambiguous_251127.nwb" not in found_files, "AO glob pattern is too greedy (matched ID-suffix)"

def test_ao_discovery_nested_duplicates(mock_production_structure, monkeypatch):
    """
    Test Case 5: Nested Duplicates.
    Parent/Subdir(ID)/Ghost(ID).nwb
    Should find both?
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    parent = mock_production_structure
    # Add a file specifically named like the animal INSIDE the animal subdir
    # Parent/AP3B2homo-240-M/AP3B2homo-240-M_duplicate.nwb
    duplicate_file = parent / "AP3B2homo-240-M" / "AP3B2homo-240-M_duplicate_251127.nwb"
    # Ensure parent dir exists (it is created by mock_production_structure but we might need subdir)
    (parent / "AP3B2homo-240-M").mkdir(exist_ok=True)
    duplicate_file.touch()
    
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    ao = results.AnimalOrganizer(
        base_folder_path=parent,
        animal_id="AP3B2homo-240-M",
        mode="concat",
        file_pattern="*.nwb",
        day_parse_kwargs=day_parse_kwargs
    )
    
    found_files = [Path(f).name for f in ao._bin_folders]
    print(f"\n[Nested Duplicate Check] Found files: {found_files}")
    
    # Should find both correctly
    assert "AP3B2homo-240-M_Correct_HOMO_251127.nwb" in found_files
    assert "AP3B2homo-240-M_duplicate_251127.nwb" in found_files

def test_ao_discovery_file_level_exclusion(mock_production_structure, monkeypatch):
    """
    Test Case 6: File Level Exclusion.
    Files in correct subdir but named differently.
    Parent/AP3B2homo-240-M/IgnoredFile.nwb
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    
    parent = mock_production_structure
    (parent / "AP3B2homo-240-M").mkdir(exist_ok=True)
    ignored_file = parent / "AP3B2homo-240-M" / "IgnoredFile_251127.nwb"
    ignored_file.touch()
    
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)
    
    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    ao = results.AnimalOrganizer(
        base_folder_path=parent,
        animal_id="AP3B2homo-240-M",
        mode="concat",
        file_pattern="*.nwb",
        day_parse_kwargs=day_parse_kwargs
    )
    
    found_files = [Path(f).name for f in ao._bin_folders]
    print(f"\n[File Level Check] Found files: {found_files}")
    
    # Should find Correct.nwb
    assert "AP3B2homo-240-M_Correct_HOMO_251127.nwb" in found_files
    
    # Should NOT find IgnoredFile.nwb (because mode=concat checks filename)
    assert "IgnoredFile_251127.nwb" not in found_files


@pytest.fixture
def mock_joint_session_structure(tmp_path):
    """
    Simulate a joint session structure where filenames contain concatenated animal IDs.
    e.g. Arx Rosa dataset: MARSH 20141125ARXROSATAM967968969418_1_Selection1.EDF
    """
    session_dir = tmp_path / "Arx Rosa  967 968 969 418"
    session_dir.mkdir(parents=True)

    # Joint session EDF files — IDs are concatenated in the filename
    (session_dir / "MARSH_20141125ARXROSATAM967968969418_Selection1_251125.EDF").touch()
    (session_dir / "MARSH_20141125ARXROSATAM967968969418_Selection2_251126.EDF").touch()

    # An unrelated file that should NOT match
    (session_dir / "UNRELATED_FILE_999999_251127.EDF").touch()

    return session_dir


def test_ao_animal_file_match_pattern_regex(mock_joint_session_structure, monkeypatch):
    """
    Test animal_file_match_pattern with a regex pattern for joint sessions.
    Without animal_file_match_pattern, AO would fail because 'ArxRosa-967' is not
    a substring of 'MARSH_20141125ARXROSATAM967968969418_Selection1_251125.EDF'.
    With animal_file_match_pattern="967|968|969|418", the regex matches '967' in the filename.
    """
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}

    # Without animal_file_match_pattern: should fail
    with pytest.raises(ValueError, match="No directories found"):
        results.AnimalOrganizer(
            base_folder_path=mock_joint_session_structure,
            animal_id="ArxRosa-967",
            mode="base",
            file_pattern="*.EDF",
            day_parse_kwargs=day_parse_kwargs,
        )

    # With animal_file_match_pattern: should succeed
    ao = results.AnimalOrganizer(
        base_folder_path=mock_joint_session_structure,
        animal_id="ArxRosa-967",
        mode="base",
        file_pattern="*.EDF",
        day_parse_kwargs=day_parse_kwargs,
        animal_file_match_pattern="967|968|969|418",
    )

    found_files = [Path(f).name for f in ao._bin_folders]

    # Should find the joint session files
    assert "MARSH_20141125ARXROSATAM967968969418_Selection1_251125.EDF" in found_files
    assert "MARSH_20141125ARXROSATAM967968969418_Selection2_251126.EDF" in found_files

    # Should NOT find the unrelated file
    assert "UNRELATED_FILE_999999_251127.EDF" not in found_files

    # animal_id should still be the original ID
    assert ao.animal_id == "ArxRosa-967"


def test_ao_animal_file_match_pattern_none_default(mock_production_structure, monkeypatch):
    """
    Test that animal_file_match_pattern=None (default) preserves existing behavior.
    """
    monkeypatch.setattr("neurodent.constants.GENOTYPE_ALIASES", MOCK_ALIASES)
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    day_parse_kwargs = {"date_patterns": [(r"\d{6}", "%y%m%d")]}
    ao = results.AnimalOrganizer(
        base_folder_path=mock_production_structure,
        animal_id="AP3B2homo-240-M",
        mode="concat",
        file_pattern="*.nwb",
        day_parse_kwargs=day_parse_kwargs,
        # animal_file_match_pattern not passed — should default to [animal_id]
    )

    found_files = [Path(f).name for f in ao._bin_folders]
    assert "AP3B2homo-240-M_Correct_HOMO_251127.nwb" in found_files
    assert ao.animal_file_match_pattern == ["AP3B2homo-240-M"]
