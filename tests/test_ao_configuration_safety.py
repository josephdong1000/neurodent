
import pytest
from pathlib import Path
import neurodent.visualization.results as results
import neurodent.constants as constants

@pytest.fixture
def mock_valid_structure(tmp_path):
    """
    Creates a valid PARENT structure.
    """
    parent = tmp_path / "Data"
    parent.mkdir()
    return parent

def test_no_files_discovered_error(mock_valid_structure, monkeypatch):
    """
    Test that when no files match the pattern, a clear error is raised.
    This replaces the old date parsing test - in the new system, we test pattern matching.
    """
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    parent = mock_valid_structure
    anim_id = "AP3B2homo-240-M"

    # Create a subdirectory but don't add files that match the pattern
    subdir = parent / anim_id
    subdir.mkdir(exist_ok=True)

    # Add files that won't match the pattern (pattern expects data/ subdirectory)
    (subdir / "file1.nwb").touch()
    (subdir / "file2.nwb").touch()

    # Pattern expects {animal}/data/{session}.nwb but we only have {animal}/{file}.nwb
    with pytest.raises(ValueError) as excinfo:
        results.AnimalOrganizer(
            pattern=str(parent) + "/{animal}/data/{session}.nwb",
            animal_id=anim_id,
        )

    error_msg = str(excinfo.value)
    print(f"\n[Scenario 1] Caught expected error: {error_msg}")

    assert "No items discovered" in error_msg


def test_pattern_matching_works(mock_valid_structure, monkeypatch):
    """
    Test that the pattern-based discovery works correctly.
    This replaces the old strange format test.
    """
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    parent = mock_valid_structure
    anim_id = "AP3B2homo-240-M"
    subdir = parent / anim_id
    subdir.mkdir(exist_ok=True)

    # Create files that match the pattern
    (subdir / "2023-01-15.nwb").touch()
    (subdir / "2023-01-16.nwb").touch()

    # Should successfully create AO
    ao = results.AnimalOrganizer(
        pattern=str(parent) + "/{animal}/{session}.nwb",
        animal_id=anim_id,
    )

    # Should have discovered 2 sessions
    assert len(ao._animalday_folder_groups) == 2
    assert "2023-01-15" in ao._animalday_folder_groups
    assert "2023-01-16" in ao._animalday_folder_groups


def test_animal_id_filtering_works(mock_valid_structure, monkeypatch):
    """
    Verify that animal_id filtering still works correctly.
    """
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    parent = mock_valid_structure
    anim_id = "AP3B2homo-240-M"

    # Create subdirectories for multiple animals
    subdir1 = parent / anim_id
    subdir1.mkdir(exist_ok=True)
    (subdir1 / "2023-01-15.nwb").touch()

    subdir2 = parent / "OtherAnimal"
    subdir2.mkdir(exist_ok=True)
    (subdir2 / "2023-01-15.nwb").touch()

    # Initialize AO with specific animal_id
    ao = results.AnimalOrganizer(
        pattern=str(parent) + "/{animal}/{session}.nwb",
        animal_id=anim_id,
    )

    # Should only discover files for the specified animal
    assert len(ao._animalday_folder_groups) == 1
    assert "2023-01-15" in ao._animalday_folder_groups

    # Verify no files from other animals were included
    for session_files in ao._animalday_folder_groups.values():
        for file_path in session_files:
            assert anim_id in str(file_path)
