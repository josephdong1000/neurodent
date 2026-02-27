"""
Tests for AnimalOrganizer file discovery with pattern-based matching.

These tests verify that FileDiscoverer correctly:
1. Discovers files using {animal}, {session}, {index} placeholders
2. Filters by animal_id when provided
3. Groups files by session
4. Handles skip_sessions correctly
5. Works with MultiFileGroup for dual-file scenarios
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import neurodent.visualization.results as results
import neurodent.core.core as core


@pytest.fixture
def simple_structure(tmp_path):
    """
    Create a simple test structure:
    tmp_path/
      A10/
        2025-01-24/
          1.rhd
          2.rhd
        2025-01-25/
          1.rhd
      A11/
        2025-01-24/
          1.rhd
    """
    # A10 sessions
    a10_day1 = tmp_path / "A10" / "2025-01-24"
    a10_day1.mkdir(parents=True)
    (a10_day1 / "1.rhd").touch()
    (a10_day1 / "2.rhd").touch()

    a10_day2 = tmp_path / "A10" / "2025-01-25"
    a10_day2.mkdir(parents=True)
    (a10_day2 / "1.rhd").touch()

    # A11 sessions
    a11_day1 = tmp_path / "A11" / "2025-01-24"
    a11_day1.mkdir(parents=True)
    (a11_day1 / "1.rhd").touch()

    return tmp_path


@pytest.fixture
def multi_file_structure(tmp_path):
    """
    Create a structure with dual files (bin + csv):
    tmp_path/
      A10/
        session1/
          data.bin
          meta.csv
        session2/
          data.bin
          meta.csv
    """
    session1 = tmp_path / "A10" / "session1"
    session1.mkdir(parents=True)
    (session1 / "data.bin").touch()
    (session1 / "meta.csv").touch()

    session2 = tmp_path / "A10" / "session2"
    session2.mkdir(parents=True)
    (session2 / "data.bin").touch()
    (session2 / "meta.csv").touch()

    return tmp_path


def test_single_pattern_discovery(simple_structure, monkeypatch):
    """Test basic pattern matching with {animal}/{session}/{index}."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    pattern = str(simple_structure) + "/{animal}/{session}/{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    # Should find 2 sessions for A10
    assert len(ao._animalday_folder_groups) == 2
    assert "2025-01-24" in ao._animalday_folder_groups
    assert "2025-01-25" in ao._animalday_folder_groups

    # Session 2025-01-24 should have 2 files
    assert len(ao._animalday_folder_groups["2025-01-24"]) == 2

    # Session 2025-01-25 should have 1 file
    assert len(ao._animalday_folder_groups["2025-01-25"]) == 1

    # Should NOT find A11 files
    for files in ao._animalday_folder_groups.values():
        for file_path in files:
            assert "A11" not in str(file_path)


def test_no_animal_filter(simple_structure, monkeypatch):
    """Test discovery without animal_id filter (should find all animals)."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    pattern = str(simple_structure) + "/{animal}/{session}/{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id=None,
    )

    # Should find sessions from both A10 and A11
    # Since we're not filtering by animal, sessions might overlap
    assert len(ao._animalday_folder_groups) >= 1

    # Should have discovered files from both animals
    all_files = [f for files in ao._animalday_folder_groups.values() for f in files]
    has_a10 = any("A10" in str(f) for f in all_files)
    has_a11 = any("A11" in str(f) for f in all_files)
    assert has_a10 and has_a11


def test_skip_sessions(simple_structure, monkeypatch):
    """Test skipping specific sessions."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    pattern = str(simple_structure) + "/{animal}/{session}/{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
        skip_sessions=["2025-01-24"],
    )

    # Should only find 2025-01-25
    assert len(ao._animalday_folder_groups) == 1
    assert "2025-01-25" in ao._animalday_folder_groups
    assert "2025-01-24" not in ao._animalday_folder_groups


def test_truncate_sessions(simple_structure, monkeypatch):
    """Test truncating to first n sessions."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    pattern = str(simple_structure) + "/{animal}/{session}/{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
        truncate=1,
    )

    # Should only find 1 session
    assert len(ao._animalday_folder_groups) == 1


def test_multi_pattern_discovery(multi_file_structure, monkeypatch):
    """Test multi-file discovery (bin + csv)."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    from neurodent.core.discovery import MultiFileGroup

    base = str(multi_file_structure)
    patterns = [
        base + "/{animal}/{session}/data.bin",
        base + "/{animal}/{session}/meta.csv",
    ]

    ao = results.AnimalOrganizer(
        pattern=patterns,
        animal_id="A10",
    )

    # Should find 2 sessions
    assert len(ao._animalday_folder_groups) == 2
    assert "session1" in ao._animalday_folder_groups
    assert "session2" in ao._animalday_folder_groups

    # Each session should have a MultiFileGroup with 2 files
    for session_files in ao._animalday_folder_groups.values():
        assert len(session_files) == 1
        assert isinstance(session_files[0], MultiFileGroup)
        assert len(session_files[0].paths) == 2


def test_pattern_without_session(tmp_path, monkeypatch):
    """Test pattern without {session} placeholder."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Create structure: tmp_path/A10/1.edf
    a10_dir = tmp_path / "A10"
    a10_dir.mkdir(parents=True)
    (a10_dir / "1.edf").touch()
    (a10_dir / "2.edf").touch()

    pattern = str(tmp_path) + "/{animal}/{index}.edf"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    # Should group all files under "unknown" session
    assert len(ao._animalday_folder_groups) == 1
    assert "unknown" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups["unknown"]) == 2


def test_pattern_without_index(tmp_path, monkeypatch):
    """Test pattern without {index} placeholder (single file per session)."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Create structure: tmp_path/A10/2025-01-24.edf
    a10_dir = tmp_path / "A10"
    a10_dir.mkdir(parents=True)
    (a10_dir / "2025-01-24.edf").touch()
    (a10_dir / "2025-01-25.edf").touch()

    pattern = str(tmp_path) + "/{animal}/{session}.edf"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    # Should find 2 sessions, each with 1 file
    assert len(ao._animalday_folder_groups) == 2
    assert "2025-01-24" in ao._animalday_folder_groups
    assert "2025-01-25" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups["2025-01-24"]) == 1
    assert len(ao._animalday_folder_groups["2025-01-25"]) == 1


def test_complex_pattern(tmp_path, monkeypatch):
    """Test pattern with different ordering: {animal}-{session}-{index}"""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Create structure: tmp_path/data/A10-2025-01-24-1.rhd
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "A10-2025-01-24-1.rhd").touch()
    (data_dir / "A10-2025-01-24-2.rhd").touch()
    (data_dir / "A10-2025-01-25-1.rhd").touch()

    pattern = str(tmp_path / "data") + "/{animal}-{session}-{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    # Should find 2 sessions
    assert len(ao._animalday_folder_groups) == 2
    assert "2025-01-24" in ao._animalday_folder_groups
    assert "2025-01-25" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups["2025-01-24"]) == 2
    assert len(ao._animalday_folder_groups["2025-01-25"]) == 1


def test_unique_animaldays_format(simple_structure, monkeypatch):
    """Test that unique_animaldays has correct format: {animal}_{session}."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    pattern = str(simple_structure) + "/{animal}/{session}/{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    # unique_animaldays should have format "{animal}_{session}"
    assert len(ao.unique_animaldays) == 2
    assert "A10_2025-01-24" in ao.unique_animaldays
    assert "A10_2025-01-25" in ao.unique_animaldays

    # animaldays should be alias
    assert ao.animaldays == ao.unique_animaldays


def test_no_files_found(tmp_path, monkeypatch):
    """Test error when no files match the pattern."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    pattern = str(tmp_path) + "/{animal}/{session}/{index}.rhd"

    with pytest.raises(ValueError, match="No items discovered"):
        results.AnimalOrganizer(
            pattern=pattern,
            animal_id="A10",
        )


def test_pattern_with_wildcards(tmp_path, monkeypatch):
    """Test that plain wildcards (no placeholders) still work but extract no metadata."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Create some files
    subdir = tmp_path / "data"
    subdir.mkdir(parents=True)
    (subdir / "file1.rhd").touch()
    (subdir / "file2.rhd").touch()

    # Use plain wildcard pattern (no placeholders)
    pattern = str(tmp_path) + "/*/*.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id=None,
    )

    # Should discover files but group under "unknown" session
    assert len(ao._animalday_folder_groups) == 1
    assert "unknown" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups["unknown"]) == 2
