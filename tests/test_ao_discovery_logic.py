"""
Tests for AnimalOrganizer file discovery with pattern-based matching.

These tests verify that FileDiscoverer correctly:
1. Discovers files using {animal}, {session}, {index} placeholders
2. Filters by animal_id when provided
3. Groups files by session
4. Handles skip_sessions correctly
5. Works with DiscoveredFile for dual-file scenarios
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import neurodent.loading.animal_organizer as results
import neurodent.loading.long_recording_organizer as core


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
    """Test skipping specific sessions with exact match (backward compat)."""
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


def test_skip_sessions_glob_pattern(tmp_path, monkeypatch):
    """Test skipping sessions with glob/wildcard patterns."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Create sessions: good_day1, good_day2, bad_day1, corrupted_day1
    for session in ["good_day1", "good_day2", "bad_day1", "corrupted_day1"]:
        d = tmp_path / "A10" / session
        d.mkdir(parents=True)
        (d / "1.rhd").touch()

    pattern = str(tmp_path) + "/{animal}/{session}/{index}.rhd"

    # Wildcard: skip anything containing "bad"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
        skip_sessions=["*bad*"],
    )
    assert "bad_day1" not in ao._animalday_folder_groups
    assert "good_day1" in ao._animalday_folder_groups
    assert "good_day2" in ao._animalday_folder_groups
    assert "corrupted_day1" in ao._animalday_folder_groups


def test_skip_sessions_multiple_glob_patterns(tmp_path, monkeypatch):
    """Test skipping sessions with multiple glob patterns (reject if any matches)."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    for session in ["good_day1", "good_day2", "bad_day1", "corrupted_day1"]:
        d = tmp_path / "A10" / session
        d.mkdir(parents=True)
        (d / "1.rhd").touch()

    pattern = str(tmp_path) + "/{animal}/{session}/{index}.rhd"

    # Multiple patterns: skip "bad" and "corrupted_*"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
        skip_sessions=["*bad*", "corrupted_*"],
    )
    assert "bad_day1" not in ao._animalday_folder_groups
    assert "corrupted_day1" not in ao._animalday_folder_groups
    assert "good_day1" in ao._animalday_folder_groups
    assert "good_day2" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups) == 2


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

    # Each session should have a DiscoveredFile with 2 files
    for session_files in ao._animalday_folder_groups.values():
        assert len(session_files) == 1
        from neurodent.loading.discovery import DiscoveredFile
        assert isinstance(session_files[0], DiscoveredFile)
        assert session_files[0].is_multi_file
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
    """Test pattern with different ordering using underscores instead of dashes to avoid ambiguity"""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Create structure: tmp_path/data/A10_20250124_1.rhd (no dashes in date to avoid ambiguity)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "A10_20250124_1.rhd").touch()
    (data_dir / "A10_20250124_2.rhd").touch()
    (data_dir / "A10_20250125_1.rhd").touch()

    pattern = str(tmp_path / "data") + "/{animal}_{session}_{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    # Should find 2 sessions
    assert len(ao._animalday_folder_groups) == 2
    assert "20250124" in ao._animalday_folder_groups
    assert "20250125" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups["20250124"]) == 2
    assert len(ao._animalday_folder_groups["20250125"]) == 1


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


def test_glob_wildcards_mixed_with_placeholders(tmp_path, monkeypatch):
    """Test glob wildcards (*) mixed with {placeholders} in patterns.

    Mirrors real sox5 data layout where the pattern contains a literal animal
    name surrounded by glob wildcards: parent/*AnimalName*/{session}/...
    The {animal} placeholder is NOT used with surrounding wildcards — instead
    the pipeline resolves the animal name into the pattern as a literal.
    """
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Structure: cohort_2mice/A10/session1/prefix-1.rhd
    parent = tmp_path / "cohort_2mice"
    for session in ["session1", "session2"]:
        d = parent / "A10" / session
        d.mkdir(parents=True)
        (d / "prefix-1.rhd").touch()
    (parent / "A10" / "session1" / "prefix-2.rhd").touch()

    # Pattern uses literal animal name with glob wildcards (like real pipeline)
    pattern = str(parent) + "/*A10*/{session}/*-{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id=None,
    )

    assert len(ao._animalday_folder_groups) == 2
    assert "session1" in ao._animalday_folder_groups
    assert "session2" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups["session1"]) == 2
    assert len(ao._animalday_folder_groups["session2"]) == 1


def test_glob_wildcards_with_animal_placeholder(tmp_path, monkeypatch):
    """Test *{animal}* pattern with animal_id — placeholder is substituted
    into the pattern before regex compilation, avoiding greediness issues."""
    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", lambda self, kw: None)

    # Structure: cohort/prefix-A10-suffix/session1/1.rhd
    parent = tmp_path / "cohort"
    for session in ["session1", "session2"]:
        d = parent / "prefix-A10-suffix" / session
        d.mkdir(parents=True)
        (d / "1.rhd").touch()
    # Also create another animal to verify filtering
    other = parent / "prefix-B20-suffix" / "session1"
    other.mkdir(parents=True)
    (other / "1.rhd").touch()

    pattern = str(parent) + "/*{animal}*/{session}/{index}.rhd"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    assert len(ao._animalday_folder_groups) == 2
    assert "session1" in ao._animalday_folder_groups
    assert "session2" in ao._animalday_folder_groups
    # Should NOT include B20's files
    all_files = [str(f) for files in ao._animalday_folder_groups.values() for f in files]
    assert not any("B20" in f for f in all_files)


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


def test_sort_lros_handles_multifile_discovered_files():
    """Regression test: _sort_lros_by_median_time must not raise TypeError when
    items are multi-file DiscoveredFile objects (e.g. .bin + .csv pairs).

    Previously, the logging code called ``Path(folder).name`` which invoked
    ``DiscoveredFile.__fspath__`` and raised::

        TypeError: Multi-file DiscoveredFile cannot be converted to a single path.
    """
    from neurodent.loading.discovery import DiscoveredFile

    ao = MagicMock(spec=results.AnimalOrganizer)
    ao._get_item_name = results.AnimalOrganizer._get_item_name.__get__(ao, results.AnimalOrganizer)
    ao._sort_lros_by_median_time = results.AnimalOrganizer._sort_lros_by_median_time.__get__(ao, results.AnimalOrganizer)
    ao._sort_lros_by_median_time_static = results.AnimalOrganizer._sort_lros_by_median_time_static

    # Two multi-file DiscoveredFile objects (bin + csv pairs) for the same session
    df1 = DiscoveredFile(
        paths=("/data/A10/session1/001_ColMajor.bin", "/data/A10/session1/001_Meta.csv"),
        metadata={"animal": "A10", "session": "session1", "index": "001"},
    )
    df2 = DiscoveredFile(
        paths=("/data/A10/session1/002_ColMajor.bin", "/data/A10/session1/002_Meta.csv"),
        metadata={"animal": "A10", "session": "session1", "index": "002"},
    )

    mock_lro1 = MagicMock()
    mock_lro1.file_end_datetimes = [datetime(2025, 1, 1, 13, 0, 0)]
    mock_lro1.LongRecording.get_duration.return_value = 3600.0

    mock_lro2 = MagicMock()
    mock_lro2.file_end_datetimes = [datetime(2025, 1, 1, 14, 0, 0)]
    mock_lro2.LongRecording.get_duration.return_value = 3600.0

    # This must not raise TypeError
    result = ao._sort_lros_by_median_time([(df1, mock_lro1), (df2, mock_lro2)])

    assert len(result) == 2
    # df1 (earlier timestamp) should come first
    assert result[0][0] is df1
    assert result[1][0] is df2


def test_pattern_with_irrelevant_path_data_and_index_sort(tmp_path, monkeypatch):
    """
    Integration test: pattern with irrelevant directory data between {session} and {index}.

    This tests the scenario described in the PR comment where a pattern like
    "{animal}/{session}/*/{index}/filename.ext" has irrelevant data (the asterisk)
    that could cause incorrect sorting if used instead of {index} metadata.

    The test verifies that:
    1. Files are discovered correctly with the wildcard pattern
    2. Files are sorted by {index} metadata, not by the irrelevant directory names
    3. Timeline computation uses index-based ordering
    """
    from neurodent.loading.discovery import DiscoveredFile
    import pandas as pd

    # Create file structure with 10 files where:
    # - Irrelevant directory names sort completely differently from indices
    # - Filenames are intentionally out of order (not data1, data2, data3... but scrambled)
    # This proves that ONLY sorting by {index} metadata gives correct ordering
    #
    # Directory structure: {animal}/{session}/{irrelevant_dir}/{index}/{filename}.bin
    # Filenames are deliberately scrambled: zebra, monkey, aardvark, etc.
    # If sorted by filename: aardvark < banana < cherry < ... < zebra (wrong order)
    # If sorted by index: 001, 002, 003, ..., 010 (correct order)

    session_dir = tmp_path / "A10" / "session1"

    # Create 10 files with:
    # - Indices: 001 through 010 (sequential, correct order)
    # - Directory names: intentionally scrambled to sort differently
    # - Filenames: alphabetically scrambled to not match index order

    files_config = [
        # (irrelevant_dir, index, filename)
        ("run999", "001", "zebra.bin"),      # index=001, filename sorts last, dir sorts last
        ("run001", "002", "monkey.bin"),     # index=002, filename sorts middle, dir sorts first
        ("run700", "003", "aardvark.bin"),   # index=003, filename sorts first, dir sorts middle
        ("run200", "004", "turtle.bin"),     # index=004
        ("run850", "005", "banana.bin"),     # index=005
        ("run100", "006", "xylophone.bin"),  # index=006
        ("run950", "007", "cherry.bin"),     # index=007
        ("run300", "008", "walrus.bin"),     # index=008
        ("run600", "009", "elephant.bin"),   # index=009
        ("run400", "010", "quokka.bin"),     # index=010
    ]

    for irrelevant_dir, index, filename in files_config:
        file_dir = session_dir / irrelevant_dir / index
        file_dir.mkdir(parents=True)
        (file_dir / filename).touch()

    # Mock _create_long_recordings to prevent actual LRO creation
    # Instead, capture what would be passed to timeline computation
    captured_items = []

    def mock_create_long_recordings(self, lro_kwargs):
        # Capture the discovered items for verification
        captured_items.extend(self._animalday_folder_groups.get("session1", []))

    monkeypatch.setattr(results.AnimalOrganizer, "_create_long_recordings", mock_create_long_recordings)

    # Use pattern with wildcard between session and index
    pattern = str(tmp_path) + "/{animal}/{session}/*/{index}/*.bin"
    ao = results.AnimalOrganizer(
        pattern=pattern,
        animal_id="A10",
    )

    # Verify discovery
    assert len(ao._animalday_folder_groups) == 1
    assert "session1" in ao._animalday_folder_groups
    assert len(ao._animalday_folder_groups["session1"]) == 10  # Now we have 10 files

    # Verify all discovered items are DiscoveredFile with index metadata
    items = ao._animalday_folder_groups["session1"]
    for item in items:
        assert isinstance(item, DiscoveredFile)
        assert hasattr(item, "metadata")
        assert "index" in item.metadata
        assert "animal" in item.metadata
        assert item.metadata["animal"] == "A10"
        assert "session" in item.metadata
        assert item.metadata["session"] == "session1"

    # Verify the index values are captured correctly
    indices = sorted([item.metadata["index"] for item in items])
    assert indices == ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]

    # Now test that _compute_global_timeline sorts by index, not by directory name or filename
    # Mock LongRecordingOrganizer to avoid file I/O
    with patch("neurodent.loading.long_recording_organizer.LongRecordingOrganizer") as mock_lro_cls:
        def side_effect(*args, **kwargs):
            m = MagicMock()
            m.LongRecording.get_duration.return_value = 3600.0  # 1 hour per file
            return m

        mock_lro_cls.side_effect = side_effect

        animalday_to_items = {"session1": items}
        base_datetime = pd.to_datetime("2025-01-01 10:00:00")
        base_lro_kwargs = {"datetimes_are_start": True}

        # Bind the real _compute_global_timeline method
        ao_instance = MagicMock(spec=results.AnimalOrganizer)
        ao_instance._compute_global_timeline = results.AnimalOrganizer._compute_global_timeline.__get__(
            ao_instance, results.AnimalOrganizer
        )
        ao_instance._items_have_index = results.AnimalOrganizer._items_have_index.__get__(
            ao_instance, results.AnimalOrganizer
        )
        ao_instance._session_sort_key = results.AnimalOrganizer._session_sort_key.__get__(
            ao_instance, results.AnimalOrganizer
        )
        ao_instance._get_item_name = results.AnimalOrganizer._get_item_name.__get__(
            ao_instance, results.AnimalOrganizer
        )
        ao_instance._sort_lros_by_median_time = results.AnimalOrganizer._sort_lros_by_median_time.__get__(
            ao_instance, results.AnimalOrganizer
        )
        ao_instance._is_item_file = results.AnimalOrganizer._is_item_file.__get__(
            ao_instance, results.AnimalOrganizer
        )
        ao_instance._get_item_key = results.AnimalOrganizer._get_item_key.__get__(
            ao_instance, results.AnimalOrganizer
        )

        result, _end_dt = ao_instance._compute_global_timeline(
            base_datetime, animalday_to_items, base_lro_kwargs,
            original_manual_datetimes=base_datetime,
        )

        # Verify timeline is sorted by index (001-010), NOT by filename or path
        # If sorted by filename alphabetically: aardvark, banana, cherry, elephant, monkey, quokka, turtle, walrus, xylophone, zebra
        # If sorted by index: 001, 002, 003, 004, 005, 006, 007, 008, 009, 010 ← correct

        # Expected order by index with their filenames:
        expected_order = [
            ("zebra.bin", "001", 0),       # index 001, hour 0
            ("monkey.bin", "002", 1),      # index 002, hour 1
            ("aardvark.bin", "003", 2),    # index 003, hour 2
            ("turtle.bin", "004", 3),      # index 004, hour 3
            ("banana.bin", "005", 4),      # index 005, hour 4
            ("xylophone.bin", "006", 5),   # index 006, hour 5
            ("cherry.bin", "007", 6),      # index 007, hour 6
            ("walrus.bin", "008", 7),      # index 008, hour 7
            ("elephant.bin", "009", 8),    # index 009, hour 8
            ("quokka.bin", "010", 9),      # index 010, hour 9
        ]

        # _get_item_key returns full paths; build a lookup from filename to the
        # actual key present in the result dict (avoids path-separator mismatches
        # on Windows where DiscoveredFile.path and _get_item_key may differ).
        filename_to_key = {}
        for key in result:
            fname = Path(key).name
            filename_to_key[fname] = key

        for filename, index, hour_offset in expected_order:
            expected_time = base_datetime + pd.Timedelta(hours=hour_offset)
            key = filename_to_key[filename]
            assert result[key] == expected_time, (
                f"{filename} (index {index}) should be at hour {hour_offset}, "
                f"but got {result.get(key)}"
            )
