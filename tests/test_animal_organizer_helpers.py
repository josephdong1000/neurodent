"""Tests for AnimalOrganizer helper methods:
_get_item_name, _is_item_file, _get_context_path."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neurodent.core.discovery import DiscoveredFile


# ---------------------------------------------------------------------------
# Helpers to call the methods without constructing a real AnimalOrganizer
# ---------------------------------------------------------------------------

def _get_item_name(item):
    """Call AnimalOrganizer._get_item_name via a minimal mock instance."""
    from neurodent.visualization.results import AnimalOrganizer
    # Bypass __init__ entirely
    instance = object.__new__(AnimalOrganizer)
    return instance._get_item_name(item)


def _is_item_file(item):
    from neurodent.visualization.results import AnimalOrganizer
    instance = object.__new__(AnimalOrganizer)
    return instance._is_item_file(item)


def _get_context_path(item):
    from neurodent.visualization.results import AnimalOrganizer
    return AnimalOrganizer._get_context_path(item)


# ---------------------------------------------------------------------------
# _get_item_name
# ---------------------------------------------------------------------------

class TestGetItemName:
    def test_discovered_file_single_path(self, tmp_path):
        f = tmp_path / "session.edf"
        f.touch()
        df = DiscoveredFile(path=str(f))
        assert _get_item_name(df) == "session.edf"

    def test_discovered_file_multiple_paths_appends_ellipsis(self, tmp_path):
        f1 = tmp_path / "data.bin"
        f2 = tmp_path / "meta.csv"
        f1.touch()
        f2.touch()
        df = DiscoveredFile(paths=(str(f1), str(f2)))
        name = _get_item_name(df)
        assert name == "data.bin..."

    def test_list_of_paths(self, tmp_path):
        f1 = str(tmp_path / "fileA.bin")
        f2 = str(tmp_path / "fileB.bin")
        assert _get_item_name([f1, f2]) == "fileA.bin"

    def test_tuple_of_paths(self, tmp_path):
        f1 = str(tmp_path / "rec.edf")
        f2 = str(tmp_path / "rec2.edf")
        assert _get_item_name((f1, f2)) == "rec.edf"

    def test_plain_string(self, tmp_path):
        p = str(tmp_path / "myfile.rhd")
        assert _get_item_name(p) == "myfile.rhd"

    def test_path_object(self, tmp_path):
        p = tmp_path / "myfile.rhd"
        assert _get_item_name(p) == "myfile.rhd"


# ---------------------------------------------------------------------------
# _is_item_file
# ---------------------------------------------------------------------------

class TestIsItemFile:
    def test_discovered_file_that_is_a_file(self, tmp_path):
        f = tmp_path / "rec.edf"
        f.touch()
        df = DiscoveredFile(path=str(f))
        assert _is_item_file(df) is True

    def test_discovered_file_that_is_a_directory(self, tmp_path):
        df = DiscoveredFile(path=str(tmp_path))
        assert _is_item_file(df) is False

    def test_list_first_element_is_file(self, tmp_path):
        f = tmp_path / "a.bin"
        f.touch()
        assert _is_item_file([str(f)]) is True

    def test_list_first_element_is_directory(self, tmp_path):
        assert _is_item_file([str(tmp_path)]) is False

    def test_plain_string_file(self, tmp_path):
        f = tmp_path / "x.edf"
        f.touch()
        assert _is_item_file(str(f)) is True

    def test_plain_string_directory(self, tmp_path):
        assert _is_item_file(str(tmp_path)) is False

    def test_discovered_file_empty_paths(self):
        df = DiscoveredFile(path=None, paths=())
        # get_path_list() returns [] → should return False
        assert _is_item_file(df) is False


# ---------------------------------------------------------------------------
# _get_context_path
# ---------------------------------------------------------------------------

class TestGetContextPath:
    def test_discovered_file_returns_path(self, tmp_path):
        f = tmp_path / "rec.edf"
        result = _get_context_path(DiscoveredFile(path=str(f)))
        assert result == Path(f)
        assert isinstance(result, Path)

    def test_discovered_file_multi_returns_first_path(self, tmp_path):
        f1 = str(tmp_path / "a.bin")
        f2 = str(tmp_path / "b.csv")
        result = _get_context_path(DiscoveredFile(paths=(f1, f2)))
        assert result == Path(f1)

    def test_list_returns_first_element_as_path(self, tmp_path):
        f1 = str(tmp_path / "first.bin")
        f2 = str(tmp_path / "second.bin")
        result = _get_context_path([f1, f2])
        assert result == Path(f1)

    def test_tuple_returns_first_element_as_path(self, tmp_path):
        f1 = str(tmp_path / "t1.edf")
        result = _get_context_path((f1,))
        assert result == Path(f1)

    def test_plain_string_returns_path(self, tmp_path):
        p = str(tmp_path / "myfile.edf")
        result = _get_context_path(p)
        assert result == Path(p)
        assert isinstance(result, Path)

    def test_path_object_returns_path(self, tmp_path):
        p = tmp_path / "myfile.edf"
        result = _get_context_path(p)
        assert result == p
