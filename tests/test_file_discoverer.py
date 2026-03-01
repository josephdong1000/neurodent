import os
import warnings

import pytest
from pathlib import Path
from neurodent.core.discovery import DiscoveredFile, FileDiscoverer, MultiFileGroup


def test_single_file_discovery(tmp_path):
    # Setup dummy files
    (tmp_path / "A10" / "2025-01-24").mkdir(parents=True)
    (tmp_path / "A10" / "2025-01-24" / "1.rhd").touch()
    (tmp_path / "A10" / "2025-01-24" / "2.rhd").touch()

    (tmp_path / "B20" / "2025-01-25").mkdir(parents=True)
    (tmp_path / "B20" / "2025-01-25" / "1.rhd").touch()

    base = str(tmp_path)
    pattern = f"{base}/{{animal}}/{{session}}/{{index}}.rhd"

    fd = FileDiscoverer(pattern)
    results = fd.discover()

    assert len(results) == 3
    # Check if elements are extracted correctly
    for r in results:
        assert "animal" in r
        assert "session" in r
        assert "index" in r
        assert "path" in r

    # test filtering
    results_filtered = fd.discover(animal="A10")
    assert len(results_filtered) == 2
    assert all(r["animal"] == "A10" for r in results_filtered)


def test_multiple_file_discovery(tmp_path):
    (tmp_path / "A10" / "sess1").mkdir(parents=True)
    (tmp_path / "A10" / "sess1" / "data.bin").touch()
    (tmp_path / "A10" / "sess1" / "meta.json").touch()

    (tmp_path / "A10" / "sess2").mkdir(parents=True)
    (tmp_path / "A10" / "sess2" / "data.bin").touch()
    # Missing meta.json for sess2

    base = str(tmp_path)
    patterns = [
        f"{base}/{{animal}}/{{session}}/data.bin",
        f"{base}/{{animal}}/{{session}}/meta.json",
    ]

    fd = FileDiscoverer(patterns)
    results = fd.discover()

    # Should only find sess1 because sess2 is missing the meta.json
    assert len(results) == 1
    assert results[0]["animal"] == "A10"
    assert results[0]["session"] == "sess1"
    assert len(results[0]["paths"]) == 2
    assert results[0]["paths"][0].endswith("data.bin")
    assert results[0]["paths"][1].endswith("meta.json")


def test_folder_discovery(tmp_path):
    (tmp_path / "A10" / "2025-01-24").mkdir(parents=True)
    (tmp_path / "A10" / "2025-01-25").mkdir(parents=True)

    base = str(tmp_path)
    pattern = f"{base}/{{animal}}/{{session}}"

    fd = FileDiscoverer(pattern)
    results = fd.discover()

    assert len(results) == 2
    assert "path" in results[0]
    assert Path(results[0]["path"]).is_dir()


# ---------------------------------------------------------------------------
# DiscoveredFile edge cases
# ---------------------------------------------------------------------------


class TestDiscoveredFileEdgeCases:
    """Tests for DiscoveredFile error paths and dict-compat API."""

    def test_no_path_or_paths_raises(self):
        with pytest.raises(ValueError, match="Either path or paths must be provided"):
            DiscoveredFile()

    def test_both_path_and_paths_raises(self):
        with pytest.raises(ValueError, match="Cannot provide both path and paths"):
            DiscoveredFile(path="/a.txt", paths=("/a.txt", "/b.txt"))

    def test_fspath_single(self):
        df = DiscoveredFile(path="/data/file.rhd", metadata={"animal": "A10"})
        assert os.fspath(df) == "/data/file.rhd"

    def test_fspath_multi_raises(self):
        df = DiscoveredFile(
            paths=("/data/a.bin", "/data/a.csv"), metadata={"animal": "A10"}
        )
        with pytest.raises(TypeError, match="Multi-file DiscoveredFile"):
            os.fspath(df)

    def test_contains_and_getitem(self):
        df = DiscoveredFile(path="/f.rhd", metadata={"animal": "A10", "session": "s1"})
        assert "path" in df
        assert "paths" not in df
        assert "animal" in df
        assert df["path"] == "/f.rhd"
        assert df["animal"] == "A10"

    def test_contains_paths_key(self):
        df = DiscoveredFile(paths=("/a.bin",), metadata={})
        assert "paths" in df
        assert "path" not in df
        assert df["paths"] == ("/a.bin",)

    def test_is_multi_file(self):
        single = DiscoveredFile(path="/a.rhd")
        multi = DiscoveredFile(paths=("/a.bin", "/a.csv"))
        assert not single.is_multi_file
        assert multi.is_multi_file

    def test_get_path_list_single(self):
        df = DiscoveredFile(path="/a.rhd")
        assert df.get_path_list() == ["/a.rhd"]

    def test_get_path_list_multi(self):
        df = DiscoveredFile(paths=("/a.bin", "/a.csv"))
        assert df.get_path_list() == ["/a.bin", "/a.csv"]

    def test_iter(self):
        df = DiscoveredFile(paths=("/a.bin", "/a.csv"))
        assert list(df) == ["/a.bin", "/a.csv"]

    def test_repr_single(self):
        df = DiscoveredFile(path="/a.rhd", metadata={"animal": "X"})
        r = repr(df)
        assert "path=" in r
        assert "X" in r

    def test_repr_multi(self):
        df = DiscoveredFile(paths=("/a.bin",), metadata={"animal": "X"})
        r = repr(df)
        assert "paths=" in r

    def test_default_metadata_is_empty_dict(self):
        df = DiscoveredFile(path="/a.rhd")
        assert df.metadata == {}

    def test_getitem_missing_key_raises(self):
        df = DiscoveredFile(path="/a.rhd", metadata={"animal": "A"})
        with pytest.raises(KeyError):
            _ = df["nonexistent"]


class TestMultiFileGroupDeprecation:
    """MultiFileGroup should emit DeprecationWarning."""

    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mfg = MultiFileGroup(paths=("/a.bin", "/b.csv"), metadata={"animal": "A10"})
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            assert mfg.paths == ("/a.bin", "/b.csv")


# ---------------------------------------------------------------------------
# FileDiscoverer edge cases
# ---------------------------------------------------------------------------


class TestFileDiscovererEdgeCases:
    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            FileDiscoverer("")

    def test_empty_list_pattern_raises(self):
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            FileDiscoverer([])

    def test_no_matches_returns_empty(self, tmp_path):
        fd = FileDiscoverer(str(tmp_path / "{animal}" / "{session}.rhd"))
        assert fd.discover() == []

    def test_pattern_without_placeholders(self, tmp_path):
        (tmp_path / "data.bin").touch()
        fd = FileDiscoverer(str(tmp_path / "*.bin"))
        results = fd.discover()
        assert len(results) == 1
        assert "path" in results[0]

    def test_filter_no_match(self, tmp_path):
        (tmp_path / "A10" / "s1").mkdir(parents=True)
        (tmp_path / "A10" / "s1" / "1.rhd").touch()
        fd = FileDiscoverer(str(tmp_path / "{animal}" / "{session}" / "{index}.rhd"))
        assert fd.discover(animal="NONEXISTENT") == []

    def test_pathlib_pattern_accepted(self, tmp_path):
        (tmp_path / "a.txt").touch()
        fd = FileDiscoverer(Path(tmp_path / "*.txt"))
        assert len(fd.discover()) == 1

    def test_multi_pattern_empty_first_returns_empty(self, tmp_path):
        patterns = [
            str(tmp_path / "{animal}" / "data.bin"),
            str(tmp_path / "{animal}" / "meta.json"),
        ]
        fd = FileDiscoverer(patterns)
        assert fd.discover() == []

    def test_discover_sorts_deterministically(self, tmp_path):
        for name in ["c.txt", "a.txt", "b.txt"]:
            (tmp_path / name).touch()
        fd = FileDiscoverer(str(tmp_path / "*.txt"))
        results = fd.discover()
        paths = [r["path"] for r in results]
        assert paths == sorted(paths)
