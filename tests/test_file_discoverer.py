import os

import pytest
from pathlib import Path
from neurodent.loading.discovery import DiscoveredFile, FileDiscoverer


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


def test_redundant_slashes_in_pattern_still_match(tmp_path):
    """A base dir with a trailing slash yields 'base//...'; glob collapses '//' but the discovery
    regex must too, or it silently matches zero files. Regression for the ap3b2_rhd 0-files bug."""
    (tmp_path / "PortA-A10-PortB-B20").mkdir(parents=True)
    (tmp_path / "PortA-A10-PortB-B20" / "rec_1.rhd").touch()
    (tmp_path / "PortA-A10-PortB-B20" / "rec_2.rhd").touch()

    base = str(tmp_path).rstrip("/") + "/"            # base WITH a trailing slash (idiomatic)
    pattern = f"{base}/*{{animal}}*/{{index}}.rhd"    # -> 'base//*{animal}*/...' (double slash)

    fd = FileDiscoverer(pattern)
    results = fd.discover(animal="A10")
    assert len(results) == 2, f"double slash in pattern matched {len(results)} files, expected 2"
    assert all(r["index"].startswith("rec_") for r in results)


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

    # -- exotic placeholder patterns -----------------------------------------

    def test_session_before_animal_placeholder_order(self, tmp_path):
        """Placeholders in {session}/{animal}/{index} order."""
        (tmp_path / "s1" / "A10").mkdir(parents=True)
        (tmp_path / "s1" / "A10" / "001.rhd").touch()
        (tmp_path / "s2" / "B20").mkdir(parents=True)
        (tmp_path / "s2" / "B20" / "002.rhd").touch()
        fd = FileDiscoverer(str(tmp_path / "{session}" / "{animal}" / "{index}.rhd"))
        results = fd.discover()
        assert len(results) == 2
        meta = {r.metadata["animal"] for r in results}
        assert meta == {"A10", "B20"}
        for r in results:
            assert "session" in r.metadata
            assert "index" in r.metadata

    def test_index_animal_session_reversed_order(self, tmp_path):
        """Fully reversed: {index}/{animal}/{session}.edf"""
        (tmp_path / "001" / "A10" / "baseline").mkdir(parents=True)
        (tmp_path / "001" / "A10" / "baseline" / "data.edf").write_text("")
        fd = FileDiscoverer(
            str(tmp_path / "{index}" / "{animal}" / "{session}" / "data.edf")
        )
        results = fd.discover()
        assert len(results) == 1
        assert results[0].metadata == {"index": "001", "animal": "A10", "session": "baseline"}

    def test_placeholder_in_filename_and_directory(self, tmp_path):
        """Placeholder embedded in filename: {animal}_{session}.nwb"""
        (tmp_path / "A10_s1.nwb").touch()
        (tmp_path / "B20_s2.nwb").touch()
        fd = FileDiscoverer(str(tmp_path / "{animal}_{session}.nwb"))
        results = fd.discover()
        assert len(results) == 2
        animals = {r.metadata["animal"] for r in results}
        sessions = {r.metadata["session"] for r in results}
        assert animals == {"A10", "B20"}
        assert sessions == {"s1", "s2"}

    def test_deeply_nested_pattern(self, tmp_path):
        """Four-level nesting: {project}/{animal}/{session}/{index}.bin"""
        (tmp_path / "proj1" / "A10" / "day1" / "rec001").mkdir(parents=True)
        (tmp_path / "proj1" / "A10" / "day1" / "rec001" / "trace.bin").touch()
        fd = FileDiscoverer(
            str(tmp_path / "{project}" / "{animal}" / "{session}" / "{index}" / "trace.bin")
        )
        results = fd.discover()
        assert len(results) == 1
        m = results[0].metadata
        assert m == {"project": "proj1", "animal": "A10", "session": "day1", "index": "rec001"}

    def test_adjacent_placeholders_in_filename(self, tmp_path):
        """Two placeholders separated by hyphen in file name: {animal}-{session}.rhd"""
        (tmp_path / "A10-baseline.rhd").touch()
        fd = FileDiscoverer(str(tmp_path / "{animal}-{session}.rhd"))
        results = fd.discover()
        assert len(results) == 1
        assert results[0].metadata == {"animal": "A10", "session": "baseline"}

    def test_single_placeholder_only(self, tmp_path):
        """Pattern with just one placeholder: {animal}.csv"""
        (tmp_path / "A10.csv").touch()
        (tmp_path / "B20.csv").touch()
        fd = FileDiscoverer(str(tmp_path / "{animal}.csv"))
        results = fd.discover()
        assert len(results) == 2
        assert {r.metadata["animal"] for r in results} == {"A10", "B20"}
