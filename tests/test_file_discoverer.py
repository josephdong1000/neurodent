import pytest
from pathlib import Path
from neurodent.core.discovery import FileDiscoverer


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
