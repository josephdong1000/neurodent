"""Tests for the atomic-write / safe-delete helpers in neurodent.core.utils."""

import json
from pathlib import Path

import pytest

from neurodent.core.utils import atomic_output_path, atomic_write_json, safe_unlink


class TestAtomicOutputPath:
    def test_commits_on_success(self, tmp_path):
        """On clean exit the temp file is renamed into place, with no leftover."""
        final = tmp_path / "data.bin"
        with atomic_output_path(final) as tmp:
            assert Path(tmp) != final
            Path(tmp).write_bytes(b"payload")
            assert not final.exists()  # not visible at the final path yet
        assert final.read_bytes() == b"payload"
        # No stray temp files left behind.
        assert list(tmp_path.glob("*.tmp")) == []

    def test_cleans_up_and_preserves_final_on_error(self, tmp_path):
        """An exception removes the temp file and leaves the final path untouched."""
        final = tmp_path / "data.bin"
        final.write_bytes(b"original")
        with pytest.raises(RuntimeError, match="boom"):
            with atomic_output_path(final) as tmp:
                Path(tmp).write_bytes(b"partial")
                raise RuntimeError("boom")
        # Original content preserved; partial temp gone.
        assert final.read_bytes() == b"original"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_does_not_create_final_on_error(self, tmp_path):
        """If the final did not exist, a failed write must not create it."""
        final = tmp_path / "data.bin"
        with pytest.raises(ValueError):
            with atomic_output_path(final) as tmp:
                Path(tmp).write_bytes(b"partial")
                raise ValueError
        assert not final.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_overwrites_existing_atomically(self, tmp_path):
        """Writing over an existing file replaces it with the new content."""
        final = tmp_path / "data.bin"
        final.write_bytes(b"old")
        with atomic_output_path(final) as tmp:
            Path(tmp).write_bytes(b"new")
        assert final.read_bytes() == b"new"
        assert list(tmp_path.glob("*.tmp")) == []


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "meta.json"
        obj = {"a": 1, "b": ["x", "y"], "c": None}
        atomic_write_json(path, obj)
        assert json.loads(path.read_text()) == obj
        assert list(tmp_path.glob("*.tmp")) == []

    def test_replaces_existing(self, tmp_path):
        path = tmp_path / "meta.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}

    def test_does_not_clobber_on_serialization_error(self, tmp_path):
        """A non-serializable object must not corrupt or remove an existing file."""
        path = tmp_path / "meta.json"
        atomic_write_json(path, {"v": 1})
        with pytest.raises(TypeError):
            atomic_write_json(path, {"bad": object()})
        assert json.loads(path.read_text()) == {"v": 1}
        assert list(tmp_path.glob("*.tmp")) == []


class TestSafeUnlink:
    def test_removes_existing_file(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("x")
        safe_unlink(path)
        assert not path.exists()

    def test_missing_file_is_noop(self, tmp_path):
        # Should not raise.
        safe_unlink(tmp_path / "does_not_exist.txt")
