"""Tests for the atomic-write / safe-delete helpers in neurodent.core.utils."""

import json
from pathlib import Path

import pytest

from neurodent import constants
from neurodent.core.utils import (
    atomic_output_path,
    atomic_write_json,
    is_si_recording_folder,
    safe_rmtree,
    safe_unlink,
)


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


def _make_zarr_folder(tmp_path, name="rec.zarr"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / ".zattrs").write_text("{}")
    return folder


def _make_binary_folder(tmp_path, name="rec_bin"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "si_folder.json").write_text("{}")
    return folder


class TestIsSiRecordingFolder:
    def test_zarr_folder_recognized(self, tmp_path):
        assert is_si_recording_folder(_make_zarr_folder(tmp_path)) is True

    def test_binary_folder_recognized(self, tmp_path):
        assert is_si_recording_folder(_make_binary_folder(tmp_path)) is True

    def test_neurodent_sidecar_recognized(self, tmp_path):
        folder = tmp_path / "anything"
        folder.mkdir()
        (folder / constants.NEURODENT_SIDECAR_NAME).write_text("{}")
        assert is_si_recording_folder(folder) is True

    def test_zarr_suffix_without_metadata_not_recognized(self, tmp_path):
        # .zarr suffix alone (no zarr metadata files) must not qualify.
        folder = tmp_path / "empty.zarr"
        folder.mkdir()
        assert is_si_recording_folder(folder) is False

    def test_foreign_dir_not_recognized(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        (folder / "notes.txt").write_text("hello")
        assert is_si_recording_folder(folder) is False

    def test_empty_dir_not_recognized(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        assert is_si_recording_folder(folder) is False

    def test_file_not_recognized(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("x")
        assert is_si_recording_folder(path) is False

    def test_missing_path_not_recognized(self, tmp_path):
        assert is_si_recording_folder(tmp_path / "nope") is False


class TestSafeRmtree:
    def test_removes_recognized_folder(self, tmp_path):
        folder = _make_binary_folder(tmp_path)
        safe_rmtree(folder)
        assert not folder.exists()

    def test_refuses_foreign_folder(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        sentinel = folder / "notes.txt"
        sentinel.write_text("keep")
        with pytest.raises(ValueError, match="Refusing to delete"):
            safe_rmtree(folder)
        assert sentinel.exists()

    def test_removes_foreign_folder_when_marker_not_required(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        (folder / "notes.txt").write_text("keep")
        safe_rmtree(folder, require_marker=False)
        assert not folder.exists()

    def test_missing_path_is_noop(self, tmp_path):
        # Should not raise.
        safe_rmtree(tmp_path / "does_not_exist")
