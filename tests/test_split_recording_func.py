"""Tests for the split_recording() standalone convenience function."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neurodent.loading.long_recording_organizer import split_recording


def _patch_lro(mock_lro):
    """Return a patch context manager that replaces LongRecordingOrganizer in
    split_recording's own globals dict.

    This approach is robust even when test_imports.py reloads the
    neurodent.loading.long_recording_organizer module (which changes the sys.modules entry but not
    the ``split_recording.__globals__`` dict that this function already has).
    """
    mock_class = MagicMock(return_value=mock_lro)
    return patch.dict(split_recording.__globals__, {"LongRecordingOrganizer": mock_class})


class TestSplitRecordingFunction:
    """Tests for split_recording() that mock LongRecordingOrganizer to avoid I/O."""

    def _make_mock_lro(self, splits=None):
        """Return a mock LRO whose .split() returns a dict of mock child LROs."""
        if splits is None:
            splits = {
                "AnimalA": MagicMock(name="lro_A"),
                "AnimalB": MagicMock(name="lro_B"),
            }
        lro = MagicMock()
        lro.split.return_value = splits
        return lro, splits

    def test_save_true_without_output_base_raises_value_error(self):
        """save=True without output_base must raise ValueError."""
        mock_lro, _ = self._make_mock_lro()
        with _patch_lro(mock_lro):
            with pytest.raises(ValueError, match="output_base is required"):
                split_recording(
                    "/fake/path",
                    groups={"A": ["ch1"], "B": ["ch2"]},
                    output_base=None,
                    save=True,
                )

    def test_save_false_returns_splits_without_saving(self):
        """save=False should return the splits dict and not call save_recording()."""
        mock_lro, splits = self._make_mock_lro()
        with _patch_lro(mock_lro):
            result = split_recording(
                "/fake/path",
                groups={"AnimalA": ["ch1"], "AnimalB": ["ch2"]},
                save=False,
            )
        assert result is splits
        for child_lro in splits.values():
            child_lro.save_recording.assert_not_called()

    def test_save_true_with_output_base_calls_save_recording(self, tmp_path):
        """save=True with output_base should call save_recording() on each child LRO."""
        mock_lro, splits = self._make_mock_lro()
        with _patch_lro(mock_lro):
            result = split_recording(
                "/fake/path",
                groups={"AnimalA": ["ch1"], "AnimalB": ["ch2"]},
                output_base=tmp_path,
                save=True,
                format="zarr",
            )
        assert result is splits
        for group_name, child_lro in splits.items():
            expected_dir = tmp_path / group_name
            child_lro.save_recording.assert_called_once_with(
                expected_dir, format="zarr", overwrite=False
            )

    def test_persist_alias_is_deprecated_but_works(self, tmp_path):
        """The deprecated persist= alias still drives save_recording() with a warning."""
        mock_lro, splits = self._make_mock_lro()
        with _patch_lro(mock_lro):
            with pytest.warns(DeprecationWarning, match="persist"):
                split_recording(
                    "/fake/path",
                    groups={"AnimalA": ["ch1"]},
                    output_base=tmp_path,
                    persist=True,
                    format="zarr",
                )
        for group_name, child_lro in splits.items():
            child_lro.save_recording.assert_called_once_with(
                tmp_path / group_name, format="zarr", overwrite=False
            )
