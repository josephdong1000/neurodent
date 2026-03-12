"""Tests for the split_recording() standalone convenience function."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neurodent.core.core import split_recording


def _patch_lro(mock_lro):
    """Return a patch context manager that replaces LongRecordingOrganizer in
    split_recording's own globals dict.

    This approach is robust even when test_imports.py reloads the
    neurodent.core.core module (which changes the sys.modules entry but not
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

    def test_persist_true_without_output_base_raises_value_error(self):
        """persist=True without output_base must raise ValueError."""
        mock_lro, _ = self._make_mock_lro()
        with _patch_lro(mock_lro):
            with pytest.raises(ValueError, match="output_base is required"):
                split_recording(
                    "/fake/path",
                    groups={"A": ["ch1"], "B": ["ch2"]},
                    output_base=None,
                    persist=True,
                )

    def test_persist_false_returns_splits_without_saving(self):
        """persist=False should return the splits dict and not call persist()."""
        mock_lro, splits = self._make_mock_lro()
        with _patch_lro(mock_lro):
            result = split_recording(
                "/fake/path",
                groups={"AnimalA": ["ch1"], "AnimalB": ["ch2"]},
                persist=False,
            )
        assert result is splits
        for child_lro in splits.values():
            child_lro.persist.assert_not_called()

    def test_persist_true_with_output_base_calls_persist(self, tmp_path):
        """persist=True with output_base should call persist() on each child LRO."""
        mock_lro, splits = self._make_mock_lro()
        with _patch_lro(mock_lro):
            result = split_recording(
                "/fake/path",
                groups={"AnimalA": ["ch1"], "AnimalB": ["ch2"]},
                output_base=tmp_path,
                persist=True,
                format="zarr",
            )
        assert result is splits
        for group_name, child_lro in splits.items():
            expected_dir = tmp_path / group_name
            child_lro.persist.assert_called_once_with(expected_dir, format="zarr")
