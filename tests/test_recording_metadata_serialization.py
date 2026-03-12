"""
Tests for RecordingMetadata serialization (to_dict / from_dict).

Ensures that serialized paths always use forward slashes regardless of the
operating system, making JSON/dict output cross-platform consistent.
"""

import pytest
from pathlib import Path

from neurodent.core.core import RecordingMetadata


class TestRecordingMetadataToDict:
    """Tests for RecordingMetadata.to_dict() serialization."""

    def _make_metadata(self, **kwargs):
        """Helper to build a RecordingMetadata without a CSV file."""
        defaults = dict(
            metadata_path=None,
            n_channels=16,
            f_s=30000.0,
            V_units="uV",
            mult_to_uV=1.0,
            channel_names=[f"ch{i}" for i in range(16)],
        )
        defaults.update(kwargs)
        return RecordingMetadata(**defaults)

    def test_metadata_path_serialized_as_string(self):
        """metadata_path must serialize with forward slashes on every OS."""
        meta = self._make_metadata()
        meta.metadata_path = Path("/some/path/meta.csv")
        result = meta.to_dict()
        assert result["metadata_path"] == "/some/path/meta.csv"

    def test_metadata_path_none_serializes_as_none(self):
        """metadata_path=None must serialize as None, not the string 'None'."""
        meta = self._make_metadata()
        result = meta.to_dict()
        assert result["metadata_path"] is None

    def test_to_dict_contains_expected_keys(self):
        """to_dict() must include all required serialization keys."""
        meta = self._make_metadata()
        result = meta.to_dict()
        for key in ("metadata_path", "n_channels", "f_s", "V_units", "mult_to_uV"):
            assert key in result, f"Missing key: {key}"
