"""Tests for LongRecordingOrganizer._extract_channel_names() static method."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from neurodent.loading.long_recording_organizer import LongRecordingOrganizer


class TestExtractChannelNames:
    def _make_recording(self, property_keys=None, channel_name_property=None,
                        channel_ids=None):
        """Build a minimal mock SI recording."""
        rec = MagicMock()
        if property_keys is not None:
            rec.get_property_keys.return_value = property_keys
        if channel_name_property is not None:
            rec.get_property.return_value = channel_name_property
        if channel_ids is not None:
            rec.get_channel_ids.return_value = channel_ids
        return rec

    def test_uses_channel_name_property_when_present(self):
        """If 'channel_name' is in property_keys, return those names."""
        rec = self._make_recording(
            property_keys=["channel_name", "gain"],
            channel_name_property=["C-009", "C-010", "C-011"],
            channel_ids=["0", "1", "2"],
        )
        names = LongRecordingOrganizer._extract_channel_names(rec)
        rec.get_property.assert_called_once_with("channel_name")
        assert names == ["C-009", "C-010", "C-011"]

    def test_channel_name_not_in_keys_falls_back_to_ids(self):
        """If 'channel_name' is not in property_keys, fall back to channel IDs."""
        rec = self._make_recording(
            property_keys=["gain"],
            channel_ids=["0", "1", "2"],
        )
        names = LongRecordingOrganizer._extract_channel_names(rec)
        assert names == ["0", "1", "2"]

    def test_attribute_error_on_get_property_keys_falls_back_to_ids(self):
        """If get_property_keys() raises AttributeError, fall back to channel IDs."""
        rec = MagicMock()
        rec.get_property_keys.side_effect = AttributeError("no property_keys")
        rec.get_channel_ids.return_value = ["ch0", "ch1"]
        names = LongRecordingOrganizer._extract_channel_names(rec)
        assert names == ["ch0", "ch1"]

    def test_type_error_on_get_property_keys_falls_back_to_ids(self):
        """If get_property_keys() raises TypeError, fall back to channel IDs."""
        rec = MagicMock()
        rec.get_property_keys.side_effect = TypeError("not callable")
        rec.get_channel_ids.return_value = ["x", "y"]
        names = LongRecordingOrganizer._extract_channel_names(rec)
        assert names == ["x", "y"]

    def test_integer_channel_ids_converted_to_strings(self, caplog):
        """Integer channel IDs should be converted to strings and log a warning."""
        import logging
        rec = self._make_recording(
            property_keys=["gain"],
            channel_ids=[0, 1, 2],
        )
        with caplog.at_level(logging.WARNING):
            names = LongRecordingOrganizer._extract_channel_names(rec)
        assert names == ["0", "1", "2"]
        assert "Channel IDs are integers" in caplog.text

    def test_integer_channel_ids_numpy_converted(self):
        """numpy integer channel IDs should also be converted to strings."""
        rec = self._make_recording(
            property_keys=[],
            channel_ids=np.array([0, 1, 2], dtype=np.int64),
        )
        names = LongRecordingOrganizer._extract_channel_names(rec)
        assert names == ["0", "1", "2"]

    def test_returns_strings(self):
        """All returned names must be plain Python strings."""
        rec = self._make_recording(
            property_keys=["channel_name"],
            channel_name_property=np.array(["A", "B"]),
            channel_ids=["0", "1"],
        )
        names = LongRecordingOrganizer._extract_channel_names(rec)
        assert all(isinstance(n, str) for n in names)
