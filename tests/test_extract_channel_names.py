"""Tests for LongRecordingOrganizer._extract_channel_identities() static method.

Returns ``(channel_ids, channel_names)``: ``channel_ids`` are always the recording's stable
identifiers (``get_channel_ids()``, what configs key on); ``channel_names`` are display labels
(the ``channel_name`` property when a reader sets one, else the ids).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from neurodent.loading.long_recording_organizer import LongRecordingOrganizer


class TestExtractChannelIdentities:
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

    def test_ids_from_get_channel_ids_names_from_property(self):
        """channel_ids come from get_channel_ids(); display names from the channel_name property."""
        rec = self._make_recording(
            property_keys=["channel_name", "gain"],
            channel_name_property=["C-009", "C-010", "C-011"],
            channel_ids=["0", "1", "2"],
        )
        ids, names = LongRecordingOrganizer._extract_channel_identities(rec)
        rec.get_property.assert_called_once_with("channel_name")
        assert ids == ["0", "1", "2"]                    # stable identity
        assert names == ["C-009", "C-010", "C-011"]      # display

    def test_names_default_to_ids_when_no_property(self):
        """If 'channel_name' is not in property_keys, display names fall back to the ids."""
        rec = self._make_recording(
            property_keys=["gain"],
            channel_ids=["0", "1", "2"],
        )
        ids, names = LongRecordingOrganizer._extract_channel_identities(rec)
        assert ids == ["0", "1", "2"]
        assert names == ["0", "1", "2"]

    def test_attribute_error_on_get_property_keys_falls_back_to_ids(self):
        """If get_property_keys() raises AttributeError, names fall back to channel IDs."""
        rec = MagicMock()
        rec.get_property_keys.side_effect = AttributeError("no property_keys")
        rec.get_channel_ids.return_value = ["ch0", "ch1"]
        ids, names = LongRecordingOrganizer._extract_channel_identities(rec)
        assert ids == ["ch0", "ch1"]
        assert names == ["ch0", "ch1"]

    def test_type_error_on_get_property_keys_falls_back_to_ids(self):
        """If get_property_keys() raises TypeError, names fall back to channel IDs."""
        rec = MagicMock()
        rec.get_property_keys.side_effect = TypeError("not callable")
        rec.get_channel_ids.return_value = ["x", "y"]
        ids, names = LongRecordingOrganizer._extract_channel_identities(rec)
        assert ids == ["x", "y"]
        assert names == ["x", "y"]

    def test_integer_channel_ids_converted_to_strings(self, caplog):
        """Integer channel IDs should be converted to strings and log a warning."""
        import logging
        rec = self._make_recording(
            property_keys=["gain"],
            channel_ids=[0, 1, 2],
        )
        with caplog.at_level(logging.WARNING):
            ids, names = LongRecordingOrganizer._extract_channel_identities(rec)
        assert ids == ["0", "1", "2"]
        assert names == ["0", "1", "2"]
        assert "Channel IDs are integers" in caplog.text

    def test_integer_channel_ids_numpy_converted(self):
        """numpy integer channel IDs should also be converted to strings."""
        rec = self._make_recording(
            property_keys=[],
            channel_ids=np.array([0, 1, 2], dtype=np.int64),
        )
        ids, names = LongRecordingOrganizer._extract_channel_identities(rec)
        assert ids == ["0", "1", "2"]
        assert names == ["0", "1", "2"]

    def test_returns_strings(self):
        """Both ids and display names must be plain Python strings."""
        rec = self._make_recording(
            property_keys=["channel_name"],
            channel_name_property=np.array(["A", "B"]),
            channel_ids=np.array(["0", "1"]),
        )
        ids, names = LongRecordingOrganizer._extract_channel_identities(rec)
        assert all(isinstance(n, str) for n in names)
        assert all(isinstance(i, str) for i in ids)
