"""Tests for RecordingMetadata serialization methods: to_dict, from_dict, to_json,
from_json, update_sampling_rate, and DDFBinaryMetadata deprecation."""

import json
import warnings
from datetime import datetime
from pathlib import Path

import pytest

from neurodent.loading.long_recording_organizer import DDFBinaryMetadata, RecordingMetadata


def _make_meta(**overrides):
    """Factory for a minimal RecordingMetadata constructed from params."""
    defaults = dict(
        metadata_path=None,
        n_channels=4,
        f_s=1000.0,
        dt_end=datetime(2023, 6, 15, 12, 0, 0),
        channel_names=["ch1", "ch2", "ch3", "ch4"],
        V_units="µV",
        mult_to_uV=1.0,
    )
    defaults.update(overrides)
    return RecordingMetadata(**defaults)


class TestRecordingMetadataToDict:
    def test_returns_dict_with_expected_keys(self):
        meta = _make_meta()
        d = meta.to_dict()
        assert isinstance(d, dict)
        for key in ("metadata_path", "n_channels", "f_s", "V_units", "mult_to_uV",
                    "precision", "dt_end", "channel_names"):
            assert key in d

    def test_dt_end_is_isoformat_string(self):
        dt = datetime(2023, 6, 15, 12, 0, 0)
        meta = _make_meta(dt_end=dt)
        d = meta.to_dict()
        assert d["dt_end"] == dt.isoformat()

    def test_dt_end_none_serializes_as_none(self):
        meta = _make_meta(dt_end=None)
        d = meta.to_dict()
        assert d["dt_end"] is None

    def test_metadata_path_serialized_as_string(self):
        meta = _make_meta()
        meta.metadata_path = Path("/some/path/meta.csv")
        d = meta.to_dict()
        assert isinstance(d["metadata_path"], str)
        assert d["metadata_path"] == "/some/path/meta.csv"

    def test_metadata_path_none_stays_none(self):
        meta = _make_meta()
        d = meta.to_dict()
        assert d["metadata_path"] is None

    def test_values_match_attributes(self):
        meta = _make_meta()
        d = meta.to_dict()
        assert d["n_channels"] == meta.n_channels
        assert d["f_s"] == meta.f_s
        assert d["channel_names"] == meta.channel_names
        assert d["V_units"] == meta.V_units
        assert d["mult_to_uV"] == meta.mult_to_uV


class TestRecordingMetadataFromDict:
    def test_roundtrip_with_dt_end(self):
        meta = _make_meta()
        restored = RecordingMetadata.from_dict(meta.to_dict())
        assert restored.n_channels == meta.n_channels
        assert restored.f_s == meta.f_s
        assert restored.dt_end == meta.dt_end
        assert restored.channel_names == meta.channel_names

    def test_roundtrip_dt_end_none(self):
        meta = _make_meta(dt_end=None)
        restored = RecordingMetadata.from_dict(meta.to_dict())
        assert restored.dt_end is None
        assert restored.n_channels == meta.n_channels

    def test_v_units_preserved(self):
        meta = _make_meta(V_units="mV")
        restored = RecordingMetadata.from_dict(meta.to_dict())
        assert restored.V_units == "mV"

    def test_mult_to_uv_preserved(self):
        meta = _make_meta(mult_to_uV=1000.0)
        restored = RecordingMetadata.from_dict(meta.to_dict())
        assert restored.mult_to_uV == 1000.0


class TestRecordingMetadataToJson:
    def test_writes_valid_json_file(self, tmp_path):
        meta = _make_meta()
        json_path = tmp_path / "meta.json"
        meta.to_json(json_path)
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["n_channels"] == meta.n_channels

    def test_json_content_matches_to_dict(self, tmp_path):
        meta = _make_meta()
        json_path = tmp_path / "meta.json"
        meta.to_json(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data == meta.to_dict()


class TestRecordingMetadataFromJson:
    def test_roundtrip(self, tmp_path):
        meta = _make_meta()
        json_path = tmp_path / "meta.json"
        meta.to_json(json_path)
        restored = RecordingMetadata.from_json(json_path)
        assert restored.n_channels == meta.n_channels
        assert restored.f_s == meta.f_s
        assert restored.dt_end == meta.dt_end
        assert restored.channel_names == meta.channel_names

    def test_v_units_preserved(self, tmp_path):
        meta = _make_meta(V_units="mV")
        json_path = tmp_path / "meta.json"
        meta.to_json(json_path)
        restored = RecordingMetadata.from_json(json_path)
        assert restored.V_units == "mV"

    def test_mult_to_uv_preserved(self, tmp_path):
        meta = _make_meta(mult_to_uV=500.0)
        json_path = tmp_path / "meta.json"
        meta.to_json(json_path)
        restored = RecordingMetadata.from_json(json_path)
        assert restored.mult_to_uV == 500.0

    def test_precision_preserved(self, tmp_path):
        meta = _make_meta()
        meta.precision = "int16"
        json_path = tmp_path / "meta.json"
        meta.to_json(json_path)
        restored = RecordingMetadata.from_json(json_path)
        assert restored.precision == "int16"

    def test_roundtrip_dt_end_none(self, tmp_path):
        meta = _make_meta(dt_end=None)
        json_path = tmp_path / "meta.json"
        meta.to_json(json_path)
        restored = RecordingMetadata.from_json(json_path)
        assert restored.dt_end is None


class TestRecordingMetadataUpdateSamplingRate:
    def test_updates_f_s(self):
        meta = _make_meta(f_s=500.0)
        meta.update_sampling_rate(2000.0)
        assert meta.f_s == 2000.0

    def test_original_f_s_replaced(self):
        meta = _make_meta(f_s=1000.0)
        meta.update_sampling_rate(250.0)
        assert meta.f_s != 1000.0
        assert meta.f_s == 250.0


class TestDDFBinaryMetadataDeprecation:
    def test_raises_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="DDFBinaryMetadata is deprecated"):
            DDFBinaryMetadata(
                None,
                n_channels=2,
                f_s=1000.0,
                dt_end=None,
                channel_names=["a", "b"],
            )

    def test_is_instance_of_recording_metadata(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            obj = DDFBinaryMetadata(
                None,
                n_channels=2,
                f_s=1000.0,
                dt_end=None,
                channel_names=["a", "b"],
            )
        assert isinstance(obj, RecordingMetadata)
        assert obj.n_channels == 2
        assert obj.f_s == 1000.0
