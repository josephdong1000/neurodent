"""
Tests targeting uncovered lines in neurodent.core.core.
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import mne
import numpy as np
import pandas as pd
import pytest

try:
    import spikeinterface.core as si
except ImportError:
    si = None

from neurodent.core.core import (
    RecordingMetadata,
    DDFBinaryMetadata,
    LongRecordingOrganizer,
    convert_ddfrowbin_to_si,
    _convert_ddfrowbin_to_si_no_resample,
)
from neurodent import constants

_DEFAULT_CH_NAMES = ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lro(item=None, **kwargs):
    """Create a LongRecordingOrganizer without loading data."""
    return LongRecordingOrganizer(item, mode=None, **kwargs)


def _make_mock_recording(
    n_channels=4,
    sfreq=None,
    duration=10.0,
    dtype=constants.GLOBAL_DTYPE,
    total_samples=None,
):
    """Return a Mock that mimics a SpikeInterface BaseRecording."""
    sfreq = sfreq or constants.GLOBAL_SAMPLING_RATE
    total_samples = int(duration * sfreq) if total_samples is None else total_samples
    rec = Mock()
    rec.get_num_channels.return_value = n_channels
    rec.get_sampling_frequency.return_value = sfreq
    rec.get_duration.return_value = duration
    rec.get_total_duration.return_value = duration
    rec.get_total_samples.return_value = total_samples
    rec.get_dtype.return_value = np.dtype(dtype)
    rec.get_channel_ids.return_value = [str(i) for i in range(n_channels)]
    rec.get_property_keys.return_value = []
    rec.has_scaleable_traces.return_value = False
    rec.get_traces.return_value = np.zeros((total_samples, n_channels), dtype=dtype)
    return rec


def _make_mock_mne_raw(n_channels=4, sfreq=None, preload=True, ch_prefix="EEG"):
    """Return a Mock that mimics an mne.io.Raw object."""
    sfreq = sfreq or constants.GLOBAL_SAMPLING_RATE
    ch_names = [f"{ch_prefix}{i}" for i in range(n_channels)]
    raw = MagicMock(spec=mne.io.Raw)
    raw.info = {
        "sfreq": sfreq,
        "nchan": n_channels,
        "ch_names": ch_names,
    }
    raw.preload = preload
    raw.get_data.return_value = np.zeros((n_channels, int(sfreq * 2)))
    raw.resample.return_value = raw
    return raw


def _export_creates_file(path, *args, **kwargs):
    """Side effect for a mocked ``mne.export.export_raw``.

    The intermediate file is now written via ``atomic_output_path`` (write to a
    temp sibling, then rename into place), so a mocked exporter must actually
    create the file at the path it is given for the rename to succeed.
    """
    Path(path).write_bytes(b"")


# ===================================================================
# RecordingMetadata
# ===================================================================


class TestRecordingMetadataEmptyCSV:
    """Line 125: ValueError when CSV has header but no data rows."""

    def test_header_only_csv(self, tmp_path):
        csv_path = tmp_path / "header_only.csv"
        pd.DataFrame(
            columns=["ProbeInfo", "SampleRate", "Units", "Precision", "LastEdit"]
        ).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Metadata file is empty"):
            RecordingMetadata(csv_path)


class TestRecordingMetadataEmptyColumn:
    """Line 172: __getsinglecolval returns None when column is empty."""

    def test_empty_column_returns_none(self, tmp_path):
        csv_path = tmp_path / "meta.csv"
        df = pd.DataFrame(
            {
                "ProbeInfo": ["ch0"],
                "SampleRate": [1000],
                "Units": ["µV"],
                "Precision": ["float32"],
                "LastEdit": ["2023-01-01T12:00:00"],
            }
        )
        df.to_csv(csv_path, index=False)
        meta = RecordingMetadata(csv_path)
        # Manually call with an empty slice to trigger size == 0
        empty_df = pd.DataFrame({"SampleRate": pd.Series([], dtype=float)})
        meta.metadata_df = empty_df
        result = meta._RecordingMetadata__getsinglecolval("SampleRate")
        assert result is None


# ===================================================================
# convert_ddfrowbin_to_si  /  _convert_ddfrowbin_to_si_no_resample
# ===================================================================


class TestConvertDdfrowbinImportError:
    """Lines 289, 359: ImportError when se is None."""

    @patch("neurodent.core.core.se", None)
    def test_convert_ddfrowbin_to_si_import_error(self):
        meta = RecordingMetadata(
            None, n_channels=2, f_s=1000, dt_end=None, channel_names=["a", "b"]
        )
        with pytest.raises(ImportError, match="SpikeInterface is required"):
            convert_ddfrowbin_to_si("dummy.bin", meta)

    @patch("neurodent.core.core.se", None)
    def test_convert_no_resample_import_error(self):
        meta = RecordingMetadata(
            None, n_channels=2, f_s=1000, dt_end=None, channel_names=["a", "b"]
        )
        with pytest.raises(ImportError, match="SpikeInterface is required"):
            _convert_ddfrowbin_to_si_no_resample("dummy.bin", meta)


class TestConvertDdfrowbinNoResamplePaths:
    """Lines 359-406: .npy.gz and .bin paths in _convert_ddfrowbin_to_si_no_resample."""

    @pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
    def test_bin_direct_read(self, tmp_path):
        """Test .bin direct read path (line 399-401)."""
        n_ch, n_samples = 2, 100
        data = np.zeros((n_samples, n_ch), dtype="float32")
        bin_path = tmp_path / "test.bin"
        data.tofile(str(bin_path))
        meta = RecordingMetadata(
            None,
            n_channels=n_ch,
            f_s=constants.GLOBAL_SAMPLING_RATE,
            dt_end=None,
            channel_names=["a", "b"],
            V_units="µV",
            mult_to_uV=1.0,
        )
        meta.precision = "float32"
        rec, temppath = _convert_ddfrowbin_to_si_no_resample(str(bin_path), meta)
        assert temppath is None
        assert rec.get_num_channels() == n_ch


# ===================================================================
# LongRecordingOrganizer.display_name
# ===================================================================


class TestDisplayName:
    """Line 642: display_name when item is a list/tuple."""

    def test_display_name_list(self):
        lro = _make_lro(item=["/data/file1.bin", "/data/file2.bin"])
        assert lro.display_name == "file1.bin"

    def test_display_name_tuple(self):
        lro = _make_lro(item=("/data/a.edf",))
        assert lro.display_name == "a.edf"

    def test_display_name_single_path(self):
        lro = _make_lro(item="/data/single.edf")
        assert lro.display_name == "single.edf"

    def test_display_name_none(self):
        lro = _make_lro(item=None)
        assert lro.display_name == "unknown"


# ===================================================================
# _resolve_func_path
# ===================================================================


class TestResolveFuncPath:
    """Line 679: ImportError when spec is None (non-existent file path)."""

    def test_no_colon_separator(self):
        with pytest.raises(ImportError, match="expected"):
            LongRecordingOrganizer._resolve_func_path("no_colon_here")

    def test_nonexistent_file_no_extension(self):
        """spec_from_file_location returns None for files without .py extension."""
        with pytest.raises(ImportError, match="Cannot load module"):
            LongRecordingOrganizer._resolve_func_path(
                "/nonexistent/path/readers:func"
            )


# ===================================================================
# detect_and_load_data
# ===================================================================


class TestDetectAndLoadData:
    """Lines 720, 730, 752, 765."""

    @patch("neurodent.core.core.si", None)
    def test_si_mode_import_error(self):
        lro = _make_lro(item="dummy")
        with pytest.raises(ImportError, match="SpikeInterface is required"):
            lro.detect_and_load_data(mode="si")

    def test_si_mode_unresolvable_extract_func(self):
        lro = _make_lro(item="dummy")
        with pytest.raises(ValueError, match="Could not resolve extractor function"):
            lro.detect_and_load_data(mode="si", extract_func="nonexistent_func_xyz")

    def test_mne_mode_unresolvable_extract_func(self):
        lro = _make_lro(item="dummy")
        with pytest.raises(ValueError, match="Could not resolve extractor function"):
            lro.detect_and_load_data(mode="mne", extract_func="nonexistent_func_xyz")

    def test_none_mode_passes(self):
        lro = _make_lro(item="dummy")
        lro.detect_and_load_data(mode=None)
        assert lro.LongRecording is None


# ===================================================================
# convert_file_with_si_to_recording
# ===================================================================


class TestConvertFileWithSI:
    """Lines 779, 801-805, 815-818, 821, 837-838."""

    @patch("neurodent.core.core.si", None)
    def test_import_error_when_si_none(self):
        lro = _make_lro(item="dummy")
        with pytest.raises(ImportError, match="SpikeInterface is required"):
            lro.convert_file_with_si_to_recording(extract_func=Mock())

    @pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
    def test_dask_parallel_mode(self):
        """Lines 800-805: dask multiprocess_mode path."""
        rec = _make_mock_recording()
        extract_func = Mock(return_value=rec)
        lro = _make_lro(item=["f1.bin", "f2.bin"])
        with patch("neurodent.core.core.dask") as mock_dask:
            mock_dask.delayed.return_value = Mock(return_value=rec)
            mock_dask.compute.return_value = [rec, rec]
            # Mock si.concatenate_recordings
            with patch("neurodent.core.core.si") as mock_si:
                concat_rec = _make_mock_recording()
                mock_si.concatenate_recordings.return_value = concat_rec
                with patch.object(lro, "_apply_resampling", return_value=concat_rec):
                    with patch.object(lro, "finalize_file_timestamps"):
                        with patch.object(lro, "_extract_channel_names", return_value=_DEFAULT_CH_NAMES):
                            lro.convert_file_with_si_to_recording(
                                extract_func=extract_func,
                                multiprocess_mode="dask",
                            )
            assert lro.LongRecording is not None

    @pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
    def test_zero_sample_filtering(self):
        """Lines 814-818: filter out 0-sample recordings in list mode."""
        good_rec = _make_mock_recording(total_samples=1000)
        empty_rec = _make_mock_recording(total_samples=0)
        extract_func = Mock(side_effect=[empty_rec, good_rec])
        lro = _make_lro(item=["f1.bin", "f2.bin"])
        concat_rec = _make_mock_recording()
        with patch("neurodent.core.core.si") as mock_si:
            mock_si.concatenate_recordings.return_value = concat_rec
            with patch.object(lro, "_apply_resampling", return_value=concat_rec):
                with patch.object(lro, "finalize_file_timestamps"):
                    with patch.object(lro, "_extract_channel_names", return_value=_DEFAULT_CH_NAMES):
                        lro.convert_file_with_si_to_recording(
                            extract_func=extract_func
                        )
        # The empty recording was skipped, only good_rec passed through
        mock_si.concatenate_recordings.assert_called_once()

    @pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
    def test_all_zero_samples_error(self):
        """Line 821: ValueError when all recordings have 0 samples."""
        empty_rec = _make_mock_recording(total_samples=0)
        extract_func = Mock(return_value=empty_rec)
        lro = _make_lro(item=["f1.bin", "f2.bin"])
        with patch("neurodent.core.core.si"):
            with pytest.raises(ValueError, match="All recordings.*0 samples"):
                lro.convert_file_with_si_to_recording(extract_func=extract_func)

    @pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
    def test_try_except_non_si_recording(self):
        """Lines 837-838: TypeError/AttributeError in 0-sample check for single file."""
        rec = Mock()
        rec.get_total_samples.side_effect = TypeError("not SI")
        rec.get_num_channels.return_value = 2
        rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        rec.get_duration.return_value = 5.0
        rec.get_dtype.return_value = np.dtype("float32")
        rec.get_channel_ids.return_value = ["0", "1"]
        rec.get_property_keys.return_value = []
        rec.has_scaleable_traces.return_value = False
        extract_func = Mock(return_value=rec)
        lro = _make_lro(item="single.bin")
        with patch("neurodent.core.core.si"):
            with patch.object(lro, "_apply_resampling", return_value=rec):
                with patch.object(lro, "finalize_file_timestamps"):
                    with patch.object(lro, "_extract_channel_names", return_value=["a", "b"]):
                        lro.convert_file_with_si_to_recording(extract_func=extract_func)
        assert lro.LongRecording is rec


# ===================================================================
# _load_and_process_mne_data
# ===================================================================


class TestLoadAndProcessMneData:
    """Lines 883-930: all input_types, preload, resampling."""

    def _get_lro(self):
        return _make_lro(item="dummy")

    def test_folder_input_type(self):
        lro = self._get_lro()
        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)
        result = lro._load_and_process_mne_data(
            extract_func, "folder", "/data/folder", None, None, n_jobs=1
        )
        extract_func.assert_called_once_with("/data/folder")
        assert result is raw

    def test_file_input_type(self):
        lro = self._get_lro()
        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)
        result = lro._load_and_process_mne_data(
            extract_func, "file", None, "/data/file.edf", None, n_jobs=1
        )
        extract_func.assert_called_once_with("/data/file.edf")

    def test_files_input_type(self):
        lro = self._get_lro()
        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)
        with patch("neurodent.core.core.mne") as mock_mne:
            mock_mne.concatenate_raws.return_value = raw
            result = lro._load_and_process_mne_data(
                extract_func,
                "files",
                None,
                None,
                ["/f1.edf", "/f2.edf"],
                n_jobs=1,
            )
        assert extract_func.call_count == 2

    def test_invalid_input_type(self):
        lro = self._get_lro()
        with pytest.raises(ValueError, match="Invalid input_type"):
            lro._load_and_process_mne_data(Mock(), "bogus", None, None, None, 1)

    def test_preload_triggers_load_data(self):
        lro = self._get_lro()
        raw = _make_mock_mne_raw(preload=False)
        extract_func = Mock(return_value=raw)
        lro._load_and_process_mne_data(
            extract_func, "file", None, "f.edf", None, n_jobs=1
        )
        raw.load_data.assert_called_once()

    def test_resampling_when_sfreq_differs(self):
        lro = self._get_lro()
        raw = _make_mock_mne_raw(sfreq=500.0)
        extract_func = Mock(return_value=raw)
        lro._load_and_process_mne_data(
            extract_func, "file", None, "f.edf", None, n_jobs=2
        )
        raw.resample.assert_called_once()


# ===================================================================
# _load_mne_data_no_resample
# ===================================================================


class TestLoadMneDataNoResample:
    """Lines 939-943, 951-952: list path with concatenation, preload."""

    def test_list_path_concatenation(self):
        lro = _make_lro(item=["f1.edf", "f2.edf"])
        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)
        with patch("neurodent.core.core.mne") as mock_mne:
            mock_mne.concatenate_raws.return_value = raw
            result = lro._load_mne_data_no_resample(extract_func)
        assert extract_func.call_count == 2
        mock_mne.concatenate_raws.assert_called_once()

    def test_preload_when_not_loaded(self):
        lro = _make_lro(item="file.edf")
        raw = _make_mock_mne_raw(preload=False)
        extract_func = Mock(return_value=raw)
        lro._load_mne_data_no_resample(extract_func)
        raw.load_data.assert_called_once()


# ===================================================================
# _get_or_create_intermediate_file
# ===================================================================


class TestGetOrCreateIntermediateFile:
    """Lines 998-1003, 1032-1047, 1080, 1190."""

    def test_cache_policy_always_missing(self, tmp_path):
        """Lines 998-1003: cache_policy='always' raises FileNotFoundError."""
        lro = _make_lro(item="dummy")
        fname = tmp_path / "nonexistent.edf"
        with pytest.raises(FileNotFoundError, match="Cache policy 'always'"):
            lro._get_or_create_intermediate_file(
                fname=fname,
                source_paths=["src.edf"],
                cache_policy="always",
                intermediate="edf",
                extract_func=Mock(),
                n_jobs=1,
            )

    def test_corrupted_metadata_json_auto(self, tmp_path):
        """Lines 1032-1047: corrupted metadata JSON fallback with auto policy."""
        lro = _make_lro(item="dummy.edf")
        fname = tmp_path / "cached.edf"
        meta_fname = fname.with_suffix(fname.suffix + ".meta.json")
        # Create files so cache validation passes
        fname.write_text("fake data")
        meta_fname.write_text("{{invalid json")

        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)

        with patch("neurodent.core.core.should_use_cache_unified", return_value=True):
            with patch("neurodent.core.core.RecordingMetadata.from_json", side_effect=json.JSONDecodeError("bad", "", 0)):
                with patch("neurodent.core.core.extract_mne_unit_info", return_value=("µV", 1.0)):
                    with patch.object(lro, "_load_mne_data_no_resample", return_value=raw):
                        with patch("neurodent.core.core.se") as mock_se:
                            mock_se.read_edf.return_value = _make_mock_recording()
                            with patch("neurodent.core.core.mne") as mock_mne:
                                mock_mne.export.export_raw.side_effect = _export_creates_file
                                rec, raw_obj, meta = lro._get_or_create_intermediate_file(
                                    fname=fname,
                                    source_paths=["src.edf"],
                                    cache_policy="auto",
                                    intermediate="edf",
                                    extract_func=extract_func,
                                    n_jobs=1,
                                )
        # Verify regeneration happened (extract_func called to rebuild metadata)
        assert extract_func.called
        assert meta is not None

    def test_item_list_extract_func(self, tmp_path):
        """Line 1080: extract_func(self.item[0]) when item is list."""
        lro = _make_lro(item=["f1.edf", "f2.edf"])
        fname = tmp_path / "inter.edf"
        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)

        with patch("neurodent.core.core.should_use_cache_unified", return_value=False):
            with patch("neurodent.core.core.extract_mne_unit_info", return_value=("µV", 1.0)):
                with patch.object(lro, "_load_mne_data_no_resample", return_value=raw):
                    with patch("neurodent.core.core.se") as mock_se:
                        mock_se.read_edf.return_value = _make_mock_recording()
                        with patch("neurodent.core.core.mne") as mock_mne:
                            mock_mne.export.export_raw.side_effect = _export_creates_file
                            rec, raw_obj, meta = lro._get_or_create_intermediate_file(
                                fname=fname,
                                source_paths=["f1.edf", "f2.edf"],
                                cache_policy="auto",
                                intermediate="edf",
                                extract_func=extract_func,
                                n_jobs=1,
                            )
        # First call to extract_func is with item[0]
        assert extract_func.call_args_list[0].args[0] == "f1.edf"

    def test_invalid_intermediate_type(self, tmp_path):
        """Line 1190: ValueError for invalid intermediate type."""
        lro = _make_lro(item="dummy.edf")
        fname = tmp_path / "inter.xyz"
        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)

        with patch("neurodent.core.core.should_use_cache_unified", return_value=False):
            with patch("neurodent.core.core.extract_mne_unit_info", return_value=("µV", 1.0)):
                with patch.object(lro, "_load_mne_data_no_resample", return_value=raw):
                    with pytest.raises(ValueError, match="Invalid intermediate"):
                        lro._get_or_create_intermediate_file(
                            fname=fname,
                            source_paths=["src.edf"],
                            cache_policy="auto",
                            intermediate="xyz",
                            extract_func=extract_func,
                            n_jobs=1,
                        )


# ===================================================================
# Intan channel name conversion & EDF export retry
# ===================================================================


class TestIntanAndEdfExport:
    """Lines 1116-1117, 1126-1161."""

    def test_intan_channel_conversion(self, tmp_path):
        """Lines 1116-1117: Intan channel names trigger conversion."""
        lro = _make_lro(item="dummy.edf")
        fname = tmp_path / "inter.edf"
        raw = _make_mock_mne_raw(ch_prefix="intan_ch")
        extract_func = Mock(return_value=raw)

        with patch("neurodent.core.core.should_use_cache_unified", return_value=False):
            with patch("neurodent.core.core.extract_mne_unit_info", return_value=("µV", 1.0)):
                with patch.object(lro, "_load_mne_data_no_resample", return_value=raw):
                    with patch("neurodent.core.core.convert_intan_chname_mne") as mock_conv:
                        with patch("neurodent.core.core.se") as mock_se:
                            mock_se.read_edf.return_value = _make_mock_recording()
                            with patch("neurodent.core.core.mne") as mock_mne:
                                mock_mne.export.export_raw.side_effect = _export_creates_file
                                lro._get_or_create_intermediate_file(
                                    fname=fname,
                                    source_paths=["src.edf"],
                                    cache_policy="auto",
                                    intermediate="edf",
                                    extract_func=extract_func,
                                    n_jobs=1,
                                )
                        mock_conv.assert_called_once_with(raw)

    def test_edf_export_retry_with_robust_range(self, tmp_path):
        """Lines 1126-1161: EDF export ValueError retry with robust range."""
        lro = _make_lro(item="dummy.edf")
        fname = tmp_path / "inter.edf"
        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)

        first_call = True

        def side_effect_export(path, *args, **kwargs):
            nonlocal first_call
            if first_call:
                first_call = False
                raise ValueError("exceeds maximum field length")
            # Successful retry: create the file so the atomic rename succeeds.
            Path(path).write_bytes(b"")

        with patch("neurodent.core.core.should_use_cache_unified", return_value=False):
            with patch("neurodent.core.core.extract_mne_unit_info", return_value=("µV", 1.0)):
                with patch.object(lro, "_load_mne_data_no_resample", return_value=raw):
                    with patch("neurodent.core.core.se") as mock_se:
                        mock_se.read_edf.return_value = _make_mock_recording()
                        with patch("neurodent.core.core.mne") as mock_mne:
                            mock_mne.export.export_raw = Mock(side_effect=side_effect_export)
                            lro._get_or_create_intermediate_file(
                                fname=fname,
                                source_paths=["src.edf"],
                                cache_policy="auto",
                                intermediate="edf",
                                extract_func=extract_func,
                                n_jobs=1,
                            )
                            assert mock_mne.export.export_raw.call_count == 2
                            # Verify second call used physical_range kwarg (robust retry)
                            second_call = mock_mne.export.export_raw.call_args_list[1]
                            assert "physical_range" in second_call.kwargs


# ===================================================================
# Self-healing read of a corrupt intermediate cache
# ===================================================================


def _write_valid_meta(meta_fname, channel_names=None):
    """Write a valid RecordingMetadata sidecar so the cached-read path is reached."""
    channel_names = channel_names or list(_DEFAULT_CH_NAMES)
    RecordingMetadata(
        metadata_path=None,
        n_channels=len(channel_names),
        f_s=constants.GLOBAL_SAMPLING_RATE,
        dt_end=None,
        channel_names=channel_names,
        V_units="µV",
        mult_to_uV=1.0,
    ).to_json(meta_fname)


class TestSelfHealingCorruptCache:
    """A corrupt cached intermediate is deleted and regenerated under 'auto'."""

    def test_corrupt_edf_cache_regenerates_auto(self, tmp_path):
        lro = _make_lro(item="dummy.edf")
        fname = tmp_path / "cached.edf"
        meta_fname = fname.with_suffix(fname.suffix + ".meta.json")
        fname.write_text("corrupt-not-edf")  # exists so cache validation passes
        _write_valid_meta(meta_fname)

        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)

        reads = {"n": 0}

        def read_edf_side_effect(path):
            reads["n"] += 1
            if reads["n"] == 1:
                raise RuntimeError("the file is not EDF(+) or BDF(+) compliant (Filesize)")
            return _make_mock_recording()

        with patch("neurodent.core.core.should_use_cache_unified", return_value=True):
            with patch("neurodent.core.core.extract_mne_unit_info", return_value=("µV", 1.0)):
                with patch.object(lro, "_load_mne_data_no_resample", return_value=raw):
                    with patch("neurodent.core.core.se") as mock_se:
                        mock_se.read_edf.side_effect = read_edf_side_effect
                        with patch("neurodent.core.core.mne") as mock_mne:
                            mock_mne.export.export_raw.side_effect = _export_creates_file
                            rec, raw_obj, meta = lro._get_or_create_intermediate_file(
                                fname=fname,
                                source_paths=["src.edf"],
                                cache_policy="auto",
                                intermediate="edf",
                                extract_func=extract_func,
                                n_jobs=1,
                            )
        # The corrupt cache was read once (failed), then regenerated and re-read.
        assert reads["n"] == 2
        assert extract_func.called
        assert rec is not None
        assert fname.exists()  # regenerated

    def test_corrupt_bin_cache_regenerates_auto(self, tmp_path):
        lro = _make_lro(item="dummy.bin")
        fname = tmp_path / "cached.bin"
        meta_fname = fname.with_suffix(fname.suffix + ".meta.json")
        fname.write_bytes(b"corrupt")
        _write_valid_meta(meta_fname)

        raw = _make_mock_mne_raw()
        extract_func = Mock(return_value=raw)

        reads = {"n": 0}

        def read_binary_side_effect(path, **kwargs):
            reads["n"] += 1
            if reads["n"] == 1:
                raise ValueError("truncated binary file")
            return _make_mock_recording()

        with patch("neurodent.core.core.should_use_cache_unified", return_value=True):
            with patch("neurodent.core.core.extract_mne_unit_info", return_value=("µV", 1.0)):
                with patch.object(lro, "_load_mne_data_no_resample", return_value=raw):
                    with patch("neurodent.core.core.se") as mock_se:
                        mock_se.read_binary.side_effect = read_binary_side_effect
                        rec, raw_obj, meta = lro._get_or_create_intermediate_file(
                            fname=fname,
                            source_paths=["src.bin"],
                            cache_policy="auto",
                            intermediate="bin",
                            extract_func=extract_func,
                            n_jobs=1,
                        )
        assert reads["n"] == 2
        assert extract_func.called
        assert rec is not None
        assert fname.exists()  # regenerated (real data.tofile write)

    def test_corrupt_cache_always_raises(self, tmp_path):
        """Under cache_policy='always' a corrupt cache is not self-healed; it raises."""
        lro = _make_lro(item="dummy.edf")
        fname = tmp_path / "cached.edf"
        meta_fname = fname.with_suffix(fname.suffix + ".meta.json")
        fname.write_text("corrupt-not-edf")
        _write_valid_meta(meta_fname)

        extract_func = Mock()

        with patch("neurodent.core.core.should_use_cache_unified", return_value=True):
            with patch("neurodent.core.core.se") as mock_se:
                mock_se.read_edf.side_effect = RuntimeError("not EDF(+) compliant")
                with pytest.raises(RuntimeError, match="not EDF"):
                    lro._get_or_create_intermediate_file(
                        fname=fname,
                        source_paths=["src.edf"],
                        cache_policy="always",
                        intermediate="edf",
                        extract_func=extract_func,
                        n_jobs=1,
                    )
        # No regeneration attempted under 'always'.
        assert not extract_func.called


# ===================================================================
# convert_file_with_mne_to_recording
# ===================================================================


class TestConvertFileWithMne:
    """Lines 1210, 1216-1218, 1246-1249, 1273-1287, 1291-1294."""

    @patch("neurodent.core.core.se", None)
    def test_import_error(self):
        lro = _make_lro(item="dummy")
        with pytest.raises(ImportError, match="SpikeInterface is required"):
            lro.convert_file_with_mne_to_recording(extract_func=Mock())

    @pytest.mark.skipif(si is None, reason="SpikeInterface not installed")
    def test_multi_file_timestamps(self, tmp_path):
        """Lines 1216-1218: multi-file list item."""
        rec = _make_mock_recording()
        raw = _make_mock_mne_raw()
        lro = _make_lro(
            item=["f1.edf", "f2.edf"],
            manual_datetimes=[datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 12, 5)],
        )

        with patch.object(
            lro,
            "_get_or_create_intermediate_file",
            return_value=(rec, raw, RecordingMetadata(
                None, n_channels=4, f_s=1000, dt_end=None,
                channel_names=_DEFAULT_CH_NAMES,
            )),
        ):
            with patch.object(lro, "_apply_resampling", return_value=rec):
                with patch.object(lro, "finalize_file_timestamps"):
                    lro.convert_file_with_mne_to_recording(
                        extract_func=Mock(),
                        intermediate_dir=str(tmp_path),
                    )
        assert lro._n_processed_files == 2

    def test_tmpdir_keyerror_fallback(self, tmp_path):
        """Lines 1246-1249: TMPDIR KeyError fallback."""
        rec = _make_mock_recording()
        raw = _make_mock_mne_raw()
        lro = _make_lro(item="file.edf")

        with patch("neurodent.core.core.se") as mock_se:
            with patch("neurodent.core.core.get_temp_directory", side_effect=KeyError("TMPDIR")):
                with patch("tempfile.gettempdir", return_value=str(tmp_path)):
                    with patch.object(
                        lro,
                        "_get_or_create_intermediate_file",
                        return_value=(rec, raw, RecordingMetadata(
                            None, n_channels=4, f_s=1000, dt_end=None,
                            channel_names=_DEFAULT_CH_NAMES,
                        )),
                    ):
                        with patch.object(lro, "_apply_resampling", return_value=rec):
                            with patch.object(lro, "finalize_file_timestamps"):
                                lro.convert_file_with_mne_to_recording(
                                    extract_func=Mock(),
                                )

    def test_cleanup_intermediate_files(self, tmp_path):
        """Lines 1273-1287: cleanup with force_regenerate policy."""
        rec = _make_mock_recording()
        raw = _make_mock_mne_raw()
        lro = _make_lro(item="file.edf")

        with patch("neurodent.core.core.se"):
            with patch("neurodent.core.core.get_temp_directory", return_value=tmp_path):
                with patch.object(
                    lro,
                    "_get_or_create_intermediate_file",
                    return_value=(rec, raw, RecordingMetadata(
                        None, n_channels=4, f_s=1000, dt_end=None,
                        channel_names=_DEFAULT_CH_NAMES,
                    )),
                ):
                    with patch.object(lro, "_apply_resampling", return_value=rec):
                        with patch.object(lro, "finalize_file_timestamps"):
                            lro.convert_file_with_mne_to_recording(
                                extract_func=Mock(),
                                cache_policy="force_regenerate",
                            )

    def test_multi_file_duration_averaging(self, tmp_path):
        """Lines 1291-1294: multi-file duration averaging."""
        rec = _make_mock_recording(duration=20.0)
        raw = _make_mock_mne_raw()
        lro = _make_lro(
            item=["f1.edf", "f2.edf"],
            manual_datetimes=datetime(2023, 1, 1, 12, 0),
        )

        with patch("neurodent.core.core.se"):
            with patch.object(
                lro,
                "_get_or_create_intermediate_file",
                return_value=(rec, raw, RecordingMetadata(
                    None, n_channels=4, f_s=1000, dt_end=None,
                    channel_names=_DEFAULT_CH_NAMES,
                )),
            ):
                with patch.object(lro, "_apply_resampling", return_value=rec):
                    lro.convert_file_with_mne_to_recording(
                        extract_func=Mock(),
                        intermediate_dir=str(tmp_path),
                    )
        assert len(lro.file_durations) == 2
        assert lro.file_durations[0] == 10.0
        assert lro.file_durations[1] == 10.0
        assert sum(lro.file_durations) == pytest.approx(20.0)  # total preserved


# ===================================================================
# cleanup_rec
# ===================================================================


class TestCleanupRec:
    """Lines 1304-1305: AttributeError when LongRecording already deleted."""

    def test_cleanup_already_deleted(self):
        lro = _make_lro(item=None)
        # Remove attribute to trigger AttributeError
        del lro.LongRecording
        lro.temppaths = []
        lro.cleanup_rec()  # Should not raise


# ===================================================================
# save_to_edf
# ===================================================================


class TestSaveToEdf:
    """Lines 1589-1590."""

    def test_save_to_edf_calls_convert_and_export(self, tmp_path):
        lro = _make_lro(item=None)
        lro.LongRecording = _make_mock_recording()
        lro.channel_names = ["ch0", "ch1", "ch2", "ch3"]

        out = tmp_path / "output.edf"
        with patch.object(lro, "convert_to_mne") as mock_conv:
            mock_raw = MagicMock()
            mock_conv.return_value = mock_raw
            with patch("neurodent.core.core.mne") as mock_mne:
                lro.save_to_edf(out)
                mock_conv.assert_called_once()
                mock_mne.export.export_raw.assert_called_once()


# ===================================================================
# compute_bad_channels
# ===================================================================


class TestComputeBadChannels:
    """Lines 1613, 1622-1624."""

    def test_existing_lof_scores(self):
        """Line 1613: uses existing LOF scores."""
        lro = _make_lro(item=None)
        lro.lof_scores = np.array([1.0, 2.0])
        lro.channel_names = ["a", "b"]
        lro.compute_bad_channels(lof_threshold=1.5)
        assert lro.bad_channel_names == ["b"]

    def test_exception_during_lof(self):
        """Lines 1622-1624: exception during LOF computation."""
        lro = _make_lro(item=None)
        lro.LongRecording = _make_mock_recording()
        lro.channel_names = ["a", "b"]
        with patch.object(
            lro, "_compute_lof_scores", side_effect=RuntimeError("fail")
        ):
            with pytest.raises(RuntimeError, match="fail"):
                lro.compute_bad_channels(force_recompute=True)


# ===================================================================
# apply_lof_threshold  /  get_lof_scores
# ===================================================================


class TestLofScores:
    """Lines 1706, 1724-1729."""

    def test_apply_lof_threshold_no_scores(self):
        lro = _make_lro(item=None)
        with pytest.raises(ValueError, match="LOF scores not available"):
            lro.apply_lof_threshold(1.5)

    def test_get_lof_scores_no_scores(self):
        lro = _make_lro(item=None)
        with pytest.raises(ValueError, match="LOF scores not available"):
            lro.get_lof_scores()


# ===================================================================
# _validate_timestamps_for_mode
# ===================================================================


class TestValidateTimestamps:
    """Line 1760: length mismatch error."""

    def test_length_mismatch(self):
        lro = _make_lro(
            item=None,
            manual_datetimes=[datetime(2023, 1, 1), datetime(2023, 1, 2)],
        )
        with pytest.raises(ValueError, match="manual_datetimes length"):
            lro._validate_timestamps_for_mode("si", expected_n_files=3)


# ===================================================================
# finalize_file_timestamps
# ===================================================================


class TestFinalizeFileTimestamps:
    """Lines 1864, 1878-1882."""

    def test_no_file_durations_early_return(self):
        """Line 1864: no file_durations → early return."""
        lro = _make_lro(item=None)
        lro.file_durations = []
        lro.finalize_file_timestamps()  # Should not raise

    def test_all_file_end_datetimes_none(self):
        """Lines 1878-1882: all file_end_datetimes are None."""
        lro = _make_lro(item=None)
        lro.file_durations = [10.0]
        lro.file_end_datetimes = [None]
        lro.manual_datetimes = None
        with pytest.raises(ValueError, match="No dates found"):
            lro.finalize_file_timestamps()


# ===================================================================
# get_date_string
# ===================================================================


class TestGetDateString:
    """Lines 1900, 1908."""

    def test_no_timestamps(self):
        lro = _make_lro(item=None)
        lro.file_end_datetimes = []
        with pytest.raises(ValueError, match="No file timestamps"):
            lro.get_date_string()

    def test_all_none_timestamps(self):
        lro = _make_lro(item=None)
        lro.file_end_datetimes = [None, None]
        lro.file_durations = [10.0, 10.0]
        with pytest.raises(ValueError, match="All file timestamps are None"):
            lro.get_date_string()


# ===================================================================
# __str__
# ===================================================================


class TestStr:
    """Lines 1919, 1948."""

    def test_no_recording_loaded(self):
        """Line 1919: no LongRecording loaded."""
        lro = _make_lro(item=None)
        assert "No recording loaded" in str(lro)

    def test_metadata_precision_units(self):
        """Line 1948: metadata with precision and units."""
        lro = _make_lro(item=None)
        lro.LongRecording = _make_mock_recording()
        lro.channel_names = ["ch0", "ch1", "ch2", "ch3"]
        lro.file_durations = [10.0]
        lro.file_end_datetimes = [datetime(2023, 1, 1)]
        meta = RecordingMetadata(
            None,
            n_channels=4,
            f_s=1000.0,
            dt_end=None,
            channel_names=["ch0", "ch1", "ch2", "ch3"],
            V_units="µV",
        )
        meta.precision = "float32"
        lro.meta = meta
        result = str(lro)
        assert "float32 precision" in result
        assert "µV units" in result


# ===================================================================
# _apply_resampling
# ===================================================================


class TestApplyResampling:
    """Lines 1982, 2001-2002, 2020-2021, 2046."""

    def test_recording_none(self):
        """Line 1982: recording is None."""
        lro = _make_lro(item=None)
        result = lro._apply_resampling(None)
        assert result is None

    def test_dtype_type_error_integer_check(self):
        """Lines 2001-2002: TypeError on dtype check for is_integer.
        Using a string that causes np.dtype() to raise TypeError."""
        lro = _make_lro(item=None)
        rec = Mock()
        # "bogus" causes np.dtype("bogus") -> TypeError, and isinstance("bogus", str) is True
        rec.get_dtype.return_value = "bogus"
        rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        rec.has_scaleable_traces.return_value = False
        with patch("neurodent.core.core.spre") as mock_spre:
            mock_spre.astype.return_value = rec
            result = lro._apply_resampling(rec)
        assert result is rec

    def test_dtype_type_error_unsigned_check(self):
        """Lines 2020-2021: TypeError on dtype check for is_unsigned.
        Second dtype call returns invalid string."""
        lro = _make_lro(item=None)
        rec = Mock()
        call_count = [0]
        def fake_get_dtype():
            call_count[0] += 1
            if call_count[0] <= 2:
                return np.dtype("float32")  # first two calls OK
            return "bogus"  # third+ calls → TypeError in unsigned check
        rec.get_dtype = fake_get_dtype
        rec.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        rec.has_scaleable_traces.return_value = False
        with patch("neurodent.core.core.spre") as mock_spre:
            mock_spre.astype.return_value = rec
            result = lro._apply_resampling(rec)
        assert result is rec

    def test_info_fallback_for_sfreq(self):
        """Line 2046: recording.info fallback when get_sampling_frequency missing."""
        lro = _make_lro(item=None)
        rec = Mock()
        rec.get_dtype.return_value = np.dtype("float32")
        rec.has_scaleable_traces.return_value = False
        rec.get_sampling_frequency.return_value = None
        rec.info = {"sfreq": constants.GLOBAL_SAMPLING_RATE}
        result = lro._apply_resampling(rec)
        assert result is rec


# ===================================================================
# merge
# ===================================================================


class TestMerge:
    """Line 2087."""

    @patch("neurodent.core.core.si", None)
    def test_merge_import_error(self):
        lro = _make_lro(item=None)
        other = _make_lro(item=None)
        with pytest.raises(ImportError, match="SpikeInterface is required"):
            lro.merge(other)


# ===================================================================
# _validate_merge_compatibility
# ===================================================================


class TestValidateMergeCompatibility:
    """Lines 2155, 2162, 2170, 2172."""

    def test_sampling_rate_mismatch(self):
        lro1 = _make_lro(item=None)
        lro2 = _make_lro(item=None)
        lro1.channel_names = ["a", "b"]
        lro2.channel_names = ["a", "b"]
        lro1.meta = Mock(f_s=1000, n_channels=2)
        lro2.meta = Mock(f_s=500, n_channels=2)
        lro1.LongRecording = _make_mock_recording(n_channels=2)
        lro2.LongRecording = _make_mock_recording(n_channels=2)
        with pytest.raises(ValueError, match="Sampling rate mismatch"):
            lro1._validate_merge_compatibility(lro2)

    def test_channel_count_mismatch(self):
        lro1 = _make_lro(item=None)
        lro2 = _make_lro(item=None)
        lro1.channel_names = ["a", "b"]
        lro2.channel_names = ["a", "b"]
        lro1.meta = Mock(f_s=1000, n_channels=2)
        lro2.meta = Mock(f_s=1000, n_channels=3)
        lro1.LongRecording = _make_mock_recording(n_channels=2)
        lro2.LongRecording = _make_mock_recording(n_channels=2)
        with pytest.raises(ValueError, match="Channel count mismatch"):
            lro1._validate_merge_compatibility(lro2)

    def test_no_valid_long_recording_self(self):
        lro1 = _make_lro(item=None)
        lro2 = _make_lro(item=None)
        lro1.channel_names = ["a"]
        lro2.channel_names = ["a"]
        lro1.meta = Mock(f_s=1000, n_channels=1)
        lro2.meta = Mock(f_s=1000, n_channels=1)
        lro1.LongRecording = None
        lro2.LongRecording = _make_mock_recording(n_channels=1)
        with pytest.raises(ValueError, match="This LRO does not have"):
            lro1._validate_merge_compatibility(lro2)

    def test_no_valid_long_recording_other(self):
        lro1 = _make_lro(item=None)
        lro2 = _make_lro(item=None)
        lro1.channel_names = ["a"]
        lro2.channel_names = ["a"]
        lro1.meta = Mock(f_s=1000, n_channels=1)
        lro2.meta = Mock(f_s=1000, n_channels=1)
        lro1.LongRecording = _make_mock_recording(n_channels=1)
        lro2.LongRecording = None
        with pytest.raises(ValueError, match="Other LRO does not have"):
            lro1._validate_merge_compatibility(lro2)
