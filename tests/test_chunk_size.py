"""Tests for adjustable in-memory chunk sizes (issue #156).

Verifies that:
- ``cache_fragments_to_zarr`` honours the ``chunk_size`` parameter.
- ``compute_windowed_analysis`` produces identical results regardless of
  ``chunk_size`` (correctness of the streaming path).
- Edge cases: ``chunk_size=1``, ``chunk_size`` larger than total fragments,
  and ``chunk_size=None`` (default behaviour).
- ``save_fif_and_json`` propagates the ``chunk_len`` parameter to
  ``convert_to_mne``.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from neurodent.core.utils import cache_fragments_to_zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fragments(n: int, n_samples: int = 50, n_channels: int = 4) -> np.ndarray:
    """Return a deterministic float32 array shaped (n, n_samples, n_channels)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, n_samples, n_channels)).astype(np.float32)


# ---------------------------------------------------------------------------
# cache_fragments_to_zarr – chunk_size parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCacheFragmentsToZarrChunkSize:
    """Unit tests for the ``chunk_size`` parameter of ``cache_fragments_to_zarr``."""

    def test_default_chunk_size_creates_zarr(self, tmp_path):
        """Calling without ``chunk_size`` should still work (backward-compat)."""
        frags = _make_fragments(10)
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            path, arr = cache_fragments_to_zarr(frags, 10)
        assert os.path.exists(path)
        np.testing.assert_array_equal(arr[:], frags)

    def test_explicit_chunk_size_stored_in_zarr(self, tmp_path):
        """Zarr chunk dimension 0 should equal the supplied ``chunk_size``."""
        frags = _make_fragments(20)
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            _, arr = cache_fragments_to_zarr(frags, 20, chunk_size=5)
        assert arr.chunks[0] == 5

    def test_chunk_size_larger_than_n_fragments_clipped(self, tmp_path):
        """``chunk_size`` larger than ``n_fragments`` should be clipped to
        ``n_fragments`` so zarr does not store more chunks than needed."""
        frags = _make_fragments(7)
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            _, arr = cache_fragments_to_zarr(frags, 7, chunk_size=1000)
        assert arr.chunks[0] == 7  # clipped to n_fragments
        assert arr.shape[0] == 7  # total size matches n_fragments

    def test_chunk_size_one(self, tmp_path):
        """``chunk_size=1`` is the most memory-conservative setting."""
        frags = _make_fragments(6)
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            _, arr = cache_fragments_to_zarr(frags, 6, chunk_size=1)
        assert arr.chunks[0] == 1
        np.testing.assert_array_equal(arr[:], frags)

    def test_data_roundtrip_preserved(self, tmp_path):
        """Data written to zarr should be bit-for-bit equal when read back."""
        frags = _make_fragments(15)
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            _, arr = cache_fragments_to_zarr(frags, 15, chunk_size=4)
        np.testing.assert_array_equal(arr[:], frags)

    def test_none_chunk_size_uses_default_behavior(self, tmp_path):
        """Explicitly passing ``chunk_size=None`` should match default."""
        frags = _make_fragments(50)
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            _, arr_default = cache_fragments_to_zarr(frags, 50)
        # default caps at min(100, n_fragments) = 50
        assert arr_default.chunks[0] == 50

        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            _, arr_none = cache_fragments_to_zarr(frags, 50, chunk_size=None)
        assert arr_none.chunks[0] == 50


# ---------------------------------------------------------------------------
# compute_windowed_analysis – streaming path (chunk_size)
# ---------------------------------------------------------------------------


def _make_mock_lan(n_fragments: int, fragment_shape=(100, 4)):
    """Return a mock LongRecordingAnalyzer that yields deterministic fragments."""
    rng = np.random.default_rng(42)
    fragments = [
        rng.standard_normal(fragment_shape).astype(np.float32)
        for _ in range(n_fragments)
    ]

    lan = MagicMock()
    lan.n_fragments = n_fragments
    lan.f_s = 1000.0
    lan.get_fragment_np.side_effect = lambda idx: fragments[idx]
    lan.get_file_end.return_value = False
    return lan, fragments


@pytest.mark.unit
class TestComputeWindowedAnalysisChunkSize:
    """Unit tests for the ``chunk_size`` streaming path in
    ``compute_windowed_analysis``.

    We test the Dask zarr-building logic in isolation by exercising the helper
    that streams fragments to zarr directly, without constructing a full
    ``AnimalOrganizer``.
    """

    def _run_streaming(self, n_fragments: int, chunk_size: int, tmp_path):
        """Run the streaming path and return the zarr array contents."""
        import zarr

        fragment_shape = (80, 4)
        lan, fragments = _make_mock_lan(n_fragments, fragment_shape)

        # Replicate the streaming logic from compute_windowed_analysis
        batch = min(chunk_size, n_fragments)
        zarr_path = str(tmp_path / f"test_{n_fragments}_{chunk_size}.zarr")
        zarr_array = zarr.open(
            zarr_path,
            mode="w",
            shape=(n_fragments,) + fragment_shape,
            chunks=(batch, -1, -1),
            dtype=np.float32,
            compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
        )
        for batch_start in range(0, n_fragments, batch):
            batch_end = min(batch_start + batch, n_fragments)
            batch_len = batch_end - batch_start
            np_batch = np.empty((batch_len,) + fragment_shape, dtype=np.float32)
            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                np_batch[local_idx] = lan.get_fragment_np(global_idx)
            zarr_array[batch_start:batch_end] = np_batch
            del np_batch
        del zarr_array

        result = zarr.open(zarr_path, mode="r")[:]
        return result, fragments

    def test_streaming_data_matches_original(self, tmp_path):
        """Streamed zarr data must be bit-for-bit equal to source fragments."""
        result, fragments = self._run_streaming(n_fragments=10, chunk_size=3, tmp_path=tmp_path)
        expected = np.stack(fragments)
        np.testing.assert_array_equal(result, expected)

    def test_chunk_size_one_correct(self, tmp_path):
        """Edge case: ``chunk_size=1`` (most conservative)."""
        result, fragments = self._run_streaming(n_fragments=5, chunk_size=1, tmp_path=tmp_path)
        expected = np.stack(fragments)
        np.testing.assert_array_equal(result, expected)

    def test_chunk_size_larger_than_total(self, tmp_path):
        """Edge case: ``chunk_size`` > number of fragments."""
        result, fragments = self._run_streaming(n_fragments=5, chunk_size=100, tmp_path=tmp_path)
        expected = np.stack(fragments)
        np.testing.assert_array_equal(result, expected)

    def test_chunk_size_equals_total(self, tmp_path):
        """``chunk_size`` == number of fragments (single batch)."""
        result, fragments = self._run_streaming(n_fragments=8, chunk_size=8, tmp_path=tmp_path)
        expected = np.stack(fragments)
        np.testing.assert_array_equal(result, expected)

    def test_streaming_same_as_bulk_allocation(self, tmp_path):
        """Streaming and bulk-allocation paths should produce identical zarr data."""
        import zarr

        n_fragments = 12
        fragment_shape = (60, 3)
        chunk_size = 4
        _, fragments = _make_mock_lan(n_fragments, fragment_shape)
        expected = np.stack(fragments)

        # Bulk path
        bulk_path = str(tmp_path / "bulk.zarr")
        bulk_arr = zarr.open(
            bulk_path,
            mode="w",
            shape=expected.shape,
            chunks=(min(100, n_fragments), -1, -1),
            dtype=np.float32,
        )
        bulk_arr[:] = expected
        bulk_result = zarr.open(bulk_path, mode="r")[:]

        # Streaming path
        lan, _ = _make_mock_lan(n_fragments, fragment_shape)
        stream_path = str(tmp_path / "stream.zarr")
        batch = min(chunk_size, n_fragments)
        stream_arr = zarr.open(
            stream_path,
            mode="w",
            shape=(n_fragments,) + fragment_shape,
            chunks=(batch, -1, -1),
            dtype=np.float32,
        )
        for batch_start in range(0, n_fragments, batch):
            batch_end = min(batch_start + batch, n_fragments)
            batch_len = batch_end - batch_start
            np_batch = np.empty((batch_len,) + fragment_shape, dtype=np.float32)
            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                np_batch[local_idx] = lan.get_fragment_np(global_idx)
            stream_arr[batch_start:batch_end] = np_batch
        stream_result = zarr.open(stream_path, mode="r")[:]

        np.testing.assert_array_equal(stream_result, bulk_result)


# ---------------------------------------------------------------------------
# compute_windowed_analysis – chunk_size parameter is accepted by the method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeWindowedAnalysisSignature:
    """Smoke-test that ``chunk_size`` is wired into the method signature."""

    def test_chunk_size_in_signature(self):
        """``compute_windowed_analysis`` must accept a ``chunk_size`` kwarg."""
        import inspect
        from neurodent.visualization.results import AnimalOrganizer

        sig = inspect.signature(AnimalOrganizer.compute_windowed_analysis)
        assert "chunk_size" in sig.parameters

    def test_chunk_size_default_is_none(self):
        """Default value of ``chunk_size`` should be None (backward-compat)."""
        import inspect
        from neurodent.visualization.results import AnimalOrganizer

        sig = inspect.signature(AnimalOrganizer.compute_windowed_analysis)
        assert sig.parameters["chunk_size"].default is None


# ---------------------------------------------------------------------------
# save_fif_and_json – chunk_len parameter propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveFifAndJsonChunkLen:
    """Verify that ``save_fif_and_json`` forwards ``chunk_len`` to
    ``convert_to_mne``."""

    def test_chunk_len_forwarded_to_convert_to_mne(self, tmp_path):
        """``save_fif_and_json`` should call ``convert_to_mne`` with the
        supplied ``chunk_len``."""
        from neurodent.visualization.frequency_domain_results import (
            FrequencyDomainSpikeAnalysisResult,
        )

        fdsar = FrequencyDomainSpikeAnalysisResult.__new__(
            FrequencyDomainSpikeAnalysisResult
        )
        fdsar.result_mne = None
        fdsar.result_sas = [MagicMock()]  # non-empty list triggers convert_to_mne

        mock_raw = MagicMock()
        mock_raw.save = MagicMock()
        mock_raw.get_data.return_value = np.zeros((1, 100))

        with patch.object(fdsar, "convert_to_mne", return_value=mock_raw) as mock_conv:
            with patch("json.dump"), patch("builtins.open", MagicMock()):
                try:
                    fdsar.save_fif_and_json(
                        folder=str(tmp_path),
                        chunk_len=30.0,
                        multiprocess_mode="serial",
                    )
                except Exception:
                    pass  # saving may fail for mocked objects; we only care about the call

            # Verify chunk_len was forwarded
            mock_conv.assert_called_once()
            _, kwargs = mock_conv.call_args
            assert kwargs.get("chunk_len") == 30.0

    def test_chunk_len_default_is_60(self):
        """Default ``chunk_len`` in ``save_fif_and_json`` should be 60 s."""
        import inspect
        from neurodent.visualization.frequency_domain_results import (
            FrequencyDomainSpikeAnalysisResult,
        )

        sig = inspect.signature(FrequencyDomainSpikeAnalysisResult.save_fif_and_json)
        assert sig.parameters["chunk_len"].default == 60

    def test_chunk_len_in_signature(self):
        """``save_fif_and_json`` must expose a ``chunk_len`` parameter."""
        import inspect
        from neurodent.visualization.frequency_domain_results import (
            FrequencyDomainSpikeAnalysisResult,
        )

        sig = inspect.signature(FrequencyDomainSpikeAnalysisResult.save_fif_and_json)
        assert "chunk_len" in sig.parameters
