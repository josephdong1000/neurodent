"""Tests for adjustable in-memory chunk sizes (issue #156).

Verifies that:
- ``cache_fragments_to_zarr`` honors the ``chunk_size`` parameter.
- ``stream_fragments_to_zarr`` streams fragments correctly with bounded peak RAM.
- ``compute_windowed_analysis`` accepts ``chunk_duration_s`` and delegates to
  ``stream_fragments_to_zarr``.
- Edge cases: ``chunk_size=1``, ``chunk_size`` larger than total fragments,
  and ``chunk_size=None`` (default behavior).
- ``save_fif_and_json`` propagates the ``chunk_duration_s`` parameter to
  ``convert_to_mne``.
"""

import os
import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurodent.core.utils import cache_fragments_to_zarr, stream_fragments_to_zarr, stream_recording_to_zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fragments(n: int, n_samples: int = 50, n_channels: int = 4) -> np.ndarray:
    """Return a deterministic float32 array shaped (n, n_samples, n_channels)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, n_samples, n_channels)).astype(np.float32)


def _make_get_fragment_fn(n_fragments: int, fragment_shape=(100, 4)):
    """Return (get_fn, fragments_list) where get_fn(idx) returns fragment."""
    rng = np.random.default_rng(42)
    fragments = [
        rng.standard_normal(fragment_shape).astype(np.float32)
        for _ in range(n_fragments)
    ]
    return lambda idx: fragments[idx], fragments


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

    def test_invalid_chunk_size_zero_raises(self, tmp_path):
        """``chunk_size=0`` should raise ``ValueError`` before touching zarr."""
        frags = _make_fragments(5)
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
                cache_fragments_to_zarr(frags, 5, chunk_size=0)

    def test_invalid_chunk_size_negative_raises(self, tmp_path):
        """Negative ``chunk_size`` should raise ``ValueError`` before touching zarr."""
        frags = _make_fragments(5)
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
                cache_fragments_to_zarr(frags, 5, chunk_size=-10)


# ---------------------------------------------------------------------------
# stream_fragments_to_zarr – correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamFragmentsToZarr:
    """Tests for the standalone ``stream_fragments_to_zarr`` helper."""

    def test_basic_correctness(self, tmp_path):
        """Data streamed to zarr must equal what ``get_fragment_fn`` returns."""
        import zarr

        get_fn, fragments = _make_get_fragment_fn(10, (80, 4))
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            path = stream_fragments_to_zarr(get_fn, 10, (80, 4), np.float32, chunk_size=3)

        result = zarr.open(path, mode="r")[:]
        expected = np.stack(fragments)
        np.testing.assert_array_equal(result, expected)

    def test_chunk_size_one(self, tmp_path):
        """Edge case: ``chunk_size=1`` streams one fragment at a time."""
        import zarr

        get_fn, fragments = _make_get_fragment_fn(5, (60, 3))
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            path = stream_fragments_to_zarr(get_fn, 5, (60, 3), np.float32, chunk_size=1)

        result = zarr.open(path, mode="r")[:]
        np.testing.assert_array_equal(result, np.stack(fragments))

    def test_chunk_size_larger_than_n_fragments(self, tmp_path):
        """When ``chunk_size`` > ``n_fragments`` a single batch is used."""
        import zarr

        get_fn, fragments = _make_get_fragment_fn(5, (40, 2))
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            path = stream_fragments_to_zarr(get_fn, 5, (40, 2), np.float32, chunk_size=1000)

        result = zarr.open(path, mode="r")[:]
        np.testing.assert_array_equal(result, np.stack(fragments))

    def test_chunk_size_equals_n_fragments(self, tmp_path):
        """``chunk_size == n_fragments`` is a single-batch degenerate case."""
        import zarr

        get_fn, fragments = _make_get_fragment_fn(8, (50, 4))
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            path = stream_fragments_to_zarr(get_fn, 8, (50, 4), np.float32, chunk_size=8)

        result = zarr.open(path, mode="r")[:]
        np.testing.assert_array_equal(result, np.stack(fragments))

    def test_invalid_chunk_size_raises(self, tmp_path):
        """``chunk_size=0`` or negative should raise ``ValueError``."""
        get_fn, _ = _make_get_fragment_fn(4)
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
                stream_fragments_to_zarr(get_fn, 4, (50, 4), np.float32, chunk_size=0)

    def test_results_match_bulk_allocation(self, tmp_path):
        """Streamed output must be identical to bulk ``np.empty`` allocation."""
        import zarr

        n_fragments = 12
        fragment_shape = (60, 3)
        get_fn, fragments = _make_get_fragment_fn(n_fragments, fragment_shape)
        expected = np.stack(fragments)

        # Bulk path via cache_fragments_to_zarr
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            _, bulk_arr = cache_fragments_to_zarr(expected, n_fragments)
        bulk_result = bulk_arr[:]

        # Streaming path
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            path = stream_fragments_to_zarr(
                get_fn, n_fragments, fragment_shape, np.float32, chunk_size=4
            )
        stream_result = zarr.open(path, mode="r")[:]

        np.testing.assert_array_equal(stream_result, bulk_result)

    def test_get_fragment_fn_called_once_per_fragment(self, tmp_path):
        """Each fragment index should be fetched exactly once."""
        call_counts = {}

        def tracking_fn(idx):
            call_counts[idx] = call_counts.get(idx, 0) + 1
            return np.zeros((40, 2), dtype=np.float32)

        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            stream_fragments_to_zarr(tracking_fn, 6, (40, 2), np.float32, chunk_size=2)

        assert len(call_counts) == 6
        assert all(c == 1 for c in call_counts.values())


# ---------------------------------------------------------------------------
# stream_recording_to_zarr – equivalence with per-fragment path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamRecordingToZarrEquivalence:
    """Verify that ``stream_recording_to_zarr`` produces the same output as
    the old per-fragment ``stream_fragments_to_zarr`` path."""

    def test_bulk_path_matches_per_fragment_path(self, tmp_path):
        """stream_recording_to_zarr must produce identical fragments to
        stream_fragments_to_zarr when given the same underlying data."""
        import spikeinterface.core as si_core
        import zarr

        rng = np.random.default_rng(42)
        n_channels = 4
        fs = 1000.0
        n_frag = 10
        n_samples_per_frag = int(5 * fs)  # 5s windows
        total_samples = n_frag * n_samples_per_frag
        chunk = 3

        # Each channel has a different sine frequency so transposition
        # would produce visibly different data.
        t = np.arange(total_samples) / fs
        data = np.column_stack(
            [
                np.sin(2 * np.pi * (5 + ch * 3) * t)
                + 0.1 * rng.standard_normal(total_samples)
                for ch in range(n_channels)
            ]
        ).astype(np.float32)

        rec = si_core.NumpyRecording(
            traces_list=[data], sampling_frequency=fs,
        )

        # Old path: per-fragment getter → stream_fragments_to_zarr
        def get_fragment(idx):
            start = idx * n_samples_per_frag
            end = start + n_samples_per_frag
            return rec.get_traces(
                start_frame=start, end_frame=end, return_scaled=True,
            )

        with patch.dict(os.environ, {"TMPDIR": str(tmp_path / "old")}):
            os.makedirs(tmp_path / "old", exist_ok=True)
            old_path = stream_fragments_to_zarr(
                get_fragment,
                n_frag,
                (n_samples_per_frag, n_channels),
                np.float32,
                chunk,
            )

        # New path: bulk recording → stream_recording_to_zarr
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path / "new")}):
            os.makedirs(tmp_path / "new", exist_ok=True)
            new_path = stream_recording_to_zarr(
                rec, n_frag, n_samples_per_frag, chunk,
            )

        old_arr = zarr.open(old_path, mode="r")[:]
        new_arr = zarr.open(new_path, mode="r")[:]

        assert old_arr.shape == new_arr.shape == (
            n_frag, n_samples_per_frag, n_channels,
        )
        np.testing.assert_array_equal(old_arr, new_arr)

    def test_channels_are_distinguishable(self, tmp_path):
        """Each channel in the zarr output must have distinct content,
        catching transposition bugs where all channels become identical."""
        import spikeinterface.core as si_core
        import zarr

        n_channels = 4
        fs = 1000.0
        n_frag = 5
        n_samples_per_frag = int(5 * fs)
        total_samples = n_frag * n_samples_per_frag

        # Channels with very different means so transposition is obvious
        t = np.arange(total_samples) / fs
        data = np.column_stack(
            [
                ch * 100.0 + np.sin(2 * np.pi * (2 + ch) * t)
                for ch in range(n_channels)
            ]
        ).astype(np.float32)

        rec = si_core.NumpyRecording(
            traces_list=[data], sampling_frequency=fs,
        )

        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            path = stream_recording_to_zarr(rec, n_frag, n_samples_per_frag, 2)

        arr = zarr.open(path, mode="r")[:]
        # Check that channel means are distinct (0, 100, 200, 300)
        means = [arr[:, :, ch].mean() for ch in range(n_channels)]
        for i in range(n_channels):
            assert abs(means[i] - i * 100.0) < 1.0, (
                f"Channel {i} mean={means[i]:.1f}, expected ~{i * 100.0}"
            )


# ---------------------------------------------------------------------------
# stream_fragments_to_zarr – peak memory scales with chunk_size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamFragmentsPeakMemory:
    """Verify that peak RAM usage scales with ``chunk_size``, not ``n_fragments``.

    Uses ``tracemalloc`` to measure allocation inside the batch loop.  With a
    large fragment array and small ``chunk_size`` the peak allocation must be
    significantly less than with a large ``chunk_size``.
    """

    @staticmethod
    def _measure_peak_bytes(get_fn, n_fragments, fragment_shape, dtype, chunk_size, tmp_path):
        """Return peak bytes allocated during ``stream_fragments_to_zarr``."""
        tracemalloc.start()
        with patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            stream_fragments_to_zarr(get_fn, n_fragments, fragment_shape, dtype, chunk_size)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    def test_small_chunk_size_uses_less_peak_ram_than_large(self, tmp_path):
        """Peak RAM with ``chunk_size=1`` must be less than with ``chunk_size=n``."""
        n_fragments = 50
        fragment_shape = (500, 8)  # ~16 KB per fragment * 50 = ~800 KB total
        dtype = np.float32

        get_fn_small, _ = _make_get_fragment_fn(n_fragments, fragment_shape)
        get_fn_large, _ = _make_get_fragment_fn(n_fragments, fragment_shape)

        small_dir = tmp_path / "small"
        small_dir.mkdir()
        peak_small = self._measure_peak_bytes(
            get_fn_small, n_fragments, fragment_shape, dtype, chunk_size=1,
            tmp_path=small_dir,
        )

        large_dir = tmp_path / "large"
        large_dir.mkdir()
        peak_large = self._measure_peak_bytes(
            get_fn_large, n_fragments, fragment_shape, dtype, chunk_size=n_fragments,
            tmp_path=large_dir,
        )

        # Peak with chunk_size=1 must be strictly less than with chunk_size=n_fragments.
        # We allow a generous 5x margin to account for zarr/numpy overhead.
        assert peak_small < peak_large, (
            f"Expected peak_small ({peak_small} B) < peak_large ({peak_large} B)"
        )

    def test_peak_memory_proportional_to_chunk_size(self, tmp_path):
        """Peak RAM should grow roughly in proportion to ``chunk_size``."""
        n_fragments = 60
        fragment_shape = (400, 8)
        dtype = np.float32
        bytes_per_fragment = int(np.prod(fragment_shape)) * np.dtype(dtype).itemsize

        get_fn1, _ = _make_get_fragment_fn(n_fragments, fragment_shape)
        get_fn10, _ = _make_get_fragment_fn(n_fragments, fragment_shape)

        cs1_dir = tmp_path / "cs1"
        cs1_dir.mkdir()
        peak1 = self._measure_peak_bytes(
            get_fn1, n_fragments, fragment_shape, dtype, chunk_size=1,
            tmp_path=cs1_dir,
        )

        cs10_dir = tmp_path / "cs10"
        cs10_dir.mkdir()
        peak10 = self._measure_peak_bytes(
            get_fn10, n_fragments, fragment_shape, dtype, chunk_size=10,
            tmp_path=cs10_dir,
        )

        # With chunk_size=10 we allocate 10 fragments at a time; chunk_size=1 allocates 1.
        # The raw data ratio is 10x; allow for overhead by requiring at least 3x difference.
        assert peak10 > peak1 * 3, (
            f"Expected peak10 ({peak10} B) > 3 * peak1 ({peak1} B). "
            f"Each fragment is ~{bytes_per_fragment} bytes."
        )


# ---------------------------------------------------------------------------
# compute_windowed_analysis – chunk_duration_s parameter is accepted by the method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeWindowedAnalysisSignature:
    """Smoke-test that ``chunk_duration_s`` is wired into the method signature."""

    def test_chunk_duration_s_in_signature(self):
        """``compute_windowed_analysis`` must accept a ``chunk_duration_s`` kwarg."""
        import inspect
        from neurodent.visualization.results import AnimalOrganizer

        sig = inspect.signature(AnimalOrganizer.compute_windowed_analysis)
        assert "chunk_duration_s" in sig.parameters

    def test_chunk_duration_s_default_is_3600(self):
        """Default value of ``chunk_duration_s`` should be 3600."""
        import inspect
        from neurodent.visualization.results import AnimalOrganizer

        sig = inspect.signature(AnimalOrganizer.compute_windowed_analysis)
        assert sig.parameters["chunk_duration_s"].default == 3600

    def test_stream_fragments_to_zarr_called_when_chunk_duration_s_set(self, tmp_path):
        """Calling ``compute_windowed_analysis(multiprocess_mode="dask",
        chunk_duration_s=...)`` must invoke ``stream_recording_to_zarr``
        inside the dask branch of AO."""
        import pandas as pd
        from neurodent.visualization.results import AnimalOrganizer

        # -- build a minimal AO instance ------------------------------------
        ao = AnimalOrganizer.__new__(AnimalOrganizer)
        ao._validate_sampling_rates = MagicMock()
        ao.long_analyzers = []
        ao.long_recordings = []
        ao.animaldays = []
        ao.animal_id = "test"
        ao.genotype = "WT"
        ao.sex = "M"
        ao.channel_names = ["LAud", "RAud"]
        ao.bad_channels_dict = {}

        # -- mock LAN returned by core.LongRecordingAnalyzer ----------------
        mock_lan = MagicMock()
        mock_lan.n_fragments = 5
        mock_lan.f_s = 1000
        mock_lan.apply_notch_filter = False

        # -- mock lrec (LongRecordingOrganizer) with a SI recording ----------
        mock_si_rec = MagicMock()
        mock_si_rec.get_num_channels.return_value = 2
        mock_lrec = MagicMock()
        mock_lrec.display_name = "rec0"
        mock_lrec.LongRecording = mock_si_rec

        ao._iter_valid_recordings = MagicMock(
            return_value=iter([(0, mock_lrec)])
        )

        with (
            patch(
                "neurodent.visualization.results.core.LongRecordingAnalyzer",
                return_value=mock_lan,
            ),
            patch(
                "neurodent.visualization.results.core.utils.stream_recording_to_zarr",
                return_value=str(tmp_path / "fake.zarr"),
            ) as mock_stream,
            patch("neurodent.visualization.results.da.from_zarr"),
            patch(
                "neurodent.visualization.results.dask.compute",
                # n_fragments_war = max(n_fragments - 1, 1) = 4
                return_value=[{"rms": 0.0}] * (mock_lan.n_fragments - 1),
            ),
            patch(
                "neurodent.visualization.pipeline.delayed",
                side_effect=lambda f: lambda *a, **kw: {"rms": 0.0},
            ),
            patch(
                "neurodent.visualization.results.core.validate_timestamps"
            ),
        ):
            ao._process_fragment_metadata = MagicMock(
                return_value={
                    "animalday": "test WT 2025",
                    "timestamp": 0.0,
                    "animal": "test",
                    "session": "2025",
                    "genotype": "WT",
                    "sex": "M",
                }
            )

            ao.compute_windowed_analysis(
                features=["rms"],
                multiprocess_mode="dask",
                chunk_duration_s=600,
            )

            mock_stream.assert_called_once()
            # Verify n_frag_per_chunk (4th positional arg) is derived from
            # chunk_duration_s=600 / window_s=5 = 120
            call_args = mock_stream.call_args
            actual_n_frag_per_chunk = call_args[0][3]
            assert actual_n_frag_per_chunk == 120, (
                f"Expected n_frag_per_chunk=120 (600/5), got {actual_n_frag_per_chunk}"
            )


# ---------------------------------------------------------------------------
# save_fif_and_json – chunk_duration_s parameter propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveFifAndJsonChunkDurationS:
    """Verify that ``save_fif_and_json`` forwards ``chunk_duration_s`` to
    ``convert_to_mne``."""

    def test_chunk_duration_s_forwarded_to_convert_to_mne(self, tmp_path):
        """``save_fif_and_json`` should call ``convert_to_mne`` with the
        supplied ``chunk_duration_s``."""
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
                        chunk_duration_s=30.0,
                        multiprocess_mode="serial",
                    )
                except Exception:
                    pass  # saving may fail for mocked objects; we only care about the call

            # Verify chunk_duration_s was forwarded
            mock_conv.assert_called_once()
            _, kwargs = mock_conv.call_args
            assert kwargs.get("chunk_duration_s") == 30.0

    def test_chunk_duration_s_default_is_60(self):
        """Default ``chunk_duration_s`` in ``save_fif_and_json`` should be 60 s."""
        import inspect
        from neurodent.visualization.frequency_domain_results import (
            FrequencyDomainSpikeAnalysisResult,
        )

        sig = inspect.signature(FrequencyDomainSpikeAnalysisResult.save_fif_and_json)
        assert sig.parameters["chunk_duration_s"].default == 60

    def test_chunk_duration_s_in_signature(self):
        """``save_fif_and_json`` must expose a ``chunk_duration_s`` parameter."""
        import inspect
        from neurodent.visualization.frequency_domain_results import (
            FrequencyDomainSpikeAnalysisResult,
        )

        sig = inspect.signature(FrequencyDomainSpikeAnalysisResult.save_fif_and_json)
        assert "chunk_duration_s" in sig.parameters


# ---------------------------------------------------------------------------
# chunked_channel_distance_matrix – unit tests for extracted utility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkedChannelDistanceMatrix:
    """Test that ``chunked_channel_distance_matrix`` returns correct Euclidean
    distances and that chunked computation matches the non-chunked baseline."""

    def _brute_force_distance(self, data):
        """Reference implementation: full pairwise Euclidean distance."""
        from scipy.spatial.distance import squareform, pdist
        return squareform(pdist(data.T, metric="euclidean"))

    def test_identity_with_single_chunk(self):
        """When chunk_samples >= n_samples, result matches brute-force."""
        from neurodent.core.utils import chunked_channel_distance_matrix
        rng = np.random.default_rng(42)
        n_channels, n_samples = 4, 200
        traces = rng.standard_normal((n_samples, n_channels))

        result = chunked_channel_distance_matrix(
            get_traces_fn=lambda s, e: traces[s:e],
            n_channels=n_channels,
            n_samples=n_samples,
            chunk_samples=n_samples,  # single chunk
        )
        expected = self._brute_force_distance(traces)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_matches_brute_force_multiple_chunks(self):
        """Chunked result must be identical to brute-force for many small chunks."""
        from neurodent.core.utils import chunked_channel_distance_matrix
        rng = np.random.default_rng(7)
        n_channels, n_samples = 6, 500
        traces = rng.standard_normal((n_samples, n_channels))

        result = chunked_channel_distance_matrix(
            get_traces_fn=lambda s, e: traces[s:e],
            n_channels=n_channels,
            n_samples=n_samples,
            chunk_samples=37,  # deliberately non-divisor
        )
        expected = self._brute_force_distance(traces)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_symmetry(self):
        """Distance matrix must be symmetric."""
        from neurodent.core.utils import chunked_channel_distance_matrix
        rng = np.random.default_rng(99)
        n_channels, n_samples = 5, 300
        traces = rng.standard_normal((n_samples, n_channels))

        result = chunked_channel_distance_matrix(
            get_traces_fn=lambda s, e: traces[s:e],
            n_channels=n_channels,
            n_samples=n_samples,
            chunk_samples=50,
        )
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_diagonal_is_zero(self):
        """Self-distances must be zero."""
        from neurodent.core.utils import chunked_channel_distance_matrix
        rng = np.random.default_rng(11)
        n_channels, n_samples = 3, 100
        traces = rng.standard_normal((n_samples, n_channels))

        result = chunked_channel_distance_matrix(
            get_traces_fn=lambda s, e: traces[s:e],
            n_channels=n_channels,
            n_samples=n_samples,
            chunk_samples=25,
        )
        np.testing.assert_allclose(np.diag(result), 0, atol=1e-6)

    def test_non_negative(self):
        """All distances must be non-negative."""
        from neurodent.core.utils import chunked_channel_distance_matrix
        rng = np.random.default_rng(55)
        n_channels, n_samples = 4, 150
        traces = rng.standard_normal((n_samples, n_channels))

        result = chunked_channel_distance_matrix(
            get_traces_fn=lambda s, e: traces[s:e],
            n_channels=n_channels,
            n_samples=n_samples,
            chunk_samples=30,
        )
        assert np.all(result >= 0)

    def test_chunk_size_one(self):
        """Edge case: chunk_samples = 1 still produces correct result."""
        from neurodent.core.utils import chunked_channel_distance_matrix
        rng = np.random.default_rng(21)
        n_channels, n_samples = 3, 10
        traces = rng.standard_normal((n_samples, n_channels))

        result = chunked_channel_distance_matrix(
            get_traces_fn=lambda s, e: traces[s:e],
            n_channels=n_channels,
            n_samples=n_samples,
            chunk_samples=1,
        )
        expected = self._brute_force_distance(traces)
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# FDSAR chunked spike detection – boundary deduplication
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFdsarChunkedDetection:
    """Verify that chunked spike detection processes the full recording and
    deduplicates spikes at chunk boundaries."""

    def test_chunked_matches_unchunked(self):
        """Spike indices from chunked processing must match unchunked."""
        try:
            import spikeinterface.core as si
        except ImportError:
            pytest.skip("SpikeInterface not available")

        from neurodent.core.frequency_domain_spike_detection import (
            FrequencyDomainSpikeDetector,
        )

        rng = np.random.default_rng(42)
        n_channels, fs, duration = 2, 1000.0, 5.0
        n_samples = int(duration * fs)
        data = rng.standard_normal((n_channels, n_samples)) * 0.1

        # Insert clear negative spikes at known positions
        spike_positions = [500, 1500, 2500, 3500, 4500]
        for ch in range(n_channels):
            for pos in spike_positions:
                width = int(0.02 * fs)
                half = width // 2
                t = np.arange(-half, half + 1)
                spike = -5.0 * np.exp(-0.5 * (t / (width / 6)) ** 2)
                start = max(0, pos - half)
                end = min(n_samples, pos + half + 1)
                data[ch, start:end] += spike[: end - start]

        rec = si.NumpyRecording(data.T, sampling_frequency=fs)

        params = {
            "bp": [3.0, 40.0],
            "notch": 60.0,
            "notch_q": 30.0,
            "freq_slices": [10.0, 20.0],
            "sneo_percentile": 98.0,
            "cluster_gap_ms": 80.0,
            "vote_k": 1,
            "baseline_ms": 500.0,
            "search_ms": 160.0,
            "k_sigma": 3.0,
            "smooth_window": 7,
            "smooth_len": 5,
            "window_s": 0.125,
        }

        # Unchunked (None = full recording at once)
        spikes_full = FrequencyDomainSpikeDetector.detect_spikes_recording(
            rec, detection_params=params, chunk_duration_s=None,
            multiprocess_mode="serial",
        )

        # Chunked: 1-second chunks → several boundary crossings
        spikes_chunked = FrequencyDomainSpikeDetector.detect_spikes_recording(
            rec, detection_params=params, chunk_duration_s=1.0,
            multiprocess_mode="serial",
        )

        # Results must be identical (no duplicates, no missing spikes)
        for ch in range(n_channels):
            np.testing.assert_array_equal(
                spikes_chunked[ch], spikes_full[ch],
                err_msg=f"Channel {ch}: chunked vs unchunked spike indices differ",
            )

    def test_no_duplicate_spikes_at_boundaries(self):
        """Spikes near chunk boundaries must not be duplicated."""
        try:
            import spikeinterface.core as si
        except ImportError:
            pytest.skip("SpikeInterface not available")

        from neurodent.core.frequency_domain_spike_detection import (
            FrequencyDomainSpikeDetector,
        )

        rng = np.random.default_rng(7)
        fs = 1000.0
        duration = 4.0
        n_samples = int(duration * fs)
        data = rng.standard_normal((1, n_samples)) * 0.1

        # Place a spike exactly at the 2-second boundary
        pos = 2000
        width = int(0.02 * fs)
        half = width // 2
        t = np.arange(-half, half + 1)
        spike = -5.0 * np.exp(-0.5 * (t / (width / 6)) ** 2)
        data[0, pos - half : pos + half + 1] += spike

        rec = si.NumpyRecording(data.T, sampling_frequency=fs)

        params = {
            "bp": [3.0, 40.0],
            "notch": 60.0,
            "notch_q": 30.0,
            "freq_slices": [10.0, 20.0],
            "sneo_percentile": 98.0,
            "cluster_gap_ms": 80.0,
            "vote_k": 1,
            "baseline_ms": 500.0,
            "search_ms": 160.0,
            "k_sigma": 3.0,
            "smooth_window": 7,
            "smooth_len": 5,
            "window_s": 0.125,
        }

        spikes = FrequencyDomainSpikeDetector.detect_spikes_recording(
            rec, detection_params=params, chunk_duration_s=2.0,
            multiprocess_mode="serial",
        )

        # No duplicate indices
        assert len(spikes[0]) == len(np.unique(spikes[0])), \
            "Duplicate spike indices found at chunk boundary"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputValidation:
    """Verify that chunk-related functions reject invalid parameters early
    instead of entering infinite loops or producing cryptic errors."""

    # --- chunked_channel_distance_matrix ---

    def test_distance_matrix_rejects_zero_chunk_samples(self):
        """chunk_samples=0 must raise ValueError."""
        from neurodent.core.utils import chunked_channel_distance_matrix

        with pytest.raises(ValueError, match="chunk_samples must be >= 1"):
            chunked_channel_distance_matrix(
                get_traces_fn=lambda s, e: np.zeros((e - s, 2)),
                n_channels=2, n_samples=10, chunk_samples=0,
            )

    def test_distance_matrix_rejects_negative_chunk_samples(self):
        """chunk_samples=-5 must raise ValueError."""
        from neurodent.core.utils import chunked_channel_distance_matrix

        with pytest.raises(ValueError, match="chunk_samples must be >= 1"):
            chunked_channel_distance_matrix(
                get_traces_fn=lambda s, e: np.zeros((e - s, 2)),
                n_channels=2, n_samples=10, chunk_samples=-5,
            )

    # --- cache_fragments_to_zarr ---

    def test_cache_fragments_rejects_zero_chunk_size(self, tmp_path):
        """chunk_size=0 must raise ValueError."""
        frags = _make_fragments(5)
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            cache_fragments_to_zarr(frags, 5, chunk_size=0, tmpdir=str(tmp_path))

    def test_cache_fragments_rejects_negative_chunk_size(self, tmp_path):
        """chunk_size=-1 must raise ValueError."""
        frags = _make_fragments(5)
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            cache_fragments_to_zarr(frags, 5, chunk_size=-1, tmpdir=str(tmp_path))

    def test_cache_fragments_rejects_non_int_chunk_size(self, tmp_path):
        """chunk_size=2.5 must raise TypeError."""
        frags = _make_fragments(5)
        with pytest.raises(TypeError, match="chunk_size must be an integer"):
            cache_fragments_to_zarr(frags, 5, chunk_size=2.5, tmpdir=str(tmp_path))

    # --- detect_spikes_recording chunk_duration_s ---

    def test_spike_detection_tiny_chunk_duration_no_infinite_loop(self):
        """Very small chunk_duration_s should not cause an infinite loop;
        chunk_samples is clamped to at least 1."""
        try:
            import spikeinterface as si
        except ImportError:
            pytest.skip("spikeinterface not installed")

        from neurodent.core.frequency_domain_spike_detection import (
            FrequencyDomainSpikeDetector,
        )

        rng = np.random.default_rng(42)
        n_channels, n_samples, fs = 1, 100, 1000.0
        traces = rng.standard_normal((n_samples, n_channels)).astype(np.float32)
        rec = si.core.NumpyRecording(
            traces_list=[traces], sampling_frequency=fs
        )

        params = {
            "baseline_ms": 50.0,
            "k_sigma": 3.0,
            "smooth_window": 7,
            "smooth_len": 5,
            "window_s": 0.05,
        }

        # chunk_duration_s so tiny it rounds to 0 samples → clamped to 1
        spikes = FrequencyDomainSpikeDetector.detect_spikes_recording(
            rec, detection_params=params,
            chunk_duration_s=1e-10,
            multiprocess_mode="serial",
        )
        # Just verify it terminates and returns correct types
        assert isinstance(spikes, list)
        assert len(spikes) == n_channels

    def test_spike_detection_none_chunk_with_empty_recording_raises(self):
        """chunk_duration_s=None on a recording with 0 samples must raise."""
        try:
            import spikeinterface as si
        except ImportError:
            pytest.skip("spikeinterface not installed")

        from neurodent.core.frequency_domain_spike_detection import (
            FrequencyDomainSpikeDetector,
        )

        traces = np.empty((0, 1), dtype=np.float32)
        rec = si.core.NumpyRecording(
            traces_list=[traces], sampling_frequency=1000.0
        )

        with pytest.raises(ValueError, match="no samples"):
            FrequencyDomainSpikeDetector.detect_spikes_recording(
                rec, chunk_duration_s=None, multiprocess_mode="serial",
            )

    # --- LOF lof_chunk_duration_s validation ---

    def test_lof_rejects_zero_chunk_duration(self):
        """lof_chunk_duration_s=0 must raise ValueError."""
        from neurodent.core.core import LongRecordingOrganizer

        lro = LongRecordingOrganizer.__new__(LongRecordingOrganizer)

        mock_rec = MagicMock()
        mock_rec.get_num_channels.return_value = 3
        mock_rec.get_total_samples.return_value = 1000
        mock_rec.get_sampling_frequency.return_value = 1000.0
        lro.LongRecording = mock_rec

        with pytest.raises(ValueError, match="lof_chunk_duration_s must be positive"):
            lro._compute_lof_scores(lof_chunk_duration_s=0)

    def test_lof_rejects_negative_chunk_duration(self):
        """lof_chunk_duration_s=-10 must raise ValueError."""
        from neurodent.core.core import LongRecordingOrganizer

        lro = LongRecordingOrganizer.__new__(LongRecordingOrganizer)

        mock_rec = MagicMock()
        mock_rec.get_num_channels.return_value = 3
        mock_rec.get_total_samples.return_value = 1000
        mock_rec.get_sampling_frequency.return_value = 1000.0
        lro.LongRecording = mock_rec

        with pytest.raises(ValueError, match="lof_chunk_duration_s must be positive"):
            lro._compute_lof_scores(lof_chunk_duration_s=-10)
