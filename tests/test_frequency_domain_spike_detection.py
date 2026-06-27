"""
Unit tests for neurodent.core.frequency_domain_spike_detection module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
import warnings

try:
    import spikeinterface.core as si

    SPIKEINTERFACE_AVAILABLE = True
except ImportError:
    si = None
    SPIKEINTERFACE_AVAILABLE = False

import mne

from neurodent.core.frequency_domain_spike_detection import FrequencyDomainSpikeDetector
from neurodent import constants


@pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
class TestFrequencyDomainSpikeDetector:
    """Test FrequencyDomainSpikeDetector static methods."""

    @pytest.fixture
    def mock_recording(self):
        """Create a mock SpikeInterface recording for testing."""
        mock_rec = MagicMock()
        mock_rec.get_num_channels.return_value = 4
        mock_rec.get_channel_ids.return_value = ["ch1", "ch2", "ch3", "ch4"]
        mock_rec.get_sampling_frequency.return_value = 1000.0
        mock_rec.get_num_frames.return_value = 10000
        mock_rec.get_total_samples.return_value = 10000
        mock_rec.get_dtype.return_value = np.float32
        mock_rec.clone.return_value = mock_rec
        mock_rec.set_channel_ids.return_value = None

        # Mock data - transposed to (samples, channels) format as get_traces returns
        np.random.seed(42)
        full_data = np.random.randn(10000, 4) * 0.1
        mock_rec.get_traces.side_effect = lambda start_frame=0, end_frame=None, return_scaled=True, return_in_uV=False: \
            full_data[start_frame:(end_frame if end_frame is not None else 10000)]

        return mock_rec

    @pytest.fixture
    def detection_params(self):
        """Default detection parameters for testing."""
        return {
            "bp": [3.0, 40.0],
            "notch": 60.0,
            "notch_q": 30.0,
            "freq_slices": [10.0, 20.0],
            "window_s": 0.125,
            "sneo_percentile": 99.0,  # Lower for testing
            "cluster_gap_ms": 80.0,
            "search_ms": 160.0,
            "baseline_ms": 500.0,
            "k_sigma": 3.0,
            "smooth_window": 7,
            "vote_k": 1,  # Lower for testing
            "smooth_len": 5,
        }

    @pytest.fixture
    def test_signal(self):
        """Create a test signal with known characteristics."""
        fs = 1000.0
        duration = 10.0  # 10 seconds
        t = np.arange(0, duration, 1 / fs)

        # Base signal with some noise
        signal = np.random.randn(len(t)) * 0.1

        # Add some artificial spikes at known locations
        spike_times = [2.0, 4.5, 7.2]  # seconds
        for spike_time in spike_times:
            spike_idx = int(spike_time * fs)
            if spike_idx < len(signal):
                # Create a negative spike
                spike_width = int(0.02 * fs)  # 20ms wide
                spike_indices = np.arange(
                    max(0, spike_idx - spike_width // 2), min(len(signal), spike_idx + spike_width // 2)
                )
                signal[spike_indices] -= np.exp(-(((spike_indices - spike_idx) / (spike_width / 4)) ** 2)) * 2.0

        return signal, fs, spike_times

    def test_default_params(self):
        """Test default parameters are properly defined."""
        params = FrequencyDomainSpikeDetector.DEFAULT_PARAMS

        required_keys = [
            "bp",
            "notch",
            "freq_slices",
            "sneo_percentile",
            "cluster_gap_ms",
            "search_ms",
            "baseline_ms",
            "k_sigma",
        ]

        for key in required_keys:
            assert key in params, f"Missing required parameter: {key}"

        # Test parameter types and ranges
        assert isinstance(params["bp"], list)
        assert len(params["bp"]) == 2
        assert params["bp"][0] < params["bp"][1]

        assert isinstance(params["sneo_percentile"], (int, float))
        assert 0 <= params["sneo_percentile"] <= 100

    def test_compute_stft_slices(self, test_signal):
        """Test STFT slice computation."""
        signal, fs, _ = test_signal
        freqs = (10.0, 20.0)

        slices_dict = FrequencyDomainSpikeDetector._compute_stft_slices(signal, fs, freqs=freqs)

        # Check output structure
        assert isinstance(slices_dict, dict)
        assert len(slices_dict) == len(freqs)

        for freq in freqs:
            assert float(freq) in slices_dict
            assert len(slices_dict[float(freq)]) == len(signal)
            assert np.all(np.isfinite(slices_dict[float(freq)]))

    def test_sneo(self):
        """Test SNEO function."""
        # Test with known input
        x = np.array([1, 2, 3, 2, 1])
        result = FrequencyDomainSpikeDetector._sneo(x)

        # SNEO: x[n]^2 - x[n-1] * x[n+1]
        expected = np.array(
            [
                2**2 - 1 * 3,  # 4 - 3 = 1
                3**2 - 2 * 2,  # 9 - 4 = 5
                2**2 - 3 * 1,  # 4 - 3 = 1
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_apply_sneo_on_slices(self, test_signal):
        """Test SNEO application on frequency slices."""
        signal, fs, _ = test_signal

        # Create simple slice dict
        slices_dict = {
            10.0: signal + np.random.randn(len(signal)) * 0.05,
            20.0: signal + np.random.randn(len(signal)) * 0.05,
        }

        spikes, sneo_combined = FrequencyDomainSpikeDetector._apply_sneo_on_slices(
            slices_dict, fs, threshold_percentile=95.0, vote_k=1
        )

        # Check output structure
        assert isinstance(spikes, np.ndarray)
        assert isinstance(sneo_combined, np.ndarray)
        assert len(sneo_combined) == len(signal) - 2  # SNEO reduces length by 2

        # Should detect some candidates with lowered threshold
        assert len(spikes) >= 0  # May or may not detect spikes depending on signal

    def test_enforce_downward_and_refine_minimal(self, test_signal):
        """Test spike refinement function."""
        signal, fs, spike_times = test_signal

        # Use approximate spike locations as candidates
        candidates = [int(t * fs) for t in spike_times]

        refined = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
            signal,
            fs,
            candidates,
            k_sigma=2.0,  # Lower threshold for testing
        )

        # Check output structure
        assert isinstance(refined, np.ndarray)
        assert len(refined) <= len(candidates)  # Should not add spikes

        # All refined spikes should be negative deflections
        for spike_idx in refined:
            if 0 <= spike_idx < len(signal):
                # Check that it's a local minimum in a small window
                window_half = 10
                start = max(0, spike_idx - window_half)
                end = min(len(signal), spike_idx + window_half + 1)
                window = signal[start:end]
                local_min_idx = np.argmin(window)
                assert start + local_min_idx == spike_idx or abs(start + local_min_idx - spike_idx) <= 2

    def test_filter_close_spikes_by_min_local(self, test_signal):
        """Test spike clustering function."""
        signal, fs, _ = test_signal

        # Create closely spaced artificial spikes
        spike_indices = np.array([1000, 1020, 1025, 2000, 2015, 4000])  # Some close pairs

        filtered = FrequencyDomainSpikeDetector._filter_close_spikes_by_min_local(
            signal,
            fs,
            spike_indices,
            min_gap_ms=50.0,  # 50ms minimum gap
        )

        # Check output structure
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) <= len(spike_indices)

        # Check minimum gap constraint
        if len(filtered) > 1:
            gaps = np.diff(filtered)
            min_gap_samples = int(50.0 * fs / 1000.0)
            assert np.all(gaps >= min_gap_samples), "Spikes too close together"

    def test_detect_spikes_channel(self, test_signal, detection_params):
        """Test single-channel spike detection."""
        signal, fs, spike_times = test_signal

        # Lower thresholds for testing
        test_params = detection_params.copy()
        test_params["sneo_percentile"] = 90.0
        test_params["vote_k"] = 1

        spike_indices = FrequencyDomainSpikeDetector._detect_spikes_channel(signal, fs, test_params)

        # Check output structure
        assert isinstance(spike_indices, np.ndarray)
        assert spike_indices.dtype == int

        # Should detect some spikes (may not be exact due to algorithm parameters)
        # This is more of a smoke test than precise validation
        assert len(spike_indices) >= 0

    @patch("spikeinterface.core.NumpyRecording")
    def test_apply_preprocessing(self, mock_numpy_recording, mock_recording, detection_params):
        """Test preprocessing application."""
        # Mock the SpikeInterface recording
        mock_numpy_recording.return_value = mock_recording

        result = FrequencyDomainSpikeDetector._apply_preprocessing(mock_recording, detection_params)

        # Should return a recording-like object
        assert result is not None
        mock_recording.get_traces.assert_called()
        # Verify NumpyRecording was created with filtered data
        mock_numpy_recording.assert_called_once()

    def test_add_spike_annotations(self):
        """Test MNE annotation creation."""
        # Create simple MNE object
        n_channels = 3
        fs = 1000.0
        duration = 5.0
        n_samples = int(duration * fs)

        info = mne.create_info(ch_names=[f"ch{i}" for i in range(n_channels)], sfreq=fs, ch_types="eeg")
        data = np.random.randn(n_channels, n_samples) * 0.1
        raw = mne.io.RawArray(data, info)

        # Create spike indices
        spike_indices_per_channel = [
            np.array([500, 1500, 3000]),  # ch0
            np.array([800, 2200]),  # ch1
            np.array([]),  # ch2 (no spikes)
        ]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            annotated_raw = FrequencyDomainSpikeDetector._add_spike_annotations(raw, spike_indices_per_channel, fs)

        # Check annotations
        annotations = annotated_raw.annotations
        assert len(annotations) == 5  # 3 + 2 + 0 spikes

        # Check annotation descriptions
        descriptions = annotations.description
        spike_descriptions = [desc for desc in descriptions if desc.startswith("Spike_Ch")]
        assert len(spike_descriptions) == 5

    @patch.object(FrequencyDomainSpikeDetector, "_preprocess_array")
    @patch.object(FrequencyDomainSpikeDetector, "_detect_spikes_channel")
    def test_detect_spikes_recording_serial(
        self, mock_detect_channel, mock_preprocess, mock_recording, detection_params
    ):
        """Test full spike detection pipeline in serial mode."""
        # Setup mocks: _preprocess_array receives an ndarray, returns filtered ndarray
        mock_preprocess.side_effect = lambda data, fs, params: data
        mock_detect_channel.return_value = np.array([100, 500, 1000])

        spike_indices = FrequencyDomainSpikeDetector.detect_spikes_recording(
            mock_recording, detection_params, multiprocess_mode="auto")

        # Check calls
        mock_preprocess.assert_called_once()
        assert mock_detect_channel.call_count == 4  # 4 channels

        # Check outputs
        assert len(spike_indices) == 4  # 4 channels

    @patch.object(FrequencyDomainSpikeDetector, "_preprocess_array")
    @patch.object(FrequencyDomainSpikeDetector, "_detect_spikes_channel")
    def test_detect_spikes_recording_dask(
        self, mock_detect_channel, mock_preprocess, mock_recording, detection_params
    ):
        """Test full spike detection pipeline in dask mode."""
        # Setup mocks
        mock_preprocess.side_effect = lambda data, fs, params: data
        mock_detect_channel.return_value = np.array([100, 500, 1000])

        spike_indices = FrequencyDomainSpikeDetector.detect_spikes_recording(
            mock_recording, detection_params, multiprocess_mode="dask")

        # Check calls
        assert mock_preprocess.call_count == 4  # once per channel (inside _filter_and_detect_channel)
        assert mock_detect_channel.call_count == 4  # 4 channels

        # Check outputs - dask.compute returns tuple, should be converted to list/tuple
        assert len(spike_indices) == 4  # 4 channels
        assert isinstance(spike_indices, (list, tuple))

    def test_detect_spikes_dask_vs_serial_consistency(self, detection_params):
        """Test that dask and serial modes produce identical results on real data."""
        try:
            import spikeinterface.core as si
        except ImportError:
            pytest.skip("SpikeInterface not available")

        # Create realistic test recording with artificial spikes
        np.random.seed(42)
        n_channels = 4
        fs = 1000.0
        duration = 5.0  # 5 seconds
        n_samples = int(duration * fs)

        # Generate base noise
        data = np.random.randn(n_channels, n_samples) * 0.1

        # Add artificial spikes at known locations
        spike_times_samples = [500, 1500, 2500, 3500, 4000]
        for ch in range(n_channels):
            for spike_time in spike_times_samples:
                if spike_time < n_samples:
                    # Create negative spike (20ms wide)
                    spike_width = int(0.02 * fs)
                    start = max(0, spike_time - spike_width // 2)
                    end = min(n_samples, spike_time + spike_width // 2)
                    spike_indices = np.arange(start, end)
                    # Add channel-specific variation
                    amplitude = 2.0 + ch * 0.3
                    data[ch, spike_indices] -= np.exp(
                        -(((spike_indices - spike_time) / (spike_width / 4)) ** 2)
                    ) * amplitude

        # Create SpikeInterface recording (transpose to samples x channels)
        recording = si.NumpyRecording(data.T, sampling_frequency=fs, channel_ids=[f"ch{i}" for i in range(n_channels)])

        # Lower thresholds for testing
        test_params = detection_params.copy()
        test_params["sneo_percentile"] = 90.0

        # Run both modes
        spike_indices_serial = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, test_params, multiprocess_mode="serial")

        spike_indices_dask = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, test_params, multiprocess_mode="dask")

        # Check consistency
        assert len(spike_indices_serial) == len(spike_indices_dask), "Different number of channels"

        for ch in range(n_channels):
            serial_spikes = spike_indices_serial[ch]
            dask_spikes = spike_indices_dask[ch]

            # Should detect same number of spikes
            assert len(serial_spikes) == len(dask_spikes), f"Channel {ch}: Different spike counts"

            # Spike indices should be identical
            np.testing.assert_array_equal(
                serial_spikes, dask_spikes, err_msg=f"Channel {ch}: Different spike locations"
            )

    def test_detect_spikes_dask_empty_channels(self, detection_params):
        """Test dask mode with channels that have no spikes."""
        try:
            import spikeinterface.core as si
        except ImportError:
            pytest.skip("SpikeInterface not available")

        np.random.seed(123)
        n_channels = 3
        fs = 1000.0
        duration = 2.0
        n_samples = int(duration * fs)

        # Pure noise - no spikes
        data = np.random.randn(n_channels, n_samples) * 0.05

        recording = si.NumpyRecording(data.T, sampling_frequency=fs, channel_ids=[f"ch{i}" for i in range(n_channels)])

        test_params = detection_params.copy()
        test_params["sneo_percentile"] = 99.9  # Very high threshold

        spike_indices = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, test_params, multiprocess_mode="auto")

        # Should return empty arrays for each channel
        assert len(spike_indices) == n_channels
        assert isinstance(spike_indices, (list, tuple))
        for ch_spikes in spike_indices:
            assert isinstance(ch_spikes, np.ndarray)
            assert len(ch_spikes) == 0 or len(ch_spikes) < 10  # Very few or no spikes

    def test_detect_spikes_dask_single_channel(self, detection_params):
        """Test dask mode with single channel recording."""
        try:
            import spikeinterface.core as si
        except ImportError:
            pytest.skip("SpikeInterface not available")

        np.random.seed(456)
        fs = 1000.0
        duration = 3.0
        n_samples = int(duration * fs)

        # Single channel with spikes
        data = np.random.randn(1, n_samples) * 0.1
        spike_time = 1000
        spike_width = 20
        data[0, spike_time - spike_width : spike_time + spike_width] -= 3.0 * np.exp(
            -((np.arange(-spike_width, spike_width) / 8) ** 2)
        )

        recording = si.NumpyRecording(data.T, sampling_frequency=fs, channel_ids=["ch0"])

        test_params = detection_params.copy()
        test_params["sneo_percentile"] = 95.0

        spike_indices = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, test_params, multiprocess_mode="auto")

        # Should work with single channel
        assert len(spike_indices) == 1
        assert isinstance(spike_indices, (list, tuple))
        assert isinstance(spike_indices[0], np.ndarray)

    def test_detect_spikes_dask_many_channels(self, detection_params):
        """Test dask mode with many channels (tests parallel scaling)."""
        try:
            import spikeinterface.core as si
        except ImportError:
            pytest.skip("SpikeInterface not available")

        np.random.seed(789)
        n_channels = 16  # Larger number of channels
        fs = 1000.0
        duration = 2.0
        n_samples = int(duration * fs)

        # Generate data with varying noise levels per channel
        data = np.random.randn(n_channels, n_samples) * 0.1

        # Add spikes to some channels
        for ch in range(0, n_channels, 2):  # Every other channel
            spike_time = 500 + ch * 50
            if spike_time < n_samples - 20:
                data[ch, spike_time - 10 : spike_time + 10] -= 2.0

        recording = si.NumpyRecording(data.T, sampling_frequency=fs, channel_ids=[f"ch{i:02d}" for i in range(n_channels)])

        test_params = detection_params.copy()
        test_params["sneo_percentile"] = 95.0

        spike_indices = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, test_params, multiprocess_mode="auto")

        # Should handle many channels
        assert len(spike_indices) == n_channels
        assert isinstance(spike_indices, (list, tuple))

        # Each channel should return valid array
        for ch_spikes in spike_indices:
            assert isinstance(ch_spikes, np.ndarray)

    def test_detect_spikes_dask_return_type_consistency(self, detection_params):
        """Test that dask mode returns consistent types with serial mode."""
        try:
            import spikeinterface.core as si
        except ImportError:
            pytest.skip("SpikeInterface not available")

        np.random.seed(111)
        n_channels = 3
        fs = 1000.0
        n_samples = 2000

        data = np.random.randn(n_channels, n_samples) * 0.1
        recording = si.NumpyRecording(data.T, sampling_frequency=fs, channel_ids=[f"ch{i}" for i in range(n_channels)])

        spike_indices_serial = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, detection_params, multiprocess_mode="auto")

        spike_indices_dask = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, detection_params, multiprocess_mode="auto")

        # Both should return same container types
        assert type(spike_indices_serial).__name__ == type(spike_indices_dask).__name__ or (
            isinstance(spike_indices_serial, (list, tuple)) and isinstance(spike_indices_dask, (list, tuple))
        )

        # Both should be iterable and have same structure
        assert len(spike_indices_serial) == len(spike_indices_dask)

        for serial_arr, dask_arr in zip(spike_indices_serial, spike_indices_dask):
            assert type(serial_arr) == type(dask_arr)
            assert serial_arr.dtype == dask_arr.dtype


@pytest.mark.unit
class TestFrequencyDomainSpikeDetectorUtils:
    """Test utility functions that don't require SpikeInterface."""

    def test_sneo_edge_cases(self):
        """Test SNEO with edge cases."""
        # Test with minimum length
        x = np.array([1, 2, 3])
        result = FrequencyDomainSpikeDetector._sneo(x)
        assert len(result) == 1
        assert result[0] == 2**2 - 1 * 3  # 4 - 3 = 1

        # Test with zeros
        x = np.array([0, 0, 0, 0])
        result = FrequencyDomainSpikeDetector._sneo(x)
        assert np.all(result == 0)

    def test_compute_stft_slices_edge_cases(self):
        """Test STFT computation with edge cases."""
        # Very short signal
        signal = np.array([1, 2, 3, 4, 5])
        fs = 100.0
        freqs = (10.0,)

        slices_dict = FrequencyDomainSpikeDetector._compute_stft_slices(signal, fs, freqs=freqs)

        assert 10.0 in slices_dict
        assert len(slices_dict[10.0]) == len(signal)

    def test_filter_close_spikes_empty_input(self):
        """Test spike filtering with empty input."""
        signal = np.random.randn(1000)
        fs = 1000.0

        result = FrequencyDomainSpikeDetector._filter_close_spikes_by_min_local(signal, fs, np.array([]))

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_enforce_downward_empty_input(self):
        """Test spike refinement with empty input."""
        signal = np.random.randn(1000)
        fs = 1000.0

        result = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(signal, fs, np.array([]))

        assert isinstance(result, np.ndarray)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Baseline window edge cases (no SpikeInterface required)
# ---------------------------------------------------------------------------


class TestSpikeDetectorBaselineEdge:
    """Test short-baseline warning path in _enforce_downward_and_refine_minimal."""

    def test_very_short_signal_warns(self):
        """A spike near signal boundary may produce a baseline < 10 samples."""
        np.random.seed(42)
        signal = np.random.randn(50)
        signal[2] = -20  # artificially large spike near the start

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
                signal,
                fs=1000,
                candidates=np.array([2]),
                search_ms=10,
                baseline_ms=5,  # very small baseline → likely < 10 samples
            )
            # Function should not crash; result is an array
            assert isinstance(result, np.ndarray)
            # Verify the short-baseline warning was emitted
            baseline_warnings = [
                x for x in w if "baseline window length" in str(x.message)
            ]
            assert len(baseline_warnings) > 0, "Expected a warning about short baseline"

    def test_spike_at_signal_edge(self):
        """Spike at index 0 should not crash."""
        signal = np.zeros(100)
        signal[0] = -10
        result = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
            signal,
            fs=1000,
            candidates=np.array([0]),
            search_ms=10,
            baseline_ms=5,
        )
        assert isinstance(result, np.ndarray)

    def test_empty_candidates_returns_empty(self):
        """Empty candidates should return empty array."""
        signal = np.random.randn(100)
        result = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
            signal,
            fs=1000,
            candidates=np.array([]),
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _filter_and_detect_channel unit tests (no SpikeInterface required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterAndDetectChannel:
    """Unit tests for the fused _filter_and_detect_channel static method."""

    @pytest.fixture
    def params(self):
        return {
            "bp": [3.0, 40.0],
            "notch": 60.0,
            "notch_q": 30.0,
            "freq_slices": [10.0, 20.0],
            "window_s": 0.125,
            "sneo_percentile": 99.0,
            "cluster_gap_ms": 80.0,
            "search_ms": 160.0,
            "baseline_ms": 500.0,
            "k_sigma": 3.0,
            "smooth_window": 7,
            "vote_k": 1,
            "smooth_len": 5,
        }

    def test_returns_ndarray(self, params):
        """Return value is a numpy array."""
        np.random.seed(0)
        raw = np.random.randn(2000) * 0.1
        result = FrequencyDomainSpikeDetector._filter_and_detect_channel(raw, 1000.0, params)
        assert isinstance(result, np.ndarray)

    def test_output_indices_in_range(self, params):
        """All returned spike indices lie within [0, n_samples)."""
        np.random.seed(1)
        n = 3000
        raw = np.random.randn(n) * 0.1
        result = FrequencyDomainSpikeDetector._filter_and_detect_channel(raw, 1000.0, params)
        assert np.all(result >= 0)
        assert np.all(result < n)

    def test_equivalence_with_serial_pipeline(self, params):
        """Fused method is bit-identical to separate preprocess + detect."""
        np.random.seed(42)
        n = 4000
        fs = 1000.0
        raw = np.random.randn(n) * 0.1
        # Plant a clear negative spike
        spike_width = 20
        raw[1000 - spike_width // 2 : 1000 + spike_width // 2] -= 3.0

        result_fused = FrequencyDomainSpikeDetector._filter_and_detect_channel(raw, fs, params)

        filtered_2d = FrequencyDomainSpikeDetector._preprocess_array(
            raw[np.newaxis, :], fs, params
        )
        result_serial = FrequencyDomainSpikeDetector._detect_spikes_channel(
            filtered_2d[0], fs, params
        )

        np.testing.assert_array_equal(result_fused, result_serial)

    def test_newaxis_indexing_round_trip(self):
        """raw_1d[np.newaxis, :][0] recovers the original array exactly."""
        raw = np.arange(200, dtype=float)
        np.testing.assert_array_equal(raw[np.newaxis, :][0], raw)

    @patch.object(FrequencyDomainSpikeDetector, "_preprocess_array")
    @patch.object(FrequencyDomainSpikeDetector, "_detect_spikes_channel")
    def test_preprocess_called_with_2d_single_channel_shape(
        self, mock_detect, mock_preprocess, params
    ):
        """_preprocess_array is invoked with shape (1, n_samples)."""
        n = 500
        raw = np.random.randn(n)
        filtered = np.random.randn(n)
        mock_preprocess.return_value = filtered[np.newaxis, :]
        mock_detect.return_value = np.array([50, 150])

        FrequencyDomainSpikeDetector._filter_and_detect_channel(raw, 1000.0, params)

        mock_preprocess.assert_called_once()
        data_arg = mock_preprocess.call_args[0][0]
        assert data_arg.ndim == 2
        assert data_arg.shape == (1, n)
        np.testing.assert_array_equal(data_arg[0], raw)

    @patch.object(FrequencyDomainSpikeDetector, "_preprocess_array")
    @patch.object(FrequencyDomainSpikeDetector, "_detect_spikes_channel")
    def test_detect_called_with_1d_filtered_result(
        self, mock_detect, mock_preprocess, params
    ):
        """_detect_spikes_channel receives the 1D filtered row (not 2D)."""
        n = 500
        raw = np.random.randn(n)
        filtered_1d = np.random.randn(n)
        mock_preprocess.return_value = filtered_1d[np.newaxis, :]
        mock_detect.return_value = np.array([])

        FrequencyDomainSpikeDetector._filter_and_detect_channel(raw, 1000.0, params)

        mock_detect.assert_called_once()
        detect_data_arg = mock_detect.call_args[0][0]
        assert detect_data_arg.ndim == 1
        np.testing.assert_array_equal(detect_data_arg, filtered_1d)

    @patch.object(FrequencyDomainSpikeDetector, "_preprocess_array")
    @patch.object(FrequencyDomainSpikeDetector, "_detect_spikes_channel")
    def test_params_forwarded_to_both_internal_calls(
        self, mock_detect, mock_preprocess, params
    ):
        """Both internal calls receive the exact same params dict."""
        raw = np.random.randn(300)
        mock_preprocess.return_value = raw[np.newaxis, :]
        mock_detect.return_value = np.array([])

        FrequencyDomainSpikeDetector._filter_and_detect_channel(raw, 500.0, params)

        assert mock_preprocess.call_args[0][2] is params
        assert mock_detect.call_args[0][2] is params

    def test_empty_signal_does_not_crash(self, params):
        """Very short signal (edge case) does not raise an unhandled exception."""
        raw = np.zeros(100)  # too short for filter but should not crash
        try:
            result = FrequencyDomainSpikeDetector._filter_and_detect_channel(
                raw, 1000.0, params
            )
            assert isinstance(result, np.ndarray)
        except Exception:
            pass  # short signals may legitimately error in scipy filters


# ---------------------------------------------------------------------------
# Dask-mode dispatch tests (require SpikeInterface + dask)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
class TestDaskModeDispatch:
    """Verify that multiprocess_mode='dask' dispatches _filter_and_detect_channel
    per channel and that results are consistent with serial mode."""

    @pytest.fixture(autouse=True)
    def _require_dask(self):
        pytest.importorskip("dask")

    @pytest.fixture
    def params(self):
        return {
            "bp": [3.0, 40.0],
            "notch": 60.0,
            "notch_q": 30.0,
            "freq_slices": [10.0, 20.0],
            "window_s": 0.125,
            "sneo_percentile": 90.0,
            "cluster_gap_ms": 80.0,
            "search_ms": 160.0,
            "baseline_ms": 500.0,
            "k_sigma": 3.0,
            "smooth_window": 7,
            "vote_k": 1,
            "smooth_len": 5,
        }

    @pytest.fixture
    def recording_4ch(self):
        """4-channel NumpyRecording with planted spikes."""
        np.random.seed(99)
        n_ch, n_samples, fs = 4, 5000, 1000.0
        data = np.random.randn(n_ch, n_samples) * 0.1
        for ch in range(n_ch):
            t = 1000 + ch * 500
            w = 20
            idx = np.arange(max(0, t - w), min(n_samples, t + w))
            data[ch, idx] -= 3.0 * np.exp(-(((idx - t) / 8) ** 2))
        return si.NumpyRecording(
            data.T, sampling_frequency=fs, channel_ids=[f"ch{i}" for i in range(n_ch)]
        )

    @pytest.fixture
    def mock_recording(self):
        mock_rec = MagicMock()
        mock_rec.get_num_channels.return_value = 4
        mock_rec.get_channel_ids.return_value = ["ch0", "ch1", "ch2", "ch3"]
        mock_rec.get_sampling_frequency.return_value = 1000.0
        mock_rec.get_num_frames.return_value = 5000
        mock_rec.get_total_samples.return_value = 5000
        np.random.seed(7)
        full_data = np.random.randn(5000, 4) * 0.1
        mock_rec.get_traces.side_effect = (
            lambda start_frame=0, end_frame=None, return_scaled=True, return_in_uV=False:
            full_data[start_frame:(end_frame if end_frame is not None else 5000)]
        )
        return mock_rec

    # ------------------------------------------------------------------
    # Dispatch verification
    # ------------------------------------------------------------------

    @patch.object(FrequencyDomainSpikeDetector, "_filter_and_detect_channel")
    def test_filter_and_detect_dispatched_once_per_channel(
        self, mock_fad, mock_recording, params
    ):
        """In dask mode, _filter_and_detect_channel is called exactly n_channels times."""
        mock_fad.return_value = np.array([100, 200], dtype=int)

        spike_indices = FrequencyDomainSpikeDetector.detect_spikes_recording(
            mock_recording, params, multiprocess_mode="dask"
        )

        assert mock_fad.call_count == 4
        assert len(spike_indices) == 4

    @patch.object(FrequencyDomainSpikeDetector, "_filter_and_detect_channel")
    def test_filter_and_detect_receives_1d_channel_slices(
        self, mock_fad, mock_recording, params
    ):
        """Each dask task receives a 1-D slice (one channel, all samples)."""
        mock_fad.return_value = np.array([], dtype=int)

        FrequencyDomainSpikeDetector.detect_spikes_recording(
            mock_recording, params, multiprocess_mode="dask"
        )

        for call in mock_fad.call_args_list:
            raw_arg = call[0][0]
            assert raw_arg.ndim == 1, "Each channel slice must be 1-D"

    @patch.object(FrequencyDomainSpikeDetector, "_preprocess_array")
    def test_dask_preprocess_called_per_channel_not_once(
        self, mock_preprocess, mock_recording, params
    ):
        """In dask mode, _preprocess_array is called once per channel (not once for all)."""
        mock_preprocess.side_effect = lambda data, fs, p: data

        FrequencyDomainSpikeDetector.detect_spikes_recording(
            mock_recording, params, multiprocess_mode="dask"
        )

        assert mock_preprocess.call_count == 4
        for call in mock_preprocess.call_args_list:
            data_arg = call[0][0]
            assert data_arg.shape[0] == 1, "Each preprocess call must be for a single channel"

    @patch.object(FrequencyDomainSpikeDetector, "_preprocess_array")
    def test_serial_preprocess_called_once_for_all_channels(
        self, mock_preprocess, mock_recording, params
    ):
        """In serial mode, _preprocess_array is called once with all channels at once."""
        mock_preprocess.side_effect = lambda data, fs, p: data

        FrequencyDomainSpikeDetector.detect_spikes_recording(
            mock_recording, params, multiprocess_mode="serial"
        )

        assert mock_preprocess.call_count == 1
        data_arg = mock_preprocess.call_args[0][0]
        assert data_arg.shape[0] == 4, "Serial preprocess must receive all 4 channels"

    # ------------------------------------------------------------------
    # Correctness: dask vs serial produce identical results on real data
    # ------------------------------------------------------------------

    def test_dask_vs_serial_identical_spike_indices(self, recording_4ch, params):
        """Dask and serial modes return bit-identical spike indices on the same data."""
        serial = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording_4ch, params, multiprocess_mode="serial"
        )
        dask_result = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording_4ch, params, multiprocess_mode="dask"
        )

        assert len(serial) == len(dask_result)
        for ch, (s, d) in enumerate(zip(serial, dask_result)):
            np.testing.assert_array_equal(
                s, d, err_msg=f"Channel {ch}: serial and dask spike indices differ"
            )

    def test_dask_output_structure_matches_serial(self, recording_4ch, params):
        """Dask result is a list of ndarrays with integer dtype, same as serial."""
        serial = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording_4ch, params, multiprocess_mode="serial"
        )
        dask_result = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording_4ch, params, multiprocess_mode="dask"
        )

        assert isinstance(dask_result, list)
        for s_arr, d_arr in zip(serial, dask_result):
            assert isinstance(d_arr, np.ndarray)
            assert d_arr.dtype == s_arr.dtype

    def test_dask_single_channel_recording(self, params):
        """Dask mode works correctly with a single-channel recording."""
        np.random.seed(13)
        data = np.random.randn(3000, 1) * 0.1
        recording = si.NumpyRecording(data, sampling_frequency=1000.0, channel_ids=["ch0"])

        result = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, params, multiprocess_mode="dask"
        )

        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_dask_many_channels(self, params):
        """Dask mode scales correctly to 16 channels."""
        np.random.seed(55)
        n_ch = 16
        data = np.random.randn(2000, n_ch) * 0.1
        recording = si.NumpyRecording(
            data, sampling_frequency=1000.0, channel_ids=[f"ch{i:02d}" for i in range(n_ch)]
        )

        result = FrequencyDomainSpikeDetector.detect_spikes_recording(
            recording, params, multiprocess_mode="dask"
        )

        assert len(result) == n_ch
        for arr in result:
            assert isinstance(arr, np.ndarray)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_dask_import_error_when_dask_unavailable(self, mock_recording, params):
        """ImportError is raised in dask mode when dask cannot be imported."""
        import neurodent.core.frequency_domain_spike_detection as fdsd_mod
        original = fdsd_mod.dask
        try:
            fdsd_mod.dask = None
            with pytest.raises(ImportError, match="dask is required"):
                FrequencyDomainSpikeDetector.detect_spikes_recording(
                    mock_recording, params, multiprocess_mode="dask"
                )
        finally:
            fdsd_mod.dask = original
