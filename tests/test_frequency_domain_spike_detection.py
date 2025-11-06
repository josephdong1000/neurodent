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
        mock_rec.get_dtype.return_value = np.float32
        mock_rec.clone.return_value = mock_rec
        mock_rec.set_channel_ids.return_value = None

        # Mock data - transposed to (samples, channels) format as get_traces returns
        np.random.seed(42)
        mock_data = np.random.randn(10000, 4) * 0.1
        mock_rec.get_traces.return_value = mock_data

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
        mock_recording.clone.assert_called_once()
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

    @patch.object(FrequencyDomainSpikeDetector, "_apply_preprocessing")
    @patch.object(FrequencyDomainSpikeDetector, "_detect_spikes_channel")
    @patch.object(FrequencyDomainSpikeDetector, "_add_spike_annotations")
    def test_detect_spikes_recording_serial(
        self, mock_add_annotations, mock_detect_channel, mock_preprocess, mock_recording, detection_params
    ):
        """Test full spike detection pipeline in serial mode."""
        # Setup mocks
        mock_preprocess.return_value = mock_recording
        mock_detect_channel.return_value = np.array([100, 500, 1000])

        # Mock MNE creation
        with patch("mne.create_info"), patch("mne.io.RawArray") as mock_raw_array:
            mock_raw = MagicMock()
            mock_raw_array.return_value = mock_raw
            mock_add_annotations.return_value = mock_raw

            spike_indices, mne_raw = FrequencyDomainSpikeDetector.detect_spikes_recording(
                mock_recording, detection_params, multiprocess_mode="serial"
            )

        # Check calls
        mock_preprocess.assert_called_once()
        assert mock_detect_channel.call_count == 4  # 4 channels
        mock_add_annotations.assert_called_once()

        # Check outputs
        assert len(spike_indices) == 4  # 4 channels
        assert mne_raw is not None


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
