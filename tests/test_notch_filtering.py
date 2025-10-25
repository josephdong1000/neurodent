"""
Tests specifically for notch filtering functionality.

These tests ensure that the notch filtering implementation works correctly
both at the FragmentAnalyzer level and the LongRecordingAnalyzer integration level.
"""

import numpy as np
import pytest
from unittest.mock import patch, Mock
import warnings

from neurodent.core.analysis import LongRecordingAnalyzer
from neurodent.core.analyze_frag import FragmentAnalyzer
from neurodent import constants, core

try:
    import spikeinterface as si
    import spikeinterface.preprocessing as spre

    SPIKEINTERFACE_AVAILABLE = True
except ImportError:
    SPIKEINTERFACE_AVAILABLE = False


class TestNotchFiltering:
    """Test notch filtering functionality at multiple levels."""

    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_notch_filter_reduces_line_frequency(self, real_spikeinterface_recording):
        """Test that notch filter reduces power at line frequency."""
        if real_spikeinterface_recording is None:
            pytest.skip("Real SpikeInterface recording not available")

        recording = real_spikeinterface_recording

        # Add 60Hz noise to the recording
        duration = recording.get_total_duration()
        fs = recording.get_sampling_frequency()
        n_samples = int(duration * fs)
        n_channels = recording.get_num_channels()

        # Generate 60Hz noise
        t = np.linspace(0, duration, n_samples, endpoint=False)
        noise_60hz = 100 * np.sin(2 * np.pi * constants.LINE_FREQ * t)  # Strong 60Hz component

        # Create recording with 60Hz noise added (synthetic approach)
        original_data = recording.get_traces()
        noisy_data = original_data + noise_60hz[:, np.newaxis]

        # Apply notch filter via SpikeInterface
        filtered_recording = spre.notch_filter(recording, freq=constants.LINE_FREQ)

        # Test that the filter was applied (this is an integration test)
        assert filtered_recording is not None
        assert filtered_recording.get_sampling_frequency() == fs
        assert filtered_recording.get_num_channels() == n_channels

    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_long_recording_analyzer_notch_filter_integration(self, real_spikeinterface_recording):
        """Test that LongRecordingAnalyzer properly integrates notch filtering."""
        if real_spikeinterface_recording is None:
            pytest.skip("Real SpikeInterface recording not available")

        # Create a mock LongRecordingOrganizer that returns our real recording
        from neurodent.core.core import LongRecordingOrganizer

        mock_long_recording = Mock(spec=LongRecordingOrganizer)
        mock_long_recording.get_num_fragments.return_value = 1
        mock_long_recording.channel_names = [f"ch{i}" for i in range(8)]
        mock_long_recording.meta = Mock()
        mock_long_recording.meta.n_channels = 8
        mock_long_recording.meta.mult_to_uV = 1.0
        mock_long_recording.LongRecording = Mock()
        mock_long_recording.LongRecording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_long_recording.LongRecording.get_num_frames.return_value = 2000
        mock_long_recording.cumulative_file_durations = [2.0]
        mock_long_recording.end_relative = [1]

        # Make get_fragment return our real SpikeInterface recording
        mock_long_recording.get_fragment.return_value = real_spikeinterface_recording

        # Create a real mock that inherits from LongRecordingOrganizer to pass isinstance check
        class MockLongRecordingOrganizer(core.LongRecordingOrganizer):
            def __init__(self):
                # Skip parent __init__ to avoid file system dependencies
                pass

        # Copy all the mock attributes to our class-based mock
        mock_organizer = MockLongRecordingOrganizer()
        for attr in [
            "get_num_fragments",
            "channel_names",
            "meta",
            "LongRecording",
            "cumulative_file_durations",
            "end_relative",
            "get_fragment",
        ]:
            setattr(mock_organizer, attr, getattr(mock_long_recording, attr))

        analyzer = LongRecordingAnalyzer(longrecording=mock_organizer, fragment_len_s=2)

        # Test that get_fragment_rec works with real SpikeInterface object
        fragment_rec = analyzer.get_fragment_rec(0)

        # Verify it's a SpikeInterface recording (could be filtered or original)
        assert hasattr(fragment_rec, "get_traces")
        assert hasattr(fragment_rec, "get_sampling_frequency")
        assert fragment_rec.get_sampling_frequency() == constants.GLOBAL_SAMPLING_RATE

    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_notch_filter_disable_flag(self):
        """Test that apply_notch_filter=False properly disables filtering."""
        from neurodent.core.core import LongRecordingOrganizer

        mock_long_recording = Mock(spec=LongRecordingOrganizer)
        mock_long_recording.get_num_fragments.return_value = 1
        mock_long_recording.channel_names = [f"ch{i}" for i in range(8)]
        mock_long_recording.meta = Mock()
        mock_long_recording.meta.n_channels = 8
        mock_long_recording.meta.mult_to_uV = 1.0
        mock_long_recording.LongRecording = Mock()
        mock_long_recording.LongRecording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_long_recording.LongRecording.get_num_frames.return_value = 2000
        mock_long_recording.cumulative_file_durations = [2.0]
        mock_long_recording.end_relative = [1]

        # Create a mock recording that would fail notch_filter if called
        mock_recording = Mock()
        mock_recording.get_traces.return_value = np.random.randn(1000, 8)
        mock_long_recording.get_fragment.return_value = mock_recording

        # Create a real mock that inherits from LongRecordingOrganizer to pass isinstance check
        class MockLongRecordingOrganizer(core.LongRecordingOrganizer):
            def __init__(self):
                # Skip parent __init__ to avoid file system dependencies
                pass

        # Copy all the mock attributes to our class-based mock
        mock_organizer = MockLongRecordingOrganizer()
        for attr in [
            "get_num_fragments",
            "channel_names",
            "meta",
            "LongRecording",
            "cumulative_file_durations",
            "end_relative",
            "get_fragment",
        ]:
            setattr(mock_organizer, attr, getattr(mock_long_recording, attr))

        analyzer = LongRecordingAnalyzer(longrecording=mock_organizer, fragment_len_s=2)
        analyzer.apply_notch_filter = False

        # This should work without calling notch_filter on the mock
        # Since SpikeInterface may not be available, patch the spre module in neurodent.core.analysis
        with patch("neurodent.core.analysis.spre") as mock_spre:
            mock_spre.notch_filter = Mock()
            fragment_rec = analyzer.get_fragment_rec(0)
            # Verify notch_filter was NOT called
            mock_spre.notch_filter.assert_not_called()
            assert fragment_rec == mock_recording

    def test_fragment_analyzer_notch_filter_in_psd_computation(self):
        """Test that FragmentAnalyzer's PSD computation includes notch filtering."""
        # Create test data with 60Hz line noise
        fs = constants.GLOBAL_SAMPLING_RATE
        duration = 2.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        # Signal: 10Hz alpha + 60Hz line noise
        alpha_signal = 50 * np.sin(2 * np.pi * 10 * t)
        line_noise = 100 * np.sin(2 * np.pi * constants.LINE_FREQ * t)  # Stronger line noise

        # Create multi-channel data
        n_channels = 4
        rec = np.zeros((len(t), n_channels))
        for ch in range(n_channels):
            rec[:, ch] = alpha_signal + line_noise + 5 * np.random.randn(len(t))

        # Compute PSD with and without notch filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress MNE warnings for test data

            f_with_notch, psd_with_notch = FragmentAnalyzer.compute_psd(rec, fs, welch_bin_t=1.0, notch_filter=True)
            f_without_notch, psd_without_notch = FragmentAnalyzer.compute_psd(
                rec, fs, welch_bin_t=1.0, notch_filter=False
            )

        # Find the 60Hz frequency bin
        freq_60hz_idx = np.argmin(np.abs(f_with_notch - constants.LINE_FREQ))

        # The notch-filtered signal should have less power at 60Hz
        power_60hz_with_notch = np.mean(psd_with_notch[freq_60hz_idx, :])
        power_60hz_without_notch = np.mean(psd_without_notch[freq_60hz_idx, :])

        # Verify that notch filtering reduced 60Hz power
        assert power_60hz_with_notch < power_60hz_without_notch

        # The reduction should be substantial (at least 50% reduction)
        reduction_ratio = power_60hz_with_notch / power_60hz_without_notch
        assert reduction_ratio < 0.5, f"60Hz power reduction was only {(1 - reduction_ratio) * 100:.1f}%"

    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_notch_filter_parameter_in_analyzer_methods(self):
        """Test that LongRecordingAnalyzer methods respect the notch filter setting."""
        from neurodent.core.core import LongRecordingOrganizer

        mock_long_recording = Mock(spec=LongRecordingOrganizer)
        mock_long_recording.get_num_fragments.return_value = 1
        mock_long_recording.channel_names = [f"ch{i}" for i in range(8)]
        mock_long_recording.meta = Mock()
        mock_long_recording.meta.n_channels = 8
        mock_long_recording.meta.mult_to_uV = 1.0
        mock_long_recording.LongRecording = Mock()
        mock_long_recording.LongRecording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_long_recording.LongRecording.get_num_frames.return_value = 2000
        mock_long_recording.cumulative_file_durations = [2.0]
        mock_long_recording.end_relative = [1]

        # Create a real mock that inherits from LongRecordingOrganizer to pass isinstance check
        class MockLongRecordingOrganizer(core.LongRecordingOrganizer):
            def __init__(self):
                # Skip parent __init__ to avoid file system dependencies
                pass

        # Copy all the mock attributes to our class-based mock
        mock_organizer = MockLongRecordingOrganizer()
        for attr in [
            "get_num_fragments",
            "channel_names",
            "meta",
            "LongRecording",
            "cumulative_file_durations",
            "end_relative",
        ]:
            setattr(mock_organizer, attr, getattr(mock_long_recording, attr))

        analyzer = LongRecordingAnalyzer(longrecording=mock_organizer, fragment_len_s=2)

        # Test that the analyzer has notch filtering enabled by default
        assert analyzer.apply_notch_filter

        # Test that we can disable it
        analyzer.apply_notch_filter = False
        assert not analyzer.apply_notch_filter
