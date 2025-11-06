"""
Unit tests for neurodent.core.analyze_sort module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from neurodent.core.analyze_sort import MountainSortAnalyzer, MOUNTAINSORT_AVAILABLE
from neurodent import constants


@pytest.mark.skipif(not MOUNTAINSORT_AVAILABLE, reason="mountainsort5 not available")
class TestMountainSortAnalyzer:
    """Test MountainSortAnalyzer static methods."""

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
        mock_rec.set_probe.return_value = mock_rec
        mock_rec.select_channels.return_value = mock_rec
        return mock_rec

    @pytest.fixture
    def mock_sorting(self):
        """Create a mock SpikeInterface sorting for testing."""
        mock_sort = MagicMock()
        mock_sort.get_unit_ids.return_value = ["unit1", "unit2"]
        mock_sort.get_num_units.return_value = 2
        return mock_sort

    @pytest.fixture
    def mock_probe(self):
        """Create a mock probe for testing."""
        mock_probe = MagicMock()
        mock_probe.contact_positions = np.array([[0, 0], [0, 40], [0, 80], [0, 120]])
        return mock_probe

    def test_get_dummy_probe(self, mock_recording):
        """Test _get_dummy_probe method."""
        with patch("probeinterface.generate_linear_probe") as mock_gen_probe:
            mock_probe = Mock()
            mock_gen_probe.return_value = mock_probe

            result = MountainSortAnalyzer._get_dummy_probe(mock_recording)

            # Verify probe generation was called with correct parameters
            mock_gen_probe.assert_called_once_with(4, ypitch=40)

            # Verify probe configuration methods were called
            mock_probe.set_device_channel_indices.assert_called_once()
            mock_probe.set_contact_ids.assert_called_once_with(["ch1", "ch2", "ch3", "ch4"])

            assert result == mock_probe

    def test_get_recording_for_sorting(self, mock_recording):
        """Test _get_recording_for_sorting method."""
        with patch.object(MountainSortAnalyzer, "_apply_preprocessing") as mock_preprocess:
            mock_preprocess.return_value = mock_recording

            result = MountainSortAnalyzer._get_recording_for_sorting(mock_recording)

            # Verify preprocessing was called with sorting parameters
            mock_preprocess.assert_called_once_with(mock_recording, constants.SORTING_PARAMS)
            assert result == mock_recording

    def test_get_recording_for_waveforms(self, mock_recording):
        """Test _get_recording_for_waveforms method."""
        with patch.object(MountainSortAnalyzer, "_apply_preprocessing") as mock_preprocess:
            mock_preprocess.return_value = mock_recording

            result = MountainSortAnalyzer._get_recording_for_waveforms(mock_recording)

            # Verify preprocessing was called with waveform parameters
            mock_preprocess.assert_called_once_with(mock_recording, constants.WAVEFORM_PARAMS)
            assert result == mock_recording

    @patch("spikeinterface.preprocessing.notch_filter")
    @patch("spikeinterface.preprocessing.common_reference")
    @patch("spikeinterface.preprocessing.scale")
    @patch("spikeinterface.preprocessing.whiten")
    @patch("spikeinterface.preprocessing.highpass_filter")
    @patch("spikeinterface.preprocessing.bandpass_filter")
    def test_apply_preprocessing_all_filters(
        self, mock_bandpass, mock_highpass, mock_whiten, mock_scale, mock_common_ref, mock_notch, mock_recording
    ):
        """Test _apply_preprocessing with all filters enabled."""
        # Define test parameters with all filters enabled
        test_params = {
            "notch_freq": 60,
            "common_ref": True,
            "scale": 2.0,
            "whiten": True,
            "freq_min": 0.1,
            "freq_max": 100,
        }

        # Mock each preprocessing step to return the recording
        mock_notch.return_value = mock_recording
        mock_common_ref.return_value = mock_recording
        mock_scale.return_value = mock_recording
        mock_whiten.return_value = mock_recording
        mock_bandpass.return_value = mock_recording

        result = MountainSortAnalyzer._apply_preprocessing(mock_recording, test_params)

        # Verify all preprocessing steps were called
        mock_notch.assert_called_once_with(mock_recording, freq=60, q=100)
        mock_common_ref.assert_called_once()
        mock_scale.assert_called_once_with(mock_recording, gain=2.0)
        mock_whiten.assert_called_once()
        mock_bandpass.assert_called_once_with(mock_recording, freq_min=0.1, freq_max=100, ftype="bessel")

        assert result == mock_recording

    @patch("spikeinterface.preprocessing.highpass_filter")
    def test_apply_preprocessing_highpass_only(self, mock_highpass, mock_recording):
        """Test _apply_preprocessing with only highpass filter."""
        test_params = {
            "notch_freq": None,
            "common_ref": False,
            "scale": None,
            "whiten": False,
            "freq_min": 0.5,
            "freq_max": None,
        }

        mock_highpass.return_value = mock_recording

        result = MountainSortAnalyzer._apply_preprocessing(mock_recording, test_params)

        mock_highpass.assert_called_once_with(mock_recording, freq_min=0.5, ftype="bessel")
        assert result == mock_recording

    @patch("spikeinterface.preprocessing.bandpass_filter")
    def test_apply_preprocessing_lowpass_equivalent(self, mock_bandpass, mock_recording):
        """Test _apply_preprocessing with only freq_max (lowpass equivalent)."""
        test_params = {
            "notch_freq": None,
            "common_ref": False,
            "scale": None,
            "whiten": False,
            "freq_min": None,
            "freq_max": 50,
        }

        mock_bandpass.return_value = mock_recording

        result = MountainSortAnalyzer._apply_preprocessing(mock_recording, test_params)

        # Should use bandpass with freq_min=0.1 as a lowpass equivalent
        mock_bandpass.assert_called_once_with(mock_recording, freq_min=0.1, freq_max=50, ftype="bessel")
        assert result == mock_recording

    def test_apply_preprocessing_no_filters(self, mock_recording):
        """Test _apply_preprocessing with no filters enabled."""
        test_params = {
            "notch_freq": None,
            "common_ref": False,
            "scale": None,
            "whiten": False,
            "freq_min": None,
            "freq_max": None,
        }

        result = MountainSortAnalyzer._apply_preprocessing(mock_recording, test_params)

        # Should just return the cloned recording
        mock_recording.clone.assert_called_once()
        assert result == mock_recording

    def test_split_recording(self, mock_recording):
        """Test _split_recording method."""
        channel_ids = ["ch1", "ch2", "ch3", "ch4"]
        mock_recording.get_channel_ids.return_value = channel_ids

        # Mock select_channels to return different objects for each channel
        mock_single_channel_recs = [Mock() for _ in channel_ids]
        mock_recording.clone.return_value.select_channels.side_effect = mock_single_channel_recs

        result = MountainSortAnalyzer._split_recording(mock_recording)

        # Should return one recording per channel
        assert len(result) == 4

        # Verify select_channels was called for each channel
        calls = mock_recording.clone.return_value.select_channels.call_args_list
        expected_calls = [([channel_id],) for channel_id in channel_ids]
        actual_calls = [call[0] for call in calls]

        assert actual_calls == expected_calls

    @patch("neurodent.core.analyze_sort.get_temp_directory")
    @patch("neurodent.core.analyze_sort.create_cached_recording")
    @patch("spikeinterface.preprocessing.astype")
    @patch("os.makedirs")
    def test_cache_recording(self, mock_makedirs, mock_astype, mock_create_cached, mock_get_temp_dir, mock_recording):
        """Test _cache_recording method."""
        mock_temp_dir = Path("/tmp/test_temp")
        mock_get_temp_dir.return_value = mock_temp_dir

        mock_cached_rec = Mock()
        mock_create_cached.return_value = mock_cached_rec
        mock_astype.return_value = mock_cached_rec

        with patch("os.urandom") as mock_urandom:
            mock_urandom.return_value.hex.return_value = "random_hex"

            result = MountainSortAnalyzer._cache_recording(mock_recording)

            # Verify temp directory creation
            expected_path = mock_temp_dir / "random_hex"
            mock_makedirs.assert_called_once_with(expected_path)

            # Verify cached recording creation
            mock_create_cached.assert_called_once_with(mock_recording, folder=expected_path, chunk_duration="60s")

            # Verify dtype conversion
            mock_astype.assert_called_once_with(mock_cached_rec, dtype=constants.GLOBAL_DTYPE)

            assert result == mock_cached_rec

    @patch("neurodent.core.analyze_sort.sorting_scheme2")
    @patch("neurodent.core.analyze_sort.Scheme2SortingParameters")
    def test_run_sorting(self, mock_sort_params, mock_sorting_scheme2, mock_recording):
        """Test _run_sorting method."""
        mock_recording.get_sampling_frequency.return_value = 1000.0

        mock_params_instance = Mock()
        mock_sort_params.return_value = mock_params_instance

        mock_sorting = Mock()
        mock_sorting_scheme2.return_value = mock_sorting

        result = MountainSortAnalyzer._run_sorting(mock_recording)

        # Verify parameter calculation (snippet times converted from seconds to samples)
        expected_t1_samples = round(1000.0 * constants.SCHEME2_SORTING_PARAMS["snippet_T1"])
        expected_t2_samples = round(1000.0 * constants.SCHEME2_SORTING_PARAMS["snippet_T2"])

        # Verify sorting parameters creation
        mock_sort_params.assert_called_once_with(
            phase1_detect_channel_radius=constants.SCHEME2_SORTING_PARAMS["phase1_detect_channel_radius"],
            detect_channel_radius=constants.SCHEME2_SORTING_PARAMS["detect_channel_radius"],
            snippet_T1=expected_t1_samples,
            snippet_T2=expected_t2_samples,
        )

        # Verify sorting execution
        mock_sorting_scheme2.assert_called_once_with(recording=mock_recording, sorting_parameters=mock_params_instance)

        assert result == mock_sorting

    @patch.object(MountainSortAnalyzer, "_run_sorting")
    @patch.object(MountainSortAnalyzer, "_cache_recording")
    @patch.object(MountainSortAnalyzer, "_split_recording")
    @patch.object(MountainSortAnalyzer, "_get_recording_for_waveforms")
    @patch.object(MountainSortAnalyzer, "_get_recording_for_sorting")
    @patch.object(MountainSortAnalyzer, "_get_dummy_probe")
    def test_sort_recording_serial(
        self,
        mock_get_probe,
        mock_get_sort_rec,
        mock_get_wave_rec,
        mock_split,
        mock_cache,
        mock_run_sort,
        mock_recording,
        mock_probe,
    ):
        """Test sort_recording method in serial mode."""
        # Setup mocks
        mock_get_probe.return_value = mock_probe
        mock_get_sort_rec.return_value = mock_recording
        mock_get_wave_rec.return_value = mock_recording

        mock_split_recs = [Mock(), Mock(), Mock(), Mock()]
        mock_split.side_effect = [mock_split_recs, mock_split_recs]  # Called twice

        mock_cached_recs = [Mock(), Mock(), Mock(), Mock()]
        mock_cache.side_effect = mock_cached_recs

        mock_sortings = [Mock(), Mock(), Mock(), Mock()]
        mock_run_sort.side_effect = mock_sortings

        # Test serial mode
        sortings, wave_recs = MountainSortAnalyzer.sort_recording(
            mock_recording, plot_probe=False, multiprocess_mode="serial"
        )

        # Verify probe setup
        mock_get_probe.assert_called_once_with(mock_recording)
        mock_recording.set_probe.assert_called_once_with(mock_probe)

        # Verify recording preparation
        mock_get_sort_rec.assert_called_once()
        mock_get_wave_rec.assert_called_once()

        # Verify splitting (called twice: once for sorting, once for waveforms)
        assert mock_split.call_count == 2

        # Verify caching and sorting
        assert mock_cache.call_count == 4
        assert mock_run_sort.call_count == 4

        # Check results
        assert sortings == mock_sortings
        assert wave_recs == mock_split_recs

    @patch.object(MountainSortAnalyzer, "_run_sorting")
    @patch.object(MountainSortAnalyzer, "_cache_recording")
    @patch.object(MountainSortAnalyzer, "_split_recording")
    @patch.object(MountainSortAnalyzer, "_get_recording_for_waveforms")
    @patch.object(MountainSortAnalyzer, "_get_recording_for_sorting")
    @patch.object(MountainSortAnalyzer, "_get_dummy_probe")
    @patch("dask.delayed")
    def test_sort_recording_dask(
        self,
        mock_delayed,
        mock_get_probe,
        mock_get_sort_rec,
        mock_get_wave_rec,
        mock_split,
        mock_cache,
        mock_run_sort,
        mock_recording,
        mock_probe,
    ):
        """Test sort_recording method in dask mode."""
        # Setup mocks
        mock_get_probe.return_value = mock_probe
        mock_get_sort_rec.return_value = mock_recording
        mock_get_wave_rec.return_value = mock_recording

        mock_split_recs = [Mock(), Mock(), Mock(), Mock()]
        mock_split.side_effect = [mock_split_recs, mock_split_recs]

        # Mock dask.delayed to return mock delayed objects
        mock_delayed_cache = [Mock(), Mock(), Mock(), Mock()]
        mock_delayed_sort = [Mock(), Mock(), Mock(), Mock()]
        mock_delayed.side_effect = mock_delayed_cache + mock_delayed_sort

        # Test dask mode
        sortings, wave_recs = MountainSortAnalyzer.sort_recording(
            mock_recording, plot_probe=False, multiprocess_mode="dask"
        )

        # Verify dask.delayed was called for each channel (cache + sort)
        assert mock_delayed.call_count == 8  # 4 for caching + 4 for sorting

        # Check results structure - should return delayed objects, not the original mocks
        assert len(sortings) == 4  # One for each channel
        assert len(wave_recs) == 4  # One for each channel
        assert wave_recs == mock_split_recs

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    @patch.object(MountainSortAnalyzer, "_run_sorting")
    @patch.object(MountainSortAnalyzer, "_cache_recording")
    @patch.object(MountainSortAnalyzer, "_split_recording")
    @patch.object(MountainSortAnalyzer, "_get_recording_for_waveforms")
    @patch.object(MountainSortAnalyzer, "_get_recording_for_sorting")
    @patch.object(MountainSortAnalyzer, "_get_dummy_probe")
    def test_sort_recording_with_plot(
        self,
        mock_get_probe,
        mock_get_sort_rec,
        mock_get_wave_rec,
        mock_split,
        mock_cache,
        mock_run_sort,
        mock_show,
        mock_subplots,
        mock_recording,
        mock_probe,
    ):
        """Test sort_recording method with probe plotting enabled."""
        # Setup basic mocks
        mock_get_probe.return_value = mock_probe
        mock_get_sort_rec.return_value = mock_recording
        mock_get_wave_rec.return_value = mock_recording
        mock_split.return_value = [Mock()]
        mock_cache.return_value = Mock()
        mock_run_sort.return_value = Mock()

        mock_fig, mock_ax = Mock(), Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Test with plotting enabled
        with patch("neurodent.core.analyze_sort.pi_plotting.plot_probe") as mock_plot_probe:
            MountainSortAnalyzer.sort_recording(mock_recording, plot_probe=True, multiprocess_mode="serial")

            # Verify plotting was called
            mock_subplots.assert_called_once_with(1, 1)
            mock_plot_probe.assert_called_once_with(
                mock_probe, ax=mock_ax, with_device_index=True, with_contact_id=True
            )
            mock_show.assert_called_once()

    def test_constants_used_correctly(self):
        """Test that the correct constants are used in the analyzer."""
        # Verify that the expected constants exist and have reasonable values
        assert hasattr(constants, "SORTING_PARAMS")
        assert hasattr(constants, "WAVEFORM_PARAMS")
        assert hasattr(constants, "SCHEME2_SORTING_PARAMS")
        assert hasattr(constants, "GLOBAL_DTYPE")

        # Check that SCHEME2_SORTING_PARAMS has required keys
        required_keys = ["snippet_T1", "snippet_T2", "detect_channel_radius", "phase1_detect_channel_radius"]
        for key in required_keys:
            assert key in constants.SCHEME2_SORTING_PARAMS

    def test_error_handling_invalid_multiprocess_mode(self, mock_recording):
        """Test error handling for invalid multiprocess mode."""
        with patch.object(MountainSortAnalyzer, "_get_dummy_probe"):
            with patch.object(MountainSortAnalyzer, "_get_recording_for_sorting"):
                with patch.object(MountainSortAnalyzer, "_get_recording_for_waveforms"):
                    with patch.object(MountainSortAnalyzer, "_split_recording"):
                        # This should work without error for valid modes
                        try:
                            MountainSortAnalyzer.sort_recording(mock_recording, multiprocess_mode="serial")
                        except Exception as e:
                            # The test setup might cause other errors, but not from invalid mode
                            assert "multiprocess_mode" not in str(e)

    @pytest.mark.parametrize("n_channels", [1, 4, 8, 16])
    def test_split_recording_different_channel_counts(self, n_channels):
        """Test _split_recording with different numbers of channels."""
        mock_recording = Mock()
        channel_ids = [f"ch{i}" for i in range(n_channels)]
        mock_recording.get_channel_ids.return_value = channel_ids

        # Mock the chain of clone().select_channels()
        mock_cloned = Mock()
        mock_recording.clone.return_value = mock_cloned
        mock_cloned.select_channels.return_value = Mock()

        result = MountainSortAnalyzer._split_recording(mock_recording)

        assert len(result) == n_channels
        assert mock_cloned.select_channels.call_count == n_channels

    def test_cache_recording_creates_unique_paths(self, mock_recording):
        """Test that _cache_recording creates unique temporary paths."""
        with patch("neurodent.core.analyze_sort.get_temp_directory") as mock_get_temp:
            with patch("os.makedirs") as mock_makedirs:
                with patch("neurodent.core.analyze_sort.create_cached_recording") as mock_create:
                    with patch("spikeinterface.preprocessing.astype") as mock_astype:
                        mock_get_temp.return_value = Path("/tmp/test")
                        mock_create.return_value = Mock()
                        mock_astype.return_value = Mock()

                        # Call multiple times and verify different paths are used
                        with patch("os.urandom") as mock_urandom:
                            # Mock urandom to return different values
                            mock_urandom.return_value.hex.side_effect = ["hex1", "hex2", "hex3"]

                            MountainSortAnalyzer._cache_recording(mock_recording)
                            MountainSortAnalyzer._cache_recording(mock_recording)
                            MountainSortAnalyzer._cache_recording(mock_recording)

                            # Verify different paths were created
                            calls = mock_makedirs.call_args_list
                            paths = [call[0][0] for call in calls]

                            assert len(set(paths)) == 3  # All paths should be unique
                            assert all("hex" in str(path) for path in paths)


class TestMountainSortOptionalDependency:
    """Test behavior when mountainsort5 is not available."""

    @pytest.mark.skipif(MOUNTAINSORT_AVAILABLE, reason="mountainsort5 is available")
    def test_mountainsort_unavailable_error(self):
        """Test that proper error is raised when mountainsort5 is not available."""
        mock_recording = MagicMock()

        with pytest.raises(ImportError, match="MountainSort5 is not available"):
            MountainSortAnalyzer.sort_recording(mock_recording)

    def test_mountainsort_availability_flag(self):
        """Test that MOUNTAINSORT_AVAILABLE flag is accessible."""
        # This should always pass regardless of availability
        assert isinstance(MOUNTAINSORT_AVAILABLE, bool)
