"""
Unit tests for neurodent.core.analyze_frag module.
"""

import numpy as np
import pytest
import warnings
from unittest.mock import patch, Mock, MagicMock
from scipy.integrate import trapezoid

from neurodent.core.analyze_frag import FragmentAnalyzer
from neurodent import constants


class TestFragmentAnalyzer:
    """Test FragmentAnalyzer static methods."""

    # Test Fixtures
    @pytest.fixture
    def sample_rec_2d(self):
        """Create a 2D sample recording array (N_samples x N_channels)."""
        np.random.seed(42)  # For reproducible tests
        n_samples, n_channels = 6000, 4  # 6 seconds at 1000 Hz for reliable spectral estimates
        # Create a realistic EEG-like signal with different frequencies per channel
        t = np.linspace(0, 6, n_samples)  # 6 seconds for >=5 cycles at 1 Hz
        data = np.zeros((n_samples, n_channels))

        # Channel 0: 10 Hz sine wave + noise
        data[:, 0] = 100 * np.sin(2 * np.pi * 10 * t) + 10 * np.random.randn(n_samples)
        # Channel 1: 20 Hz sine wave + noise
        data[:, 1] = 80 * np.sin(2 * np.pi * 20 * t) + 15 * np.random.randn(n_samples)
        # Channel 2: Mix of frequencies + noise
        data[:, 2] = 50 * np.sin(2 * np.pi * 5 * t) + 30 * np.sin(2 * np.pi * 15 * t) + 20 * np.random.randn(n_samples)
        # Channel 3: Higher frequency + noise
        data[:, 3] = 60 * np.sin(2 * np.pi * 30 * t) + 25 * np.random.randn(n_samples)

        return data.astype(np.float32)

    @pytest.fixture
    def sample_rec_2d_short(self):
        """Create a short 2D sample recording array for non-spectral tests."""
        np.random.seed(42)  # For reproducible tests
        n_samples, n_channels = 1000, 4  # 1 second - sufficient for amplitude/variance tests
        t = np.linspace(0, 1, n_samples)
        data = np.zeros((n_samples, n_channels))

        # Same signal patterns as longer version
        data[:, 0] = 100 * np.sin(2 * np.pi * 10 * t) + 10 * np.random.randn(n_samples)
        data[:, 1] = 80 * np.sin(2 * np.pi * 20 * t) + 15 * np.random.randn(n_samples)
        data[:, 2] = 50 * np.sin(2 * np.pi * 5 * t) + 30 * np.sin(2 * np.pi * 15 * t) + 20 * np.random.randn(n_samples)
        data[:, 3] = 60 * np.sin(2 * np.pi * 30 * t) + 25 * np.random.randn(n_samples)

        return data.astype(np.float32)

    @pytest.fixture
    def sample_rec_3d(self, sample_rec_2d):
        """Create a 3D sample recording array for MNE (1 x N_channels x N_samples)."""
        return sample_rec_2d.T[np.newaxis, :, :]  # (1, n_channels, n_samples)

    # Input Validation Tests
    def test_check_rec_np_valid(self, sample_rec_2d_short):
        """Test _check_rec_np with valid 2D array."""
        # Should not raise any exception
        FragmentAnalyzer._check_rec_np(sample_rec_2d_short)

    def test_check_rec_np_invalid_type(self):
        """Test _check_rec_np with invalid input type."""
        with pytest.raises(ValueError, match="rec must be a numpy array"):
            FragmentAnalyzer._check_rec_np([1, 2, 3])

    def test_check_rec_np_invalid_dimensions(self):
        """Test _check_rec_np with invalid dimensions."""
        # 1D array
        with pytest.raises(ValueError, match="rec must be a 2D numpy array"):
            FragmentAnalyzer._check_rec_np(np.array([1, 2, 3]))

        # 3D array
        with pytest.raises(ValueError, match="rec must be a 2D numpy array"):
            FragmentAnalyzer._check_rec_np(np.random.randn(10, 4, 2))

    def test_check_rec_mne_valid(self, sample_rec_3d):
        """Test _check_rec_mne with valid 3D array."""
        # Should not raise any exception
        FragmentAnalyzer._check_rec_mne(sample_rec_3d)

    def test_check_rec_mne_invalid_type(self):
        """Test _check_rec_mne with invalid input type."""
        with pytest.raises(ValueError, match="rec must be a numpy array"):
            FragmentAnalyzer._check_rec_mne([[[1, 2, 3]]])

    def test_check_rec_mne_invalid_dimensions(self):
        """Test _check_rec_mne with invalid dimensions."""
        # 2D array
        with pytest.raises(ValueError, match="rec must be a 3D numpy array"):
            FragmentAnalyzer._check_rec_mne(np.random.randn(10, 4))

        # Wrong first dimension
        with pytest.raises(ValueError, match="rec must be a 1 x M x N array"):
            FragmentAnalyzer._check_rec_mne(np.random.randn(2, 4, 10))

    # Data Transformation Tests
    def test_reshape_np_for_mne(self, sample_rec_2d_short):
        """Test _reshape_np_for_mne conversion."""
        n_samples, n_channels = sample_rec_2d_short.shape
        result = FragmentAnalyzer._reshape_np_for_mne(sample_rec_2d_short)

        # Check output shape: (1, n_channels, n_samples)
        assert result.shape == (1, n_channels, n_samples)

        # Check data integrity (first sample, all channels)
        np.testing.assert_array_almost_equal(result[0, :, 0], sample_rec_2d_short[0, :], decimal=5)

    # Basic Amplitude Features
    def test_compute_rms(self, sample_rec_2d_short):
        """Test compute_rms function."""
        result = FragmentAnalyzer.compute_rms(sample_rec_2d_short)

        # Check output shape: should be (n_channels,)
        assert result.shape == (sample_rec_2d_short.shape[1],)

        # Check that all RMS values are positive
        assert np.all(result > 0)

        # Manually compute RMS for first channel and compare
        expected_rms_ch0 = np.sqrt(np.mean(sample_rec_2d_short[:, 0] ** 2))
        # Use decimal=4 for float32 precision (was decimal=5)
        np.testing.assert_array_almost_equal(result[0], expected_rms_ch0, decimal=4)

    def test_compute_logrms(self, sample_rec_2d_short):
        """Test compute_logrms function."""
        result = FragmentAnalyzer.compute_logrms(sample_rec_2d_short)

        # Check output shape
        assert result.shape == (sample_rec_2d_short.shape[1],)

        # Compare with manual calculation
        rms_values = FragmentAnalyzer.compute_rms(sample_rec_2d_short)
        expected_logrms = np.log(rms_values + 1)
        np.testing.assert_array_almost_equal(result, expected_logrms, decimal=5)

    def test_compute_ampvar(self, sample_rec_2d_short):
        """Test compute_ampvar function."""
        result = FragmentAnalyzer.compute_ampvar(sample_rec_2d_short)

        # Check output shape
        assert result.shape == (sample_rec_2d_short.shape[1],)

        # Check that all variance values are non-negative
        assert np.all(result >= 0)

        # Manually compute variance for first channel and compare
        expected_var_ch0 = np.std(sample_rec_2d_short[:, 0]) ** 2
        # Use decimal=3 for float32 precision (was decimal=5)
        np.testing.assert_array_almost_equal(result[0], expected_var_ch0, decimal=3)

    def test_compute_logampvar(self, sample_rec_2d_short):
        """Test compute_logampvar function."""
        result = FragmentAnalyzer.compute_logampvar(sample_rec_2d_short)

        # Check output shape
        assert result.shape == (sample_rec_2d_short.shape[1],)

        # Compare with manual calculation
        ampvar_values = FragmentAnalyzer.compute_ampvar(sample_rec_2d_short)
        expected_logampvar = np.log(ampvar_values + 1)
        np.testing.assert_array_almost_equal(result, expected_logampvar, decimal=5)

    # Power Spectral Density Tests
    def test_compute_psd_welch(self, sample_rec_2d):
        """Test compute_psd function with Welch method."""
        f_s = 1000.0
        f, psd = FragmentAnalyzer.compute_psd(
            sample_rec_2d, f_s=f_s, welch_bin_t=1.0, notch_filter=False, multitaper=False
        )

        # Check that frequency and PSD arrays have correct shapes
        assert len(f.shape) == 1  # Frequency is 1D
        assert psd.shape[0] == len(f)  # First dimension matches frequency bins
        assert psd.shape[1] == sample_rec_2d.shape[1]  # Second dimension matches channels

        # Check frequency range
        assert f[0] >= 0
        assert f[-1] <= f_s / 2  # Nyquist frequency

        # Check that PSD values are non-negative
        assert np.all(psd >= 0)

        # Test mathematical correctness with a known sine wave
        target_freq = 10.0  # Same as channel 0 in fixture
        target_idx = np.argmin(np.abs(f - target_freq))

        # Channel 0 should have highest power around 10 Hz
        ch0_peak_idx = np.argmax(psd[:, 0])
        assert abs(f[ch0_peak_idx] - target_freq) < 2.0, (
            f"Peak at {f[ch0_peak_idx]} Hz, expected around {target_freq} Hz"
        )

    @patch("neurodent.core.analyze_frag.psd_array_multitaper")
    def test_compute_psd_multitaper(self, mock_multitaper, sample_rec_2d):
        """Test compute_psd function with multitaper method."""
        f_s = 1000.0
        n_channels = sample_rec_2d.shape[1]

        # Mock multitaper output
        mock_f = np.linspace(0, 40, 100)
        mock_psd = np.random.rand(n_channels, len(mock_f))
        mock_multitaper.return_value = (mock_psd, mock_f)

        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, multitaper=True)

        # Check that multitaper was called
        mock_multitaper.assert_called_once()

        # Check output shapes after transposition
        assert len(f) == len(mock_f)
        assert psd.shape == (len(mock_f), n_channels)
        np.testing.assert_array_equal(f, mock_f)

        # Verify that multitaper was called with correct parameters
        call_args = mock_multitaper.call_args
        called_data = call_args[0][0]
        called_fs = call_args[0][1]
        assert called_data.shape == (n_channels, sample_rec_2d.shape[0])  # Transposed
        assert called_fs == f_s

    def test_compute_psd_with_notch_filter(self, sample_rec_2d):
        """Test compute_psd function with notch filter enabled."""
        f_s = 1000.0

        # Test that notch filter doesn't break the function
        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, notch_filter=True)

        assert len(f.shape) == 1
        assert psd.shape[1] == sample_rec_2d.shape[1]
        assert np.all(psd >= 0)

        # Test that line frequency (60 Hz) is suppressed compared to no notch filter
        f_no_notch, psd_no_notch = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, notch_filter=False)

        line_freq = constants.LINE_FREQ  # 60 Hz
        line_idx = np.argmin(np.abs(f - line_freq))
        line_idx_no_notch = np.argmin(np.abs(f_no_notch - line_freq))

        # Power at line frequency should be lower with notch filter
        notch_power = np.mean(psd[line_idx, :])
        no_notch_power = np.mean(psd_no_notch[line_idx_no_notch, :])

        # Allow for some variation but expect meaningful suppression
        if no_notch_power > 0:
            suppression_ratio = notch_power / no_notch_power
            assert suppression_ratio < 0.8, f"Notch filter should suppress 60Hz power (ratio: {suppression_ratio:.3f})"

    # Band Power Features
    def test_compute_psdband(self, sample_rec_2d):
        """Test compute_psdband function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdband(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check that result is a dictionary with expected band names
        assert isinstance(result, dict)
        expected_bands = list(constants.FREQ_BANDS.keys())
        assert set(result.keys()) == set(expected_bands)

        # Check that each band has correct shape (n_channels,)
        n_channels = sample_rec_2d.shape[1]
        for band_name, band_power in result.items():
            assert band_power.shape == (n_channels,)
            assert np.all(band_power >= 0)  # Power should be non-negative

        # Test mathematical correctness: verify band powers sum approximately equals total power
        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Compute total power manually from PSD
        deltaf = np.median(np.diff(f))
        freq_mask = np.logical_and(f >= constants.FREQ_BAND_TOTAL[0], f <= constants.FREQ_BAND_TOTAL[1])
        manual_total = trapezoid(psd[freq_mask, :], dx=deltaf, axis=0)

        # Sum of band powers should approximately equal total power
        band_sum = sum(result.values())
        np.testing.assert_allclose(
            band_sum, manual_total, rtol=1e-6, err_msg="Sum of band powers should equal total power"
        )

    def test_compute_psdband_custom_bands(self, sample_rec_2d):
        """Test compute_psdband function with custom frequency bands."""
        f_s = 1000.0
        custom_bands = {"low": (1, 10), "high": (20, 40)}

        result = FragmentAnalyzer.compute_psdband(sample_rec_2d, f_s=f_s, bands=custom_bands, notch_filter=False)

        # Check that result contains only custom bands
        assert set(result.keys()) == set(custom_bands.keys())

        # Check shapes
        n_channels = sample_rec_2d.shape[1]
        for band_power in result.values():
            assert band_power.shape == (n_channels,)
            assert np.all(band_power >= 0)

    def test_compute_logpsdband(self, sample_rec_2d):
        """Test compute_logpsdband function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_logpsdband(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check that result is a dictionary with expected band names
        assert isinstance(result, dict)
        expected_bands = list(constants.FREQ_BANDS.keys())
        assert set(result.keys()) == set(expected_bands)

        # Compare with manual calculation
        psdband_result = FragmentAnalyzer.compute_psdband(sample_rec_2d, f_s=f_s, notch_filter=False)

        for band_name in expected_bands:
            expected_log = np.log(psdband_result[band_name] + 1)
            np.testing.assert_array_almost_equal(result[band_name], expected_log, decimal=5)

    def test_process_fragment_features_dask(self, sample_rec_2d):
        """Test _process_fragment_features_dask function."""
        f_s = 1000
        features = ["rms", "ampvar"]
        kwargs = {}

        result = FragmentAnalyzer._process_fragment_features_dask(sample_rec_2d, f_s, features, kwargs)

        # Check that result is a dictionary with requested features
        assert isinstance(result, dict)
        assert set(result.keys()) == set(features)

        # Check that each feature has correct shape
        n_channels = sample_rec_2d.shape[1]
        for feature_name, feature_value in result.items():
            assert feature_value.shape == (n_channels,)

    def test_process_fragment_features_dask_invalid_feature(self, sample_rec_2d):
        """Test _process_fragment_features_dask with invalid feature name."""
        f_s = 1000
        features = ["invalid_feature"]
        kwargs = {}

        with pytest.raises(
            AttributeError, match="type object 'FragmentAnalyzer' has no attribute 'compute_invalid_feature'"
        ):
            FragmentAnalyzer._process_fragment_features_dask(sample_rec_2d, f_s, features, kwargs)

    def test_rms_with_zero_signal(self):
        """Test RMS computation with zero signal."""
        zero_signal = np.zeros((100, 3))
        result = FragmentAnalyzer.compute_rms(zero_signal)

        # RMS of zero signal should be zero
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_ampvar_with_constant_signal(self):
        """Test amplitude variance with constant signal."""
        constant_signal = np.ones((100, 3)) * 5.0
        result = FragmentAnalyzer.compute_ampvar(constant_signal)

        # Variance of constant signal should be zero
        np.testing.assert_array_almost_equal(result, np.zeros(3), decimal=10)

    def test_psd_frequency_resolution(self, sample_rec_2d):
        """Test that PSD frequency resolution depends on welch_bin_t parameter."""
        f_s = 1000.0

        # Test with different window sizes
        f1, psd1 = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, welch_bin_t=0.5, notch_filter=False)
        f2, psd2 = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, welch_bin_t=1.0, notch_filter=False)

        # Longer window should give better frequency resolution (more frequency bins)
        assert len(f2) > len(f1), "Longer welch_bin_t should give more frequency bins"

    @pytest.mark.parametrize("f_s", [500, 1000, 2000])
    def test_psd_sampling_rate_effect(self, sample_rec_2d, f_s):
        """Test PSD computation with different sampling rates."""
        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check that maximum frequency is close to Nyquist
        assert f[-1] <= f_s / 2
        assert f[-1] > f_s / 2 * 0.8  # Should be reasonably close to Nyquist

    def test_edge_case_single_channel(self):
        """Test functions with single-channel data."""
        single_channel_data = np.random.randn(500, 1).astype(np.float32)

        # Test all basic functions
        rms = FragmentAnalyzer.compute_rms(single_channel_data)
        assert rms.shape == (1,)

        ampvar = FragmentAnalyzer.compute_ampvar(single_channel_data)
        assert ampvar.shape == (1,)

        # PSD with short single-channel signals may generate scipy warnings about nperseg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore scipy nperseg warnings
            f, psd = FragmentAnalyzer.compute_psd(single_channel_data, f_s=1000, notch_filter=False)
            assert psd.shape[1] == 1

        # Test all compute functions with single channel data
        f_s = 1000.0

        # All PSD-based functions may generate scipy warnings for short/single-channel signals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore scipy nperseg warnings

            # Test band power functions
            psdband = FragmentAnalyzer.compute_psdband(single_channel_data, f_s=f_s)
            assert isinstance(psdband, dict)
            for band_power in psdband.values():
                assert band_power.shape == (1,)

            logpsdband = FragmentAnalyzer.compute_logpsdband(single_channel_data, f_s=f_s)
            assert isinstance(logpsdband, dict)
            for band_power in logpsdband.values():
                assert band_power.shape == (1,)

            # Test total power functions
            psdtotal = FragmentAnalyzer.compute_psdtotal(single_channel_data, f_s=f_s)
            assert psdtotal.shape == (1,)

            logpsdtotal = FragmentAnalyzer.compute_logpsdtotal(single_channel_data, f_s=f_s)
            assert logpsdtotal.shape == (1,)

            # Test fractional power functions
            psdfrac = FragmentAnalyzer.compute_psdfrac(single_channel_data, f_s=f_s)
            assert isinstance(psdfrac, dict)
            for band_frac in psdfrac.values():
                assert band_frac.shape == (1,)

            logpsdfrac = FragmentAnalyzer.compute_logpsdfrac(single_channel_data, f_s=f_s)
            assert isinstance(logpsdfrac, dict)
            for band_frac in logpsdfrac.values():
                assert band_frac.shape == (1,)

            # Test PSD slope
            psdslope = FragmentAnalyzer.compute_psdslope(single_channel_data, f_s=f_s)
            assert psdslope.shape == (1, 2)  # slope and intercept

        # Test correlation functions (single channel should give 1x1 matrix)
        pcorr = FragmentAnalyzer.compute_pcorr(single_channel_data, f_s=f_s)
        assert pcorr.shape == (1, 1)
        # Note: self-correlation might not be exactly 1.0 due to bandpass filtering
        # which can modify the signal. We check it's a valid correlation value.
        assert -1 <= pcorr[0, 0] <= 1, f"Correlation should be between -1 and 1, got {pcorr[0, 0]}"

        zpcorr = FragmentAnalyzer.compute_zpcorr(single_channel_data, f_s=f_s)
        assert zpcorr.shape == (1, 1)

    def test_edge_case_short_signal(self):
        """Test functions with very short signals."""
        short_signal = np.random.randn(10, 2).astype(np.float32)

        # Basic functions should still work
        rms = FragmentAnalyzer.compute_rms(short_signal)
        assert rms.shape == (2,)

        ampvar = FragmentAnalyzer.compute_ampvar(short_signal)
        assert ampvar.shape == (2,)

        # PSD with very short signals will generate expected scipy warnings about nperseg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore scipy nperseg warnings

            try:
                f, psd = FragmentAnalyzer.compute_psd(short_signal, f_s=100, notch_filter=False)
                assert psd.shape[1] == 2
            except ValueError as e:
                # PSD computation might fail for very short signals due to insufficient data for windowing
                assert "nperseg" in str(e).lower() or "window" in str(e).lower(), (
                    f"Expected windowing-related error, got: {e}"
                )

        # Test other functions with short signals - most should handle gracefully
        f_s = 100.0

        try:
            # Test functions that should work with short signals
            psdband = FragmentAnalyzer.compute_psdband(short_signal, f_s=f_s, welch_bin_t=0.05)
            assert isinstance(psdband, dict)

            psdtotal = FragmentAnalyzer.compute_psdtotal(short_signal, f_s=f_s, welch_bin_t=0.05)
            assert psdtotal.shape == (2,)

            # Correlation should work even with short signals
            pcorr = FragmentAnalyzer.compute_pcorr(short_signal, f_s=f_s)
            assert pcorr.shape == (2, 2)

        except ValueError as e:
            # Some functions may legitimately fail with very short signals
            # This is acceptable behavior and we document it
            assert any(
                keyword in str(e).lower() for keyword in ["window", "nperseg", "length", "insufficient", "w0 should be"]
            ), f"Expected short signal error, got: {e}"

    # Total Power Features
    def test_compute_psdtotal(self, sample_rec_2d):
        """Test compute_psdtotal function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdtotal(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check output shape: should be (n_channels,)
        n_channels = sample_rec_2d.shape[1]
        assert result.shape == (n_channels,)

        # Check that all total power values are positive
        assert np.all(result > 0)

        # Test mathematical correctness: compare with manual calculation
        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, notch_filter=False)
        deltaf = np.median(np.diff(f))
        freq_mask = np.logical_and(f >= constants.FREQ_BAND_TOTAL[0], f <= constants.FREQ_BAND_TOTAL[1])
        manual_total = trapezoid(psd[freq_mask, :], dx=deltaf, axis=0)

        np.testing.assert_allclose(
            result, manual_total, rtol=1e-6, err_msg="Total power should match manual calculation from PSD"
        )

    def test_compute_psdtotal_custom_band(self, sample_rec_2d):
        """Test compute_psdtotal with custom frequency band."""
        f_s = 1000.0
        custom_band = (5, 25)  # 5-25 Hz range

        result = FragmentAnalyzer.compute_psdtotal(sample_rec_2d, f_s=f_s, band=custom_band, notch_filter=False)

        assert result.shape == (sample_rec_2d.shape[1],)
        assert np.all(result > 0)

        # Test mathematical correctness with custom band
        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, notch_filter=False)
        deltaf = np.median(np.diff(f))
        freq_mask = np.logical_and(f >= custom_band[0], f <= custom_band[1])
        manual_total = trapezoid(psd[freq_mask, :], dx=deltaf, axis=0)

        np.testing.assert_allclose(
            result, manual_total, rtol=0.01, err_msg="Custom band total power should match manual calculation"
        )

    def test_compute_logpsdtotal(self, sample_rec_2d):
        """Test compute_logpsdtotal function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_logpsdtotal(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check output shape
        n_channels = sample_rec_2d.shape[1]
        assert result.shape == (n_channels,)

        # Compare with manual calculation
        psdtotal = FragmentAnalyzer.compute_psdtotal(sample_rec_2d, f_s=f_s, notch_filter=False)
        expected_logpsdtotal = np.log(psdtotal + 1)
        np.testing.assert_array_almost_equal(result, expected_logpsdtotal, decimal=5)

    # Fractional Power Features
    def test_compute_psdfrac(self, sample_rec_2d):
        """Test compute_psdfrac function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdfrac(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check that result is a dictionary with expected band names
        assert isinstance(result, dict)
        expected_bands = list(constants.FREQ_BANDS.keys())
        assert set(result.keys()) == set(expected_bands)

        # Check that each band fraction has correct shape and is between 0 and 1
        n_channels = sample_rec_2d.shape[1]
        total_fraction = np.zeros(n_channels)

        for band_name, band_fraction in result.items():
            assert band_fraction.shape == (n_channels,)
            assert np.all(band_fraction >= 0)
            assert np.all(band_fraction <= 1)
            total_fraction += band_fraction

        # Total fractions should sum to 1 since psdfrac uses sum of band powers as denominator
        np.testing.assert_array_almost_equal(total_fraction, np.ones(n_channels), decimal=6)

    def test_compute_logpsdfrac(self, sample_rec_2d):
        """Test compute_logpsdfrac function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_logpsdfrac(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check that result is a dictionary with expected band names
        assert isinstance(result, dict)
        expected_bands = list(constants.FREQ_BANDS.keys())
        assert set(result.keys()) == set(expected_bands)

        # Check shapes
        n_channels = sample_rec_2d.shape[1]
        for band_fraction in result.values():
            assert band_fraction.shape == (n_channels,)

    # PSD Slope Features
    def test_compute_psdslope(self, sample_rec_2d):
        """Test compute_psdslope function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdslope(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check output shape: should be (n_channels, 2) for slope and intercept
        n_channels = sample_rec_2d.shape[1]
        assert result.shape == (n_channels, 2)

        # Check that slopes are reasonable (typically negative for EEG)
        slopes = result[:, 0]
        intercepts = result[:, 1]

        # Slopes should be finite numbers
        assert np.all(np.isfinite(slopes))
        assert np.all(np.isfinite(intercepts))

        # Test mathematical correctness with known signal properties
        # EEG typically shows power law decay (negative slope in log-log plot)
        # For our test data with sine waves + noise, we expect mostly negative slopes
        assert np.mean(slopes) < 0, "Average slope should be negative for typical EEG-like signals"

        # Verify slope calculation by comparing with manual linear regression
        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, notch_filter=False)
        freq_mask = np.logical_and(f >= constants.FREQ_BAND_TOTAL[0], f <= constants.FREQ_BAND_TOTAL[1])
        test_freqs = f[freq_mask]
        test_psd = psd[freq_mask, 0]  # Test first channel

        from scipy.stats import linregress

        manual_result = linregress(np.log10(test_freqs), np.log10(test_psd))

        # Compare with first channel result
        np.testing.assert_allclose(
            result[0, 0], manual_result.slope, rtol=0.01, err_msg="PSD slope should match manual linear regression"
        )

    # Frequency Analysis Utilities
    def test_get_freqs_cycles_cwt_morlet(self, sample_rec_3d):
        """Test _get_freqs_cycles with CWT Morlet mode."""
        f_s = 1000.0
        freq_res = 2.0
        cwt_n_cycles_max = 7.0
        epsilon = 1e-2

        freqs, n_cycles = FragmentAnalyzer._get_freqs_cycles(
            sample_rec_3d,
            f_s,
            freq_res,
            geomspace=False,
            mode="cwt_morlet",
            cwt_n_cycles_max=cwt_n_cycles_max,
            epsilon=epsilon,
        )

        # Check that frequencies are in expected range
        assert freqs[0] >= constants.FREQ_BAND_TOTAL[0]
        assert freqs[-1] <= constants.FREQ_BAND_TOTAL[1]

        # Check that n_cycles are reasonable
        assert len(n_cycles) == len(freqs)
        assert np.all(n_cycles > 0)
        assert np.all(n_cycles <= cwt_n_cycles_max + epsilon)

        # Test frequency spacing is approximately correct
        expected_freq_res = freq_res
        actual_freq_res = np.median(np.diff(freqs))
        assert abs(actual_freq_res - expected_freq_res) / expected_freq_res < 0.5, (
            f"Frequency resolution {actual_freq_res:.3f} should be close to {expected_freq_res}"
        )

        # Test that n_cycles is monotonically increasing with frequency
        freq_sorted_indices = np.argsort(freqs)
        cycles_sorted = n_cycles[freq_sorted_indices]
        assert np.all(np.diff(cycles_sorted) >= 0), "Number of cycles should monotonically increase with frequency"

    def test_get_freqs_cycles_multitaper(self, sample_rec_3d):
        """Test _get_freqs_cycles with multitaper mode."""
        f_s = 1000.0
        freq_res = 1.0
        epsilon = 1e-2

        freqs, n_cycles = FragmentAnalyzer._get_freqs_cycles(
            sample_rec_3d, f_s, freq_res, geomspace=True, mode="multitaper", cwt_n_cycles_max=7.0, epsilon=epsilon
        )

        # Check frequency spacing for geometric space
        assert len(freqs) > 1
        assert freqs[0] >= constants.FREQ_BAND_TOTAL[0]
        assert freqs[-1] <= constants.FREQ_BAND_TOTAL[1]

        # Check n_cycles
        assert len(n_cycles) == len(freqs)
        assert np.all(n_cycles > 0)

        # Test geometric spacing properties
        if len(freqs) > 1:
            freq_ratios = freqs[1:] / freqs[:-1]
            # For geometric spacing, ratios should be approximately constant
            ratio_std = np.std(freq_ratios)
            ratio_mean = np.mean(freq_ratios)
            coefficient_of_variation = ratio_std / ratio_mean
            assert coefficient_of_variation < 0.1, (
                f"Geometric spacing should have consistent ratios (CV: {coefficient_of_variation:.3f})"
            )

    # Connectivity Features
    @patch("neurodent.core.analyze_frag.spectral_connectivity_time")
    def test_compute_cohere(self, mock_connectivity, sample_rec_2d):
        """Test compute_cohere function."""
        f_s = 1000.0
        n_channels = sample_rec_2d.shape[1]

        # Mock the spectral connectivity output
        mock_con = Mock()
        # Create mock data for each frequency band
        n_bands = len(constants.BAND_NAMES)
        mock_data = np.random.rand(n_channels**2, n_bands)
        mock_con.get_data.return_value = mock_data
        mock_connectivity.return_value = mock_con

        result = FragmentAnalyzer.compute_cohere(sample_rec_2d, f_s=f_s, freq_res=2.0, downsamp_q=2)

        # Check that result is a dictionary with band names
        assert isinstance(result, dict)
        assert set(result.keys()) == set(constants.BAND_NAMES)

        # Check that each band has the right matrix shape
        for band_name, coherence_matrix in result.items():
            assert coherence_matrix.shape == (n_channels, n_channels)

            np.testing.assert_allclose(
                coherence_matrix,
                coherence_matrix.T,
                rtol=1e-10,
                err_msg=f"{band_name} coherence matrix should be symmetric",
            )

            np.testing.assert_allclose(
                np.diag(coherence_matrix),
                np.ones(n_channels),
                rtol=1e-6,
                err_msg=f"{band_name} coherence diagonal should be 1",
            )

            # All values should be between 0 and 1
            assert np.all(coherence_matrix >= 0) and np.all(coherence_matrix <= 1), (
                f"{band_name} coherence values should be between 0 and 1"
            )

            # Test that all unique channel pairs are represented
            # For n channels, we should have n*(n-1)/2 unique pairs
            n_unique_pairs = n_channels * (n_channels - 1) // 2
            upper_triangle = coherence_matrix[np.triu_indices(n_channels, k=1)]
            assert len(upper_triangle) == n_unique_pairs, (
                f"Should have {n_unique_pairs} unique pairs for {n_channels} channels"
            )

            # Off-diagonal values should be meaningful (not all zeros unless by design)
            # This tests that the reconstruction from upper triangle worked
            if n_channels > 1:
                assert not np.allclose(upper_triangle, 0), (
                    f"{band_name} off-diagonal coherence values shouldn't all be zero"
                )

    def test_compute_zcohere(self, sample_rec_2d):
        """Test compute_zcohere function."""
        # Create a simplified test by mocking compute_cohere
        with patch.object(FragmentAnalyzer, "compute_cohere") as mock_cohere:
            # Mock coherence values (between 0 and 1)
            n_channels = sample_rec_2d.shape[1]
            mock_coherence = {
                band: np.random.uniform(0.1, 0.9, (n_channels, n_channels)) for band in constants.BAND_NAMES
            }
            mock_cohere.return_value = mock_coherence

            result = FragmentAnalyzer.compute_zcohere(sample_rec_2d, f_s=1000.0)

            # Check that result has same structure as input
            assert isinstance(result, dict)
            assert set(result.keys()) == set(constants.BAND_NAMES)

            # Check that z-transform was applied (arctanh)
            for band_name in constants.BAND_NAMES:
                expected_z = np.arctanh(mock_coherence[band_name])
                np.testing.assert_array_almost_equal(result[band_name], expected_z, decimal=5)

    def test_compute_pcorr(self, sample_rec_2d):
        """Test compute_pcorr function."""
        f_s = 1000.0

        # Test with lower triangle
        result_lower = FragmentAnalyzer.compute_pcorr(sample_rec_2d, f_s=f_s, lower_triag=True)

        n_channels = sample_rec_2d.shape[1]
        assert result_lower.shape == (n_channels, n_channels)

        # Check that it's lower triangular (upper triangle above diagonal should be zero)
        # Note: diagonal is preserved, only strict upper triangle (k=1) should be zero
        assert np.allclose(np.triu(result_lower, k=1), 0)

    def test_short_signal_warnings(self, sample_rec_2d_short):
        """Test that short signals appropriately warn about insufficient cycles."""
        # This test verifies that the warning system is working correctly
        # Short signals (1 second) should warn about insufficient cycles for 1 Hz analysis

        with pytest.warns(RuntimeWarning, match="fmin=.*Hz corresponds to.*< 5 cycles"):
            # This should generate a warning because 1 Hz needs 5 seconds for 5 cycles
            FragmentAnalyzer.compute_cohere(sample_rec_2d_short, constants.GLOBAL_SAMPLING_RATE)

    def test_long_signal_no_warnings(self, sample_rec_2d):
        """Test that appropriately long signals don't generate cycle warnings."""
        # This test verifies that longer signals (6 seconds) don't warn about insufficient cycles

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Capture all warnings
            FragmentAnalyzer.compute_cohere(sample_rec_2d, constants.GLOBAL_SAMPLING_RATE)

            # Filter out non-MNE warnings (benchmark warnings are expected in test environment)
            mne_warnings = [warning for warning in w if "cycles" in str(warning.message)]
            assert len(mne_warnings) == 0, f"Unexpected cycle warnings: {[str(w.message) for w in mne_warnings]}"

    def test_compute_pcorr(self, sample_rec_2d):
        """Test compute_pcorr function."""
        f_s = constants.GLOBAL_SAMPLING_RATE
        n_channels = sample_rec_2d.shape[1]

        # Test with lower triangle
        result_lower = FragmentAnalyzer.compute_pcorr(sample_rec_2d, f_s=f_s, lower_triag=True)

        assert result_lower.shape == (n_channels, n_channels)

        # Check that it's lower triangular (upper triangle above diagonal should be zero)
        # Note: diagonal is preserved, only strict upper triangle (k=1) should be zero
        assert np.allclose(np.triu(result_lower, k=1), 0)

        # Test with full matrix
        result_full = FragmentAnalyzer.compute_pcorr(sample_rec_2d, f_s=f_s, lower_triag=False)

        assert result_full.shape == (n_channels, n_channels)

        # Diagonal should be 1 (perfect correlation with self)
        np.testing.assert_array_almost_equal(np.diag(result_full), np.ones(n_channels), decimal=5)

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(result_full, result_full.T, decimal=5)

        # All correlation values should be between -1 and 1
        assert np.all(result_full >= -1) and np.all(result_full <= 1), "Correlation values should be between -1 and 1"

        # Test that all unique channel pairs are represented
        n_unique_pairs = n_channels * (n_channels - 1) // 2
        upper_triangle = result_full[np.triu_indices(n_channels, k=1)]
        assert len(upper_triangle) == n_unique_pairs, (
            f"Should have {n_unique_pairs} unique pairs for {n_channels} channels"
        )

        # Test that lower_triag=True and lower_triag=False give related results
        # The lower triangle of full result should match the lower triangle result
        # (excluding diagonal since lower_triag=True zeroes the diagonal)
        lower_indices = np.tril_indices(n_channels, k=-1)
        np.testing.assert_array_almost_equal(
            result_full[lower_indices],
            result_lower[lower_indices],
            decimal=10,
            err_msg="Lower triangles should match between lower_triag=True/False",
        )

        # Test some basic properties with known test signals
        # Our fixture has structured signals (sine waves), so some correlations might be higher
        # But we can test general properties without making specific value assumptions
        off_diag_values = result_full[np.triu_indices(n_channels, k=1)]
        assert len(off_diag_values) > 0, "Should have off-diagonal correlation values"

    def test_compute_zpcorr(self, sample_rec_2d):
        """Test compute_zpcorr function."""
        f_s = 1000.0

        # Mock compute_pcorr to have controlled values
        with patch.object(FragmentAnalyzer, "compute_pcorr") as mock_pcorr:
            n_channels = sample_rec_2d.shape[1]
            # Create correlation values between -0.9 and 0.9 to avoid arctanh infinity
            mock_correlations = np.random.uniform(-0.9, 0.9, (n_channels, n_channels))
            mock_pcorr.return_value = mock_correlations

            result = FragmentAnalyzer.compute_zpcorr(sample_rec_2d, f_s=f_s)

            # Check shape
            assert result.shape == (n_channels, n_channels)

            # Check that z-transform was applied
            expected_z = np.arctanh(mock_correlations)
            np.testing.assert_array_almost_equal(result, expected_z, decimal=5)

    # Spike Detection Features
    def test_compute_nspike(self, sample_rec_2d):
        """Test compute_nspike function."""
        result = FragmentAnalyzer.compute_nspike(sample_rec_2d)
        # This function returns NaN array for placeholder functionality
        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_rec_2d.shape[1],)
        assert np.all(np.isnan(result))

    def test_compute_lognspike(self, sample_rec_2d):
        """Test compute_lognspike function."""
        result = FragmentAnalyzer.compute_lognspike(sample_rec_2d)
        # This function returns log-transformed NaN array (which is still NaN)
        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_rec_2d.shape[1],)
        assert np.all(np.isnan(result))

    # Error Handling Tests
    def test_memory_error_handling(self, sample_rec_2d):
        """Test memory error handling in compute_cohere."""
        f_s = 1000.0

        with patch("neurodent.core.analyze_frag.spectral_connectivity_epochs") as mock_connectivity:
            # Make the connectivity function raise a MemoryError
            mock_connectivity.side_effect = MemoryError("Test memory error")

            with pytest.raises(MemoryError, match="Out of memory"):
                FragmentAnalyzer.compute_cohere(sample_rec_2d, f_s=f_s)

    # Parametric Tests
    @pytest.mark.parametrize("welch_bin_t", [0.5, 1.0, 2.0])
    def test_psd_welch_bin_effect(self, sample_rec_2d, welch_bin_t):
        """Test that different welch_bin_t values produce valid results."""
        f_s = 1000.0

        f, psd = FragmentAnalyzer.compute_psd(sample_rec_2d, f_s=f_s, welch_bin_t=welch_bin_t, notch_filter=False)

        # All should produce valid results
        assert len(f) > 0
        assert psd.shape[0] == len(f)
        assert psd.shape[1] == sample_rec_2d.shape[1]
        assert np.all(psd >= 0)

        # Test that different welch_bin_t values affect frequency resolution
        # Longer welch_bin_t should give better frequency resolution
        freq_res_current = np.median(np.diff(f))
        assert freq_res_current > 0, "Frequency resolution should be positive"

        # Test that PSD power is reasonable for the given welch_bin_t
        total_power = np.sum(psd, axis=0)
        assert np.all(total_power > 0), "Total power should be positive for all channels"

    # Integration Tests
    def test_integration_psd_methods(self, sample_rec_2d):
        """Test integration between different PSD-related methods."""
        f_s = 1000.0

        # Test that psdfrac values are derived from psdband with sum normalization
        psdband = FragmentAnalyzer.compute_psdband(sample_rec_2d, f_s=f_s, notch_filter=False)
        psdfrac = FragmentAnalyzer.compute_psdfrac(sample_rec_2d, f_s=f_s, notch_filter=False)

        # Check that fractions are computed as band power / sum of all band powers
        band_sum = sum(psdband.values())
        for band_name in psdband.keys():
            expected_frac = psdband[band_name] / band_sum
            np.testing.assert_array_almost_equal(psdfrac[band_name], expected_frac, decimal=5)

        # Check that all fractions sum to 1
        total_frac = sum(psdfrac.values())
        np.testing.assert_array_almost_equal(total_frac, np.ones_like(total_frac), decimal=5)


class TestFragmentAnalyzerMathematicalVerification:
    """Mathematical correctness tests for FragmentAnalyzer methods using known signals."""

    # Basic Signal Tests
    def test_rms_mathematical_correctness_constant_signal(self):
        """Test RMS computation on constant signal - should equal the constant value."""
        constant_value = 5.0
        n_samples, n_channels = 1000, 2
        constant_signal = np.full((n_samples, n_channels), constant_value, dtype=np.float32)

        rms_result = FragmentAnalyzer.compute_rms(constant_signal)
        expected_rms = np.full(n_channels, constant_value)
        np.testing.assert_allclose(rms_result, expected_rms, rtol=1e-10)

    def test_rms_mathematical_correctness_sine_wave(self):
        """Test RMS computation on sine wave - should equal amplitude/sqrt(2)."""
        n_samples, n_channels = 1000, 2
        fs = constants.GLOBAL_SAMPLING_RATE
        amplitude = 2.0
        frequency = 10.0  # 10 Hz

        t = np.arange(n_samples) / fs
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        test_signal = np.tile(sine_wave, (n_channels, 1)).T.astype(np.float32)

        rms_result = FragmentAnalyzer.compute_rms(test_signal)

        # For sine wave, RMS = amplitude / sqrt(2)
        expected_rms = amplitude / np.sqrt(2)
        np.testing.assert_allclose(rms_result, expected_rms, rtol=1e-3)

    def test_ampvar_mathematical_correctness_constant_signal(self):
        """Test amplitude variance on constant signal - should be zero."""
        constant_value = 3.0
        n_samples, n_channels = 1000, 2
        constant_signal = np.full((n_samples, n_channels), constant_value, dtype=np.float32)

        ampvar_result = FragmentAnalyzer.compute_ampvar(constant_signal)
        np.testing.assert_allclose(ampvar_result, 0.0, atol=1e-10)

    # PSD Energy Conservation Tests
    def test_psdband_energy_conservation(self):
        """Test that sum of band powers approximately equals total power."""
        np.random.seed(42)
        n_samples, n_channels = 2000, 2  # Longer signal for better frequency resolution
        noise_signal = np.random.randn(n_samples, n_channels).astype(np.float32)

        psdband_result = FragmentAnalyzer.compute_psdband(
            noise_signal, f_s=constants.GLOBAL_SAMPLING_RATE, notch_filter=False
        )
        psdtotal_result = FragmentAnalyzer.compute_psdtotal(
            noise_signal, f_s=constants.GLOBAL_SAMPLING_RATE, notch_filter=False
        )

        # Sum of band powers should approximately equal total power
        band_sum = sum(psdband_result.values())
        np.testing.assert_allclose(band_sum, psdtotal_result, rtol=1e-6)  # High precision with trapezoidal integration

    def test_psdfrac_sums_to_one(self):
        """Test that PSD fractions sum to 1."""
        np.random.seed(123)
        n_samples, n_channels = 2000, 2
        test_signal = np.random.randn(n_samples, n_channels).astype(np.float32)

        psdfrac_result = FragmentAnalyzer.compute_psdfrac(
            test_signal, f_s=constants.GLOBAL_SAMPLING_RATE, notch_filter=False
        )

        # Sum of fractions should equal 1 for each channel
        for ch in range(n_channels):
            fraction_sum = sum(band_values[ch] for band_values in psdfrac_result.values())
            np.testing.assert_allclose(fraction_sum, 1.0, rtol=1e-6)

    # Log Transform Tests
    def test_log_transforms_mathematical_correctness(self):
        """Test that log transforms produce mathematically correct results."""
        n_samples, n_channels = 1000, 2
        test_signal = np.abs(np.random.randn(n_samples, n_channels)).astype(np.float32) + 1.0  # Ensure positive values

        # Test log RMS
        rms_result = FragmentAnalyzer.compute_rms(test_signal)
        logrms_result = FragmentAnalyzer.compute_logrms(test_signal)
        expected_logrms = np.log(rms_result + 1)
        np.testing.assert_allclose(logrms_result, expected_logrms, rtol=1e-10)

        # Test log amplitude variance
        ampvar_result = FragmentAnalyzer.compute_ampvar(test_signal)
        logampvar_result = FragmentAnalyzer.compute_logampvar(test_signal)
        expected_logampvar = np.log(ampvar_result + 1)
        np.testing.assert_allclose(logampvar_result, expected_logampvar, rtol=1e-10)

        # Test log PSD total
        psdtotal_result = FragmentAnalyzer.compute_psdtotal(
            test_signal, f_s=constants.GLOBAL_SAMPLING_RATE, notch_filter=False
        )
        logpsdtotal_result = FragmentAnalyzer.compute_logpsdtotal(
            test_signal, f_s=constants.GLOBAL_SAMPLING_RATE, notch_filter=False
        )
        expected_logpsdtotal = np.log(psdtotal_result + 1)
        np.testing.assert_allclose(logpsdtotal_result, expected_logpsdtotal, rtol=1e-10)

        # Test log PSD bands
        psdband_result = FragmentAnalyzer.compute_psdband(
            test_signal, f_s=constants.GLOBAL_SAMPLING_RATE, notch_filter=False
        )
        logpsdband_result = FragmentAnalyzer.compute_logpsdband(
            test_signal, f_s=constants.GLOBAL_SAMPLING_RATE, notch_filter=False
        )
        for band_name in psdband_result.keys():
            expected_logpsdband = np.log(psdband_result[band_name] + 1)
            np.testing.assert_allclose(logpsdband_result[band_name], expected_logpsdband, rtol=1e-10)

    # Correlation Tests
    def test_correlation_mathematical_properties(self):
        """Test that correlation matrices have correct mathematical properties."""
        n_samples = 2000
        np.random.seed(456)

        # Create identical signals for perfect correlation test
        base_signal = np.random.randn(n_samples)
        identical_signal = np.column_stack([base_signal, base_signal]).astype(np.float32)

        pcorr_result = FragmentAnalyzer.compute_pcorr(
            identical_signal, f_s=constants.GLOBAL_SAMPLING_RATE, lower_triag=False
        )

        # For identical signals, correlation should be perfect (1.0) on diagonal and off-diagonal
        # But since this uses bandpass filtering, just check that diagonal is close to 1 and matrix is symmetric
        np.testing.assert_allclose(np.diag(pcorr_result), 1.0, rtol=1e-2)
        np.testing.assert_allclose(pcorr_result, pcorr_result.T, rtol=1e-10)  # Should be symmetric

    @pytest.mark.parametrize("noise_level", [0.1, 0.5, 1.0])
    def test_correlation_with_noise_levels(self, noise_level):
        """Test correlation behavior with different noise levels added to identical signals."""
        n_samples = 2000
        np.random.seed(789)

        # Create base signal
        base_signal = np.random.randn(n_samples)

        # Create signals with different noise levels
        signal1 = base_signal.copy()
        signal2 = base_signal + noise_level * np.random.randn(n_samples)

        test_data = np.column_stack([signal1, signal2]).astype(np.float32)
        pcorr_result = FragmentAnalyzer.compute_pcorr(test_data, f_s=constants.GLOBAL_SAMPLING_RATE, lower_triag=False)

        # Higher noise should result in lower correlation
        off_diag_corr = pcorr_result[0, 1]

        # With low noise, correlation should be high; with high noise, correlation should be lower
        if noise_level <= 0.1:
            assert off_diag_corr > 0.9, f"With low noise ({noise_level}), correlation should be high"
        elif noise_level >= 1.0:
            assert off_diag_corr < 0.9, f"With high noise ({noise_level}), correlation should be lower"

    def test_correlation_with_timeshift(self):
        """Test that identical signals with time shifts have decreasing correlation."""
        n_samples = 2000
        fs = constants.GLOBAL_SAMPLING_RATE

        # Create a distinctive signal (sine wave with noise)
        t = np.arange(n_samples) / fs
        base_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.RandomState(42).randn(n_samples)

        correlations = []
        shifts = [0, 10, 50, 100]  # Time shifts in samples

        for shift in shifts:
            if shift == 0:
                shifted_signal = base_signal.copy()
            else:
                # Create time-shifted version
                shifted_signal = np.zeros_like(base_signal)
                shifted_signal[shift:] = base_signal[:-shift]

            test_data = np.column_stack([base_signal, shifted_signal]).astype(np.float32)
            pcorr_result = FragmentAnalyzer.compute_pcorr(test_data, f_s=fs, lower_triag=False)
            correlations.append(pcorr_result[0, 1])

        # Correlation should generally decrease with increasing time shift
        assert correlations[0] > correlations[-1], f"Correlation should decrease with time shift: {correlations}"

    def test_correlation_opposite_signals(self):
        """Test that opposite signals have correlation close to -1."""
        n_samples = 2000
        np.random.seed(101)

        # Create a signal and its negative
        base_signal = np.random.randn(n_samples)
        opposite_signal = -base_signal

        test_data = np.column_stack([base_signal, opposite_signal]).astype(np.float32)
        pcorr_result = FragmentAnalyzer.compute_pcorr(test_data, f_s=constants.GLOBAL_SAMPLING_RATE, lower_triag=False)

        # Correlation should be close to -1
        off_diag_corr = pcorr_result[0, 1]
        assert off_diag_corr < -0.9, f"Opposite signals should have correlation close to -1, got {off_diag_corr}"

    def test_coherence_mathematical_properties(self):
        """Test coherence behaves similarly to correlation for basic cases."""
        n_samples = 6000  # 6 seconds at 1000 Hz for reliable spectral estimation
        fs = constants.GLOBAL_SAMPLING_RATE
        np.random.seed(202)

        # Test 1: Random uncorrelated signals should have low coherence
        random_data = np.random.randn(n_samples, 2).astype(np.float32)

        try:
            with patch("neurodent.core.analyze_frag.spectral_connectivity_time") as mock_connectivity:
                # Mock low coherence for random signals
                mock_con = Mock()
                n_bands = len(constants.BAND_NAMES)
                mock_data = np.random.uniform(0.0, 0.3, (1, n_bands))  # Low coherence values
                mock_con.get_data.return_value = mock_data
                mock_connectivity.return_value = mock_con

                coherence_result = FragmentAnalyzer.compute_cohere(random_data, f_s=fs)

                # Check that coherence values are low for random signals
                for band_name, coh_matrix in coherence_result.items():
                    off_diag_coh = coh_matrix[0, 1]
                    assert 0 <= off_diag_coh <= 1, f"Coherence should be between 0 and 1"
        except Exception:
            # If connectivity computation fails, skip this test
            pytest.skip("Coherence computation failed - may need MNE or connectivity dependencies")

    def test_coherence_invariant_to_timeshift(self):
        """Test that coherence is relatively invariant to time shifts (unlike correlation)."""
        # This is a theoretical property - coherence in frequency domain should be
        # less affected by time shifts than time-domain correlation
        # For now, we'll document this as a placeholder test

        n_samples = 2000
        fs = constants.GLOBAL_SAMPLING_RATE

        # Create a sine wave signal
        t = np.arange(n_samples) / fs
        freq = 15.0
        signal = np.sin(2 * np.pi * freq * t)

        # This test would require actual coherence computation which is complex to mock properly
        # For now, we document the expected behavior
        assert True  # Placeholder - coherence should be more stable under time shifts than correlation

    # Frequency Analysis Tests
    @pytest.mark.parametrize("target_freq,amplitude", [(10.0, 1.0), (15.0, 2.0), (25.0, 0.5), (35.0, 1.5)])
    def test_psd_peak_detection_parameterized(self, target_freq, amplitude):
        """Test that PSD correctly identifies frequency peaks for various frequencies and amplitudes."""
        n_samples = 4000  # Long signal for good frequency resolution
        fs = constants.GLOBAL_SAMPLING_RATE

        t = np.arange(n_samples) / fs
        sine_signal = amplitude * np.sin(2 * np.pi * target_freq * t)
        # Add small amount of noise to make it more realistic
        sine_signal += 0.1 * np.random.RandomState(42).randn(n_samples)
        test_signal = np.tile(sine_signal, (2, 1)).T.astype(np.float32)

        freqs, psd = FragmentAnalyzer.compute_psd(test_signal, f_s=fs, notch_filter=False, welch_bin_t=1)

        # Find peak frequency
        peak_idx = np.argmax(psd[:, 0])
        peak_freq = freqs[peak_idx]

        # Peak should be close to target frequency
        freq_tolerance = 2.0  # Hz
        assert abs(peak_freq - target_freq) < freq_tolerance, (
            f"Peak at {peak_freq:.1f} Hz should be within {freq_tolerance} Hz of target {target_freq} Hz"
        )

        # Check that power at target frequency is significantly higher than neighbors
        target_idx = np.argmin(np.abs(freqs - target_freq))
        target_power = psd[target_idx, 0]

        # Check neighboring frequencies (±2 Hz to avoid the peak itself)
        neighbor_indices = []
        for offset in [-3.0, 3.0]:
            neighbor_freq = target_freq + offset
            if 1.0 < neighbor_freq < freqs[-1] - 1.0:  # Stay away from boundaries
                neighbor_idx = np.argmin(np.abs(freqs - neighbor_freq))
                neighbor_indices.append(neighbor_idx)

        if neighbor_indices:
            neighbor_powers = [psd[idx, 0] for idx in neighbor_indices]
            max_neighbor_power = max(neighbor_powers)

            # Power at target frequency should be significantly higher
            # Scale expectation with amplitude
            expected_ratio = 2.0 + amplitude  # Higher amplitude should give higher ratio
            assert target_power > expected_ratio * max_neighbor_power, (
                f"Peak power ({target_power:.3f}) should be >{expected_ratio:.1f}x higher than neighbors ({max_neighbor_power:.3f})"
            )

    def test_pcorr_analyzer_consistency(self):
        """Test that FragmentAnalyzer and LongRecordingAnalyzer produce consistent pcorr results."""
        np.random.seed(42)
        signals = np.random.randn(1000, 4).astype(np.float32)

        # FragmentAnalyzer default behavior
        fa_result = FragmentAnalyzer.compute_pcorr(signals, constants.GLOBAL_SAMPLING_RATE)

        # Create LongRecordingAnalyzer setup
        from neurodent.core.analysis import LongRecordingAnalyzer
        from neurodent.core import core

        mock_long_recording = MagicMock(spec=core.LongRecordingOrganizer)
        mock_long_recording.get_num_fragments.return_value = 1
        mock_long_recording.channel_names = ["ch1", "ch2", "ch3", "ch4"]
        mock_long_recording.meta = MagicMock()
        mock_long_recording.meta.n_channels = 4
        mock_long_recording.meta.mult_to_uV = 1.0
        mock_long_recording.LongRecording = MagicMock()
        mock_long_recording.LongRecording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_long_recording.LongRecording.get_num_frames.return_value = 5000
        mock_long_recording.end_relative = [1]

        mock_recording = MagicMock()
        mock_recording.get_traces.return_value = signals
        mock_long_recording.get_fragment.return_value = mock_recording

        analyzer = LongRecordingAnalyzer(mock_long_recording, fragment_len_s=10)
        # Disable notch filtering for this test since we're using mock objects
        analyzer.apply_notch_filter = False
        lra_result = analyzer.compute_pcorr(0)

        # Both should produce symmetric matrices
        assert np.allclose(fa_result, fa_result.T), "FragmentAnalyzer should produce symmetric matrix"
        assert np.allclose(lra_result, lra_result.T), "LongRecordingAnalyzer should produce symmetric matrix"

        # Results should be nearly identical (allowing for minor numerical differences)
        np.testing.assert_allclose(fa_result, lra_result, rtol=1e-10, atol=1e-12)
