"""
Unit tests for pythoneeg.core.analyze_frag module.
"""
import numpy as np
import pytest
from unittest.mock import patch, Mock

from pythoneeg.core.analyze_frag import FragmentAnalyzer
from pythoneeg import constants


class TestFragmentAnalyzer:
    """Test FragmentAnalyzer static methods."""
    
    @pytest.fixture
    def sample_rec_2d(self):
        """Create a 2D sample recording array (N_samples x N_channels)."""
        np.random.seed(42)  # For reproducible tests
        n_samples, n_channels = 1000, 4
        # Create a realistic EEG-like signal with different frequencies per channel
        t = np.linspace(0, 1, n_samples)
        data = np.zeros((n_samples, n_channels))
        
        # Channel 0: 10 Hz sine wave + noise
        data[:, 0] = 100 * np.sin(2 * np.pi * 10 * t) + 10 * np.random.randn(n_samples)
        # Channel 1: 20 Hz sine wave + noise  
        data[:, 1] = 80 * np.sin(2 * np.pi * 20 * t) + 15 * np.random.randn(n_samples)
        # Channel 2: Mix of frequencies + noise
        data[:, 2] = (50 * np.sin(2 * np.pi * 5 * t) + 
                     30 * np.sin(2 * np.pi * 15 * t) + 
                     20 * np.random.randn(n_samples))
        # Channel 3: Higher frequency + noise
        data[:, 3] = 60 * np.sin(2 * np.pi * 30 * t) + 25 * np.random.randn(n_samples)
        
        return data.astype(np.float32)
    
    @pytest.fixture
    def sample_rec_3d(self, sample_rec_2d):
        """Create a 3D sample recording array for MNE (1 x N_channels x N_samples)."""
        return sample_rec_2d.T[np.newaxis, :, :]  # (1, n_channels, n_samples)
    
    def test_check_rec_np_valid(self, sample_rec_2d):
        """Test _check_rec_np with valid 2D array."""
        # Should not raise any exception
        FragmentAnalyzer._check_rec_np(sample_rec_2d)
    
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
    
    def test_reshape_np_for_mne(self, sample_rec_2d):
        """Test _reshape_np_for_mne conversion."""
        n_samples, n_channels = sample_rec_2d.shape
        result = FragmentAnalyzer._reshape_np_for_mne(sample_rec_2d)
        
        # Check output shape: (1, n_channels, n_samples)
        assert result.shape == (1, n_channels, n_samples)
        
        # Check data integrity (first sample, all channels)
        np.testing.assert_array_almost_equal(
            result[0, :, 0], sample_rec_2d[0, :], decimal=5
        )
    
    def test_compute_rms(self, sample_rec_2d):
        """Test compute_rms function."""
        result = FragmentAnalyzer.compute_rms(sample_rec_2d)
        
        # Check output shape: should be (n_channels,)
        assert result.shape == (sample_rec_2d.shape[1],)
        
        # Check that all RMS values are positive
        assert np.all(result > 0)
        
        # Manually compute RMS for first channel and compare
        expected_rms_ch0 = np.sqrt(np.mean(sample_rec_2d[:, 0] ** 2))
        # Use decimal=4 for float32 precision (was decimal=5)
        np.testing.assert_array_almost_equal(result[0], expected_rms_ch0, decimal=4)
    
    def test_compute_logrms(self, sample_rec_2d):
        """Test compute_logrms function."""
        result = FragmentAnalyzer.compute_logrms(sample_rec_2d)
        
        # Check output shape
        assert result.shape == (sample_rec_2d.shape[1],)
        
        # Compare with manual calculation
        rms_values = FragmentAnalyzer.compute_rms(sample_rec_2d)
        expected_logrms = np.log(rms_values + 1)
        np.testing.assert_array_almost_equal(result, expected_logrms, decimal=5)
    
    def test_compute_ampvar(self, sample_rec_2d):
        """Test compute_ampvar function."""
        result = FragmentAnalyzer.compute_ampvar(sample_rec_2d)
        
        # Check output shape
        assert result.shape == (sample_rec_2d.shape[1],)
        
        # Check that all variance values are non-negative
        assert np.all(result >= 0)
        
        # Manually compute variance for first channel and compare
        expected_var_ch0 = np.std(sample_rec_2d[:, 0]) ** 2
        # Use decimal=3 for float32 precision (was decimal=5)
        np.testing.assert_array_almost_equal(result[0], expected_var_ch0, decimal=3)
    
    def test_compute_logampvar(self, sample_rec_2d):
        """Test compute_logampvar function."""
        result = FragmentAnalyzer.compute_logampvar(sample_rec_2d)
        
        # Check output shape
        assert result.shape == (sample_rec_2d.shape[1],)
        
        # Compare with manual calculation
        ampvar_values = FragmentAnalyzer.compute_ampvar(sample_rec_2d)
        expected_logampvar = np.log(ampvar_values + 1)
        np.testing.assert_array_almost_equal(result, expected_logampvar, decimal=5)
    
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
    
    @patch('pythoneeg.core.analyze_frag.psd_array_multitaper')
    def test_compute_psd_multitaper(self, mock_multitaper, sample_rec_2d):
        """Test compute_psd function with multitaper method."""
        f_s = 1000.0
        n_channels = sample_rec_2d.shape[1]
        
        # Mock multitaper output
        mock_f = np.linspace(0, 40, 100)
        mock_psd = np.random.rand(n_channels, len(mock_f))
        mock_multitaper.return_value = (mock_psd, mock_f)
        
        f, psd = FragmentAnalyzer.compute_psd(
            sample_rec_2d, f_s=f_s, multitaper=True
        )
        
        # Check that multitaper was called
        mock_multitaper.assert_called_once()
        
        # Check output shapes after transposition
        assert len(f) == len(mock_f)
        assert psd.shape == (len(mock_f), n_channels)
        np.testing.assert_array_equal(f, mock_f)
    
    def test_compute_psd_with_notch_filter(self, sample_rec_2d):
        """Test compute_psd function with notch filter enabled."""
        f_s = 1000.0
        
        # Test that notch filter doesn't break the function
        f, psd = FragmentAnalyzer.compute_psd(
            sample_rec_2d, f_s=f_s, notch_filter=True
        )
        
        assert len(f.shape) == 1
        assert psd.shape[1] == sample_rec_2d.shape[1]
        assert np.all(psd >= 0)
    
    def test_compute_psdband(self, sample_rec_2d):
        """Test compute_psdband function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdband(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
        # Check that result is a dictionary with expected band names
        assert isinstance(result, dict)
        expected_bands = list(constants.FREQ_BANDS.keys())
        assert set(result.keys()) == set(expected_bands)
        
        # Check that each band has correct shape (n_channels,)
        n_channels = sample_rec_2d.shape[1]
        for band_name, band_power in result.items():
            assert band_power.shape == (n_channels,)
            assert np.all(band_power >= 0)  # Power should be non-negative
    
    def test_compute_psdband_custom_bands(self, sample_rec_2d):
        """Test compute_psdband function with custom frequency bands."""
        f_s = 1000.0
        custom_bands = {"low": (1, 10), "high": (20, 40)}
        
        result = FragmentAnalyzer.compute_psdband(
            sample_rec_2d, f_s=f_s, bands=custom_bands, notch_filter=False
        )
        
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
        result = FragmentAnalyzer.compute_logpsdband(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
        # Check that result is a dictionary with expected band names
        assert isinstance(result, dict)
        expected_bands = list(constants.FREQ_BANDS.keys())
        assert set(result.keys()) == set(expected_bands)
        
        # Compare with manual calculation
        psdband_result = FragmentAnalyzer.compute_psdband(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
        for band_name in expected_bands:
            expected_log = np.log(psdband_result[band_name] + 1)
            np.testing.assert_array_almost_equal(
                result[band_name], expected_log, decimal=5
            )
    
    def test_process_fragment_features_dask(self, sample_rec_2d):
        """Test _process_fragment_features_dask function."""
        f_s = 1000
        features = ["rms", "ampvar"]
        kwargs = {}
        
        result = FragmentAnalyzer._process_fragment_features_dask(
            sample_rec_2d, f_s, features, kwargs
        )
        
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
        
        with pytest.raises(AttributeError, match="type object 'FragmentAnalyzer' has no attribute 'compute_invalid_feature'"):
            FragmentAnalyzer._process_fragment_features_dask(
                sample_rec_2d, f_s, features, kwargs
            )
    
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
        f1, psd1 = FragmentAnalyzer.compute_psd(
            sample_rec_2d, f_s=f_s, welch_bin_t=0.5, notch_filter=False
        )
        f2, psd2 = FragmentAnalyzer.compute_psd(
            sample_rec_2d, f_s=f_s, welch_bin_t=1.0, notch_filter=False
        )
        
        # Longer window should give better frequency resolution (more frequency bins)
        assert len(f2) > len(f1), "Longer welch_bin_t should give more frequency bins"
    
    @pytest.mark.parametrize("f_s", [500, 1000, 2000])
    def test_psd_sampling_rate_effect(self, sample_rec_2d, f_s):
        """Test PSD computation with different sampling rates."""
        f, psd = FragmentAnalyzer.compute_psd(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
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
        
        f, psd = FragmentAnalyzer.compute_psd(single_channel_data, f_s=1000, notch_filter=False)
        assert psd.shape[1] == 1
    
    def test_edge_case_short_signal(self):
        """Test functions with very short signals."""
        short_signal = np.random.randn(10, 2).astype(np.float32)
        
        # Basic functions should still work
        rms = FragmentAnalyzer.compute_rms(short_signal)
        assert rms.shape == (2,)
        
        ampvar = FragmentAnalyzer.compute_ampvar(short_signal)
        assert ampvar.shape == (2,)
        
        # PSD might have issues with very short signals, but should not crash
        try:
            f, psd = FragmentAnalyzer.compute_psd(short_signal, f_s=100, notch_filter=False)
            assert psd.shape[1] == 2
        except ValueError:
            # This is acceptable for very short signals
            pass

    def test_compute_psdtotal(self, sample_rec_2d):
        """Test compute_psdtotal function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdtotal(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
        # Check output shape: should be (n_channels,)
        n_channels = sample_rec_2d.shape[1]
        assert result.shape == (n_channels,)
        
        # Check that all total power values are positive
        assert np.all(result > 0)
    
    def test_compute_psdtotal_custom_band(self, sample_rec_2d):
        """Test compute_psdtotal with custom frequency band."""
        f_s = 1000.0
        custom_band = (5, 25)  # 5-25 Hz range
        
        result = FragmentAnalyzer.compute_psdtotal(
            sample_rec_2d, f_s=f_s, band=custom_band, notch_filter=False
        )
        
        assert result.shape == (sample_rec_2d.shape[1],)
        assert np.all(result > 0)
    
    def test_compute_logpsdtotal(self, sample_rec_2d):
        """Test compute_logpsdtotal function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_logpsdtotal(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
        # Check output shape
        n_channels = sample_rec_2d.shape[1]
        assert result.shape == (n_channels,)
        
        # Compare with manual calculation
        psdtotal = FragmentAnalyzer.compute_psdtotal(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        expected_logpsdtotal = np.log(psdtotal + 1)
        np.testing.assert_array_almost_equal(result, expected_logpsdtotal, decimal=5)
    
    def test_compute_psdfrac(self, sample_rec_2d):
        """Test compute_psdfrac function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdfrac(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
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
        
        # Total fractions should approximately sum to 1
        # KNOWN BUG: Frequency bands have overlapping boundaries causing double-counting
        # at band edges. This test will fail until the core implementation is fixed.
        np.testing.assert_array_almost_equal(total_fraction, np.ones(n_channels), decimal=1)
    
    def test_compute_logpsdfrac(self, sample_rec_2d):
        """Test compute_logpsdfrac function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_logpsdfrac(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
        # Check that result is a dictionary with expected band names
        assert isinstance(result, dict)
        expected_bands = list(constants.FREQ_BANDS.keys())
        assert set(result.keys()) == set(expected_bands)
        
        # Check shapes
        n_channels = sample_rec_2d.shape[1]
        for band_fraction in result.values():
            assert band_fraction.shape == (n_channels,)
    
    def test_compute_psdslope(self, sample_rec_2d):
        """Test compute_psdslope function."""
        f_s = 1000.0
        result = FragmentAnalyzer.compute_psdslope(
            sample_rec_2d, f_s=f_s, notch_filter=False
        )
        
        # Check output shape: should be (n_channels, 2) for slope and intercept
        n_channels = sample_rec_2d.shape[1]
        assert result.shape == (n_channels, 2)
        
        # Check that slopes are reasonable (typically negative for EEG)
        slopes = result[:, 0]
        intercepts = result[:, 1]
        
        # Slopes should be finite numbers
        assert np.all(np.isfinite(slopes))
        assert np.all(np.isfinite(intercepts))
    
    def test_get_freqs_cycles_cwt_morlet(self, sample_rec_3d):
        """Test _get_freqs_cycles with CWT Morlet mode."""
        f_s = 1000.0
        freq_res = 2.0
        cwt_n_cycles_max = 7.0
        epsilon = 1e-2
        
        freqs, n_cycles = FragmentAnalyzer._get_freqs_cycles(
            sample_rec_3d, f_s, freq_res, geomspace=False, 
            mode="cwt_morlet", cwt_n_cycles_max=cwt_n_cycles_max, epsilon=epsilon
        )
        
        # Check that frequencies are in expected range
        assert freqs[0] >= constants.FREQ_BAND_TOTAL[0]
        assert freqs[-1] <= constants.FREQ_BAND_TOTAL[1]
        
        # Check that n_cycles are reasonable
        assert len(n_cycles) == len(freqs)
        assert np.all(n_cycles > 0)
        assert np.all(n_cycles <= cwt_n_cycles_max + epsilon)
    
    def test_get_freqs_cycles_multitaper(self, sample_rec_3d):
        """Test _get_freqs_cycles with multitaper mode."""
        f_s = 1000.0
        freq_res = 1.0
        epsilon = 1e-2
        
        freqs, n_cycles = FragmentAnalyzer._get_freqs_cycles(
            sample_rec_3d, f_s, freq_res, geomspace=True, 
            mode="multitaper", cwt_n_cycles_max=7.0, epsilon=epsilon
        )
        
        # Check frequency spacing for geometric space
        assert len(freqs) > 1
        assert freqs[0] >= constants.FREQ_BAND_TOTAL[0]
        assert freqs[-1] <= constants.FREQ_BAND_TOTAL[1]
        
        # Check n_cycles
        assert len(n_cycles) == len(freqs)
        assert np.all(n_cycles > 0)
    
    @patch('pythoneeg.core.analyze_frag.spectral_connectivity_time')
    def test_compute_cohere(self, mock_connectivity, sample_rec_2d):
        """Test compute_cohere function."""
        f_s = 1000.0
        n_channels = sample_rec_2d.shape[1]
        
        # Mock the spectral connectivity output
        mock_con = Mock()
        # Create mock data for each frequency band
        n_bands = len(constants.BAND_NAMES)
        mock_data = np.random.rand(n_channels * (n_channels - 1) // 2, n_bands)
        mock_con.get_data.return_value = mock_data
        mock_connectivity.return_value = mock_con
        
        result = FragmentAnalyzer.compute_cohere(
            sample_rec_2d, f_s=f_s, freq_res=2.0, downsamp_q=2
        )
        
        # Check that result is a dictionary with band names
        assert isinstance(result, dict)
        assert set(result.keys()) == set(constants.BAND_NAMES)
        
        # Check that each band has the right matrix shape
        for band_name, coherence_matrix in result.items():
            assert coherence_matrix.shape == (n_channels, n_channels)
    
    def test_compute_zcohere(self, sample_rec_2d):
        """Test compute_zcohere function."""
        # Create a simplified test by mocking compute_cohere
        with patch.object(FragmentAnalyzer, 'compute_cohere') as mock_cohere:
            # Mock coherence values (between 0 and 1)
            n_channels = sample_rec_2d.shape[1]
            mock_coherence = {
                band: np.random.uniform(0.1, 0.9, (n_channels, n_channels))
                for band in constants.BAND_NAMES
            }
            mock_cohere.return_value = mock_coherence
            
            result = FragmentAnalyzer.compute_zcohere(sample_rec_2d, f_s=1000.0)
            
            # Check that result has same structure as input
            assert isinstance(result, dict)
            assert set(result.keys()) == set(constants.BAND_NAMES)
            
            # Check that z-transform was applied (arctanh)
            for band_name in constants.BAND_NAMES:
                expected_z = np.arctanh(mock_coherence[band_name])
                np.testing.assert_array_almost_equal(
                    result[band_name], expected_z, decimal=5
                )
    
    def test_compute_pcorr(self, sample_rec_2d):
        """Test compute_pcorr function."""
        f_s = 1000.0
        
        # Test with lower triangle
        result_lower = FragmentAnalyzer.compute_pcorr(
            sample_rec_2d, f_s=f_s, lower_triag=True
        )
        
        n_channels = sample_rec_2d.shape[1]
        assert result_lower.shape == (n_channels, n_channels)
        
        # Check that it's lower triangular (upper triangle should be zero)
        assert np.allclose(np.triu(result_lower, k=0), 0)
        
        # Test with full matrix
        result_full = FragmentAnalyzer.compute_pcorr(
            sample_rec_2d, f_s=f_s, lower_triag=False
        )
        
        assert result_full.shape == (n_channels, n_channels)
        
        # Diagonal should be 1 (perfect correlation with self)
        np.testing.assert_array_almost_equal(
            np.diag(result_full), np.ones(n_channels), decimal=5
        )
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            result_full, result_full.T, decimal=5
        )
    
    def test_compute_zpcorr(self, sample_rec_2d):
        """Test compute_zpcorr function."""
        f_s = 1000.0
        
        # Mock compute_pcorr to have controlled values
        with patch.object(FragmentAnalyzer, 'compute_pcorr') as mock_pcorr:
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
    
    def test_compute_nspike(self, sample_rec_2d):
        """Test compute_nspike function."""
        result = FragmentAnalyzer.compute_nspike(sample_rec_2d)
        # This function should return None as documented
        assert result is None
    
    def test_compute_lognspike(self, sample_rec_2d):
        """Test compute_lognspike function."""
        result = FragmentAnalyzer.compute_lognspike(sample_rec_2d)
        # This function should return None as documented
        assert result is None
    
    def test_memory_error_handling(self, sample_rec_2d):
        """Test memory error handling in compute_cohere."""
        f_s = 1000.0
        
        with patch('pythoneeg.core.analyze_frag.spectral_connectivity_time') as mock_connectivity:
            # Make the connectivity function raise a MemoryError
            mock_connectivity.side_effect = MemoryError("Test memory error")
            
            with pytest.raises(MemoryError, match="Out of memory"):
                FragmentAnalyzer.compute_cohere(sample_rec_2d, f_s=f_s)
    
    @pytest.mark.parametrize("welch_bin_t", [0.5, 1.0, 2.0])
    def test_psd_welch_bin_effect(self, sample_rec_2d, welch_bin_t):
        """Test that different welch_bin_t values produce valid results."""
        f_s = 1000.0
        
        f, psd = FragmentAnalyzer.compute_psd(
            sample_rec_2d, f_s=f_s, welch_bin_t=welch_bin_t, notch_filter=False
        )
        
        # All should produce valid results
        assert len(f) > 0
        assert psd.shape[0] == len(f)
        assert psd.shape[1] == sample_rec_2d.shape[1]
        assert np.all(psd >= 0)
    
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
            np.testing.assert_array_almost_equal(
                psdfrac[band_name], expected_frac, decimal=5
            )
        
        # Check that all fractions sum to 1
        total_frac = sum(psdfrac.values())
        np.testing.assert_array_almost_equal(
            total_frac, np.ones_like(total_frac), decimal=5
        )
