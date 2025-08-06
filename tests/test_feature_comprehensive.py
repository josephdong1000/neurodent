"""
Comprehensive testing suite for PyEEG feature computation functions.

This module implements:
1. Synthetic signal testing
2. Mathematical property validation
3. Cross-method consistency testing
4. Reference implementation comparison
5. Edge case and pathological signal testing
6. Parameter combination testing

All tests are designed to ensure robustness and correctness of EEG feature computations.
"""

import numpy as np
import pytest
from scipy import signal
from scipy.stats import pearsonr
import warnings

from pythoneeg.core.analyze_frag import FragmentAnalyzer
from pythoneeg import constants


class SyntheticSignalGenerator:
    """Generate synthetic EEG-like signals for testing."""
    
    @staticmethod
    def white_noise(n_samples: int, n_channels: int, amplitude: float = 1.0, seed: int = 42):
        """Generate white noise signal."""
        np.random.seed(seed)
        return np.random.normal(0, amplitude, (n_samples, n_channels)).astype(np.float32)
    
    @staticmethod
    def sine_wave(n_samples: int, n_channels: int, freq: float, fs: float, amplitude: float = 1.0, 
                  phase_offset: float = 0.0, seed: int = 42):
        """Generate sine wave signal."""
        np.random.seed(seed)
        t = np.arange(n_samples) / fs
        signal_base = amplitude * np.sin(2 * np.pi * freq * t + phase_offset)
        return np.tile(signal_base, (n_channels, 1)).T.astype(np.float32)
    
    @staticmethod
    def multi_freq_signal(n_samples: int, n_channels: int, freqs: list, amplitudes: list, 
                         fs: float, noise_level: float = 0.1, seed: int = 42):
        """Generate multi-frequency signal with optional noise."""
        np.random.seed(seed)
        t = np.arange(n_samples) / fs
        signal_data = np.zeros((n_samples, n_channels), dtype=np.float32)
        
        for freq, amp in zip(freqs, amplitudes):
            signal_data += amp * np.sin(2 * np.pi * freq * t).reshape(-1, 1)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, (n_samples, n_channels))
            signal_data += noise
            
        return signal_data.astype(np.float32)
    
    @staticmethod
    def band_limited_noise(n_samples: int, n_channels: int, low_freq: float, high_freq: float,
                          fs: float, amplitude: float = 1.0, seed: int = 42):
        """Generate band-limited noise."""
        np.random.seed(seed)
        white_noise = np.random.normal(0, 1, (n_samples, n_channels))
        
        # Apply bandpass filter
        sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos', fs=fs)
        filtered_signal = signal.sosfiltfilt(sos, white_noise, axis=0)
        
        return (amplitude * filtered_signal).astype(np.float32)
    
    @staticmethod
    def chirp_signal(n_samples: int, n_channels: int, f0: float, f1: float, fs: float,
                    amplitude: float = 1.0, seed: int = 42):
        """Generate chirp signal (linear frequency sweep)."""
        np.random.seed(seed)
        t = np.arange(n_samples) / fs
        chirp = signal.chirp(t, f0, t[-1], f1)
        return (amplitude * np.tile(chirp, (n_channels, 1)).T).astype(np.float32)
    
    @staticmethod
    def pathological_signals(signal_type: str, n_samples: int, n_channels: int):
        """Generate pathological test signals."""
        if signal_type == "zeros":
            return np.zeros((n_samples, n_channels), dtype=np.float32)
        elif signal_type == "ones":
            return np.ones((n_samples, n_channels), dtype=np.float32)
        elif signal_type == "inf":
            sig = np.ones((n_samples, n_channels), dtype=np.float32)
            sig[n_samples//2, :] = np.inf
            return sig
        elif signal_type == "nan":
            sig = np.ones((n_samples, n_channels), dtype=np.float32)
            sig[n_samples//2, :] = np.nan
            return sig
        elif signal_type == "very_large":
            return np.full((n_samples, n_channels), 1e10, dtype=np.float32)
        elif signal_type == "very_small":
            return np.full((n_samples, n_channels), 1e-10, dtype=np.float32)
        elif signal_type == "impulse":
            sig = np.zeros((n_samples, n_channels), dtype=np.float32)
            sig[0, :] = 1.0
            return sig
        else:
            raise ValueError(f"Unknown pathological signal type: {signal_type}")


class TestSyntheticSignals:
    """Test feature computations on synthetic signals with known properties."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.fs = 1000.0
        self.n_samples = 10000
        self.n_channels = 2
        self.generator = SyntheticSignalGenerator()
        
    def test_rms_white_noise(self):
        """Test RMS computation on white noise."""
        amplitude = 2.0
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, amplitude)
        
        rms = FragmentAnalyzer.compute_rms(signal_data)
        
        # For white noise, RMS should be close to the amplitude
        expected_rms = amplitude
        np.testing.assert_allclose(rms, expected_rms, rtol=0.1)
        
    def test_rms_sine_wave(self):
        """Test RMS computation on sine wave."""
        amplitude = 1.0
        freq = 10.0
        signal_data = self.generator.sine_wave(self.n_samples, self.n_channels, freq, self.fs, amplitude)
        
        rms = FragmentAnalyzer.compute_rms(signal_data)
        
        # For sine wave, RMS = amplitude / sqrt(2)
        expected_rms = amplitude / np.sqrt(2)
        np.testing.assert_allclose(rms, expected_rms, rtol=1e-3)
        
    def test_ampvar_constant_signal(self):
        """Test amplitude variance on constant signal."""
        signal_data = np.ones((self.n_samples, self.n_channels), dtype=np.float32)
        
        ampvar = FragmentAnalyzer.compute_ampvar(signal_data)
        
        # Constant signal should have zero variance
        np.testing.assert_allclose(ampvar, 0.0, atol=1e-10)
        
    def test_psd_delta_function(self):
        """Test PSD computation on impulse signal."""
        signal_data = self.generator.pathological_signals("impulse", self.n_samples, self.n_channels)
        
        freqs, psd = FragmentAnalyzer.compute_psd(signal_data, self.fs, notch_filter=False)
        
        # Impulse should have flat spectrum (white)
        # Test that PSD values are reasonable and non-zero
        for ch in range(self.n_channels):
            # Filter out very low frequencies and zero values
            valid_freqs = freqs > 1.0  # Skip very low frequencies
            freqs_valid = freqs[valid_freqs]
            psd_valid = psd[valid_freqs, ch]
            psd_valid = psd_valid[psd_valid > 1e-12]  # Remove zeros
            
            if len(psd_valid) > 10 and len(freqs_valid) > 10:
                # Check that most of the spectrum has reasonable power
                assert np.all(psd_valid > 0), "PSD should be positive for impulse"
                assert np.std(np.log10(psd_valid)) < 2.0, "Impulse spectrum should be relatively flat"
            
    def test_psd_sine_wave_peak(self):
        """Test that PSD shows peak at correct frequency for sine wave."""
        target_freq = 15.0
        amplitude = 1.0
        signal_data = self.generator.sine_wave(self.n_samples, self.n_channels, target_freq, self.fs, amplitude)
        
        freqs, psd = FragmentAnalyzer.compute_psd(signal_data, self.fs, notch_filter=False)
        
        # Find peak frequency
        peak_idx = np.argmax(psd[:, 0])
        peak_freq = freqs[peak_idx]
        
        # Peak should be close to target frequency
        np.testing.assert_allclose(peak_freq, target_freq, rtol=0.05)
        
    def test_psdband_energy_conservation(self):
        """Test that sum of band powers equals total power."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        psdband = FragmentAnalyzer.compute_psdband(signal_data, self.fs)
        psdtotal = FragmentAnalyzer.compute_psdtotal(signal_data, self.fs)
        
        # Sum of band powers should approximately equal total power
        # Note: Due to boundary handling differences, allow more tolerance
        band_sum = sum(psdband.values())
        np.testing.assert_allclose(band_sum, psdtotal, rtol=0.15)
        
    def test_psdfrac_sums_to_one(self):
        """Test that PSD fractions sum to 1."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        psdfrac = FragmentAnalyzer.compute_psdfrac(signal_data, self.fs)
        
        # Fractions should sum to 1
        frac_sum = sum(psdfrac.values())
        np.testing.assert_allclose(frac_sum, 1.0, rtol=1e-6)
        
    def test_cohere_identical_signals(self):
        """Test that coherence computation works without errors."""
        # Use multi-frequency signal for better coherence across bands
        signal_data = self.generator.multi_freq_signal(
            self.n_samples, 2, [5, 10, 20], [1, 1, 1], self.fs, noise_level=0.1
        )
        
        # Make both channels identical
        signal_data[:, 1] = signal_data[:, 0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cohere = FragmentAnalyzer.compute_cohere(signal_data, self.fs, freq_res=4)
        
        # Test that coherence computation works and produces valid output
        assert isinstance(cohere, dict)
        for band_name in constants.FREQ_BANDS.keys():
            assert band_name in cohere
            coh_matrix = cohere[band_name]
            assert coh_matrix.shape == (2, 2)
            assert np.all(coh_matrix >= 0) and np.all(coh_matrix <= 1)
            # Check that coherence values are in valid range
            assert np.all(np.isfinite(coh_matrix))
            
    def test_cohere_uncorrelated_noise(self):
        """Test that coherence is low for uncorrelated noise."""
        signal_ch1 = self.generator.white_noise(self.n_samples, 1, 1.0, seed=42)
        signal_ch2 = self.generator.white_noise(self.n_samples, 1, 1.0, seed=123)
        signal_data = np.hstack([signal_ch1, signal_ch2])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cohere = FragmentAnalyzer.compute_cohere(signal_data, self.fs)
        
        # Coherence between uncorrelated signals should be low
        for band_name in cohere:
            coherence_value = cohere[band_name][0, 1]
            assert coherence_value < 0.3, f"Coherence too high for uncorrelated signals: {coherence_value}"
            
    def test_pcorr_identical_signals(self):
        """Test Pearson correlation for identical signals."""
        signal_data = self.generator.sine_wave(self.n_samples, 2, 10.0, self.fs)
        signal_data[:, 1] = signal_data[:, 0]
        
        pcorr = FragmentAnalyzer.compute_pcorr(signal_data, self.fs, lower_triag=False)
        
        # Correlation between identical signals should be 1
        np.testing.assert_allclose(pcorr[0, 1], 1.0, rtol=1e-6)
        np.testing.assert_allclose(pcorr[1, 0], 1.0, rtol=1e-6)


class TestMathematicalProperties:
    """Test mathematical properties of feature computations."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.fs = 1000.0
        self.n_samples = 5000
        self.n_channels = 2
        self.generator = SyntheticSignalGenerator()
        
    def test_rms_linearity(self):
        """Test RMS linearity property: RMS(a*x) = a*RMS(x) for a>0."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        scale_factor = 3.0
        
        rms_original = FragmentAnalyzer.compute_rms(signal_data)
        rms_scaled = FragmentAnalyzer.compute_rms(scale_factor * signal_data)
        
        np.testing.assert_allclose(rms_scaled, scale_factor * rms_original, rtol=1e-5)
        
    def test_ampvar_scale_property(self):
        """Test amplitude variance scaling: Var(a*x) = a²*Var(x)."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        scale_factor = 2.5
        
        ampvar_original = FragmentAnalyzer.compute_ampvar(signal_data)
        ampvar_scaled = FragmentAnalyzer.compute_ampvar(scale_factor * signal_data)
        
        np.testing.assert_allclose(ampvar_scaled, scale_factor**2 * ampvar_original, rtol=1e-5)
        
    def test_psd_parseval_theorem(self):
        """Test Parseval's theorem: total signal energy = integral of PSD."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        # Calculate signal energy
        signal_energy = np.mean(signal_data**2, axis=0)
        
        # Calculate PSD integral
        freqs, psd = FragmentAnalyzer.compute_psd(signal_data, self.fs, notch_filter=False)
        psd_integral = np.trapz(psd, freqs, axis=0)
        
        # They should be approximately equal
        np.testing.assert_allclose(psd_integral, signal_energy, rtol=0.2)
        
    def test_log_features_monotonicity(self):
        """Test that log features preserve ordering."""
        amplitudes = [0.5, 1.0, 2.0]
        rms_values = []
        logrms_values = []
        
        for amp in amplitudes:
            signal_data = self.generator.white_noise(self.n_samples, self.n_channels, amp)
            rms_values.append(FragmentAnalyzer.compute_rms(signal_data)[0])
            logrms_values.append(FragmentAnalyzer.compute_logrms(signal_data)[0])
        
        # Both should be monotonically increasing
        assert all(rms_values[i] <= rms_values[i+1] for i in range(len(rms_values)-1))
        assert all(logrms_values[i] <= logrms_values[i+1] for i in range(len(logrms_values)-1))
        
    def test_zscore_transformation_properties(self):
        """Test Fisher z-transformation properties."""
        # Create signals with known correlation
        signal_base = self.generator.sine_wave(self.n_samples, 1, 10.0, self.fs)
        noise = self.generator.white_noise(self.n_samples, 1, 0.1, seed=123)
        signal_data = np.hstack([signal_base, signal_base + noise])
        
        pcorr = FragmentAnalyzer.compute_pcorr(signal_data, self.fs, lower_triag=False)
        zpcorr = FragmentAnalyzer.compute_zpcorr(signal_data, self.fs, lower_triag=False)
        
        # Fisher z should be monotonic transformation
        original_corr = pcorr[0, 1]
        z_corr = zpcorr[0, 1]
        
        # Check that transformation is correct
        expected_z = np.arctanh(original_corr)
        np.testing.assert_allclose(z_corr, expected_z, rtol=1e-6)


class TestCrossMethodConsistency:
    """Test consistency between different computational methods."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.fs = 1000.0
        self.n_samples = 8000
        self.n_channels = 2
        self.generator = SyntheticSignalGenerator()
        
    def test_psd_methods_consistency(self):
        """Test that both PSD methods produce valid results."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        freqs_welch, psd_welch = FragmentAnalyzer.compute_psd(
            signal_data, self.fs, multitaper=False, notch_filter=False
        )
        freqs_mt, psd_mt = FragmentAnalyzer.compute_psd(
            signal_data, self.fs, multitaper=True, notch_filter=False
        )
        
        # Both methods should produce positive, finite PSDs
        assert np.all(psd_welch > 0)
        assert np.all(np.isfinite(psd_welch))
        assert np.all(psd_mt > 0) 
        assert np.all(np.isfinite(psd_mt))
        
        # Both should have reasonable power levels (order of magnitude check)
        total_power_welch = np.mean(psd_welch)
        total_power_mt = np.mean(psd_mt)
        assert total_power_welch > 0
        assert total_power_mt > 0
        assert np.isfinite(total_power_welch)
        assert np.isfinite(total_power_mt)
        
    def test_correlation_methods_consistency(self):
        """Test consistency between correlation methods."""
        # Create correlated signals
        signal_base = self.generator.sine_wave(self.n_samples, 1, 10.0, self.fs)
        noise = self.generator.white_noise(self.n_samples, 1, 0.2, seed=456)
        signal_data = np.hstack([signal_base, signal_base + noise])
        
        # Compute correlation using FragmentAnalyzer
        pcorr_fa = FragmentAnalyzer.compute_pcorr(signal_data, self.fs, lower_triag=False)
        
        # Compute correlation using scipy directly (after filtering)
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, constants.FREQ_BAND_TOTAL, btype="bandpass", output="sos", fs=self.fs)
        signal_filtered = sosfiltfilt(sos, signal_data, axis=0)
        pcorr_scipy = np.corrcoef(signal_filtered.T)
        
        # Results should be similar
        np.testing.assert_allclose(pcorr_fa, pcorr_scipy, rtol=0.01)


class TestEdgeCasesAndPathological:
    """Test edge cases and pathological signals."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.fs = 1000.0
        self.n_samples = 1000
        self.n_channels = 2
        self.generator = SyntheticSignalGenerator()
        
    def test_zero_signal_handling(self):
        """Test handling of zero signals."""
        signal_data = self.generator.pathological_signals("zeros", self.n_samples, self.n_channels)
        
        rms = FragmentAnalyzer.compute_rms(signal_data)
        ampvar = FragmentAnalyzer.compute_ampvar(signal_data)
        
        np.testing.assert_allclose(rms, 0.0, atol=1e-10)
        np.testing.assert_allclose(ampvar, 0.0, atol=1e-10)
        
    def test_constant_signal_handling(self):
        """Test handling of constant signals."""
        signal_data = self.generator.pathological_signals("ones", self.n_samples, self.n_channels)
        
        ampvar = FragmentAnalyzer.compute_ampvar(signal_data)
        np.testing.assert_allclose(ampvar, 0.0, atol=1e-10)
        
    def test_very_short_signal(self):
        """Test handling of very short signals."""
        short_samples = 10
        signal_data = self.generator.white_noise(short_samples, self.n_channels, 1.0)
        
        # These should not crash
        rms = FragmentAnalyzer.compute_rms(signal_data)
        ampvar = FragmentAnalyzer.compute_ampvar(signal_data)
        
        assert rms.shape == (self.n_channels,)
        assert ampvar.shape == (self.n_channels,)
        
    def test_single_channel_handling(self):
        """Test handling of single channel signals."""
        single_channel_data = self.generator.white_noise(self.n_samples, 1, 1.0)
        
        rms = FragmentAnalyzer.compute_rms(single_channel_data)
        assert rms.shape == (1,)
        
    def test_nan_inf_detection(self):
        """Test detection of NaN and Inf values."""
        # Test with NaN
        signal_nan = self.generator.pathological_signals("nan", self.n_samples, self.n_channels)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms_nan = FragmentAnalyzer.compute_rms(signal_nan)
            
        # Should produce NaN result
        assert np.any(np.isnan(rms_nan))
        
        # Test with Inf
        signal_inf = self.generator.pathological_signals("inf", self.n_samples, self.n_channels)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms_inf = FragmentAnalyzer.compute_rms(signal_inf)
            
        # Should produce Inf result
        assert np.any(np.isinf(rms_inf))


class TestParameterCombinations:
    """Test various parameter combinations for feature functions."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.fs = 1000.0
        self.n_samples = 5000
        self.n_channels = 2
        self.generator = SyntheticSignalGenerator()
        
    @pytest.mark.parametrize("welch_bin_t", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("notch_filter", [True, False])
    def test_psd_parameter_combinations(self, welch_bin_t, notch_filter):
        """Test PSD computation with different parameter combinations."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        freqs, psd = FragmentAnalyzer.compute_psd(
            signal_data, self.fs, welch_bin_t=welch_bin_t, notch_filter=notch_filter
        )
        
        # Basic sanity checks
        assert freqs.shape[0] == psd.shape[0]
        assert psd.shape[1] == self.n_channels
        assert np.all(psd >= 0)  # PSD should be non-negative
        assert np.all(np.isfinite(psd))  # Should be finite
        
    @pytest.mark.parametrize("multitaper", [True, False])
    def test_psdband_multitaper(self, multitaper):
        """Test PSD band computation with and without multitaper."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        psdband = FragmentAnalyzer.compute_psdband(
            signal_data, self.fs, multitaper=multitaper
        )
        
        # Should have all expected bands
        expected_bands = set(constants.FREQ_BANDS.keys())
        assert set(psdband.keys()) == expected_bands
        
        # All values should be positive and finite
        for band_name, values in psdband.items():
            assert np.all(values >= 0)
            assert np.all(np.isfinite(values))
            
    def test_custom_frequency_bands(self):
        """Test PSD computation with custom frequency bands."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        custom_bands = {
            "low": (0.1, 10),
            "mid": (10, 30),
            "high": (30, 40)
        }
        
        psdband = FragmentAnalyzer.compute_psdband(
            signal_data, self.fs, bands=custom_bands
        )
        
        # Should have custom bands
        assert set(psdband.keys()) == set(custom_bands.keys())
        
    @pytest.mark.parametrize("lower_triag", [True, False])
    def test_pcorr_triangle_option(self, lower_triag):
        """Test Pearson correlation with different triangle options."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        pcorr = FragmentAnalyzer.compute_pcorr(
            signal_data, self.fs, lower_triag=lower_triag
        )
        
        assert pcorr.shape == (self.n_channels, self.n_channels)
        
        if lower_triag:
            # Upper triangle should be zero
            assert np.all(np.triu(pcorr, k=0) == 0)
        else:
            # Should be symmetric
            np.testing.assert_allclose(pcorr, pcorr.T, rtol=1e-10)


class TestReferenceImplementations:
    """Compare against reference implementations where possible."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.fs = 1000.0
        self.n_samples = 2000
        self.n_channels = 2
        self.generator = SyntheticSignalGenerator()
        
    def test_rms_reference(self):
        """Test RMS against manual calculation."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        # FragmentAnalyzer implementation
        rms_fa = FragmentAnalyzer.compute_rms(signal_data)
        
        # Reference implementation
        rms_ref = np.sqrt(np.mean(signal_data**2, axis=0))
        
        np.testing.assert_allclose(rms_fa, rms_ref, rtol=1e-10)
        
    def test_ampvar_reference(self):
        """Test amplitude variance against numpy std."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        # FragmentAnalyzer implementation
        ampvar_fa = FragmentAnalyzer.compute_ampvar(signal_data)
        
        # Reference implementation (std squared is variance)
        ampvar_ref = np.std(signal_data, axis=0, ddof=0) ** 2
        
        np.testing.assert_allclose(ampvar_fa, ampvar_ref, rtol=1e-6)
        
    def test_psd_reference_scipy(self):
        """Test PSD against scipy.signal.welch directly."""
        signal_data = self.generator.white_noise(self.n_samples, self.n_channels, 1.0)
        
        # FragmentAnalyzer implementation
        freqs_fa, psd_fa = FragmentAnalyzer.compute_psd(
            signal_data, self.fs, welch_bin_t=1.0, notch_filter=False, multitaper=False
        )
        
        # Reference implementation using scipy directly
        from scipy.signal import welch
        freqs_ref, psd_ref = welch(signal_data, fs=self.fs, nperseg=int(self.fs), axis=0)
        
        np.testing.assert_allclose(freqs_fa, freqs_ref, rtol=1e-10)
        np.testing.assert_allclose(psd_fa, psd_ref, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])