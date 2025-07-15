"""
Unit tests for pythoneeg.core.analysis module.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pythoneeg.core import analysis
from pythoneeg import constants


class TestLongRecordingAnalyzer:
    """Test LongRecordingAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self, sample_multi_channel_eeg_data, sample_metadata):
        """Create a LongRecordingAnalyzer instance for testing."""
        with patch('pythoneeg.core.analysis.LongRecordingOrganizer'):
            analyzer = analysis.LongRecordingAnalyzer(
                data_path="/fake/path",
                animal_param=(1, "_"),
                day_sep="_"
            )
            analyzer.data = sample_multi_channel_eeg_data
            analyzer.metadata = sample_metadata
            analyzer.sampling_rate = constants.GLOBAL_SAMPLING_RATE
            return analyzer
    
    def test_init(self):
        """Test LongRecordingAnalyzer initialization."""
        with patch('pythoneeg.core.analysis.LongRecordingOrganizer'):
            analyzer = analysis.LongRecordingAnalyzer(
                data_path="/fake/path",
                animal_param=(1, "_"),
                day_sep="_"
            )
            
            assert analyzer.data_path == "/fake/path"
            assert analyzer.animal_param == (1, "_")
            assert analyzer.day_sep == "_"
            
    def test_load_data(self, analyzer):
        """Test data loading functionality."""
        with patch.object(analyzer, '_load_eeg_data') as mock_load:
            mock_load.return_value = np.random.randn(1000, 8)
            
            analyzer.load_data()
            
            mock_load.assert_called_once()
            assert analyzer.data is not None
            
    def test_compute_rms(self, analyzer):
        """Test RMS computation."""
        # Create test data
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        analyzer.data = test_data
        
        result = analyzer.compute_rms()
        
        # RMS should be computed for each channel
        expected = np.sqrt(np.mean(test_data**2, axis=1))
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_compute_psd(self, analyzer):
        """Test PSD computation."""
        # Create test data with known frequency components
        fs = constants.GLOBAL_SAMPLING_RATE
        t = np.linspace(0, 1, fs)
        test_data = np.array([
            np.sin(2 * np.pi * 10 * t),  # 10 Hz component
            np.sin(2 * np.pi * 20 * t),  # 20 Hz component
        ])
        analyzer.data = test_data
        analyzer.sampling_rate = fs
        
        result = analyzer.compute_psd()
        
        # Should return PSD for each channel
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == test_data.shape[0]
        
    def test_compute_band_power(self, analyzer):
        """Test band power computation."""
        # Create test data
        fs = constants.GLOBAL_SAMPLING_RATE
        t = np.linspace(0, 1, fs)
        test_data = np.array([
            np.sin(2 * np.pi * 10 * t),  # Alpha band
            np.sin(2 * np.pi * 30 * t),  # Gamma band
        ])
        analyzer.data = test_data
        analyzer.sampling_rate = fs
        
        result = analyzer.compute_band_power()
        
        # Should return band power for each channel and band
        assert isinstance(result, dict)
        assert "delta" in result
        assert "theta" in result
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result
        
    def test_compute_coherence(self, analyzer):
        """Test coherence computation."""
        # Create test data
        test_data = np.random.randn(8, 1000)
        analyzer.data = test_data
        
        result = analyzer.compute_coherence()
        
        # Should return coherence matrix
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == test_data.shape[0]
        assert result.shape[1] == test_data.shape[0]
        
    def test_compute_correlation(self, analyzer):
        """Test correlation computation."""
        # Create test data
        test_data = np.random.randn(8, 1000)
        analyzer.data = test_data
        
        result = analyzer.compute_correlation()
        
        # Should return correlation matrix
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == test_data.shape[0]
        assert result.shape[1] == test_data.shape[0]
        
    def test_extract_features(self, analyzer):
        """Test feature extraction."""
        # Create test data
        test_data = np.random.randn(8, 1000)
        analyzer.data = test_data
        
        result = analyzer.extract_features()
        
        # Should return dictionary of features
        assert isinstance(result, dict)
        assert "rms" in result
        assert "psdtotal" in result
        assert "cohere" in result
        
    def test_save_results(self, analyzer, temp_dir):
        """Test results saving."""
        # Create test results
        test_results = {
            "rms": np.random.randn(8),
            "psdtotal": np.random.randn(8),
            "metadata": {"test": "data"}
        }
        
        output_path = temp_dir / "test_results.npz"
        
        with patch('numpy.savez') as mock_savez:
            analyzer.save_results(test_results, output_path)
            mock_savez.assert_called_once()
            
    def test_load_results(self, analyzer, temp_dir):
        """Test results loading."""
        output_path = temp_dir / "test_results.npz"
        
        # Create mock data
        mock_data = {
            "rms": np.random.randn(8),
            "psdtotal": np.random.randn(8),
            "metadata": np.array([{"test": "data"}], dtype=object)
        }
        
        with patch('numpy.load') as mock_load:
            mock_load.return_value = Mock()
            mock_load.return_value.__getitem__ = lambda self, key: mock_data[key]
            mock_load.return_value.files = list(mock_data.keys())
            
            result = analyzer.load_results(output_path)
            
            assert isinstance(result, dict)
            assert "rms" in result
            assert "psdtotal" in result
            
    def test_validate_data(self, analyzer):
        """Test data validation."""
        # Test with valid data
        test_data = np.random.randn(8, 1000)
        analyzer.data = test_data
        
        # Should not raise any exceptions
        analyzer.validate_data()
        
        # Test with invalid data
        analyzer.data = None
        with pytest.raises(ValueError, match="Data not loaded"):
            analyzer.validate_data()
            
    def test_get_channel_names(self, analyzer):
        """Test channel name generation."""
        analyzer.data = np.random.randn(8, 1000)
        
        result = analyzer.get_channel_names()
        
        assert len(result) == 8
        assert all(isinstance(name, str) for name in result)
        
    def test_get_recording_duration(self, analyzer):
        """Test recording duration calculation."""
        fs = constants.GLOBAL_SAMPLING_RATE
        n_samples = 5000
        test_data = np.random.randn(8, n_samples)
        analyzer.data = test_data
        analyzer.sampling_rate = fs
        
        result = analyzer.get_recording_duration()
        
        expected = n_samples / fs
        assert result == expected
        
    def test_compute_statistics(self, analyzer):
        """Test statistical computation."""
        test_data = np.random.randn(8, 1000)
        analyzer.data = test_data
        
        result = analyzer.compute_statistics()
        
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        
    @pytest.mark.slow
    def test_full_analysis_pipeline(self, analyzer):
        """Test complete analysis pipeline."""
        # Create realistic test data
        fs = constants.GLOBAL_SAMPLING_RATE
        duration = 5  # seconds
        t = np.linspace(0, duration, duration * fs)
        
        # Create multi-channel data with different frequency components
        test_data = np.zeros((8, len(t)))
        for ch in range(8):
            freq = 2 + ch * 3  # Different frequency per channel
            test_data[ch] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        analyzer.data = test_data
        analyzer.sampling_rate = fs
        
        # Run full analysis
        results = analyzer.run_analysis()
        
        # Check that all expected features are present
        expected_features = ["rms", "psdtotal", "psdslope", "psdband", "cohere", "pcorr"]
        for feature in expected_features:
            assert feature in results
            
    def test_error_handling_invalid_path(self):
        """Test error handling for invalid data path."""
        with pytest.raises(FileNotFoundError):
            analysis.LongRecordingAnalyzer(
                data_path="/nonexistent/path",
                animal_param=(1, "_"),
                day_sep="_"
            )
            
    def test_error_handling_invalid_parameters(self, analyzer):
        """Test error handling for invalid parameters."""
        # Test with invalid sampling rate
        analyzer.sampling_rate = 0
        with pytest.raises(ValueError, match="Invalid sampling rate"):
            analyzer.validate_data()
            
        # Test with invalid data shape
        analyzer.sampling_rate = constants.GLOBAL_SAMPLING_RATE
        analyzer.data = np.random.randn(1000)  # 1D instead of 2D
        with pytest.raises(ValueError, match="Data must be 2D"):
            analyzer.validate_data()


class TestAnalysisUtilities:
    """Test analysis utility functions."""
    
    def test_compute_spectral_features(self):
        """Test spectral feature computation."""
        # Create test signal
        fs = constants.GLOBAL_SAMPLING_RATE
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        
        result = analysis.compute_spectral_features(signal, fs)
        
        assert isinstance(result, dict)
        assert "psd" in result
        assert "frequencies" in result
        
    def test_compute_connectivity_features(self):
        """Test connectivity feature computation."""
        # Create test multi-channel data
        test_data = np.random.randn(8, 1000)
        
        result = analysis.compute_connectivity_features(test_data)
        
        assert isinstance(result, dict)
        assert "coherence" in result
        assert "correlation" in result
        
    def test_compute_time_domain_features(self):
        """Test time domain feature computation."""
        # Create test signal
        signal = np.random.randn(1000)
        
        result = analysis.compute_time_domain_features(signal)
        
        assert isinstance(result, dict)
        assert "rms" in result
        assert "variance" in result
        assert "peak_to_peak" in result
        
    def test_normalize_features(self):
        """Test feature normalization."""
        # Create test features
        features = {
            "rms": np.random.randn(10),
            "psdtotal": np.random.randn(10),
            "cohere": np.random.randn(10, 10)
        }
        
        result = analysis.normalize_features(features)
        
        assert isinstance(result, dict)
        assert "rms" in result
        assert "psdtotal" in result
        assert "cohere" in result
        
        # Check that features are normalized
        for key, value in result.items():
            if value.ndim == 1:
                assert np.std(value) > 0  # Should have some variance
            else:
                assert np.std(value) > 0  # Should have some variance 