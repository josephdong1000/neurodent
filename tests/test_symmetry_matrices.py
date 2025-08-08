"""
Tests for symmetric matrix behavior in pcorr and cohere functions.

This test module verifies the behavior of correlation and coherence matrices
when lower_triag=False, ensuring they are properly symmetric and that
downstream averaging functions work correctly.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from pythoneeg.core.analysis import LongRecordingAnalyzer
from pythoneeg.core.analyze_frag import FragmentAnalyzer
from pythoneeg.core import core
from pythoneeg import constants


class TestSymmetricMatrices:
    """Test symmetric matrix behavior for correlation and coherence."""

    def test_pcorr_symmetry_consistency(self):
        """Test that FragmentAnalyzer and LongRecordingAnalyzer have consistent defaults."""
        np.random.seed(42)
        signals = np.random.randn(1000, 4).astype(np.float32)
        
        # FragmentAnalyzer default behavior
        fa_default = FragmentAnalyzer.compute_pcorr(signals, constants.GLOBAL_SAMPLING_RATE)
        
        # Create LongRecordingAnalyzer setup
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
        
        analyzer = LongRecordingAnalyzer(mock_long_recording, fragment_len_s=10, notch_freq=60)
        lra_default = analyzer.compute_pcorr(0)
        
        # Both should default to symmetric matrices
        assert np.allclose(fa_default, fa_default.T), "FragmentAnalyzer should produce symmetric matrix by default"
        assert np.allclose(lra_default, lra_default.T), "LongRecordingAnalyzer should produce symmetric matrix by default"
        
        # The lower triangles should be identical (up to numerical precision)
        fa_lower_tri = np.tril(fa_default, k=-1)
        lra_lower_tri = np.tril(lra_default, k=-1)
        np.testing.assert_allclose(fa_lower_tri, lra_lower_tri, atol=1e-10)

    def test_cohere_matrices_are_symmetric(self):
        """Test that coherence matrices are symmetric."""
        np.random.seed(42)
        signals = np.random.randn(2000, 4).astype(np.float32)  # Longer for better coherence estimation
        
        cohere_dict = FragmentAnalyzer.compute_cohere(signals, constants.GLOBAL_SAMPLING_RATE)
        
        for band_name, matrix in cohere_dict.items():
            assert matrix.shape[0] == matrix.shape[1], f"Matrix for {band_name} should be square"
            assert np.allclose(matrix, matrix.T, atol=1e-10), f"Matrix for {band_name} should be symmetric"
            
            # Diagonal should be 1.0 for coherence
            diagonal = np.diag(matrix)
            np.testing.assert_allclose(diagonal, 1.0, atol=1e-6, 
                                       err_msg=f"Diagonal for {band_name} should be 1.0")

    def test_zcohere_matrices_are_symmetric(self):
        """Test that Fisher z-transformed coherence matrices are symmetric."""
        np.random.seed(42)
        signals = np.random.randn(2000, 3).astype(np.float32)
        
        zcohere_dict = FragmentAnalyzer.compute_zcohere(signals, constants.GLOBAL_SAMPLING_RATE)
        
        for band_name, matrix in zcohere_dict.items():
            assert matrix.shape[0] == matrix.shape[1], f"Matrix for {band_name} should be square"
            assert np.allclose(matrix, matrix.T, atol=1e-10), f"Matrix for {band_name} should be symmetric"
            
            # Should not contain inf values (arctanh issue would cause this)
            assert not np.isinf(matrix).any(), f"Matrix for {band_name} should not contain inf values"

    def test_visualization_averaging_equivalence(self):
        """Test that averaging using lower triangle vs all off-diagonal gives same result for symmetric matrices."""
        # Create a test symmetric correlation matrix
        test_matrix = np.array([
            [1.0, 0.8, 0.6, 0.4],
            [0.8, 1.0, 0.5, 0.3], 
            [0.6, 0.5, 1.0, 0.2],
            [0.4, 0.3, 0.2, 1.0]
        ])
        
        # Current visualization logic: lower triangle, excluding diagonal
        tril_indices = np.tril_indices(4, k=-1)
        avg_lower = np.nanmean(test_matrix[tril_indices])
        
        # Alternative: all off-diagonal elements
        off_diag_mask = ~np.eye(4, dtype=bool)
        avg_all_offdiag = np.nanmean(test_matrix[off_diag_mask])
        
        # Should be mathematically equivalent for symmetric matrices
        np.testing.assert_allclose(avg_lower, avg_all_offdiag, atol=1e-15)

    def test_backward_compatibility_lower_triag_true(self):
        """Test that explicit lower_triag=True still works as before."""
        np.random.seed(42)
        signals = np.random.randn(1000, 3).astype(np.float32)
        
        # Test FragmentAnalyzer
        fa_full = FragmentAnalyzer.compute_pcorr(signals, constants.GLOBAL_SAMPLING_RATE, lower_triag=False)
        fa_lower = FragmentAnalyzer.compute_pcorr(signals, constants.GLOBAL_SAMPLING_RATE, lower_triag=True)
        
        # Full matrix should be symmetric
        assert np.allclose(fa_full, fa_full.T)
        
        # Lower triangle matrix should have zero upper triangle
        assert np.allclose(np.triu(fa_lower, k=1), 0)
        
        # Lower triangles should match
        lower_indices = np.tril_indices(3, k=-1)
        np.testing.assert_allclose(fa_full[lower_indices], fa_lower[lower_indices])

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 8])
    def test_symmetry_different_channel_counts(self, n_channels):
        """Test matrix symmetry for different numbers of channels."""
        np.random.seed(42)
        signals = np.random.randn(1000, n_channels).astype(np.float32)
        
        pcorr = FragmentAnalyzer.compute_pcorr(signals, constants.GLOBAL_SAMPLING_RATE, lower_triag=False)
        
        assert pcorr.shape == (n_channels, n_channels)
        assert np.allclose(pcorr, pcorr.T, atol=1e-10), f"Matrix should be symmetric for {n_channels} channels"
        
        # Diagonal should be close to 1.0 (perfect self-correlation)
        diagonal = np.diag(pcorr) 
        np.testing.assert_allclose(diagonal, 1.0, atol=0.1, 
                                   err_msg=f"Diagonal should be ~1.0 for {n_channels} channels")

    def test_no_bias_from_diagonal_inclusion(self):
        """Test that visualization averaging doesn't accidentally include diagonal values."""
        # Create test matrix where diagonal would bias the average
        test_matrix = np.array([
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.1], 
            [0.1, 0.1, 1.0]
        ])
        
        # Current visualization uses k=-1 (excludes diagonal)
        tril_indices = np.tril_indices(3, k=-1)
        avg_no_diag = np.nanmean(test_matrix[tril_indices])
        
        # If diagonal were accidentally included (k=0)
        tril_with_diag = np.tril_indices(3, k=0)
        avg_with_diag = np.nanmean(test_matrix[tril_with_diag])
        
        # Should be different - this confirms k=-1 is correct
        assert abs(avg_no_diag - avg_with_diag) > 0.1, "k=-1 should exclude diagonal and give different result"
        assert np.isclose(avg_no_diag, 0.1), "Without diagonal, average should be 0.1"
        assert avg_with_diag > avg_no_diag, "Including diagonal should increase average"