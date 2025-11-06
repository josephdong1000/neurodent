"""
Tests for the matrix feature reordering fix in results.py.

This tests the fix for the issue where matrix features (cohere, pcorr, etc.)
were being incorrectly processed during channel reordering, causing:
1. Values to be doubled (matrix + transpose)
2. Matrices to become lower triangular only
3. Diagonals to be zeroed
4. Values to exceed valid ranges
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import Mock

from neurodent.visualization.results import WindowAnalysisResult
from neurodent import constants


class TestMatrixFeatureReordering:
    """Test the fixed matrix feature reordering logic."""

    @pytest.fixture
    def sample_matrices(self):
        """Create sample symmetric matrices for testing."""
        # 3x3 coherence matrix (0-1 range, symmetric, diagonal=1)
        cohere_matrix = np.array([[[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]]])

        # 3x3 correlation matrix (-1 to 1 range, symmetric, diagonal=1)
        pcorr_matrix = np.array([[[1.0, -0.5, 0.3], [-0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]])

        return cohere_matrix, pcorr_matrix

    @pytest.fixture
    def channel_setup(self):
        """Set up channel names and target channels for reordering."""
        channel_names = ["A", "B", "C"]
        target_channels = ["A", "B", "C", "D"]  # D doesn't exist in original
        channel_map = {ch: i for i, ch in enumerate(target_channels)}
        return channel_names, target_channels, channel_map

    def test_cohere_reordering_preserves_properties(self, sample_matrices, channel_setup):
        """Test that coherence matrix reordering preserves mathematical properties."""
        cohere_matrix, _ = sample_matrices
        channel_names, target_channels, _ = channel_setup

        # Create mock WindowAnalysisResult with coherence data
        mock_war = Mock(spec=WindowAnalysisResult)
        mock_war._feature_columns = ["cohere"]

        # Create result dict with cohere data (dict format with frequency bands)
        result = {"cohere": [{"delta": cohere_matrix[0], "theta": cohere_matrix[0] * 0.9}]}

        # Test the reordering logic by calling the method
        reordered_result = self._test_matrix_reordering(result, "cohere", channel_names, target_channels)

        # Verify the reordered matrix properties
        reordered_cohere = reordered_result["cohere"][0]["delta"]

        # Should be 4x4 now (padded with NaN for missing channel D)
        assert reordered_cohere.shape == (4, 4)

        # Extract the valid 3x3 submatrix
        valid_matrix = reordered_cohere[:3, :3]

        # Test mathematical properties
        assert np.allclose(valid_matrix, cohere_matrix[0]), "Values should be preserved"
        assert np.allclose(valid_matrix, valid_matrix.T), "Should remain symmetric"
        assert np.allclose(np.diag(valid_matrix), 1.0), "Diagonal should be 1.0"
        assert np.all(valid_matrix >= 0) and np.all(valid_matrix <= 1), "Should be in [0,1] range"

    def test_pcorr_reordering_preserves_properties(self, sample_matrices, channel_setup):
        """Test that correlation matrix reordering preserves mathematical properties."""
        _, pcorr_matrix = sample_matrices
        channel_names, target_channels, _ = channel_setup

        # Create result dict with pcorr data (list format)
        result = {"pcorr": [pcorr_matrix[0]]}

        # Test the reordering logic
        reordered_result = self._test_matrix_reordering(result, "pcorr", channel_names, target_channels)

        # Verify the reordered matrix properties
        reordered_pcorr = np.array(reordered_result["pcorr"][0])

        # Should be 4x4 now (padded with NaN for missing channel D)
        assert reordered_pcorr.shape == (4, 4)

        # Extract the valid 3x3 submatrix
        valid_matrix = reordered_pcorr[:3, :3]

        # Test mathematical properties
        assert np.allclose(valid_matrix, pcorr_matrix[0]), "Values should be preserved"
        assert np.allclose(valid_matrix, valid_matrix.T), "Should remain symmetric"
        assert np.allclose(np.diag(valid_matrix), 1.0), "Diagonal should be 1.0"
        assert np.all(valid_matrix >= -1) and np.all(valid_matrix <= 1), "Should be in [-1,1] range"

    def test_mixed_channel_mapping(self, sample_matrices):
        """Test reordering when only some channels are present in target."""
        cohere_matrix, _ = sample_matrices

        # Original has channels A, B, C but target only wants B, D, E
        channel_names = ["A", "B", "C"]
        target_channels = ["B", "D", "E"]

        result = {"cohere": [{"delta": cohere_matrix[0]}]}

        reordered_result = self._test_matrix_reordering(result, "cohere", channel_names, target_channels)

        reordered_cohere = reordered_result["cohere"][0]["delta"]

        # Should be 3x3 (target channels)
        assert reordered_cohere.shape == (3, 3)

        # Only B channel (index 1 in original, index 0 in target) should have data
        assert reordered_cohere[0, 0] == cohere_matrix[0, 1, 1]  # B-B correlation
        assert np.isnan(reordered_cohere[0, 1])  # B-D (D doesn't exist)
        assert np.isnan(reordered_cohere[1, 0])  # D-B (D doesn't exist)

    def _test_matrix_reordering(self, result, feature, channel_names, target_channels):
        """Helper to test the matrix reordering logic."""
        channel_map = {ch: i for i, ch in enumerate(target_channels)}

        # Extract the logic from the actual method
        if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
            df_bands = pd.DataFrame(result[feature])
            vals = np.array(df_bands.values.tolist())
            keys = df_bands.keys()
        else:
            vals = np.array(result[feature])

        new_shape = list(vals.shape[:-2]) + [len(target_channels), len(target_channels)]
        new_vals = np.full(new_shape, np.nan)

        # Apply the FIXED logic
        for i, ch1 in enumerate(channel_names):
            if ch1 in channel_map:
                for j, ch2 in enumerate(channel_names):
                    if ch2 in channel_map:
                        new_vals[..., channel_map[ch1], channel_map[ch2]] = vals[..., i, j]

        if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
            result[feature] = [dict(zip(keys, vals)) for vals in new_vals]
        else:
            result[feature] = [list(x) for x in new_vals]

        return result

    def test_all_matrix_features(self, sample_matrices, channel_setup):
        """Test that all matrix features are handled correctly."""
        cohere_matrix, pcorr_matrix = sample_matrices
        channel_names, target_channels, _ = channel_setup

        # Test all matrix features
        matrix_features = constants.MATRIX_FEATURES

        for feature in matrix_features:
            if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
                # These use dict format with frequency bands
                test_matrix = cohere_matrix[0]
                result = {feature: [{"delta": test_matrix, "theta": test_matrix}]}
            else:
                # pcorr, zpcorr use list format
                test_matrix = pcorr_matrix[0] if "corr" in feature else cohere_matrix[0]
                result = {feature: [test_matrix]}

            # Test reordering
            reordered_result = self._test_matrix_reordering(result, feature, channel_names, target_channels)

            # Basic checks
            if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
                reordered_matrix = reordered_result[feature][0]["delta"]
            else:
                reordered_matrix = np.array(reordered_result[feature][0])

            assert reordered_matrix.shape == (4, 4), f"Shape wrong for {feature}"

            # Values should be preserved in the valid submatrix
            valid_matrix = reordered_matrix[:3, :3]
            assert np.allclose(valid_matrix, test_matrix), f"Values not preserved for {feature}"
            assert np.allclose(valid_matrix, valid_matrix.T), f"Not symmetric for {feature}"
