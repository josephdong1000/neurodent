"""Tests for channel averaging functionality in WindowAnalysisResult"""
import numpy as np
import pandas as pd
import pytest
from neurodent import visualization, constants


class TestChannelAveraging:
    """Test the get_channel_averaged_result() method and its helper functions"""

    def test_extract_band_features(self, mock_war_with_bands):
        """Test that band features are properly extracted into separate columns"""
        war = mock_war_with_bands
        df = war.get_windowed_result()

        # Call private method for testing
        df_extracted = war._extract_band_features(df, 'logpsdband', constants.BAND_NAMES)

        # Check that band columns were created
        assert 'logpsdband_delta' in df_extracted.columns
        assert 'logpsdband_theta' in df_extracted.columns
        assert 'logpsdband_alpha' in df_extracted.columns
        assert 'logpsdband_beta' in df_extracted.columns
        assert 'logpsdband_gamma' in df_extracted.columns

        # Check that each band column contains arrays (not yet averaged)
        first_val = df_extracted['logpsdband_delta'].iloc[0]
        assert isinstance(first_val, np.ndarray)
        assert first_val.ndim == 1  # Vector across channels

    def test_extract_matrix_features(self, mock_war_with_matrices):
        """Test that matrix features are properly extracted into per-band columns"""
        war = mock_war_with_matrices
        df = war.get_windowed_result()

        # Call private method for testing
        df_extracted = war._extract_matrix_features(df, 'zcohere', constants.BAND_NAMES)

        # Check that band columns were created
        assert 'zcohere_delta' in df_extracted.columns
        assert 'zcohere_theta' in df_extracted.columns
        assert 'zcohere_alpha' in df_extracted.columns
        assert 'zcohere_beta' in df_extracted.columns
        assert 'zcohere_gamma' in df_extracted.columns

        # Check that each band column contains 2D matrices (not yet averaged)
        first_val = df_extracted['zcohere_delta'].iloc[0]
        assert isinstance(first_val, np.ndarray)
        assert first_val.ndim == 2  # Matrix across channels

    def test_average_across_channels_linear(self, mock_war_with_linear):
        """Test that linear features are averaged to scalars"""
        war = mock_war_with_linear
        df = war.get_windowed_result()

        # Before averaging: logrms should be arrays
        assert isinstance(df['logrms'].iloc[0], np.ndarray)

        # Call private method for testing
        df_averaged = war._average_across_channels(df, ['logrms'])

        # After averaging: logrms should be scalars
        assert isinstance(df_averaged['logrms'].iloc[0], (int, float, np.number))
        assert not isinstance(df_averaged['logrms'].iloc[0], np.ndarray)

    def test_average_across_channels_matrix(self, mock_war_with_simple_matrix):
        """Test that matrix features are averaged using upper triangle"""
        war = mock_war_with_simple_matrix
        df = war.get_windowed_result()

        # Before averaging: zpcorr should be 2D arrays
        assert isinstance(df['zpcorr'].iloc[0], np.ndarray)
        assert df['zpcorr'].iloc[0].ndim == 2

        # Call private method for testing
        df_averaged = war._average_across_channels(df, ['zpcorr'])

        # After averaging: zpcorr should be scalars
        assert isinstance(df_averaged['zpcorr'].iloc[0], (int, float, np.number))
        assert not isinstance(df_averaged['zpcorr'].iloc[0], np.ndarray)

    def test_get_channel_averaged_result_integration(self, mock_war_full):
        """Integration test: full pipeline from WAR to channel-averaged dataframe"""
        war = mock_war_full

        # Request all feature types
        features = ['logrms', 'zpcorr', 'logpsdband', 'zcohere', 'zimcoh']
        df = war.get_channel_averaged_result(features=features)

        # Check that all expected columns exist
        expected_cols = [
            'timestamp', 'genotype',  # Metadata
            'logrms',  # Linear feature
            'zpcorr',  # Matrix feature (non-banded)
            # Band features expanded
            'logpsdband_delta', 'logpsdband_theta', 'logpsdband_alpha', 'logpsdband_beta', 'logpsdband_gamma',
            # Matrix band features expanded
            'zcohere_delta', 'zcohere_theta', 'zcohere_alpha', 'zcohere_beta', 'zcohere_gamma',
            'zimcoh_delta', 'zimcoh_theta', 'zimcoh_alpha', 'zimcoh_beta', 'zimcoh_gamma',
        ]

        for col in expected_cols:
            assert col in df.columns, f"Expected column {col} not found"

        # Check that all feature values are scalars
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'genotype']]
        for col in feature_cols:
            first_val = df[col].iloc[0]
            assert isinstance(first_val, (int, float, np.number)), \
                f"Column {col} should contain scalars, got {type(first_val)}"
            assert not isinstance(first_val, np.ndarray), \
                f"Column {col} should not contain arrays"

    def test_exclude_features(self, mock_war_full):
        """Test that exclude parameter properly filters features"""
        war = mock_war_full

        features = ['logrms', 'zpcorr', 'logpsdband']
        exclude = ['zpcorr']

        df = war.get_channel_averaged_result(features=features, exclude=exclude)

        # zpcorr should not be in the result
        assert 'zpcorr' not in df.columns

        # logrms and logpsdband should be present
        assert 'logrms' in df.columns
        assert 'logpsdband_delta' in df.columns

    def test_missing_feature_handling(self, mock_war_with_linear):
        """Test that requesting missing features is handled gracefully"""
        war = mock_war_with_linear

        # Request a feature that doesn't exist
        features = ['logrms', 'nonexistent_feature']

        # Should not raise an error, just skip the missing feature
        df = war.get_channel_averaged_result(features=features)

        assert 'logrms' in df.columns
        assert 'nonexistent_feature' not in df.columns


# Pytest fixtures to create mock WARs for testing
@pytest.fixture
def mock_war_with_bands():
    """Create a mock WAR with band features (logpsdband)"""
    # TODO: Implement mock WAR creation
    # This would require creating a minimal WindowAnalysisResult object
    # with synthetic data for testing
    pytest.skip("Mock WAR creation not yet implemented")


@pytest.fixture
def mock_war_with_matrices():
    """Create a mock WAR with matrix band features (zcohere)"""
    pytest.skip("Mock WAR creation not yet implemented")


@pytest.fixture
def mock_war_with_linear():
    """Create a mock WAR with linear features (logrms)"""
    pytest.skip("Mock WAR creation not yet implemented")


@pytest.fixture
def mock_war_with_simple_matrix():
    """Create a mock WAR with non-banded matrix features (zpcorr)"""
    pytest.skip("Mock WAR creation not yet implemented")


@pytest.fixture
def mock_war_full():
    """Create a mock WAR with all feature types"""
    pytest.skip("Mock WAR creation not yet implemented")


class TestZeitgeberPipelineIntegration:
    """Test the zeitgeber pipeline integration with the new method"""

    @pytest.mark.integration
    def test_zeitgeber_feature_extraction_script(self):
        """Test that the zeitgeber script works with real data"""
        # This would require running the actual snakemake pipeline
        # or at least loading real WAR files
        pytest.skip("Integration test requires real data")

    @pytest.mark.integration
    def test_zeitgeber_plots_generation(self):
        """Test that zeitgeber plots are generated correctly"""
        pytest.skip("Integration test requires real data")


class TestBackwardCompatibility:
    """Test that old functionality still works"""

    def test_experiment_plotter_collapse_channels(self):
        """Test that ExperimentPlotter.pull_timeseries_dataframe still works"""
        pytest.skip("Requires mock ExperimentPlotter setup")

    def test_get_windowed_result_unchanged(self):
        """Test that get_windowed_result() without averaging still works"""
        pytest.skip("Requires mock WAR setup")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
