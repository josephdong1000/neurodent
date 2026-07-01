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
        df = war.result.copy()

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

    def test_extract_banded_matrix_features(self, mock_war_with_matrices):
        """Test that banded matrix features are properly extracted into per-band columns"""
        war = mock_war_with_matrices
        df = war.result.copy()

        # Call private method for testing
        df_extracted = war._extract_banded_matrix_features(df, 'zcohere', constants.BAND_NAMES)

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
        df = war.result.copy()

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
        df = war.result.copy()

        # Before averaging: zpcorr should be 2D arrays
        assert isinstance(df['zpcorr'].iloc[0], np.ndarray)
        assert df['zpcorr'].iloc[0].ndim == 2

        # Call private method for testing
        df_averaged = war._average_across_channels(df, ['zpcorr'])

        # After averaging: zpcorr should be scalars
        assert isinstance(df_averaged['zpcorr'].iloc[0], (int, float, np.number))
        assert not isinstance(df_averaged['zpcorr'].iloc[0], np.ndarray)

    def test_simple_matrix_features_not_expanded(self, mock_war_with_simple_matrix):
        """Test that simple matrix features (pcorr, zpcorr) are NOT expanded into bands"""
        war = mock_war_with_simple_matrix

        # Request zpcorr (a simple matrix feature)
        df = war.get_channel_averaged_result(features=['zpcorr'])

        # zpcorr should remain as 'zpcorr', NOT expanded into 'zpcorr_delta', etc.
        assert 'zpcorr' in df.columns, "Simple matrix feature should remain as single column"
        
        # Should NOT have band-expanded columns
        for band in constants.BAND_NAMES:
            assert f'zpcorr_{band}' not in df.columns, \
                f"Simple matrix feature should NOT be expanded into {band} band"

        # Value should be a scalar (upper triangle mean)
        assert isinstance(df['zpcorr'].iloc[0], (int, float, np.number))
        assert not isinstance(df['zpcorr'].iloc[0], np.ndarray)

    def test_get_channel_averaged_result_integration(self, mock_war_full):
        """Integration test: full pipeline from WAR to channel-averaged dataframe"""
        war = mock_war_full

        # Request all feature types
        features = ['logrms', 'zpcorr', 'logpsdband', 'zcohere', 'zimcoh']
        df = war.get_channel_averaged_result(features=features)

        # Check that all expected columns exist
        # Note: zpcorr is a SIMPLE matrix feature (no bands), so it stays as 'zpcorr' (not expanded)
        expected_cols = [
            'timestamp', 'genotype', 'animalday',  # Metadata
            'logrms',  # Linear feature
            'zpcorr',  # Simple matrix feature (no bands - just averaged directly)
            # Band features expanded
            'logpsdband_delta', 'logpsdband_theta', 'logpsdband_alpha', 'logpsdband_beta', 'logpsdband_gamma',
            # Banded matrix features expanded
            'zcohere_delta', 'zcohere_theta', 'zcohere_alpha', 'zcohere_beta', 'zcohere_gamma',
            'zimcoh_delta', 'zimcoh_theta', 'zimcoh_alpha', 'zimcoh_beta', 'zimcoh_gamma',
        ]

        for col in expected_cols:
            assert col in df.columns, f"Expected column {col} not found"

        # Check that all feature values are scalars
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'genotype', 'animalday']]
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

        # Request a feature that exists (logrms) and one that's not in the WAR (zpcorr)
        # zpcorr is a valid feature name, but not present in mock_war_with_linear
        features = ['logrms', 'zpcorr']

        # Should not raise an error, just skip the missing feature
        df = war.get_channel_averaged_result(features=features)

        assert 'logrms' in df.columns
        # zpcorr bands should not be present since zpcorr wasn't in the data
        assert 'zpcorr_delta' not in df.columns

    def test_list_format_linear_features(self, mock_war_with_linear_lists):
        """Test that linear features stored as lists are converted to arrays"""
        war = mock_war_with_linear_lists
        df = war.result.copy()

        # Before averaging: logrms should be lists
        assert isinstance(df['logrms'].iloc[0], list)

        # Call channel averaging - should convert lists to arrays automatically
        df_averaged = war.get_channel_averaged_result(features=['logrms'])

        # After averaging: logrms should be scalars (not lists or arrays)
        assert isinstance(df_averaged['logrms'].iloc[0], (int, float, np.number))
        assert not isinstance(df_averaged['logrms'].iloc[0], (np.ndarray, list))

    def test_list_format_simple_matrix_features(self, mock_war_with_matrix_lists):
        """Test that simple matrix features stored as lists are converted to arrays and averaged"""
        war = mock_war_with_matrix_lists
        df = war.result.copy()

        # Before averaging: zpcorr should be lists (legacy format)
        first_val = df['zpcorr'].iloc[0]
        assert isinstance(first_val, list)

        # Call channel averaging - should handle list format
        # zpcorr is a SIMPLE matrix feature (no bands), so it should NOT be expanded
        df_averaged = war.get_channel_averaged_result(features=['zpcorr'])

        # After averaging: zpcorr should be a scalar (NOT expanded into bands)
        assert 'zpcorr' in df_averaged.columns, "zpcorr should remain as a single column"
        assert 'zpcorr_delta' not in df_averaged.columns, "zpcorr should NOT be expanded into bands"
        assert isinstance(df_averaged['zpcorr'].iloc[0], (int, float, np.number))
        assert not isinstance(df_averaged['zpcorr'].iloc[0], (np.ndarray, list))

    def test_dict_with_list_matrix_features(self, mock_war_with_matrix_dict_lists):
        """Test that banded matrix features with dicts containing lists are handled (THE REAL BUG)"""
        war = mock_war_with_matrix_dict_lists
        df = war.result.copy()

        # Before averaging: zcohere/zimcoh should be dicts with lists inside
        first_zcohere = df['zcohere'].iloc[0]
        first_zimcoh = df['zimcoh'].iloc[0]
        assert isinstance(first_zcohere, dict)
        assert isinstance(first_zimcoh, dict)
        # Check that the values inside the dict are lists, not numpy arrays
        assert isinstance(first_zcohere['delta'], list)
        assert isinstance(first_zimcoh['delta'], list)

        # Call channel averaging - should convert lists inside dicts to arrays
        df_averaged = war.get_channel_averaged_result(features=['zcohere', 'zimcoh'])

        # After averaging: all band columns should exist with non-NaN scalars
        for band in constants.BAND_NAMES:
            zcohere_col = f'zcohere_{band}'
            zimcoh_col = f'zimcoh_{band}'

            assert zcohere_col in df_averaged.columns, f"Expected column {zcohere_col}"
            assert zimcoh_col in df_averaged.columns, f"Expected column {zimcoh_col}"

            # Check values are scalars and NOT NaN (this was the bug!)
            zcohere_val = df_averaged[zcohere_col].iloc[0]
            zimcoh_val = df_averaged[zimcoh_col].iloc[0]

            assert isinstance(zcohere_val, (int, float, np.number)), f"{zcohere_col} should be scalar"
            assert isinstance(zimcoh_val, (int, float, np.number)), f"{zimcoh_col} should be scalar"
            assert not isinstance(zcohere_val, (np.ndarray, list)), f"{zcohere_col} should not be array/list"
            assert not isinstance(zimcoh_val, (np.ndarray, list)), f"{zimcoh_col} should not be array/list"

            # Most importantly: values should NOT be NaN
            assert not np.isnan(zcohere_val), f"{zcohere_col} should not be NaN"
            assert not np.isnan(zimcoh_val), f"{zimcoh_col} should not be NaN"

    def test_banded_matrix_as_2d_array_raises(self, mock_war_with_banded_matrix_as_2d_array):
        """Test that a 2D array for a banded feature raises ValueError (avoiding silent failure)"""
        war = mock_war_with_banded_matrix_as_2d_array

        # Requesting zcohere (stored as 2D array) should now raise ValueError
        with pytest.raises(ValueError, match="is stored as a 2D array, but is defined as a banded feature"):
            war.get_channel_averaged_result(features=["zcohere"])

    def test_banded_matrix_as_3d_array_success(self, mock_war_with_banded_matrix_as_3d_array):
        """Test that a 3D array (Bands, Ch, Ch) is correctly extracted into bands"""
        war = mock_war_with_banded_matrix_as_3d_array

        # Request zcohere (stored as 3D array)
        df_averaged = war.get_channel_averaged_result(features=["zcohere"])

        # It should have expanded it to all bands by indexing the first dimension
        for band in constants.BAND_NAMES:
            col = f"zcohere_{band}"
            assert col in df_averaged.columns
            val = df_averaged[col].iloc[0]
            assert isinstance(val, (int, float, np.number))
            assert not np.isnan(val)

    def test_extract_band_features_non_dict_raises(self, mock_war_with_linear):
        """Test that _extract_band_features raises ValueError for non-dict features"""
        war = mock_war_with_linear
        df = war.result.copy()
        
        # logrms is a linear feature (array), not a dict. Calling _extract_band_features should fail.
        with pytest.raises(ValueError, match="Band feature logrms must be a dictionary"):
            war._extract_band_features(df, 'logrms', constants.BAND_NAMES)

    def test_average_across_channels_inconsistent_shapes_raises(self, mock_war_with_linear):
        """Test that _average_across_channels raises ValueError for inconsistent array shapes"""
        war = mock_war_with_linear
        df = war.result.copy()
        
        # Corrupt one row to have a different shape
        original_shape = df['logrms'].iloc[0].shape
        df.at[df.index[1], 'logrms'] = np.random.randn(original_shape[0] + 2)  # Different length
        
        # Should raise ValueError about inconsistent channel counts
        with pytest.raises(ValueError, match="inconsistent channel counts"):
            war._average_across_channels(df, ['logrms'])

    def test_3d_array_band_count_validation(self, mock_war_with_banded_matrix_as_3d_array):
        """Test that 3D array extraction validates band count matches"""
        war = mock_war_with_banded_matrix_as_3d_array
        df = war.result.copy()
        
        # Corrupt first element to have wrong number of bands
        wrong_bands = np.random.randn(3, 8, 8)  # Only 3 bands, should be 5
        df.at[df.index[0], 'zcohere'] = wrong_bands
        
        # Should raise ValueError about band count mismatch
        with pytest.raises(ValueError, match="has 3 bands, but 5 were expected"):
            war._extract_banded_matrix_features(df, 'zcohere', constants.BAND_NAMES)


# Pytest fixtures to create mock WARs for testing
@pytest.fixture
def mock_war_with_bands():
    """Create a mock WAR with band features (logpsdband)"""
    n_windows = 10
    n_channels = 8

    # Band features are stored as dicts: {'delta': array, 'theta': array, ...}
    band_data = []
    for i in range(n_windows):
        band_dict = {
            band: np.random.randn(n_channels) for band in constants.BAND_NAMES
        }
        band_data.append(band_dict)

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        'logpsdband': band_data,
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_with_matrices():
    """Create a mock WAR with matrix band features (zcohere as dict)"""
    n_windows = 10
    n_channels = 8

    # Matrix band features are stored as dicts: {'delta': 2D_array, 'theta': 2D_array, ...}
    matrix_data = []
    for i in range(n_windows):
        matrix_dict = {
            band: np.random.randn(n_channels, n_channels) for band in constants.BAND_NAMES
        }
        matrix_data.append(matrix_dict)

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        'zcohere': matrix_data,
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_with_linear():
    """Create a mock WAR with linear features (logrms)"""
    n_windows = 10
    n_channels = 8

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        'logrms': [np.random.randn(n_channels) for _ in range(n_windows)],
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_with_simple_matrix():
    """Create a mock WAR with non-banded matrix features (zpcorr as 2D array)"""
    n_windows = 10
    n_channels = 8

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        'zpcorr': [np.random.randn(n_channels, n_channels) for _ in range(n_windows)],
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_full():
    """Create a mock WAR with all feature types"""
    n_windows = 10
    n_channels = 8

    # Band features (dict format)
    band_data = []
    for i in range(n_windows):
        band_dict = {
            band: np.random.randn(n_channels) for band in constants.BAND_NAMES
        }
        band_data.append(band_dict)

    # Matrix band features (dict format)
    matrix_data_zcohere = []
    matrix_data_zimcoh = []
    for i in range(n_windows):
        matrix_dict_zcohere = {
            band: np.random.randn(n_channels, n_channels) for band in constants.BAND_NAMES
        }
        matrix_dict_zimcoh = {
            band: np.random.randn(n_channels, n_channels) for band in constants.BAND_NAMES
        }
        matrix_data_zcohere.append(matrix_dict_zcohere)
        matrix_data_zimcoh.append(matrix_dict_zimcoh)

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        'logrms': [np.random.randn(n_channels) for _ in range(n_windows)],
        'logpsdband': band_data,
        'zpcorr': [np.random.randn(n_channels, n_channels) for _ in range(n_windows)],
        'zcohere': matrix_data_zcohere,
        'zimcoh': matrix_data_zimcoh,
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_with_linear_lists():
    """Create a mock WAR with linear features stored as lists (legacy format)"""
    n_windows = 10
    n_channels = 8

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        # Store as Python lists instead of numpy arrays (legacy format)
        'logrms': [list(np.random.randn(n_channels)) for _ in range(n_windows)],
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_with_matrix_lists():
    """Create a mock WAR with matrix features stored as nested lists (legacy format)"""
    n_windows = 10
    n_channels = 8

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        # Store as nested Python lists instead of numpy arrays (legacy format)
        'zpcorr': [[list(row) for row in np.random.randn(n_channels, n_channels)] for _ in range(n_windows)],
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_with_matrix_dict_lists():
    """Create a mock WAR with banded matrix features as dicts containing lists (real legacy format)"""
    n_windows = 10
    n_channels = 8

    # Create matrix data as dicts with LISTS inside (not numpy arrays)
    # This mimics the actual format in real WAR files that causes the bug
    matrix_data_zcohere = []
    matrix_data_zimcoh = []
    for i in range(n_windows):
        matrix_dict_zcohere = {
            band: [[float(x) for x in row] for row in np.random.randn(n_channels, n_channels)]
            for band in constants.BAND_NAMES
        }
        matrix_dict_zimcoh = {
            band: [[float(x) for x in row] for row in np.random.randn(n_channels, n_channels)]
            for band in constants.BAND_NAMES
        }
        matrix_data_zcohere.append(matrix_dict_zcohere)
        matrix_data_zimcoh.append(matrix_dict_zimcoh)

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_windows, freq='4s'),
        'genotype': ['WT'] * n_windows,
        'animalday': ['A001_1'] * n_windows,
        'zcohere': matrix_data_zcohere,
        'zimcoh': matrix_data_zimcoh,
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id='A001',
        genotype='WT',
        channel_names=['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis'],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={}
    )
    return war


@pytest.fixture
def mock_war_with_banded_matrix_as_2d_array():
    """Create a mock WAR where a banded feature (zcohere) is stored as a 2D array directly (now an error)"""
    n_windows = 10
    n_channels = 8

    data = {
        "timestamp": pd.date_range("2025-01-01", periods=n_windows, freq="4s"),
        "genotype": ["WT"] * n_windows,
        "animalday": ["A001_1"] * n_windows,
        # zcohere is in BANDED_MATRIX_FEATURES, storing as 2D should fail
        "zcohere": [np.random.randn(n_channels, n_channels) for _ in range(n_windows)],
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id="A001",
        genotype="WT",
        channel_names=["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={},
    )
    return war


@pytest.fixture
def mock_war_with_banded_matrix_as_3d_array():
    """Create a mock WAR where a banded feature (zcohere) is stored as 3D array (Bands, Ch, Ch)"""
    n_windows = 10
    n_channels = 8
    n_bands = len(constants.BAND_NAMES)

    data = {
        "timestamp": pd.date_range("2025-01-01", periods=n_windows, freq="4s"),
        "genotype": ["WT"] * n_windows,
        "animalday": ["A001_1"] * n_windows,
        # zcohere as 3D array: (Bands, Channels, Channels)
        "zcohere": [np.random.randn(n_bands, n_channels, n_channels) for _ in range(n_windows)],
    }

    df = pd.DataFrame(data)

    war = visualization.WindowAnalysisResult(
        result=df,
        animal_id="A001",
        genotype="WT",
        channel_names=["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"],
        assume_from_number=True,
        bad_channels_dict={},
        suppress_short_interval_error=True,
        lof_scores_dict={},
    )
    return war


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
