
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from neurodent.visualization.plotting import ZeitgeberPlotter
from neurodent import constants

@patch("neurodent.visualization.plotting.zeitgeber_plotter.so.Plot")
@patch("neurodent.visualization.plotting.zeitgeber_plotter.mpl.figure.Figure")
@patch("neurodent.visualization.plotting.zeitgeber_plotter.plt.close")
def test_plot_single_feature(mock_close, mock_fig, mock_plot, tmp_path):
    """Verify plot_feature calls plotting commands correctly."""
    df = pd.DataFrame({
        "total_minutes": [0, 60, 120, 180],
        "sex": ["Male", "Male", "Female", "Female"],
        "gene": ["WT", "WT", "WT", "WT"],
        "test_feat": [1, 2, 3, 4]
    })
    
    plotter = ZeitgeberPlotter(df)
    
    # Mock the chain of calls for seaborn objects
    mock_plot_instance = mock_plot.return_value
    mock_plot_instance.facet.return_value = mock_plot_instance
    mock_plot_instance.add.return_value = mock_plot_instance
    mock_plot_instance.layout.return_value = mock_plot_instance
    mock_plot_instance.theme.return_value = mock_plot_instance
    mock_plot_instance.label.return_value = mock_plot_instance
    
    output_path = tmp_path / "test_plot.png"
    
    plotter.plot_feature(feature="test_feat", output_path=output_path, figsize=[10, 10], dpi=100)
    
    # Verify Plot was initialized with our df
    # so.Plot(df, ...)
    args, kwargs = mock_plot.call_args
    pd.testing.assert_frame_equal(args[0], df)
    assert kwargs['x'] == "total_minutes"
    assert kwargs['y'] == "test_feat"
    
    # Verify figure creation
    mock_fig.assert_called_with(figsize=[10, 10])
    
    # Verify savefig was called with the output path
    mock_fig.return_value.savefig.assert_called()
    call_args = mock_fig.return_value.savefig.call_args
    # The first positional argument should be the output path
    assert call_args[0][0] == output_path


def test_get_feature_label_utility():
    """Test the new get_feature_label utility function directly."""
    from neurodent.core import get_feature_label
    
    # Base features
    assert get_feature_label("rms") == "RMS"
    assert get_feature_label("logrms") == "Log(RMS)"
    assert get_feature_label("alphadelta") == "Alpha/Delta Ratio"
    
    # Banded features - should use parentheses
    assert get_feature_label("logpsdband_delta") == "Log Band Power (Delta)"
    assert get_feature_label("zcohere_theta") == "Z-Coherence (Theta)"
    
    # Baseline-subtracted
    assert get_feature_label("logrms_nobase") == "Log(RMS) - Baseline"
    assert get_feature_label("logpsdband_delta_nobase") == "Log Band Power (Delta) - Baseline"
    
    # Unknown feature - should return as-is
    assert get_feature_label("unknown_feature") == "unknown_feature"
    
    # Edge case: alphadelta should NOT be confused with band names
    # (alpha is a band name, but alphadelta is a derived feature, not a banded one)
    assert "Alpha" not in get_feature_label("alphadelta") or "Ratio" in get_feature_label("alphadelta")


def test_zeitgeber_plotter_from_zars():
    """Test ZeitgeberPlotter initialization from list of ZARs."""
    from unittest.mock import MagicMock
    from neurodent.core import zeitgeber
    
    # Create mock ZARs
    mock_zar1 = MagicMock()
    mock_zar1.animal_id = "Animal1"
    mock_zar1.get_channel_averaged_result.return_value = pd.DataFrame({
        "total_minutes": [0, 60, 120],
        "genotype": ["M_WT", "M_WT", "M_WT"],
        "feature1": [1.0, 2.0, 3.0]
    })
    
    mock_zar2 = MagicMock()
    mock_zar2.animal_id = "Animal2"
    mock_zar2.get_channel_averaged_result.return_value = pd.DataFrame({
        "total_minutes": [0, 60, 120],
        "genotype": ["F_Mut", "F_Mut", "F_Mut"],
        "feature1": [4.0, 5.0, 6.0]
    })
    
    # Mock prepare_plot_data to avoid full pipeline
    with patch.object(zeitgeber, "prepare_plot_data") as mock_prep:
        mock_prep.side_effect = lambda df, **kwargs: df  # Return df as-is
        
        plotter = ZeitgeberPlotter([mock_zar1, mock_zar2], features=["feature1"])
        
        # Verify aggregation happened
        assert "animal" in plotter.df.columns
        assert set(plotter.df["animal"].unique()) == {"Animal1", "Animal2"}
        
        # Verify prepare_plot_data was called
        mock_prep.assert_called_once()


def test_zeitgeber_plotter_from_dataframe():
    """Test ZeitgeberPlotter initialization from DataFrame (backward compat)."""
    df = pd.DataFrame({
        "total_minutes": [0, 60, 120],
        "sex": ["Male", "Male", "Male"],
        "gene": ["WT", "WT", "WT"],
        "feature1": [1.0, 2.0, 3.0]
    })
    
    plotter = ZeitgeberPlotter(df)
    
    # Should store df directly
    pd.testing.assert_frame_equal(plotter.df, df)


def test_zeitgeber_plotter_invalid_input():
    """Test ZeitgeberPlotter raises on invalid input."""
    with pytest.raises(ValueError, match="must be a DataFrame or list"):
        ZeitgeberPlotter("invalid_string")
    
    with pytest.raises(ValueError, match="cannot be empty"):
        ZeitgeberPlotter([])

