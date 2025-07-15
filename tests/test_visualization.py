"""
Unit tests for pythoneeg.visualization module.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import matplotlib.pyplot as plt

from pythoneeg.visualization import results
from pythoneeg import constants


class TestResultsVisualizer:
    """Test ResultsVisualizer class."""
    
    @pytest.fixture
    def visualizer(self, sample_dataframe):
        """Create a ResultsVisualizer instance for testing."""
        return results.ResultsVisualizer(sample_dataframe)
    
    def test_init(self, sample_dataframe):
        """Test ResultsVisualizer initialization."""
        visualizer = results.ResultsVisualizer(sample_dataframe)
        
        assert visualizer.df is sample_dataframe
        assert isinstance(visualizer.df, pd.DataFrame)
        
    def test_validate_dataframe(self, visualizer):
        """Test DataFrame validation."""
        # Test with valid DataFrame
        visualizer.validate_dataframe()
        
        # Test with empty DataFrame
        visualizer.df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            visualizer.validate_dataframe()
            
    def test_get_unique_values(self, visualizer):
        """Test getting unique values from columns."""
        result = visualizer.get_unique_values("channel")
        
        expected = ["LAud", "RAud"]
        assert set(result) == set(expected)
        
    def test_filter_dataframe(self, visualizer):
        """Test DataFrame filtering."""
        filtered = visualizer.filter_dataframe({"genotype": "WT"})
        
        assert isinstance(filtered, pd.DataFrame)
        assert all(filtered["genotype"] == "WT")
        
    def test_compute_summary_statistics(self, visualizer):
        """Test summary statistics computation."""
        stats = visualizer.compute_summary_statistics("rms", groupby="genotype")
        
        assert isinstance(stats, pd.DataFrame)
        assert "mean" in stats.columns
        assert "std" in stats.columns
        
    def test_create_line_plot(self, visualizer):
        """Test line plot creation."""
        with patch('matplotlib.pyplot.figure') as mock_fig:
            with patch('matplotlib.pyplot.subplot') as mock_subplot:
                with patch('matplotlib.pyplot.plot') as mock_plot:
                    with patch('matplotlib.pyplot.show'):
                        visualizer.create_line_plot("rms", "channel")
                        
                        mock_fig.assert_called()
                        mock_subplot.assert_called()
                        mock_plot.assert_called()
                        
    def test_create_bar_plot(self, visualizer):
        """Test bar plot creation."""
        with patch('matplotlib.pyplot.figure') as mock_fig:
            with patch('matplotlib.pyplot.subplot') as mock_subplot:
                with patch('matplotlib.pyplot.bar') as mock_bar:
                    with patch('matplotlib.pyplot.show'):
                        visualizer.create_bar_plot("rms", "channel")
                        
                        mock_fig.assert_called()
                        mock_subplot.assert_called()
                        mock_bar.assert_called()
                        
    def test_create_box_plot(self, visualizer):
        """Test box plot creation."""
        with patch('matplotlib.pyplot.figure') as mock_fig:
            with patch('matplotlib.pyplot.subplot') as mock_subplot:
                with patch('matplotlib.pyplot.boxplot') as mock_boxplot:
                    with patch('matplotlib.pyplot.show'):
                        visualizer.create_box_plot("rms", "channel")
                        
                        mock_fig.assert_called()
                        mock_subplot.assert_called()
                        mock_boxplot.assert_called()
                        
    def test_create_heatmap(self, visualizer):
        """Test heatmap creation."""
        with patch('matplotlib.pyplot.figure') as mock_fig:
            with patch('matplotlib.pyplot.subplot') as mock_subplot:
                with patch('matplotlib.pyplot.imshow') as mock_imshow:
                    with patch('matplotlib.pyplot.colorbar') as mock_colorbar:
                        with patch('matplotlib.pyplot.show'):
                            visualizer.create_heatmap("cohere")
                            
                            mock_fig.assert_called()
                            mock_subplot.assert_called()
                            mock_imshow.assert_called()
                            mock_colorbar.assert_called()
                            
    def test_create_scatter_plot(self, visualizer):
        """Test scatter plot creation."""
        with patch('matplotlib.pyplot.figure') as mock_fig:
            with patch('matplotlib.pyplot.subplot') as mock_subplot:
                with patch('matplotlib.pyplot.scatter') as mock_scatter:
                    with patch('matplotlib.pyplot.show'):
                        visualizer.create_scatter_plot("rms", "psdtotal")
                        
                        mock_fig.assert_called()
                        mock_subplot.assert_called()
                        mock_scatter.assert_called()
                        
    def test_save_plot(self, visualizer, temp_dir):
        """Test plot saving."""
        output_path = temp_dir / "test_plot.png"
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.figure') as mock_fig:
                visualizer.save_plot(output_path)
                mock_savefig.assert_called_with(output_path, dpi=300, bbox_inches='tight')
                
    def test_create_multi_panel_plot(self, visualizer):
        """Test multi-panel plot creation."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            with patch('matplotlib.pyplot.show'):
                mock_fig, mock_axes = Mock(), [Mock(), Mock()]
                mock_subplots.return_value = (mock_fig, mock_axes)
                
                visualizer.create_multi_panel_plot(["rms", "psdtotal"], "channel")
                
                mock_subplots.assert_called()
                
    def test_add_statistical_annotations(self, visualizer):
        """Test statistical annotation addition."""
        with patch('matplotlib.pyplot.annotate') as mock_annotate:
            visualizer.add_statistical_annotations("rms", "channel")
            mock_annotate.assert_called()
            
    def test_customize_plot_style(self, visualizer):
        """Test plot style customization."""
        with patch('matplotlib.pyplot.style.use') as mock_style:
            visualizer.customize_plot_style("seaborn-v0_8")
            mock_style.assert_called_with("seaborn-v0_8")
            
    def test_create_interactive_plot(self, visualizer):
        """Test interactive plot creation."""
        with patch('plotly.graph_objects.Figure') as mock_fig:
            with patch('plotly.graph_objects.Scatter') as mock_scatter:
                visualizer.create_interactive_plot("rms", "channel")
                mock_fig.assert_called()
                mock_scatter.assert_called()


class TestPlottingFunctions:
    """Test standalone plotting functions."""
    
    def test_plot_eeg_timeseries(self, sample_eeg_data):
        """Test EEG timeseries plotting."""
        with patch('matplotlib.pyplot.plot') as mock_plot:
            with patch('matplotlib.pyplot.show'):
                results.plot_eeg_timeseries(sample_eeg_data, fs=constants.GLOBAL_SAMPLING_RATE)
                mock_plot.assert_called()
                
    def test_plot_spectrogram(self, sample_eeg_data):
        """Test spectrogram plotting."""
        with patch('matplotlib.pyplot.specgram') as mock_specgram:
            with patch('matplotlib.pyplot.show'):
                results.plot_spectrogram(sample_eeg_data, fs=constants.GLOBAL_SAMPLING_RATE)
                mock_specgram.assert_called()
                
    def test_plot_power_spectrum(self, sample_eeg_data):
        """Test power spectrum plotting."""
        with patch('matplotlib.pyplot.plot') as mock_plot:
            with patch('matplotlib.pyplot.show'):
                results.plot_power_spectrum(sample_eeg_data, fs=constants.GLOBAL_SAMPLING_RATE)
                mock_plot.assert_called()
                
    def test_plot_connectivity_matrix(self):
        """Test connectivity matrix plotting."""
        connectivity_matrix = np.random.randn(8, 8)
        
        with patch('matplotlib.pyplot.imshow') as mock_imshow:
            with patch('matplotlib.pyplot.colorbar') as mock_colorbar:
                with patch('matplotlib.pyplot.show'):
                    results.plot_connectivity_matrix(connectivity_matrix)
                    mock_imshow.assert_called()
                    mock_colorbar.assert_called()
                    
    def test_plot_feature_comparison(self, sample_dataframe):
        """Test feature comparison plotting."""
        with patch('matplotlib.pyplot.subplot') as mock_subplot:
            with patch('matplotlib.pyplot.boxplot') as mock_boxplot:
                with patch('matplotlib.pyplot.show'):
                    results.plot_feature_comparison(sample_dataframe, "rms", "channel")
                    mock_subplot.assert_called()
                    mock_boxplot.assert_called()
                    
    def test_plot_time_frequency_analysis(self, sample_eeg_data):
        """Test time-frequency analysis plotting."""
        with patch('matplotlib.pyplot.contourf') as mock_contourf:
            with patch('matplotlib.pyplot.colorbar') as mock_colorbar:
                with patch('matplotlib.pyplot.show'):
                    results.plot_time_frequency_analysis(sample_eeg_data, fs=constants.GLOBAL_SAMPLING_RATE)
                    mock_contourf.assert_called()
                    mock_colorbar.assert_called()


class TestDataProcessingForVisualization:
    """Test data processing functions for visualization."""
    
    def test_prepare_data_for_plotting(self, sample_dataframe):
        """Test data preparation for plotting."""
        result = results.prepare_data_for_plotting(sample_dataframe, "rms", "channel")
        
        assert isinstance(result, dict)
        assert "data" in result
        assert "labels" in result
        
    def test_compute_confidence_intervals(self, sample_dataframe):
        """Test confidence interval computation."""
        result = results.compute_confidence_intervals(sample_dataframe, "rms", "genotype")
        
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns
        
    def test_normalize_data_for_comparison(self, sample_dataframe):
        """Test data normalization for comparison."""
        result = results.normalize_data_for_comparison(sample_dataframe, "rms")
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_dataframe)
        
    def test_create_summary_table(self, sample_dataframe):
        """Test summary table creation."""
        result = results.create_summary_table(sample_dataframe, "rms", ["genotype", "channel"])
        
        assert isinstance(result, pd.DataFrame)
        assert "count" in result.columns
        assert "mean" in result.columns
        assert "std" in result.columns


class TestPlotCustomization:
    """Test plot customization functions."""
    
    def test_set_plot_style(self):
        """Test plot style setting."""
        with patch('matplotlib.pyplot.style.use') as mock_style:
            results.set_plot_style("seaborn-v0_8")
            mock_style.assert_called_with("seaborn-v0_8")
            
    def test_customize_colors(self):
        """Test color customization."""
        colors = results.customize_colors(["WT", "KO"])
        
        assert isinstance(colors, dict)
        assert "WT" in colors
        assert "KO" in colors
        assert all(isinstance(color, str) for color in colors.values())
        
    def test_set_figure_size(self):
        """Test figure size setting."""
        with patch('matplotlib.pyplot.figure') as mock_fig:
            results.set_figure_size(width=10, height=8)
            mock_fig.assert_called_with(figsize=(10, 8))
            
    def test_add_plot_labels(self):
        """Test plot label addition."""
        with patch('matplotlib.pyplot.xlabel') as mock_xlabel:
            with patch('matplotlib.pyplot.ylabel') as mock_ylabel:
                with patch('matplotlib.pyplot.title') as mock_title:
                    results.add_plot_labels("X Label", "Y Label", "Title")
                    mock_xlabel.assert_called_with("X Label")
                    mock_ylabel.assert_called_with("Y Label")
                    mock_title.assert_called_with("Title")


class TestExportFunctions:
    """Test export functions."""
    
    def test_export_to_pdf(self, sample_dataframe, temp_dir):
        """Test PDF export."""
        output_path = temp_dir / "test_report.pdf"
        
        with patch('matplotlib.backends.backend_pdf.PdfPages') as mock_pdf:
            with patch('matplotlib.pyplot.figure') as mock_fig:
                results.export_to_pdf(sample_dataframe, output_path)
                mock_pdf.assert_called_with(output_path)
                
    def test_export_to_html(self, sample_dataframe, temp_dir):
        """Test HTML export."""
        output_path = temp_dir / "test_report.html"
        
        with patch('plotly.graph_objects.Figure.write_html') as mock_write_html:
            results.export_to_html(sample_dataframe, output_path)
            mock_write_html.assert_called_with(output_path)
            
    def test_create_report(self, sample_dataframe, temp_dir):
        """Test report creation."""
        output_path = temp_dir / "test_report"
        
        with patch('matplotlib.pyplot.figure') as mock_fig:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                results.create_report(sample_dataframe, output_path)
                mock_savefig.assert_called()


class TestErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_invalid_dataframe_type(self):
        """Test handling of invalid DataFrame type."""
        with pytest.raises(TypeError, match="DataFrame expected"):
            results.ResultsVisualizer("not a dataframe")
            
    def test_missing_column(self, sample_dataframe):
        """Test handling of missing columns."""
        visualizer = results.ResultsVisualizer(sample_dataframe)
        
        with pytest.raises(ValueError, match="Column 'missing_column' not found"):
            visualizer.get_unique_values("missing_column")
            
    def test_empty_data_after_filtering(self, sample_dataframe):
        """Test handling of empty data after filtering."""
        visualizer = results.ResultsVisualizer(sample_dataframe)
        
        # Filter to get empty DataFrame
        filtered = visualizer.filter_dataframe({"genotype": "INVALID"})
        
        with pytest.raises(ValueError, match="No data remaining after filtering"):
            visualizer.create_line_plot("rms", "channel", data=filtered)
            
    def test_invalid_plot_type(self, sample_dataframe):
        """Test handling of invalid plot type."""
        visualizer = results.ResultsVisualizer(sample_dataframe)
        
        with pytest.raises(ValueError, match="Invalid plot type"):
            visualizer.create_plot("invalid_type", "rms", "channel") 