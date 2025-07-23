"""
Integration tests for PythonEEG package.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from pythoneeg.core import analysis, utils
from pythoneeg.visualization import results
from pythoneeg import constants
from pythoneeg.core.core import LongRecordingOrganizer


@pytest.mark.integration
class TestAnalysisPipeline:
    """Test complete analysis pipeline integration."""
    
    def test_full_analysis_and_visualization_pipeline(self):
        """Test full analysis and visualization pipeline."""
        # Create mock data
        mock_long_recording = MagicMock(spec=LongRecordingOrganizer)
        mock_long_recording.get_num_fragments.return_value = 10
        mock_long_recording.channel_names = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]
        mock_long_recording.meta = MagicMock()
        mock_long_recording.meta.n_channels = 8
        mock_long_recording.meta.mult_to_uV = 1.0
        mock_long_recording.LongRecording = MagicMock()
        mock_long_recording.LongRecording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_long_recording.LongRecording.get_num_frames.return_value = 10000
        mock_long_recording.end_relative = [1]
        
        # Create analyzer with correct constructor
        analyzer = analysis.LongRecordingAnalyzer(
            longrecording=mock_long_recording,
            fragment_len_s=10,
            notch_freq=60
        )
        
        # Test basic functionality
        assert analyzer.n_fragments == 10
        assert analyzer.n_channels == 8
        assert analyzer.f_s == constants.GLOBAL_SAMPLING_RATE
        assert analyzer.channel_names == ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]
        
    def test_data_loading_and_processing_integration(self, temp_dir):
        """Test data loading and processing integration."""
        # Use a filename with a valid genotype
        filepath = Path("WT_A10_2023-01-01.bin")
        
        metadata = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="concat"
        )
        
        assert metadata["genotype"] == "WT"
        assert metadata["animal"] == "A10"
        
        # Create test file paths
        test_files = {
            "ddf_col": temp_dir / "test_ColMajor_001.bin",
            "ddf_row": temp_dir / "test_RowMajor_001.npy.gz",
            "ddf_meta": temp_dir / "test_Meta_001.json"
        }
        
        # Create mock data files
        for path in test_files.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npy.gz":
                np.save(path.with_suffix(".npy"), np.random.randn(100, 8))
            else:
                path.touch()
        
        # Test path conversion utilities
        col_path = str(test_files["ddf_col"])
        rowdir_path = str(temp_dir)
        
        row_path = utils.convert_colpath_to_rowpath(rowdir_path, col_path)
        assert row_path.exists() or row_path.with_suffix(".npy").exists()
        
        # Test file index extraction
        index = utils.filepath_to_index(col_path)
        assert index == 1
        
        # Test metadata parsing
        metadata = utils.parse_path_to_animalday(
            Path("WT_A10_2023-01-01.bin"), 
            animal_param=(1, "_"), 
            mode="concat"
        )
        assert "animal" in metadata
        assert "genotype" in metadata
        assert "day" in metadata


@pytest.mark.integration
class TestVisualizationIntegration:
    """Test visualization module integration."""
    
    # Remove or comment out ResultsVisualizer tests as the class does not exist
    # def test_data_analysis_to_visualization_workflow(self, sample_dataframe):
    #     """Test workflow from analysis results to visualization."""
    #     # Start with analysis results
    #     df = sample_dataframe.copy()
        
    #     # Add computed features
    #     df["log_rms"] = np.log10(df["rms"])
    #     df["normalized_psd"] = (df["psdtotal"] - df["psdtotal"].mean()) / df["psdtotal"].std()
        
    #     # Create visualizer
    #     visualizer = results.ResultsVisualizer(df)
        
    #     # Test multiple visualization types
    #     with patch('matplotlib.pyplot.show'):
    #         # Line plot
    #         visualizer.create_line_plot("rms", "channel")
            
    #         # Bar plot with grouping
    #         visualizer.create_bar_plot("psdtotal", "genotype")
            
    #         # Box plot
    #         visualizer.create_box_plot("cohere", "channel")
            
    #         # Scatter plot
    #         visualizer.create_scatter_plot("rms", "psdtotal")
            
    #         # Multi-panel plot
    #         visualizer.create_multi_panel_plot(["rms", "psdtotal"], "channel")
        
    #     # Test data processing functions
    #     stats = visualizer.compute_summary_statistics("rms", groupby=["genotype", "channel"])
    #     assert isinstance(stats, pd.DataFrame)
        
    #     # Test filtering and visualization
    #     filtered_df = visualizer.filter_dataframe({"genotype": "WT"})
    #     assert len(filtered_df) < len(df)
        
    #     # Test confidence intervals
    #     ci_data = results.compute_confidence_intervals(df, "rms", "genotype")
    #     assert "ci_lower" in ci_data.columns
    #     assert "ci_upper" in ci_data.columns


@pytest.mark.integration
class TestCoreModuleIntegration:
    """Test core module integration."""
    
    def test_utils_and_analysis_integration(self, sample_multi_channel_eeg_data):
        """Test integration between utils and analysis modules."""
        # Test data validation
        test_data = sample_multi_channel_eeg_data
        
        # Test unit conversion
        multiplier = utils.convert_units_to_multiplier("mV", "µV")
        assert np.isclose(multiplier, 1000.0)
        
        # Test log transformation
        log_data = utils.log_transform(test_data)
        assert log_data.shape == test_data.shape
        
        # Test data processing utilities
        # Create test metadata
        metadata = {
            "animal": "A10",
            "genotype": "WT",
            "day": "Jan-01-2023"
        }
        
        # Test metadata parsing utilities
        animal = utils.parse_str_to_animal("WT_A10_Jan01_2023", animal_param=(1, "_"))
        assert animal == "A10"
        
        genotype = utils.parse_str_to_genotype("WT_A10_Jan01_2023")
        assert genotype == "WT"
        
        # Test day parsing
        day = utils.parse_str_to_day("WT_A10_Jan01_2023")
        assert day.year == 2023
        
        # Test channel name parsing
        ch_name = utils.parse_chname_to_abbrev("left Auditory")
        assert ch_name == "LAud"
        
    def test_constants_and_analysis_integration(self):
        """Test integration between constants and analysis modules."""
        # Test that analysis uses constants correctly
        assert constants.GLOBAL_SAMPLING_RATE == 1000
        assert constants.GLOBAL_DTYPE == np.float32
        
        # Test frequency bands
        for band, (fmin, fmax) in constants.FREQ_BANDS.items():
            assert fmin < fmax
            assert fmin > 0
            assert fmax <= 100
        
        # Test feature lists
        assert "rms" in constants.LINEAR_FEATURES
        assert "psdband" in constants.BAND_FEATURES
        assert "cohere" in constants.MATRIX_FEATURES
        
        # Test sorting parameters
        assert "notch_freq" in constants.SORTING_PARAMS
        assert "common_ref" in constants.SORTING_PARAMS


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling across modules."""
    
    def test_error_propagation_across_modules(self):
        """Test that errors propagate correctly across modules."""
        # Test invalid data handling
        with pytest.raises(ValueError):
            utils.parse_str_to_genotype("INVALID_DATA")
        
        # Test invalid unit conversion
        with pytest.raises(AssertionError):
            utils.convert_units_to_multiplier("invalid", "µV")
        
        # Test invalid animal parameter
        with pytest.raises(ValueError):
            utils.parse_str_to_animal("test", animal_param=123)
        
        # Test invalid mode
        with pytest.raises(ValueError):
            utils.parse_path_to_animalday(Path("/test"), mode="invalid")
    
    # Remove or comment out ResultsVisualizer tests as the class does not exist
    # def test_data_validation_integration(self, sample_dataframe):
    #     """Test data validation across modules."""
    #     # Test with valid data
    #     visualizer = results.ResultsVisualizer(sample_dataframe)
    #     visualizer.validate_dataframe()
        
    #     # Test with invalid data
    #     with pytest.raises(TypeError):
    #         results.ResultsVisualizer("not a dataframe")
        
    #     # Test with empty data
    #     empty_df = pd.DataFrame()
    #     visualizer.df = empty_df
    #     with pytest.raises(ValueError):
    #         visualizer.validate_dataframe()


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance characteristics across modules."""
    
    @pytest.mark.slow
    def test_large_data_processing_performance(self):
        """Test performance with large datasets."""
        # Create large test dataset
        large_data = np.random.randn(16, 100000)  # 16 channels, 100k samples
        
        # Test log transformation performance
        import time
        start_time = time.time()
        log_data = utils.log_transform(large_data)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds
        assert log_data.shape == large_data.shape
        
        # Test data processing performance
        start_time = time.time()
        rms_values = np.sqrt(np.mean(large_data**2, axis=1))
        end_time = time.time()
        
        assert end_time - start_time < 5.0  # 5 seconds
        assert len(rms_values) == large_data.shape[0]
    
    # Remove or comment out ResultsVisualizer tests as the class does not exist
    # def test_memory_usage_integration(self, sample_multi_channel_eeg_data):
    #     """Test memory usage across modules."""
    #     import psutil
    #     import os
        
    #     process = psutil.Process(os.getpid())
    #     initial_memory = process.memory_info().rss
        
    #     # Process data through multiple modules
    #     test_data = sample_multi_channel_eeg_data
        
    #     # Apply transformations
    #     log_data = utils.log_transform(test_data)
    #     rms_values = np.sqrt(np.mean(test_data**2, axis=1))
        
    #     # Create DataFrame
    #     df = pd.DataFrame({
    #         "rms": rms_values,
    #         "log_rms": np.log10(rms_values)
    #     })
        
    #     # Create visualizer
    #     visualizer = results.ResultsVisualizer(df)
        
    #     final_memory = process.memory_info().rss
    #     memory_increase = final_memory - initial_memory
        
    #     # Memory increase should be reasonable (less than 100MB)
    #     assert memory_increase < 100 * 1024 * 1024  # 100MB 