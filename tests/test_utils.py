"""
Unit tests for pythoneeg.core.utils module.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock

from pythoneeg.core import utils
from pythoneeg import constants


class TestConvertUnitsToMultiplier:
    """Test convert_units_to_multiplier function."""
    
    def test_valid_conversions(self):
        """Test valid unit conversions."""
        # Test basic conversions
        assert utils.convert_units_to_multiplier("µV", "µV") == 1.0
        assert utils.convert_units_to_multiplier("mV", "µV") == 1000.0
        assert utils.convert_units_to_multiplier("V", "µV") == 1000000.0
        assert utils.convert_units_to_multiplier("nV", "µV") == 0.001
        
    def test_invalid_current_unit(self):
        """Test error handling for invalid current unit."""
        with pytest.raises(AssertionError, match="No valid current unit"):
            utils.convert_units_to_multiplier("invalid", "µV")
            
    def test_invalid_target_unit(self):
        """Test error handling for invalid target unit."""
        with pytest.raises(AssertionError, match="No valid target unit"):
            utils.convert_units_to_multiplier("µV", "invalid")


class TestIsDay:
    """Test is_day function."""
    
    def test_day_time(self):
        """Test daytime hours."""
        dt = datetime(2023, 1, 1, 12, 0)  # Noon
        assert utils.is_day(dt) is True
        
    def test_night_time(self):
        """Test nighttime hours."""
        dt = datetime(2023, 1, 1, 2, 0)  # 2 AM
        assert utils.is_day(dt) is False
        
    def test_sunrise_edge(self):
        """Test sunrise edge case."""
        dt = datetime(2023, 1, 1, 6, 0)  # 6 AM (sunrise)
        assert utils.is_day(dt) is True
        
    def test_sunset_edge(self):
        """Test sunset edge case."""
        dt = datetime(2023, 1, 1, 18, 0)  # 6 PM (sunset)
        assert utils.is_day(dt) is False
        
    def test_custom_hours(self):
        """Test custom sunrise/sunset hours."""
        dt = datetime(2023, 1, 1, 10, 0)  # 10 AM
        assert utils.is_day(dt, sunrise=8, sunset=20) is True
        assert utils.is_day(dt, sunrise=12, sunset=14) is False


class TestConvertColpathToRowpath:
    """Test convert_colpath_to_rowpath function."""
    
    def test_basic_conversion(self):
        """Test basic path conversion."""
        col_path = "/path/to/data_ColMajor_001.bin"
        rowdir_path = "/output/dir"
        
        result = utils.convert_colpath_to_rowpath(rowdir_path, col_path)
        expected = Path("/output/dir/data_RowMajor_001.npy.gz")
        assert result == expected
        
    def test_without_gzip(self):
        """Test conversion without gzip compression."""
        col_path = "/path/to/data_ColMajor_001.bin"
        rowdir_path = "/output/dir"
        
        result = utils.convert_colpath_to_rowpath(rowdir_path, col_path, gzip=False)
        expected = Path("/output/dir/data_RowMajor_001.bin")
        assert result == expected
        
    def test_as_string(self):
        """Test conversion returning string."""
        col_path = "/path/to/data_ColMajor_001.bin"
        rowdir_path = "/output/dir"
        
        result = utils.convert_colpath_to_rowpath(rowdir_path, col_path, aspath=False)
        expected = "/output/dir/data_RowMajor_001.npy.gz"
        assert result == expected


class TestFilepathToIndex:
    """Test filepath_to_index function."""
    
    def test_basic_extraction(self):
        """Test basic index extraction."""
        filepath = "/path/to/data_ColMajor_001.bin"
        assert utils.filepath_to_index(filepath) == 1
        
    def test_with_different_suffixes(self):
        """Test with different file suffixes."""
        filepath = "/path/to/data_RowMajor_005.npy.gz"
        assert utils.filepath_to_index(filepath) == 5
        
    def test_with_meta_suffix(self):
        """Test with meta suffix."""
        filepath = "/path/to/data_Meta_010.json"
        assert utils.filepath_to_index(filepath) == 10
        
    def test_with_multiple_numbers(self):
        """Test with multiple numbers in filename."""
        filepath = "/path/to/data_2023_ColMajor_015.bin"
        assert utils.filepath_to_index(filepath) == 15


class TestParseTruncate:
    """Test parse_truncate function."""
    
    def test_boolean_true(self):
        """Test boolean True input."""
        assert utils.parse_truncate(True) == 10
        
    def test_boolean_false(self):
        """Test boolean False input."""
        assert utils.parse_truncate(False) == 0
        
    def test_integer_input(self):
        """Test integer input."""
        assert utils.parse_truncate(5) == 5
        assert utils.parse_truncate(0) == 0
        
    def test_invalid_input(self):
        """Test invalid input type."""
        with pytest.raises(ValueError, match="Invalid truncate value"):
            utils.parse_truncate("invalid")


class TestNanaverage:
    """Test nanaverage function."""
    
    def test_basic_averaging(self):
        """Test basic averaging with weights."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = np.array([0.5, 0.3, 0.2])
        
        result = utils.nanaverage(A, weights, axis=0)
        expected = np.average(A, weights=weights, axis=0)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_with_nans(self):
        """Test averaging with NaN values."""
        A = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, np.nan]])
        weights = np.array([0.5, 0.3, 0.2])
        
        result = utils.nanaverage(A, weights, axis=0)
        # Should handle NaN values appropriately
        assert not np.any(np.isnan(result[0]))  # First column has no NaN
        assert np.isnan(result[1])  # Second column has NaN
        assert not np.any(np.isnan(result[2]))  # Third column has NaN but should be handled


class TestParsePathToAnimalday:
    """Test parse_path_to_animalday function."""
    
    def test_nest_mode(self):
        """Test nest mode parsing."""
        filepath = Path("/parent/WT_A10_Jan01_2023/data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="nest"
        )
        
        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Jan-01-2023"
        assert result["animalday"] == "A10 WT Jan-01-2023"
        
    def test_concat_mode(self):
        """Test concat mode parsing."""
        filepath = Path("/path/WT_A10_Jan01_2023_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="concat"
        )
        
        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Jan-01-2023"
        
    def test_noday_mode(self):
        """Test noday mode parsing."""
        filepath = Path("/path/WT_A10_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="noday"
        )
        
        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == constants.DEFAULT_DAY.strftime("%b-%d-%Y")
        
    def test_invalid_mode(self):
        """Test invalid mode handling."""
        filepath = Path("/path/test.bin")
        
        with pytest.raises(ValueError, match="Invalid mode"):
            utils.parse_path_to_animalday(filepath, mode="invalid")


class TestParseStrToGenotype:
    """Test parse_str_to_genotype function."""
    
    def test_wt_parsing(self):
        """Test WT genotype parsing."""
        assert utils.parse_str_to_genotype("WT_A10_data") == "WT"
        assert utils.parse_str_to_genotype("wildtype_A10_data") == "WT"
        
    def test_ko_parsing(self):
        """Test KO genotype parsing."""
        assert utils.parse_str_to_genotype("KO_A10_data") == "KO"
        assert utils.parse_str_to_genotype("knockout_A10_data") == "KO"
        
    def test_no_match(self):
        """Test no match handling."""
        with pytest.raises(ValueError):
            utils.parse_str_to_genotype("INVALID_A10_data")


class TestParseStrToAnimal:
    """Test parse_str_to_animal function."""
    
    def test_tuple_param(self):
        """Test tuple parameter parsing."""
        string = "WT_A10_Jan01_2023"
        result = utils.parse_str_to_animal(string, animal_param=(1, "_"))
        assert result == "A10"
        
    def test_regex_param(self):
        """Test regex parameter parsing."""
        string = "WT_A10_Jan01_2023"
        result = utils.parse_str_to_animal(string, animal_param=r"A\d+")
        assert result == "A10"
        
    def test_list_param(self):
        """Test list parameter parsing."""
        string = "WT_A10_Jan01_2023"
        result = utils.parse_str_to_animal(string, animal_param=["A10", "A11", "A12"])
        assert result == "A10"
        
    def test_no_match(self):
        """Test no match handling."""
        string = "WT_A10_Jan01_2023"
        with pytest.raises(ValueError):
            utils.parse_str_to_animal(string, animal_param=["B10", "B11"])
            
    def test_invalid_param_type(self):
        """Test invalid parameter type."""
        string = "WT_A10_Jan01_2023"
        with pytest.raises(ValueError, match="Invalid animal_param type"):
            utils.parse_str_to_animal(string, animal_param=123)


class TestParseStrToDay:
    """Test parse_str_to_day function."""
    
    def test_basic_date_parsing(self):
        """Test basic date parsing."""
        string = "WT_A10_Jan01_2023_data"
        result = utils.parse_str_to_day(string)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1
        
    def test_with_separator(self):
        """Test date parsing with separator."""
        string = "WT_A10_Jan01_2023_data"
        result = utils.parse_str_to_day(string, sep="_")
        assert result.year == 2023
        
    def test_no_valid_date(self):
        """Test handling when no valid date is found."""
        string = "WT_A10_invalid_data"
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day(string)


class TestParseChnameToAbbrev:
    """Test parse_chname_to_abbrev function."""
    
    def test_basic_abbreviation(self):
        """Test basic channel name abbreviation."""
        assert utils.parse_chname_to_abbrev("Auditory") == "Aud"
        assert utils.parse_chname_to_abbrev("Visual") == "Vis"
        assert utils.parse_chname_to_abbrev("Hippocampus") == "Hip"
        
    def test_case_insensitive(self):
        """Test case insensitive matching."""
        assert utils.parse_chname_to_abbrev("auditory") == "Aud"
        assert utils.parse_chname_to_abbrev("VISUAL") == "Vis"
        
    def test_no_match(self):
        """Test no match handling."""
        with pytest.raises(ValueError, match="No matching channel"):
            utils.parse_chname_to_abbrev("InvalidChannel")


class TestLogTransform:
    """Test log_transform function."""
    
    def test_basic_transform(self):
        """Test basic log transformation."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data)
        
        # Should be log-transformed
        expected = np.log10(data)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_with_offset(self):
        """Test log transformation with offset."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data, offset=1)
        
        # Should be log-transformed with offset
        expected = np.log10(data + 1)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_with_negative_values(self):
        """Test log transformation with negative values."""
        data = np.array([-1, 0, 1, 10])
        result = utils.log_transform(data, offset=2)
        
        # Should handle negative values with offset
        expected = np.log10(data + 2)
        np.testing.assert_array_almost_equal(result, expected)


class TestSortDataframeByPlotOrder:
    """Test sort_dataframe_by_plot_order function."""
    
    def test_basic_sorting(self):
        """Test basic DataFrame sorting."""
        df = pd.DataFrame({
            "channel": ["RAud", "LAud", "RMot", "LMot"],
            "genotype": ["KO", "WT", "WT", "KO"],
            "band": ["gamma", "delta", "alpha", "beta"]
        })
        
        result = utils.sort_dataframe_by_plot_order(df)
        
        # Should be sorted according to constants.DF_SORT_ORDER
        assert result.iloc[0]["channel"] == "LMot"  # First in channel order
        assert result.iloc[0]["genotype"] == "WT"   # First in genotype order
        
    def test_with_custom_sort_order(self):
        """Test sorting with custom sort order."""
        df = pd.DataFrame({
            "channel": ["RAud", "LAud"],
            "custom_col": ["B", "A"]
        })
        
        custom_order = {"custom_col": ["A", "B"]}
        result = utils.sort_dataframe_by_plot_order(df, custom_order)
        
        assert result.iloc[0]["custom_col"] == "A"


class TestTempDirectory:
    """Test temporary directory functions."""
    
    def test_set_and_get_temp_directory(self):
        """Test setting and getting temp directory."""
        test_path = "/tmp/test_dir"
        utils.set_temp_directory(test_path)
        
        result = utils.get_temp_directory()
        assert result == Path(test_path)
        
    @patch("os.environ.get")
    def test_get_temp_directory_default(self, mock_get):
        """Test getting default temp directory."""
        mock_get.return_value = None
        
        result = utils.get_temp_directory()
        assert isinstance(result, Path) 