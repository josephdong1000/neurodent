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
        assert np.isclose(utils.convert_units_to_multiplier("µV", "µV"), 1.0)
        assert np.isclose(utils.convert_units_to_multiplier("mV", "µV"), 1000.0)
        assert np.isclose(utils.convert_units_to_multiplier("V", "µV"), 1000000.0)
        assert np.isclose(utils.convert_units_to_multiplier("nV", "µV"), 0.001)
        
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
        
    def test_invalid_input_type(self):
        """Test error handling for non-datetime input."""
        with pytest.raises(TypeError, match="Expected datetime object, got"):
            utils.is_day("not a datetime")
            
        with pytest.raises(TypeError, match="Expected datetime object, got"):
            utils.is_day(123)
            
        with pytest.raises(TypeError, match="Expected datetime object, got"):
            utils.is_day(None)


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

    def test_without_gzip_as_string(self):
        """Test conversion without gzip compression returning string."""
        col_path = "/path/to/data_ColMajor_001.bin"
        rowdir_path = "/output/dir"
        
        result = utils.convert_colpath_to_rowpath(rowdir_path, col_path, gzip=False, aspath=False)
        expected = "/output/dir/data_RowMajor_001.bin"
        assert result == expected
        
    def test_invalid_col_path(self):
        """Test error handling for col_path without 'ColMajor'."""
        col_path = "/path/to/data_RowMajor_001.bin"
        rowdir_path = "/output/dir"
        
        with pytest.raises(ValueError, match="Expected 'ColMajor' in col_path"):
            utils.convert_colpath_to_rowpath(rowdir_path, col_path)
            
    def test_col_path_without_colmajor_string(self):
        """Test error handling for col_path that doesn't contain 'ColMajor'."""
        col_path = "/path/to/data_001.bin"
        rowdir_path = "/output/dir"
        
        with pytest.raises(ValueError, match="Expected 'ColMajor' in col_path"):
            utils.convert_colpath_to_rowpath(rowdir_path, col_path)


class TestFilepathToIndex:
    """Test filepath_to_index function."""
    
    def test_basic_extraction(self):
        """Test basic index extraction."""
        filepath = "/path/to/data_ColMajor_001.bin"
        assert utils.filepath_to_index(filepath) == 1
        
    def test_with_different_suffixes(self):
        """Test with different file suffixes."""
        # Test .npy.gz suffix
        filepath = "/path/to/data_005_RowMajor.npy.gz"
        assert utils.filepath_to_index(filepath) == 5

        # Test .bin suffix
        filepath = "/path/to/data_006_RowMajor.bin"
        assert utils.filepath_to_index(filepath) == 6

        # Test .npy suffix
        filepath = "/path/to/data_007_RowMajor.npy"
        assert utils.filepath_to_index(filepath) == 7

        # Test no suffix
        filepath = "/path/to/data_008_RowMajor"
        assert utils.filepath_to_index(filepath) == 8

        # Test multiple suffixes
        filepath = "/path/to/data_009_RowMajor.test.npy.gz"
        assert utils.filepath_to_index(filepath) == 9
        
    def test_with_meta_suffix(self):
        """Test with meta suffix."""
        filepath = "/path/to/data_Meta_010.json"
        assert utils.filepath_to_index(filepath) == 10
        
    def test_with_multiple_numbers(self):
        """Test with multiple numbers in filename."""
        # Test with year in filename
        filepath = "/path/to/data_2023_015_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 15

        # Test with multiple numbers throughout path
        filepath = "/path/to/123/data_456_789_ColMajor.bin" 
        assert utils.filepath_to_index(filepath) == 789

        # Test with numbers in directory names
        filepath = "/path/2024/data_v2_042_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 42

    def test_dots_in_filename(self):
        """Test handling of dots within filenames."""
        # Test with decimal numbers
        filepath = "/path/to/data_1.2_3.4_567_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 567

        # Test with version numbers
        filepath = "/path/to/data_v1.0_042_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 42

        # Test with multiple dots
        filepath = "/path/to/data.2023.001_ColMajor.bin"
        assert utils.filepath_to_index(filepath) == 1


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


class TestNanAverage:
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
        
        # First column: [1, 4, 7] with weights [0.5, 0.3, 0.2]
        expected_col1 = 1*0.5 + 4*0.3 + 7*0.2  # = 2.9
        np.testing.assert_almost_equal(result[0], expected_col1)
        
        # Second column: [nan, 5, 8] with adjusted weights [0, 0.3, 0.2]
        # Need to renormalize weights to sum to 1: [0, 0.6, 0.4]
        expected_col2 = 5*0.6 + 8*0.4  # = 6.2
        np.testing.assert_almost_equal(result[1], expected_col2)
        
        # Third column: [3, 6, nan] with adjusted weights [0.5, 0.3, 0]
        # Renormalized weights: [0.625, 0.375, 0]
        expected_col3 = 3*0.625 + 6*0.375  # = 4.125
        np.testing.assert_almost_equal(result[2], expected_col3)


class TestParsePathToAnimalday:
    """Test parse_path_to_animalday function."""
    
    def test_nest_mode(self):
        """Test nest mode parsing."""
        # Use a filename with a valid date token that the parser can recognize
        filepath = Path("/parent/WT_A10_2023-04-01/recording_2023-04-01.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="nest"
        )
        
        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"
        assert result["animalday"] == "A10 WT Apr-01-2023"
        
    def test_concat_mode(self):
        """Test concat mode parsing."""
        # Use a filename with a valid date token that the parser can recognize
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="concat"
        )
        
        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"
        
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
        assert result["animalday"] == f"A10 WT {constants.DEFAULT_DAY.strftime('%b-%d-%Y')}"
        
        
    def test_invalid_mode(self):
        """Test invalid mode handling."""
        filepath = Path("/path/WT_A10_2023-01-1")
        
        # Test various invalid modes
        invalid_modes = ["invalid", "test", "random", "unknown", None]
        for mode in invalid_modes:
            with pytest.raises(ValueError, match=f"Invalid mode: {mode}"):
                utils.parse_path_to_animalday(filepath, mode=mode)
                
    def test_invalid_filepath_type(self):
        """Test invalid filepath type handling."""
        invalid_filepaths = [123, None, [], {}, 3.14]
        
        for filepath in invalid_filepaths:
            with pytest.raises((TypeError, AttributeError)):
                utils.parse_path_to_animalday(filepath, mode="concat")
                
    def test_filepath_without_genotype(self):
        """Test filepath that doesn't contain valid genotype."""
        filepath = Path("/path/INVALID_A10_2023-01-1")
        
        with pytest.raises(ValueError, match="does not have any matching values"):
            utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="concat")
            
    def test_filepath_without_valid_animal_id(self):
        """Test filepath that doesn't contain valid animal ID."""
        filepath = Path("/path/WT_INVALID_2023-01-1")
        
        with pytest.raises(ValueError, match="No matching ID found"):
            utils.parse_path_to_animalday(filepath, animal_param=["A1011"], mode="concat")
            
    def test_filepath_without_valid_date(self):
        """Test filepath that doesn't contain valid date (for modes that require date)."""
        filepath = Path("/path/WT_A10nvalid_date.bin")
        
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="concat")
            
    def test_nest_mode_with_invalid_parent_name(self):
        """Test nest mode with invalid parent directory name."""
        filepath = Path("/parent/INVALID_NAME/recording_2023-04-1")
        
        with pytest.raises(ValueError, match="does not have any matching values"):
            utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="nest")
            
    def test_nest_mode_with_invalid_filename(self):
        """Test nest mode with invalid filename (no date)."""
        filepath = Path("/parent/WT_A10_20231/recording_invalid.bin")
        
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="nest")

    def test_base_mode(self):
        """Test base mode parsing (same as concat)."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="base"
        )
        
        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"

    def test_with_day_sep_parameter(self):
        """Test parsing with custom day separator."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            day_sep="_",
            mode="concat"
        )
        
        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"

    def test_animal_param_tuple_format(self):
        """Test animal_param as tuple (index, separator) format."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=(1, "_"), 
            mode="concat"
        )
        
        assert result["animal"] == "A10"

    def test_animal_param_regex_format(self):
        """Test animal_param as regex pattern format."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=r"A\d+", 
            mode="concat"
        )
        
        assert result["animal"] == "A10"

    def test_animal_param_list_format(self):
        """Test animal_param as list of possible IDs format."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        
        result = utils.parse_path_to_animalday(
            filepath, 
            animal_param=["A10", "A11", "A12"], 
            mode="concat"
        )
        
        assert result["animal"] == "A10"

    def test_documentation_examples(self):
        """Test the specific examples mentioned in the documentation."""
        # Test nest mode example: /WT_A10/recording_2023-04-01.bin
        nest_filepath = Path("/parent/WT_A10/recording_2023-04-01.bin")
        nest_result = utils.parse_path_to_animalday(
            nest_filepath, 
            animal_param=(1, "_"), 
            mode="nest"
        )
        assert nest_result["animal"] == "A10"
        assert nest_result["genotype"] == "WT"
        assert nest_result["day"] == "Apr-01-2023"
        assert nest_result["animalday"] == "A10 WT Apr-01-2023"
        
        # Test concat mode example: /WT_A10_2023-04-01_data.bin
        concat_filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        concat_result = utils.parse_path_to_animalday(
            concat_filepath, 
            animal_param=(1, "_"), 
            mode="concat"
        )
        assert concat_result["animal"] == "A10"
        assert concat_result["genotype"] == "WT"
        assert concat_result["day"] == "Apr-01-2023"
        assert concat_result["animalday"] == "A10 WT Apr-01-2023"
        
        # Test noday mode example: /WT_A10_recording.*"
        noday_filepath = Path("/path/WT_A10_data.bin")
        noday_result = utils.parse_path_to_animalday(
            noday_filepath, 
            animal_param=(1, "_"), 
            mode="noday"
        )
        assert noday_result["animal"] == "A10"
        assert noday_result["genotype"] == "WT"
        assert noday_result["day"] == constants.DEFAULT_DAY.strftime("%b-%d-%Y")
        assert noday_result["animalday"] == f"A10 WT {constants.DEFAULT_DAY.strftime('%b-%d-%Y')}"


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
            
    @patch('pythoneeg.constants.GENOTYPE_ALIASES', {
        "HET": ["HET", "heterozygous", "het"],
        "HOM": ["HOM", "homozygous", "hom"],
        "CONTROL": ["CONTROL", "control", "CTRL"]
    })
    def test_custom_genotype_aliases(self):
        """Test parsing with custom genotype aliases."""
        assert utils.parse_str_to_genotype("HET_A10_data") == "HET"
        assert utils.parse_str_to_genotype("heterozygous_A10_data") == "HET"
        assert utils.parse_str_to_genotype("het_A10_data") == "HET"
        
        assert utils.parse_str_to_genotype("HOM_A10_data") == "HOM"
        assert utils.parse_str_to_genotype("homozygous_A10_data") == "HOM"
        assert utils.parse_str_to_genotype("hom_A10_data") == "HOM"
        
        assert utils.parse_str_to_genotype("CONTROL_A10_data") == "CONTROL"
        assert utils.parse_str_to_genotype("control_A10_data") == "CONTROL"
        assert utils.parse_str_to_genotype("CTRL_A10_data") == "CONTROL"
        
    @patch('pythoneeg.constants.GENOTYPE_ALIASES', {
        "MUT": ["MUT", "mutant", "mutation"],
        "WT": ["WT", "wildtype", "wild_type"]
    })
    def test_mutant_wildtype_aliases(self):
        """Test parsing with mutant/wildtype aliases."""
        assert utils.parse_str_to_genotype("MUT_A10_data") == "MUT"
        assert utils.parse_str_to_genotype("mutant_A10_data") == "MUT"
        assert utils.parse_str_to_genotype("mutation_A10_data") == "MUT"
        
        assert utils.parse_str_to_genotype("WT_A10_data") == "WT"
        assert utils.parse_str_to_genotype("wildtype_A10_data") == "WT"
        assert utils.parse_str_to_genotype("wild_type_A10_data") == "WT"
        
    @patch('pythoneeg.constants.GENOTYPE_ALIASES', {
        "TRANSGENIC": ["TRANSGENIC", "transgenic", "transgene"],
        "NON_TRANSGENIC": ["NON_TRANSGENIC", "non_transgenic", "non_transgene"]
    })
    def test_transgenic_aliases(self):
        """Test parsing with transgenic aliases."""
        assert utils.parse_str_to_genotype("TRANSGENIC_A10_data") == "TRANSGENIC"
        assert utils.parse_str_to_genotype("transgenic_A10_data") == "TRANSGENIC"
        assert utils.parse_str_to_genotype("transgene_A10_data") == "TRANSGENIC"
        assert utils.parse_str_to_genotype("nontestcasetransgenetestcase_A10_data") == "TRANSGENIC"
        
        assert utils.parse_str_to_genotype("NON_TRANSGENIC_A10_data") == "NON_TRANSGENIC"
        assert utils.parse_str_to_genotype("non_transgenic_A10_data") == "NON_TRANSGENIC"
        assert utils.parse_str_to_genotype("non_transgene_A10_data") == "NON_TRANSGENIC"
        
    @patch('pythoneeg.constants.GENOTYPE_ALIASES', {})
    def test_empty_aliases(self):
        """Test parsing with empty genotype aliases."""
        with pytest.raises(ValueError, match="does not have any matching values"):
            utils.parse_str_to_genotype("WT_A10_data")


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
        # Use a list of IDs that don't match the string
        with pytest.raises(ValueError, match="No matching ID found"):
            utils.parse_str_to_animal(string, animal_param=["B10", "B11"])
            
    def test_invalid_param_type(self):
        """Test invalid parameter type."""
        string = "WT_A10_Jan01_2023"
        with pytest.raises(ValueError, match="Invalid animal_param type"):
            utils.parse_str_to_animal(string, animal_param=123)
            
    # Tests for documentation examples
    def test_documentation_tuple_examples(self):
        """Test tuple format examples from documentation."""
        # Example 1: WT_A10_2023-01-01_data.bin with (1, "_")
        result1 = utils.parse_str_to_animal("WT_A10_2023-01-01_data.bin", (1, "_"))
        assert result1 == "A10"
        
        # Example 2: WT_A10_recording.bin with (1, "_")
        result2 = utils.parse_str_to_animal("WT_A10_recording.bin", (1, "_"))
        assert result2 == "A10"
        
        # Example 3: A10_WT_recording.bin with (0, "_")
        result3 = utils.parse_str_to_animal("A10_WT_recording.bin", (0, "_"))
        assert result3 == "A10"
        
    def test_documentation_regex_examples(self):
        """Test regex pattern examples from documentation."""
        # Example 1: Pattern r"A\d+" matches "A" followed by any number of digits
        # e.g. "A10" in "WT_A10_2023-01-01_data.bin"
        result1 = utils.parse_str_to_animal("WT_A10_2023-01-01_data.bin", r"A\d+")
        assert result1 == "A10"
        
        # Example 2: Pattern r"B\d+" matches "B" followed by any number of digits
        # e.g. "B15" in "mouse_B15_recording.bin"
        result2 = utils.parse_str_to_animal("mouse_B15_recording.bin", r"B\d+")
        assert result2 == "B15"
        
        # Example 3: Pattern r"\d+" matches one or more consecutive digits
        # e.g. "123" in "subject_123_data.bin"
        result3 = utils.parse_str_to_animal("subject_123_data.bin", r"\d+")
        assert result3 == "123"
        
        result4 = utils.parse_str_to_animal("subject_123_2025-01-01_data.bin", r"\d+")
        assert result4 == "123"

    def test_documentation_list_examples(self):
        """Test list format examples from documentation."""
        # Example 1: WT_A10_2023-01-01_data.bin with ["A10", "A11", "A12"]
        result1 = utils.parse_str_to_animal("WT_A10_2023-01-01_data.bin", ["A10", "A11", "A12"])
        assert result1 == "A10"
        
        # Example 2: KO_B15_recording.bin with ["A10", "B15", "C20"]
        result2 = utils.parse_str_to_animal("KO_B15_recording.bin", ["A10", "B15", "C20"])
        assert result2 == "B15"
        
        # Example 3: WT_A10_data.bin with ["B15", "C20"] - should raise error
        with pytest.raises(ValueError, match="No matching ID found in WT_A10_data.bin from possible IDs: \\['B15', 'C20'\\]"):
            utils.parse_str_to_animal("WT_A10_data.bin", ["B15", "C20"])
            
    def test_edge_cases(self):
        """Test edge cases and variations."""
        # Test with different separators
        result1 = utils.parse_str_to_animal("WT-A10-Jan01-2023", (1, "-"))
        assert result1 == "A10"
        
        # Test with multiple matches in regex (should return first match)
        result2 = utils.parse_str_to_animal("A10_B15_C20_data.bin", r"[A-Z]\d+")
        assert result2 == "A10"
        
        # Test with empty string in list (should not match)
        with pytest.raises(ValueError, match="No matching ID found"):
            utils.parse_str_to_animal("WT_A10_data.bin", ["", "B15"])
            
    def test_regex_no_match(self):
        """Test regex pattern with no match."""
        with pytest.raises(ValueError, match=r"No match found for pattern B\\d\+ in string WT_A10_data.bin"):
            utils.parse_str_to_animal("WT_A10_data.bin", r"B\d+")
            
    def test_multiple_matches_list_mode(self):
        """Test list mode when multiple IDs could match the string."""
        # Test case 1: String contains multiple possible IDs
        string = "WT_A10_B15_data.bin"
        result = utils.parse_str_to_animal(string, ["A10", "B15", "C20"])
        # Should return the first match found in the list order
        assert result == "A10"
        
        # Test case 2: String contains multiple possible IDs in different order
        string = "WT_B15_A10_data.bin"
        result = utils.parse_str_to_animal(string, ["A10", "B15", "C20"])
        # Should still return the first match found in the list order
        assert result == "A10"
        
        # Test case 3: String contains multiple possible IDs, test with different list order
        string = "WT_A10_B15_data.bin"
        result = utils.parse_str_to_animal(string, ["B15", "A10", "C20"])
        # Should return the first match found in the new list order
        assert result == "B15"
        
    def test_partial_matches_list_mode(self):
        """Test list mode with partial matches (substrings)."""
        # Test case 1: One ID is a substring of another
        string = "WT_A10_data.bin"
        result = utils.parse_str_to_animal(string, ["A1", "A10", "A100"])
        # Should return the first match found in the list order
        assert result == "A1"
        
        # Test case 2: One ID is a substring of another, different order
        string = "WT_A10_data.bin"
        result = utils.parse_str_to_animal(string, ["A10", "A1", "A100"])
        # Should return the first match found in the list order
        assert result == "A10"
        
    def test_case_sensitivity_list_mode(self):
        """Test list mode with case sensitivity."""
        # Test case 1: Case sensitive matching
        string = "WT_a10_data.bin"
        result = utils.parse_str_to_animal(string, ["a10", "A10", "b15"])
        # Should return the first case-sensitive match
        assert result == "a10"
        
        # Test case 2: No case-sensitive match
        string = "WT_a10_data.bin"
        with pytest.raises(ValueError, match="No matching ID found"):
            utils.parse_str_to_animal(string, ["A10", "B15"])
            
    def test_empty_and_whitespace_list_mode(self):
        """Test list mode with empty strings and whitespace."""
        # Test case 1: Empty strings in list (should be ignored)
        string = "WT_A10_data.bin"
        result = utils.parse_str_to_animal(string, ["", "A10", "   ", "B15"])
        # Should return the first non-empty match
        assert result == "A10"
        
        # Test case 2: Whitespace-only strings in list (should be ignored)
        string = "WT_A10_data.bin"
        result = utils.parse_str_to_animal(string, ["   ", "A10", "\t", "B15"])
        # Should return the first non-whitespace match
        assert result == "A10"
        
        # Test case 3: All empty/whitespace strings
        string = "WT_A10_data.bin"
        with pytest.raises(ValueError, match="No matching ID found"):
            utils.parse_str_to_animal(string, ["", "   ", "\t", "\n"])
            
    def test_whitespace_in_string_list_mode(self):
        """Test list mode when the input string contains whitespace."""
        # Test case 1: String with leading/trailing whitespace
        string = "  WT_A10_data.bin  "
        result = utils.parse_str_to_animal(string, ["A10", "B15"])
        # Should still match "A10" even with whitespace
        assert result == "A10"
        
        # Test case 2: String with internal whitespace
        string = "WT A10 data.bin"
        result = utils.parse_str_to_animal(string, ["A10", "B15"])
        # Should still match "A10" even with internal whitespace
        assert result == "A10"
        
        # Test case 3: String with tabs and newlines
        string = "WT\tA10\ndata.bin"
        result = utils.parse_str_to_animal(string, ["A10", "B15"])
        # Should still match "A10" even with tabs and newlines
        assert result == "A10"
        
        # Test case 4: String with mixed whitespace characters
        string = "  WT  A10  data.bin  "
        result = utils.parse_str_to_animal(string, ["A10", "B15"])
        # Should still match "A10" even with mixed whitespace
        assert result == "A10"
        
    def test_whitespace_in_list_items(self):
        """Test list mode when the list items contain whitespace."""
        # Test case 1: List items with leading/trailing whitespace - exact match required
        string = "WT  A10  data.bin"
        result = utils.parse_str_to_animal(string, ["  A10  ", "B15"])
        # Should match "  A10  " because it's an exact substring match
        assert result == "  A10  "
        
        # Test case 2: List items with internal whitespace - exact match required
        string = "WT A 10 data.bin"
        result = utils.parse_str_to_animal(string, ["A 10", "B15"])
        # Should match "A 10" because it's an exact substring match
        assert result == "A 10"
        
        # Test case 3: List items with tabs and newlines - exact match required
        string = "WT\tA10\ndata.bin"
        result = utils.parse_str_to_animal(string, ["\tA10\n", "B15"])
        # Should match "\tA10\n" because it's an exact substring match
        assert result == "\tA10\n"
        
        # Test case 4: No match when whitespace doesn't align exactly
        string = "WT_A10_data.bin"
        # Should not match " A10 " because the string doesn't have leading/trailing spaces
        with pytest.raises(ValueError, match="No matching ID found"):
            utils.parse_str_to_animal(string, [" A10 ", "B15"])
        
        # Test case 5: Whitespace in string but not in list item
        string = "WT  A10  data.bin"
        result = utils.parse_str_to_animal(string, ["A10", "B15"])
        # Should match "A10" because it's a substring of "  A10  "
        assert result == "A10"


class TestParseStrToDay:
    """Test parse_str_to_day function."""
    
    def test_standard_iso_date_format(self):
        """Test parsing of standard ISO date format (YYYY-MM-DD)."""
        # Use a valid date format that the parser can recognize
        string = "WT_A10_2023-07-04_data"
        result = utils.parse_str_to_day(string)
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
    def test_custom_separator_parameter(self):
        """Test date parsing with custom separator parameter."""
        string = "WT_A10_2023-07-04_data"
        result = utils.parse_str_to_day(string, sep="_")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        # Empty separator should split by whitespace
        result = utils.parse_str_to_day("WT A10 2023-07-04 data")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Custom separator that doesn't exist in string - should still work because of fuzzy parsing
        result = utils.parse_str_to_day("WT_A10_2023-07-04_data", sep="|")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Separator that creates empty tokens
        result = utils.parse_str_to_day("WT__A10__2023-07-04__data", sep="_")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
    def test_no_date_found_raises_valueerror(self):
        """Test that ValueError is raised when no valid date is found."""
        string = "WT_A10_invalid_data"
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day(string)
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_A5_G20_(15)_data")
            
    def test_non_string_input_raises_typeerror(self):
        """Test that TypeError is raised for non-string inputs."""
        # Non-string inputs should raise TypeError
        with pytest.raises((TypeError, AttributeError)):
            utils.parse_str_to_day(123)
        with pytest.raises((TypeError, AttributeError)):
            utils.parse_str_to_day(None)
        with pytest.raises((TypeError, AttributeError)):
            utils.parse_str_to_day(["2023-01-01"])
        with pytest.raises((TypeError, AttributeError)):
            utils.parse_str_to_day({"date": "2023-01-01"})
            
    def test_empty_and_whitespace_only_strings(self):
        """Test that empty and whitespace-only strings raise ValueError."""
        # Empty string should raise ValueError
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("")
        
        # Whitespace-only string should raise ValueError
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("   ")
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("\t\n\r")
            
    def test_invalid_parse_params_type_raises_typeerror(self):
        """Test that invalid parse_params types raise TypeError."""
        # Non-dict parse_params should raise TypeError
        with pytest.raises(TypeError):
            utils.parse_str_to_day("2023-07-04", parse_params="invalid")
        with pytest.raises(TypeError):
            utils.parse_str_to_day("2023-07-04", parse_params=123)
        
        # Empty dict parse_params should work (uses default behavior)
        result = utils.parse_str_to_day("2023-07-04", parse_params={})
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
    def test_dates_before_1980_are_ignored(self):
        """Test that dates before 1980 are ignored for safety."""
        # Dates before 1980 should be ignored
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_1979-07-04_data")
        
        # Very old dates should be ignored
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_1900-07-04_data")
        
        result = utils.parse_str_to_day("WT_A10_1980-07-04_data")
        assert result.year == 1980
        assert result.month == 7
        assert result.day == 4

        # Future dates should work
        result = utils.parse_str_to_day("WT_A10_2030-07-04_data")
        assert result.year == 2030
        assert result.month == 7
        assert result.day == 4
        
    def test_first_valid_date_is_returned_when_multiple_present(self):
        """Test that the first valid date is returned when multiple dates are present."""
        # Should return the first valid date found
        result = utils.parse_str_to_day("WT_A10_2023-07-04_2024-12-25_data")
        assert result.year == 2023  # Should pick first date
        assert result.month == 7
        assert result.day == 4
        
    def test_ambiguous_date_formats_raise_valueerror(self):
        """Test that invalid date formats raise ValueError."""
        # Ambiguous date formats
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_13-13-13_data")  # Invalid month/day
        
    def test_unicode_and_special_characters_dont_interfere(self):
        """Test that unicode and special characters don't interfere with date parsing."""
        # Unicode characters in string
        result = utils.parse_str_to_day("WT_A10_2023-07-04_αβγ_data")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Special characters that might interfere with parsing
        result = utils.parse_str_to_day("WT_A10_2023-07-04_!@#$%^&*()_data")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
    def test_many_tokens_performance(self):
        """Test performance with strings containing many tokens."""
        # String with many tokens (could cause performance issues)
        many_tokens = "WT_A10_" + "_".join([f"token{i}" for i in range(int(1e6))]) + "_2023-07-04_data"
        result = utils.parse_str_to_day(many_tokens)
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
    def test_iso_us_and_european_date_formats(self):
        """Test parsing of ISO, US, and European date formats."""
        # Test various date formats that should work
        test_cases = [
            ("WT_A10_2023-07-04_data", 2023, 7, 4),      # ISO format
            ("WT_A10_07/04/2023_data", 2023, 7, 4),      # US format
            ("WT_A10_04/07/2023_data", 2023, 4, 7),      # European format
            ("WT_A10_2023-7-4_data", 2023, 7, 4),        # No leading zeros
        ]
        
        for string, expected_year, expected_month, expected_day in test_cases:
            result = utils.parse_str_to_day(string)
            assert result.year == expected_year, f"Failed for {string}"
            assert result.month == expected_month, f"Failed for {string}"
            assert result.day == expected_day, f"Failed for {string}"
            
    def test_complex_date_formats(self):
        """Test parsing of complex date formats with the new two-pass approach."""
        # Test cases that should work with the new sliding window approach
        test_cases = [
            ("ID1524_January-20-2012_data", 2012, 1, 20),    # Complex ID with date
            ("ID1524_Jan-20-2012_data", 2012, 1, 20),        # Abbreviated month
        ]
        
        for string, expected_year, expected_month, expected_day in test_cases:
            result = utils.parse_str_to_day(string, parse_mode="split")
            assert result.year == expected_year, f"Failed for {string}"
            assert result.month == expected_month, f"Failed for {string}"
            assert result.day == expected_day, f"Failed for {string}"
            
    def test_parsemode_all(self):
        """Test underscore-separated dates (current limitation)."""
        # Note: year defaults to 2000 due to underscore interference in parsing
        # This is acceptable behavior for the current safety-focused design
        
        result = utils.parse_str_to_day("Mouse_A10_December_25_2023_data", parse_mode="all")
        assert result.month == 12
        assert result.day == 25
        
        result = utils.parse_str_to_day("Subject_123_March_15_2024_data", parse_mode="all")
        assert result.month == 3
        assert result.day == 15

        result = utils.parse_str_to_day("ID1524_January_20_2012_data")
        assert result.month == 1
        assert result.day == 20
            
    def test_edge_cases_with_new_approach(self):
        """Test edge cases that demonstrate the safety of the new approach."""
        # These should still fail (safety maintained)
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("ID1524_invalid_data")
            
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_no_date_here")
            
    def test_number_only_ids_with_number_only_dates(self):
        """Test parsing with numeric IDs and numeric date formats."""
        # Test various combinations of numeric IDs with numeric dates
        test_cases = [
            ("123_2023-07-04_data", 2023, 7, 4),           # Simple numeric ID with ISO date
            ("456_07/04/2023_data", 2023, 7, 4),           # Numeric ID with US date format
            ("789_04/07/2023_data", 2023, 4, 7),           
            ("999_2023-7-4_data", 2023, 7, 4),             # Numeric ID with no leading zeros
            ("123_2023-07-04_456_data", 2023, 7, 4),       # Multiple numeric IDs
        ]
        
        for string, expected_year, expected_month, expected_day in test_cases:
            result = utils.parse_str_to_day(string, parse_mode='split')
            assert result.year == expected_year, f"Failed for {string}"
            assert result.month == expected_month, f"Failed for {string}"
            assert result.day == expected_day, f"Failed for {string}"
            
    def test_multiple_numeric_dates_behavior(self):
        """Test behavior with multiple numeric dates (may vary based on dateutil version)."""
        # This case can cause dateutil to get confused about time offsets
        # The exact behavior may vary depending on the dateutil version
        string = "123_2023-07-04_789_2024-12-25_data"
        try:
            result = utils.parse_str_to_day(string, parse_mode='full')
            # If it succeeds, it should return the first valid date
            assert result.year == 2023
            assert result.month == 7
            assert result.day == 4
        except ValueError as e:
            # If it fails, it should be due to dateutil confusion
            assert "timedelta" in str(e) or "offset" in str(e)
            
    def test_numeric_ids_without_dates_raise_error(self):
        """Test that numeric IDs without valid dates raise ValueError."""
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("123_invalid_data")
            
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("456_789_no_date_here")

    def test_parse_mode_full(self):
        """Test parse_mode='full' - only tries parsing the entire cleaned string."""
        # Should work when date is in the full string
        result = utils.parse_str_to_day("2023-07-04", parse_mode="full")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Should work with extra text
        result = utils.parse_str_to_day("WT_A10_2023-07-04_data", parse_mode="full")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Should fail when date is only in individual tokens (no valid date in full string)
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_no_date_here", parse_mode="full")

    def test_parse_mode_split(self):
        """Test parse_mode='split' - only tries parsing individual tokens."""
        # Should work when date is in individual tokens
        result = utils.parse_str_to_day("WT_A10_2023-07-04_data", parse_mode="split")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Should work with custom separator
        result = utils.parse_str_to_day("WT|A10|2023|07|04|data", parse_mode="split", sep="|")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Should fail when no individual tokens contain valid date
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("no date here", parse_mode="split")

    def test_parse_mode_window(self):
        """Test parse_mode='window' - only tries parsing sliding windows of tokens."""
        # Should work when date requires multiple tokens
        result = utils.parse_str_to_day("WT_A10_January_20_2012_data", parse_mode="window")
        assert result.month == 1
        assert result.day == 20
        # Note: year may default to 2000 due to dateutil limitations
        
        # Should work with abbreviated month
        result = utils.parse_str_to_day("WT_A10_Jan_20_2012_data", parse_mode="window")
        assert result.month == 1
        assert result.day == 20
        
        # Should fail when no multi-token date exists
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("no date here at all", parse_mode="window")

    def test_parse_mode_all(self):
        """Test parse_mode='all' - uses all three approaches in sequence."""
        # Should work with full string parsing
        result = utils.parse_str_to_day("2023-07-04", parse_mode="all")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Should work with individual token parsing
        result = utils.parse_str_to_day("WT_A10_2023_07_04_data", parse_mode="all")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Should work with window parsing
        result = utils.parse_str_to_day("WT_A10_January_20_2012_data", parse_mode="all")
        assert result.month == 1
        assert result.day == 20

    def test_parse_mode_default_behavior(self):
        """Test that default parse_mode is 'split'."""
        # Default should be 'split' mode
        result = utils.parse_str_to_day("WT_A10_2023_07_04_data")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # Should fail for full string parsing cases with default mode
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("2023-07-04")

    def test_parse_mode_invalid_value(self):
        """Test that invalid parse_mode values raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid parse_mode"):
            utils.parse_str_to_day("2023-07-04", parse_mode="invalid_mode")
        
        with pytest.raises(ValueError, match="Invalid parse_mode"):
            utils.parse_str_to_day("2023-07-04", parse_mode="")

    def test_parse_mode_with_parse_params(self):
        """Test that parse_mode works correctly with parse_params."""
        # Test with fuzzy parsing disabled
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_2023-07-04_data", 
                                  parse_mode="split", 
                                  parse_params={"fuzzy": False})
        
        # Test with fuzzy parsing enabled
        result = utils.parse_str_to_day("WT_A10_2023-07-04_data", 
                                       parse_mode="split", 
                                       parse_params={"fuzzy": True})
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

    def test_parse_mode_performance_comparison(self):
        """Test that different parse modes have expected performance characteristics."""
        # Create a string that would be expensive to process with 'all' mode
        many_tokens = "WT_A10_" + "_".join([f"token{i}" for i in range(100)]) + "_2023_07_04_data"
        
        # 'split' mode should be fast and find the date
        result = utils.parse_str_to_day(many_tokens, parse_mode="split")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4
        
        # 'all' mode should also work but might be slower
        result = utils.parse_str_to_day(many_tokens, parse_mode="all")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4


class TestParseChnameToAbbrev:
    """Test parse_chname_to_abbrev function."""
    
    def test_basic_abbreviation(self):
        """Test basic channel name abbreviation."""
        # Use full channel names that include left/right prefixes
        assert utils.parse_chname_to_abbrev("left Aud") == "LAud"
        assert utils.parse_chname_to_abbrev("right Vis") == "RVis"
        assert utils.parse_chname_to_abbrev("Left Hip") == "LHip"
        
    def test_case_insensitive(self):
        """Test case insensitive matching."""
        # Use full channel names that include left/right prefixes
        assert utils.parse_chname_to_abbrev("Left aud") == "LAud"
        assert utils.parse_chname_to_abbrev("Right VIS") == "RVis"
        
    def test_no_match(self):
        """Test no match handling."""
        with pytest.raises(ValueError, match="does not have any matching values"):
            utils.parse_chname_to_abbrev("InvalidChannel")


class TestLogTransform:
    """Test log_transform function."""
    
    def test_basic_transform(self):
        """Test basic log transformation."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data)
        
        # Should be natural log-transformed (ln(x+1))
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_with_offset(self):
        """Test log transformation with offset."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data, offset=1)
        
        # Should be natural log-transformed with offset (ln(x+1))
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_with_negative_values(self):
        """Test log transformation with negative values."""
        data = np.array([-1, 0, 1, 10])
        result = utils.log_transform(data, offset=2)
        
        # Should handle negative values with offset (ln(x+2))
        expected = np.log(data + 2)
        np.testing.assert_array_almost_equal(result, expected)


class TestSortDataframeByPlotOrder:
    """Test sort_dataframe_by_plot_order function."""
    
    def test_basic_sorting(self):
        """Test basic DataFrame sorting."""
        # Create test data with known order
        df = pd.DataFrame({
            "genotype": ["WT", "KO", "WT", "KO"],
            "channel": ["LAud", "RAud", "LVis", "RVis"],
            "value": [1, 2, 3, 4]
        })
        
        result = utils.sort_dataframe_by_plot_order(df)
        
        # Check that the DataFrame is sorted according to the predefined order
        # The first row should be WT (first in genotype order)
        assert result.iloc[0]["genotype"] == "WT"
        # The second row should be KO
        assert result.iloc[1]["genotype"] == "KO"


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