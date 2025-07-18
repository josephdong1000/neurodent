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
    
    def test_basic_date_parsing(self):
        """Test basic date parsing."""
        # Use a valid date format that the parser can recognize
        string = "WT_A10_2023-01-01_data"
        result = utils.parse_str_to_day(string)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1
        
    def test_with_separator(self):
        """Test date parsing with separator."""
        string = "WT_A10_2023-01-01_data"
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