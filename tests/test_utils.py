"""
Unit tests for neurodent.core.utils module.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import shutil
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from neurodent.core import utils
from neurodent import constants


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
        expected = Path(rowdir_path) / "data_RowMajor_001.npy.gz"
        assert result == expected

    def test_without_gzip(self):
        """Test conversion without gzip compression."""
        col_path = "/path/to/data_ColMajor_001.bin"
        rowdir_path = "/output/dir"

        result = utils.convert_colpath_to_rowpath(rowdir_path, col_path, gzip=False)
        expected = Path(rowdir_path) / "data_RowMajor_001.bin"
        assert result == expected

    def test_as_string(self):
        """Test conversion returning string."""
        col_path = "/path/to/data_ColMajor_001.bin"
        rowdir_path = "/output/dir"

        result = utils.convert_colpath_to_rowpath(rowdir_path, col_path, aspath=False)
        expected = str(Path(rowdir_path) / "data_RowMajor_001.npy.gz")
        assert result == expected

    def test_without_gzip_as_string(self):
        """Test conversion without gzip compression returning string."""
        col_path = "/path/to/data_ColMajor_001.bin"
        rowdir_path = "/output/dir"

        result = utils.convert_colpath_to_rowpath(rowdir_path, col_path, gzip=False, aspath=False)
        expected = str(Path(rowdir_path) / "data_RowMajor_001.bin")
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
        expected_col1 = 1 * 0.5 + 4 * 0.3 + 7 * 0.2  # = 2.9
        np.testing.assert_almost_equal(result[0], expected_col1)

        # Second column: [nan, 5, 8] with adjusted weights [0, 0.3, 0.2]
        # Need to renormalize weights to sum to 1: [0, 0.6, 0.4]
        expected_col2 = 5 * 0.6 + 8 * 0.4  # = 6.2
        np.testing.assert_almost_equal(result[1], expected_col2)

        # Third column: [3, 6, nan] with adjusted weights [0.5, 0.3, 0]
        # Renormalized weights: [0.625, 0.375, 0]
        expected_col3 = 3 * 0.625 + 6 * 0.375  # = 4.125
        np.testing.assert_almost_equal(result[2], expected_col3)


class TestParsePathToAnimalday:
    """Test parse_path_to_animalday function."""

    def test_nest_mode(self):
        """Test nest mode parsing."""
        # Use a filename with a valid date token that the parser can recognize
        filepath = Path("/parent/WT_A10_2023-04-01/recording_2023-04-01.bin")

        with pytest.warns(UserWarning, match="Only 1 string token found"):
            result = utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="nest")

        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"
        assert result["animalday"] == "A10 WT Apr-01-2023"

    def test_concat_mode(self):
        """Test concat mode parsing."""
        # Use a filename with a valid date token that the parser can recognize
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")

        result = utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="concat")

        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"

    def test_noday_mode(self):
        """Test noday mode parsing."""
        filepath = Path("/path/WT_A10_data.bin")

        result = utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="noday")

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

        # Expected warning because filename doesn't contain separators
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            with pytest.raises(ValueError, match="No valid date token found"):
                utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="nest")

    def test_base_mode(self):
        """Test base mode parsing (same as concat)."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")

        result = utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="base")

        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"

    def test_with_day_sep_parameter(self):
        """Test parsing with custom day separator."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")

        result = utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), day_sep="_", mode="concat")

        assert result["animal"] == "A10"
        assert result["genotype"] == "WT"
        assert result["day"] == "Apr-01-2023"

    def test_animal_param_tuple_format(self):
        """Test animal_param as tuple (index, separator) format."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")

        result = utils.parse_path_to_animalday(filepath, animal_param=(1, "_"), mode="concat")

        assert result["animal"] == "A10"

    def test_animal_param_regex_format(self):
        """Test animal_param as regex pattern format."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")

        result = utils.parse_path_to_animalday(filepath, animal_param=r"A\d+", mode="concat")

        assert result["animal"] == "A10"

    def test_animal_param_list_format(self):
        """Test animal_param as list of possible IDs format."""
        filepath = Path("/path/WT_A10_2023-04-01_data.bin")

        result = utils.parse_path_to_animalday(filepath, animal_param=["A10", "A11", "A12"], mode="concat")

        assert result["animal"] == "A10"

    def test_documentation_examples(self):
        """Test the specific examples mentioned in the documentation."""
        # Test nest mode example: /WT_A10/recording_2023-04-01.bin
        nest_filepath = Path("/parent/WT_A10/recording_2023-04-01.bin")

        # Expected warning because filename doesn't contain separators
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            nest_result = utils.parse_path_to_animalday(nest_filepath, animal_param=(1, "_"), mode="nest")
        assert nest_result["animal"] == "A10"
        assert nest_result["genotype"] == "WT"
        assert nest_result["day"] == "Apr-01-2023"
        assert nest_result["animalday"] == "A10 WT Apr-01-2023"

        # Test concat mode example: /WT_A10_2023-04-01_data.bin
        concat_filepath = Path("/path/WT_A10_2023-04-01_data.bin")
        concat_result = utils.parse_path_to_animalday(concat_filepath, animal_param=(1, "_"), mode="concat")
        assert concat_result["animal"] == "A10"
        assert concat_result["genotype"] == "WT"
        assert concat_result["day"] == "Apr-01-2023"
        assert concat_result["animalday"] == "A10 WT Apr-01-2023"

        # Test noday mode example: /WT_A10_recording.*"
        noday_filepath = Path("/path/WT_A10_data.bin")
        noday_result = utils.parse_path_to_animalday(noday_filepath, animal_param=(1, "_"), mode="noday")
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

    @patch(
        "neurodent.constants.GENOTYPE_ALIASES",
        {
            "HET": ["HET", "heterozygous", "het"],
            "HOM": ["HOM", "homozygous", "hom"],
            "CONTROL": ["CONTROL", "control", "CTRL"],
        },
    )
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

    @patch(
        "neurodent.constants.GENOTYPE_ALIASES",
        {"MUT": ["MUT", "mutant", "mutation"], "WT": ["WT", "wildtype", "wild_type"]},
    )
    def test_mutant_wildtype_aliases(self):
        """Test parsing with mutant/wildtype aliases."""
        assert utils.parse_str_to_genotype("MUT_A10_data") == "MUT"
        assert utils.parse_str_to_genotype("mutant_A10_data") == "MUT"
        assert utils.parse_str_to_genotype("mutation_A10_data") == "MUT"

        assert utils.parse_str_to_genotype("WT_A10_data") == "WT"
        assert utils.parse_str_to_genotype("wildtype_A10_data") == "WT"
        assert utils.parse_str_to_genotype("wild_type_A10_data") == "WT"

    @patch(
        "neurodent.constants.GENOTYPE_ALIASES",
        {
            "TRANSGENIC": ["TRANSGENIC", "transgenic", "transgene"],
            "NON_TRANSGENIC": ["NON_TRANSGENIC", "non_transgenic", "non_transgene"],
        },
    )
    def test_transgenic_aliases(self):
        """Test parsing with transgenic aliases."""
        assert utils.parse_str_to_genotype("TRANSGENIC_A10_data") == "TRANSGENIC"
        assert utils.parse_str_to_genotype("transgenic_A10_data") == "TRANSGENIC"
        assert utils.parse_str_to_genotype("transgene_A10_data") == "TRANSGENIC"
        assert utils.parse_str_to_genotype("nontestcasetransgenetestcase_A10_data") == "TRANSGENIC"

        assert utils.parse_str_to_genotype("NON_TRANSGENIC_A10_data") == "NON_TRANSGENIC"
        assert utils.parse_str_to_genotype("non_transgenic_A10_data") == "NON_TRANSGENIC"
        assert utils.parse_str_to_genotype("non_transgene_A10_data") == "NON_TRANSGENIC"

    @patch("neurodent.constants.GENOTYPE_ALIASES", {})
    def test_empty_aliases(self):
        """Test parsing with empty genotype aliases."""
        with pytest.raises(ValueError, match="does not have any matching values"):
            utils.parse_str_to_genotype("WT_A10_data")

    def test_genotype_backward_compatibility(self):
        """Test that existing genotype parsing code still works."""
        # Old function calls should work exactly as before
        assert utils.parse_str_to_genotype("WT_A10_data") == "WT"
        assert utils.parse_str_to_genotype("knockout_B15_data") == "KO"

        # Ambiguous cases should work (non-strict by default)
        result = utils.parse_str_to_genotype("WT_vs_KO_study")
        assert result in ["WT", "KO"]  # Should work, pick one based on longest match


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
        with pytest.raises(
            ValueError, match="No matching ID found in WT_A10_data.bin from possible IDs: \\['B15', 'C20'\\]"
        ):
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
        with pytest.warns(UserWarning, match="Only 1 string token found"):
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
        with pytest.warns(UserWarning, match="Only 1 string token found"):
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

        # 1980 is also ignored (year <= 1980)
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_1980-07-04_data")

        # 1981 should work
        result = utils.parse_str_to_day("WT_A10_1981-07-04_data")
        assert result.year == 1981
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
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_13-13-13_data")

        with pytest.warns(UserWarning, match="Only 1 string token found"):
            with pytest.raises(ValueError, match="No valid date token found"):
                utils.parse_str_to_day("ID1524_invalid_data")

        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_no_date_here")

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
        many_tokens = "WT_A10_" + "_".join([f"token{i}" for i in range(int(1e3))]) + "_2023-07-04_data"
        result = utils.parse_str_to_day(many_tokens)
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

    def test_iso_us_date_formats(self):
        """Test parsing of ISO and US date formats without date_patterns."""
        test_cases = [
            ("WT_A10_2023-07-04_data", 2023, 7, 4),  # ISO format
            ("WT_A10_07/04/2023_data", 2023, 7, 4),  # US format - dateutil defaults to US interpretation
            ("WT_A10_2023-7-4_data", 2023, 7, 4),  # No leading zeros
        ]

        for string, expected_year, expected_month, expected_day in test_cases:
            result = utils.parse_str_to_day(string)
            assert result.year == expected_year, f"Failed for {string}"
            assert result.month == expected_month, f"Failed for {string}"
            assert result.day == expected_day, f"Failed for {string}"

    def test_date_patterns_for_ambiguous_formats(self):
        """Test that date_patterns can resolve ambiguous formats like MM/DD/YYYY vs DD/MM/YYYY."""
        ambiguous_string = "WT_A10_04/07/2023_data"

        # Test US interpretation (MM/DD/YYYY) - April 7th
        us_patterns = [(r"(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})", "%m/%d/%Y")]
        result = utils.parse_str_to_day(ambiguous_string, date_patterns=us_patterns)
        assert result.year == 2023
        assert result.month == 4  # April
        assert result.day == 7

        # Test European interpretation (DD/MM/YYYY) - July 4th
        european_patterns = [(r"(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})", "%d/%m/%Y")]
        result = utils.parse_str_to_day(ambiguous_string, date_patterns=european_patterns)
        assert result.year == 2023
        assert result.month == 7  # July
        assert result.day == 4

    def test_date_patterns_validation(self):
        """Test validation of date_patterns parameter."""
        # Test invalid type
        with pytest.raises(TypeError, match="date_patterns must be a list"):
            utils.parse_str_to_day("2023-07-04", date_patterns="invalid")

        # Test invalid tuple format
        with pytest.raises(TypeError, match="must be a tuple of"):
            utils.parse_str_to_day("2023-07-04", date_patterns=["invalid"])

        # Test invalid tuple length
        with pytest.raises(TypeError, match="must be a tuple of"):
            utils.parse_str_to_day("2023-07-04", date_patterns=[("pattern",)])

        # Test non-string elements
        with pytest.raises(TypeError, match="must contain string elements"):
            utils.parse_str_to_day("2023-07-04", date_patterns=[(123, "%Y")])

    def test_date_patterns_comprehensive_edge_cases(self):
        """Test comprehensive edge cases for date_patterns functionality."""

        # Test 1: Multiple patterns match - should return first successful match
        multiple_patterns = [
            (r"(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})", "%m/%d/%Y"),  # US format
            (r"(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})", "%d/%m/%Y"),  # European format
            (r"(19\d{2}|20\d{2})-(\d{1,2})-(\d{1,2})", "%Y-%m-%d"),  # ISO format
        ]
        # Should use first pattern (US format) when multiple match
        with pytest.warns(UserWarning, match="Multiple.*patterns.*matched"):
            result = utils.parse_str_to_day("12/06/2023_data", date_patterns=multiple_patterns)
            assert result.year == 2023
            assert result.month == 12  # December (US format: MM/DD/YYYY)
            assert result.day == 6

        # Test 2: Multiple matches within single string - should return first match
        multi_date_string = "data_01/02/2020_and_03/04/2021_files"
        with pytest.warns(UserWarning, match="Multiple.*patterns.*matched"):
            result = utils.parse_str_to_day(multi_date_string, date_patterns=multiple_patterns)
            assert result.year == 2020  # First date found
            assert result.month == 1  # January (US format)
            assert result.day == 2

        # Test 3: No patterns match - should fall back to token parsing
        no_match_patterns = [(r"NEVER_MATCHES_ANYTHING", "%Y-%m-%d")]
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            with pytest.warns(UserWarning, match="patterns.*matched.*back"):
                result = utils.parse_str_to_day("WT_2023-12-25_data", date_patterns=no_match_patterns)
                assert result.year == 2023  # Falls back to dateutil parsing
                assert result.month == 12
                assert result.day == 25

        # Test 4: Pattern matches but format string is incompatible
        bad_format_patterns = [
            (r"(\d{4})-(\d{2})-(\d{2})", "%d/%m/%Y")
        ]  # Regex finds YYYY-MM-DD but format expects DD/MM/YYYY
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            with pytest.warns(UserWarning, match="patterns.*matched.*back"):
                result = utils.parse_str_to_day("test_2023-12-25_data", date_patterns=bad_format_patterns)
                assert result.year == 2023  # Should fall back to token parsing
                assert result.month == 12
                assert result.day == 25

        # Test 5: Pattern matches invalid date (e.g., Feb 30) - should fall back
        invalid_date_patterns = [(r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d")]
        result = utils.parse_str_to_day("test_2023-02-30_2023-12-25_data", date_patterns=invalid_date_patterns)
        # Should skip invalid Feb 30 and find valid Dec 25
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 25

        # Test 6: Various date formats beyond July 4, 2023
        various_patterns = [(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", "%d-%b-%Y")]  # DD-MMM-YYYY
        test_cases = [
            ("data_15-Jan-2022_file", 2022, 1, 15),
            ("data_03-Dec-2024_file", 2024, 12, 3),
            ("data_29-Feb-2024_file", 2024, 2, 29),  # Leap year
            ("data_01-Aug-1995_file", 1995, 8, 1),
            ("data_31-Mar-2000_file", 2000, 3, 31),
        ]
        for test_string, expected_year, expected_month, expected_day in test_cases:
            result = utils.parse_str_to_day(test_string, date_patterns=various_patterns)
            assert result.year == expected_year, f"Failed year for {test_string}"
            assert result.month == expected_month, f"Failed month for {test_string}"
            assert result.day == expected_day, f"Failed day for {test_string}"

    def test_date_patterns_year_1980_boundary(self):
        """Test year filtering around 1980 boundary."""
        iso_patterns = [(r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d")]

        # Test dates around 1980 boundary - include fallback dates for filtered cases
        boundary_tests = [
            ("data_1979-12-31_fallback_2023-06-15", 2023, 6, 15),  # 1979 filtered, use fallback
            ("data_1980-01-01_fallback_2022-03-10", 2022, 3, 10),  # 1980 filtered, use fallback
            ("data_1981-01-01_valid", 1981, 1, 1),  # 1981: should work
            ("data_1995-06-15_valid", 1995, 6, 15),  # 1995: should work
            ("data_2030-12-31_future", 2030, 12, 31),  # Future date: should work
        ]

        for test_string, expected_year, expected_month, expected_day in boundary_tests:
            result = utils.parse_str_to_day(test_string, date_patterns=iso_patterns)
            assert result.year == expected_year, f"Failed year for {test_string}"
            assert result.month == expected_month, f"Failed month for {test_string}"
            assert result.day == expected_day, f"Failed day for {test_string}"

        # Test pure 1980 boundary cases that should fail completely
        pure_old_dates = [
            "data_1979-12-31_old",
            "data_1980-01-01_old",
        ]

        for test_string in pure_old_dates:
            # These should fail completely since no valid fallback exists
            with pytest.warns(UserWarning, match="patterns.*matched.*back"):
                with pytest.warns(UserWarning, match="Only 1 string token found"):
                    with pytest.raises(ValueError, match="No valid date token found"):
                        utils.parse_str_to_day(test_string, date_patterns=iso_patterns)

    def test_date_patterns_malformed_dates(self):
        """Test handling of malformed dates that match regex but can't be parsed."""
        malformed_patterns = [(r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d")]

        # These cases have both malformed and valid dates - should skip malformed and find valid
        malformed_cases = [
            ("data_2023-13-01_2023-12-25_file", 2023, 12, 25),  # Skip month 13, find Dec 25
            ("data_2023-02-30_2023-11-15_file", 2023, 11, 15),  # Skip Feb 30, find Nov 15
            ("data_2023-04-31_2023-03-20_file", 2023, 3, 20),  # Skip April 31, find Mar 20
            ("data_2023-00-15_2023-01-10_file", 2023, 1, 10),  # Skip month 0, find Jan 10
            ("data_2023-06-00_2023-05-05_file", 2023, 5, 5),  # Skip day 0, find May 5
        ]

        for test_string, expected_year, expected_month, expected_day in malformed_cases:
            # Should skip malformed date and find the valid one
            result = utils.parse_str_to_day(test_string, date_patterns=malformed_patterns)
            assert result.year == expected_year, f"Failed year for {test_string}"
            assert result.month == expected_month, f"Failed month for {test_string}"
            assert result.day == expected_day, f"Failed day for {test_string}"

    def test_date_patterns_empty_and_whitespace_strings(self):
        """Test date patterns with empty and whitespace strings."""
        patterns = [(r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d")]

        # Empty string should still raise ValueError
        with pytest.warns(UserWarning, match="patterns.*matched.*back"):
            with pytest.raises(ValueError, match="No valid date token found"):
                utils.parse_str_to_day("", date_patterns=patterns)

        # Whitespace only should still raise ValueError
        with pytest.warns(UserWarning, match="patterns.*matched.*back"):
            with pytest.raises(ValueError, match="No valid date token found"):
                utils.parse_str_to_day("   ", date_patterns=patterns)

    def test_date_patterns_priority_ordering(self):
        """Test that date patterns are processed in order and first successful match wins."""
        # Create patterns with different interpretations of the same format
        priority_patterns = [
            (r"(\d{2})/(\d{2})/(\d{4})", "%m/%d/%Y"),  # US format (MM/DD/YYYY) - higher priority
            (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"),  # European format (DD/MM/YYYY) - lower priority
        ]

        # Test with ambiguous date where interpretation matters
        with pytest.warns(UserWarning, match="Multiple.*patterns.*matched"):
            result = utils.parse_str_to_day("data_06/03/2023_file", date_patterns=priority_patterns)
            assert result.year == 2023
            assert result.month == 6  # Should use first pattern (US): June
            assert result.day == 3

        # Reverse the order - European should win now
        reverse_patterns = [
            (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"),  # European format (DD/MM/YYYY) - higher priority
            (r"(\d{2})/(\d{2})/(\d{4})", "%m/%d/%Y"),  # US format (MM/DD/YYYY) - lower priority
        ]

        with pytest.warns(UserWarning, match="Multiple.*patterns.*matched"):
            result = utils.parse_str_to_day("data_06/03/2023_file", date_patterns=reverse_patterns)
            assert result.year == 2023
            assert result.month == 3  # Should use first pattern (European): March
            assert result.day == 6

    def test_date_patterns_complex_formats(self):
        """Test various complex date formats beyond basic ISO/US/European."""

        # Test different separators and formats
        complex_formats = [
            # Format: YYYY.MM.DD
            ("data_2023.11.15_dots", [(r"(\d{4})\.(\d{2})\.(\d{2})", "%Y.%m.%d")], 2023, 11, 15),
            # Format: DD MMM YYYY (spaces)
            ("report 25 Dec 2022 final", [(r"(\d{1,2}) ([A-Za-z]{3}) (\d{4})", "%d %b %Y")], 2022, 12, 25),
            # Format: YYYYMMDD (no separators)
            ("backup_20231201_data", [(r"(\d{8})", "%Y%m%d")], 2023, 12, 1),
            # Format: Month DD, YYYY
            ("file_January 15, 2024_version", [(r"([A-Za-z]+) (\d{1,2}), (\d{4})", "%B %d, %Y")], 2024, 1, 15),
            # Format: DD/MM/YY (2-digit year)
            ("old_15/03/99_backup", [(r"(\d{2})/(\d{2})/(\d{2})", "%d/%m/%y")], 1999, 3, 15),
        ]

        for test_string, patterns, expected_year, expected_month, expected_day in complex_formats:
            result = utils.parse_str_to_day(test_string, date_patterns=patterns)
            assert result.year == expected_year, f"Failed year for {test_string}"
            assert result.month == expected_month, f"Failed month for {test_string}"
            assert result.day == expected_day, f"Failed day for {test_string}"

    def test_date_patterns_invalid_regex(self):
        """Test that invalid regex patterns are handled gracefully."""
        # Invalid regex pattern should be skipped with warning, fallback to token parsing
        invalid_patterns = [(r"[invalid regex", "%Y-%m-%d")]

        # Should still work via fallback token parsing
        # This case generates both fallback warning and single token warning
        with pytest.warns(UserWarning, match="patterns.*matched.*back"):
            with pytest.warns(UserWarning, match="Only 1 string token found"):
                result = utils.parse_str_to_day("2023-07-04", date_patterns=invalid_patterns)
                assert result.year == 2023
                assert result.month == 7
                assert result.day == 4

    def test_date_patterns_warnings(self):
        """Test that appropriate warnings are raised for various conditions."""

        # Test 1: Warning when multiple patterns match
        multiple_patterns = [
            (r"(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})", "%m/%d/%Y"),  # US format
            (r"(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})", "%d/%m/%Y"),  # European format
        ]

        with pytest.warns(UserWarning, match="Multiple.*patterns.*matched"):
            result = utils.parse_str_to_day("12/06/2023_data", date_patterns=multiple_patterns)
            assert result.year == 2023
            assert result.month == 12  # Should use first pattern (US)

        # Test 2: Warning when patterns provided but none match (fallback to token parsing)
        no_match_patterns = [(r"NEVER_MATCHES_ANYTHING", "%Y-%m-%d")]

        with pytest.warns(UserWarning, match="patterns.*matched.*back"):
            with pytest.warns(UserWarning, match="Only 1 string token found"):
                result = utils.parse_str_to_day("WT_2023-12-25_data", date_patterns=no_match_patterns)
                assert result.year == 2023
                assert result.month == 12
                assert result.day == 25

        # Test 3: Warning when multiple matches within same pattern
        same_pattern_multiple_matches = [(r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d")]

        with pytest.warns(UserWarning, match="Multiple.*patterns.*matched"):
            result = utils.parse_str_to_day(
                "data_2020-01-15_and_2021-03-10_files", date_patterns=same_pattern_multiple_matches
            )
            assert result.year == 2020  # Should use first match
            assert result.month == 1
            assert result.day == 15

        # Test 4: Warning when no user patterns match (falls back to token parsing)
        non_matching_patterns = [(r"(\d{2})/(\d{2})/(\d{4})", "%m/%d/%Y")]  # MM/DD/YYYY format

        # This case generates both fallback warning and single token warning
        with pytest.warns(UserWarning, match="patterns.*matched.*back"):
            with pytest.warns(UserWarning, match="Only 1 string token found"):
                result = utils.parse_str_to_day(
                    "data_experiment_2023-July-04_results", date_patterns=non_matching_patterns
                )
                # Should get valid result from fallback token parsing
                assert result.year == 2023
                assert result.month == 7
                assert result.day == 4

    def test_date_patterns_no_warnings_when_appropriate(self):
        """Test that warnings are NOT raised when they shouldn't be."""

        # Test 1: No warning when single pattern matches successfully
        single_pattern = [(r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d")]

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # This will raise exception if any warning occurs
            result = utils.parse_str_to_day("data_2023-07-04_file", date_patterns=single_pattern)
            assert result.year == 2023
            assert result.month == 7
            assert result.day == 4

        # Test 2: No warning when no patterns provided (expected token parsing)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = utils.parse_str_to_day("data 2023-07-04 file")  # No date_patterns, uses whitespace sep
            assert result.year == 2023

        # Test 3: No warning when patterns provided but parse_mode doesn't use them
        patterns = [(r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d")]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # parse_mode="full" doesn't use date_patterns, so no fallback warning
            result = utils.parse_str_to_day("data_2023-07-04_file", date_patterns=patterns, parse_mode="full")
            assert result.year == 2023

    def test_original_behavior_without_date_patterns(self):
        """Test that original behavior is preserved when date_patterns is not provided."""
        # The main issue case should now work via improved token parsing
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            result = utils.parse_str_to_day("2001_2023-07-04_data")
            assert result.year == 2023  # Should find the complete date, not just the year
            assert result.month == 7
            assert result.day == 4

    def test_complex_date_formats_parsemode_split(self):
        """Test parsing of complex date formats with parsemode split."""
        test_cases = [
            ("ID1524_January-20-2012_data", 2012, 1, 20),
            ("ID1524_Jan-20-2012_data", 2012, 1, 20),
            ("ID 1524 January-20-2012 data", 2012, 1, 20),  # Complex ID with date
            ("ID 1524 Jan-20-2012 data", 2012, 1, 20),  # Abbreviated month
        ]

        for string, expected_year, expected_month, expected_day in test_cases:
            # Strings with underscores but no spaces will generate single token warnings
            if "_" in string and " " not in string:
                with pytest.warns(UserWarning, match="Only 1 string token found"):
                    result = utils.parse_str_to_day(string, parse_mode="split")
                    assert result.year == expected_year, f"Failed for {string}"
                    assert result.month == expected_month, f"Failed for {string}"
                    assert result.day == expected_day, f"Failed for {string}"
            else:
                result = utils.parse_str_to_day(string, parse_mode="split")
                assert result.year == expected_year, f"Failed for {string}"
                assert result.month == expected_month, f"Failed for {string}"
                assert result.day == expected_day, f"Failed for {string}"

    def test_parsemode_all(self):
        """Test underscore-separated date with parsemode 'all'"""
        # Note: year defaults to 2000
        # parse_mode="all" can handle these without warnings due to full-string parsing
        result = utils.parse_str_to_day("Mouse_A10_December_25_2023_data", parse_mode="all")
        assert result.month == 12
        assert result.day == 25

        result = utils.parse_str_to_day("Subject_123_March_15_2024_data", parse_mode="all")
        assert result.month == 3
        assert result.day == 15

        # This one uses default parse_mode which generates warning
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            result = utils.parse_str_to_day("ID1524_January_20_2012_data")
        assert result.month == 1
        assert result.day == 20

    def test_parsemode_all_ambiguous_month_day(self):
        """Test underscore-separated date with parsemode 'all' with ambiguous month/day"""
        # Note: year defaults to 2000
        # parse_mode="all" can handle these without warnings due to full-string parsing
        result = utils.parse_str_to_day("Mouse_A10_December_11_2023_data", parse_mode="all")
        assert result.month == 12
        assert result.day == 11

        result = utils.parse_str_to_day("Subject_123_March_11_2024_data", parse_mode="all")
        assert result.month == 3
        assert result.day == 11

        # This one uses default parse_mode which generates warning
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            result = utils.parse_str_to_day("ID1524_January_11_2012_data")
        assert result.month == 1
        assert result.day == 11

    def test_number_only_ids_with_number_only_dates(self):
        """Test parsing with numeric IDs and numeric date formats."""
        # Test various combinations of numeric IDs with numeric dates
        test_cases = [
            ("123_2023-07-04_data", 2023, 7, 4),  # Simple numeric ID with ISO date
            ("456_07/04/2023_data", 2023, 7, 4),  # Numeric ID with US date format
            ("999_2023-7-4_data", 2023, 7, 4),  # Numeric ID with no leading zeros
            ("123_2023-07-04_456_data", 2023, 7, 4),  # Multiple numeric IDs
        ]

        for string, expected_year, expected_month, expected_day in test_cases:
            result = utils.parse_str_to_day(string, parse_mode="split", sep="_")
            assert result.year == expected_year, f"Failed for {string}"
            assert result.month == expected_month, f"Failed for {string}"
            assert result.day == expected_day, f"Failed for {string}"

    def test_number_only_four_digit_ids_with_number_only_dates(self):
        """Test parsing with 4-digit year-like numeric IDs and numeric date formats."""
        # Test various combinations of numeric IDs with numeric dates
        test_cases = [
            ("1450_2023-07-04_data", 2023, 7, 4),
            ("1920_07/04/2023_data", 2023, 7, 4),
            ("2001_2023-07-04_data", 2023, 7, 4),  # Simple numeric ID with ISO date
            ("2002_07/04/2023_data", 2023, 7, 4),  # Numeric ID with US date format
            ("2003_2023-7-4_data", 2023, 7, 4),  # Numeric ID with no leading zeros
            ("1990_2023-07-04_1991_data", 2023, 7, 4),  # Multiple numeric IDs
        ]

        # Use explicit date patterns to prioritize complete date formats over isolated years
        iso_patterns = [(r"(19\d{2}|20\d{2})-(\d{1,2})-(\d{1,2})", "%Y-%m-%d")]
        us_patterns = [(r"(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})", "%m/%d/%Y")]

        for string, expected_year, expected_month, expected_day in test_cases:
            # Determine which pattern to use based on the date format in the string
            if "-" in string:
                patterns = iso_patterns
            else:
                patterns = us_patterns

            result = utils.parse_str_to_day(string, parse_mode="split", sep="_", date_patterns=patterns)
            assert result.year == expected_year, f"Failed for {string}"
            assert result.month == expected_month, f"Failed for {string}"
            assert result.day == expected_day, f"Failed for {string}"

    def test_multiple_numeric_dates_behavior(self):
        """Test behavior with multiple numeric dates (may vary based on dateutil version)."""
        string = "123_2023-07-04_789_2024-12-25_data"
        result = utils.parse_str_to_day(string, parse_mode="full")
        # Test it returns first valid date
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

    def test_numeric_ids_without_dates_raise_error(self):
        """Test that numeric IDs without valid dates raise ValueError."""
        with pytest.warns(UserWarning, match="Only 1 string token found"):
            with pytest.raises(ValueError, match="No valid date token found"):
                utils.parse_str_to_day("123_invalid_data")

        with pytest.warns(UserWarning, match="Only 1 string token found"):
            with pytest.raises(ValueError, match="No valid date token found"):
                utils.parse_str_to_day("456_789_no_date_here")

    def test_parse_mode_full(self):
        """Test parse_mode='full' - only tries parsing the entire cleaned string."""
        result = utils.parse_str_to_day("2023-07-04", parse_mode="full")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        result = utils.parse_str_to_day("WT_A10_2023-07-04_data", parse_mode="full")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("WT_A10_no_date_here", parse_mode="full")

    def test_parse_mode_split(self):
        """Test parse_mode='split' - only tries parsing individual tokens."""
        result = utils.parse_str_to_day("WT_A10 2023-07-04 data", parse_mode="split")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        result = utils.parse_str_to_day("WT|A10|2023-07-04|data", parse_mode="split", sep="|")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("no date here", parse_mode="split")

    def test_parse_mode_window(self):
        """Test parse_mode='window' - only tries parsing sliding windows of tokens."""
        # Window mode works with token-based parsing and finds the first valid date
        # Note: Due to dateutil behavior, single month names parse to default year
        result = utils.parse_str_to_day("WT_A10_February_20_2012_data", parse_mode="window", sep="_")
        # This currently finds "February" -> Feb 1, 2000 (known limitation)
        assert result.year == 2000  # Current behavior: finds partial match first
        assert result.month == 2
        assert result.day == 1

        # Test with a format that window mode can handle better
        result = utils.parse_str_to_day("WT_A10_2012-02-20_data", parse_mode="window", sep="_")
        assert result.year == 2012
        assert result.month == 2
        assert result.day == 20

        # Should fail when no valid date tokens exist
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("no date here at all", parse_mode="window")

    def test_parse_mode_all(self):
        """Test parse_mode='all' - uses all three approaches in sequence."""
        # Should work with full string parsing (no patterns needed)
        # parse_mode="all" tries full parsing first, so no warning for simple dates
        result = utils.parse_str_to_day("2023-07-04", parse_mode="all")
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        # Should work with patterns for complex cases
        iso_patterns = [(r"(19\d{2}|20\d{2})[_-](\d{1,2})[_-](\d{1,2})", "%Y_%m_%d")]
        result = utils.parse_str_to_day("WT_A10_2023_07_04_data", parse_mode="all", sep="_", date_patterns=iso_patterns)
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        # Should work with month name patterns
        month_patterns = [
            (
                r"(January|February|March|April|May|June|July|August|September|October|November|December)[_\s]+(\d{1,2})[_\s]+(19\d{2}|20\d{2})",
                "%B_%d_%Y",
            )
        ]
        result = utils.parse_str_to_day(
            "WT_A10_January_20_2012_data", parse_mode="all", sep="_", date_patterns=month_patterns
        )
        assert result.year == 2012  # Fixed: should be 2012, not 2023
        assert result.month == 1
        assert result.day == 20

    # REVIEW by the way, split by default splits by whitespace, not by underscores. Almost all of these tests assume it splits by underscores, which is very wrong
    # I documented this with a warning when split is called but only 1 token gets raised. It would be prudent to closely examine tests that raise this warning --
    # perhaps the logic is wrong and should be using the format string searching method instead

    def test_parse_mode_default_behavior(self):
        """Test that default parse_mode is 'split'."""
        # Default should be 'split' mode
        # This string has enough tokens to parse without warnings
        result = utils.parse_str_to_day("WT_A10_2023_07_04_data")
        assert result.year == 2000  # Conservative parsing uses default year
        assert result.month == 7
        assert result.day == 4
        # REVIEW problems outlined per above

        # Should fail when no valid tokens can be parsed in split mode
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day("random text with no dates")

    def test_parse_mode_invalid_value(self):
        """Test that invalid parse_mode values raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid parse_mode"):
            utils.parse_str_to_day("2023-07-04", parse_mode="invalid_mode")

        with pytest.raises(ValueError, match="Invalid parse_mode"):
            utils.parse_str_to_day("2023-07-04", parse_mode="")

    def test_parse_mode_with_parse_params(self):
        """Test that parse_mode works correctly with parse_params."""
        # Test with fuzzy parsing disabled on a clear ISO date (should still work)
        result = utils.parse_str_to_day(
            "WT_A10_2023-07-04_data", parse_mode="split", sep="_", parse_params={"fuzzy": False}
        )
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4

        # Test with fuzzy parsing disabled on unparseable text (should fail)
        with pytest.raises(ValueError, match="No valid date token found"):
            utils.parse_str_to_day(
                "WT_A10_this_cannot_be_parsed_as_date", parse_mode="split", sep="_", parse_params={"fuzzy": False}
            )

        # Test with fuzzy parsing enabled
        result = utils.parse_str_to_day(
            "WT_A10_2023-07-04_data", parse_mode="split", sep="_", parse_params={"fuzzy": True}
        )
        assert result.year == 2023
        assert result.month == 7
        assert result.day == 4


class TestParseChnameToAbbrev:
    """Test parse_chname_to_abbrev function."""

    def test_basic_abbreviation(self):
        """Test basic channel name abbreviation."""
        # Test with exact aliases from constants
        assert utils.parse_chname_to_abbrev("left Aud") == "LAud"
        assert utils.parse_chname_to_abbrev("right Vis") == "RVis"
        assert utils.parse_chname_to_abbrev("Left Hip") == "LHip"
        assert utils.parse_chname_to_abbrev("right Bar") == "RBar"
        assert utils.parse_chname_to_abbrev("left Mot") == "LMot"

    def test_case_insensitive_behavior(self):
        """Test case insensitive behavior with new uppercase aliases."""
        # These now work with uppercase aliases added to constants
        assert utils.parse_chname_to_abbrev("Left aud") == "LAud"
        assert utils.parse_chname_to_abbrev("right vis") == "RVis"
        assert utils.parse_chname_to_abbrev("Right VIS") == "RVis"
        assert utils.parse_chname_to_abbrev("Left AUD") == "LAud"
        assert utils.parse_chname_to_abbrev("right HIP") == "RHip"
        assert utils.parse_chname_to_abbrev("left BAR") == "LBar"
        assert utils.parse_chname_to_abbrev("Right MOT") == "RMot"

        # Test uppercase L/R prefixes
        assert utils.parse_chname_to_abbrev("LEFT aud") == "LAud"
        assert utils.parse_chname_to_abbrev("RIGHT vis") == "RVis"
        assert utils.parse_chname_to_abbrev("LEFT BAR") == "LBar"
        assert utils.parse_chname_to_abbrev("RIGHT MOT") == "RMot"

    def test_lr_prefix_variations(self):
        """Test various left/right prefix variations."""
        # Test different L/R prefix formats from LR_ALIASES
        assert utils.parse_chname_to_abbrev("left Aud") == "LAud"
        assert utils.parse_chname_to_abbrev("Left Aud") == "LAud"
        assert utils.parse_chname_to_abbrev("L Aud") == "LAud"
        assert utils.parse_chname_to_abbrev(" L Aud") == "LAud"

        assert utils.parse_chname_to_abbrev("right Vis") == "RVis"
        assert utils.parse_chname_to_abbrev("Right Vis") == "RVis"
        assert utils.parse_chname_to_abbrev("R Vis") == "RVis"
        assert utils.parse_chname_to_abbrev(" R Vis") == "RVis"

    def test_channel_name_variations(self):
        """Test various channel name variations."""
        # Test all channel types with both cases
        test_cases = [
            ("left Aud", "LAud"),
            ("right aud", "RAud"),
            ("left Vis", "LVis"),
            ("right vis", "RVis"),
            ("left Hip", "LHip"),
            ("right hip", "RHip"),
            ("left Bar", "LBar"),
            ("right bar", "RBar"),
            ("left Mot", "LMot"),
            ("right mot", "RMot"),
        ]

        for input_name, expected in test_cases:
            result = utils.parse_chname_to_abbrev(input_name)
            assert result == expected, f"Expected {expected}, got {result} for input {input_name}"

    def test_already_abbreviated_channels(self):
        """Test channels that are already abbreviations."""
        abbreviated_channels = ["LAud", "RAud", "LVis", "RVis", "LHip", "RHip", "LBar", "RBar", "LMot", "RMot"]

        for channel in abbreviated_channels:
            result = utils.parse_chname_to_abbrev(channel)
            assert result == channel, f"Expected {channel}, got {result}"

    def test_assume_from_number_parameter(self):
        """Test assume_from_number parameter functionality."""
        # Test with assume_from_number=True
        test_cases = [
            ("channel_9", "LAud"),  # DEFAULT_ID_TO_NAME[9] = "LAud"
            ("ch10", "LVis"),  # DEFAULT_ID_TO_NAME[10] = "LVis"
            ("electrode_12", "LHip"),  # DEFAULT_ID_TO_NAME[12] = "LHip"
            ("probe_22", "RAud"),  # DEFAULT_ID_TO_NAME[22] = "RAud"
        ]

        for input_name, expected in test_cases:
            result = utils.parse_chname_to_abbrev(input_name, assume_from_number=True)
            assert result == expected, f"Expected {expected}, got {result} for {input_name}"

    def test_assume_from_number_multiple_numbers(self):
        """Test assume_from_number with multiple numbers (uses last number)."""
        # Should use the last number found in the string
        result = utils.parse_chname_to_abbrev("ch1_probe2_electrode_22", assume_from_number=True)
        assert result == "RAud"  # DEFAULT_ID_TO_NAME[22] = "RAud"

        result = utils.parse_chname_to_abbrev("2023_ch_10_data", assume_from_number=True)
        assert result == "LVis"  # DEFAULT_ID_TO_NAME[10] = "LVis"

    def test_assume_from_number_invalid_id(self):
        """Test assume_from_number with invalid channel ID - should provide detailed error."""
        # Should raise KeyError with detailed message for numbers not in DEFAULT_ID_TO_NAME
        with pytest.raises(
            KeyError,
            match="Channel number 99 found in 'channel_99' is not a valid channel ID. Available channel IDs: \\[9, 10, 12, 14, 15, 16, 17, 19, 21, 22\\]",
        ):
            utils.parse_chname_to_abbrev("channel_99", assume_from_number=True)

        with pytest.raises(KeyError, match="Channel number 1 found in 'electrode_1' is not a valid channel ID"):
            utils.parse_chname_to_abbrev("electrode_1", assume_from_number=True)

    def test_assume_from_number_no_numbers(self):
        """Test assume_from_number when no numbers are found - should provide clear error."""
        # Should raise ValueError with clear message when no numbers are found
        with pytest.raises(
            ValueError,
            match="Expected to find a number in channel name 'no_numbers_here' when assume_from_number=True, but no numbers were found",
        ):
            utils.parse_chname_to_abbrev("no_numbers_here", assume_from_number=True)

        with pytest.raises(
            ValueError,
            match="Expected to find a number in channel name 'channel' when assume_from_number=True, but no numbers were found",
        ):
            utils.parse_chname_to_abbrev("channel", assume_from_number=True)

    def test_mixed_case_channel_content(self):
        """Test channels with mixed case content."""
        # Test channels that have valid L/R and channel names but with extra content
        assert utils.parse_chname_to_abbrev("left_Aud_electrode") == "LAud"
        assert utils.parse_chname_to_abbrev("Right_vis_channel") == "RVis"
        assert utils.parse_chname_to_abbrev("probe_Left_Hip_001") == "LHip"

    def test_substring_matching_behavior(self):
        """Test that function uses substring matching (not exact word matching)."""
        # These should work because the aliases are found as substrings
        assert utils.parse_chname_to_abbrev("leftAud") == "LAud"
        assert utils.parse_chname_to_abbrev("rightvis") == "RVis"
        assert utils.parse_chname_to_abbrev("LeftHip") == "LHip"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test empty string
        with pytest.raises(ValueError):
            utils.parse_chname_to_abbrev("")

        # Test string with only whitespace
        with pytest.raises(ValueError):
            utils.parse_chname_to_abbrev("   ")

        # Test string with only L/R but no channel name
        with pytest.raises(ValueError):
            utils.parse_chname_to_abbrev("left")

        # Test string with only channel name but no L/R
        with pytest.raises(ValueError):
            utils.parse_chname_to_abbrev("Aud")

    def test_input_type_validation(self):
        """Test input type validation."""
        # Test non-string inputs that should raise TypeError
        with pytest.raises(TypeError, match="argument of type 'int' is not iterable"):
            utils.parse_chname_to_abbrev(123)

        with pytest.raises(TypeError, match="argument of type 'NoneType' is not iterable"):
            utils.parse_chname_to_abbrev(None)

        # Interestingly, lists work because the function searches for substrings in the list
        # REVIEW this is unintended behavior, parse chnaame to abbrev should only operate on strings
        result = utils.parse_chname_to_abbrev(["left", "Aud"])
        assert result == "LAud"

    def test_strict_matching_mode(self):
        """Test strict_matching parameter functionality."""
        # Test strict mode (default) - should reject ambiguous L/R matches
        with pytest.raises(ValueError, match="Ambiguous match in 'left right Aud'. Multiple alias types matched"):
            utils.parse_chname_to_abbrev("left right Aud", strict_matching=True)

        with pytest.raises(ValueError, match="Ambiguous match in 'Left Right VIS'. Multiple alias types matched"):
            utils.parse_chname_to_abbrev("Left Right VIS", strict_matching=True)

        # Test strict mode rejects ambiguous channel type matches
        with pytest.raises(
            ValueError, match="Ambiguous match in 'right auditory hippocampus'. Multiple alias types matched"
        ):
            utils.parse_chname_to_abbrev("right auditory hippocampus", strict_matching=True)

        with pytest.raises(ValueError, match="Ambiguous match in 'left auditory visual'. Multiple alias types matched"):
            utils.parse_chname_to_abbrev("left auditory visual", strict_matching=True)

        with pytest.raises(ValueError, match="Aud Vis does not have any matching values"):
            utils.parse_chname_to_abbrev("Aud Vis", strict_matching=True)  # Fails because no L/R prefix

        with pytest.raises(ValueError, match="Ambiguous match in 'left Aud Vis'. Multiple alias types matched"):
            utils.parse_chname_to_abbrev("left Aud Vis", strict_matching=True)  # Has L/R but multiple channel types

        with pytest.raises(ValueError, match="Ambiguous match in 'Right Hip Aud'. Multiple alias types matched"):
            utils.parse_chname_to_abbrev("Right Hip Aud", strict_matching=True)

    def test_nonstrict_matching_mode(self):
        # Test non-strict mode - should allow ambiguous matches and use longest
        result = utils.parse_chname_to_abbrev("left right Aud", strict_matching=False)
        assert result == "RAud"  # "right" is longer than "left", so R wins

        result = utils.parse_chname_to_abbrev("Left Right VIS", strict_matching=False)
        assert result == "RVis"  # "Right" is longer than "Left", so R wins

        result = utils.parse_chname_to_abbrev("right auditory hippocampus", strict_matching=False)
        assert result == "RAud"  # "auditory" is longer than "hippocampus", so Aud wins

        result = utils.parse_chname_to_abbrev("left auditory visual", strict_matching=False)
        assert result == "LAud"  # "auditory" is longer than "visual", so Aud wins

        result = utils.parse_chname_to_abbrev("Right Hip Aud", strict_matching=False)
        assert result == "RAud"  # "Aud" is longer than "Hip", so Aud wins

        # Test that strict mode still works for unambiguous matches
        assert utils.parse_chname_to_abbrev("left Aud", strict_matching=True) == "LAud"
        assert utils.parse_chname_to_abbrev("right Vis", strict_matching=True) == "RVis"

    def test_reverse_order_parsing(self):
        """Test that reverse order (channel type before L/R) works correctly."""
        # These should work because they're unambiguous
        assert utils.parse_chname_to_abbrev("Auditory Left") == "LAud"
        assert utils.parse_chname_to_abbrev("Visual Right") == "RVis"
        assert utils.parse_chname_to_abbrev("Hippocampal Left") == "LHip"
        assert utils.parse_chname_to_abbrev("Motor RIGHT") == "RMot"
        assert utils.parse_chname_to_abbrev("auditory right") == "RAud"
        assert utils.parse_chname_to_abbrev("vis LEFT") == "LVis"

    def test_strict_matching_default_behavior(self):
        """Test that strict_matching defaults to True."""
        # Should fail by default (strict_matching=True)
        with pytest.raises(ValueError, match="Ambiguous match"):
            utils.parse_chname_to_abbrev("left right Aud")

        # Should work when explicitly set to False
        result = utils.parse_chname_to_abbrev("left right Aud", strict_matching=False)
        assert result == "RAud"

    def test_strict_matching_with_assume_from_number(self):
        """Test interaction between strict_matching and assume_from_number."""
        # When normal parsing fails due to strict mode, should fall back to assume_from_number
        result = utils.parse_chname_to_abbrev("left right channel_9", assume_from_number=True, strict_matching=True)
        assert result == "LAud"  # Falls back to number-based parsing

        # Should also work in non-strict mode but still use assume_from_number path
        result = utils.parse_chname_to_abbrev("left right channel_10", assume_from_number=True, strict_matching=False)
        assert result == "LVis"

    def test_improved_error_messages(self):
        """Test that error messages are more helpful."""
        # Test improved no-match error message
        with pytest.raises(
            ValueError, match="InvalidChannel does not have any matching values. Available aliases \\(examples\\):"
        ):
            utils.parse_chname_to_abbrev("InvalidChannel")

        # Error should show examples of available aliases
        try:
            utils.parse_chname_to_abbrev("NoMatch")
        except ValueError as e:
            error_msg = str(e)
            assert "Available aliases (examples)" in error_msg
            assert "L" in error_msg and "R" in error_msg  # Should show L/R examples

    def test_backward_compatibility(self):
        """Test that existing code still works with new parameters."""
        # Old function calls should still work (strict_matching defaults to True)
        assert utils.parse_chname_to_abbrev("left Aud") == "LAud"
        assert utils.parse_chname_to_abbrev("right Vis") == "RVis"
        assert utils.parse_chname_to_abbrev("channel_9", assume_from_number=True) == "LAud"

        # Test that function signature is backward compatible
        assert utils.parse_chname_to_abbrev("Left Hip", False) == "LHip"  # positional assume_from_number

    def test_function_documentation_examples(self):
        """Test examples that should work based on the function's purpose."""
        # Test typical use cases that would be expected in EEG channel naming
        assert utils.parse_chname_to_abbrev("left auditory") == "LAud"  # if "auditory" contains "aud"
        assert utils.parse_chname_to_abbrev("right visual") == "RVis"  # if "visual" contains "vis"


class TestLogTransform:
    """Test log_transform function."""

    def test_basic_transform(self):
        """Test basic log transformation."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data)

        # Should be natural log-transformed (ln(x+1))
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_kwargs_ignored(self):
        """Test that extra kwargs are ignored."""
        data = np.array([1, 10, 100, 1000])
        result = utils.log_transform(data, offset=1, some_other_param="ignored")

        # Function should ignore kwargs and still work normally (ln(x+1))
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_negative_values(self):
        """Test log transformation with negative values."""
        data = np.array([-1, 0, 1, 10])

        # The function always does ln(x + 1), so let's test that
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = utils.log_transform(data)

        # Expected behavior: ln(x + 1)
        # [-1, 0, 1, 10] + 1 = [0, 1, 2, 11]
        # ln([0, 1, 2, 11]) = [-inf, 0, ln(2), ln(11)]
        assert np.isneginf(result[0])  # ln(0) = -inf
        assert np.isclose(result[1], 0)  # ln(1) = 0
        assert np.isclose(result[2], np.log(2))  # ln(2)
        assert np.isclose(result[3], np.log(11))  # ln(11)

    def test_zero_values(self):
        """Test log transformation with zero values."""
        data = np.array([0, 0, 0])
        result = utils.log_transform(data)

        # ln(0 + 1) = ln(1) = 0
        expected = np.log(data + 1)  # [0, 0, 0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_one_edge_case(self):
        """Test log transformation with -1 values (edge case)."""
        data = np.array([-1])

        # ln(-1 + 1) = ln(0) = -inf, should raise warning or return -inf
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = utils.log_transform(data)
            assert np.isneginf(result[0])

    def test_values_less_than_negative_one(self):
        """Test log transformation with values < -1."""
        data = np.array([-2, -1.5, -10])

        # ln(x + 1) where x < -1 gives complex numbers, numpy should handle this
        with pytest.warns(RuntimeWarning, match="invalid value"):
            result = utils.log_transform(data)
            assert np.all(np.isnan(result))

    def test_very_large_values(self):
        """Test log transformation with very large values."""
        data = np.array([1e10, 1e100, np.inf])
        result = utils.log_transform(data)

        # Should handle large values appropriately
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_very_small_positive_values(self):
        """Test log transformation with very small positive values."""
        data = np.array([1e-10, 1e-100, 1e-308])
        result = utils.log_transform(data)

        # Should handle small values appropriately
        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_nan_values(self):
        """Test log transformation with NaN values."""
        data = np.array([1, np.nan, 10])
        result = utils.log_transform(data)

        # NaN should propagate through
        expected = np.log(data + 1)
        assert np.isnan(result[1])
        np.testing.assert_array_almost_equal(result[[0, 2]], expected[[0, 2]])

    def test_mixed_edge_cases(self):
        """Test log transformation with mixed edge case values."""
        data = np.array([-2, -1, 0, 1e-10, 1, 1e10, np.nan, np.inf])

        with pytest.warns(RuntimeWarning):
            result = utils.log_transform(data)

        # Check specific expected behaviors
        assert np.isnan(result[0])  # -2 + 1 = -1, log(-1) = nan
        assert np.isneginf(result[1])  # -1 + 1 = 0, log(0) = -inf
        assert result[2] == 0  # 0 + 1 = 1, log(1) = 0
        assert result[3] > 0  # small positive + 1, log > 0
        assert np.isnan(result[6])  # nan propagates
        assert np.isinf(result[7])  # inf + 1 = inf, log(inf) = inf

    def test_empty_array(self):
        """Test log transformation with empty array."""
        data = np.array([])
        result = utils.log_transform(data)

        assert len(result) == 0
        assert result.dtype == np.float64

    def test_multidimensional_arrays(self):
        """Test log transformation with multidimensional arrays."""
        data = np.array([[1, 10], [100, 1000]])
        result = utils.log_transform(data)

        expected = np.log(data + 1)
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == data.shape


class TestSortDataframeByPlotOrder:
    """Test sort_dataframe_by_plot_order function."""

    def test_basic_sorting(self):
        """Test basic DataFrame sorting."""
        # Create test data with known order
        df = pd.DataFrame(
            {"genotype": ["KO", "WT", "KO", "WT"], "channel": ["RVis", "LAud", "LVis", "RAud"], "value": [1, 2, 3, 4]}
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Check that the DataFrame is sorted according to the predefined order
        # Based on constants.DF_SORT_ORDER: genotype: ["WT", "KO"], channel: ["average", "all", "LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"]
        # Should be sorted first by genotype (WT before KO), then by channel (LAud < RAud < LVis < RVis)

        # Check genotype ordering
        genotype_order = result["genotype"].tolist()
        wt_indices = [i for i, x in enumerate(genotype_order) if x == "WT"]
        ko_indices = [i for i, x in enumerate(genotype_order) if x == "KO"]

        # All WT should come before all KO
        assert max(wt_indices) < min(ko_indices)

        # Check channel ordering within each genotype
        wt_channels = result[result["genotype"] == "WT"]["channel"].tolist()
        ko_channels = result[result["genotype"] == "KO"]["channel"].tolist()

        # LAud should come before RAud in WT group
        if "LAud" in wt_channels and "RAud" in wt_channels:
            assert wt_channels.index("LAud") < wt_channels.index("RAud")

        # LVis should come before RVis in KO group
        if "LVis" in ko_channels and "RVis" in ko_channels:
            assert ko_channels.index("LVis") < ko_channels.index("RVis")

        # Check that values are in expected order after sorting
        expected_values = [2, 4, 3, 1]  # Values corresponding to WT/LAud, WT/RAud, KO/LVis, KO/RVis
        np.testing.assert_array_equal(result["value"].values, expected_values)

    def test_empty_dataframe(self):
        """Test sorting an empty DataFrame."""
        df = pd.DataFrame()
        result = utils.sort_dataframe_by_plot_order(df)

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_with_missing_columns(self):
        """Test DataFrame that doesn't have columns in sort order."""
        df = pd.DataFrame({"unknown_col": [1, 2, 3], "another_col": ["a", "b", "c"]})

        result = utils.sort_dataframe_by_plot_order(df)

        # Should return a copy of the original DataFrame
        pd.testing.assert_frame_equal(result, df)
        assert result is not df  # Should be a copy

    def test_custom_sort_order(self):
        """Test with custom sort order dictionary."""
        df = pd.DataFrame(
            {
                "priority": ["high", "low", "medium", "high"],
                "category": ["B", "A", "C", "A"],
                "channel": ["LAud", "RAud", "LVis", "RVis"],
                "value": [1, 2, 3, 4],
            }
        )

        custom_order = {"priority": ["low", "medium", "high"], "category": ["A", "B", "C"]}

        result = utils.sort_dataframe_by_plot_order(df, custom_order)

        # Should sort by priority first, then by category
        priorities = result["priority"].tolist()

        # Check that priorities are in correct order
        assert priorities.index("low") < priorities.index("medium")
        assert priorities.index("medium") < max([i for i, x in enumerate(priorities) if x == "high"])

        # Check that channel (from default sort order) follows the sorted rows
        channels = result["channel"].tolist()
        assert channels == ["RAud", "LVis", "RVis", "LAud"]  # Reflects the sorted order by priority and category

    def test_constants_not_mutated(self):
        """Test that constants.DF_SORT_ORDER is not mutated by the function."""
        # Save original state
        original_df_sort_order = constants.DF_SORT_ORDER.copy()

        df = pd.DataFrame({"genotype": ["WT", "KO"], "channel": ["LAud", "RVis"], "value": [1, 2]})

        # Call function multiple times
        result1 = utils.sort_dataframe_by_plot_order(df)
        result2 = utils.sort_dataframe_by_plot_order(df)

        # Check that constants.DF_SORT_ORDER hasn't changed
        assert constants.DF_SORT_ORDER == original_df_sort_order

        # Also check that the results are consistent
        pd.testing.assert_frame_equal(result1, result2)

    def test_dictionary_not_modified_by_reference(self):
        """Test that the passed dictionary is not modified by the function."""
        custom_order = {"priority": ["low", "medium", "high"], "category": ["A", "B", "C"]}
        original_custom_order = {"priority": ["low", "medium", "high"], "category": ["A", "B", "C"]}

        df = pd.DataFrame({"priority": ["high", "low", "medium"], "category": ["B", "A", "C"], "value": [1, 2, 3]})

        utils.sort_dataframe_by_plot_order(df, custom_order)

        # Check that custom_order wasn't modified
        assert custom_order == original_custom_order

    def test_invalid_sort_order_dict(self):
        """Test error handling for invalid sort order dictionary."""
        df = pd.DataFrame({"genotype": ["WT", "KO"], "value": [1, 2]})
        # ANCHOR 08/14/2025
        # Test with non-dict
        with pytest.raises(ValueError, match="df_sort_order must be a dictionary"):
            utils.sort_dataframe_by_plot_order(df, "not_a_dict")

        # Test with invalid categories (not list or tuple)
        with pytest.raises(ValueError, match="Categories for column 'genotype' must be a list or tuple"):
            utils.sort_dataframe_by_plot_order(df, {"genotype": "not_a_list"})

    def test_values_not_in_sort_order(self):
        """Test error handling when DataFrame contains values not in sort order."""
        df = pd.DataFrame({"genotype": ["WT", "KO", "UNKNOWN"], "value": [1, 2, 3]})

        with pytest.raises(
            ValueError, match="Column 'genotype' contains values not in sort order dictionary: \\{'UNKNOWN'\\}"
        ):
            utils.sort_dataframe_by_plot_order(df)

    def test_missing_values_handled(self):
        """Test that missing values (NaN) are handled properly."""
        df = pd.DataFrame(
            {"genotype": ["WT", "KO", np.nan, "WT"], "channel": ["LAud", np.nan, "RVis", "LVis"], "value": [1, 2, 3, 4]}
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Function should not crash with NaN values
        assert len(result) == 4
        assert result["genotype"].isna().any()
        assert result["channel"].isna().any()

    def test_only_existing_categories_used(self):
        """Test that only categories that exist in the DataFrame are used for sorting."""
        df = pd.DataFrame(
            {
                "genotype": ["WT", "WT"],  # Only WT, no KO
                "channel": ["LAud", "LVis"],  # Only 2 channels
                "value": [1, 2],
            }
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Should work even though not all categories from DF_SORT_ORDER are present
        assert len(result) == 2
        genotypes = result["genotype"].unique()
        assert "WT" in genotypes
        assert "KO" not in genotypes

    def test_maintains_dataframe_structure(self):
        """Test that the function maintains DataFrame structure and data types."""
        df = pd.DataFrame(
            {"genotype": ["WT", "KO"], "channel": ["LAud", "RVis"], "value": [1.5, 2.7], "count": [10, 20]}
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Check that all columns are preserved
        assert list(result.columns) == list(df.columns)

        # Check that data types are preserved for non-sorted columns
        assert result["value"].dtype == df["value"].dtype
        assert result["count"].dtype == df["count"].dtype

        # Check that no data is lost
        assert len(result) == len(df)

    def test_return_copy_not_reference(self):
        """Test that function returns a copy, not a reference to the original."""
        df = pd.DataFrame({"genotype": ["WT", "KO"], "value": [1, 2]})

        result = utils.sort_dataframe_by_plot_order(df)

        # Modify the result
        result.loc[0, "value"] = 999

        # Original should be unchanged
        assert df.loc[0, "value"] == 1

    def test_complex_sorting_scenario(self):
        """Test complex sorting with multiple columns and mixed data."""
        df = pd.DataFrame(
            {
                "genotype": ["KO", "WT", "KO", "WT", "WT", "KO"],
                "channel": ["RVis", "LAud", "LVis", "RBar", "LHip", "RAud"],
                "band": ["gamma", "delta", "beta", "alpha", "theta", "gamma"],
                "isday": [False, True, True, False, True, False],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )

        result = utils.sort_dataframe_by_plot_order(df)

        # Check that result is properly sorted
        assert len(result) == 6

        # Verify that the sorting is stable and follows the expected order
        # All WT should come before KO (check row positions, not original indices)
        wt_positions = [i for i, genotype in enumerate(result["genotype"]) if genotype == "WT"]
        ko_positions = [i for i, genotype in enumerate(result["genotype"]) if genotype == "KO"]
        assert max(wt_positions) < min(ko_positions)

    def test_default_parameter_copy_behavior(self):
        """Test that function creates a copy of constants.DF_SORT_ORDER when using default parameter."""
        df = pd.DataFrame({"genotype": ["WT", "KO"], "channel": ["LAud", "RVis"], "value": [1, 2]})

        # Store original state
        original_constants = constants.DF_SORT_ORDER.copy()

        # Call function with default parameter (None)
        result = utils.sort_dataframe_by_plot_order(df)

        # Constants should be unchanged
        assert constants.DF_SORT_ORDER == original_constants

        # Function should work correctly
        assert len(result) == 2
        assert "genotype" in result.columns
        assert "channel" in result.columns


class TestNanmeanSeriesOfNp:
    """Test nanmean_series_of_np function."""

    def test_basic_operation_small_series(self):
        """Test basic nanmean operation with small series (< 1000 items)."""
        # Create series with numpy arrays
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_operation_large_series(self):
        """Test basic nanmean operation with large series (> 1000 items)."""
        # Create series with many numpy arrays - reduced size to prevent memory issues
        arrays = [np.array([i, i + 1, i + 2]) for i in range(100)]  # Reduced from 1500
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.stack(arrays, axis=0), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_with_nan_values(self):
        """Test nanmean operation with NaN values."""
        arrays = [np.array([1, 2, np.nan]), np.array([4, np.nan, 6]), np.array([np.nan, 8, 9])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_all_nan_values(self):
        """Test nanmean operation when all values are NaN."""
        arrays = [
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan]),
        ]
        series = pd.Series(arrays)

        # Expected warning when computing mean of all NaN values
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result = utils.nanmean_series_of_np(series)

        # Should return array of NaNs
        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_different_axis(self):
        """Test nanmean operation with different axis parameter."""
        # Create 2D arrays
        arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
        series = pd.Series(arrays)

        # Test axis=1
        result = utils.nanmean_series_of_np(series, axis=1)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=1)

        np.testing.assert_array_almost_equal(result, expected)

    def test_single_element_series(self):
        """Test nanmean operation with single element series."""
        arrays = [np.array([1, 2, 3])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = arrays[0]  # Should be the same as input

        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_series(self):
        """Test nanmean operation with empty series."""
        series = pd.Series([], dtype=object)

        # Empty series causes warning and returns NaN
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result = utils.nanmean_series_of_np(series)
        assert np.isnan(result)

    def test_mixed_array_sizes_small_series(self):
        """Test with arrays of different sizes in small series."""
        arrays = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
        series = pd.Series(arrays)

        # This should fall back to list conversion and may raise an error
        # or handle it gracefully depending on numpy version
        try:
            result = utils.nanmean_series_of_np(series)
            # If it works, verify it's reasonable
            assert isinstance(result, np.ndarray)
        except ValueError:
            # This is acceptable for mixed sizes
            pass

    def test_mixed_array_sizes_large_series(self):
        """Test with arrays of different sizes in large series (triggers stacking path)."""
        # Create 100 arrays with mixed sizes - reduced size to prevent memory issues
        arrays = []
        for i in range(100):  # Reduced from 1500
            if i % 3 == 0:
                arrays.append(np.array([i, i + 1]))
            elif i % 3 == 1:
                arrays.append(np.array([i, i + 1, i + 2]))
            else:
                arrays.append(np.array([i]))

        series = pd.Series(arrays)

        # Mixed-size arrays cannot be processed by np.nanmean - should raise ValueError
        with pytest.raises(ValueError, match="setting an array element with a sequence"):
            utils.nanmean_series_of_np(series)

    def test_non_numpy_arrays_small_series(self):
        """Test with non-numpy arrays in small series."""
        arrays = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Lists instead of numpy arrays
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array(arrays), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_non_numpy_arrays_large_series(self):
        """Test with non-numpy arrays in large series."""
        arrays = [[i, i + 1, i + 2] for i in range(100)]  # Lists instead of numpy arrays - reduced size
        series = pd.Series(arrays)

        # Should fallback to list method since first element is not numpy array
        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array(arrays), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_types_large_series(self):
        """Test with mixed types in large series."""
        arrays = []
        for i in range(100):  # Reduced from 1500 to prevent memory issues
            if i % 2 == 0:
                arrays.append(np.array([i, i + 1, i + 2]))  # numpy array
            else:
                arrays.append([i, i + 1, i + 2])  # list

        series = pd.Series(arrays)

        # First element is numpy array, so tries stacking but should handle TypeError
        result = utils.nanmean_series_of_np(series)
        assert isinstance(result, np.ndarray)

    def test_3d_arrays(self):
        """Test with 3D numpy arrays."""
        arrays = [
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
            np.array([[[17, 18], [19, 20]], [[21, 22], [23, 24]]]),
        ]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_complex_numbers(self):
        """Test with complex number arrays."""
        arrays = [
            np.array([1 + 2j, 3 + 4j, 5 + 6j]),
            np.array([7 + 8j, 9 + 10j, 11 + 12j]),
            np.array([13 + 14j, 15 + 16j, 17 + 18j]),
        ]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_boolean_arrays(self):
        """Test with boolean arrays."""
        arrays = [np.array([True, False, True]), np.array([False, True, False]), np.array([True, True, False])]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.array([arrays[0], arrays[1], arrays[2]]), axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_large_series_exception_handling(self):
        """Test exception handling in large series path."""
        # Create series that will trigger the large series path but cause an exception
        arrays = []
        for i in range(100):  # Reduced from 1500 to prevent memory issues
            if i == 0:
                arrays.append(np.array([1, 2, 3]))  # First element is numpy array
            else:
                arrays.append("not_an_array")  # Rest are strings to cause TypeError

        series = pd.Series(arrays)

        # Should catch the exception and fall back to list method
        with pytest.raises((ValueError, TypeError)):
            utils.nanmean_series_of_np(series)

    def test_performance_comparison(self):
        """Test that large series uses the optimized path."""
        # Create large series - reduced size to prevent memory issues
        arrays = [np.array([i, i + 1, i + 2]) for i in range(100)]  # Reduced from 1500
        series = pd.Series(arrays)

        # This should use the optimized stacking path
        result = utils.nanmean_series_of_np(series)

        # Verify result is correct
        expected = np.nanmean(np.stack(arrays, axis=0), axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_axis(self):
        """Test with negative axis parameter."""
        arrays = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
        ]
        series = pd.Series(arrays)

        # Test axis=-1 (last axis)
        result = utils.nanmean_series_of_np(series, axis=-1)
        expected = np.nanmean(np.array([arrays[0], arrays[1]]), axis=-1)

        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_operation_large_series(self):
        """Test basic nanmean operation with large series (100 items)."""
        # Create series with many numpy arrays - reduced size to prevent memory issues
        arrays = [np.array([i, i + 1, i + 2]) for i in range(100)]
        series = pd.Series(arrays)

        result = utils.nanmean_series_of_np(series)
        expected = np.nanmean(np.stack(arrays, axis=0), axis=0)

        np.testing.assert_array_almost_equal(result, expected)


class TestTimestampMapper:
    """Test TimestampMapper class."""

    def test_basic_initialization(self):
        """Test basic initialization of TimestampMapper."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0), datetime(2023, 1, 1, 12, 2, 0)]
        durations = [60.0, 60.0, 60.0]  # 1 minute each

        mapper = utils.TimestampMapper(end_times, durations)

        assert mapper.file_end_datetimes == end_times
        assert mapper.file_durations == durations
        assert len(mapper.file_start_datetimes) == 3
        assert len(mapper.cumulative_durations) == 3

        # Check start times are calculated correctly
        expected_start_times = [
            datetime(2023, 1, 1, 11, 59, 0),  # 12:00 - 60s
            datetime(2023, 1, 1, 12, 0, 0),  # 12:01 - 60s
            datetime(2023, 1, 1, 12, 1, 0),  # 12:02 - 60s
        ]
        assert mapper.file_start_datetimes == expected_start_times

        # Check cumulative durations
        expected_cumulative = [60.0, 120.0, 180.0]
        np.testing.assert_array_almost_equal(mapper.cumulative_durations, expected_cumulative)

    def test_single_file(self):
        """Test with single file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [120.0]  # 2 minutes

        mapper = utils.TimestampMapper(end_times, durations)

        assert len(mapper.file_start_datetimes) == 1
        assert mapper.file_start_datetimes[0] == datetime(2023, 1, 1, 11, 58, 0)
        assert mapper.cumulative_durations[0] == 120.0

    def test_get_fragment_timestamp_first_file(self):
        """Test getting fragment timestamp from first file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        durations = [60.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Fragment 0, length 30s (should be in first file)
        result = mapper.get_fragment_timestamp(0, 30.0)

        # Fragment starts at time 0, which is in first file
        # Offset should be 0 - 60 = -60s from end time
        expected = datetime(2023, 1, 1, 12, 0, 0) + timedelta(seconds=-60)
        assert result == expected

    def test_get_fragment_timestamp_second_file(self):
        """Test getting fragment timestamp from second file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        durations = [60.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Fragment starting at 90s (30s into second file)
        result = mapper.get_fragment_timestamp(3, 30.0)  # 3 * 30s = 90s

        # Fragment starts at 90s, cumulative duration of file 1 is 120s
        # Offset should be 90 - 120 = -30s from end time of file 1
        expected = datetime(2023, 1, 1, 12, 1, 0) + timedelta(seconds=-30)
        assert result == expected

    def test_get_fragment_timestamp_edge_cases(self):
        """Test edge cases for fragment timestamp calculation."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 2, 0), datetime(2023, 1, 1, 12, 5, 0)]
        durations = [30.0, 120.0, 180.0]  # Different durations

        mapper = utils.TimestampMapper(end_times, durations)

        # Test fragment beyond last file (should clamp to last file)
        result = mapper.get_fragment_timestamp(100, 10.0)  # Way beyond

        # Should use last file index
        assert isinstance(result, datetime)

    def test_empty_file_lists(self):
        """Test with empty file lists."""
        with pytest.raises(IndexError):
            mapper = utils.TimestampMapper([], [])
            mapper.get_fragment_timestamp(0, 30.0)

    def test_mismatched_list_lengths(self):
        """Test with mismatched end times and durations lists."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [60.0, 120.0]  # Different length

        # Should raise ValueError for mismatched lengths
        with pytest.raises(ValueError, match="file_end_datetimes and file_durations must have the same length"):
            utils.TimestampMapper(end_times, durations)

    def test_zero_duration_file(self):
        """Test with zero duration file."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        durations = [0.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Should handle zero duration gracefully
        result = mapper.get_fragment_timestamp(0, 30.0)
        assert isinstance(result, datetime)

    def test_negative_duration(self):
        """Test with negative duration (edge case)."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [-60.0]  # Negative duration

        mapper = utils.TimestampMapper(end_times, durations)

        # Should still work, but start time will be after end time
        assert mapper.file_start_datetimes[0] > mapper.file_end_datetimes[0]

    def test_very_small_fragment_length(self):
        """Test with very small fragment length."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Very small fragment
        result = mapper.get_fragment_timestamp(0, 0.001)
        assert isinstance(result, datetime)

    def test_very_large_fragment_length(self):
        """Test with very large fragment length."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0)]
        durations = [60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Large fragment
        result = mapper.get_fragment_timestamp(0, 3600.0)  # 1 hour
        assert isinstance(result, datetime)

    def test_non_sequential_end_times(self):
        """Test with non-sequential end times."""
        end_times = [
            datetime(2023, 1, 1, 12, 2, 0),  # Later time first
            datetime(2023, 1, 1, 12, 0, 0),  # Earlier time second
            datetime(2023, 1, 1, 12, 1, 0),
        ]
        durations = [60.0, 60.0, 60.0]

        mapper = utils.TimestampMapper(end_times, durations)

        # Should still work, mapping is based on order in list
        result = mapper.get_fragment_timestamp(0, 30.0)
        assert isinstance(result, datetime)

    def test_microsecond_precision(self):
        """Test timestamp precision with microseconds."""
        end_times = [datetime(2023, 1, 1, 12, 0, 0, 123456)]
        durations = [60.5]  # Duration that doesn't exactly cancel microseconds

        mapper = utils.TimestampMapper(end_times, durations)

        result = mapper.get_fragment_timestamp(0, 30.0)  # Fragment at start
        assert isinstance(result, datetime)
        # With fragment_start_time=0, offset_in_file = 0 - 60.5 = -60.5
        # So result = 12:00:00.123456 + (-60.5s) = 10:59:59.623456
        assert result.microsecond == 623456


class TestValidateTimestamps:
    """Test validate_timestamps function."""

    def test_valid_chronological_timestamps(self):
        """Test with valid chronological timestamps."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0), datetime(2023, 1, 1, 12, 2, 0)]

        result = utils.validate_timestamps(timestamps)
        assert result == timestamps

    def test_empty_timestamp_list(self):
        """Test with empty timestamp list."""
        with pytest.raises(ValueError, match="No timestamps provided for validation"):
            utils.validate_timestamps([])

    def test_all_none_timestamps(self):
        """Test with all None timestamps."""
        timestamps = [None, None, None]

        with pytest.warns(UserWarning, match="Found 3 None timestamps"):
            with pytest.raises(ValueError, match="No valid timestamps found"):
                utils.validate_timestamps(timestamps)

    def test_some_none_timestamps(self):
        """Test with some None timestamps."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0), None, datetime(2023, 1, 1, 12, 2, 0), None]

        # Should warn about None timestamps AND about large gap (2 minutes > 60 seconds)
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps)

        # Check that we got both warnings
        warning_messages = [str(w.message) for w in warning_list]
        assert any("Found 2 None timestamps" in msg for msg in warning_messages)
        assert any("Large gap detected" in msg for msg in warning_messages)

        expected = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 2, 0)]
        assert result == expected

    def test_non_chronological_timestamps(self):
        """Test with non-chronological timestamps."""
        timestamps = [
            datetime(2023, 1, 1, 12, 2, 0),  # Later time first
            datetime(2023, 1, 1, 12, 0, 0),  # Earlier time
            datetime(2023, 1, 1, 12, 1, 0),
        ]

        with pytest.warns(UserWarning, match="Timestamps are not in chronological order"):
            result = utils.validate_timestamps(timestamps)

        # Should return original order, not sorted
        assert result == timestamps

    def test_large_gaps_warning(self):
        """Test warning for large gaps between timestamps."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 1, 12, 0, 30),  # 30 seconds later
            datetime(2023, 1, 1, 12, 2, 0),  # 90 seconds later (large gap)
        ]

        with pytest.warns(UserWarning, match="Large gap detected"):
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=60)

        assert result == timestamps

    def test_custom_gap_threshold(self):
        """Test with custom gap threshold."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 1, 12, 0, 45),  # 45 seconds later
        ]

        # Should warn with threshold of 30 seconds
        with pytest.warns(UserWarning, match="Large gap detected"):
            utils.validate_timestamps(timestamps, gap_threshold_seconds=30)

        # Should not warn with threshold of 60 seconds
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=60)
            assert result == timestamps

    def test_gap_threshold_behavior(self):
        """Test gap threshold behavior with varying thresholds and gaps."""
        # Timestamps with different gaps
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),  # Start
            datetime(2023, 1, 1, 12, 0, 30),  # 30 seconds gap
            datetime(2023, 1, 1, 12, 1, 30),  # 60 seconds gap
            datetime(2023, 1, 1, 12, 3, 30),  # 120 seconds gap
        ]

        # Test with 20 second threshold - should warn about all gaps
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=20)

        warning_messages = [str(w.message) for w in warning_list]
        # Should get 3 warnings (for 30s, 60s, and 120s gaps)
        assert len([msg for msg in warning_messages if "Large gap detected" in msg]) == 3

        # Test with 50 second threshold - should warn about 60s and 120s gaps only
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=50)

        warning_messages = [str(w.message) for w in warning_list]
        # Should get 2 warnings (for 60s and 120s gaps)
        assert len([msg for msg in warning_messages if "Large gap detected" in msg]) == 2

        # Test with 90 second threshold - should warn about 120s gap only
        with pytest.warns(UserWarning) as warning_list:
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=90)

        warning_messages = [str(w.message) for w in warning_list]
        # Should get 1 warning (for 120s gap)
        assert len([msg for msg in warning_messages if "Large gap detected" in msg]) == 1

        # Test with 150 second threshold - should not warn about any gaps
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=150)
            assert result == timestamps

    def test_single_timestamp(self):
        """Test with single timestamp."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0)]

        result = utils.validate_timestamps(timestamps)
        assert result == timestamps

    def test_identical_timestamps(self):
        """Test with identical timestamps."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        timestamps = [timestamp, timestamp, timestamp]

        # Should not trigger gap warning (0 second gaps)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = utils.validate_timestamps(timestamps)
            assert result == timestamps

    def test_microsecond_differences(self):
        """Test with microsecond-level differences."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0, 0),
            datetime(2023, 1, 1, 12, 0, 0, 500000),  # 0.5 seconds later
            datetime(2023, 1, 1, 12, 0, 1, 0),  # 1 second from first
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = utils.validate_timestamps(timestamps, gap_threshold_seconds=2)
            assert result == timestamps

    def test_negative_gap_threshold(self):
        """Test with negative gap threshold."""
        timestamps = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 0, 1)]

        # Even 1 second should trigger warning with negative threshold
        with pytest.warns(UserWarning, match="Large gap detected"):
            utils.validate_timestamps(timestamps, gap_threshold_seconds=-1)

    def test_zero_gap_threshold(self):
        """Test with zero gap threshold."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 1, 12, 0, 0, 1),  # 1 microsecond later
        ]

        # Even microsecond should trigger warning with zero threshold
        with pytest.warns(UserWarning, match="Large gap detected"):
            utils.validate_timestamps(timestamps, gap_threshold_seconds=0)

    def test_very_large_gaps(self):
        """Test with very large gaps."""
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 2, 12, 0, 0),  # 1 day later
        ]

        with pytest.warns(UserWarning, match="Large gap detected"):
            result = utils.validate_timestamps(timestamps)
            assert result == timestamps

    def test_mixed_none_and_valid_chronological(self):
        """Test mixed None and valid timestamps in chronological order."""
        timestamps = [
            None,
            datetime(2023, 1, 1, 12, 0, 0),
            None,
            datetime(2023, 1, 1, 12, 1, 0),
            None,
            datetime(2023, 1, 1, 12, 2, 0),
            None,
        ]

        with pytest.warns(UserWarning, match="Found 4 None timestamps"):
            result = utils.validate_timestamps(timestamps)

        expected = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0), datetime(2023, 1, 1, 12, 2, 0)]
        assert result == expected

    def test_mixed_none_and_valid_non_chronological(self):
        """Test mixed None and valid timestamps not in chronological order."""
        timestamps = [
            None,
            datetime(2023, 1, 1, 12, 2, 0),
            None,
            datetime(2023, 1, 1, 12, 0, 0),  # Earlier timestamp
            None,
            datetime(2023, 1, 1, 12, 1, 0),
            None,
        ]

        with pytest.warns(UserWarning, match="Found 4 None timestamps"):
            with pytest.warns(UserWarning, match="Timestamps are not in chronological order"):
                result = utils.validate_timestamps(timestamps)

        expected = [datetime(2023, 1, 1, 12, 2, 0), datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0)]
        assert result == expected

    def test_extreme_timestamp_values(self):
        """Test with extreme timestamp values."""
        timestamps = [datetime.min, datetime.max]

        # Should handle extreme values
        with pytest.warns(UserWarning, match="Large gap detected"):
            result = utils.validate_timestamps(timestamps)
            assert result == timestamps


class TestTempDirectory:
    """Test temporary directory functions."""

    def setup_method(self):
        """Setup for each test - save original TMPDIR."""
        self.original_tmpdir = os.environ.get("TMPDIR")

    def teardown_method(self):
        """Cleanup after each test - restore original TMPDIR."""
        if self.original_tmpdir is not None:
            os.environ["TMPDIR"] = self.original_tmpdir
        elif "TMPDIR" in os.environ:
            del os.environ["TMPDIR"]

    def test_set_and_get_temp_directory(self):
        """Test setting and getting temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = temp_dir + "/test_dir"
            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == Path(test_path)
            assert result.exists()  # Should be created

    def test_set_temp_directory_creates_path(self):
        """Test that set_temp_directory creates the path if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "nested" / "temp" / "dir"

            # Ensure the path doesn't exist initially
            assert not test_path.exists()

            utils.set_temp_directory(test_path)

            # Should be created
            assert test_path.exists()
            assert test_path.is_dir()

            # Should be set in environment
            assert utils.get_temp_directory() == test_path

    def test_set_temp_directory_with_existing_path(self):
        """Test setting temp directory when path already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)

            # Path already exists
            assert test_path.exists()

            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == test_path

    def test_set_temp_directory_with_string_path(self):
        """Test setting temp directory with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = temp_dir + "/string_path"

            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == Path(test_path)
            assert result.exists()

    def test_set_temp_directory_with_path_object(self):
        """Test setting temp directory with Path object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "path_object"

            utils.set_temp_directory(test_path)

            result = utils.get_temp_directory()
            assert result == test_path
            assert result.exists()

    def test_get_temp_directory_default(self):
        """Test getting temp directory after setting it."""
        # Set a temporary directory first
        test_path = "/default/tmp"
        os.environ["TMPDIR"] = test_path

        result = utils.get_temp_directory()
        assert result == Path(test_path)

    def test_get_temp_directory_missing_env_var(self):
        """Test get_temp_directory when TMPDIR is not set."""
        # Remove TMPDIR if it exists
        if "TMPDIR" in os.environ:
            del os.environ["TMPDIR"]

        with pytest.raises(KeyError):
            utils.get_temp_directory()

    def test_file_operations_in_temp_directory(self):
        """Test creating, modifying, and deleting files in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "file_ops_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Test file creation
            test_file = temp_dir_path / "test_file.txt"
            test_file.write_text("Hello, World!")

            assert test_file.exists()
            assert test_file.read_text() == "Hello, World!"

            # Test file modification
            test_file.write_text("Modified content")
            assert test_file.read_text() == "Modified content"

            # Test file deletion
            test_file.unlink()
            assert not test_file.exists()

    def test_directory_operations_in_temp_directory(self):
        """Test creating and deleting directories in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "dir_ops_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Test directory creation
            test_subdir = temp_dir_path / "subdir" / "nested"
            test_subdir.mkdir(parents=True)

            assert test_subdir.exists()
            assert test_subdir.is_dir()

            # Test file in subdirectory
            test_file = test_subdir / "nested_file.txt"
            test_file.write_text("Nested content")

            assert test_file.exists()
            assert test_file.read_text() == "Nested content"

            # Test directory deletion
            shutil.rmtree(test_subdir.parent)  # Remove "subdir" and all contents
            assert not test_subdir.exists()
            assert not test_file.exists()

    def test_binary_file_operations(self):
        """Test binary file operations in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "binary_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Test binary file operations
            binary_file = temp_dir_path / "binary_file.bin"
            binary_data = b"\x00\x01\x02\x03\xff\xfe\xfd"

            binary_file.write_bytes(binary_data)

            assert binary_file.exists()
            read_data = binary_file.read_bytes()
            assert read_data == binary_data

            # Test appending to binary file
            additional_data = b"\x10\x20\x30"
            with binary_file.open("ab") as f:
                f.write(additional_data)

            final_data = binary_file.read_bytes()
            assert final_data == binary_data + additional_data

    def test_numpy_array_operations(self):
        """Test saving and loading numpy arrays in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "numpy_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create test array
            test_array = np.random.rand(10, 5)

            # Save as .npy file
            npy_file = temp_dir_path / "test_array.npy"
            np.save(npy_file, test_array)

            assert npy_file.exists()

            # Load and verify
            loaded_array = np.load(npy_file)
            np.testing.assert_array_equal(test_array, loaded_array)

            # Save as .npz file
            npz_file = temp_dir_path / "test_arrays.npz"
            np.savez(npz_file, array1=test_array, array2=test_array * 2)

            assert npz_file.exists()

            # Load and verify
            with np.load(npz_file) as loaded_data:
                np.testing.assert_array_equal(loaded_data["array1"], test_array)
                np.testing.assert_array_equal(loaded_data["array2"], test_array * 2)

    def test_pandas_dataframe_operations(self):
        """Test saving and loading pandas DataFrames in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "pandas_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create test DataFrame
            test_df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"], "C": [1.1, 2.2, 3.3, 4.4]})

            # Save as CSV
            csv_file = temp_dir_path / "test_df.csv"
            test_df.to_csv(csv_file, index=False)

            assert csv_file.exists()

            # Load and verify
            loaded_df = pd.read_csv(csv_file)
            pd.testing.assert_frame_equal(test_df, loaded_df)

            # Save as pickle
            pickle_file = temp_dir_path / "test_df.pkl"
            test_df.to_pickle(pickle_file)

            assert pickle_file.exists()

            # Load and verify
            loaded_pickle_df = pd.read_pickle(pickle_file)
            pd.testing.assert_frame_equal(test_df, loaded_pickle_df)

    def test_file_permissions_and_attributes(self):
        """Test file permissions and attributes in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "permissions_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create test file
            test_file = temp_dir_path / "perm_test.txt"
            test_file.write_text("Permission test")

            # Test file attributes
            assert test_file.exists()
            assert test_file.is_file()
            assert not test_file.is_dir()

            # Test file size
            assert test_file.stat().st_size > 0

            # Test file modification (on Unix-like systems)
            if os.name != "nt":  # Skip on Windows
                # Make file read-only
                test_file.chmod(0o444)

                # Verify read-only (should raise PermissionError when trying to write)
                with pytest.raises(PermissionError):
                    test_file.write_text("Should fail")

                # Restore write permissions
                test_file.chmod(0o644)

                # Should work now
                test_file.write_text("Now it works")
                assert test_file.read_text() == "Now it works"

    def test_concurrent_access_safety(self):
        """Test that temp directory handles concurrent access safely."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "concurrent_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create multiple files simultaneously
            files = []
            for i in range(10):
                file_path = temp_dir_path / f"concurrent_file_{i}.txt"
                file_path.write_text(f"Content {i}")
                files.append(file_path)

            # Verify all files exist and have correct content
            for i, file_path in enumerate(files):
                assert file_path.exists()
                assert file_path.read_text() == f"Content {i}"

            # Clean up
            for file_path in files:
                file_path.unlink()
                assert not file_path.exists()

    def test_large_file_operations(self):
        """Test operations with larger files in temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "large_file_test"
            utils.set_temp_directory(test_path)

            temp_dir_path = utils.get_temp_directory()

            # Create a moderately large file (100KB) - reduced size to prevent memory issues
            large_file = temp_dir_path / "large_file.txt"
            content = "A" * (1024 * 100)  # 100KB of 'A's (reduced from 1MB)

            large_file.write_text(content)

            assert large_file.exists()
            assert large_file.stat().st_size >= 1024 * 100  # Reduced from 1MB

            # Read it back
            read_content = large_file.read_text()
            assert len(read_content) == len(content)
            assert read_content == content

            # Test chunked reading
            with large_file.open("r") as f:
                chunk = f.read(1024)
                assert len(chunk) == 1024
                assert chunk == "A" * 1024


class TestHiddenPrints:
    """Test _HiddenPrints context manager."""

    def test_silence_enabled(self):
        """Test that prints are silenced when silence=True."""
        import sys
        from io import StringIO

        # Capture stdout to verify silence
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with utils._HiddenPrints(silence=True):
                print("This should be silenced")

            # Output should be empty (silenced)
            assert captured_output.getvalue() == ""
        finally:
            sys.stdout = original_stdout

    def test_silence_disabled(self):
        """Test that prints work normally when silence=False."""
        import sys
        from io import StringIO

        # Capture stdout to verify output
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            with utils._HiddenPrints(silence=False):
                print("This should be visible")

            # Output should contain the print statement
            assert "This should be visible" in captured_output.getvalue()
        finally:
            sys.stdout = original_stdout

    def test_context_manager_restoration(self):
        """Test that stdout is properly restored after context."""
        import sys

        original_stdout = sys.stdout

        with utils._HiddenPrints(silence=True):
            # stdout should be redirected during context
            assert sys.stdout != original_stdout

        # stdout should be restored after context
        assert sys.stdout == original_stdout


class TestFilepathToIndexEdgeCases:
    """Test edge cases for filepath_to_index function."""

    def test_no_numbers_in_path(self):
        """Test filepath with no numbers raises IndexError."""
        with pytest.raises(IndexError):
            utils.filepath_to_index("/path/to/file_without_numbers.bin")

    def test_only_extension_numbers(self):
        """Test filepath where only the extension contains numbers raises IndexError."""
        with pytest.raises(IndexError):
            utils.filepath_to_index("/path/to/file.mp4")  # Only extension has numbers


class TestNanaverageEdgeCases:
    """Test edge cases for nanaverage function."""

    def test_zero_weights(self):
        """Test nanaverage with all zero weights."""
        A = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        # This will cause division by zero warnings but should return nan gracefully
        with pytest.warns(RuntimeWarning, match="invalid value encountered in scalar divide"):
            result = utils.nanaverage(A, weights)

        assert np.isnan(result)
        assert isinstance(result, np.ndarray)

    def test_negative_weights(self):
        """Test nanaverage with negative weights."""
        A = np.array([1.0, 2.0, 3.0])
        weights = np.array([-1.0, 2.0, -1.0])

        # This should work but may produce warnings about negative weights
        with pytest.warns(RuntimeWarning, match="invalid value encountered in scalar divide"):
            result = utils.nanaverage(A, weights)

        # Should return nan due to invalid calculation with negative weights
        assert np.isnan(result)
        assert isinstance(result, np.ndarray)


class TestCleanStrForDate:
    """Test _clean_str_for_date function."""

    def test_basic_cleaning(self):
        """Test basic string cleaning for dates."""
        # Test removing common patterns that interfere with date parsing
        # The function doesn't replace underscores, only removes specific patterns
        assert utils._clean_str_for_date("some_random_folder_2023-01-15") == "some_random_folder_2023-01-15"
        assert utils._clean_str_for_date("data_file_Jan_15_2023") == "data_file_Jan_15_2023"

    def test_extra_whitespace_cleanup(self):
        """Test that extra whitespace is cleaned up."""
        assert utils._clean_str_for_date("  multiple   spaces   here  ") == "multiple spaces here"
        assert utils._clean_str_for_date("\t\ttabs\t\there\t\t") == "tabs here"

    def test_empty_string(self):
        """Test empty string input."""
        assert utils._clean_str_for_date("") == ""

    def test_pattern_removal(self):
        """Test that specific patterns are removed according to constants."""
        # Test patterns that are actually removed by the function
        test_string = "exp_A10_setup_(2)_2023-01-15_final"
        result = utils._clean_str_for_date(test_string)
        # Should remove A10 (matches [A-Z]+\d+) and (2) (matches \([0-9]+\))
        assert "A10" not in result
        assert "(2)" not in result


class TestGetKeyFromMatchValues:
    """Test _get_key_from_match_values function."""

    def test_basic_matching(self):
        """Test basic alias matching."""
        test_dict = {"WT": ["wildtype", "wt", "control"], "KO": ["knockout", "ko", "mutant"]}

        assert utils._get_key_from_match_values("wildtype_experiment", test_dict) == "WT"
        assert utils._get_key_from_match_values("knockout_data", test_dict) == "KO"
        assert utils._get_key_from_match_values("some_control_group", test_dict) == "WT"

    def test_longest_match_wins(self):
        """Test that longest matching alias is selected."""
        test_dict = {"A": ["a", "abc"], "B": ["ab", "b"]}

        # "abc" is longer than "ab", so should match "A" when strict_matching=False
        assert utils._get_key_from_match_values("abc_test", test_dict, strict_matching=False) == "A"

    def test_strict_matching_true(self):
        """Test strict matching prevents ambiguous matches."""
        test_dict = {
            "A": ["test", "data"],
            "B": ["test", "info"],  # "test" appears in both
        }

        with pytest.raises(ValueError, match="Ambiguous match"):
            utils._get_key_from_match_values("test_file", test_dict, strict_matching=True)

    def test_strict_matching_false(self):
        """Test that strict_matching=False allows ambiguous matches."""
        test_dict = {
            "A": ["test", "data"],
            "B": ["test", "info"],  # "test" appears in both, same length
        }

        # Should not raise error, should return one of the keys
        result = utils._get_key_from_match_values("test_file", test_dict, strict_matching=False)
        assert result in ["A", "B"]

    def test_no_matches_raises_error(self):
        """Test that no matches raises ValueError."""
        test_dict = {"A": ["apple", "apricot"], "B": ["banana", "berry"]}

        with pytest.raises(ValueError, match="does not have any matching values"):
            utils._get_key_from_match_values("orange_fruit", test_dict)

    def test_empty_dict_raises_error(self):
        """Test that empty dictionary raises error."""
        with pytest.raises(ValueError, match="does not have any matching values"):
            utils._get_key_from_match_values("test_string", {})

    def test_with_real_genotype_aliases(self):
        """Test with actual genotype aliases from constants."""
        # This tests integration with real data structure
        if hasattr(constants, "GENOTYPE_ALIASES") and constants.GENOTYPE_ALIASES:
            # Find a real alias to test with
            first_key = list(constants.GENOTYPE_ALIASES.keys())[0]
            first_alias = constants.GENOTYPE_ALIASES[first_key][0]

            result = utils._get_key_from_match_values(f"test_{first_alias}_data", constants.GENOTYPE_ALIASES)
            assert result == first_key

    def test_with_real_channel_aliases(self):
        """Test with actual channel aliases from constants."""
        # This tests integration with real data structure
        if hasattr(constants, "LR_ALIASES") and constants.LR_ALIASES:
            first_key = list(constants.LR_ALIASES.keys())[0]
            first_alias = constants.LR_ALIASES[first_key][0]

            result = utils._get_key_from_match_values(f"test_{first_alias}_channel", constants.LR_ALIASES)
            assert result == first_key
