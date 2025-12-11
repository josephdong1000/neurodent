"""
Unit tests for neurodent.constants module.
"""

import pytest
from datetime import datetime
import numpy as np

from neurodent import constants


class TestConstants:
    """Test constants module functionality."""

    def test_default_id_to_lr(self):
        """Test DEFAULT_ID_TO_LR mapping."""
        assert constants.DEFAULT_ID_TO_LR[9] == "L"
        assert constants.DEFAULT_ID_TO_LR[16] == "R"
        assert len(constants.DEFAULT_ID_TO_LR) == 10

    def test_genotype_aliases(self):
        """Test GENOTYPE_ALIASES mapping."""
        assert "WT" in constants.GENOTYPE_ALIASES
        assert "KO" in constants.GENOTYPE_ALIASES
        assert constants.GENOTYPE_ALIASES["WT"] == ["WT", "wildtype"]
        assert constants.GENOTYPE_ALIASES["KO"] == ["KO", "knockout"]

    def test_chname_aliases(self):
        """Test CHNAME_ALIASES mapping."""
        expected_channels = ["Aud", "Vis", "Hip", "Bar", "Mot"]
        for ch in expected_channels:
            assert ch in constants.CHNAME_ALIASES
            assert len(constants.CHNAME_ALIASES[ch]) == 3  # Each has lowercase, uppercase, and ALL_CAPS variants

    def test_lr_aliases(self):
        """Test LR_ALIASES mapping."""
        assert "L" in constants.LR_ALIASES
        assert "R" in constants.LR_ALIASES
        assert "left" in constants.LR_ALIASES["L"]
        assert "right" in constants.LR_ALIASES["R"]

    def test_default_id_to_name(self):
        """Test DEFAULT_ID_TO_NAME mapping."""
        assert constants.DEFAULT_ID_TO_NAME[9] == "LAud"
        assert constants.DEFAULT_ID_TO_NAME[16] == "RMot"
        assert len(constants.DEFAULT_ID_TO_NAME) == 10

    def test_df_sort_order(self):
        """Test DF_SORT_ORDER structure."""
        expected_keys = ["channel", "genotype", "sex", "isday", "band"]
        for key in expected_keys:
            assert key in constants.DF_SORT_ORDER
            assert isinstance(constants.DF_SORT_ORDER[key], list)

    def test_dateparser_patterns(self):
        """Test DATEPARSER_PATTERNS_TO_REMOVE."""
        assert isinstance(constants.DATEPARSER_PATTERNS_TO_REMOVE, list)
        assert len(constants.DATEPARSER_PATTERNS_TO_REMOVE) > 0
        for pattern in constants.DATEPARSER_PATTERNS_TO_REMOVE:
            assert isinstance(pattern, str)

    def test_default_day(self):
        """Test DEFAULT_DAY constant."""
        assert isinstance(constants.DEFAULT_DAY, datetime)
        assert constants.DEFAULT_DAY.year == 2000
        assert constants.DEFAULT_DAY.month == 1
        assert constants.DEFAULT_DAY.day == 1

    def test_global_constants(self):
        """Test global constants."""
        assert constants.GLOBAL_SAMPLING_RATE == 1000
        assert constants.GLOBAL_DTYPE == np.float32

    def test_feature_constants(self):
        """Test feature-related constants."""
        assert isinstance(constants.LINEAR_FEATURES, list)
        assert isinstance(constants.BAND_FEATURES, list)
        assert isinstance(constants.MATRIX_FEATURES, list)
        assert isinstance(constants.HIST_FEATURES, list)
        assert isinstance(constants.FEATURES, list)
        assert isinstance(constants.WAR_FEATURES, list)

        # Check that all feature lists contain expected items
        assert "rms" in constants.LINEAR_FEATURES
        assert "ampvar" in constants.LINEAR_FEATURES
        assert "psdtotal" in constants.LINEAR_FEATURES
        assert "psdslope" in constants.LINEAR_FEATURES
        assert "nspike" in constants.LINEAR_FEATURES
        assert "logrms" in constants.LINEAR_FEATURES
        assert "logampvar" in constants.LINEAR_FEATURES
        assert "logpsdtotal" in constants.LINEAR_FEATURES
        assert "lognspike" in constants.LINEAR_FEATURES
        assert "psdband" in constants.BAND_FEATURES
        assert "psdfrac" in constants.BAND_FEATURES
        assert "logpsdband" in constants.BAND_FEATURES
        assert "logpsdfrac" in constants.BAND_FEATURES
        assert "cohere" in constants.MATRIX_FEATURES
        assert "zcohere" in constants.MATRIX_FEATURES
        assert "imcoh" in constants.MATRIX_FEATURES
        assert "zimcoh" in constants.MATRIX_FEATURES
        assert "pcorr" in constants.MATRIX_FEATURES
        assert "zpcorr" in constants.MATRIX_FEATURES
        assert "psd" in constants.HIST_FEATURES

    def test_feature_plot_height_ratios(self):
        """Test FEATURE_PLOT_HEIGHT_RATIOS for both linear and matrix features."""
        assert isinstance(constants.FEATURE_PLOT_HEIGHT_RATIOS, dict)

        # Test structure and data types
        for feature, ratio in constants.FEATURE_PLOT_HEIGHT_RATIOS.items():
            assert isinstance(feature, str)
            assert isinstance(ratio, (int, float))
            assert ratio > 0

        # Test that both linear and matrix features are included
        linear_features = ["rms", "ampvar", "psdtotal", "psdslope", "psdband", "psdfrac", "nspike"]
        matrix_features = ["cohere", "zcohere", "pcorr", "zpcorr"]

        for feature in linear_features + matrix_features:
            assert feature in constants.FEATURE_PLOT_HEIGHT_RATIOS, f"Missing feature: {feature}"

    def test_freq_bands(self):
        """Test FREQ_BANDS structure."""
        expected_bands = ["delta", "theta", "alpha", "beta", "gamma"]
        for band in expected_bands:
            assert band in constants.FREQ_BANDS
            freq_range = constants.FREQ_BANDS[band]
            assert isinstance(freq_range, tuple)
            assert len(freq_range) == 2
            assert freq_range[0] < freq_range[1]

    def test_band_names(self):
        """Test BAND_NAMES."""
        assert constants.BAND_NAMES == list(constants.FREQ_BANDS.keys())

    def test_freq_constants(self):
        """Test frequency-related constants."""
        assert isinstance(constants.FREQ_BAND_TOTAL, tuple)
        assert len(constants.FREQ_BAND_TOTAL) == 2
        assert constants.FREQ_BAND_TOTAL[0] < constants.FREQ_BAND_TOTAL[1]

        assert isinstance(constants.FREQ_MINS, list)
        assert isinstance(constants.FREQ_MAXS, list)
        assert len(constants.FREQ_MINS) == len(constants.FREQ_MAXS)

        assert constants.LINE_FREQ == 60

    def test_freq_bands_contiguity(self):
        """Test that frequency bands are contiguous without gaps or overlaps."""
        band_items = list(constants.FREQ_BANDS.items())

        # Test contiguity between adjacent bands
        for i in range(len(band_items) - 1):
            current_name, (current_low, current_high) = band_items[i]
            next_name, (next_low, next_high) = band_items[i + 1]

            # Bands should be perfectly contiguous (current_high == next_low)
            assert current_high == next_low, (
                f"Gap/overlap between {current_name} (ends at {current_high}) and {next_name} (starts at {next_low})"
            )

        # Test that combined range matches FREQ_BAND_TOTAL
        combined_range = (band_items[0][1][0], band_items[-1][1][1])
        assert combined_range == constants.FREQ_BAND_TOTAL, (
            f"Combined band range {combined_range} does not match FREQ_BAND_TOTAL {constants.FREQ_BAND_TOTAL}"
        )

    def test_sorting_params(self):
        """Test SORTING_PARAMS."""
        expected_keys = ["notch_freq", "common_ref", "scale", "whiten", "freq_min", "freq_max"]
        for key in expected_keys:
            assert key in constants.SORTING_PARAMS

    def test_scheme2_sorting_params(self):
        """Test SCHEME2_SORTING_PARAMS."""
        expected_keys = ["detect_channel_radius", "phase1_detect_channel_radius", "snippet_T1", "snippet_T2"]
        for key in expected_keys:
            assert key in constants.SCHEME2_SORTING_PARAMS

    def test_waveform_params(self):
        """Test WAVEFORM_PARAMS."""
        expected_keys = ["notch_freq", "common_ref", "scale", "whiten", "freq_min", "freq_max"]
        for key in expected_keys:
            assert key in constants.WAVEFORM_PARAMS


class TestOkabeItoColors:
    """Test Okabe-Ito colorblind-friendly color palette."""

    def test_okabe_ito_colors_exists(self):
        """Test that OKABE_ITO_COLORS dictionary exists."""
        assert hasattr(constants, "OKABE_ITO_COLORS")
        assert isinstance(constants.OKABE_ITO_COLORS, dict)

    def test_okabe_ito_colors_count(self):
        """Test that the palette has exactly 8 colors."""
        assert len(constants.OKABE_ITO_COLORS) == 8

    def test_okabe_ito_colors_keys(self):
        """Test that all expected color names are present."""
        expected_colors = ["black", "orange", "blue", "green", "yellow", "lightblue", "red", "purple"]
        assert set(constants.OKABE_ITO_COLORS.keys()) == set(expected_colors)

    def test_okabe_ito_colors_values(self):
        """Test that color values match the reference Okabe-Ito palette."""
        expected_values = {
            "black": "#000000",
            "orange": "#E69F00",
            "blue": "#0072B2",
            "green": "#009E73",
            "yellow": "#F5C710",
            "lightblue": "#56B4E9",
            "red": "#D55E00",
            "purple": "#CC79A7",
        }
        for color_name, expected_hex in expected_values.items():
            assert constants.OKABE_ITO_COLORS[color_name] == expected_hex

    def test_okabe_ito_colors_format(self):
        """Test that all colors are valid hex color strings."""
        for color_name, hex_value in constants.OKABE_ITO_COLORS.items():
            # Should be a string
            assert isinstance(hex_value, str)
            # Should start with #
            assert hex_value.startswith("#")
            # Should be 7 characters long (#RRGGBB)
            assert len(hex_value) == 7
            # Should be valid hex (0-9, A-F)
            assert all(c in "0123456789ABCDEFabcdef#" for c in hex_value)

    def test_individual_color_variables_exist(self):
        """Test that individual color variables are exported."""
        color_vars = ["black", "orange", "blue", "green", "yellow", "lightblue", "red", "purple"]
        for color_name in color_vars:
            assert hasattr(constants, color_name)

    def test_individual_color_variables_values(self):
        """Test that individual color variables match dictionary values."""
        assert constants.black == constants.OKABE_ITO_COLORS["black"]
        assert constants.orange == constants.OKABE_ITO_COLORS["orange"]
        assert constants.blue == constants.OKABE_ITO_COLORS["blue"]
        assert constants.green == constants.OKABE_ITO_COLORS["green"]
        assert constants.yellow == constants.OKABE_ITO_COLORS["yellow"]
        assert constants.lightblue == constants.OKABE_ITO_COLORS["lightblue"]
        assert constants.red == constants.OKABE_ITO_COLORS["red"]
        assert constants.purple == constants.OKABE_ITO_COLORS["purple"]

    def test_colors_are_strings(self):
        """Test that individual color variables are strings, not other types."""
        assert isinstance(constants.black, str)
        assert isinstance(constants.orange, str)
        assert isinstance(constants.blue, str)
        assert isinstance(constants.green, str)
        assert isinstance(constants.yellow, str)
        assert isinstance(constants.lightblue, str)
        assert isinstance(constants.red, str)
        assert isinstance(constants.purple, str)

    def test_colors_not_in_top_level_package(self):
        """Test that colors are NOT exported at package top level."""
        import neurodent

        # Colors should NOT be available at neurodent.blue
        assert not hasattr(neurodent, "blue")
        assert not hasattr(neurodent, "red")
        assert not hasattr(neurodent, "OKABE_ITO_COLORS")

    def test_colors_matplotlib_compatible(self):
        """Test that colors work with matplotlib."""
        import matplotlib.colors as mcolors

        # All colors should be valid matplotlib color specifications
        for color_name, hex_value in constants.OKABE_ITO_COLORS.items():
            assert mcolors.is_color_like(hex_value)

    def test_backward_compatibility_import(self):
        """Test that the import style is backward compatible with okabeito package."""
        # This import pattern should work (replacing 'from okabeito import ...')
        from neurodent.constants import black, blue, green, lightblue, orange, purple, red, yellow

        # Verify they're the correct values (should match the palette in constants.py)
        assert black == "#000000"
        assert orange == "#E69F00"
        assert blue == "#0072B2"
        assert green == "#009E73"
        assert yellow == "#F5C710"
        assert lightblue == "#56B4E9"
        assert red == "#D55E00"
        assert purple == "#CC79A7"
