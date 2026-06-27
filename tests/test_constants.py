"""
Unit tests for neurodent.constants module.
"""

import pytest
from datetime import datetime
import numpy as np

from neurodent import constants


class TestConstants:
    """Test constants module functionality."""

    def test_genotype_aliases(self):
        """Test GENOTYPE_ALIASES mapping."""
        assert "WT" in constants.GENOTYPE_ALIASES
        assert "KO" in constants.GENOTYPE_ALIASES
        assert constants.GENOTYPE_ALIASES["WT"] == ["WT", "wildtype"]
        assert constants.GENOTYPE_ALIASES["KO"] == ["KO", "knockout"]

    def test_channel_aliases(self):
        """Test CHANNEL_MAP: the flat channel-identity source of truth."""
        expected_channels = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]
        for ch in expected_channels:
            assert ch in constants.CHANNEL_MAP
            assert isinstance(constants.CHANNEL_MAP[ch], list)

    def test_channel_abbrevs_derived_from_aliases(self):
        """CHANNEL_ABBREVS is derived from CHANNEL_MAP keys (single source, no drift)."""
        assert constants.CHANNEL_ABBREVS == list(constants.CHANNEL_MAP)

    def test_channel_abbrev_by_raw_is_inverse(self):
        """CHANNEL_ABBREV_BY_RAW is the exact reverse of CHANNEL_MAP (the resolution table)."""
        expected = {
            raw: abbrev
            for abbrev, raws in constants.CHANNEL_MAP.items()
            for raw in raws
        }
        assert constants.CHANNEL_ABBREV_BY_RAW == expected

    def test_df_sort_order(self):
        """Test DF_SORT_ORDER structure and that its channel entry derives from CHANNEL_ABBREVS."""
        expected_keys = ["channel", "genotype", "sex", "isday", "band"]
        for key in expected_keys:
            assert key in constants.DF_SORT_ORDER
            assert isinstance(constants.DF_SORT_ORDER[key], list)
        assert constants.DF_SORT_ORDER["channel"] == ["average", "all", *constants.CHANNEL_ABBREVS]

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
        assert isinstance(constants.LINEAR_2D_FEATURES, list)
        assert isinstance(constants.BAND_FEATURES, list)
        assert isinstance(constants.MATRIX_FEATURES, list)
        assert isinstance(constants.HIST_FEATURES, list)
        assert isinstance(constants.FEATURES, list)
        assert isinstance(constants.WAR_FEATURES, list)

        # Check that all feature lists contain expected items
        assert "rms" in constants.LINEAR_FEATURES
        assert "ampvar" in constants.LINEAR_FEATURES
        assert "psdtotal" in constants.LINEAR_FEATURES
        assert "nspike" in constants.LINEAR_FEATURES
        assert "logrms" in constants.LINEAR_FEATURES
        assert "logampvar" in constants.LINEAR_FEATURES
        assert "logpsdtotal" in constants.LINEAR_FEATURES
        assert "lognspike" in constants.LINEAR_FEATURES
        assert "psdslope" in constants.LINEAR_2D_FEATURES
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


class TestFeatureType:
    """Test FeatureType enum, FEATURE_TYPES mapping, and classify_feature()."""

    def test_feature_type_enum_members(self):
        """Test that FeatureType enum has the expected members."""
        assert hasattr(constants.FeatureType, "LINEAR")
        assert hasattr(constants.FeatureType, "LINEAR_2D")
        assert hasattr(constants.FeatureType, "BAND")
        assert hasattr(constants.FeatureType, "BANDED_MATRIX")
        assert hasattr(constants.FeatureType, "SIMPLE_MATRIX")
        assert hasattr(constants.FeatureType, "HIST")
        assert len(constants.FeatureType) == 6

    def test_classify_feature_linear(self):
        """Test that all LINEAR_FEATURES classify as LINEAR."""
        for feat in constants.LINEAR_FEATURES:
            assert constants.classify_feature(feat) is constants.FeatureType.LINEAR

    def test_classify_feature_linear_2d(self):
        """Test that all LINEAR_2D_FEATURES classify as LINEAR_2D."""
        for feat in constants.LINEAR_2D_FEATURES:
            assert constants.classify_feature(feat) is constants.FeatureType.LINEAR_2D

    def test_classify_feature_band(self):
        """Test that all BAND_FEATURES classify as BAND."""
        for feat in constants.BAND_FEATURES:
            assert constants.classify_feature(feat) is constants.FeatureType.BAND

    def test_classify_feature_banded_matrix(self):
        """Test that all BANDED_MATRIX_FEATURES classify as BANDED_MATRIX."""
        for feat in constants.BANDED_MATRIX_FEATURES:
            assert constants.classify_feature(feat) is constants.FeatureType.BANDED_MATRIX

    def test_classify_feature_simple_matrix(self):
        """Test that all SIMPLE_MATRIX_FEATURES classify as SIMPLE_MATRIX."""
        for feat in constants.SIMPLE_MATRIX_FEATURES:
            assert constants.classify_feature(feat) is constants.FeatureType.SIMPLE_MATRIX

    def test_classify_feature_hist(self):
        """Test that all HIST_FEATURES classify as HIST."""
        for feat in constants.HIST_FEATURES:
            assert constants.classify_feature(feat) is constants.FeatureType.HIST

    def test_classify_feature_unknown_raises(self):
        """Test that classify_feature raises ValueError for unknown features."""
        with pytest.raises(ValueError, match="Unknown feature"):
            constants.classify_feature("nonexistent_feature")

    def test_feature_types_dict_covers_all_features(self):
        """Test that FEATURE_TYPES maps every known feature."""
        for feat in constants.FEATURES:
            assert feat in constants.FEATURE_TYPES

    def test_feature_types_dict_has_no_extras(self):
        """Test that FEATURE_TYPES doesn't contain features not in FEATURES."""
        features_set = set(constants.FEATURES)
        for feat in constants.FEATURE_TYPES:
            assert feat in features_set

    def test_is_linear_property(self):
        """Test the is_linear convenience property."""
        assert constants.FeatureType.LINEAR.is_linear is True
        assert constants.FeatureType.LINEAR_2D.is_linear is True
        assert constants.FeatureType.BAND.is_linear is False
        assert constants.FeatureType.BANDED_MATRIX.is_linear is False
        assert constants.FeatureType.SIMPLE_MATRIX.is_linear is False
        assert constants.FeatureType.HIST.is_linear is False

    def test_is_matrix_property(self):
        """Test the is_matrix convenience property."""
        assert constants.FeatureType.BANDED_MATRIX.is_matrix is True
        assert constants.FeatureType.SIMPLE_MATRIX.is_matrix is True
        assert constants.FeatureType.LINEAR.is_matrix is False
        assert constants.FeatureType.LINEAR_2D.is_matrix is False
        assert constants.FeatureType.BAND.is_matrix is False
        assert constants.FeatureType.HIST.is_matrix is False

    def test_is_dict_stored_property(self):
        """Test the is_dict_stored convenience property."""
        assert constants.FeatureType.BAND.is_dict_stored is True
        assert constants.FeatureType.BANDED_MATRIX.is_dict_stored is True
        assert constants.FeatureType.LINEAR.is_dict_stored is False
        assert constants.FeatureType.LINEAR_2D.is_dict_stored is False
        assert constants.FeatureType.SIMPLE_MATRIX.is_dict_stored is False
        assert constants.FeatureType.HIST.is_dict_stored is False

    def test_matrix_features_consistent(self):
        """Test that MATRIX_FEATURES = BANDED_MATRIX_FEATURES + SIMPLE_MATRIX_FEATURES."""
        matrix_from_types = sorted(
            f for f, ft in constants.FEATURE_TYPES.items() if ft.is_matrix
        )
        assert matrix_from_types == sorted(constants.MATRIX_FEATURES)

    def test_dict_stored_features_consistent(self):
        """Test that dict-stored features = BAND_FEATURES + BANDED_MATRIX_FEATURES."""
        dict_stored = sorted(
            f for f, ft in constants.FEATURE_TYPES.items() if ft.is_dict_stored
        )
        assert dict_stored == sorted(constants.BAND_FEATURES + constants.BANDED_MATRIX_FEATURES)

    def test_linear_features_consistent(self):
        """Test that is_linear features = LINEAR_FEATURES + LINEAR_2D_FEATURES."""
        linear_from_types = sorted(
            f for f, ft in constants.FEATURE_TYPES.items() if ft.is_linear
        )
        assert linear_from_types == sorted(constants.LINEAR_FEATURES + constants.LINEAR_2D_FEATURES)

    def test_psdslope_is_linear_2d(self):
        """Test that psdslope is correctly classified as LINEAR_2D."""
        assert constants.classify_feature("psdslope") is constants.FeatureType.LINEAR_2D
        assert constants.FeatureType.LINEAR_2D.is_linear
        assert not constants.FeatureType.LINEAR_2D.is_matrix
        assert not constants.FeatureType.LINEAR_2D.is_dict_stored


class TestFeatureShapes:
    """Test FEATURE_SHAPES registry and FeatureType shape properties."""

    def test_feature_shapes_exists(self):
        """Test that FEATURE_SHAPES is exported from constants."""
        assert hasattr(constants, "FEATURE_SHAPES")
        assert isinstance(constants.FEATURE_SHAPES, dict)

    def test_feature_shapes_covers_all_types(self):
        """Test that every FeatureType member has an entry in FEATURE_SHAPES."""
        for ftype in constants.FeatureType:
            assert ftype in constants.FEATURE_SHAPES, f"Missing entry for {ftype}"

    def test_feature_shapes_keys(self):
        """Test that each entry has the required keys."""
        required_keys = {"extracted_shape", "cell_shape", "channel_axes", "semantic_axes", "description"}
        for ftype, info in constants.FEATURE_SHAPES.items():
            for key in required_keys:
                assert key in info, f"{ftype} missing key '{key}'"

    def test_channel_axes_values(self):
        """Test channel_axes for each FeatureType matches convention."""
        assert constants.FeatureType.LINEAR.channel_axes == (1,)
        assert constants.FeatureType.LINEAR_2D.channel_axes == (1,)
        assert constants.FeatureType.BAND.channel_axes == (1,)
        assert constants.FeatureType.HIST.channel_axes == (1,)
        assert constants.FeatureType.SIMPLE_MATRIX.channel_axes == (1, 2)
        assert constants.FeatureType.BANDED_MATRIX.channel_axes == (1, 2)

    def test_semantic_axes_values(self):
        """Test semantic_axes for each FeatureType."""
        assert constants.FeatureType.LINEAR.semantic_axes == {}
        assert constants.FeatureType.LINEAR_2D.semantic_axes == {"components": 2}
        assert constants.FeatureType.BAND.semantic_axes == {"bands": 2}
        assert constants.FeatureType.HIST.semantic_axes == {"freq_bins": 2}
        assert constants.FeatureType.SIMPLE_MATRIX.semantic_axes == {}
        assert constants.FeatureType.BANDED_MATRIX.semantic_axes == {"bands": 3}

    def test_extracted_shape_property(self):
        """Test the extracted_shape property on FeatureType."""
        assert constants.FeatureType.LINEAR.extracted_shape == "W, C"
        assert constants.FeatureType.LINEAR_2D.extracted_shape == "W, C, K"
        assert constants.FeatureType.BAND.extracted_shape == "W, C, B"
        assert constants.FeatureType.SIMPLE_MATRIX.extracted_shape == "W, C, C"
        assert constants.FeatureType.BANDED_MATRIX.extracted_shape == "W, C, C, B"
        assert constants.FeatureType.HIST.extracted_shape == "W, C, F"

    def test_channel_axes_property(self):
        """Test the channel_axes property on FeatureType."""
        for ftype in constants.FeatureType:
            assert ftype.channel_axes == constants.FEATURE_SHAPES[ftype]["channel_axes"]

    def test_semantic_axes_property(self):
        """Test the semantic_axes property on FeatureType."""
        for ftype in constants.FeatureType:
            assert ftype.semantic_axes == constants.FEATURE_SHAPES[ftype]["semantic_axes"]

    def test_channel_axes_always_start_at_1(self):
        """Test that channel axes always start at axis 1 (axis 0 is windows)."""
        for ftype in constants.FeatureType:
            axes = ftype.channel_axes
            assert axes[0] == 1, f"{ftype} channel axes should start at 1, got {axes}"


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

    def test_use_dictionary_for_colors(self):
        """Test that colors can be accessed via OKABE_ITO_COLORS dictionary."""
        # Preferred access pattern
        assert constants.OKABE_ITO_COLORS["black"] == "#000000"
        assert constants.OKABE_ITO_COLORS["orange"] == "#E69F00"
        assert constants.OKABE_ITO_COLORS["blue"] == "#0072B2"
        assert constants.OKABE_ITO_COLORS["green"] == "#009E73"
        assert constants.OKABE_ITO_COLORS["yellow"] == "#F5C710"
        assert constants.OKABE_ITO_COLORS["lightblue"] == "#56B4E9"
        assert constants.OKABE_ITO_COLORS["red"] == "#D55E00"
        assert constants.OKABE_ITO_COLORS["purple"] == "#CC79A7"

