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
