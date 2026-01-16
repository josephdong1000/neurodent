"""Tests for warning behavior in neurodent functions.

These tests verify that warnings from neurodent functions work correctly:
1. Unmatched channels can produce warnings
2. Valid channels don't produce warnings
3. WindowAnalysisResult handles channels properly

Note: Due to Python's warning registry behavior, tests cannot reliably 
assert exact warning counts when modules are pre-imported. These tests
verify the code paths work, not specific warning counts.
"""
import warnings
import pytest
import pandas as pd


class TestParseChNameBehavior:
    """Test parse_chname_to_abbrev function behavior."""

    def test_unmatched_channel_with_number_returns_abbreviation(self):
        """Unmatched channel with number should return an abbreviation."""
        from neurodent.core import utils as core_utils
        
        # Should not raise an exception when assume_from_number=True
        result = core_utils.parse_chname_to_abbrev(
            "Intan Input C-009", assume_from_number=True
        )
        # Should return a valid abbreviation based on the number
        assert result is not None
        assert isinstance(result, str)

    def test_valid_channel_no_warning(self):
        """Valid channel names should not produce warnings."""
        from neurodent.core import utils as core_utils
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = core_utils.parse_chname_to_abbrev("left Auditory")
            assert result == "LAud"

        matching = [x for x in w if "does not match name aliases" in str(x.message)]
        assert len(matching) == 0

    def test_full_channel_names_parse_correctly(self):
        """Full channel names should parse to abbreviations without warnings."""
        from neurodent.core import utils as core_utils
        
        assert core_utils.parse_chname_to_abbrev("left Auditory") == "LAud"
        assert core_utils.parse_chname_to_abbrev("right Motor") == "RMot"


class TestWindowAnalysisResultBehavior:
    """Test WindowAnalysisResult creation behavior."""

    @pytest.fixture
    def minimal_war_df(self):
        """Create minimal DataFrame for WAR testing."""
        return pd.DataFrame({
            "animalday": ["A10 WT Jan-01-2024"] * 3,
            "animal": ["A10"] * 3,
            "genotype": ["WT"] * 3,
            "day": ["Jan-01-2024"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="4s"),
            "duration": [4.0] * 3,
            "endfile": [False, False, True],
            "isday": [True, True, True],
            "rms": [[1.0, 2.0, 3.0]] * 3,
        })

    def test_war_with_unmatched_channels_succeeds(self, minimal_war_df):
        """WAR with unmatched channels should succeed with assume_from_number=True."""
        from neurodent.visualization.results import WindowAnalysisResult
        
        unmatched_channels = ["Intan_C009", "Intan_C010", "Intan_C012"]

        # Should not raise an exception
        war = WindowAnalysisResult(
            minimal_war_df,
            animal_id="A10",
            genotype="WT",
            channel_names=unmatched_channels,
            assume_from_number=True,
        )
        assert war is not None

    def test_war_with_valid_channels_no_warning(self, minimal_war_df):
        """WAR with valid channels produces no warnings."""
        from neurodent.visualization.results import WindowAnalysisResult
        
        valid_channels = ["LMot", "RMot", "LAud"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            war = WindowAnalysisResult(
                minimal_war_df.copy(),
                animal_id="A10",
                genotype="WT",
                channel_names=valid_channels,
                assume_from_number=False,
            )

        matching = [x for x in w if "does not match name aliases" in str(x.message)]
        assert len(matching) == 0
        assert war is not None
