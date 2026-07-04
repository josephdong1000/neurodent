"""Tests for channel-name resolution behavior (exact, loud, no inference).

After the move to an explicit channel map:
1. Canonical / configured channel names resolve silently.
2. Unconfigured channel names raise loudly (no fuzzy or number inference).
3. resolve_channels() warns (not silently) on an unmappable name.
"""
import warnings
import pytest
import pandas as pd


class TestResolveChannelBehavior:
    """resolve_channel resolves by exact lookup only."""

    def test_canonical_channel_resolves_silently(self):
        from neurodent.core import utils as core_utils

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert core_utils.resolve_channel("LAud") == "LAud"
            assert core_utils.resolve_channel("RMot") == "RMot"
        assert len(w) == 0

    def test_unconfigured_channel_raises(self):
        from neurodent.core import utils as core_utils

        # Free-text names that used to resolve via the fuzzy fallback now raise.
        with pytest.raises(ValueError, match="not in the configured channel map"):
            core_utils.resolve_channel("left Auditory")
        # Numeric / device names that used to resolve via assume_from_number now raise.
        with pytest.raises(ValueError, match="not in the configured channel map"):
            core_utils.resolve_channel("Intan Input C-009")

    def test_abbreviate_warns_on_unmapped(self):
        from neurodent.core import utils as core_utils

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = core_utils.resolve_channels(["LAud", "Intan_C009"])
        assert out == ["LAud", "Intan_C009"]  # unmapped kept as-is
        assert any("could not be mapped" in str(x.message) for x in w)


class TestWindowAnalysisResultBehavior:
    """WindowAnalysisResult channel handling under exact resolution."""

    @pytest.fixture
    def minimal_war_df(self):
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

    def test_war_with_unmatched_channels_raises(self, minimal_war_df):
        """WAR construction parses channel names; unconfigured names raise loudly."""
        from neurodent.results.window_analysis_result import WindowAnalysisResult

        with pytest.raises(ValueError, match="not in the configured channel map"):
            WindowAnalysisResult(
                minimal_war_df,
                animal_id="A10",
                genotype="WT",
                channel_names=["Intan_C009", "Intan_C010", "Intan_C012"],
            )

    def test_war_with_valid_channels_no_warning(self, minimal_war_df):
        """WAR with canonical channel names constructs without channel warnings."""
        from neurodent.results.window_analysis_result import WindowAnalysisResult

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            war = WindowAnalysisResult(
                minimal_war_df.copy(),
                animal_id="A10",
                genotype="WT",
                channel_names=["LMot", "RMot", "LAud"],
            )
        matching = [x for x in w if "could not be mapped" in str(x.message)]
        assert len(matching) == 0
        assert war is not None
