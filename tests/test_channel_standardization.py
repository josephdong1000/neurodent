"""Tests for the flat channel model: single-source derivation, set_channel_map,
standardization keeping off-default channels (e.g. hippocampus), loud partial-drop
warnings, and the opt-in global reject list.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from neurodent import constants
from neurodent.visualization.results import WindowAnalysisResult


def _make_linear_war(channel_abbrevs, rms_row):
    """Minimal WAR carrying one linear feature ('rms'), enough to exercise
    reorder_and_pad_channels(inplace=False) without full construction."""
    war = WindowAnalysisResult.__new__(WindowAnalysisResult)
    war.channel_names = list(channel_abbrevs)
    war.channel_abbrevs = list(channel_abbrevs)
    war._feature_columns = ["rms"]
    war.result = pd.DataFrame({"rms": [list(rms_row)]})
    return war


class TestChannelDerivation:
    """CHANNEL_MAP is the single source; everything else derives from it (no drift)."""

    def test_channel_abbrevs_is_alias_keys(self):
        assert constants.CHANNEL_ABBREVS == list(constants.CHANNEL_MAP)

    def test_df_sort_order_channel_derived(self):
        assert constants.DF_SORT_ORDER["channel"] == ["average", "all", *constants.CHANNEL_ABBREVS]

    def test_reverse_map_is_exact_inverse(self):
        """CHANNEL_ABBREV_BY_RAW is the exact inverse of CHANNEL_MAP."""
        expected = {
            raw: abbrev
            for abbrev, raws in constants.CHANNEL_MAP.items()
            for raw in raws
        }
        assert constants.CHANNEL_ABBREV_BY_RAW == expected


@pytest.mark.mutates_constants
class TestConfigureChannels:
    """The package front door updates the source and re-derives everything."""

    def test_recompute_on_configure(self):
        constants.set_channel_map(
            {"LMot": ["LMot"], "RMot": ["RMot"], "LFoo": ["LFoo", "left Foo"], "RFoo": ["RFoo"]}
        )
        assert constants.CHANNEL_ABBREVS == ["LMot", "RMot", "LFoo", "RFoo"]
        assert constants.DF_SORT_ORDER["channel"] == ["average", "all", "LMot", "RMot", "LFoo", "RFoo"]

    def test_new_region_resolves_via_explicit_alias(self):
        from neurodent.core import utils

        constants.set_channel_map(
            {"LMot": ["LMot"], "RMot": ["RMot"], "LFoo": ["LFoo", "L Foo Ctx"], "RFoo": ["RFoo"]}
        )
        # Exact match against the configured raw alias.
        assert utils.resolve_channel("L Foo Ctx") == "LFoo"
        # An unconfigured spelling is NOT inferred -> loud raise.
        with pytest.raises(ValueError, match="not in the configured channel map"):
            utils.resolve_channel("left Foo")

    def test_duplicate_raw_name_raises(self):
        # Same raw name mapped to two abbreviations is a config error -> loud raise.
        with pytest.raises(ValueError, match="mapped to both"):
            constants.set_channel_map({"LMot": ["shared"], "RMot": ["shared"]})


class TestExactResolution:
    """resolve_channel resolves by exact lookup only; everything else raises."""

    def test_already_canonical(self):
        from neurodent.core import utils

        assert utils.resolve_channel("LMot") == "LMot"
        assert utils.resolve_channel("  RVis  ") == "RVis"  # stripped

    def test_unconfigured_raises_loudly(self):
        from neurodent.core import utils

        with pytest.raises(ValueError, match="not in the configured channel map"):
            utils.resolve_channel("totally unknown channel")

    def test_no_number_inference(self):
        from neurodent.core import utils

        # "channel_9" used to infer LAud via assume_from_number; now it must raise.
        with pytest.raises(ValueError, match="not in the configured channel map"):
            utils.resolve_channel("channel_9")

    def test_abbreviate_warns_on_unmapped(self):
        from neurodent.core import utils

        with pytest.warns(UserWarning, match="could not be mapped"):
            out = utils.resolve_channels(["LMot", "no such channel"])
        assert out == ["LMot", "no such channel"]  # unmapped kept as-is


class TestReorderKeepsHippocampus:
    """Standardization must never silently drop a real channel (the arx-Hip bug)."""

    def test_hip_survives_aud_padded(self):
        data_channels = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LVis", "RVis"]
        rms = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # LHip=5, RHip=6
        war = _make_linear_war(data_channels, rms)
        target = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]
        new_df = war.reorder_and_pad_channels(target, use_abbrevs=True, inplace=False)
        out = new_df["rms"].iloc[0]
        assert len(out) == len(target)
        assert out[target.index("LHip")] == 5.0  # hippocampus preserved, not dropped
        assert out[target.index("RHip")] == 6.0
        assert np.isnan(out[target.index("LAud")])  # absent auditory -> NaN-padded
        assert np.isnan(out[target.index("RAud")])

    def test_default_target_uses_channel_abbrevs(self):
        data_channels = list(constants.CHANNEL_ABBREVS)
        war = _make_linear_war(data_channels, [float(i) for i in range(len(data_channels))])
        new_df = war.reorder_and_pad_channels(use_abbrevs=True, inplace=False)
        assert len(new_df["rms"].iloc[0]) == len(constants.CHANNEL_ABBREVS)

    def test_streaming_reorder_none_target_defaults(self):
        """Streaming reorder_and_pad_channels(None) defaults to CHANNEL_ABBREVS.

        Regression: war_standardize passes ``channel_reorder=None`` when no
        ``channel_reorder`` is configured. The streaming path must mirror the
        in-memory WindowAnalysisResult default and not crash on a None target
        (previously: ``TypeError: 'NoneType' object is not iterable``).
        """
        from neurodent.visualization.streaming import (
            LazyWindowAnalysisResult,
            ReorderAndPadChannels,
        )

        # Transform-level: the crash site.
        assert ReorderAndPadChannels(None).target_channels == list(constants.CHANNEL_ABBREVS)

        # Method-level: the public streaming entry point used by standardize.
        lazy = LazyWindowAnalysisResult.__new__(LazyWindowAnalysisResult)
        lazy._pending = []
        lazy.reorder_and_pad_channels(None)
        assert lazy._pending[0].target_channels == list(constants.CHANNEL_ABBREVS)


class TestLoudDrop:
    """A channel present in data but absent from the target warns loudly."""

    def test_off_montage_channel_warns(self):
        war = _make_linear_war(["LMot", "RMot", "LFoo"], [1.0, 2.0, 3.0])
        with pytest.warns(UserWarning, match="dropping channels"):
            war.reorder_and_pad_channels(["LMot", "RMot"], use_abbrevs=True, inplace=False)


class TestRejectOptIn:
    """The global reject lists are empty by default (opt-in only)."""

    def test_global_reject_channels_empty(self):
        cfg = yaml.safe_load(Path("config/config.yaml").read_text())
        manual = cfg["analysis"]["channel_filter_config"]["manual"]
        assert manual["reject_channels"] == []
        assert "reject_channels_by_session" not in manual  # toggle removed
        assert cfg["analysis"]["channel_filter_config"]["lof"]["reject_channels"] == []
