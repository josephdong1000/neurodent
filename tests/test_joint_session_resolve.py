"""Regression test for the joint-session channel-split path.

Joint datasets (arx_parv / arx_rosa / ap3b2_rhd) record multiple animals together,
each owning a per-animal ``channel_subset``.  ``generate_wars.py`` splits the joint
recording per animal via the **real** ``LongRecordingOrganizer.split()`` and then builds
a WAR whose channel names must resolve exactly to canonical abbreviations.

This path is never exercised end-to-end elsewhere: ``mini_real`` (the only full-pipeline
integration test) has no joint sessions, and ``test_split_ao.py`` / ``test_core_split.py``
**mock** the recording with fake channel names (``"ch1"``…), so real channel resolution
after a split was never tested.  This test covers it with a real in-memory recording:
real split -> WAR construction (resolution) -> standardization (NaN-pad to the montage).
"""

import numpy as np
import pandas as pd
import pytest
import spikeinterface.core as si_core

from neurodent import constants, set_channel_map
from neurodent.core import LongRecordingOrganizer
from neurodent.visualization.results import WindowAnalysisResult

# Both tests reconfigure the channel map via set_channel_map (restored by the autouse
# conftest fixture); declare the intentional mutation so it isn't flagged.
pytestmark = pytest.mark.mutates_constants


# Two animals co-recorded on the same files: animal A on the C-port, animal B on the
# D-port.  Both raw spellings of a region resolve to the same canonical abbreviation.
_MONTAGE = {
    "LMot": ["C-015", "D-015"], "RMot": ["C-016", "D-016"],
    "LBar": ["C-014", "D-014"], "RBar": ["C-017", "D-017"],
    "LHip": ["C-012", "D-012"], "RHip": ["C-019", "D-019"],
    "LAud": ["C-009", "D-009"], "RAud": ["C-022", "D-022"],
    "LVis": ["C-010", "D-010"], "RVis": ["C-021", "D-021"],
}
# The joint recording carries all 20 channels (10 C-ports + 10 D-ports).
_RAW_ORDER = [
    "C-015", "C-016", "C-014", "C-017", "C-012", "C-019", "C-009", "C-022", "C-010", "C-021",
    "D-015", "D-016", "D-014", "D-017", "D-012", "D-019", "D-009", "D-022", "D-010", "D-021",
]
# Each animal owns 8 of its port's channels (LVis/RVis absent — like the 8-channel arx
# animals), so standardization must NaN-pad up to the full 10-region montage.
_SUBSET_A = ["C-015", "C-016", "C-014", "C-017", "C-012", "C-019", "C-009", "C-022"]
_SUBSET_B = ["D-015", "D-016", "D-014", "D-017", "D-012", "D-019", "D-009", "D-022"]
_EXPECTED_ABBREVS = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud"]


def _make_war(channel_names):
    """Real ``WindowAnalysisResult`` — goes through ``__init__`` /
    ``_update_instance_vars``, which derives ``channel_abbrevs`` via
    :func:`~neurodent.core.resolve_channel` (the resolution under test)."""
    n = len(channel_names)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01 10:00:00", periods=4, freq="1min"),
            "animalday": ["joint_20230101"] * 4,
            "duration": [1.0] * 4,
            "rms": [list(np.arange(n, dtype=float)) for _ in range(4)],
        }
    )
    return WindowAnalysisResult(df, animal_id="joint", channel_names=list(channel_names))


def test_joint_split_channels_resolve_and_standardize(tmp_path):
    set_channel_map(_MONTAGE)

    # A real in-memory recording carrying the joint montage's 20 channels.
    data = np.random.default_rng(0).standard_normal((1000, 20)).astype(np.float32)
    rec = si_core.NumpyRecording(
        traces_list=[data], sampling_frequency=1000, channel_ids=_RAW_ORDER
    )

    lro = LongRecordingOrganizer(tmp_path, mode=None)
    lro.LongRecording = rec
    lro.channel_names = list(_RAW_ORDER)

    # The REAL split (not mocked): slices the recording into per-animal subsets.
    splits = lro.split({"animalA": _SUBSET_A, "animalB": _SUBSET_B})
    assert splits["animalA"].channel_names == _SUBSET_A
    assert splits["animalB"].channel_names == _SUBSET_B

    # WAR construction must resolve each split child's RAW names — this is exactly where
    # an exact-resolution regression would raise for a joint dataset.
    war_a = _make_war(splits["animalA"].channel_names)
    war_b = _make_war(splits["animalB"].channel_names)
    assert war_a.channel_abbrevs == _EXPECTED_ABBREVS
    assert war_b.channel_abbrevs == _EXPECTED_ABBREVS

    # Standardization: pad each animal's 8 channels up to the full 10-region montage,
    # exactly as war_standardize does (target=None -> constants.CHANNEL_ABBREVS).
    out = war_a.reorder_and_pad_channels(None, use_abbrevs=True, inplace=False)
    row = out["rms"].iloc[0]
    assert len(row) == len(constants.CHANNEL_ABBREVS) == 10
    # Owned regions are populated; the two regions the animal lacks are NaN-padded.
    assert not np.isnan(row[constants.CHANNEL_ABBREVS.index("LAud")])
    assert np.isnan(row[constants.CHANNEL_ABBREVS.index("LVis")])
    assert np.isnan(row[constants.CHANNEL_ABBREVS.index("RVis")])


def test_split_missing_channel_raises(tmp_path):
    """A channel_subset naming a channel absent from the recording must raise loudly
    (config error surfaced, not silently dropped)."""
    set_channel_map(_MONTAGE)
    data = np.random.default_rng(1).standard_normal((500, 20)).astype(np.float32)
    rec = si_core.NumpyRecording(
        traces_list=[data], sampling_frequency=1000, channel_ids=_RAW_ORDER
    )
    lro = LongRecordingOrganizer(tmp_path, mode=None)
    lro.LongRecording = rec
    lro.channel_names = list(_RAW_ORDER)

    with pytest.raises(ValueError, match="not found in recording"):
        lro.split({"animalA": ["C-015", "NOPE-999"]})
