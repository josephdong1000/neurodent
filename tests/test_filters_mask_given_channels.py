"""Channel filters mask only the channels they are given, in eager and streaming use alike.

A reject_channels_by_session filter with no dict used to mask the WAR's stored
bad_channels_dict in a WindowAnalysisResult and nothing in a LazyWindowAnalysisResult, and
apply_filters and filter_all included it by default. The stored dict held LOF's picks from WAR
generation, so the eager default quietly applied LOF while the streaming pipeline did not.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from neurodent import constants
from neurodent.results import WindowAnalysisResult

DAYS = ["A1 WT Jan-01-2023", "A1 WT Jan-02-2023"]


def _war(channels, windows=12):
    """A WAR in µV over two days, which the default fragment filters mostly keep."""
    rng = np.random.default_rng(len(channels))
    rows = []
    for d, day in enumerate(DAYS):
        start = datetime(2023, 1, 1 + d, 8)
        for w in range(windows):
            rms = rng.uniform(100, 200, len(channels))
            frac = {band: list(np.full(len(channels), 0.2)) for band in constants.BAND_NAMES}
            rows.append({
                "animal": "A1", "genotype": "WT", "animalday": day,
                "timestamp": start + timedelta(seconds=5 * w), "duration": 5.0,
                "rms": list(rms), "logrms": list(np.log(rms)), "psdfrac": frac,
            })
    # LOF's picks, as WAR generation used to store them
    stored = {DAYS[0]: [channels[0]], DAYS[1]: [channels[1]]}
    return WindowAnalysisResult(
        pd.DataFrame(rows), animal_id="A1", genotype="WT", channel_names=list(channels),
        bad_channels_dict=stored,
    )


def _rms(war):
    return np.array(war.result["rms"].tolist())


def _streamed(war, tmp_path, **apply_filters_kwargs):
    war.save_parquet_and_json(tmp_path / "src", filename="war")
    lazy = WindowAnalysisResult.scan_parquet_and_json(tmp_path / "src", filename="war")
    lazy.apply_filters(**apply_filters_kwargs)
    lazy.save_parquet_and_json(tmp_path / "out", filename="war", batch_size=7)
    return WindowAnalysisResult.load_parquet_and_json(tmp_path / "out")


CHANNEL_SETS = [["LMot", "RMot", "LBar"], ["LMot", "RMot", "LBar", "RBar", "LAud"]]


@pytest.mark.parametrize("channels", CHANNEL_SETS)
def test_the_default_filters_leave_the_stored_channels_alone(channels, tmp_path):
    war = _war(channels)

    eager = war.apply_filters(min_valid_channels=1)
    streamed = _streamed(war, tmp_path, min_valid_channels=1)

    np.testing.assert_array_equal(_rms(eager), _rms(streamed))
    # The stored channels used to be masked by the eager default only
    assert not np.isnan(_rms(eager)[:12, 0]).all()
    assert not np.isnan(_rms(eager)[12:, 1]).all()


@pytest.mark.parametrize("channels", CHANNEL_SETS)
def test_filter_all_leaves_the_stored_channels_alone(channels):
    war = _war(channels)
    expected = war.apply_filters(min_valid_channels=1)

    filtered = war.filter_all(inplace=False, min_valid_channels=1)

    np.testing.assert_array_equal(_rms(filtered), _rms(expected))
    # filter_all used to mask the stored channels as well
    assert not np.isnan(_rms(filtered)[:12, 0]).all()
    assert not np.isnan(_rms(filtered)[12:, 1]).all()


@pytest.mark.parametrize("channels", CHANNEL_SETS)
def test_a_given_dict_is_masked_alike_in_both(channels, tmp_path):
    given = {DAYS[0]: [channels[-1]], DAYS[1]: []}
    config = {"reject_channels_by_session": {"bad_channels_dict": given}}
    war = _war(channels)

    eager = war.apply_filters(filter_config=config, min_valid_channels=1)
    streamed = _streamed(war, tmp_path, filter_config=config, min_valid_channels=1)

    rms = _rms(eager)
    np.testing.assert_array_equal(rms, _rms(streamed))
    assert np.isnan(rms[:12, -1]).all() and not np.isnan(rms[12:, -1]).any()
    assert not np.isnan(rms[:, :-1]).any(), "only the given channel is masked, not the stored ones"


def test_an_empty_dict_masks_nothing(tmp_path):
    war = _war(CHANNEL_SETS[0])

    filtered = war.filter_reject_channels_by_session({})

    np.testing.assert_array_equal(_rms(filtered), _rms(war))


class TestNoDictRaises:
    def test_filter_reject_channels_by_session(self):
        with pytest.raises(ValueError, match="needs a bad_channels_dict"):
            _war(CHANNEL_SETS[0]).filter_reject_channels_by_session()

    @pytest.mark.parametrize("params", [{}, None, {"save_bad_channels": "union"}])
    def test_apply_filters(self, params):
        with pytest.raises(ValueError, match="needs a bad_channels_dict"):
            _war(CHANNEL_SETS[0]).apply_filters({"reject_channels_by_session": params})

    @pytest.mark.parametrize("params", [{}, None])
    def test_streaming_apply_filters_raises_when_called(self, params, tmp_path):
        _war(CHANNEL_SETS[0]).save_parquet_and_json(tmp_path / "src", filename="war")
        lazy = WindowAnalysisResult.scan_parquet_and_json(tmp_path / "src", filename="war")

        with pytest.raises(ValueError, match="needs a bad_channels_dict"):
            lazy.apply_filters({"reject_channels_by_session": params})
