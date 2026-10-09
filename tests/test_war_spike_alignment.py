"""Each window's spike count must come from that window.

A WAR's rows are sorted by timestamp, and spike counts are binned per animalday. The counts used
to be concatenated in sorted animalday-name order and assigned to the rows by position. Names
sort differently from dates across some month boundaries ('Feb-01-2020' before 'Jan-31-2020'), so
for an animal recorded across one, every row got another window's count, shifted by the length of
its earliest animalday. IQSEC2 76 and 78 were shifted by about 13 hours, which also put most counts
in the wrong light phase. Totals across the whole animal still agreed, which is why it went unseen.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neurodent.results import WindowAnalysisResult

WINDOW_S = 5.0
JAN31 = "A1 WT Jan-31-2020"
FEB01 = "A1 WT Feb-01-2020"

# Channel 0 has a different count in every window, so any misalignment changes a value.
# Spike times are seconds from the start of each animalday's recording.
SPIKES = {
    JAN31: [[1.0, 6.0, 6.5, 11.0, 12.0, 13.0], []],          # windows: 1, 2, 3
    FEB01: [[1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 9.5], []],  # windows: 4, 5
}
EXPECTED_CH0 = [1, 2, 3, 4, 5]


def _war():
    """Three Jan-31 windows then two Feb-01 windows, rows in timestamp order."""
    rows = []
    for day, start, n in ((JAN31, datetime(2020, 1, 31, 23, 59, 45), 3),
                          (FEB01, datetime(2020, 2, 1, 0, 30, 0), 2)):
        for i in range(n):
            rows.append({
                "animal": "A1", "genotype": "WT", "animalday": day,
                "timestamp": start + timedelta(seconds=WINDOW_S * i),
                "duration": WINDOW_S, "rms": [1.0, 1.0],
            })
    return WindowAnalysisResult(
        pd.DataFrame(rows), animal_id="A1", genotype="WT", channel_names=["LMot", "RMot"],
    )


def _channel0(war):
    return [int(np.asarray(v)[0]) for v in war.result["nspike"]]


class TestSpikeCountsLandOnTheirOwnRows:
    def test_names_sorting_differently_from_dates(self):
        """Without names, spike sets follow sorted name order: Feb-01 first."""
        war = _war()
        assert list(war.result["animalday"]) == [JAN31] * 3 + [FEB01] * 2

        war._read_from_spikes_all([SPIKES[FEB01], SPIKES[JAN31]])

        assert _channel0(war) == EXPECTED_CH0, (
            "the old code assigned these as [4, 5, 1, 2, 3]"
        )

    def test_explicit_names_in_any_order(self):
        war = _war()

        war._read_from_spikes_all([SPIKES[JAN31], SPIKES[FEB01]], animaldays=[JAN31, FEB01])

        assert _channel0(war) == EXPECTED_CH0

    def test_lognspike_follows_nspike(self):
        war = _war()
        war._read_from_spikes_all([SPIKES[JAN31], SPIKES[FEB01]], animaldays=[JAN31, FEB01])

        log0 = [float(np.asarray(v)[0]) for v in war.result["lognspike"]]
        assert log0 == sorted(log0), "a monotonic transform of 1..5 must stay increasing"

    def test_a_missing_animalday_raises(self):
        war = _war()

        with pytest.raises(ValueError, match="No spike results"):
            war._read_from_spikes_all([SPIKES[JAN31]], animaldays=[JAN31])

    def test_count_mismatch_raises(self):
        war = _war()

        with pytest.raises(ValueError, match="spike sets were given"):
            war._read_from_spikes_all([SPIKES[JAN31]], animaldays=[JAN31, FEB01])


def test_read_sars_spikes_matches_by_each_results_own_animalday():
    """The pipeline's entry point passes every spike result's animal_day, so order is irrelevant."""

    def fake_sar(day):
        def sa(times):
            sorting = SimpleNamespace(
                get_unit_ids=lambda: [0],
                get_unit_spike_train=lambda unit_id: np.round(np.asarray(times) * 1000).astype(int),
                get_sampling_frequency=lambda: 1000.0,
            )
            return SimpleNamespace(sorting=sorting)

        return SimpleNamespace(animal_day=day, result_sas=[sa(t) for t in SPIKES[day]])

    war = _war()
    # Deliberately in neither date order nor name order's opposite: whatever order, names decide.
    war.read_sars_spikes([fake_sar(JAN31), fake_sar(FEB01)], read_mode="sa", inplace=True)

    assert _channel0(war) == EXPECTED_CH0


def test_results_named_in_another_format_still_land_on_their_own_rows(caplog):
    """Spike results saved by an earlier version can carry names the WAR does not use.

    They fall back to the old pairing, in sorted animalday order, with a warning, and the counts
    still go to each animalday's own rows, so the shift is gone either way.
    """
    import logging

    def fake_sar(name, day):
        def sa(times):
            sorting = SimpleNamespace(
                get_unit_ids=lambda: [0],
                get_unit_spike_train=lambda unit_id: np.round(np.asarray(times) * 1000).astype(int),
                get_sampling_frequency=lambda: 1000.0,
            )
            return SimpleNamespace(sorting=sorting)

        return SimpleNamespace(animal_day=name, result_sas=[sa(t) for t in SPIKES[day]])

    war = _war()
    sars = [fake_sar("A1_session_feb", FEB01), fake_sar("A1_session_jan", JAN31)]  # sorted order
    with caplog.at_level(logging.WARNING):
        war.read_sars_spikes(sars, read_mode="sa", inplace=True)

    assert _channel0(war) == EXPECTED_CH0
    assert "Pairing them with the animaldays in sorted order" in caplog.text
