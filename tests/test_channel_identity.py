"""Channels are matched by what they are, not where they sit, and keep one name throughout.

Two things went wrong before. Recordings of one animal on different days that listed the
same electrodes in a different order were only logged, and every day's columns were then
labelled with the first day's order, so a channel's data sat under another channel's name.
Within one day the same difference raised instead, so whether it failed or mislabelled
depended on the dates. And a respelled recording ('L Barrel Ctx' against 'L Barrel') had
its ``channel_names`` changed but not its recording ids, which spike results are named
from and which a later merge has to match.

Now both checks use one rule: same names, matched exactly, or the same abbreviations once
each. The recording is reordered and respelled to the reference, data and ids included.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
import spikeinterface.core as si

from neurodent.loading import AnimalOrganizer, LongRecordingOrganizer
from neurodent.loading.lro_merge import conform_channels, match_channels

RATE = 1000
DAY1 = datetime(2020, 1, 30, 9)
DAY2 = datetime(2020, 1, 31, 9)

# Each channel carries its own constant, so a misplaced column is visible in the data.
VALUE = {"LMot": 1.0, "RMot": 2.0, "LBar": 3.0}


@pytest.fixture
def spellings():
    """Two spellings per electrode, as real exports have."""
    from neurodent.workflow import apply_samples_config

    apply_samples_config({"channels": {
        "LMot": ["L Motor", "L Motor Ctx"], "RMot": ["R Motor", "R Motor Ctx"],
        "LBar": ["L Barrel", "L Barrel Ctx"],
    }})


def _recording(name, start, names, seconds=10, property_names=None):
    """A recording whose channel k holds the constant for the electrode names[k] stands for."""
    from neurodent.core.utils import resolve_channels

    values = [VALUE[a] for a in resolve_channels(list(names))]
    traces = np.tile(np.asarray(values, dtype=np.float32), (seconds * RATE, 1))
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=RATE, channel_ids=list(names))
    if property_names is not None:
        rec.set_property("channel_name", list(property_names))
    lro = LongRecordingOrganizer(item=None, recording=rec)
    lro.item = name
    lro.file_end_datetimes = [start + timedelta(seconds=seconds)]
    return lro


def _columns(lro):
    """The constant each column holds, read from the data."""
    return [float(v) for v in lro.LongRecording.get_traces(start_frame=0, end_frame=1)[0]]


def _ids(lro):
    return [str(i) for i in lro.LongRecording.get_channel_ids()]


@pytest.mark.mutates_constants
@pytest.mark.parametrize("same_day", [True, False], ids=["merged into one day", "two days"])
def test_a_different_order_is_reordered_whether_or_not_the_days_merge(spellings, same_day):
    """The two paths used to disagree: same day raised, different days mislabelled."""
    first = _recording("first", DAY1, ["L Motor", "R Motor", "L Barrel"])
    second_start = DAY1 + timedelta(seconds=10) if same_day else DAY2
    second = _recording("second", second_start, ["L Barrel", "L Motor", "R Motor"])

    ao = AnimalOrganizer.from_lros([first, second], animal_id="A1")

    assert ao.channel_names == ["L Motor", "R Motor", "L Barrel"]
    for lro in ao.long_recordings:
        assert lro.channel_names == ao.channel_names
        assert _ids(lro) == ao.channel_names
        assert _columns(lro) == [1.0, 2.0, 3.0], "each column must hold its own electrode"


@pytest.mark.mutates_constants
def test_a_respelled_day_takes_the_reference_spelling_in_its_ids_too(spellings):
    """Spike results are named from the recording ids, the WAR from channel_names."""
    jan = _recording("jan", DAY1, ["L Motor", "R Motor"])
    feb = _recording("feb", DAY2, ["L Motor Ctx", "R Motor Ctx"])

    ao = AnimalOrganizer.from_lros([feb, jan], animal_id="A1")

    assert ao.channel_names == ["L Motor", "R Motor"]
    assert [_ids(lro) for lro in ao.long_recordings] == [["L Motor", "R Motor"]] * 2


@pytest.mark.mutates_constants
def test_a_later_merge_onto_a_respelled_recording_works(spellings):
    """It failed in SpikeInterface: the base's ids still had the old spelling."""
    jan = _recording("jan", DAY1, ["L Motor", "R Motor"])
    feb = _recording("feb", DAY2, ["L Motor Ctx", "R Motor Ctx"])
    AnimalOrganizer.from_lros([jan, feb], animal_id="A1")

    later = _recording("later", DAY2 + timedelta(seconds=10), ["L Motor", "R Motor"])
    feb.merge_many([later])

    assert feb.LongRecording.get_num_samples() == 20 * RATE
    assert _columns(feb) == [1.0, 2.0]


def test_a_name_property_that_differs_from_the_ids_is_brought_into_line():
    """Intan keeps native ids and puts the names in a property; split re-reads the property."""
    lro = _recording("intan", DAY1, ["LMot", "RMot"], property_names=["LMot", "RMot"])
    lro.channel_names = ["LMot", "RMot"]

    conform_channels(lro, ["RMot", "LMot"])

    assert _ids(lro) == ["RMot", "LMot"]
    assert list(lro.LongRecording.get_property("channel_name")) == ["RMot", "LMot"]
    assert _columns(lro) == [2.0, 1.0]


@pytest.mark.mutates_constants
def test_splitting_a_respelled_recording_keeps_names_and_ids_together(spellings):
    lro = _recording("feb", DAY2, ["L Motor Ctx", "R Motor Ctx"], property_names=["L Motor Ctx", "R Motor Ctx"])
    conform_channels(lro, ["L Motor", "R Motor"])

    child = lro.split({"A1": ["R Motor"]})["A1"]

    assert child.channel_names == ["R Motor"] == _ids(child)
    assert _columns(child) == [2.0]


def test_a_recording_of_several_animals_matches_by_exact_name():
    """One abbreviation per animal repeats, which only an exact-name match can tell apart."""
    order = match_channels(["A-015", "B-015"], ["B-015", "A-015"])

    assert order == [1, 0]


@pytest.mark.mutates_constants
def test_a_respelled_recording_of_several_animals_cannot_be_matched():
    """Matching by abbreviation would pick one of the repeated electrodes arbitrarily.

    The old check renamed through an abbreviation-to-name map here, which sent both of the
    second recording's motor channels to whichever name came last.
    """
    from neurodent.workflow import apply_samples_config

    apply_samples_config({"channels": {"LMot": ["A-015", "B-015", "PortA L Motor", "PortB L Motor"]}})

    with pytest.raises(ValueError, match=r"abbreviations \['LMot'\] stand for more than one channel"):
        match_channels(["A-015", "B-015"], ["PortA L Motor", "PortB L Motor"])


def test_repeated_names_raise():
    with pytest.raises(ValueError, match=r"lists channel names \['LMot'\] more than once"):
        match_channels(["LMot", "RMot"], ["LMot", "LMot"])


def test_no_channel_names_raise():
    """An empty reference used to skip the check altogether."""
    with pytest.raises(ValueError, match="has no channel names"):
        match_channels([], ["LMot"])


def test_different_channels_raise():
    with pytest.raises(ValueError, match="Channel names mismatch"):
        match_channels(["LMot", "RMot"], ["LMot", "LBar"])
