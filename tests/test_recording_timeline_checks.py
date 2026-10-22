"""Two recordings of one animal must not cover the same time.

Nothing compared recording times before this. Merging concatenated whatever it was given,
so two recordings that overlapped, or the same files reached by two sources, were joined
without a word and every feature was then computed on doubled or misordered data. Recordings
on different dates are never merged at all, so an overlap between animaldays had no check
anywhere.

Now each merge compares the incoming recording's first file with the end of everything
already merged, and the finished animal is checked as a whole. Overlaps up to
``RECORDING_TIME_TOLERANCE_S`` are clock jitter (the largest measured on real data is 3.5 s)
and pass. A larger overlap raises. A gap is real data, since recordings get stopped and
restarted, so it is only logged.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pytest
import spikeinterface.core as si

from neurodent.constants import RECORDING_TIME_TOLERANCE_S
from neurodent.loading import AnimalOrganizer, LongRecordingOrganizer

RATE = 1000
T0 = datetime(2020, 1, 31, 10, 0, 0)


def _recording(name, start, seconds, channels=("LMot", "RMot")):
    """An in-memory recording of ``seconds`` that starts at ``start``."""
    traces = np.zeros((int(seconds * RATE), len(channels)), dtype=np.float32)
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=RATE, channel_ids=list(channels))
    lro = LongRecordingOrganizer(item=None, recording=rec)
    lro.item = name
    lro.file_end_datetimes = [start + timedelta(seconds=seconds)]
    return lro


def _after(first, offset_s, name="second", seconds=1800):
    """A recording that starts ``offset_s`` after ``first`` ends (negative overlaps it)."""
    return _recording(name, first.file_end_datetimes[-1] + timedelta(seconds=offset_s), seconds)


class TestMergeComparesTimes:
    @pytest.mark.parametrize("method", ["merge", "merge_many"])
    def test_an_overlap_of_most_of_a_file_raises(self, method):
        """The case Stage 1 met: a file placed where the previous one still runs."""
        first = _recording("first", T0, 1800)
        second = _after(first, -1797)

        with pytest.raises(ValueError, match=r"second starts at .*, 1797\.0 s before .*\(first\)"):
            getattr(first, method)(second if method == "merge" else [second])

    @pytest.mark.parametrize("offset_s", [-3.5, -2.47, 0.0, RECORDING_TIME_TOLERANCE_S])
    def test_clock_jitter_and_contiguity_pass_quietly(self, offset_s, caplog):
        first = _recording("first", T0, 1800)

        with caplog.at_level(logging.WARNING):
            first.merge_many([_after(first, offset_s)])

        assert "gap inside one merged recording" not in caplog.text

    def test_the_tolerance_is_the_edge_for_an_overlap(self):
        first = _recording("first", T0, 1800)
        first.merge_many([_after(first, -RECORDING_TIME_TOLERANCE_S)])

        with pytest.raises(ValueError, match="before the recording it would be appended to"):
            first.merge_many([_after(first, -RECORDING_TIME_TOLERANCE_S - 0.1, name="third")])

    def test_a_gap_is_logged_not_raised(self, caplog):
        first = _recording("first", T0, 1800)

        with caplog.at_level(logging.WARNING):
            first.merge_many([_after(first, RECORDING_TIME_TOLERANCE_S + 0.1)])

        assert "10.1 s gap inside one merged recording" in caplog.text

    def test_each_entry_is_compared_with_everything_merged_before_it(self):
        """The third recording overlaps the second, not the base."""
        first = _recording("first", T0, 1800)
        second = _after(first, 0)
        third = _after(second, -600, name="third")

        with pytest.raises(ValueError, match=r"third starts at .*, 600\.0 s before"):
            first.merge_many([second, third])

    def test_recordings_without_file_times_are_not_checked(self):
        """The metadata merge has its own rule for those; this check stays out of the way."""
        first = _recording("first", T0, 1800)
        untimed = _recording("untimed", T0, 1800)
        del untimed.file_end_datetimes

        first._validate_merge_timeline(untimed)
        untimed._validate_merge_timeline(first)

    def test_zero_duration_entries_are_ignored(self):
        """0-sample files keep a 0 duration and must not count as covering any time."""
        first = _recording("first", T0, 1800)
        second = _after(first, 0)
        second.file_end_datetimes.insert(0, T0 + timedelta(seconds=900))
        second.file_durations.insert(0, 0)

        first.merge_many([second])


class TestTheWholeAnimalIsChecked:
    def test_the_same_files_loaded_twice_raise(self):
        """Two sources reaching the same files gave a recording of doubled data."""
        with pytest.raises(ValueError, match=r"1800\.0 s before"):
            AnimalOrganizer.from_lros(
                [_recording("a", T0, 1800), _recording("b", T0, 1800)], animal_id="A1"
            )

    def test_an_overlap_between_animaldays_raises(self):
        """Different start dates, so no merge ever compares them."""
        late_night = _recording("late_night", datetime(2020, 1, 31, 23, 0), 7200)
        next_day = _recording("next_day", datetime(2020, 2, 1, 0, 30), 1800)

        with pytest.raises(ValueError, match=r"next_day starts at .*, 1800\.0 s before a file of late_night"):
            AnimalOrganizer.from_lros([next_day, late_night], animal_id="A1")

    def test_a_timing_error_is_not_described_as_a_channel_mismatch(self):
        with pytest.raises(ValueError) as raised:
            AnimalOrganizer.from_lros(
                [_recording("a", T0, 1800), _recording("b", T0 + timedelta(seconds=60), 1800)],
                animal_id="A1",
            )

        message = str(raised.value)
        assert "Cannot merge the recordings that start on Jan-31-2020 into one animalday for A1" in message
        assert "sampling rate" not in message and "same channels" not in message

    def test_adjacent_animaldays_pass(self):
        before_midnight = _recording("before", datetime(2020, 1, 31, 22, 0), 7200)
        after_midnight = _recording("after", datetime(2020, 2, 1, 0, 0), 1800)

        ao = AnimalOrganizer.from_lros([before_midnight, after_midnight], animal_id="A1")

        assert len(ao.long_recordings) == 2


def test_split_children_do_not_share_timing_lists():
    """Merging into one child used to add the files to its siblings and its parent as well."""
    parent = _recording("parent", T0, 60)
    children = parent.split({"A": ["LMot"], "B": ["RMot"]})

    later = _recording("later", T0 + timedelta(seconds=60), 60, channels=("LMot",))
    children["A"].merge_many([later])

    assert len(children["A"].file_end_datetimes) == 2
    assert len(children["B"].file_end_datetimes) == 1
    assert len(parent.file_end_datetimes) == 1
    assert len(children["B"].file_durations) == 1


HEADER = "Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit"


def _pair(folder, stem, seconds):
    """One ColMajor bin and its sidecar, two channels of zeros."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = [f"{i},{i},{n},{n},{RATE},µV,float32," for i, n in enumerate(["LMot", "RMot"], start=1)]
    (folder / f"{stem}_Meta.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    (folder / f"{stem}_ColMajor.bin").write_bytes(np.zeros((seconds * RATE, 2), np.float32).tobytes())


@pytest.mark.integration
@pytest.mark.mutates_constants
def test_overlapping_per_file_times_fail_the_load(tmp_path):
    """A per-file start list that places a file inside its neighbour stops the load."""
    from neurodent.workflow import apply_samples_config
    from neurodent.workflow.utils import expand_animals_config, load_animal_recordings

    for index in (1, 2):
        _pair(tmp_path / "S1", f"rec-{index}", seconds=30)
    samples = expand_animals_config({
        "data_root": str(tmp_path),
        "animals": [{
            "id": "X1", "genotype": "WT", "sex": "M",
            "pattern": [f"{{data_root}}/{{session}}/rec-{{index}}{s}" for s in ("_ColMajor.bin", "_Meta.csv")],
            "lro_kwargs": {"mode": "si", "extract_func": "neurodent.readers:read_bin_csv_pair",
                           "datetimes_are_start": True},
            "manual_datetime": {"S1": ["2020-01-31 10:00:00", "2020-01-31 10:00:05"]},
        }],
    })
    apply_samples_config(samples)
    config = {"analysis": {"war_generation": {"pattern": "{index}", "lro_kwargs": {"mode": "si"}}}}

    with pytest.raises(ValueError, match=r"rec-2_ColMajor\.bin.* starts at .*, 25\.0 s before"):
        load_animal_recordings(samples, config, [("", "X1", "")], "X1")
