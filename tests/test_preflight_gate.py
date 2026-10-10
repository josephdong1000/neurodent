"""The preflight opens what a load would open, the way the load would open it.

Its file gate used to rebuild each animal's pattern and reader settings itself, open only
rhd files with a hard-coded call, and skip every other reader without printing anything,
so a DataWave, EDF or sox5 animal looked checked when it was not. It now takes the files and
the reader settings from the loader's own dry run, opens them with the loader's reader,
lists every animal, and says what a load would do with each failure.
"""

import errno
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neurodent.loading.discovery import DiscoveredFile
from neurodent.loading.long_recording_organizer import LongRecordingOrganizer
from neurodent.loading.preflight import failure_outcome, reader_kwargs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "iqsec2" / "validate_config.py"
HEADER = "Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit"
RATE = 1000
READER = {"mode": "si", "extract_func": "neurodent.readers:read_bin_csv_pair", "datetimes_are_start": True}
PAIR = DiscoveredFile(paths=("/d/rec-1_ColMajor.bin", "/d/rec-1_Meta.csv"))
SINGLE = DiscoveredFile(path="/d/rec-1.rhd")


class TestWhatALoadWouldDo:
    def test_a_corrupt_file_group_is_dropped(self):
        assert failure_outcome(PAIR, READER, ValueError("0-byte payload")) == "drop"

    def test_a_storage_error_stops_the_load(self):
        """A vanished mount is not a corrupt file; the loader re-raises it."""
        assert failure_outcome(PAIR, READER, OSError(errno.ESTALE, "stale handle")) == "abort"

    def test_a_single_file_stops_the_load_unless_told_to_skip(self):
        assert failure_outcome(SINGLE, {"mode": "si"}, ValueError("bad header")) == "abort"
        assert failure_outcome(SINGLE, {"mode": "si", "skip_unreadable_files": True}, ValueError("x")) == "drop"

    def test_an_mne_failure_stops_the_load(self):
        assert failure_outcome(SINGLE, {"mode": "mne"}, ValueError("bad edf")) == "abort"

    def test_only_reader_settings_reach_the_reader(self):
        kwargs = {**READER, "stream_id": "0", "manual_datetimes": "x", "skip_unreadable_files": True}

        assert reader_kwargs(kwargs) == {"stream_id": "0"}

    def test_readers_resolve_as_the_loader_resolves_them(self):
        import spikeinterface.extractors as se

        from neurodent.readers import read_bin_csv_pair

        assert LongRecordingOrganizer.resolve_extract_func("si", "read_intan") is se.read_intan
        assert LongRecordingOrganizer.resolve_extract_func("si", READER["extract_func"]) is read_bin_csv_pair
        with pytest.raises(ValueError, match="Could not resolve extractor function"):
            LongRecordingOrganizer.resolve_extract_func("si", "read_nothing")


def _pair(folder, stem, names, seconds=2, empty=False):
    folder.mkdir(parents=True, exist_ok=True)
    rows = [f"{i},{i},{n},{n},{RATE},µV,float32," for i, n in enumerate(names, start=1)]
    (folder / f"{stem}_Meta.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    payload = b"" if empty else np.zeros((seconds * RATE, len(names)), np.float32).tobytes()
    (folder / f"{stem}_ColMajor.bin").write_bytes(payload)


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("validate_config_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded by __name__, so nothing runs
    return module


@pytest.mark.mutates_constants
def test_the_gate_lists_every_animal_and_says_what_a_load_would_do(tmp_path, script, capsys):
    from neurodent.workflow import apply_samples_config
    from neurodent.workflow.utils import expand_animals_config

    bins = [f"{{data_root}}/{{animal}}/{{session}}/rec-{{index}}{s}" for s in ("_ColMajor.bin", "_Meta.csv")]
    # GOOD: two readable pairs and one zero-byte pair, which a load drops and carries on.
    _pair(tmp_path / "GOOD" / "S1", "rec-1", ["LMot", "RMot"])
    _pair(tmp_path / "GOOD" / "S1", "rec-2", ["LMot", "RMot"])
    _pair(tmp_path / "GOOD" / "S1", "rec-3", ["LMot", "RMot"], empty=True)
    # SPLIT: split to a channel its files do not have, which the load would raise on.
    _pair(tmp_path / "SPLIT" / "S1", "rec-1", ["LMot", "RMot"])
    # DATES: its times do not fit its files, so the dry run fails and nothing is opened.
    _pair(tmp_path / "DATES" / "S1", "rec-1", ["LMot", "RMot"])
    samples = expand_animals_config({
        "data_root": str(tmp_path),
        "animals": [
            {"id": "GOOD", "genotype": "WT", "sex": "M", "pattern": bins, "lro_kwargs": READER,
             "manual_datetime": {"S1": ["2020-01-01 00:00:00", "2020-01-01 00:00:02", "2020-01-01 00:00:04"]}},
            {"id": "SPLIT", "genotype": "WT", "sex": "M", "pattern": bins, "lro_kwargs": READER,
             "channel_subset": ["LMot", "LBar"], "manual_datetime": {"S1": "2020-01-01 00:00:00"}},
            {"id": "DATES", "genotype": "WT", "sex": "M", "pattern": bins, "lro_kwargs": READER,
             "manual_datetime": {"S1": ["2020-01-01 00:00:00", "2020-01-01 00:00:02"]}},
        ],
    })
    apply_samples_config(samples)
    config = {"analysis": {"war_generation": {"pattern": "{index}", "lro_kwargs": {}}}}

    discovery = script.gate_discovery(config, samples, logger=None)
    readable = {r["animal"]: r for r in script.gate_readable(discovery, logger=None)}
    printed = capsys.readouterr().out

    assert set(readable) == {"GOOD", "SPLIT", "DATES"}, "every animal gets a row"
    good = readable["GOOD"]
    assert (good["failed"], good["n_files"], len(good["drops"])) == (False, 3, 1)
    assert "2/3 open, 1 would be dropped" in printed
    assert readable["SPLIT"]["failed"]
    assert "split to channels the file does not have: ['LBar']" in readable["SPLIT"]["problems"][0]["error"]
    assert readable["DATES"] == {"animal": "DATES", "checked": False, "reason": "gate 1 failed"}
    assert "SKIP  DATES" in printed


def test_the_dry_run_reports_what_each_source_would_read(tmp_path):
    from neurodent.workflow.utils import expand_animals_config, load_animal_recordings
    from neurodent import constants

    bins = [f"{{data_root}}/{{session}}/rec-{{index}}{s}" for s in ("_ColMajor.bin", "_Meta.csv")]
    _pair(tmp_path / "S1", "rec-1", ["A-010", "B-010"])
    samples = expand_animals_config({
        "data_root": str(tmp_path),
        "animals": [{"id": "X1", "genotype": "WT", "sex": "M", "channel_subset": ["A-010"],
                     "sources": [{"pattern": bins, "lro_kwargs": READER,
                                  "manual_datetime": {"S1": "2020-01-01 00:00:00"}}]}],
    })
    constants.ANIMAL_METADATA["X1"] = {"genotype": "WT", "sex": "M"}
    try:
        summary = load_animal_recordings(
            samples, {"analysis": {"war_generation": {"pattern": "{index}", "lro_kwargs": {"mode": "si"}}}},
            [("", "X1", "")], "X1", validate_only=True,
        )
    finally:
        del constants.ANIMAL_METADATA["X1"]

    (source,) = summary["sources"]
    assert source["source"] == 0
    assert source["split_to"] == ["A-010"]
    assert source["lro_kwargs"]["extract_func"] == READER["extract_func"]
    assert "manual_datetimes" not in source["lro_kwargs"]
    assert [item.get_path_list()[0] for item in source["items"]] == [str(tmp_path / "S1" / "rec-1_ColMajor.bin")]
