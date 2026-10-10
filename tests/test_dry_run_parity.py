"""The dry run refuses what a real load would, wherever no file has to be read for it.

``validate_only`` returned before most of the manual_datetime checks: a per-file list with
the wrong number of entries, an unparseable value, a repeated start and an all-null dict all
passed the dry run and failed on the cluster. A top-level list did the same. Each case here
must now fail the same way, with the same message, whether or not anything is loaded.

Keys for sessions that skip_sessions excluded used to fail both ways instead: a config that
kept a 'bad files' entry for the record could not load at all, which stopped 14 sox5_bin
animals. Such an entry is now ignored, unless a null session would be timed from it.
"""

from pathlib import Path

import numpy as np
import pytest

from neurodent.loading import AnimalOrganizer

HEADER = "Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit"
RATE = 1000
T = "2020-01-01 00:00:00"


def _session(root, name, n_files, seconds=2):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    rows = [f"{i},{i},{n},{n},{RATE},µV,float32," for i, n in enumerate(["LMot", "RMot"], start=1)]
    for index in range(1, n_files + 1):
        (folder / f"rec-{index}_Meta.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
        (folder / f"rec-{index}_ColMajor.bin").write_bytes(np.zeros((seconds * RATE, 2), np.float32).tobytes())


def _organizer(root, manual, validate_only, skip_sessions=()):
    return AnimalOrganizer(
        [str(Path(root) / "{session}" / f"rec-{{index}}{s}") for s in ("_ColMajor.bin", "_Meta.csv")],
        animal_id="X1",
        skip_sessions=list(skip_sessions),
        lro_kwargs={
            "mode": "si", "extract_func": "neurodent.readers:read_bin_csv_pair",
            "datetimes_are_start": True, "manual_datetimes": manual,
        },
        validate_only=validate_only,
    )


CASES = {
    "short list": ({"S1": [T]}, "has 1 entries but the session has 2"),
    "long list": ({"S1": [T, "2020-01-01 00:00:02", "2020-01-01 00:00:04"]}, "has 3 entries"),
    "all null": ({"S1": None}, "every session is null"),
    "unparseable anchor": ({"S1": "not a time"}, "not a time"),
    "unparseable list entry": ({"S1": [T, "not a time"]}, "not a time"),
    "repeated list entry": ({"S1": [T, T]}, "to more than one file"),
    "top-level list": ([T, "2020-01-01 00:00:02"], "A top-level list is not supported"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_dry_run_and_the_load_refuse_alike(tmp_path, case):
    manual, expected = CASES[case]
    _session(tmp_path, "S1", n_files=2)

    errors = []
    for validate_only in (True, False):
        with pytest.raises(ValueError) as raised:
            _organizer(tmp_path, manual, validate_only)
        errors.append((type(raised.value), str(raised.value)))

    assert errors[0] == errors[1], "the dry run must fail exactly as the load does"
    assert expected in errors[0][1]


def test_a_valid_dry_run_reads_no_file(tmp_path, monkeypatch):
    """Checking per-file lists must not cost a load of every file."""
    from neurodent.loading import long_recording_organizer

    _session(tmp_path, "S1", n_files=2)

    def refuse(*args, **kwargs):
        raise AssertionError("the dry run built a recording")

    monkeypatch.setattr(long_recording_organizer.LongRecordingOrganizer, "__init__", refuse)

    _organizer(tmp_path, {"S1": [T, "2020-01-01 00:00:02"]}, validate_only=True)


@pytest.mark.parametrize("validate_only", [True, False])
def test_a_time_kept_for_a_skipped_session_is_ignored(tmp_path, validate_only):
    _session(tmp_path, "S1", n_files=1)
    _session(tmp_path, "bad files", n_files=1)

    ao = _organizer(
        tmp_path, {"S1": T, "bad files": "2020-01-02 00:00:00"}, validate_only, skip_sessions=["*bad*"]
    )

    assert ao.animaldays == ["X1_S1"]


@pytest.mark.parametrize("validate_only", [True, False])
def test_a_null_session_timed_from_a_skipped_one_is_refused(tmp_path, validate_only):
    """Its start would come from a recording that is never read, so its length is unknown."""
    for name in ("S1", "bad files", "S2"):
        _session(tmp_path, name, n_files=1)

    with pytest.raises(ValueError, match="would be worked out from session 'bad files'"):
        _organizer(
            tmp_path, {"S1": T, "bad files": "2020-01-02 00:00:00", "S2": None},
            validate_only, skip_sessions=["*bad*"],
        )


def test_a_key_that_matches_nothing_still_raises(tmp_path):
    """Only keys for sessions skip_sessions excluded are let through, not typos."""
    _session(tmp_path, "S1", n_files=1)

    with pytest.raises(ValueError, match="keys not in discovered sessions"):
        _organizer(tmp_path, {"S1": T, "S9": T}, validate_only=True)
