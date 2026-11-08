"""Tests for ``scripts/labeling/extract_dev_features.py`` and the provenance it shares with ``score_detectors.py``.

Unit tests run ``main`` on a synthetic cohort: the loader and the fit-pool cutter are replaced with fakes
built here, the WAR is a real WindowAnalysisResult of synthetic rows, the consensus is synthetic, and the
autoreject grid is the real one on small arrays. They cover the config lists matched through the dates of
session names, the channels and lists recorded per recording, the standardized channels of every dataset config
against those the selection counts and the pipeline pads to, the filter scores against the pipeline's own
filters, the refusal before anything loads of a recording map made under another src tree, listing a
recording twice or read while src has uncommitted changes, and of unmapped recordings, per-animal failures,
resume with byte-identical outputs, the refusal of a mapped recording without cells, what is printed for each
split, the autoreject scores file, the provenance, files that change during a run and outputs an earlier run
left, and the dev and test outputs as ``select_detectors``, ``report_event_cells`` and ``score_test`` read
them. One test runs the real loader on empty files laid out as MMUT(2)_cohort6_group1's sessions, with its
real config entries and synthetic durations, to show how its sessions group into recordings and that each
recording gets its list. A slow test runs ``main`` end to end on mini_real with fake raters. No real label
file is read.
"""

import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zlib
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import zscore

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "labeling"))
import build_cohort_bundle as C  # noqa: E402
import extract_dev_features as EDF  # noqa: E402
import score_detectors as SDET  # noqa: E402
from neurodent.results import WindowAnalysisResult  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402

GRID = adr.grid_config_names()
# The fragment filters at their config/config.yaml settings: name: (WAR accessor, params, reject from the cells).
FILTERS = {"logrms_range": ("get_filter_logrms_range", {"z_range": 3}, lambda d: d["logrmsz_abs"] > 3),
           "high_rms": ("get_filter_high_rms", {"max_rms": 500}, lambda d: d["rms"] > 500),
           "low_rms": ("get_filter_low_rms", {"min_rms": 50}, lambda d: d["rms"] < 50),
           "high_beta": ("get_filter_high_beta", {"max_beta_prop": 0.4}, lambda d: d["beta_wmax"] > 0.4)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    """Return the output of a git command run in the repository."""
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def _read_back(war, folder):
    """Return ``war`` as the pipeline's fragment filter reads it: saved to parquet and loaded again."""
    war.save_parquet_and_json(folder, filename="war")
    return WindowAnalysisResult.load_parquet_and_json(folder, filename="war")


def _war_rows(cells, war):
    """Return the WAR row and column of each cell: the window within its animalday's rows, and its channel."""
    animaldays = war.result["animalday"].to_numpy()
    first = {ad: int(np.flatnonzero(animaldays == ad)[0]) for ad in pd.unique(animaldays)}
    rows = (cells["animalday"].map(first) + cells["window"]).to_numpy()
    cols = cells["channel"].map({c: j for j, c in enumerate(war.channel_abbrevs)}).to_numpy()
    return rows, cols


def _check_filters(cells, piped):
    """Assert that one animal's extracted scores give the pipeline's decisions on the WAR it reads back.

    logrmsz_abs must also equal, exactly, the absolute z score the pipeline's logrms_range takes, which runs in
    float64 on the values read back from parquet.

    Returns:
        ``{filter: number of the cells the pipeline rejects}``.
    """
    rows, cols = _war_rows(cells, piped)
    rejected = {}
    for name, (method, params, ours) in FILTERS.items():
        reject = ~getattr(piped, method)(**params)[rows, cols]
        assert (ours(cells).to_numpy() == reject).all(), name
        rejected[name] = int(reject.sum())
    z = zscore(np.log(extract_linear_array(piped.result["rms"])), axis=0, nan_policy="omit")
    assert (cells["logrmsz_abs"].to_numpy() == np.abs(z[rows, cols])).all(), "logrmsz_abs is the pipeline's z"
    return rejected


# --------------------------------------------------------------------------- config lists


@pytest.mark.unit
def test_date_named():
    assert EDF.date_named("M3 MHet Jan-08-2022") == date(2022, 1, 8)
    assert EDF.date_named("cohort 2 group 6 mouse M3 cage3A Het Mar-19-2021") == date(2021, 3, 19)
    assert EDF.date_named("A10_062921") is None
    assert EDF.date_named("") is None


@pytest.mark.unit
def test_session_date():
    assert EDF.session_date("031922_cohort6_group1_MMUT(2)_files64-80") == date(2022, 3, 19)
    assert EDF.session_date("032021 (2)") == date(2021, 3, 20)
    assert EDF.session_date("060921") == date(2021, 6, 9)
    assert EDF.session_date("2022-06-21 F9_MUT_cohort10_cage3_files0-5") == date(2022, 6, 21)
    assert EDF.session_date("2022-06-14") == date(2022, 6, 14)
    for name in ("unknown", "bad files", "0319221_x", "20220621", "133022_x", "023022", "x_031922", "", None):
        assert EDF.session_date(name) is None, name


@pytest.mark.unit
def test_list_form_applies_to_every_recording():
    sessions = {"a__0": ["010822_a"], "a__1": ["010922_a", "011022_a"]}
    lists, matches, unmatched, uncovered = EDF.match_config_lists({"_all": ["LMot", "RMot", "LMot"]}, sessions)
    assert lists == {"a__0": ["LMot", "RMot"], "a__1": ["LMot", "RMot"]}
    assert matches == [{"key": "_all", "recording": "a__0", "sessions": ["010822_a"]},
                       {"key": "_all", "recording": "a__1", "sessions": ["010922_a", "011022_a"]}]
    assert unmatched == [] and uncovered == []
    assert EDF.match_config_lists({"_all": []}, sessions)[0] == {"a__0": [], "a__1": []}, "an empty list is a list"
    undated = {"a__0": ["unknown"], "a__1": []}
    assert EDF.match_config_lists({"_all": ["LMot"]}, undated)[0] == {"a__0": ["LMot"], "a__1": ["LMot"]}, \
        "a list-form entry needs no session dates"
    assert EDF.match_config_lists(None, sessions) == ({}, [], [], [])
    assert EDF.match_config_lists({}, sessions) == ({}, [], [], [])


@pytest.mark.unit
def test_dict_form_follows_the_dates_session_names_carry():
    """A key applies to each recording built from sessions named with its date, whatever date the recording
    starts on; a recording whose sessions do not settle on one list gets none, and each key and recording left
    out comes back with the reason."""
    sessions = {
        "a__0": ["031722_a_files0-17"],                                  # starts on Mar-17
        "a__1": ["031922_a_files64-80"],                                 # starts on Mar-18, named Mar-19
        "a__2": ["031922_a_files81-94", "032022_a_files113-129"],        # two dates, the same channels
        "a__3": ["032022_a_files130-142", "032122_a_files161-177"],      # two dates, different lists
        "a__4": ["032322_a_files200-210"],                               # a date no key names
        "a__5": ["unknown"],                                             # a session name with no date
        "a__6": [],                                                      # no session recorded
    }
    entry = {
        "OLD MMut Mar-17-2022": ["LAud", "LHip"],   # the id and genotype spellings are not read
        "OLD MMut Mar-19-2022": ["RHip", "LAud"],
        "OLD MMut Mar-20-2022": ["LAud", "RHip", "LAud"],
        "OLD MMut Mar-21-2022": ["LMot"],
        "OLD MMut Mar-25-2022": ["LMot"],           # no labeled recording carries the date
        "OLD MMut Mar-19-2022 copy": ["RAud"],      # no date as its last word
    }
    lists, matches, unmatched, uncovered = EDF.match_config_lists(entry, sessions)
    assert lists == {"a__0": ["LAud", "LHip"], "a__1": ["RHip", "LAud"], "a__2": ["RHip", "LAud"]}
    assert matches == [
        {"key": "OLD MMut Mar-17-2022", "recording": "a__0", "sessions": ["031722_a_files0-17"]},
        {"key": "OLD MMut Mar-19-2022", "recording": "a__1", "sessions": ["031922_a_files64-80"]},
        {"key": "OLD MMut Mar-19-2022", "recording": "a__2", "sessions": ["031922_a_files81-94"]},
        {"key": "OLD MMut Mar-20-2022", "recording": "a__2", "sessions": ["032022_a_files113-129"]}]
    assert {u["recording"]: u["reason"] for u in uncovered} == {
        "a__3": "its sessions carry dates with different lists: Mar-20-2022 ['LAud', 'RHip', 'LAud'], "
                "Mar-21-2022 ['LMot']",
        "a__4": "no key names Mar-23-2022, the date of sessions ['032322_a_files200-210']",
        "a__5": "sessions ['unknown'] start with no date",
        "a__6": "the loader recorded no session for this recording"}
    assert uncovered[0]["sessions"] == [{"session": "032022_a_files130-142", "date": "Mar-20-2022"},
                                        {"session": "032122_a_files161-177", "date": "Mar-21-2022"}]
    assert uncovered[2]["sessions"] == [{"session": "unknown", "date": None}]
    assert unmatched == [
        {"key": "OLD MMut Mar-19-2022 copy", "reason": "names no date in the form %b-%d-%Y"},
        {"key": "OLD MMut Mar-21-2022",
         "reason": "the labeled recordings built from sessions named with Mar-21-2022 get no list: ['a__3']"},
        {"key": "OLD MMut Mar-25-2022", "reason": "no labeled recording is built from a session named with "
                                                  "Mar-25-2022"}]


@pytest.mark.unit
def test_dict_form_keys_on_one_date():
    sessions = {"b__0": ["010822_b"], "b__1": ["010922_b"]}
    same = {"B Het Jan-08-2022": ["LMot", "RMot"], "Other Jan-08-2022": ["RMot", "LMot"], "B Het Jan-09-2022": None}
    lists, matches, unmatched, uncovered = EDF.match_config_lists(same, sessions)
    assert lists == {"b__0": ["LMot", "RMot"], "b__1": []}, "an empty YAML entry is a list with no channels"
    assert [(m["key"], m["recording"]) for m in matches] == [
        ("B Het Jan-08-2022", "b__0"), ("Other Jan-08-2022", "b__0"), ("B Het Jan-09-2022", "b__1")]
    assert unmatched == [] and uncovered == []

    differ = {"B Het Jan-08-2022": ["LMot"], "Other Jan-08-2022": ["RMot"]}
    lists, matches, unmatched, uncovered = EDF.match_config_lists(differ, sessions)
    assert lists == {} and matches == []
    assert [(u["recording"], u["reason"]) for u in uncovered] == [
        ("b__0", "keys ['B Het Jan-08-2022', 'Other Jan-08-2022'] name Jan-08-2022 with different lists"),
        ("b__1", "no key names Jan-09-2022, the date of sessions ['010922_b']")]
    assert [u["key"] for u in unmatched] == ["B Het Jan-08-2022", "Other Jan-08-2022"]
    assert {u["reason"] for u in unmatched} == {
        "the labeled recordings built from sessions named with Jan-08-2022 get no list: ['b__0']"}


@pytest.mark.unit
def test_list_beside_dated_keys_joins_the_lists_recordings_get():
    """The pipeline merges _all into every dated key's list, so it reaches only the recordings a key reaches."""
    sessions = {"c__0": ["010822_c"], "c__1": ["unknown"]}
    lists, matches, unmatched, uncovered = EDF.match_config_lists(
        {"_all": ["LMot", "RAud"], "C WT Jan-08-2022": ["RMot", "LMot"]}, sessions)
    assert lists == {"c__0": ["RMot", "LMot", "RAud"]}
    assert [(m["key"], m["recording"]) for m in matches] == [("C WT Jan-08-2022", "c__0"), ("_all", "c__0")]
    assert [u["recording"] for u in uncovered] == ["c__1"] and unmatched == []
    _, _, unmatched, _ = EDF.match_config_lists({"_all": ["LMot"], "C WT Jan-09-2022": ["RMot"]}, sessions)
    assert [u["key"] for u in unmatched] == ["C WT Jan-09-2022", "_all"]


@pytest.mark.unit
def test_recording_sessions_follow_the_start_date_of_each_session_recording():
    built = [("031722_x", "Mar-17-2022"), ("031922_x_a", "Mar-18-2022"), ("031922_x_b", "Mar-19-2022"),
             ("032022_x", "Mar-19-2022"), ("031922_x_b", "Mar-19-2022"), ("lost", None)]
    animaldays = ["X Mut Mar-17-2022", "X Mut Mar-18-2022", "X Mut Mar-19-2022", "X Mut Mar-20-2022"]
    assert EDF.recording_sessions(animaldays, built) == {
        "X Mut Mar-17-2022": ["031722_x"], "X Mut Mar-18-2022": ["031922_x_a"],
        "X Mut Mar-19-2022": ["031922_x_b", "032022_x"], "X Mut Mar-20-2022": []}


@pytest.mark.unit
def test_built_sessions_leaves_the_loader_as_it_was():
    from neurodent.loading import AnimalOrganizer

    build, own = AnimalOrganizer._create_long_recordings, dict(AnimalOrganizer.__dict__)
    with pytest.raises(RuntimeError, match="load failed"):
        with EDF.built_sessions():
            assert AnimalOrganizer._create_long_recordings is not build
            raise RuntimeError("load failed")
    assert AnimalOrganizer._create_long_recordings is build
    assert dict(AnimalOrganizer.__dict__) == own, "the class is left as it was, an inherited method not copied in"


@pytest.mark.unit
def test_standardized_channels():
    samples = {"channels": {"LMot": ["a"], "RMot": ["b"], "LHip": ["c"]}}
    assert EDF.standardized_channels({}, samples) == ["LMot", "RMot", "LHip"]
    reorder = {"analysis": {"standardization": {"channel_reorder": ["LHip", "LMot"]}}}
    assert EDF.standardized_channels(reorder, samples) == ["LHip", "LMot"]
    with pytest.raises(ValueError, match="names no channels"):
        EDF.standardized_channels({"analysis": {"standardization": {"use_abbrevs": True}}}, {})


@pytest.mark.unit
@pytest.mark.mutates_constants
@pytest.mark.parametrize("dataset", [*C.REAL_STRAINS, "mini_real"])
def test_every_dataset_config_standardizes_to_the_channels_the_selection_counts(dataset, monkeypatch):
    """For every dataset config, the channels the extraction keeps per recording (``standardized_channels`` on the
    config and samples config the cohort seam prepares) are the channels ``select_detectors.canonical_channels``
    counts and those ``standardize_wars.py`` pads each WAR to: channel_reorder, or the ``CHANNEL_ABBREVS`` that
    ``apply_samples_config`` installs. The selection refuses a ``recording_channels`` entry outside them, so a
    disagreement would otherwise show only after the extraction."""
    import select_detectors as SD
    from neurodent import constants

    monkeypatch.chdir(REPO)
    monkeypatch.setattr(C, "set_temp_directory", lambda path: None)  # the configs name cluster directories
    config, samples_config = C._prepare(dataset)  # installs the dataset's channel map
    kept = EDF.standardized_channels(config, samples_config)
    assert kept == SD.canonical_channels(SD.CONFIG_PATH, dataset)
    reorder = ((config.get("analysis") or {}).get("standardization") or {}).get("channel_reorder")
    assert kept == (list(reorder) if reorder is not None else list(constants.CHANNEL_ABBREVS))


MMUT2 = "MMUT(2)_cohort6_group1"


def _sox5_cohort(tmp_path, file_s):
    """Return the sox5_bin config and samples config, MMUT(2)_cohort6_group1's real entries in them, with
    ``data_root`` pointing at empty bin and csv files laid out as on the cluster and a reader that gives each file
    ``file_s`` seconds of noise, or less where that would start a session before the previous one ended."""
    import spikeinterface.core as si
    from neurodent.workflow.utils import expand_animals_config, load_dataset_config, resolve_samples_config

    config = load_dataset_config("sox5_bin", config_path=REPO / "config/config.yaml",
                                 local_path=tmp_path / "no_local.yaml", datasets_dir=REPO / "config/datasets")
    samples = expand_animals_config(resolve_samples_config(config))
    seconds, previous_end = {}, None
    for session, end in samples["manual_datetimes"][MMUT2].items():
        first, last = (int(i) for i in re.search(r"files(\d+)-(\d+)", session).groups())
        end = datetime.fromisoformat(end)
        total = (last - first + 1) * file_s
        if previous_end is not None and (end - previous_end).total_seconds() < total:
            total = (end - previous_end).total_seconds()
        seconds[session] = total / (last - first + 1)
        previous_end = end
        folder = tmp_path / "data" / "Cohort 6" / f"Group 1 {MMUT2} cage" / session
        folder.mkdir(parents=True)
        for i in range(first, last + 1):
            (folder / f"rec-{i}_ColMajor.bin").write_bytes(b"")
            (folder / f"rec-{i}_Meta.csv").write_text("")

    def read(item, **kwargs):
        rec = si.NoiseGeneratorRecording(num_channels=3, sampling_frequency=1000.0, noise_levels=50.0,
                                         durations=[seconds[Path(item.get_path_list()[0]).parent.name]],
                                         dtype="float32", seed=0)
        rec = rec.rename_channels(new_channel_ids=["LMot", "RMot", "LHip"])
        rec.set_channel_gains(1.0)
        rec.set_channel_offsets(0.0)
        return rec

    samples["data_root"] = str(tmp_path / "data")
    config["analysis"]["war_generation"]["lro_kwargs"] = {"mode": "si", "extract_func": read,
                                                          "multiprocess_mode": "serial"}
    return config, samples


@pytest.mark.unit
@pytest.mark.mutates_constants
@pytest.mark.parametrize("file_s, grouping", [
    (1800, {"Mar-17-2022": ["031722_files0-17"], "Mar-18-2022": ["031922_files64-80"],
            "Mar-19-2022": ["031922_files81-94", "031922_files95-112", "032022_files113-129"],
            "Mar-20-2022": ["032022_files130-142", "032022_files143-160", "032122_files161-177"],
            "Mar-21-2022": ["032122_files178-192"]}),
    (1200, {"Mar-17-2022": ["031722_files0-17"],
            "Mar-19-2022": ["031922_files64-80", "031922_files81-94", "031922_files95-112"],
            "Mar-20-2022": ["032022_files113-129", "032022_files130-142", "032022_files143-160"],
            "Mar-21-2022": ["032122_files161-177", "032122_files178-192"]})])
def test_every_recording_of_mmut2_gets_its_list(tmp_path, monkeypatch, file_s, grouping):
    """The case the amendment was written for, through the loader itself on synthetic files.

    MMUT(2)_cohort6_group1's keys name Mar-17, 19, 20 and 21, and its sessions are named 031722, 031922, 032022
    and 032122. The loader builds one recording per session folder, dates it by its first file's start (the
    folder names end times, so starts depend on the file durations) and merges those that start on one date.
    With 30-minute files the first 031922 session starts on Mar-18, which no key names, and two recordings hold
    sessions of two dates; with 20-minute files every recording holds one. Either way each recording gets the
    list, since every session date has a key and all keys carry the same channels.
    """
    pytest.importorskip("spikeinterface")
    from neurodent.workflow.utils import apply_samples_config

    monkeypatch.chdir(REPO)
    config, samples = _sox5_cohort(tmp_path, file_s)
    apply_samples_config(samples)
    ao, sessions = EDF.load_animal(samples, config, MMUT2)

    def short(name):
        return name.replace("_cohort6_group1_MMUT(2)", "")

    assert {ad.rsplit(" ", 1)[-1]: [short(s) for s in ss] for ad, ss in sessions.items()} == grouping
    assert list(sessions) == list(ao.animaldays)
    assert sorted(s for ss in sessions.values() for s in ss) == sorted(samples["manual_datetimes"][MMUT2])

    entry = samples["bad_channels"][MMUT2]
    assert {EDF.date_named(k) for k in entry} == {date(2022, 3, d) for d in (17, 19, 20, 21)}
    lists, matches, unmatched, uncovered = EDF.match_config_lists(entry, sessions)
    assert lists == {ad: ["LAud", "LHip", "RHip"] for ad in ao.animaldays}
    assert unmatched == [] and uncovered == []
    assert {m["key"] for m in matches} == set(entry)
    if file_s == 1800:
        mar18 = f"{MMUT2} Mut Mar-18-2022"
        assert EDF.date_named(mar18) not in {EDF.date_named(k) for k in entry}, "the recording's own date has no key"
        assert [m["key"] for m in matches if m["recording"] == mar18] == ["MMUT(2) MMut Mar-19-2022"]
        assert [m["key"] for m in matches if m["recording"] == f"{MMUT2} Mut Mar-19-2022"] == [
            "MMUT(2) MMut Mar-19-2022", "MMUT(2) MMut Mar-20-2022"]


@pytest.mark.unit
def test_resolve_list_reports_names_the_recording_lacks():
    def resolve(name):
        known = {"LMot": "LMot", "EEG E9-REF2": "LMot", "RAud": "RAud"}
        if name not in known:
            raise ValueError(name)
        return known[name]

    found, unknown = EDF.resolve_list(["EEG E9-REF2", "RAud", "Bogus"], ["LMot", "RMot"], resolve)
    assert found == {"LMot"}
    assert unknown == [{"channel": "RAud", "abbrev": "RAud", "reason": "RAud is not among the recording's channels"},
                       {"channel": "Bogus", "reason": "not in the dataset's channel map"}]
    _, unknown = EDF.resolve_list(["EEG E9-REF2"], ["RMot"], resolve)
    assert unknown == [{"channel": "EEG E9-REF2", "abbrev": "LMot", "reason": "LMot is not among the recording's "
                                                                              "channels"}], "a raw name's abbreviation"


# --------------------------------------------------------------------------- recording map


@pytest.mark.unit
def test_recording_map_status_reasons_and_position_fallback(tmp_path):
    path = tmp_path / "recording_map.csv"
    pd.DataFrame([
        {"july_recording": "A__0", "animal": "A", "animalday": "A WT Jan-02-2025", "port_position": "1",
         "status": "mapped", "reason": "", "extra": "ignored"},
        {"july_recording": "A__1", "animal": "A", "animalday": "", "port_position": "",
         "status": "unmapped", "reason": "cells differ"},
        {"july_recording": "A__2", "animal": "A", "animalday": "", "port_position": "3",
         "status": "shifted", "reason": "shifted by window offset +1 at port position 3"},
        {"july_recording": "032__0", "animal": "032", "animalday": "", "port_position": "0",
         "status": "mapped", "reason": ""},
    ]).to_csv(path, index=False)
    rmap = SDET.RecordingMap(path)
    assert rmap.label("A", 1, "A WT Jan-02-2025") == "A__0"
    assert rmap.label("A", 0, "A WT Jan-01-2025") is None
    assert rmap.label("A", 1, "A WT Jan-05-2025") is None, "a row with an animalday is not matched by position"
    assert rmap.label("A", 3, "A WT Jan-04-2025") is None, "only mapped rows map"
    assert rmap.label("032", 0, "032 WT Jan-01-2025") == "032__0", "ids keep their leading zeros"
    assert rmap.unmapped(["A__0", "A__1", "A__2", "A__3", "032__0"]) == [
        {"recording": "A__1", "animal": "A", "status": "unmapped", "reason": "cells differ"},
        {"recording": "A__2", "animal": "A", "status": "shifted",
         "reason": "shifted by window offset +1 at port position 3"},
        {"recording": "A__3", "animal": "A", "status": "absent", "reason": "not in the recording map"},
    ]
    assert SDET.RecordingMap(positional=True).unmapped(["A__0"]) == []


@pytest.mark.unit
def test_recording_map_names_the_src_tree_it_was_made_under(tmp_path):
    """Only mapped rows are read, and each must name the caller's src tree in port_src_tree. A label on several
    rows is refused, since a later row would hide an earlier mapped one from the check."""
    path = tmp_path / "recording_map.csv"

    def problems(trees, src_tree="tree1", column=True):
        rows = [{"july_recording": rec, "animal": "A", "animalday": f"A WT Jan-0{i + 1}-2025", "port_position": i,
                 "status": status, "reason": ""} for i, (rec, status) in enumerate(
                     [("A__0", "mapped"), ("A__1", "mapped"), ("A__2", "unmapped")])]
        for row, tree in zip(rows, trees):
            if column:
                row["port_src_tree"] = tree
        pd.DataFrame(rows).to_csv(path, index=False)
        return SDET.RecordingMap(path).port_src_tree_problems(src_tree)

    assert problems(["tree1", "tree1", "tree1"]) == []
    assert problems(["tree1", "tree1", ""]) == [], "an unmapped row maps nothing"
    assert problems(["tree1", "tree1", "tree0"]) == []
    assert problems(["tree1", "tree1", "tree1"], column=False) == [
        "it has no port_src_tree column, so the code it was made under is unknown"]
    assert problems(["tree1", "", "tree1"]) == ["port_src_tree is empty on mapped rows ['A__1']"]
    assert problems(["tree0", "tree0;tree1", "tree1"]) == [
        "its mapped rows were made under src tree tree0, tree0;tree1, and this run's src tree is tree1"]
    assert problems(["tree1", "tree1", "tree1"], src_tree=None) == [
        "git cannot tell this run's src tree (git rev-parse HEAD:src failed)"]
    assert SDET.RecordingMap(positional=True).port_src_tree_problems(None) == []

    stale = {"july_recording": "A__0", "animal": "A", "animalday": "A WT Jan-01-2025", "port_position": 0,
             "status": "mapped", "reason": "", "port_src_tree": "tree0"}
    for again in ({"animalday": "", "port_position": "", "status": "unmapped", "reason": "x", "port_src_tree": ""},
                  {"port_src_tree": "tree1"}):
        pd.DataFrame([stale, {**stale, **again}]).to_csv(path, index=False)
        with pytest.raises(SystemExit, match=r"recording_map.csv: july_recording repeated on several rows: "
                                             r"\['A__0'\]\. compare_window_hashes\.py writes one row per labeled"):
            SDET.RecordingMap(path)


# --------------------------------------------------------------------------- provenance


@pytest.mark.unit
def test_code_files_cover_the_package_and_the_feature_scripts():
    import select_detectors as SD

    files = set(SDET.CODE_FILES)
    assert set((REPO / "src/neurodent").rglob("*.py")) <= files
    for rel in ("scripts/labeling/extract_dev_features.py", "scripts/labeling/score_detectors.py",
                "scripts/labeling/build_cohort_bundle.py", "scripts/labeling/render_context.py", "uv.lock"):
        assert REPO / rel in files, rel
    for rel in SD.SCORING_CODE:  # recorded by the selection, so editing one does not invalidate an extraction
        assert REPO / rel not in files, rel
    assert SDET.CODE_FILES[:2] == [REPO / "src/neurodent/results/autoreject_detector.py",
                                   REPO / "src/neurodent/results/scoring.py"], "the listed entries stay first"
    assert len(files) == len(SDET.CODE_FILES), "no file is listed twice"


@pytest.mark.unit
def test_provenance_records_the_src_tree_and_the_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    marker = tmp_path / "input.txt"
    marker.write_text("x")
    inputs = {"marker": marker, **SDET.config_files(["mini_real"])}
    prov = SDET.provenance(inputs, script="test")
    assert {"git_commit", "git_dirty", "src_tree", "src_dirty", "inputs", "code", "params"} <= set(prov)
    assert prov["src_tree"] == _git("rev-parse", "HEAD:src").strip(), "the tree of src at HEAD, not the commit"
    assert prov["src_dirty"] is bool(_git("status", "--porcelain", "--", "src").strip())
    for rel in ("config/config.yaml", "config/datasets/mini_real.yaml"):
        assert prov["inputs"][rel] == {"path": str(REPO / rel), "sha256": _sha(REPO / rel)}
    assert prov["code"]["uv.lock"] == _sha(REPO / "uv.lock")
    assert prov["params"] == {"script": "test"}


@pytest.mark.unit
def test_score_detectors_refuses_another_working_directory(tmp_path, monkeypatch):
    """A copy of the configs in the working directory would load while the provenance hashes the repository's."""
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config/config.yaml", tmp_path / "config/config.yaml")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run from the repository root"):
        SDET.main(["A=labels_A.csv", "--all"])


# --------------------------------------------------------------------------- autoreject evidence


@pytest.mark.unit
def test_ar_cell_evidence_reads_scores_and_cutoffs_from_the_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 50e-6, (10, 3, 2500))
    X[4, 1] *= 40.0
    fits = adr.fit_grid(X, 500)
    cons = pd.DataFrame([{"recording": "a__0", "window": w, "channel": ch, "t_start_s": 5.0 * w}
                         for w in (2, 4) for ch in ("LMot", "RMot")])
    ev = EDF.ar_cell_evidence(fits, ["LMot", "RMot", "LHip"], np.arange(10) * 5.0, cons)
    bits = EDF.ar_cell_bits({n: f.mask for n, f in fits.items()}, ["LMot", "RMot", "LHip"], np.arange(10) * 5.0,
                            cons)
    assert set(ev) == set(bits) == {(2, "LMot"), (2, "RMot"), (4, "LMot"), (4, "RMot")}
    for (w, ch), per_config in ev.items():
        ci = ["LMot", "RMot", "LHip"].index(ch)
        for name, (score, cutoff) in per_config.items():
            assert score == fits[name].scores[w, ci] and cutoff == fits[name].thresholds[ci]
            assert bits[(w, ch)][name] == int(score > cutoff)


# --------------------------------------------------------------------------- main on a synthetic cohort

DS = "mini_real"  # a dataset with a config file, so the config hashes are real; its animals are faked
CH = ["LMot", "RMot", "LHip"]
FS = 500
N_FRAG = 12
WINDOWS = [2, 5, 8]
ANIMALS = {"a1": ("WT", ["Jan-01-2025", "Jan-02-2025"]),
           "a2": ("Het", ["Jan-01-2025", "Jan-02-2025"]),
           "a3": ("WT", ["Jan-03-2025"])}
UNLABELED = {"a3": ["Jan-04-2025"]}  # recordings in the animal's WAR that no rater saw
# The channels of config/datasets/mini_real.yaml, in order: what its WARs are standardized to.
CANONICAL = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]
# Each recording is built from one session named with its date, except a2's second, which also holds a session
# named with the next day.
TWO_SESSIONS = {("a2", 1): ["010225_a2_files0-9", "010325_a2_files10-19"]}


def _animalday(animal, i):
    genotype, days = ANIMALS[animal]
    return f"{animal} {genotype} {(days + UNLABELED.get(animal, []))[i]}"


def _sessions(animal):
    """Return ``{animalday: [session name, ...]}`` as the loader records them for one synthetic animal."""
    days = ANIMALS[animal][1] + UNLABELED.get(animal, [])
    return {_animalday(animal, i): TWO_SESSIONS.get((animal, i), [
        f"{datetime.strptime(day, '%b-%d-%Y'):%m%d%y}_{animal}"]) for i, day in enumerate(days)}


def _recordings():
    return [(a, i) for a, (_, days) in ANIMALS.items() for i in range(len(days))]


def _consensus():
    rng = np.random.default_rng(1)
    rows = []
    for a, i in _recordings():
        for w in WINDOWS:
            for ch in CH:
                y = int(rng.random() < 0.3)
                rows.append({"recording": f"{a}__{i}", "window": w, "channel": ch, "t_start_s": 5.0 * w,
                             "n_raters": 2, "any_event": False, "y_true": y,
                             "cat_A": "artifact" if y else "clean", "cat_B": "artifact" if y else "event"})
    return pd.DataFrame(rows)


def _fake_animal(animal, nan):
    """Return a fake AnimalOrganizer and a real WindowAnalysisResult for one synthetic animal.

    rms is float32, as in a WAR the loader computes; psdband and psdtotal are float64, as there. Each fragment
    filter rejects some labeled cell:

    - a1: the second recording runs about four times louder than the first, with one window at the first one's
      level; the first recording has a window whose beta proportion is 0/0 on RMot and over 0.4 on LHip.
    - a2: one LMot cell over 500 and one RMot cell under 50; two LHip windows of the first recording are loud
      enough that their log-RMS z score over the whole WAR is over 3, but not over their recording alone.
    - a3: its unlabeled second recording runs three times louder, and one LMot window of the labeled one is
      over 3 in z for that recording alone, but not over the whole WAR.
    """
    genotype, _ = ANIMALS[animal]
    days = ANIMALS[animal][1] + UNLABELED.get(animal, [])
    rng = np.random.default_rng(zlib.crc32(animal.encode()))
    animaldays = [_animalday(animal, i) for i in range(len(days))]
    n = N_FRAG * len(days)
    rms = rng.uniform(80.0, 120.0, (n, len(CH)))
    beta = rng.uniform(0.05, 0.3, (n, len(CH)))
    total = np.full((n, len(CH)), 2.0)
    w0, w1, w2 = WINDOWS
    if animal == "a1":
        rms[N_FRAG:] *= 4.0
        rms[N_FRAG + w1, 0] = 120.0  # low for its own recording, ordinary for the animal
        beta[w2, 2], total[w2, 1] = 0.5, 0.0  # over 0.4 on LHip; 0/0 on RMot
    if animal == "a2":
        rms[w0, 0], rms[N_FRAG + w2, 1] = 650.0, 30.0
        rms[[w0, w2], 2] = 1000.0
    if animal == "a3":
        rms[N_FRAG:] *= 3.0
        rms[w1, 0] = 1000.0
    rms = rms.astype(np.float32)
    band = beta * total
    rows = []
    for k, ad in enumerate(animaldays):
        start = pd.Timestamp(datetime.strptime(ad.split()[-1], "%b-%d-%Y")) + pd.Timedelta(hours=6)
        for w in range(N_FRAG):
            i = k * N_FRAG + w
            rows.append({"animal": animal, "animalday": ad, "genotype": genotype, "duration": 5.0,
                         "isday": bool(w % 2), "timestamp": start + pd.Timedelta(seconds=5 * w),
                         "rms": rms[i], "psdtotal": total[i].tolist(),
                         "psdband": {b: (band[i] if b == "beta" else np.full(len(CH), 0.1)).tolist()
                                     for b in ("delta", "theta", "alpha", "beta", "gamma")}})
    lof = {ad: {"lof_scores": list(rng.uniform(1.0, 3.0, len(CH))), "channel_names": list(CH)} for ad in animaldays}
    war = WindowAnalysisResult(pd.DataFrame(rows), animal_id=animal, genotype=genotype, channel_names=list(CH),
                               lof_scores_dict=lof)
    lros = [SimpleNamespace(name=f"{animal}/{i}", nan=nan) for i in range(len(days))]
    return SimpleNamespace(animaldays=animaldays, long_recordings=lros), war


def _fake_fit_epochs(lro, needed_windows=None, *, fragment_len_s=5.0, max_fit_fragments=None, seed=0, **_):
    rng = np.random.default_rng(zlib.crc32(lro.name.encode()))
    X = rng.normal(0.0, 50e-6, (N_FRAG, len(CH), int(fragment_len_s * FS)))
    X[WINDOWS[0], 1] *= 30.0  # one loud fragment
    if lro.nan:
        X[1, 0, 7] = np.nan
    return adr.FitEpochs(X, list(CH), FS, np.arange(N_FRAG) * fragment_len_s)


def _resolve(name):
    if name not in CANONICAL:
        raise ValueError(f"{name} is not in the channel map")
    return name


def _freeze_code_state(monkeypatch):
    """Pin the code hashes and the src state at their current values, so files edited while a test runs
    cannot change a run's shard key between two runs."""
    code, state = SDET.code_hashes(), SDET.src_state()
    monkeypatch.setattr(SDET, "code_hashes", lambda: dict(code))
    monkeypatch.setattr(SDET, "src_state", lambda: dict(state))
    return code, state


class FakeRun:
    """A synthetic cohort wired into ``EDF.main``; records what was loaded and whether dask started.

    The src state is the repository's src tree with no uncommitted changes, as a run that reads a map requires,
    whatever the working tree holds.
    """

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.out = tmp_path / "out" / "features.parquet"
        self.scores = self.out.with_name("features.autoreject_scores.parquet")
        self.prov_path = Path(str(self.out) + ".provenance.json")
        self.errors_path = Path(str(self.out) + ".errors.json")
        self.parts = Path(str(self.out) + ".parts")
        self.cons = _consensus()
        self.bad_channels = {"a1": {"_all": ["LMot", "RAud", "Bogus"]},
                             "a2": {"A2 Het Jan-02-2025": ["RMot", "LHip"], "A2 Het Jan-03-2025": ["LHip", "RMot"],
                                    "A2 Het Jan-09-2025": ["LMot"]}}
        self.fail, self.nan, self.loaded, self.clusters, self.wars = {}, set(), [], 0, {}
        self.raters = []
        for r in ("A", "B"):
            p = tmp_path / f"labels_{r}.csv"
            p.write_text(f"synthetic {r}\n")
            self.raters.append(f"{r}={p}")
        self.keymap = tmp_path / "keymap.csv"
        self.keymap.write_text("synthetic\n")
        self.write_split("dev")
        monkeypatch.chdir(REPO)
        self.code, self.state = _freeze_code_state(monkeypatch)
        self.state["src_dirty"] = False
        self.map_path = tmp_path / "recording_map.csv"
        self.write_map()

        def iter_cohort(datasets, *, animal=None, limit_per_dataset=None, seed=0):
            samples_config = {"bad_channels": self.bad_channels, "_animal_groups": {"a1": "g12", "a2": "g12"},
                              "channels": {c: [f"raw {c}"] for c in CANONICAL}}
            for a in ANIMALS:
                yield DS, samples_config, {}, a

        def load_and_measure(samples_config, config, animal_id, *, lof_chunk_duration_s, chunk_duration_s):
            self.loaded.append(animal_id)
            if animal_id in self.fail:
                raise self.fail[animal_id]
            ao, self.wars[animal_id] = _fake_animal(animal_id, animal_id in self.nan)
            return ao, self.wars[animal_id], _sessions(animal_id)

        @contextlib.contextmanager
        def dask_cluster(n_workers):
            self.clusters += 1
            yield None

        monkeypatch.setattr(C, "iter_cohort_animals", iter_cohort)
        monkeypatch.setattr(EDF, "load_and_measure", load_and_measure)
        monkeypatch.setattr(EDF, "dask_cluster", dask_cluster)
        monkeypatch.setattr(EDF, "build_consensus", lambda manifests, keymap, rule: self.cons.copy())
        monkeypatch.setattr(EDF, "resolve_channel", _resolve)
        monkeypatch.setattr(adr, "build_fit_epochs", _fake_fit_epochs)

    def write_split(self, which):
        self.split = self.tmp / "split.json"
        other = "test" if which == "dev" else "dev"
        self.split.write_text(json.dumps({which: list(ANIMALS), other: []}))

    def write_map(self, unmapped=(), port_src_tree=None):
        """Write the recording map. Every row names this run's src tree in port_src_tree unless ``port_src_tree``
        gives one value for every row, a ``{recording: value}`` dict over the run's tree, or False to leave the
        column out, as compare_window_hashes.py writes it on every row."""
        rows = []
        for a, i in _recordings():
            rec = f"{a}__{i}"
            if rec in unmapped:
                rows.append({"july_recording": rec, "animal": a, "animalday": "", "port_position": "",
                             "status": "unmapped", "reason": f"{rec} cells differ"})
            else:
                rows.append({"july_recording": rec, "animal": a, "animalday": _animalday(a, i), "port_position": i,
                             "status": "mapped", "reason": ""})
            if port_src_tree is not False:
                trees = port_src_tree if isinstance(port_src_tree, dict) else {}
                default = port_src_tree if isinstance(port_src_tree, str) else self.state["src_tree"]
                rows[-1]["port_src_tree"] = trees.get(rec, default)
        pd.DataFrame(rows).to_csv(self.map_path, index=False)

    def run(self, *extra, which="dev", positional=True):
        argv = [*self.raters, "--keymap", str(self.keymap), "--split", str(self.split), "--which", which,
                "--datasets", DS, "--workers", "1", "--out", str(self.out), *extra]
        argv += ["--positional-recordings"] if positional else ["--recording-map", str(self.map_path)]
        return EDF.main(argv)

    def frame(self):
        return pd.read_parquet(self.out)

    def provenance(self):
        return json.loads(self.prov_path.read_text())

    def output_bytes(self):
        return self.out.read_bytes(), self.scores.read_bytes()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return FakeRun(tmp_path, monkeypatch)


@pytest.mark.unit
def test_cells_carry_config_lists(fake):
    assert fake.run() == 0
    df = fake.frame()
    assert list(df.columns) == EDF.CELL_COLS + ["cat_A", "cat_B"] + GRID
    assert len(df) == len(fake.cons) and not df[GRID].isna().any().any()
    assert df["config_list"].dtype == np.int64 and df["config_list_defined"].dtype == bool
    a1, a2, a3 = (df[df["animal"] == a] for a in ("a1", "a2", "a3"))
    assert a1["config_list_defined"].all() and (a1["config_list"] == (a1["channel"] == "LMot")).all()
    on_day2 = a2["recording"] == "a2__1"
    assert (a2["config_list_defined"] == on_day2).all()
    assert (a2["config_list"] == (on_day2 & a2["channel"].isin(["RMot", "LHip"]))).all()
    assert not a3["config_list_defined"].any() and not a3["config_list"].any()
    assert set(df["cv_group"]) == {"g12", "a3"} and set(df["dataset"]) == {DS}

    prov = fake.provenance()
    assert prov["config_list_unmatched"] == [{"animal": "a2", "dataset": DS, "key": "A2 Het Jan-09-2025",
                                              "reason": "no labeled recording is built from a session named with "
                                                        "Jan-09-2025"}]
    assert [(m["animal"], m["key"], m["recording"], m["animalday"], m["sessions"])
            for m in prov["config_list_matches"]] == [
        ("a1", "_all", "a1__0", "a1 WT Jan-01-2025", ["010125_a1"]),
        ("a1", "_all", "a1__1", "a1 WT Jan-02-2025", ["010225_a1"]),
        ("a2", "A2 Het Jan-02-2025", "a2__1", "a2 Het Jan-02-2025", ["010225_a2_files0-9"]),
        ("a2", "A2 Het Jan-03-2025", "a2__1", "a2 Het Jan-02-2025", ["010325_a2_files10-19"])]
    assert prov["config_list_uncovered"] == [{
        "animal": "a2", "dataset": DS, "recording": "a2__0", "animalday": "a2 Het Jan-01-2025",
        "reason": "no key names Jan-01-2025, the date of sessions ['010125_a2']",
        "sessions": [{"session": "010125_a2", "date": "Jan-01-2025"}]}]
    unknown = {(u["recording"], u["channel"], u["reason"]) for u in prov["config_list_unknown_channels"]}
    assert unknown == {(rec, ch, reason) for rec in ("a1__0", "a1__1") for ch, reason in (
        ("RAud", "RAud is not among the recording's channels"), ("Bogus", "not in the dataset's channel map"))}

    recordings = [f"{a}__{i}" for a, i in _recordings()]
    assert prov["recording_channels"] == {rec: ["LMot", "RMot", "LHip"] for rec in recordings}, \
        "the WAR's channels in the standardized order"
    assert prov["recording_sessions"] == {"a1__0": ["010125_a1"], "a1__1": ["010225_a1"], "a2__0": ["010125_a2"],
                                          "a2__1": ["010225_a2_files0-9", "010325_a2_files10-19"],
                                          "a3__0": ["010325_a3"]}
    none = {"defined": False, "channels": [], "absent": []}
    by_recording = prov["config_list_by_recording"]
    assert by_recording == {"a1__0": {"defined": True, "channels": ["LMot"], "absent": ["RAud"]},
                            "a1__1": {"defined": True, "channels": ["LMot"], "absent": ["RAud"]},
                            "a2__0": none, "a2__1": {"defined": True, "channels": ["RMot", "LHip"], "absent": []},
                            "a3__0": none}
    for rec, cells in df.groupby("recording"):
        assert (cells["config_list_defined"] == by_recording[rec]["defined"]).all(), rec
        assert (cells["config_list"] == cells["channel"].isin(by_recording[rec]["channels"])).all(), rec


@pytest.mark.unit
def test_per_recording_reports_must_cover_the_extracted_recordings(fake, monkeypatch):
    cells_of = EDF.animal_cells

    def without_a2__1(*args, **kwargs):
        cells, scores, report = cells_of(*args, **kwargs)
        report["recording_channels"].pop("a2__1", None)
        return cells, scores, report

    monkeypatch.setattr(EDF, "animal_cells", without_a2__1)
    with pytest.raises(SystemExit, match=r"do not cover exactly the extracted recordings \(recording_channels "
                                         r"\['a2__1'\]\); refusing"):
        fake.run()
    assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path))


@pytest.mark.unit
def test_recording_channels_are_those_standardization_keeps(fake, monkeypatch):
    """A channel_reorder that leaves out a WAR channel drops it from the recording's channels, and a listed channel
    outside it is neither one the recording has nor a padded one it lacks."""
    iter_cohort = C.iter_cohort_animals
    config = {"analysis": {"standardization": {"channel_reorder": ["LHip", "LMot", "RAud"]}}}

    def reordered(*args, **kwargs):
        for ds, samples_config, _, a in iter_cohort(*args, **kwargs):
            yield ds, samples_config, config, a

    monkeypatch.setattr(C, "iter_cohort_animals", reordered)
    fake.bad_channels["a1"] = {"_all": ["LMot", "RMot", "RAud"]}
    assert fake.run() == 0
    prov = fake.provenance()
    assert set(map(tuple, prov["recording_channels"].values())) == {("LHip", "LMot")}
    assert prov["config_list_by_recording"]["a1__0"] == {"defined": True, "channels": ["LMot"], "absent": ["RAud"]}
    assert ("a1__0", "RMot", "RMot is not among the recording's channels") in {
        (u["recording"], u["channel"], u["reason"]) for u in prov["config_list_unknown_channels"]}
    df = fake.frame()
    assert not df.loc[df["channel"] == "RMot", "config_list"].any(), "a channel standardization drops is never listed"


@pytest.mark.unit
def test_filter_scores_follow_the_pipeline_filters(fake, tmp_path):
    """At the config settings, the extracted scores give the decisions the pipeline's filters take on the WAR it
    reads back from parquet, which holds every recording of the animal, labeled or not."""
    assert fake.run() == 0
    df = fake.frame()
    rejected = dict.fromkeys(FILTERS, 0)
    piped = {}
    for animal, war in fake.wars.items():
        assert extract_linear_array(war.result["rms"]).dtype == np.float32, "rms as the loader computes it"
        piped[animal] = _read_back(war, tmp_path / "wars" / animal)
        for name, n in _check_filters(df[df["animal"] == animal], piped[animal]).items():
            rejected[name] += n
    assert all(rejected.values()), f"every filter must reject some labeled cell: {rejected}"

    # The planted windows tell the whole-WAR z from a z over one recording (a2), or over the labeled recordings
    # only (a3, whose louder second recording is unlabeled).
    cells = df.set_index(["recording", "window", "channel"])
    for rec, w, ch, whole in (("a2__0", WINDOWS[0], "LHip", True), ("a2__0", WINDOWS[2], "LHip", True),
                              ("a3__0", WINDOWS[1], "LMot", False)):
        animal, i = rec.split("__")
        result = piped[animal].result
        rms = extract_linear_array(result["rms"])[(result["animalday"] == _animalday(animal, int(i))).to_numpy()]
        z_recording = zscore(np.log(rms), axis=0, nan_policy="omit")[w, CH.index(ch)]
        assert (cells.loc[(rec, w, ch), "logrmsz_abs"] > 3) == whole and (abs(z_recording) > 3) != whole, rec
    assert cells.loc[("a1__1", WINDOWS[1], "LMot"), "logrmsz_abs"] < 1.0, "z over the whole WAR, not per recording"
    assert cells.loc[("a1__0", WINDOWS[2], "LMot"), "beta_wmax"] == pytest.approx(0.5), "a 0/0 hides nothing"


@pytest.mark.unit
def test_scores_file_holds_the_fit_scores_and_cutoffs(fake):
    assert fake.run() == 0
    df, scores = fake.frame(), pd.read_parquet(fake.scores)
    assert list(scores.columns) == EDF.SCORE_COLS and len(scores) == len(df) * len(GRID)
    assert list(scores["config"][:len(GRID)]) == GRID
    merged = scores.merge(df.melt(id_vars=["recording", "window", "channel"], value_vars=GRID,
                                  var_name="config", value_name="bit"),
                          on=["recording", "window", "channel", "config"], validate="one_to_one")
    assert len(merged) == len(scores)
    assert ((merged["score"] > merged["cutoff"]).astype(int) == merged["bit"]).all()
    assert scores.groupby(["recording", "channel", "config"])["cutoff"].nunique().eq(1).all(), \
        "one learned cutoff per channel, configuration and recording"
    assert fake.provenance()["outputs"]["autoreject_scores"] == {"path": str(fake.scores),
                                                                 "sha256": _sha(fake.scores)}


@pytest.mark.unit
def test_provenance_of_a_run(fake):
    assert fake.run() == 0
    prov = fake.provenance()
    assert prov["src_tree"] == _git("rev-parse", "HEAD:src").strip()
    assert prov["src_dirty"] is fake.state["src_dirty"] is False, "the src state the run read"
    for a in ANIMALS:
        key = json.loads(EDF.shard_paths(fake.parts, DS, a)["json"].read_text())["key"]
        assert {k: key[k] for k in ("code", "src_tree", "src_dirty")} == {
            k: prov[k] for k in ("code", "src_tree", "src_dirty")}, "the state the shards were keyed under"
    for rel in ("config/config.yaml", f"config/datasets/{DS}.yaml"):
        assert prov["inputs"][rel]["sha256"] == _sha(REPO / rel)
    assert prov["inputs"]["labels_A"]["sha256"] == _sha(fake.tmp / "labels_A.csv")
    assert prov["outputs"]["features"] == {"path": str(fake.out), "sha256": _sha(fake.out)}
    params = prov["params"]
    assert params["which"] == "dev" and params["configs"] == GRID and params["datasets"] == [DS]
    assert params["dropped_recordings"] == [] and prov["dropped_animals"] == [] and prov["resumed_animals"] == []


@pytest.mark.unit
def test_unmapped_recordings_are_refused_before_anything_loads(fake):
    fake.write_map(unmapped={"a2__1"})
    with pytest.raises(SystemExit, match=r"leaves unmapped: \['a2__1'\]"):
        fake.run(positional=False)
    assert fake.loaded == [] and fake.clusters == 0
    assert not fake.out.exists() and not fake.parts.exists()


@pytest.mark.unit
def test_allow_unmapped_skips_animals_without_a_mapped_recording(fake):
    fake.write_map(unmapped={"a1__1", "a3__0"})
    assert fake.run("--allow-unmapped", positional=False) == 0
    assert fake.loaded == ["a1", "a2"], "an animal with no mapped labeled recording is never loaded"
    df = fake.frame()
    assert set(df["recording"]) == {"a1__0", "a2__0", "a2__1"}
    prov = fake.provenance()
    assert prov["dropped_animals"] == ["a3"]
    assert prov["unmapped_recordings"] == [
        {"recording": "a1__1", "animal": "a1", "status": "unmapped", "reason": "a1__1 cells differ"},
        {"recording": "a3__0", "animal": "a3", "status": "unmapped", "reason": "a3__0 cells differ"}]
    assert prov["params"]["dropped_recordings"] == ["a1__1", "a3__0"] and prov["params"]["allow_unmapped"]


REMAKE_MAP = "Rerun runbook steps 2 and 3 (the sample-hash check and the map commit) with the current code"


@pytest.mark.unit
@pytest.mark.parametrize("trees, state, named, fix", [
    (False, {}, r"it has no port_src_tree column", REMAKE_MAP),
    ({"a2__0": ""}, {}, r"port_src_tree is empty on mapped rows \['a2__0'\]", REMAKE_MAP),
    ({"a1__1": "c" * 40}, {}, r"its mapped rows were made under src tree c{40}, and this run's src tree is ",
     REMAKE_MAP),
    ("tree0;tree1", {"src_tree": "tree1"}, r"made under src tree tree0;tree1, and this run's src tree is tree1",
     REMAKE_MAP),
    ("tree1", {"src_tree": None}, r"git cannot tell this run's src tree", REMAKE_MAP),
    (None, {"src_dirty": True}, r"^src has uncommitted changes \(git status --porcelain -- src lists files\), so "
     r"the code that would load the recordings may not be src tree ", "Remove the changes under src."),
    (None, {"src_dirty": None}, r"^git cannot tell whether src has uncommitted changes \(git status --porcelain "
     r"-- src failed\)", "Run where git can tell.")],
    ids=["no_column", "empty_tree", "other_tree", "joined_trees", "src_tree_unknown", "src_dirty", "dirty_unknown"])
def test_a_map_made_under_other_code_is_refused_before_anything_is_read(fake, monkeypatch, trees, state, named,
                                                                        fix):
    """The map holds only for the src tree of the port run that made it, so another tree, or uncommitted changes
    under src that make the code run differ from that tree, refuse the run before the labels are read, anything
    loads or dask starts, and name the fix."""
    fake.state.update(state)
    fake.write_map(port_src_tree=trees)
    monkeypatch.setattr(EDF, "build_consensus", lambda *a: pytest.fail("the labels were read"))
    with pytest.raises(SystemExit, match=named) as err:
        fake.run(positional=False)
    message = str(err.value)
    if fix == REMAKE_MAP:
        assert message.startswith(f"recording map {fake.map_path}: ")
    else:
        assert f"the only code recording map {fake.map_path} holds for, and the selection refuses features " \
            "extracted from such a src. Refusing before loading anything." in message
        assert REMAKE_MAP not in message, "the map itself is fine"
    assert fix in message
    assert fake.loaded == [] and fake.clusters == 0
    assert not fake.out.exists() and not fake.parts.exists()


@pytest.mark.unit
@pytest.mark.parametrize("again", [
    {"animalday": "", "port_position": "", "status": "unmapped", "reason": "listed again", "port_src_tree": ""},
    {}], ids=["then_unmapped", "then_mapped"])
def test_a_map_listing_a_recording_twice_is_refused_before_anything_is_read(fake, monkeypatch, again):
    """A second row for a labeled recording, unmapped or mapped under this run's tree, would hide the first row,
    made under other code, from the src tree check; the run is refused before the labels are read."""
    fake.write_map(port_src_tree={"a1__0": "0" * 40})
    rows = pd.read_csv(fake.map_path, dtype=str, keep_default_na=False)
    first = rows[rows["july_recording"] == "a1__0"]
    pd.concat([rows, first.assign(**{"port_src_tree": fake.state["src_tree"], **again})]).to_csv(
        fake.map_path, index=False)
    monkeypatch.setattr(EDF, "build_consensus", lambda *a: pytest.fail("the labels were read"))
    with pytest.raises(SystemExit, match=rf"^{re.escape(str(fake.map_path))}: july_recording repeated on several "
                                         r"rows: \['a1__0'\]"):
        fake.run(positional=False)
    assert fake.loaded == [] and fake.clusters == 0
    assert not fake.out.exists() and not fake.parts.exists()


@pytest.mark.unit
def test_only_mapped_rows_name_the_src_tree_and_positional_runs_read_no_map(fake):
    fake.write_map(unmapped={"a3__0"}, port_src_tree={"a3__0": ""})
    assert fake.run("--allow-unmapped", positional=False) == 0, "an unmapped row maps nothing"
    fake.write_map(port_src_tree=False)
    fake.state.update(src_tree="tree0", src_dirty=True)
    assert fake.run() == 0, "--positional-recordings reads no map, so neither src check applies"
    assert fake.provenance()["src_dirty"] is True, "recorded for the selection, which refuses it"


@pytest.mark.unit
def test_failed_animals_are_listed_and_no_parquet_is_written(fake, capsys):
    fake.fail["a2"] = RuntimeError("cannot read the session files")
    fake.nan.add("a3")
    assert fake.run() == 1
    assert fake.loaded == ["a1", "a2", "a3"], "the run goes on after a failure"
    assert not fake.out.exists() and not fake.scores.exists() and not fake.prov_path.exists()
    errors = {e["animal"]: e for e in json.loads(fake.errors_path.read_text())}
    assert set(errors) == {"a2", "a3"}
    assert (errors["a2"]["error_type"], errors["a2"]["message"]) == ("RuntimeError", "cannot read the session files")
    assert errors["a3"]["error_type"] == "ValueError" and "NaN" in errors["a3"]["message"], "the detector's refusal"
    assert all(e["dataset"] == DS and "Traceback" in e["traceback"] for e in errors.values())
    assert "2 of 3 animal(s) failed" in capsys.readouterr().err
    assert EDF.shard_paths(fake.parts, DS, "a1")["json"].exists(), "the finished animal is kept"

    fake.fail.clear()
    fake.nan.clear()
    assert fake.run("--resume") == 0
    assert fake.loaded[3:] == ["a2", "a3"] and fake.provenance()["resumed_animals"] == ["a1"]
    assert not fake.errors_path.exists(), "a successful run removes the earlier errors file"


@pytest.mark.unit
def test_resume_reads_matching_shards_and_writes_the_same_bytes(fake):
    assert fake.run() == 0
    fresh = fake.output_bytes()

    assert fake.run("--resume") == 0
    assert fake.loaded == ["a1", "a2", "a3"] and fake.clusters == 1, "nothing reloaded, no dask started"
    assert fake.output_bytes() == fresh
    assert fake.provenance()["resumed_animals"] == ["a1", "a2", "a3"]

    # A shard computed under another key, and one whose file changed, are recomputed.
    a1, a3 = (EDF.shard_paths(fake.parts, DS, a) for a in ("a1", "a3"))
    doc = json.loads(a1["json"].read_text())
    doc["key"]["params"]["seed"] = 99
    a1["json"].write_text(json.dumps(doc))
    a3["cells"].write_bytes(a3["cells"].read_bytes() + b"x")
    assert fake.run("--resume") == 0
    assert fake.loaded[3:] == ["a1", "a3"] and fake.provenance()["resumed_animals"] == ["a2"]
    assert fake.output_bytes() == fresh

    # Without --resume every animal is recomputed, with the same outputs.
    assert fake.run() == 0
    assert fake.loaded[5:] == ["a1", "a2", "a3"] and fake.output_bytes() == fresh

    # Other parameters, or other code, match no shard.
    assert fake.run("--resume", "--ar-max-fit-fragments", "7") == 0
    assert fake.loaded[8:] == ["a1", "a2", "a3"] and fake.provenance()["resumed_animals"] == []
    assert fake.run("--resume") == 0
    assert fake.loaded[11:] == ["a1", "a2", "a3"]
    fake.code["scripts/labeling/extract_dev_features.py"] = "edited"
    assert fake.run("--resume") == 0
    assert fake.loaded[14:] == ["a1", "a2", "a3"] and fake.output_bytes() == fresh


def _stale(path):
    return path.with_name(path.name + ".stale")


@pytest.mark.unit
def test_a_failed_rerun_leaves_no_outputs_at_out(fake):
    """An earlier run's outputs are renamed to .stale when a run starts, so a failed rerun leaves none at --out."""
    assert fake.run() == 0
    first = [p.read_bytes() for p in (fake.out, fake.scores, fake.prov_path)]
    fake.code["scripts/labeling/extract_dev_features.py"] = "edited"
    fake.fail["a2"] = RuntimeError("cannot read the session files")
    assert fake.run() == 1
    assert fake.errors_path.exists()
    assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path))
    assert [_stale(p).read_bytes() for p in (fake.out, fake.scores, fake.prov_path)] == first

    # A run killed during the loop leaves nothing at --out either, and no errors file of the run before it.
    fake.fail["a2"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        fake.run()
    assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path, fake.errors_path))
    assert _stale(fake.errors_path).exists()


@pytest.mark.unit
def test_mapped_recording_without_cells_is_refused(fake):
    assert fake.run(positional=False) == 0
    late = fake.cons["recording"] == "a1__1"
    fake.cons.loc[late, "window"] += 1000  # beyond the recording, so no cell aligns
    fake.cons.loc[late, "t_start_s"] = fake.cons.loc[late, "window"] * 5.0
    for extra in ((), ("--allow-unmapped",)):
        with pytest.raises(SystemExit, match=r"no extracted cells: \['a1__1'\]"):
            fake.run(*extra, positional=False)
        assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path)), "nor the earlier run's"


def _edit(fake, what):
    if what == "code":
        fake.code["src/neurodent/results/autoreject_detector.py"] = "edited"
    elif what == "src_tree":
        fake.state["src_tree"] = "f" * 40
    else:
        (fake.tmp / "labels_A.csv").write_text("edited\n")


@pytest.mark.unit
@pytest.mark.parametrize("what, named", [("code", "code src/neurodent/results/autoreject_detector.py"),
                                         ("src_tree", "src_tree"), ("labels", "inputs labels_A")])
def test_files_changed_during_the_run_are_refused(fake, monkeypatch, what, named):
    """A change while animals are computed stops the run before the shard of the animal it happened during."""
    load = EDF.load_and_measure

    def load_then_edit(*args, **kwargs):
        out = load(*args, **kwargs)
        if fake.loaded == ["a1", "a2"]:
            _edit(fake, what)
        return out

    monkeypatch.setattr(EDF, "load_and_measure", load_then_edit)
    with pytest.raises(SystemExit, match=f"{named} changed during the run"):
        fake.run()
    assert fake.loaded == ["a1", "a2"]
    a1, a2 = (EDF.shard_paths(fake.parts, DS, a)["json"] for a in ("a1", "a2"))
    assert a1.exists() and not a2.exists(), "no shard is keyed under a state its animal did not run under"
    assert not fake.out.exists() and not fake.prov_path.exists()


@pytest.mark.unit
@pytest.mark.parametrize("what, named", [("code", "code src/neurodent/results/autoreject_detector.py"),
                                         ("src_tree", "src_tree")])
def test_files_changed_after_the_last_animal_are_refused(fake, monkeypatch, what, named):
    """The provenance describes the state every shard was keyed under, so a change before it is written refuses."""
    assert fake.run() == 0
    read = EDF.read_shard

    def read_then_edit(paths):
        _edit(fake, what)
        return read(paths)

    monkeypatch.setattr(EDF, "read_shard", read_then_edit)
    with pytest.raises(SystemExit, match=f"{named} changed during the run"):
        fake.run("--resume")
    assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path))


@pytest.mark.unit
def test_cell_count_is_printed_for_dev_only(fake, capsys):
    assert fake.run() == 0
    assert f"wrote {len(fake.cons)} cells over 3 animals, 5 recordings, {len(GRID)} autoreject" in \
        capsys.readouterr().out
    fake.write_split("test")
    fake.out = fake.tmp / "out" / "test_features.parquet"
    assert fake.run(which="test") == 0
    out = capsys.readouterr().out
    assert f"wrote 3 animals, 5 recordings, {len(GRID)} autoreject configurations" in out
    assert "cells" not in out


@pytest.mark.unit
def test_run_refuses_another_working_directory(fake, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run from the repository root"):
        fake.run()


@pytest.mark.unit
def test_outputs_go_through_selection_event_cells_and_test_scoring(fake, monkeypatch):
    """The dev and test outputs of main, read by select_detectors, report_event_cells and score_test as written.

    The split lists the three synthetic animals in both partitions, and the test run's consensus flips every
    label. a1's list also names RVis by a raw name, which the selection finds through the extraction's abbrev.
    The cells come from one dataset, so the leave-one-dataset-out runs are recorded as not run.
    """
    import report_event_cells as EV
    import score_test as ST
    import select_detectors as SD

    monkeypatch.setattr(EDF, "resolve_channel", lambda name: "RVis" if name == "raw R Vis" else _resolve(name))
    fake.bad_channels["a1"] = {"_all": ["LMot", "RAud", "Bogus", "raw R Vis"]}
    fake.state.update(src_tree="tree0", src_dirty=False)  # a committed, clean src, as the selection requires
    fake.split.write_text(json.dumps({"dev": list(ANIMALS), "test": list(ANIMALS)}))
    out = fake.tmp / "out"
    dev, selection = out / "dev_features.parquet", out / "dev_selection.json"
    fake.out = dev
    assert fake.run() == 0
    dev_prov = json.loads(Path(f"{dev}.provenance.json").read_text())
    assert "scripts/labeling/select_detectors.py" not in dev_prov["code"]
    assert [k for k in ST.MATCHED_PARAMS if k not in dev_prov["params"]] == [], "every parameter score_test compares"
    assert set(ST.matched_code()) == set(dev_prov["code"]), "the code score_test compares is the code recorded"

    SD.main(["--features", str(dev), "--split", str(fake.split), "--out", str(selection), "--boot", "30"])
    picked = json.loads(selection.read_text())
    k = picked["pick"]["autoreject_config"]
    assert k in GRID and picked["features"]["sha256"] == _sha(dev)
    canonical = picked["settings"]["canonical_channels"][DS]
    assert canonical == CANONICAL, "the selection reads the channels the fake cohort standardizes to"
    assert dev_prov["recording_channels"]["a1__0"] == [c for c in canonical if c in CH]
    assert dev_prov["config_list_by_recording"]["a1__0"] == {"defined": True, "channels": ["LMot"],
                                                             "absent": ["RAud", "RVis"]}
    assert [u["channel"] for u in dev_prov["config_list_unknown_channels"] if "abbrev" not in u] == ["Bogus", "Bogus"]
    masks = picked["member_masks"]
    assert masks["padded_channels"]["a1__0"] == [c for c in canonical if c not in CH]
    assert masks["M"]["listed_without_cells"] == {"a1__0": ["RAud", "RVis"], "a1__1": ["RAud", "RVis"]}, \
        "RVis by the extraction's abbrev of its raw name"
    assert masks["M"]["n_defined"] == int(pd.read_parquet(dev)["config_list_defined"].sum())
    assert "not_run" not in picked["nested_replace_logo"]
    assert picked["nested_replace_lodo"]["not_run"] == "leaving one dataset out needs at least 2, and the cells hold 1"

    events = out / "dev_event_cells.csv"
    EV.main(["--features", str(dev), "--selection", str(selection), "--split", str(fake.split), "--out", str(events)])
    assert list(pd.unique(pd.read_csv(events)["detector"]))[:len(SD.COMBINATIONS)] == list(SD.COMBINATIONS)

    flipped = 1 - fake.cons["y_true"]  # with the categories that vote for it, which the scorers check y against
    fake.cons = fake.cons.assign(y_true=flipped, cat_A=np.where(flipped == 1, "artifact", "clean"),
                                 cat_B=np.where(flipped == 1, "artifact", "event"))
    keymap = fake.keymap.read_text()

    def score(name, features):
        return ["--features", str(features), "--pick", str(selection), "--split", str(fake.split),
                "--out", str(out / name), "--boot", "30"]

    fake.keymap.write_text("edited\n")  # an input the two extractions must share
    fake.out = out / "test_other.parquet"
    assert fake.run(which="test") == 0
    with pytest.raises(SystemExit, match="produced differently") as err:
        ST.main(score("refused.json", fake.out))
    assert "input keymap" in str(err.value) and not (out / "refused.json").exists()

    fake.keymap.write_text(keymap)
    fake.out = out / "test_features.parquet"
    assert fake.run(which="test") == 0
    ST.main(score("test_score.json", fake.out))
    result = json.loads((out / "test_score.json").read_text())
    assert result["pick"]["autoreject_config"] == k and list(result["combinations"]) == list(SD.COMBINATIONS)
    assert result["features"]["sha256"] == _sha(fake.out) and result["settings"] == picked["settings"]
    assert {rel: result["provenance"]["code"][rel] for rel in SD.SCORING_CODE} == SD.scoring_code_hashes()
    assert set(pd.DataFrame(result["event_cells"])["detector"]) == set(SD.COMBINATIONS)
    with pytest.raises(SystemExit, match="scored once"):
        ST.main(score("again.json", fake.out))


# --------------------------------------------------------------------------- end to end on mini_real


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_main_end_to_end_mini_real(tmp_path, monkeypatch):
    """Bundle mini_real, fake three raters, run main through a recording map twice (the second with --resume) and
    read the outputs back.

    F22's files sit in a session folder named 051125 while its manual_datetime starts it on May-10, so its
    recording starts on May-10 and is built from a session named with May-11, as the first 031922 session of
    MMUT(2)_cohort6_group1 may start on Mar-18: the May-11 key applies and the May-10 key applies to nothing.
    A10 keeps mini_real's layout, whose pattern has no session. A map made under another src tree, or read while
    src has uncommitted changes, is refused before anything loads. The src state is otherwise pinned clean, so
    the run goes through whatever the working tree holds.
    """
    pytest.importorskip("spikeinterface")
    monkeypatch.chdir(REPO)
    _, src = _freeze_code_state(monkeypatch)
    src["src_dirty"] = False
    real_set = C.set_temp_directory
    monkeypatch.setattr(C, "set_temp_directory", lambda p: real_set(tmp_path / "tmp_prepare"))
    source, data = REPO / ".tests/integration/data", tmp_path / "data"
    for animal, folder in (("A10", data / "A10"), ("F22", data / "F22" / "051125_F22")):
        folder.mkdir(parents=True)
        for f in sorted(source.joinpath(animal).iterdir()):
            if f.suffix in (".bin", ".csv"):
                (folder / f.name).symlink_to(f)
    real_prepare = C._prepare

    def prepare(ds):
        config, samples_config = real_prepare(ds)
        samples_config["data_root"] = str(data)
        samples_config["manual_datetimes"]["F22"] = {"051125_F22": "2025-05-10 10:00:00"}
        samples_config.setdefault("_animal_overrides", {})["F22"] = {"pattern": [
            "{data_root}/{animal}/{session}/{index}_ColMajor.bin", "{data_root}/{animal}/{session}/{index}_Meta.csv"]}
        samples_config["bad_channels"] = {"A10": {"_all": ["LMot", "Nope"]},
                                          "F22": {"F22 KO May-10-2025": ["RHip"], "F22 KO May-11-2025": ["LAud"]}}
        return config, samples_config

    monkeypatch.setattr(C, "_prepare", prepare)
    wars = {}
    measure = EDF.load_and_measure

    def keep_war(samples_config, config, animal_id, **kwargs):
        ao, wars[animal_id], sessions = measure(samples_config, config, animal_id, **kwargs)
        return ao, wars[animal_id], sessions

    monkeypatch.setattr(EDF, "load_and_measure", keep_war)

    assert C.build_cohort(["mini_real"], tmp_path / "bundle_root", n_per_animal=1, seed=0, blind_seed=1,
                          make_bundle=False) == 0
    keymap = next((tmp_path / "bundle_root").rglob("keymap.csv"))
    manifest = pd.read_csv(next((tmp_path / "bundle_root").rglob("manifest.csv")))
    rng = np.random.default_rng(0)
    raters = []
    for r in ("r1", "r2", "r3"):
        filled = manifest.copy()
        for c in (c for c in filled.columns if c.startswith("label_")):
            filled[c] = rng.choice(["bad", "clean", "event"], size=len(filled))
        path = tmp_path / f"fake_{r}.csv"
        filled.to_csv(path, index=False)
        raters.append(f"{r}={path}")
    animals = sorted(set(pd.read_csv(keymap)["animal"].astype(str)))
    assert animals == ["A10", "F22"]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": animals, "test": []}))
    out = tmp_path / "out" / "dev_features.parquet"
    rmap = tmp_path / "recording_map.csv"

    def write_map(tree):  # rows without an animalday map by position, as for a bundle rendered by this code
        rows = [{"july_recording": f"{a}__0", "animal": a, "animalday": "", "port_position": 0, "status": "mapped",
                 "reason": "", "port_src_tree": tree} for a in animals]
        pd.DataFrame(rows).to_csv(rmap, index=False)

    argv = [*raters, "--keymap", str(keymap), "--split", str(split), "--which", "dev", "--recording-map", str(rmap),
            "--datasets", "mini_real", "--workers", "1", "--temp-dir", str(tmp_path / "tmp_extract"),
            "--out", str(out)]
    write_map("0" * 40)
    with pytest.raises(SystemExit, match=r"made under src tree 0{40}.*Rerun runbook steps 2 and 3"):
        EDF.main(argv)
    assert wars == {} and not out.exists()
    write_map(_git("rev-parse", "HEAD:src").strip())
    src["src_dirty"] = True
    with pytest.raises(SystemExit, match=r"^src has uncommitted changes .*Remove the changes under src\."):
        EDF.main(argv)
    assert wars == {} and not out.exists()
    src["src_dirty"] = False
    assert EDF.main(argv) == 0

    df = pd.read_parquet(out)
    assert set(df["animal"]) == {"A10", "F22"} and set(df["recording"]) == {"A10__0", "F22__0"}
    assert not df[GRID + ["rms", "logrmsz_abs", "beta_wmax", "lof"]].isna().any().any()
    assert set(wars) == {"A10", "F22"}
    for animal, war in wars.items():  # the loader's float32 rms against the WAR the pipeline reads back
        assert extract_linear_array(war.result["rms"]).dtype == np.float32
        _check_filters(df[df["animal"] == animal], _read_back(war, tmp_path / "wars" / animal))
    assert df["config_list_defined"].all()
    assert (df["config_list"] == np.where(df["animal"] == "A10", df["channel"] == "LMot",
                                          df["channel"] == "LAud")).all()

    scores_path = out.with_name("dev_features.autoreject_scores.parquet")
    scores = pd.read_parquet(scores_path)
    assert len(scores) == len(df) * len(GRID) and not scores[["score", "cutoff"]].isna().any().any()
    bits = df.melt(id_vars=["recording", "window", "channel"], value_vars=GRID, var_name="config", value_name="bit")
    merged = scores.merge(bits, on=["recording", "window", "channel", "config"], validate="one_to_one")
    assert ((merged["score"] > merged["cutoff"]).astype(int) == merged["bit"]).all()

    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["src_tree"] == _git("rev-parse", "HEAD:src").strip()
    assert prov["src_dirty"] is False, "the pinned src state"
    for rel in ("config/config.yaml", "config/datasets/mini_real.yaml"):
        assert prov["inputs"][rel]["sha256"] == _sha(REPO / rel)
    assert prov["outputs"]["autoreject_scores"]["sha256"] == _sha(scores_path)
    assert prov["recording_sessions"] == {"A10__0": ["unknown"], "F22__0": ["051125_F22"]}, \
        "the loader's own sessions: a pattern without one gives 'unknown'"
    assert [(m["animal"], m["key"], m["animalday"], m["sessions"]) for m in prov["config_list_matches"]] == [
        ("A10", "_all", "A10 WT May-10-2025", ["unknown"]),
        ("F22", "F22 KO May-11-2025", "F22 KO May-10-2025", ["051125_F22"])]
    assert [(u["animal"], u["key"], u["reason"]) for u in prov["config_list_unmatched"]] == [
        ("F22", "F22 KO May-10-2025", "no labeled recording is built from a session named with May-10-2025")]
    assert [(u["animal"], u["channel"]) for u in prov["config_list_unknown_channels"]] == [("A10", "Nope")]
    assert prov["config_list_uncovered"] == []
    assert prov["config_list_by_recording"] == {"A10__0": {"defined": True, "channels": ["LMot"], "absent": []},
                                                "F22__0": {"defined": True, "channels": ["LAud"], "absent": []}}
    for animal, war in wars.items():
        rec = f"{animal}__0"
        assert list(war.channel_abbrevs) != CANONICAL, "the WAR's own order is not the standardized one"
        assert prov["recording_channels"][rec] == [c for c in CANONICAL if c in set(war.channel_abbrevs)] == CANONICAL
        assert set(df.loc[df["recording"] == rec, "channel"]) <= set(prov["recording_channels"][rec])

    first = out.read_bytes(), scores_path.read_bytes()
    assert EDF.main([*argv, "--resume"]) == 0
    assert (out.read_bytes(), scores_path.read_bytes()) == first
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["resumed_animals"] == ["A10", "F22"]
