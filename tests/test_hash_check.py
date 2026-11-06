"""Tests for the label sample-hash check in ``scripts/labeling/``.

``hash_labeled_windows.py`` runs twice on the committed mini_real recordings (A10 and F22) with a synthetic
keymap and manifest, into separate outputs and temp directories; the two outputs must agree hash for hash,
and ``compare_window_hashes.py`` must call every cell identical, classify perturbed copies of the port
output, and leave a recording unmapped when its July-tree length does not reproduce the July draw. Those
tests load recordings through SpikeInterface and are marked integration and slow. The other tests run on
synthetic tables and draws: classification edge cases, the recording map, its July-tree length check and
the exit code, the map's reader in ``score_detectors.py``, and the recovery of July fragment counts by
``july_fragment_counts.py``. Fake recordings stand in for the loader to test read_error rows (and that
``compare_window_hashes.py`` counts them as not hashed), the output written one animal at a time to
``<out>.partial``, and ``first_file`` for intermediates in the temp directory.
"""

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
LABELING = REPO / "scripts" / "labeling"
sys.path.insert(0, str(LABELING))
import compare_window_hashes as CW  # noqa: E402
import hash_labeled_windows as HW  # noqa: E402
import july_fragment_counts as JF  # noqa: E402
import render_context as R  # noqa: E402

# mini_real: A10 is 120.36 s (25 fragments) and F22 466.44 s (94 fragments), both at 1000 Hz with the same
# ten canonical channels. The last window of each is a partial fragment.
MINI = {
    "A10": ("WT", [0, 12, 24], ["LMot", "RHip", "LAud"]),
    "F22": ("KO", [3, 50, 93], ["LVis", "RMot", "RAud"]),
}
KEY = ["animal", "position", "window", "channel"]


def _write_inputs(d):
    """Write a keymap, manifest and fragment-count table for the mini_real animals into ``d``."""
    rows = []
    for animal, (genotype, windows, channels) in MINI.items():
        for w in windows:
            for slot, ch in zip(R.neutral_labels(len(channels)), channels):
                rows.append({"recording": f"{animal}__0", "window": w, "slot": slot, "channel": ch,
                             "animal": animal, "genotype": genotype})
    keymap = d / "keymap.csv"
    pd.DataFrame(rows).to_csv(keymap, index=False)
    manifest = d / "cohort_manifest.json"
    manifest.write_text(json.dumps({"datasets": ["mini_real"], "seed": 0, "n_per_animal": 3,
                                    "rendered_animals": list(MINI), "failures": []}))
    july_n = d / "july_n_fragments.csv"
    july_n.write_text("recording,matches\nA10__0,25\nF22__0,93;94\n")
    return keymap, manifest, july_n


@pytest.fixture(scope="module")
def mini_hashes(tmp_path_factory):
    """Run hash_labeled_windows.py twice on mini_real, as two separate processes from the repository root."""
    pytest.importorskip("spikeinterface")
    d = tmp_path_factory.mktemp("hash_check")
    keymap, manifest, july_n = _write_inputs(d)
    out = {"dir": d, "keymap": keymap, "july_n": july_n}
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    for tree in ("july", "port"):
        out[tree] = d / f"hashes_{tree}.csv"
        cmd = [sys.executable, str(LABELING / "hash_labeled_windows.py"), "--tree", tree,
               "--keymap", str(keymap), "--manifest", str(manifest), "--july-n-fragments", str(july_n),
               "--out", str(out[tree]), "--temp-dir", str(d / f"tmp_{tree}")]
        res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr[-4000:]
    return out


def _read_rows(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def _write_rows(path, fields, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _find(rows, animal, window, channel, position="0"):
    hits = [r for r in rows if (r["animal"], r["position"], r["window"], r["channel"])
            == (animal, position, str(window), channel)]
    assert len(hits) == 1, (animal, position, window, channel)
    return hits[0]


def _compare(mini_hashes, port, tmp_path, july_n=None):
    """Run compare_window_hashes.py against the July output; return (exit code, cells, recording map).

    ``july_n`` is the July fragment-count table; the fixture's table by default.
    """
    cells, rmap = tmp_path / "cells.csv", tmp_path / "recording_map.csv"
    rc = CW.main(["--july", str(mini_hashes["july"]), "--port", str(port), "--keymap", str(mini_hashes["keymap"]),
                  "--july-n-fragments", str(july_n or mini_hashes["july_n"]),
                  "--out", str(cells), "--recording-map", str(rmap)])
    return rc, pd.read_csv(cells, dtype={"port_position": str}), pd.read_csv(rmap).set_index("july_recording")


# --------------------------------------------------------------------------- mini_real runs

@pytest.mark.integration
@pytest.mark.slow
def test_two_runs_agree(mini_hashes):
    """Two runs hash the same cells to the same values, and the values are the samples in the file."""
    read = {"dtype": {"animal": str, "july_n_fragments": str}}
    july, port = pd.read_csv(mini_hashes["july"], **read), pd.read_csv(mini_hashes["port"], **read)
    assert len(july) == len(port) == sum(len(w) * len(c) for _, w, c in MINI.values())
    assert (july["status"] == "ok").all()
    both = july.merge(port, on=KEY, suffixes=("_j", "_p"))
    assert len(both) == len(july)
    assert (both["raw_sha256_j"] == both["raw_sha256_p"]).all()
    assert (both["view_sha256_j"] == both["view_sha256_p"]).all()
    assert set(july["tree"]) == {"july"} and set(port["tree"]) == {"port"}
    assert july["error"].isna().all() and port["error"].isna().all()
    assert not Path(f"{mini_hashes['july']}.partial").exists(), "a finished run renames its output"

    assert july["raw_sha256"].nunique() == len(july), "each cell holds its own samples"
    assert (july["raw_sha256"] != july["view_sha256"]).all(), "the notch filter changes the rater view"
    assert (july["july_recording"] == july["animal"] + "__0").all()
    assert (july["fs"] == 1000).all() and (july["samples_per_fragment"] == 5000).all()
    per_animal = july.groupby("animal").first()
    assert per_animal.loc["A10", "n_fragments"] == 25 and per_animal.loc["F22", "n_fragments"] == 94
    assert per_animal.loc["A10", "july_n_fragments"] == "25" and per_animal.loc["F22", "july_n_fragments"] == "93;94"
    assert per_animal.loc["A10", "animalday"] == "A10 WT May-10-2025"
    last = july[july["window"].isin([24, 93])].groupby("animal")["n_samples"].first()
    assert last.to_dict() == {"A10": 360, "F22": 1440}, "the last fragment is partial"

    # The raw hash is the sha256 of the float32 samples in the file (stored one channel after another).
    row = july[(july["animal"] == "A10") & (july["window"] == 12) & (july["channel"] == "LAud")].iloc[0]
    data = np.fromfile(REPO / ".tests/integration/data/A10/Cage 2 A10-0_ColMajor.bin", dtype="<f4")
    data = data.reshape(10, -1)
    assert data.shape[1] == 120360
    expected = hashlib.sha256(np.ascontiguousarray(data[int(row["column"]), 60000:65000]).tobytes()).hexdigest()
    assert row["raw_sha256"] == expected


@pytest.mark.integration
@pytest.mark.slow
def test_compare_identical_runs(mini_hashes, tmp_path):
    rc, cells, rmap = _compare(mini_hashes, mini_hashes["port"], tmp_path)
    assert rc == 0
    assert len(cells) == 18 and (cells["class"] == "identical").all()
    assert (cells["view_match"]).all()
    assert (rmap["status"] == "mapped").all() and (rmap["length_check"] == "ok").all()
    assert rmap.loc["A10__0", "port_position"] == 0 and rmap.loc["A10__0", "animalday"] == "A10 WT May-10-2025"
    assert rmap.loc["F22__0", "date_string"] == "May-10-2025"
    assert rmap.loc["F22__0", "july_n_fragments"] == 94 and rmap.loc["F22__0", "recovered_n_fragments"] == "93;94"


@pytest.mark.integration
@pytest.mark.slow
def test_compare_july_length_mismatch(mini_hashes, tmp_path):
    """Both trees agree on F22, but its July-tree length does not reproduce the July draw, so it stays unmapped."""
    table = tmp_path / "july_n_fragments.csv"
    table.write_text("recording,matches\nA10__0,25\nF22__0,17222;17223\n")
    rc, cells, rmap = _compare(mini_hashes, mini_hashes["port"], tmp_path, july_n=table)
    assert rc == 1
    assert (cells["class"] == "identical").all()
    assert (rmap.loc["A10__0", "status"], rmap.loc["A10__0", "length_check"]) == ("mapped", "ok")
    f22 = rmap.loc["F22__0"]
    assert (f22["status"], f22["length_check"]) == ("unmapped", "mismatch")
    assert pd.isna(f22["port_position"]) and pd.isna(f22["animalday"])
    assert f22["reason"] == "July-tree n_fragments 94 is not among the July draw lengths 17222;17223"


@pytest.mark.integration
@pytest.mark.slow
def test_compare_perturbed_port(mini_hashes, tmp_path):
    """A swapped channel, a moved row, an altered hash, altered samples and a dropped row are each told apart."""
    fields, rows = _read_rows(mini_hashes["port"])
    a, b = _find(rows, "F22", 50, "LVis"), _find(rows, "F22", 50, "RMot")
    a["channel"], b["channel"] = b["channel"], a["channel"]
    _find(rows, "A10", 24, "LAud")["position"] = "1"
    _find(rows, "A10", 12, "RHip")["raw_sha256"] = "0" * 64                # same RMS and first samples
    changed = _find(rows, "F22", 3, "LVis")
    changed["raw_sha256"], changed["raw_rms"] = "1" * 64, repr(2 * float(changed["raw_rms"]))
    rows.remove(_find(rows, "F22", 93, "RAud"))
    port = tmp_path / "perturbed.csv"
    _write_rows(port, fields, rows)

    rc, cells, rmap = _compare(mini_hashes, port, tmp_path)
    assert rc == 1
    expected = {
        ("F22__0", 50, "LVis"): "channel_moved",
        ("F22__0", 50, "RMot"): "channel_moved",
        ("A10__0", 24, "LAud"): "remapped",
        ("A10__0", 12, "RHip"): "numeric_only",
        ("F22__0", 3, "LVis"): "different",
        ("F22__0", 93, "RAud"): "missing",
    }
    got = {(r.july_recording, r.window, r.channel): r for r in cells.itertuples(index=False)}
    for key, cell in got.items():
        assert cell[cells.columns.get_loc("class")] == expected.get(key, "identical"), key
    assert got[("F22__0", 50, "LVis")].port_channel == "RMot"
    assert got[("A10__0", 24, "LAud")].port_position == "1"
    assert got[("A10__0", 12, "RHip")].port_position == "0"
    assert (rmap["status"] == "unmapped").all()
    assert rmap["port_position"].isna().all() and rmap["animalday"].isna().all()
    assert "numeric_only 1" in rmap.loc["A10__0", "reason"]


@pytest.mark.integration
@pytest.mark.slow
def test_compare_moved_recording(mini_hashes, tmp_path):
    """A recording found whole at another position is remapped, and the map points at the new position."""
    fields, rows = _read_rows(mini_hashes["port"])
    for r in rows:
        if r["animal"] == "F22":
            r["position"] = "2"
    port = tmp_path / "moved.csv"
    _write_rows(port, fields, rows)

    rc, cells, rmap = _compare(mini_hashes, port, tmp_path)
    assert rc == 0
    f22 = cells[cells["animal"] == "F22"]
    assert (f22["class"] == "remapped").all() and (f22["port_position"] == "2").all()
    assert (cells.loc[cells["animal"] == "A10", "class"] == "identical").all()
    assert rmap.loc["F22__0", "status"] == "mapped" and rmap.loc["F22__0", "port_position"] == 2
    assert rmap.loc["F22__0", "n_remapped"] == 9


# --------------------------------------------------------------------------- synthetic tables

def _hash_row(animal, position, window, channel, sha, rms=1.0, head=(1.0, 2.0, 3.0, 4.0), n=100):
    row = {"animal": animal, "position": str(position), "window": str(window), "channel": channel,
           "raw_sha256": sha, "view_sha256": "v" + sha, "raw_rms": repr(rms),
           "animalday": f"{animal} WT day{position}", "date_string": f"day{position}", "n_fragments": str(n),
           "july_n_fragments": ""}
    row.update({f"raw_head_{k}": repr(v) for k, v in enumerate(head)})
    return row


def _index(rows):
    return {(r["animal"], int(r["position"]), int(r["window"]), r["channel"]): r for r in rows}


def _lengths(*labels, n=100):
    """July draw lengths giving each recording the fragment count of :func:`_hash_row`."""
    return {label: [n] for label in labels}


def _run_main(d, cells, july_rows, port_rows, lengths):
    """Write synthetic inputs, run compare_window_hashes.main, and return (exit code, map rows by label).

    Args:
        d (Path): A new directory for the inputs and outputs.
        cells (list[tuple]): ``(july_recording, animal, july_position, window, channel)`` for the keymap.
        july_rows (list[dict]): July hash rows from :func:`_hash_row`.
        port_rows (list[dict]): Port hash rows from :func:`_hash_row`.
        lengths (dict[str, str] | None): July recording label to its July draw lengths as the table writes
            them, or None to leave the table out.
    """
    d.mkdir(parents=True)
    keymap = d / "keymap.csv"
    pd.DataFrame([{"recording": label, "window": w, "slot": "Ch A", "channel": ch, "animal": animal,
                   "genotype": "WT"} for label, animal, _, w, ch in cells]).to_csv(keymap, index=False)
    hashes = {}
    for tree, rows in (("july", july_rows), ("port", port_rows)):
        hashes[tree] = d / f"hashes_{tree}.csv"
        _write_rows(hashes[tree], list(rows[0]), rows)
    table = d / "july_n_fragments.csv"
    if lengths is not None:
        _write_rows(table, ["recording", "matches"], [{"recording": k, "matches": v} for k, v in lengths.items()])
    rmap = d / "recording_map.csv"
    rc = CW.main(["--july", str(hashes["july"]), "--port", str(hashes["port"]), "--keymap", str(keymap),
                  "--july-n-fragments", str(table), "--out", str(d / "cells.csv"), "--recording-map", str(rmap)])
    _, rows = _read_rows(rmap)
    return rc, {r["july_recording"]: r for r in rows}


def test_classify_and_map_synthetic():
    """Shared samples resolve to the July position; ambiguous, split and moved recordings stay unmapped."""
    july = _index([
        _hash_row("X", 0, 20, "A", "flat"), _hash_row("X", 0, 20, "B", "b0"),
        _hash_row("X", 1, 20, "A", "flat"), _hash_row("X", 1, 20, "B", "b1"),
        _hash_row("Y", 0, 7, "A", "flat"),
        _hash_row("Z", 0, 5, "A", "z1"), _hash_row("Z", 0, 5, "B", "z2"),
        _hash_row("N", 0, 9, "A", "n0", rms=5.0),
    ])
    port = _index([
        _hash_row("X", 0, 20, "A", "flat"), _hash_row("X", 0, 20, "B", "b0"),
        _hash_row("X", 1, 20, "A", "flat"), _hash_row("X", 1, 20, "B", "b1"),
        _hash_row("Y", 1, 7, "A", "flat"), _hash_row("Y", 2, 7, "A", "flat"),
        _hash_row("Z", 1, 5, "A", "z1"), _hash_row("Z", 2, 5, "B", "z2"),
        _hash_row("N", 0, 9, "A", "other", rms=9.0), _hash_row("N", 1, 9, "A", "n0-float", rms=5.0),
    ])
    cells = [("X__0", "X", 0, 20, "A"), ("X__0", "X", 0, 20, "B"), ("X__1", "X", 1, 20, "A"),
             ("X__1", "X", 1, 20, "B"), ("Y__0", "Y", 0, 7, "A"), ("Z__0", "Z", 0, 5, "A"),
             ("Z__0", "Z", 0, 5, "B"), ("N__0", "N", 0, 9, "A"), ("Q__0", "Q", 0, 1, "A")]
    out = {(c["july_recording"], c["channel"]): c for c in CW.classify(cells, july, port)}
    assert out[("X__0", "A")]["class"] == "identical" and out[("X__0", "A")]["hash_positions"] == "0;1"
    assert out[("X__1", "B")]["class"] == "identical"
    assert out[("Y__0", "A")]["class"] == "remapped" and out[("Y__0", "A")]["port_position"] == "1;2"
    assert out[("N__0", "A")]["class"] == "numeric_only" and out[("N__0", "A")]["port_position"] == "1"
    assert out[("Q__0", "A")]["class"] == "july_missing"

    lengths = _lengths("X__0", "X__1", "Y__0", "Z__0", "N__0", "Q__0")
    rmap = {r["july_recording"]: r for r in CW.recording_map(list(out.values()), july, port, lengths)}
    assert (rmap["X__0"]["status"], rmap["X__0"]["port_position"]) == ("mapped", 0)
    assert (rmap["X__1"]["status"], rmap["X__1"]["port_position"]) == ("mapped", 1)
    assert rmap["X__1"]["animalday"] == "X WT day1"
    assert rmap["Y__0"]["status"] == "unmapped" and "1;2" in rmap["Y__0"]["reason"]
    assert rmap["Z__0"]["status"] == "unmapped" and "no single port position" in rmap["Z__0"]["reason"]
    assert rmap["N__0"]["status"] == "unmapped" and rmap["N__0"]["port_position"] == ""


def test_recording_map_is_one_to_one():
    """Two July recordings landing on one port position: the one whose position it is keeps it."""
    july = _index([_hash_row("X", 0, 20, "A", "s"), _hash_row("X", 1, 20, "A", "s"),
                   _hash_row("Y", 0, 4, "A", "t"), _hash_row("Y", 1, 4, "A", "t")])
    port = _index([_hash_row("X", 1, 20, "A", "s"), _hash_row("Y", 2, 4, "A", "t")])
    cells = [("X__0", "X", 0, 20, "A"), ("X__1", "X", 1, 20, "A"), ("Y__0", "Y", 0, 4, "A"), ("Y__1", "Y", 1, 4, "A")]
    lengths = _lengths("X__0", "X__1", "Y__0", "Y__1")
    rmap = {r["july_recording"]: r for r in CW.recording_map(CW.classify(cells, july, port), july, port, lengths)}
    assert (rmap["X__1"]["status"], rmap["X__1"]["port_position"]) == ("mapped", 1)
    assert rmap["X__0"]["status"] == "unmapped" and rmap["X__0"]["port_position"] == ""
    assert "also holds the cells of X__0;X__1" in rmap["X__0"]["reason"]
    assert rmap["Y__0"]["status"] == rmap["Y__1"]["status"] == "unmapped"


def test_recording_map_reader(tmp_path):
    """score_detectors.RecordingMap resolves mapped July recordings and ignores unmapped ones."""
    import score_detectors as SDET

    july = _index([_hash_row("X", 0, 20, "A", "a"), _hash_row("Y", 0, 3, "A", "y"),
                   _hash_row("W", 0, 8, "A", "w", n=9000)])
    port = _index([_hash_row("X", 1, 20, "A", "a"), _hash_row("Y", 0, 3, "A", "changed", rms=2.0),
                   _hash_row("W", 0, 8, "A", "w", n=9000)])
    cells = CW.classify([("X__0", "X", 0, 20, "A"), ("Y__0", "Y", 0, 3, "A"), ("W__0", "W", 0, 8, "A")], july, port)
    path = tmp_path / "recording_map.csv"
    CW._write(path, CW.MAP_FIELDS, CW.recording_map(cells, july, port, _lengths("X__0", "Y__0", "W__0")))
    rmap = SDET.RecordingMap(path)
    assert rmap.label("X", 1, "X WT day1") == "X__0"
    assert rmap.label("X", 0, "X WT day0") is None
    assert rmap.label("Y", 0, "Y WT day0") is None
    assert rmap.label("W", 0, "W WT day0") is None, "identical cells, but the July-tree length fails the check"


def test_main_fails_when_a_recording_is_unmapped(tmp_path):
    """Cells that are all identical or remapped still fail the check when their recording is left unmapped."""
    # Held whole at port positions 1 and 2, neither of them the July position.
    rc, rmap = _run_main(tmp_path / "ambiguous", [("Y__0", "Y", 0, 7, "A")], [_hash_row("Y", 0, 7, "A", "t")],
                         [_hash_row("Y", 1, 7, "A", "t"), _hash_row("Y", 2, 7, "A", "t")], {"Y__0": "100"})
    assert rc == 1
    assert rmap["Y__0"]["status"] == "unmapped" and "port positions 1;2" in rmap["Y__0"]["reason"]

    # Each cell is held at another port position.
    rc, rmap = _run_main(tmp_path / "split", [("Z__0", "Z", 0, 5, "A"), ("Z__0", "Z", 0, 5, "B")],
                         [_hash_row("Z", 0, 5, "A", "z1"), _hash_row("Z", 0, 5, "B", "z2")],
                         [_hash_row("Z", 1, 5, "A", "z1"), _hash_row("Z", 2, 5, "B", "z2")], {"Z__0": "100"})
    assert rc == 1 and "no single port position" in rmap["Z__0"]["reason"]

    # Two July recordings land on one port position, which keeps the recording at that July position.
    rc, rmap = _run_main(tmp_path / "one_to_one", [("X__0", "X", 0, 20, "A"), ("X__1", "X", 1, 20, "A")],
                         [_hash_row("X", 0, 20, "A", "s"), _hash_row("X", 1, 20, "A", "s")],
                         [_hash_row("X", 1, 20, "A", "s")], {"X__0": "100", "X__1": "100"})
    assert rc == 1
    assert rmap["X__1"]["status"] == "mapped" and rmap["X__0"]["status"] == "unmapped"

    # Held whole at one other port position: mapped there, and the check passes.
    rc, rmap = _run_main(tmp_path / "moved", [("Y__0", "Y", 0, 7, "A")], [_hash_row("Y", 0, 7, "A", "t")],
                         [_hash_row("Y", 2, 7, "A", "t")], {"Y__0": "100"})
    assert rc == 0 and (rmap["Y__0"]["status"], rmap["Y__0"]["port_position"]) == ("mapped", "2")


def test_july_tree_length_must_reproduce_the_draw(tmp_path, capsys):
    """A recording both trees agree on stays unmapped unless its July-tree length reproduces the July draw."""
    cells = [("X__0", "X", 0, 20, "A")]
    july, port = [_hash_row("X", 0, 20, "A", "s", n=9000)], [_hash_row("X", 0, 20, "A", "s", n=9000)]
    rc, rmap = _run_main(tmp_path / "mismatch", cells, july, port, {"X__0": "17222;17223"})
    x = rmap["X__0"]
    assert rc == 1
    assert (x["status"], x["length_check"], x["port_position"], x["animalday"]) == ("unmapped", "mismatch", "", "")
    assert x["reason"] == "July-tree n_fragments 9000 is not among the July draw lengths 17222;17223"
    assert (x["july_n_fragments"], x["recovered_n_fragments"]) == ("9000", "17222;17223")

    match = [_hash_row("X", 0, 20, "A", "s", n=17223)]
    rc, rmap = _run_main(tmp_path / "match", cells, match, match, {"X__0": "17222;17223"})
    assert rc == 0 and (rmap["X__0"]["status"], rmap["X__0"]["length_check"]) == ("mapped", "ok")

    rc, rmap = _run_main(tmp_path / "not_in_table", cells, july, port, {"Y__0": "9000"})
    assert rc == 1 and (rmap["X__0"]["status"], rmap["X__0"]["length_check"]) == ("unmapped", "unchecked")
    assert rmap["X__0"]["reason"] == "no July draw length to check the July-tree n_fragments 9000 against"

    capsys.readouterr()
    rc, rmap = _run_main(tmp_path / "no_table", cells, july, port, None)
    assert rc == 1 and (rmap["X__0"]["status"], rmap["X__0"]["length_check"]) == ("unmapped", "unchecked")
    assert "no July fragment-count table" in capsys.readouterr().out


def test_check_length():
    assert CW.check_length("17223", [17222, 17223]) == ("ok", "")
    assert CW.check_length(9000, [17222])[0] == "mismatch"
    assert CW.check_length("9000", []) == ("unchecked", "no July draw length to check the July-tree n_fragments "
                                                        "9000 against")
    assert CW.check_length("", [17222]) == ("unchecked", "the July output has no fragment count at the July position")


def test_cells_in_recording():
    cells = {(10, "LMot"): {0}, (30, "LMot"): {1}, (30, "RMot"): {0, 1}}
    assert HW.cells_in_recording(cells, "X", 1, 31) == [(10, "LMot", ""), (30, "LMot", "X__1"), (30, "RMot", "X__1")]
    assert HW.cells_in_recording(cells, "X", 0, 30) == [(10, "LMot", "X__0")]


def test_read_keymap_cells(tmp_path):
    path = tmp_path / "keymap.csv"
    pd.DataFrame([
        {"recording": "a__b__0", "window": 5, "slot": "Ch A", "channel": "LMot", "animal": "a__b", "genotype": "WT"},
        {"recording": "a__b__2", "window": 5, "slot": "Ch B", "channel": "LMot", "animal": "a__b", "genotype": "WT"},
    ]).to_csv(path, index=False)
    assert HW.read_keymap_cells(path) == {"a__b": {(5, "LMot"): {0, 2}}}
    pd.DataFrame([{"recording": "a__0", "window": 5, "slot": "Ch A", "channel": "LMot", "animal": "b",
                   "genotype": "WT"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        HW.read_keymap_cells(path)


def test_temp_dir_must_be_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))     # prepare_temp_dir sets TMPDIR; restored after the test
    fresh = tmp_path / "fresh"
    assert HW.prepare_temp_dir(fresh) == fresh.resolve() and fresh.is_dir()
    (fresh / "rec_mne-to-rec.edf").write_bytes(b"0")
    with pytest.raises(SystemExit, match="not empty"):
        HW.prepare_temp_dir(fresh)
    assert HW.prepare_temp_dir(fresh, reuse=True) == fresh.resolve()


def test_tree_check(tmp_path, monkeypatch):
    """neurodent must be imported from the tree in the working directory."""
    monkeypatch.chdir(REPO)
    assert HW.check_tree() == (REPO / "src" / "neurodent").resolve()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="not inside the working directory"):
        HW.check_tree()


def test_first_source_path_of_split_recording():
    """A recording split out of a shared file has no item; its construction arguments name the file."""
    parent = SimpleNamespace(_kwargs={"file_path": "/data/shared_mne-to-rec.edf"})
    rec = SimpleNamespace(_kwargs={"parent_recording": SimpleNamespace(_kwargs={"parent_recording": parent})})
    assert HW.first_source_path(SimpleNamespace(item=None, LongRecording=rec)) == "/data/shared_mne-to-rec.edf"
    item = SimpleNamespace(get_path_list=lambda: ["/data/x_ColMajor.bin", "/data/x_Meta.csv"])
    assert HW.first_source_path(SimpleNamespace(item=item, LongRecording=rec)) == "/data/x_ColMajor.bin"


# --------------------------------------------------------------------------- fake recordings

FAKE_FS = 100                           # samples per second of the fake recordings
FAKE_SPF = int(HW.FRAG_S * FAKE_FS)     # samples per labeled window


class _FakeTraces:
    """Stands in for a SpikeInterface fragment whose traces are a fixed array."""

    def __init__(self, x):
        self.x = x

    def get_traces(self, return_in_uV=True):
        return self.x


class _FakeLRO:
    """A LongRecordingOrganizer stand-in over a (samples, channels) array at ``FAKE_FS``.

    Reading a window in ``bad_windows`` raises OSError, and ``unopenable`` makes :class:`_FakeAnalyzer` fail
    before the length is known. ``channels`` may name more columns than ``data`` has.
    """

    def __init__(self, data, channels, bad_windows=(), unopenable=False, file_path="/data/x_ColMajor.bin"):
        self.data, self.channel_names = data, list(channels)
        self.bad_windows, self.unopenable = set(bad_windows), unopenable
        self.item, self.animalday, self.file_durations = None, "X WT day0", [len(data) / FAKE_FS]
        self.LongRecording = SimpleNamespace(_kwargs={"file_path": file_path},
                                             time_to_sample_index=lambda t: int(round(t * FAKE_FS)),
                                             get_channel_ids=lambda: list(channels))

    def get_date_string(self):
        return "day0"

    def get_fragment(self, fragment_len_s, index):
        if index in self.bad_windows:
            raise OSError(f"window {index} is unreadable")
        n = int(round(fragment_len_s * FAKE_FS))
        return _FakeTraces(self.data[index * n:(index + 1) * n])


class _FakeAnalyzer:
    """LongRecordingAnalyzer stand-in whose rater view is the raw samples halved."""

    def __init__(self, lro, fragment_len_s, apply_notch_filter):
        if lro.unopenable:
            raise OSError("the recording cannot be opened")
        self.lro, self.fragment_len_s, self.f_s = lro, fragment_len_s, FAKE_FS
        self.n_fragments = math.ceil(len(lro.data) / (fragment_len_s * FAKE_FS))

    def get_fragment_np(self, index):
        return self.lro.get_fragment(self.fragment_len_s, index).get_traces() / 2


def _fake_resolve(names):
    """resolve_channels stand-in: every name is canonical, except that "bad" cannot be resolved."""
    if "bad" in names:
        raise ValueError("channel 'bad' has no canonical name")
    return list(names)


@pytest.fixture
def fake_loader(monkeypatch):
    """Replace the loader: after ``fake_loader(lros)``, every animal loads as the fake recordings ``lros``."""
    monkeypatch.setattr(HW, "LongRecordingAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(HW, "resolve_channels", _fake_resolve)

    def install(lros):
        monkeypatch.setattr(HW, "load_animal_recordings", lambda *a, **k: SimpleNamespace(long_recordings=lros))
    return install


def _signal(n_windows, n_channels, seed):
    """Random float32 samples shaped (samples, channels), ``n_windows`` labeled windows long."""
    return np.random.default_rng(seed).standard_normal((n_windows * FAKE_SPF, n_channels)).astype("<f4")


def _sha(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def test_read_errors_cost_only_the_cells_they_cover(fake_loader):
    """Each failure gives read_error rows for the cells it covers, with no hash, and keeps every other row.

    Position 0 has an unreadable window and names a channel its samples lack; position 1 cannot be opened
    and its length is unknown; position 2 fails after its length is known; position 3 is sound.
    """
    cells = {(0, "A"): {0}, (1, "A"): {0}, (1, "B"): {0}, (2, "A"): {1}, (2, "C"): {0}, (5, "A"): {1}}
    data0, data3 = _signal(3, 2, seed=0), _signal(6, 3, seed=3)
    fake_loader([
        _FakeLRO(data0, ["A", "B", "C"], bad_windows={1}),
        _FakeLRO(_signal(3, 2, seed=1), ["A", "B"], unopenable=True),
        _FakeLRO(_signal(2, 2, seed=2), ["A", "bad"]),
        _FakeLRO(data3, ["A", "B", "C"]),
    ])
    rows = HW.hash_animal("port", "X", {}, {}, cells, {})
    got = {(r["position"], r["window"], r["channel"]): r for r in rows}
    assert len(got) == len(rows)
    assert {k: r["status"] for k, r in got.items()} == {
        (0, 0, "A"): "ok", (0, 1, "A"): "read_error", (0, 1, "B"): "read_error", (0, 2, "A"): "ok",
        (0, 2, "C"): "read_error",
        **{(1, w, ch): "read_error" for w, ch in cells},
        (2, 0, "A"): "read_error", (2, 1, "A"): "read_error", (2, 1, "B"): "read_error",
        **{(3, w, ch): "ok" for w, ch in cells},
    }

    for r in rows:
        if r["status"] == "read_error":
            assert r["error"] and r["raw_sha256"] == r["view_sha256"] == r["raw_rms"] == r["raw_head_0"] == ""
        else:
            assert r["error"] == ""
    assert got[(0, 1, "A")]["error"] == got[(0, 1, "B")]["error"] == "OSError: window 1 is unreadable"
    assert got[(0, 2, "C")]["error"].startswith("IndexError: ")
    assert got[(1, 2, "A")]["error"] == "OSError: the recording cannot be opened"
    assert got[(2, 1, "B")]["error"] == "ValueError: channel 'bad' has no canonical name"

    # A read_error row carries what was known of its recording, so the comparison can still use it.
    assert (got[(0, 1, "A")]["n_fragments"], got[(0, 1, "A")]["start_sample"]) == (3, FAKE_SPF)
    assert (got[(1, 5, "A")]["n_fragments"], got[(1, 5, "A")]["start_sample"]) == ("", "")
    assert got[(1, 5, "A")]["animalday"] == "X WT day0" and got[(1, 2, "A")]["july_recording"] == "X__1"
    assert (got[(2, 1, "A")]["n_fragments"], got[(2, 1, "A")]["start_sample"]) == (2, FAKE_SPF)

    for (p, w, ch), data in [((0, 0, "A"), data0), ((0, 2, "A"), data0)] + [((3, w, ch), data3) for w, ch in cells]:
        x = data[w * FAKE_SPF:(w + 1) * FAKE_SPF, "ABC".index(ch)]
        assert (got[(p, w, ch)]["raw_sha256"], got[(p, w, ch)]["view_sha256"]) == (_sha(x), _sha(x / 2))


def test_compare_counts_read_errors_as_not_hashed(tmp_path, fake_loader):
    """compare_window_hashes.py never calls a read_error cell identical: missing in the port, july_missing in July."""
    keyed = [("X__0", "X", 0, 0, "A"), ("X__0", "X", 0, 1, "A"), ("X__0", "X", 0, 1, "B"),
             ("X__0", "X", 0, 2, "B"), ("X__1", "X", 1, 3, "A")]
    cells = {}
    for _, _, p, w, ch in keyed:
        cells.setdefault((w, ch), set()).add(p)
    d0, d1 = _signal(4, 2, seed=0), _signal(4, 2, seed=1)
    fake_loader([_FakeLRO(d0, ["A", "B"]), _FakeLRO(d1, ["A", "B"], unopenable=True)])
    july = HW.hash_animal("july", "X", {}, {}, cells, {})
    fake_loader([_FakeLRO(d0, ["A", "B"], bad_windows={1}), _FakeLRO(d1, ["A", "B"])])
    port = HW.hash_animal("port", "X", {}, {}, cells, {})
    assert sum(r["status"] == "read_error" for r in july) == len(cells)
    assert sum(r["status"] == "read_error" for r in port) == 2

    rc, rmap = _run_main(tmp_path / "run", keyed, july, port, {"X__0": "4", "X__1": "4"})
    _, cell_rows = _read_rows(tmp_path / "run" / "cells.csv")
    assert {(c["july_recording"], c["window"], c["channel"]): c["class"] for c in cell_rows} == {
        ("X__0", "0", "A"): "identical", ("X__0", "1", "A"): "missing", ("X__0", "1", "B"): "missing",
        ("X__0", "2", "B"): "identical", ("X__1", "3", "A"): "july_missing"}
    assert rc == 1
    assert rmap["X__0"]["status"] == "unmapped" and "missing 2" in rmap["X__0"]["reason"]
    assert rmap["X__1"]["status"] == "unmapped" and "july_missing 1" in rmap["X__1"]["reason"]


def test_first_file_in_the_temp_dir_does_not_name_the_run(tmp_path, fake_loader):
    """An intermediate in the run's temp directory is written as <temp>/ plus its name, the same in both trees."""
    def split(path):
        parent = SimpleNamespace(_kwargs={"file_path": str(path)})
        return SimpleNamespace(item=None, LongRecording=SimpleNamespace(_kwargs={"parent_recording": parent}))

    name = "shared_mne-to-rec.edf"
    temps = {tree: tmp_path / f"hash_tmp_{tree}" for tree in ("july", "port")}
    for t in temps.values():
        t.mkdir()
    assert {HW.first_source_path(split(t / name), t) for t in temps.values()} == {f"<temp>/{name}"}

    link = tmp_path / "link_to_port_tmp"
    link.symlink_to(temps["port"], target_is_directory=True)
    assert HW.first_source_path(split(temps["port"] / name), link) == f"<temp>/{name}"
    assert HW.first_source_path(split(link / name), temps["port"]) == f"<temp>/{name}"

    outside = tmp_path / "data" / name
    assert HW.first_source_path(split(outside), temps["port"]) == str(outside)
    escaped = temps["port"] / ".." / "data" / name
    assert HW.first_source_path(split(escaped), temps["port"]) == str(escaped)
    assert HW.first_source_path(split(temps["port"] / name)) == str(temps["port"] / name)
    item = SimpleNamespace(get_path_list=lambda: [str(temps["july"] / "x_ColMajor.bin")])
    assert HW.first_source_path(SimpleNamespace(item=item), temps["july"]) == "<temp>/x_ColMajor.bin"

    fake_loader([_FakeLRO(_signal(1, 1, seed=0), ["A"], file_path=str(temps["port"] / name))])
    rows = HW.hash_animal("port", "X", {}, {}, {(0, "A"): {0}}, {}, temp_dir=temps["port"])
    assert [(r["status"], r["first_file"]) for r in rows] == [("ok", f"<temp>/{name}")]


# --------------------------------------------------------------------------- output written per animal

class _Killed(BaseException):
    """Stands in for the signal that ends a job at its time or memory limit."""


def _fake_main(monkeypatch, d, outcomes):
    """Prepare hash_labeled_windows.main to run on fake animals, without data.

    Args:
        monkeypatch (pytest.MonkeyPatch): The test's monkeypatch fixture.
        d (Path): A new directory for the inputs, the temp directory and the output.
        outcomes (dict[str, list[str] | BaseException]): Animal to the statuses of the rows it gives, or to
            the exception it raises.

    Returns:
        tuple[list[str], dict[str, list[str]], Path, Path]: main's arguments; a dict filled as main runs,
        giving for each animal started the animals whose rows ``<out>.partial`` held on disk at that moment;
        ``<out>``; and ``<out>.partial``.
    """
    d.mkdir()
    keymap, manifest, out = d / "keymap.csv", d / "cohort_manifest.json", d / "out" / "hashes.csv"
    pd.DataFrame([{"recording": f"{a}__0", "window": 1, "slot": "Ch A", "channel": "A", "animal": a,
                   "genotype": "WT"} for a in outcomes]).to_csv(keymap, index=False)
    manifest.write_text(json.dumps({"datasets": ["ds"], "rendered_animals": list(outcomes)}))
    argv = ["--tree", "port", "--keymap", str(keymap), "--manifest", str(manifest),
            "--july-n-fragments", str(d / "absent.csv"), "--out", str(out), "--temp-dir", str(d / "tmp")]
    partial = out.with_name(out.name + ".partial")
    seen = {}

    def hash_animal(tree, animal, *args, **kwargs):
        seen[animal] = [r["animal"] for r in _read_rows(partial)[1]] if partial.exists() else None
        if isinstance(outcomes[animal], BaseException):
            raise outcomes[animal]
        return [dict(dict.fromkeys(HW.FIELDS, ""), tree=tree, animal=animal, position=0, window=w, channel="A",
                     status=s, raw_sha256="h" if s == "ok" else "") for w, s in enumerate(outcomes[animal])]

    monkeypatch.setattr(HW, "check_tree", lambda: None)
    monkeypatch.setattr(HW, "find_datasets", lambda datasets, animals: ({a: "ds" for a in animals},
                                                                        {"ds": ({}, {})}, []))
    monkeypatch.setattr(HW, "apply_samples_config", lambda samples_config: None)
    monkeypatch.setattr(HW, "hash_animal", hash_animal)
    monkeypatch.setenv("TMPDIR", str(d))     # main sets TMPDIR; restored after the test
    return argv, seen, out, partial


def test_rows_are_flushed_per_animal_and_renamed_at_the_end(tmp_path, monkeypatch):
    """Each animal's rows reach the disk before the next animal starts, and a finished run renames the file."""
    argv, seen, out, partial = _fake_main(monkeypatch, tmp_path / "run", {
        "A": ["ok", "ok"], "B": RuntimeError("cannot load B"), "C": ["ok", "read_error"]})
    out.parent.mkdir()
    out.write_text("left by an earlier run\n")
    assert HW.main(argv) == 1
    assert seen == {"A": [], "B": ["A", "A"], "C": ["A", "A"]}
    assert out.exists() and not partial.exists()
    fields, rows = _read_rows(out)
    assert fields == HW.FIELDS
    assert [(r["animal"], r["status"]) for r in rows] == [("A", "ok"), ("A", "ok"), ("C", "ok"), ("C", "read_error")]

    for name, outcomes, code in [("clean", {"A": ["ok"]}, 0), ("read_error", {"A": ["ok", "read_error"]}, 1)]:
        argv, _, out, partial = _fake_main(monkeypatch, tmp_path / name, outcomes)
        assert HW.main(argv) == code, name
        assert out.exists() and not partial.exists()


def test_a_killed_run_leaves_only_the_partial_file(tmp_path, monkeypatch):
    """A run stopped during an animal keeps the animals it finished in <out>.partial and leaves no <out>."""
    argv, seen, out, partial = _fake_main(monkeypatch, tmp_path / "run", {
        "A": ["ok"], "B": ["ok", "ok"], "C": _Killed(), "D": ["ok"]})
    out.parent.mkdir()
    out.write_text("left by an earlier run\n")
    with pytest.raises(_Killed):
        HW.main(argv)
    assert not out.exists(), "an earlier run's output must not pass for this run's"
    assert list(seen) == ["A", "B", "C"]
    fields, rows = _read_rows(partial)
    assert fields == HW.FIELDS and [r["animal"] for r in rows] == ["A", "B", "B"]


# --------------------------------------------------------------------------- July fragment counts

def test_july_draw_is_select_random():
    """The replica draws what render_context.select_random draws (unchanged since commit 41353542)."""
    for n in (25, 94, 300, 2107, 17280):
        for k in (1, 5, 9):
            if n - 2 * JF.FLANK < k:
                continue
            for seed in (0, 42):
                assert JF.july_draw(n, k, seed) == R.select_random(n, k, seed=seed)
    with pytest.raises(ValueError):
        JF.july_draw(30, 7, 42)


def test_divmod_split():
    assert JF.july_draw_sizes(25, 3) == [9, 8, 8]
    assert JF.july_draw_sizes(25, 4) == [7, 6, 6, 6]
    assert JF.infer_n_recordings(25, {0: 9, 1: 8, 2: 8}) == 3
    assert JF.infer_n_recordings(25, {0: 7, 1: 6, 2: 6}) == 4, "a fourth recording July could not draw from"
    assert JF.infer_n_recordings(25, {0: 3}) == 9
    assert JF.infer_n_recordings(25, {0: 5, 1: 6}) is None, "a later position never gets more windows"


def test_recover_synthetic_draw():
    """Each recovered count reproduces the draw, the true count is among them, and none is missed."""
    seed, n_per_animal, bound = 42, 7, 2500
    true_n = [700, 1234, 2000]
    sizes = JF.july_draw_sizes(n_per_animal, len(true_n))
    recordings = {f"S__{i}": ("S", i, JF.july_draw(n, k, seed)) for i, (n, k) in enumerate(zip(true_n, sizes))}
    rows = {r["recording"]: r for r in JF.recover(recordings, seed, n_per_animal, max_fragments=bound, jobs=1)}
    for i, n in enumerate(true_n):
        r = rows[f"S__{i}"]
        windows = recordings[f"S__{i}"][2]
        brute = [m for m in range(windows[-1] + JF.FLANK + 1, bound + 1) if JF.july_draw(m, sizes[i], seed) == windows]
        matches = [int(m) for m in r["matches"].split(";")]
        assert matches == brute and n in matches
        assert r["n_fragments"] == matches[0] and r["n_matches"] == len(matches) and r["unique"] == (len(matches) == 1)
        assert r["n_recordings"] == 3 and r["divmod_size"] == r["n_windows"] == sizes[i]


def test_scan_in_worker_processes():
    windows = JF.july_draw(1500, 4, 7)
    targets = {4: {tuple(windows)}}
    assert JF.scan(targets, 30, 1601, 7, jobs=2, chunk=400) == JF.scan(targets, 30, 1601, 7, jobs=1)


def test_committed_july_n_fragments():
    """Every n_fragments in the committed table reproduces its recording's keymap windows."""
    manifest = json.loads((REPO / "config/labeling/cohort_manifest.json").read_text())
    recordings = JF.read_keymap_windows(REPO / "config/labeling/keymap.csv")
    table = pd.read_csv(REPO / "config/labeling/july_n_fragments.csv", dtype={"matches": str, "animal": str})
    assert list(table["recording"]) == list(recordings)
    draw_lengths = CW.read_draw_lengths(REPO / "config/labeling/july_n_fragments.csv")
    assert set(draw_lengths) == set(recordings) and all(draw_lengths.values())
    for r in table.itertuples(index=False):
        _, _, windows = recordings[r.recording]
        matches = [int(m) for m in r.matches.split(";")]
        assert r.n_fragments == matches[0] and r.n_matches == len(matches)
        assert r.n_fragments >= windows[-1] + JF.FLANK + 1
        for n in matches:
            assert JF.july_draw(n, len(windows), int(manifest["seed"])) == windows, (r.recording, n)
