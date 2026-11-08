#!/usr/bin/env python
"""Compare the labeled-window hashes of the July tree and the port, and map July recordings to the port's.

Joins a July output and a port output of ``hash_labeled_windows.py``. For each labeled cell in
``keymap.csv`` (July recording, window, channel) it takes the July raw sha256 at the July position and
classifies the port's samples:

* identical: the same raw sha256 at the same position, window and channel.
* remapped: the same raw sha256 at the same window and channel, at another position of the animal.
* channel_moved: the same raw sha256 at the same window under another channel.
* numeric_only: no sha256 match, but RMS and the first samples agree to ``--rtol`` at the same window and
  channel, at the same position or, failing that, at another position of the animal.
* shifted: only with ``--port-scan``, a cell that would be different or missing whose July raw sha256 the
  scan holds at another window: another window of the same port recording, or any window of another one.
* different: no match, and the port has other samples at the same position, window and channel.
* missing: the port output has no samples for the cell at its position, and nothing matches elsewhere.
* july_missing: the July output has no samples for the cell, so nothing can be compared.

The July output stands for what the raters saw only if the July tree still loads the recordings July
rendered. If session files were added or removed since, both trees can agree on samples the raters never
saw. One check of this needs no data: July drew each recording's windows from its fragment count, and
``july_fragment_counts.py`` recovers from the keymap the counts that reproduce each draw
(``config/labeling/july_n_fragments.csv``). A July recording whose fragment count in the July output, at
its July position, is not among them is left unmapped, and so is one that cannot be checked.

It prints counts per class and per animal, writes the per-cell table, and writes ``recording_map.csv``.
The map gives a July recording label the port's stable key (animal, animalday, date string, port
position) when every cell of the recording is identical or remapped, the cells agree on one port position,
and its July-tree fragment count passes the check above. The other July recordings are written with the
key left empty and the reasons. ``score_detectors.py`` and ``extract_dev_features.py`` attach labels to
recordings through this map, so they leave those recordings out. Two rules settle the rare case of the same
samples at several positions (see :func:`recording_map`): a recording held whole at several port positions
maps to its July position if that is one of them, and a port position claimed by several July recordings
stays with the one whose July position it is.

Every map row, mapped or not, also records the July run's n_files, n_fragments and date at the July
position, and the positions, n_files and n_fragments of the port recordings of that animal on that date.
For an unmapped recording, different counts point to files added or dropped, which moves the windows; with
equal counts, a scan (below) tells windows moved some other way, such as files read in another order,
from changed samples. A port recording in which no labeled window fits has no rows in a hash run, so it is
listed only when a scan covers its animal. The map also names the code of each hash run, from the hash
outputs' git_commit, src_dirty and src_tree columns: ``july_commit`` and ``port_commit``, whether src
differed from them (``july_src_dirty``, ``port_src_dirty``), and the git object ids of their ``src``
(``july_src_tree``, ``port_src_tree``). Several values are joined by semicolons. ``extract_dev_features.py``
refuses a map whose port_src_tree is empty or is not its own src tree, so a src tree is written only when it
names the code of every row of its hash output: each row records it and records src_dirty False. Rows of a
run that recorded no tree, or that ran with uncommitted changes under src, must not pass under a tree. The
printout says when the July and port hashes record the same commit or src tree: the July run then did not use
the July code, and the comparison checks the port against itself.

``--port-scan`` reads a port scan, written by ``hash_labeled_windows.py --all-windows`` for some animals. A
scan is used for an animal only when it holds the port hashes' samples: it must share at least one row with
the port hashes (same animal, position, window and channel, hashed in both), and every shared row must have
the same raw sha256. For each animal it passes, the July raw sha256 of each different or missing cell is
looked up among the animal's scan rows. Found anywhere but at the cell's own July position and window, the
cell is classified shifted, and ``port_position``, ``port_window``, ``port_channel`` and ``window_offset``
(port window minus July window, 0 for the same window of another recording) list where, up to
:data:`MAX_LISTED` matches, with ``n_scan_matches`` counting them all; several matches are listed, not
resolved. A window of constant samples matches every window of the same constant, so its matches say little.
A shifted recording stays unmapped and its reason names the port position and window offset of its shifted
cells. A cell that is not shifted keeps its class, and the reason says why: the scan could not read the
animal, was not used for it, or does not hold the samples at another window, the last with the number of the
animal's scan rows it could not read. The scan compares whole windows on the port's 5 s grid, so samples moved
by part of a window are not found. A scan changes the map's reasons and date lists but not its mapping, so
``--port-scan`` refuses to write the committed map and needs ``--recording-map`` with another path.

Run from the repository root after both hash runs::

    python scripts/labeling/compare_window_hashes.py --july hashes_july.csv --port hashes_port.csv \\
        --out window_hash_cells.csv

With a port scan, written to scratch paths::

    python scripts/labeling/compare_window_hashes.py --july hashes_july.csv --port hashes_port.csv \\
        --port-scan scan_port.csv --out window_hash_cells_scan.csv --recording-map recording_map_scan.csv

Exits 1 unless every cell is identical or remapped and every July recording is mapped.
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_KEYMAP = REPO / "config/labeling/keymap.csv"
CONFIG_RECORDING_MAP = REPO / "config/labeling/recording_map.csv"
CONFIG_JULY_N_FRAGMENTS = REPO / "config/labeling/july_n_fragments.csv"

CLASSES = ["identical", "remapped", "channel_moved", "numeric_only", "shifted", "different", "missing",
           "july_missing"]
MAPPABLE = {"identical", "remapped"}
LOOKED_UP = ("different", "missing")    # the classes a port scan may turn into shifted
N_HEAD = 4      # hash_labeled_windows.N_HEAD
SCAN, SCAN_ERROR = "scan", "scan_read_error"    # hash_labeled_windows.SCAN and SCAN_ERROR
MAX_LISTED = 10     # scan matches listed per shifted cell; n_scan_matches counts them all

CELL_FIELDS = [
    "july_recording", "animal", "july_position", "window", "channel", "class", "port_position", "port_channel",
    "hash_positions", "july_raw_sha256", "port_raw_sha256", "july_view_sha256", "port_view_sha256",
    "view_match", "july_rms", "port_rms", "port_window", "window_offset", "n_scan_matches",
]
MAP_FIELDS = [
    "july_recording", "animal", "animalday", "date_string", "port_position", "july_position", "status",
    "reason", "n_cells", "n_identical", "n_remapped", "july_n_fragments", "port_n_fragments",
    "recovered_n_fragments", "length_check", "july_n_files", "july_date_string", "port_positions_on_date",
    "port_n_files_on_date", "port_n_fragments_on_date", "july_commit", "july_src_dirty", "port_commit",
    "port_src_dirty", "july_src_tree", "port_src_tree",
]


def read_keymap_cells(path):
    """Return the labeled cells of a keymap, in keymap order.

    Args:
        path (str | Path): ``keymap.csv`` with columns recording, window, channel and animal.

    Returns:
        list[tuple[str, str, int, int, str]]: ``(july_recording, animal, july_position, window, channel)``.

    Raises:
        ValueError: If a recording label is malformed or disagrees with the row's animal.
    """
    cells, seen = [], set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            animal, _, position = row["recording"].rpartition("__")
            if animal != row["animal"] or not position.isdigit():
                raise ValueError(f"{path}: recording {row['recording']!r} does not match animal {row['animal']!r}")
            cell = (row["recording"], animal, int(position), int(row["window"]), row["channel"])
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
    return cells


def read_hashes(path):
    """Return the rows of a ``hash_labeled_windows.py`` output keyed by (animal, position, window, channel).

    Args:
        path (str | Path): The output CSV.

    Returns:
        dict[tuple[str, int, int, str], dict]: The rows.

    Raises:
        ValueError: If two rows share a key, or a row is a scan row (``--all-windows``), which belongs to
            ``--port-scan``.
    """
    index = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in (SCAN, SCAN_ERROR):
                raise ValueError(f"{path} holds scan rows of hash_labeled_windows.py --all-windows; pass a scan "
                                 "with --port-scan")
            key = (row["animal"], int(row["position"]), int(row["window"]), row["channel"])
            if key in index:
                raise ValueError(f"{path}: two rows for {key}")
            index[key] = row
    return index


def read_draw_lengths(path):
    """Return the fragment counts that reproduce each July recording's window draw.

    Args:
        path (str | Path): ``july_n_fragments.csv`` from ``july_fragment_counts.py``, with columns recording
            and matches, the counts separated by semicolons.

    Returns:
        dict[str, list[int]] | None: July recording label to its counts, or None if the file does not exist.

    Raises:
        ValueError: If the file has no recording or matches column.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"recording", "matches"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return {r["recording"]: [int(n) for n in r["matches"].split(";") if n] for r in reader}


def check_length(n_fragments, lengths):
    """Check a July recording's fragment count in the July output against the counts that reproduce its draw.

    Args:
        n_fragments (str | int): The July output's fragment count at the recording's July position, empty
            when the July output has no row there.
        lengths (list[int]): Fragment counts that reproduce the recording's July window draw.

    Returns:
        tuple[str, str]: The outcome, one of ``"ok"``, ``"mismatch"`` and ``"unchecked"``, and the reason
        when it is not ``"ok"``.
    """
    if n_fragments == "" or n_fragments is None:
        return "unchecked", "the July output has no fragment count at the July position"
    n = int(n_fragments)
    if not lengths:
        return "unchecked", f"no July draw length to check the July-tree n_fragments {n} against"
    if n in lengths:
        return "ok", ""
    return "mismatch", (f"July-tree n_fragments {n} is not among the July draw lengths "
                        + ";".join(str(m) for m in lengths))


def recordings_of(hashes):
    """Return one row per (animal, position) of a hash output, for its recording-level columns."""
    out = {}
    for (animal, position, _, _), row in hashes.items():
        out.setdefault((animal, position), row)
    return out


def _joined(values):
    """Distinct non-empty values, sorted and joined by semicolons."""
    return ";".join(sorted({v for v in values if v}))


def code_of(rows):
    """Return the git commits, src states and src trees that hash rows record, each set of distinct values joined.

    A row's src tree names the code it ran only when the row also records src_dirty False, and
    ``extract_dev_features.py`` takes the port's trees for the code the whole map was made under. So the trees
    are joined only when every row records one with src_dirty False: one row without a tree, or whose src
    differed from its commit or was not checked, leaves them empty.

    Args:
        rows (Iterable[dict]): Hash rows; a row without git_commit, src_dirty or src_tree records nothing
            there.

    Returns:
        tuple[str, str, str]: The commits, the src_dirty values and the src trees, each joined by semicolons
        when the rows record several. The commits and src_dirty values are empty when no row records one,
        the src trees unless every row records one with src_dirty False.
    """
    commits, dirty, trees, clean = set(), set(), set(), True
    for r in rows:
        commits.add(r.get("git_commit") or "")
        value = r.get("src_dirty")
        dirty.add("" if value is None else str(value))
        tree = r.get("src_tree") or ""
        trees.add(tree)
        clean = clean and bool(tree) and str(value) == "False"
    return _joined(commits), _joined(dirty), _joined(trees) if clean else ""


def shared_code(row):
    """Whether a map row's July and port hash runs record a commit or src tree in common.

    Args:
        row (dict): A row of :func:`recording_map`; its code columns may hold several values joined by
            semicolons.

    Returns:
        bool: True when a commit, or a src tree, is among both runs' values. Empty values match nothing.
    """
    def values(column):
        return {v for v in str(row[column]).split(";") if v}

    return bool((values("july_commit") & values("port_commit"))
                or (values("july_src_tree") & values("port_src_tree")))


def recordings_on_date(recordings, animal, date):
    """Return the positions, n_files and n_fragments of an animal's port recordings on one date.

    Args:
        recordings (dict[tuple[str, int], dict]): (animal, position) to a row of that recording.
        animal (str): Animal id.
        date (str): The date string; empty matches nothing.

    Returns:
        dict[str, str]: Values of port_positions_on_date, port_n_files_on_date and port_n_fragments_on_date,
        each a list in position order separated by semicolons.
    """
    on_date = sorted(((q, r) for (a, q), r in recordings.items() if a == animal and date
                      and r.get("date_string") == date), key=lambda item: item[0])
    return {"port_positions_on_date": ";".join(str(q) for q, _ in on_date),
            "port_n_files_on_date": ";".join(str(r.get("n_files", "")) for _, r in on_date),
            "port_n_fragments_on_date": ";".join(str(r.get("n_fragments", "")) for _, r in on_date)}


def read_scan(path, wanted=None, port=None):
    """Read a port scan, the output of ``hash_labeled_windows.py --all-windows``, and check it against the port hashes.

    Args:
        path (str | Path): The scan CSV.
        wanted (set[tuple[str, str]] | None, optional): (animal, raw sha256) pairs whose matches to keep;
            None keeps every scan row's. Defaults to None.
        port (dict | None, optional): The port hash rows, from :func:`read_hashes`. A scan row at the key
            (animal, position, window, channel) of a port row with hashed samples is shared with that row and
            compared with it. None compares nothing, so :func:`scan_usable` passes no animal. Defaults to None.

    Returns:
        dict: ``matches``, (animal, raw sha256) to the (position, window, channel, raw_rms) of each scan row
        holding it; ``recordings``, (animal, position) to one row of that recording, for its
        recording-level columns; ``rows`` and ``errors``, animal to its number of scan and scan_read_error
        rows; ``shared`` and ``disagreements``, animal to its number of scan rows compared with a port row
        and of those with another raw sha256; ``trees``, the tree labels; and ``code``, the commits, src
        states and src trees as :func:`code_of` gives them.

    Raises:
        ValueError: If a row is not a scan row.
    """
    matches, recordings = defaultdict(list), {}
    rows, errors, shared, disagreements, trees, code = Counter(), Counter(), Counter(), Counter(), set(), []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            status = row.get("status")
            if status not in (SCAN, SCAN_ERROR):
                raise ValueError(f"{path}: a row with status {status!r} is not a scan row; --port-scan takes the "
                                 "output of hash_labeled_windows.py --all-windows")
            animal, position = row["animal"], int(row["position"])
            if (animal, position) not in recordings:
                recordings[(animal, position)] = row
                trees.add(row.get("tree") or "")
                code.append(row)
            if status == SCAN_ERROR:
                errors[animal] += 1
                continue
            rows[animal] += 1
            window = int(row["window"])
            port_row = port.get((animal, position, window, row["channel"])) if port is not None else None
            if _hashed(port_row):
                shared[animal] += 1
                if port_row["raw_sha256"] != row["raw_sha256"]:
                    disagreements[animal] += 1
            key = (animal, row["raw_sha256"])
            if wanted is None or key in wanted:
                matches[key].append((position, window, row["channel"], row["raw_rms"]))
    return {"matches": dict(matches), "recordings": recordings, "rows": rows, "errors": errors, "shared": shared,
            "disagreements": disagreements, "trees": _joined(trees), "code": code_of(code)}


def scan_usable(scan, animal):
    """Whether a port scan holds the port hashes' samples of an animal: it shares rows with them and agrees at all.

    Args:
        scan (dict): Output of :func:`read_scan`.
        animal (str): Animal id.

    Returns:
        bool: True when the scan shares at least one row with the port hashes for the animal and holds the
        same raw sha256 at every shared row.
    """
    return bool(scan.get("shared", {}).get(animal)) and not scan.get("disagreements", {}).get(animal)


def find_shifts(cell_rows, scan):
    """Classify as shifted each different or missing cell whose July samples a port scan holds at another window.

    The July raw sha256 of the cell is looked up among the scan rows of its animal, for the animals the scan
    may speak for (:func:`scan_usable`). A match anywhere but at the cell's own July position and window, so
    at another window of the same port recording or at any window of another one, makes the cell shifted; a
    match at its own window, under any channel, is not a shift and changes nothing.

    Args:
        cell_rows (list[dict]): Output of :func:`classify`, updated in place.
        scan (dict): Output of :func:`read_scan`.

    Returns:
        list[dict]: ``cell_rows``. A shifted cell lists its matches in port_position, port_window,
        port_channel and window_offset (port window minus July window), those in its July position first,
        up to :data:`MAX_LISTED`, and n_scan_matches counts them all. Its port_raw_sha256 is the July one
        and its port_rms that of the first match; a scan has no rater view, so port_view_sha256 and
        view_match are empty.
    """
    for rec in cell_rows:
        if rec["class"] not in LOOKED_UP or not rec["july_raw_sha256"] or not scan_usable(scan, rec["animal"]):
            continue
        p, w = int(rec["july_position"]), int(rec["window"])
        hits = sorted((q != p, q, v, ch, rms)
                      for q, v, ch, rms in scan["matches"].get((rec["animal"], rec["july_raw_sha256"]), [])
                      if (q, v) != (p, w))
        if not hits:
            continue
        listed = hits[:MAX_LISTED]
        rec.update({"class": "shifted", "port_position": ";".join(str(q) for _, q, _, _, _ in listed),
                    "port_window": ";".join(str(v) for _, _, v, _, _ in listed),
                    "port_channel": ";".join(ch for _, _, _, ch, _ in listed),
                    "window_offset": ";".join(str(v - w) for _, _, v, _, _ in listed),
                    "n_scan_matches": len(hits), "port_raw_sha256": rec["july_raw_sha256"],
                    "port_view_sha256": "", "port_rms": listed[0][4], "view_match": ""})
    return cell_rows


def _count(n, noun):
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def shift_reasons(cells):
    """Name where the shifted cells of one July recording were found.

    Args:
        cells (list[dict]): The recording's rows from :func:`classify` and :func:`find_shifts`.

    Returns:
        list[str]: One reason per (port position, window offset) of the cells with a single match, then one
        for the cells with several matches, if any, naming the port positions and offsets their listed
        matches share, up to five.
    """
    groups, several, places = Counter(), 0, set()
    for c in cells:
        if c["class"] != "shifted":
            continue
        positions, offsets = str(c["port_position"]).split(";"), str(c["window_offset"]).split(";")
        if int(c["n_scan_matches"]) == 1:
            groups[(int(positions[0]), int(offsets[0]))] += 1
        else:
            several += 1
            places |= {(int(q), int(o)) for q, o in zip(positions, offsets)}
    reasons = [f"shifted by window offset {offset:+d} at port position {q}: {_count(n, 'cell')}"
               for (q, offset), n in sorted(groups.items())]
    if several:
        named = ", ".join(f"offset {o:+d} at port position {q}" for q, o in sorted(places)[:5])
        more = f" and {len(places) - 5} more" if len(places) > 5 else ""
        reasons.append(f"shifted with several matches each: {_count(several, 'cell')}, among them {named}{more}")
    return reasons


def unshifted_reasons(cells, scan):
    """Say why a port scan left the different and missing cells of one July recording as they were.

    Args:
        cells (list[dict]): The recording's rows from :func:`classify` and :func:`find_shifts`.
        scan (dict): The port scan from :func:`read_scan`.

    Returns:
        list[str]: Nothing when no such cell is left or the scan has no row of the animal. Otherwise one
        reason with the cells' counts: the scan could not read the animal, or was not used for it
        (:func:`scan_usable`), or does not hold the samples at another window, in which case the number of
        the animal's scan_read_error rows follows when there are any.
    """
    animal = cells[0]["animal"]
    left = Counter(c["class"] for c in cells if c["class"] in LOOKED_UP and c["july_raw_sha256"])
    n_rows, n_errors = scan["rows"].get(animal, 0), scan["errors"].get(animal, 0)
    if not left or not (n_rows or n_errors):
        return []
    n_shared, n_other = scan["shared"].get(animal, 0), scan["disagreements"].get(animal, 0)
    if not n_rows:
        why = "not looked up in the port scan, which could not read this animal"
    elif n_other:
        why = (f"not looked up in the port scan, which disagrees with the port hashes at {n_other} of "
               f"{_count(n_shared, 'shared row')} of this animal")
    elif not n_shared:
        why = "not looked up in the port scan, which shares no hashed row with the port hashes of this animal"
    else:
        why = "not found at another window in the port scan"
        if n_errors:
            why += f", which could not read {_count(n_errors, 'row')} of this animal"
    return [why + ": " + ", ".join(f"{k} {left[k]}" for k in LOOKED_UP if k in left)]


def _close(a, b, rtol):
    """Whether two numbers written by ``hash_labeled_windows.py`` agree to ``rtol`` relative."""
    if a == "" or b == "":
        return False
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= rtol * max(abs(a), abs(b))


def numeric_match(july_row, port_row, rtol):
    """Whether the RMS and the first samples of two cells agree to ``rtol`` relative."""
    keys = ["raw_rms"] + [f"raw_head_{k}" for k in range(N_HEAD)]
    return all(_close(july_row.get(k, ""), port_row.get(k, ""), rtol) for k in keys)


def _hashed(row):
    """Return whether a hash row holds hashed samples: status ok and a raw sha256 (read_error rows hold none)."""
    return row is not None and row.get("status", "ok") == "ok" and bool(row["raw_sha256"])


def classify(cells, july, port, rtol=1e-6):
    """Classify every labeled cell by where, if anywhere, the port holds its July samples.

    Args:
        cells (list[tuple]): ``(july_recording, animal, july_position, window, channel)`` from the keymap.
        july (dict): The July hash rows, from :func:`read_hashes`.
        port (dict): The port hash rows, from :func:`read_hashes`.
        rtol (float, optional): Relative tolerance of the numeric comparison. Defaults to 1e-6.

    Returns:
        list[dict]: One row per cell with the columns in :data:`CELL_FIELDS`. ``hash_positions`` lists every
        port position holding the July raw sha256 at the same window and channel.
    """
    hashed = defaultdict(list)          # (animal, window): port rows with samples
    for (animal, _, window, _), row in port.items():
        if _hashed(row):
            hashed[(animal, window)].append(row)

    out = []
    for label, animal, p, w, ch in cells:
        rec = dict.fromkeys(CELL_FIELDS, "")
        rec.update(july_recording=label, animal=animal, july_position=p, window=w, channel=ch)
        j = july.get((animal, p, w, ch))
        if not _hashed(j):
            rec["class"] = "july_missing"
            out.append(rec)
            continue
        rec.update(july_raw_sha256=j["raw_sha256"], july_view_sha256=j["view_sha256"], july_rms=j["raw_rms"])

        h = j["raw_sha256"]
        same = port.get((animal, p, w, ch))
        same = same if _hashed(same) else None
        candidates = hashed.get((animal, w), [])
        positions = sorted({int(r["position"]) for r in candidates if r["channel"] == ch and r["raw_sha256"] == h})
        rec["hash_positions"] = ";".join(str(q) for q in positions)
        moved = sorted((int(r["position"]) != p, int(r["position"]), r["channel"])
                       for r in candidates if r["channel"] != ch and r["raw_sha256"] == h)
        near = sorted((int(r["position"]) != p, int(r["position"]))
                      for r in candidates if r["channel"] == ch and numeric_match(j, r, rtol))

        match = None
        if p in positions:
            rec["class"], match = "identical", same
            rec["port_position"] = p
        elif positions:
            rec["class"], match = "remapped", port[(animal, positions[0], w, ch)]
            rec["port_position"] = ";".join(str(q) for q in positions)
        elif moved:
            rec["class"], match = "channel_moved", port[(animal, moved[0][1], w, moved[0][2])]
            rec["port_position"] = ";".join(str(q) for _, q, _ in moved)
            rec["port_channel"] = ";".join(c for _, _, c in moved)
        elif near:
            rec["class"], match = "numeric_only", port[(animal, near[0][1], w, ch)]
            rec["port_position"] = ";".join(str(q) for _, q in near)
        elif same is not None:
            rec["class"], match = "different", same
            rec["port_position"] = p
        else:
            rec["class"] = "missing"
        if match is not None:
            rec.update(port_raw_sha256=match["raw_sha256"], port_view_sha256=match["view_sha256"],
                       port_rms=match["raw_rms"], view_match=match["view_sha256"] == j["view_sha256"])
            if not rec["port_channel"]:
                rec["port_channel"] = match["channel"]
        out.append(rec)
    return out


def recording_map(cell_rows, july, port, draw_lengths, scan=None):
    """Map each July recording to the port recording that holds all of its labeled samples.

    A July recording is mapped when every one of its cells is identical or remapped, one port position
    holds the July raw sha256 of every cell, and its fragment count in the July output is among its July
    draw lengths (:func:`check_length`). When several positions hold every cell, the July position is taken
    if it is one of them; otherwise the recording is left unmapped as ambiguous. The map is one to one: when
    several July recordings land on one port position, only the one whose July position it is keeps it,
    since ``score_detectors.RecordingMap`` would otherwise keep whichever row it read last. An unmapped row
    leaves animalday and port_position empty, so ``RecordingMap`` resolves nothing to it.

    Every row, mapped or not, also gets the July output's n_files and date at the July position, the port
    recordings of the animal on that date (:func:`recordings_on_date`), and the code of both hash runs
    (:func:`code_of`). A shifted cell leaves its recording unmapped, and the reason names where the
    shifted cells were found (:func:`shift_reasons`).

    Args:
        cell_rows (list[dict]): Output of :func:`classify`, and of :func:`find_shifts` with a scan.
        july (dict): The July hash rows, from :func:`read_hashes`.
        port (dict): The port hash rows, from :func:`read_hashes`.
        draw_lengths (dict[str, list[int]]): July recording label to the fragment counts that reproduce its
            window draw, from :func:`read_draw_lengths`. A recording with none cannot be checked and is left
            unmapped.
        scan (dict | None, optional): The port scan from :func:`read_scan`. Its recordings join the port's in
            the date lists, and the reason of an unmapped recording of a scanned animal says why its different
            and missing cells were not shifted (:func:`unshifted_reasons`). Defaults to None.

    Returns:
        list[dict]: One row per July recording with the columns in :data:`MAP_FIELDS`. The reason column
        lists every reason a recording is unmapped, separated by "; ".
    """
    july_recs, port_recs = recordings_of(july), recordings_of(port)
    listed = dict(port_recs)
    if scan is not None:
        for key, r in scan["recordings"].items():
            listed.setdefault(key, r)
    july_commit, july_dirty, july_tree = code_of(july.values())
    port_commit, port_dirty, port_tree = code_of(port.values())
    by_rec = defaultdict(list)
    for c in cell_rows:
        by_rec[c["july_recording"]].append(c)

    rows = []
    for label, cs in by_rec.items():
        animal, p = cs[0]["animal"], cs[0]["july_position"]
        counts = Counter(c["class"] for c in cs)
        j = july_recs.get((animal, p), {})
        lengths = draw_lengths.get(label, [])
        check, length_reason = check_length(j.get("n_fragments", ""), lengths)
        row = dict.fromkeys(MAP_FIELDS, "")
        row.update(july_recording=label, animal=animal, july_position=p, status="unmapped", n_cells=len(cs),
                   n_identical=counts["identical"], n_remapped=counts["remapped"],
                   july_n_fragments=j.get("n_fragments", ""),
                   recovered_n_fragments=";".join(str(n) for n in lengths), length_check=check,
                   july_n_files=j.get("n_files", ""), july_date_string=j.get("date_string", ""),
                   **recordings_on_date(listed, animal, j.get("date_string", "")),
                   july_commit=july_commit, july_src_dirty=july_dirty, port_commit=port_commit,
                   port_src_dirty=port_dirty, july_src_tree=july_tree, port_src_tree=port_tree)
        other = {k: v for k, v in counts.items() if k not in MAPPABLE}
        reasons, q = [], None
        if other:
            reasons.append("cells neither identical nor remapped: " + ", ".join(
                f"{k} {other[k]}" for k in CLASSES if k in other))
            reasons += shift_reasons(cs)
            if scan is not None:
                reasons += unshifted_reasons(cs, scan)
        else:
            common = set.intersection(*({int(x) for x in c["hash_positions"].split(";")} for c in cs))
            if p in common:
                q = p
            elif len(common) == 1:
                q = common.pop()
            elif not common:
                reasons.append("no single port position holds every cell")
            else:
                reasons.append("every cell is held at port positions " + ";".join(str(x) for x in sorted(common)))
        if check != "ok":
            reasons.append(length_reason)
        row["reason"] = "; ".join(reasons)
        if q is not None and not reasons:
            r = port_recs[(animal, q)]
            row.update(status="mapped", port_position=q, animalday=r["animalday"], date_string=r["date_string"],
                       port_n_fragments=r["n_fragments"])
        rows.append(row)

    claims = defaultdict(list)
    for row in rows:
        if row["status"] == "mapped":
            claims[(row["animal"], row["port_position"])].append(row)
    for (_, q), claimants in claims.items():
        if len(claimants) < 2:
            continue
        names = ";".join(c["july_recording"] for c in claimants)
        own = [c for c in claimants if c["july_position"] == q]
        for c in claimants:
            if len(own) == 1 and c is own[0]:
                continue
            c.update(status="unmapped", reason=f"port position {q} also holds the cells of {names}",
                     port_position="", animalday="", date_string="", port_n_fragments="")
    return rows


def _write(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")  # LF, as git stores the committed map
        w.writeheader()
        w.writerows(rows)


def _dash(value):
    return "-" if value is None or value == "" else str(value)


def day_text(r):
    """Describe a map row's July recording and the port recordings of its animal on its date, as one line."""
    positions = [q for q in str(r["port_positions_on_date"]).split(";") if q]
    files, frags = str(r["port_n_files_on_date"]).split(";"), str(r["port_n_fragments_on_date"]).split(";")
    port = ", ".join(f"position {q} (n_files {_dash(f)}, n_fragments {_dash(n)})"
                     for q, f, n in zip(positions, files, frags))
    return (f"July n_files {_dash(r['july_n_files'])}, date {_dash(r['july_date_string'])}; port recordings of "
            f"the animal that day: {port or 'none'}")


def summarize(cell_rows, map_rows, scan=None):
    """Print counts per class and per animal, the code of each hash run, the port scan, and the recording map.

    The code lines warn when a run's code is not named by one commit and one src tree, when the July and port
    runs share a commit or src tree (:func:`shared_code`), and when the map's port_src_tree is empty or names
    several trees, which ``extract_dev_features.py`` refuses. They change neither the map nor the exit code.

    Args:
        cell_rows (list[dict]): Output of :func:`classify`, and of :func:`find_shifts` with a scan.
        map_rows (list[dict]): Output of :func:`recording_map`.
        scan (dict | None, optional): The port scan from :func:`read_scan`. Defaults to None.
    """
    total = Counter(c["class"] for c in cell_rows)
    print(f"{len(cell_rows)} labeled cells")
    for k in CLASSES:
        print(f"  {k:<14} {total.get(k, 0)}")
    held = [c for c in cell_rows if c["class"] in MAPPABLE]
    n_view = sum(1 for c in held if c["view_match"] is False)
    print(f"identical or remapped cells whose rater-view sha256 differs: {n_view} of {len(held)}")

    per_animal = defaultdict(Counter)
    for c in cell_rows:
        per_animal[c["animal"]][c["class"]] += 1
    width = max([len("animal")] + [len(a) for a in per_animal])
    print("\n" + f"{'animal':<{width}} " + " ".join(f"{k:>13}" for k in CLASSES))
    for animal, counts in per_animal.items():
        print(f"{animal:<{width}} " + " ".join(f"{counts.get(k, 0):>13}" for k in CLASSES))

    if map_rows:
        print()
        for tree in ("july", "port"):
            commit, dirty, src = (str(map_rows[0][f"{tree}_{k}"]) for k in ("commit", "src_dirty", "src_tree"))
            print(f"{tree} hashes: git commit {commit or 'not recorded'}, src dirty {dirty or 'not recorded'}, "
                  f"src tree {src or 'not written'}")
            if ";" in commit:
                print(f"  the {tree} hashes come from several commits")
            if ";" in src:
                print(f"  the {tree} hashes come from several src trees")
            if "True" in dirty.split(";"):
                print(f"  src differed from the commit in a {tree} hash run, so neither the commit nor the src tree "
                      "names the code")
            elif not src:
                print(f"  the {tree} hash rows do not all record a src tree with src_dirty False")
        if shared_code(map_rows[0]):
            print("  the July and port hashes record the same commit or src tree, so the July run did not use the "
                  "July code and the comparison checks the port against itself")
        port_tree = str(map_rows[0]["port_src_tree"])
        if not port_tree or ";" in port_tree:
            print("  port_src_tree names no single src tree, so extract_dev_features.py refuses this map; rerun the "
                  "port hash run with the current code, committed and with src clean, from the root of the port's "
                  "git worktree, then this comparison")
    if scan is not None:
        n_shifted = sum(1 for c in cell_rows if c["class"] == "shifted")
        print(f"\nport scan: {sum(scan['rows'].values())} scan rows of {len(scan['rows'])} animal(s), tree "
              f"{scan['trees'] or 'not recorded'}, git commit {scan['code'][0] or 'not recorded'}, src dirty "
              f"{scan['code'][1] or 'not recorded'}, src tree {scan['code'][2] or 'not recorded'}; different or "
              f"missing cells found at another window: {n_shifted}")
        port_code = (tuple(str(map_rows[0][f"port_{k}"]) for k in ("commit", "src_dirty", "src_tree"))
                     if map_rows else ("", "", ""))
        if scan["code"] != port_code:
            print("  the scan and the port hashes record different code, so a shift found in the scan may not hold "
                  "for the port hashes")
        for animal in sorted(set(scan["rows"]) | set(scan["errors"])):
            n_shared, n_other = scan["shared"].get(animal, 0), scan["disagreements"].get(animal, 0)
            if n_other:
                print(f"  {animal}: the scan disagrees with the port hashes at {n_other} of "
                      f"{_count(n_shared, 'shared row')}, so it does not hold the port's samples and is not used for "
                      "this animal")
            elif not n_shared:
                print(f"  {animal}: the scan shares no hashed row with the port hashes, so it cannot be checked and "
                      "is not used for this animal")
            else:
                print(f"  {animal}: the scan agrees with the port hashes at {_count(n_shared, 'shared row')}")
            if scan["errors"].get(animal):
                print(f"  {animal}: {scan['errors'][animal]} scan row(s) could not be read, so samples there cannot "
                      "be found")

    checks = Counter(r["length_check"] for r in map_rows)
    print(f"\nJuly-tree n_fragments among the July draw lengths: {checks.get('ok', 0)} of {len(map_rows)} July "
          f"recordings; not among them: {checks.get('mismatch', 0)}; not checked: {checks.get('unchecked', 0)}")
    mapped = [r for r in map_rows if r["status"] == "mapped"]
    print(f"{len(mapped)} of {len(map_rows)} July recordings mapped")
    for r in map_rows:
        target = f"port position {r['port_position']} ({r['animalday']})" if r["status"] == "mapped" else r["reason"]
        line = (f"  {r['july_recording']}: {r['status']}, {target}; n_fragments July {r['july_n_fragments'] or '-'}, "
                f"port {r['port_n_fragments'] or '-'}, July draw {r['recovered_n_fragments'] or '-'}")
        if r["status"] != "mapped":
            line += f"; {day_text(r)}"
        print(line)


def main(argv=None):
    """Classify every labeled cell, write the cell table and the recording map, and print a summary.

    Args:
        argv (list[str], optional): Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 when every cell is identical or remapped and every July recording is mapped, 1 otherwise.

    Raises:
        SystemExit: With status 2 for invalid arguments, among them ``--port-scan`` with the committed
            recording map as ``--recording-map``, before anything is read or written.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--july", type=Path, required=True, help="hash_labeled_windows.py output of the July tree")
    p.add_argument("--port", type=Path, required=True, help="hash_labeled_windows.py output of the port")
    p.add_argument("--keymap", type=Path, default=CONFIG_KEYMAP, help="keymap.csv of the labeled cells")
    p.add_argument("--july-n-fragments", type=Path, default=CONFIG_JULY_N_FRAGMENTS,
                   help="july_n_fragments.csv from july_fragment_counts.py, the fragment counts that reproduce "
                        "each July window draw (default %(default)s)")
    p.add_argument("--out", type=Path, required=True, help="per-cell output CSV")
    p.add_argument("--recording-map", type=Path, default=CONFIG_RECORDING_MAP,
                   help="recording map output CSV (default %(default)s); another path is required with --port-scan")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance on RMS and first samples for numeric_only (default %(default)s)")
    p.add_argument("--port-scan", type=Path, default=None,
                   help="a scan of the port by hash_labeled_windows.py --all-windows: a different or missing cell "
                        "whose July samples it holds at another window is classified shifted, for each animal "
                        "where the scan agrees with the port hashes")
    args = p.parse_args(argv)
    if args.port_scan is not None and args.recording_map.resolve() == CONFIG_RECORDING_MAP.resolve():
        p.error(f"--port-scan writes a map that differs from the committed one at {CONFIG_RECORDING_MAP}; pass "
                "--recording-map with a scratch path")

    cells = read_keymap_cells(args.keymap)
    if not cells:
        raise SystemExit(f"{args.keymap} lists no labeled cells")
    july, port = read_hashes(args.july), read_hashes(args.port)
    draw_lengths = read_draw_lengths(args.july_n_fragments)
    if draw_lengths is None:
        print(f"no July fragment-count table at {args.july_n_fragments}, so no July-tree n_fragments can be "
              "checked and no July recording is mapped; write the table with july_fragment_counts.py")
        draw_lengths = {}
    cell_rows = classify(cells, july, port, rtol=args.rtol)
    scan = None
    if args.port_scan is not None:
        wanted = {(c["animal"], c["july_raw_sha256"]) for c in cell_rows
                  if c["class"] in LOOKED_UP and c["july_raw_sha256"]}
        scan = read_scan(args.port_scan, wanted, port=port)
        find_shifts(cell_rows, scan)
    map_rows = recording_map(cell_rows, july, port, draw_lengths, scan=scan)
    _write(args.out, CELL_FIELDS, cell_rows)
    _write(args.recording_map, MAP_FIELDS, map_rows)
    summarize(cell_rows, map_rows, scan=scan)
    print(f"\nwrote {args.out} and {args.recording_map}")
    ok = all(c["class"] in MAPPABLE for c in cell_rows) and all(r["status"] == "mapped" for r in map_rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
