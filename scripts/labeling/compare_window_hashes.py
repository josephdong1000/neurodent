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
recordings through this map, so they leave those recordings out.

Run from the repository root after both hash runs::

    python scripts/labeling/compare_window_hashes.py --july hashes_july.csv --port hashes_port.csv \\
        --out window_hash_cells.csv

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

CLASSES = ["identical", "remapped", "channel_moved", "numeric_only", "different", "missing", "july_missing"]
MAPPABLE = {"identical", "remapped"}
N_HEAD = 4      # hash_labeled_windows.N_HEAD

CELL_FIELDS = [
    "july_recording", "animal", "july_position", "window", "channel", "class", "port_position", "port_channel",
    "hash_positions", "july_raw_sha256", "port_raw_sha256", "july_view_sha256", "port_view_sha256",
    "view_match", "july_rms", "port_rms",
]
MAP_FIELDS = [
    "july_recording", "animal", "animalday", "date_string", "port_position", "july_position", "status",
    "reason", "n_cells", "n_identical", "n_remapped", "july_n_fragments", "port_n_fragments",
    "recovered_n_fragments", "length_check",
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
        ValueError: If two rows share a key.
    """
    index = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
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


def recording_map(cell_rows, july, port, draw_lengths):
    """Map each July recording to the port recording that holds all of its labeled samples.

    A July recording is mapped when every one of its cells is identical or remapped, one port position
    holds the July raw sha256 of every cell, and its fragment count in the July output is among its July
    draw lengths (:func:`check_length`). When several positions hold every cell, the July position is taken
    if it is one of them; otherwise the recording is left unmapped as ambiguous. The map is one to one: when
    several July recordings land on one port position, only the one whose July position it is keeps it,
    since ``score_detectors.RecordingMap`` would otherwise keep whichever row it read last. An unmapped row
    leaves animalday and port_position empty, so ``RecordingMap`` resolves nothing to it.

    Args:
        cell_rows (list[dict]): Output of :func:`classify`.
        july (dict): The July hash rows, from :func:`read_hashes`.
        port (dict): The port hash rows, from :func:`read_hashes`.
        draw_lengths (dict[str, list[int]]): July recording label to the fragment counts that reproduce its
            window draw, from :func:`read_draw_lengths`. A recording with none cannot be checked and is left
            unmapped.

    Returns:
        list[dict]: One row per July recording with the columns in :data:`MAP_FIELDS`. The reason column
        lists every reason a recording is unmapped, separated by "; ".
    """
    july_recs, port_recs = recordings_of(july), recordings_of(port)
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
                   recovered_n_fragments=";".join(str(n) for n in lengths), length_check=check)
        other = {k: v for k, v in counts.items() if k not in MAPPABLE}
        reasons, q = [], None
        if other:
            reasons.append("cells neither identical nor remapped: " + ", ".join(
                f"{k} {other[k]}" for k in CLASSES if k in other))
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
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(cell_rows, map_rows):
    """Print counts per class, per animal, and the recording map."""
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

    checks = Counter(r["length_check"] for r in map_rows)
    print(f"\nJuly-tree n_fragments among the July draw lengths: {checks.get('ok', 0)} of {len(map_rows)} July "
          f"recordings; not among them: {checks.get('mismatch', 0)}; not checked: {checks.get('unchecked', 0)}")
    mapped = [r for r in map_rows if r["status"] == "mapped"]
    print(f"{len(mapped)} of {len(map_rows)} July recordings mapped")
    for r in map_rows:
        target = f"port position {r['port_position']} ({r['animalday']})" if r["status"] == "mapped" else r["reason"]
        print(f"  {r['july_recording']}: {r['status']}, {target}; n_fragments July {r['july_n_fragments'] or '-'}, "
              f"port {r['port_n_fragments'] or '-'}, July draw {r['recovered_n_fragments'] or '-'}")


def main(argv=None):
    """Classify every labeled cell, write the cell table and the recording map, and print a summary.

    Args:
        argv (list[str], optional): Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 when every cell is identical or remapped and every July recording is mapped, 1 otherwise.
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
                   help="recording map output CSV (default %(default)s)")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance on RMS and first samples for numeric_only (default %(default)s)")
    args = p.parse_args(argv)

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
    map_rows = recording_map(cell_rows, july, port, draw_lengths)
    _write(args.out, CELL_FIELDS, cell_rows)
    _write(args.recording_map, MAP_FIELDS, map_rows)
    summarize(cell_rows, map_rows)
    print(f"\nwrote {args.out} and {args.recording_map}")
    ok = all(c["class"] in MAPPABLE for c in cell_rows) and all(r["status"] == "mapped" for r in map_rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
