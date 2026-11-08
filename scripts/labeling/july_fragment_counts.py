#!/usr/bin/env python
"""Recover each July recording's fragment count from the committed keymap and manifest.

July drew the labeled windows of each recording with ``select_random`` in ``render_context.py``: a fresh
``numpy.random.RandomState(seed)``, with the manifest seed for every recording, chose ``n_select`` distinct
windows from ``range(12, n_fragments - 12)``. ``build_cohort_bundle.render_animal`` set ``n_select`` by
dividing the manifest's ``n_per_animal`` windows over the animal's recordings with ``divmod``, the first
``n_per_animal % n_recordings`` recordings taking one extra window. The draw depends only on ``n_fragments``
and ``n_select``, so a recording's window set in ``keymap.csv`` constrains the ``n_fragments`` that produced
it. For each recording this scans ``n_fragments`` from ``max(window) + 13`` up to a bound and records the
smallest value that reproduces the window set, and whether it is the only one in that range.

No recording is read, only the committed keymap and manifest. ``hash_labeled_windows.py`` logs the
fragment counts of the recordings it loads next to these, so a loader run shows whether it reproduces the
July recording lengths at the July positions. ``compare_window_hashes.py`` leaves a July recording unmapped
when the July tree's count at its July position is not among them, since the July tree is then not loading
the recording July rendered.

Run from the repository root::

    python scripts/labeling/july_fragment_counts.py --out config/labeling/july_n_fragments.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger("july_fragment_counts")

REPO = Path(__file__).resolve().parents[2]
CONFIG_KEYMAP = REPO / "config/labeling/keymap.csv"
CONFIG_MANIFEST = REPO / "config/labeling/cohort_manifest.json"
CONFIG_OUT = REPO / "config/labeling/july_n_fragments.csv"

# render_context.py at commit 41353542: 5 s fragments, and the widest context drawn around a window (the
# 60 s overview strip) reserves ceil(60 / 5) = 12 fragments at each end of the recording.
FRAG_S = 5.0
CONTEXT_S = 15.0
OVERVIEW_S = 60.0
FLANK = int(np.ceil(max(CONTEXT_S, OVERVIEW_S) / FRAG_S))
MAX_FRAGMENTS = 200_000

FIELDS = [
    "recording", "animal", "position", "n_windows", "n_recordings", "divmod_size", "min_window",
    "max_window", "search_from", "search_to", "n_fragments", "n_matches", "unique", "matches",
]


def july_draw(n_fragments, n_select, seed):
    """Return the windows July's ``select_random`` drew for a recording of ``n_fragments`` fragments.

    A copy of ``render_context.select_random`` as of commit 41353542.

    Args:
        n_fragments (int): Number of 5 s fragments in the recording.
        n_select (int): Number of windows drawn.
        seed (int): The manifest seed.

    Returns:
        list[int]: The drawn window indices, sorted.

    Raises:
        ValueError: If the recording has fewer than ``n_select`` eligible windows, as July raised.
    """
    rng = np.random.RandomState(seed)
    pool = range(FLANK, n_fragments - FLANK)
    if len(pool) < n_select:
        raise ValueError(
            f"cannot draw {n_select} distinct windows: only {len(pool)} eligible "
            f"({n_fragments} windows, {FLANK} reserved as flank at each end)"
        )
    return sorted(int(w) for w in rng.choice(pool, size=n_select, replace=False))


def july_draw_sizes(n_per_animal, n_recordings):
    """Return the number of windows July drew from each of an animal's recordings.

    Args:
        n_per_animal (int): Windows per animal, from the manifest.
        n_recordings (int): Number of recordings the animal had.

    Returns:
        list[int]: Windows drawn at each position, as ``build_cohort_bundle.render_animal`` divided them.
    """
    base, rem = divmod(n_per_animal, n_recordings)
    return [base + (1 if i < rem else 0) for i in range(n_recordings)]


def infer_n_recordings(n_per_animal, sizes_by_position):
    """Return the smallest recording count whose ``divmod`` split fits the observed window counts.

    A recording July skipped (too short to draw from) has no keymap rows, so the count cannot be read off
    the highest position.

    Args:
        n_per_animal (int): Windows per animal, from the manifest.
        sizes_by_position (dict[int, int]): Number of keymap windows at each position of one animal.

    Returns:
        int | None: The smallest count that gives every observed position its observed number of
        windows, or None if none up to ``n_per_animal`` past the highest position does.
    """
    first = max(sizes_by_position) + 1
    for n in range(first, first + n_per_animal + 1):
        sizes = july_draw_sizes(n_per_animal, n)
        if all(sizes[i] == k for i, k in sizes_by_position.items()):
            return n
    return None


def split_label(recording):
    """Split a recording label ``"{animal}__{i}"`` into ``(animal, i)``."""
    animal, _, position = recording.rpartition("__")
    if not animal or not position.isdigit():
        raise ValueError(f"recording label {recording!r} is not of the form '<animal>__<position>'")
    return animal, int(position)


def _scan_range(task):
    """Find the target window sets that July's draw produces for each ``n_fragments`` in a range.

    Legacy ``RandomState.choice(pool, size, replace=False)`` returns ``pool[permutation(len(pool))[:size]]``,
    so one permutation per ``n_fragments`` gives the draw for every size at once. The caller checks each
    match again with :func:`july_draw`.

    Args:
        task (tuple): ``(lo, hi, seed, targets)``: scan ``range(lo, hi)``; ``targets`` maps a draw size to
            the set of sorted window tuples to look for.

    Returns:
        list[tuple[int, int, tuple[int, ...]]]: ``(n_fragments, size, windows)`` for each match.
    """
    lo, hi, seed, targets = task
    rng = np.random.RandomState(seed)
    state = rng.get_state()
    largest = max(targets)
    found = []
    for n in range(lo, hi):
        population = n - 2 * FLANK
        if population <= 0:
            continue
        rng.set_state(state)
        head = rng.permutation(population)[:largest] + FLANK
        for size, wanted in targets.items():
            if size > population:
                continue
            key = tuple(sorted(int(w) for w in head[:size]))
            if key in wanted:
                found.append((n, size, key))
    return found


def scan(targets, lo, hi, seed, jobs=1, chunk=1000):
    """Return every ``n_fragments`` in ``range(lo, hi)`` whose July draw is one of the target window sets.

    Args:
        targets (dict[int, set[tuple[int, ...]]]): Draw size to the sorted window tuples to look for.
        lo (int): First ``n_fragments`` scanned.
        hi (int): One past the last ``n_fragments`` scanned.
        seed (int): The manifest seed.
        jobs (int, optional): Worker processes; 1 scans in this process. Defaults to 1.
        chunk (int, optional): ``n_fragments`` values per task. Defaults to 1000.

    Returns:
        dict[tuple[int, tuple[int, ...]], list[int]]: ``(size, windows)`` to the matching ``n_fragments``.
    """
    tasks = [(a, min(a + chunk, hi), seed, targets) for a in range(lo, hi, chunk)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            parts = list(pool.map(_scan_range, tasks))
    else:
        parts = [_scan_range(t) for t in tasks]
    found = defaultdict(list)
    for part in parts:
        for n, size, key in part:
            found[(size, key)].append(n)
    return {k: sorted(v) for k, v in found.items()}


def read_keymap_windows(path):
    """Return each keymap recording's animal, position and sorted distinct windows, in keymap order.

    Args:
        path (str | Path): ``keymap.csv`` with columns recording, window and animal.

    Returns:
        dict[str, tuple[str, int, list[int]]]: Recording label to ``(animal, position, windows)``.

    Raises:
        ValueError: If a label's animal disagrees with the row's animal column.
    """
    windows, animals = {}, {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rec = row["recording"]
            animal, _ = split_label(rec)
            if animal != row["animal"]:
                raise ValueError(f"{path}: recording {rec!r} has animal column {row['animal']!r}")
            windows.setdefault(rec, set()).add(int(row["window"]))
            animals[rec] = animal
    return {rec: (animals[rec], split_label(rec)[1], sorted(ws)) for rec, ws in windows.items()}


def recover(recordings, seed, n_per_animal, max_fragments=MAX_FRAGMENTS, jobs=1):
    """Recover the July ``n_fragments`` of each recording from its window set.

    Args:
        recordings (dict[str, tuple[str, int, list[int]]]): Recording label to ``(animal, position,
            windows)``, as :func:`read_keymap_windows` returns.
        seed (int): The manifest seed.
        n_per_animal (int): Windows per animal, from the manifest.
        max_fragments (int, optional): Largest ``n_fragments`` scanned. Defaults to 200000.
        jobs (int, optional): Worker processes for the scan. Defaults to 1.

    Returns:
        list[dict]: One row per recording, with the columns in :data:`FIELDS`. ``n_fragments`` is the
        smallest match and is empty when nothing up to ``max_fragments`` reproduces the window set.

    Raises:
        RuntimeError: If a match from the scan is not reproduced by :func:`july_draw`.
    """
    sizes = defaultdict(dict)
    targets = defaultdict(set)
    for animal, position, windows in recordings.values():
        sizes[animal][position] = len(windows)
        targets[len(windows)].add(tuple(windows))
    n_recordings = {animal: infer_n_recordings(n_per_animal, s) for animal, s in sizes.items()}

    lo = min(max(ws) + FLANK + 1 for _, _, ws in recordings.values())
    log.info(f"scanning n_fragments {lo} to {max_fragments} for {len(recordings)} recordings, "
             f"draw sizes {sorted(targets)}, seed {seed}, {jobs} job(s)")
    found = scan(dict(targets), lo, max_fragments + 1, seed, jobs=jobs)

    rows = []
    for rec, (animal, position, windows) in recordings.items():
        matches = found.get((len(windows), tuple(windows)), [])
        for n in matches:
            if july_draw(n, len(windows), seed) != windows:
                raise RuntimeError(f"{rec}: the scan matched n_fragments={n}, but july_draw does not reproduce it")
        n_rec = n_recordings[animal]
        rows.append({
            "recording": rec,
            "animal": animal,
            "position": position,
            "n_windows": len(windows),
            "n_recordings": "" if n_rec is None else n_rec,
            "divmod_size": "" if n_rec is None else july_draw_sizes(n_per_animal, n_rec)[position],
            "min_window": windows[0],
            "max_window": windows[-1],
            "search_from": windows[-1] + FLANK + 1,
            "search_to": max_fragments,
            "n_fragments": matches[0] if matches else "",
            "n_matches": len(matches),
            "unique": len(matches) == 1,
            "matches": ";".join(str(n) for n in matches),
        })
    return rows


def main(argv=None):
    """Recover the July fragment counts and write them as CSV.

    Args:
        argv (list[str], optional): Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 when every recording has at least one match and fits the divmod split, 1 otherwise. Several
        matches are not an error: draws for neighbouring ``n_fragments`` are correlated, so a window set
        can fit a short run of them.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keymap", type=Path, default=CONFIG_KEYMAP, help="keymap.csv of the labeled windows")
    p.add_argument("--manifest", type=Path, default=CONFIG_MANIFEST,
                   help="cohort_manifest.json, for the seed and n_per_animal")
    p.add_argument("--out", type=Path, default=CONFIG_OUT, help="output CSV")
    p.add_argument("--max-fragments", type=int, default=MAX_FRAGMENTS,
                   help="largest n_fragments scanned (default %(default)s)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                   help="worker processes for the scan (default: all CPUs)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    manifest = json.loads(args.manifest.read_text())
    recordings = read_keymap_windows(args.keymap)
    rows = recover(recordings, int(manifest["seed"]), int(manifest["n_per_animal"]),
                   max_fragments=args.max_fragments, jobs=args.jobs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    misfit = [r["recording"] for r in rows if r["n_recordings"] == "" or r["divmod_size"] != r["n_windows"]]
    for rec in misfit:
        log.warning(f"{rec}: its keymap window count does not fit the divmod split of its animal")
    n_unique = sum(1 for r in rows if r["unique"])
    n_none = sum(1 for r in rows if r["n_matches"] == 0)
    log.info(f"{len(rows)} recordings: {n_unique} with exactly one n_fragments, "
             f"{len(rows) - n_unique - n_none} with several, {n_none} with none; wrote {args.out}")
    return 1 if (n_none or misfit) else 0


if __name__ == "__main__":
    sys.exit(main())
