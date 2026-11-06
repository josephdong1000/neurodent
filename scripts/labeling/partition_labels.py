#!/usr/bin/env python
"""Split rater label CSVs into ``<name>_dev.csv`` and ``<name>_test.csv`` by animal, per ``split.json``.

Development and selection work loads only the ``_dev`` files, so a test animal cannot be scored by
accident. Run it on each new rater's CSV as it arrives; the animal lists in ``split.json`` do not change.

Usage::

    uv run python scripts/labeling/partition_labels.py scripts/labeling/labels/labels_cohort4strains_JD.csv
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLIT = json.loads((REPO / "config/labeling/split.json").read_text())
TEST = set(SPLIT["test"])


def animal_of(recording):
    """Return the animal id of a ``"{animal}__{i}"`` recording name."""
    return recording.rsplit("__", 1)[0]


paths = sys.argv[1:]
if not paths:
    sys.exit("usage: partition_labels.py <rater_csv> [<rater_csv> ...]")

for path in paths:
    p = Path(path)
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    unknown = {animal_of(r["recording"]) for r in rows} - set(SPLIT["test"]) - set(SPLIT["dev"])
    if unknown:
        sys.exit(f"{p.name}: rows for animals not in split.json: {sorted(unknown)}")
    parts = {"dev": [r for r in rows if animal_of(r["recording"]) not in TEST],
             "test": [r for r in rows if animal_of(r["recording"]) in TEST]}
    assert len(parts["dev"]) + len(parts["test"]) == len(rows)
    print(f"{p.name}: {len(rows)} rows")
    for suffix, rr in parts.items():
        out = p.with_name(p.stem + f"_{suffix}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rr)
        n_animals = len({animal_of(r["recording"]) for r in rr})
        print(f"  wrote {out.name}: {len(rr)} rows, {n_animals} animals")
