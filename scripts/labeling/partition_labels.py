#!/usr/bin/env python
"""Physically partition rater label CSV(s) into <name>_dev.csv / <name>_test.csv by animal, per split.json.

Structural leak prevention: dev/selection work only ever loads the _dev file, so a test animal cannot be
scored by accident. Re-run on each new rater's CSV as it arrives (the split.json animal list never changes).

    uv run python scripts/labeling/partition_labels.py scripts/labeling/labels/labels_cohort4strains_JD*.csv
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLIT = json.loads((REPO / "config/labeling/split.json").read_text())
TEST = set(SPLIT["test"])


def animal_of(recording):
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
        print(f"  -> {out.name}: {len(rr)} rows, {n_animals} animals")
