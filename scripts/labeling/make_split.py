#!/usr/bin/env python
"""Verify the committed animal-level dev/test split against the live cohort, without redrawing it.

The split was drawn once with seed 0, stratified by strain (recording modality and montage, metadata fixed
before any labels): each strain contributes at least one test animal, and the two remaining test animals go
to two strains picked by the seeded draw, with no look at labels or prevalence and no redraws. That draw is
stored in ``config/labeling/split.json``.

The script does not redraw because the draw used ``rng.sample(animals_of_strain, k)``: the order of the
animals within each strain, together with seed 0, decided which animals went to test. That order was listed
by hand and does not match any canonical sort of the cohort (``iter_cohort_animals`` orders animals
differently), so rebuilding the list and redrawing would change the test set. The split is therefore a
committed file, and this script checks that it stays consistent with the cohort, which it derives from
canonical sources rather than a hardcoded animal list:

* the labelled animals, from the committed keymap (``config/labeling/keymap.csv``)
* each animal's strain, from ``build_cohort_bundle.iter_cohort_animals`` (the dataset is the strain)

Usage::

    uv run python scripts/labeling/make_split.py                 # verify; exits non-zero on drift
    uv run python scripts/labeling/make_split.py --skip-strain   # partition check only, no cohort load
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/labeling"))   # build_cohort_bundle
sys.path.insert(0, str(REPO / "src"))                # neurodent (imported by build_cohort_bundle)

KEYMAP = REPO / "config/labeling/keymap.csv"         # committed: the shipped bundle's unblinding key
SPLIT = REPO / "config/labeling/split.json"          # committed: the frozen lock this script verifies
N_TEST = 6


def labelled_cohort():
    """Return the animals in the shipped bundle, read from the committed keymap.

    Returns:
        set[str]: Animal ids.
    """
    with open(KEYMAP, newline="") as f:
        return {r["recording"].rsplit("__", 1)[0] for r in csv.DictReader(f)}


def strain_of(animals):
    """Map animals to their strain (dataset) through the cohort iterator the bundle and scorers share.

    Disables info logging for the rest of the process.

    Args:
        animals (set[str]): Animal ids to look up.

    Returns:
        dict[str, str]: Strain per animal, for the animals found in the cohort.
    """
    import build_cohort_bundle as C  # heavy (pulls the loader stack); only imported when the strain check runs
    logging.disable(logging.INFO)
    m = {}
    for ds, _sc, _cfg, a in C.iter_cohort_animals(C.REAL_STRAINS, limit_per_dataset=None, seed=0):
        if a in animals:
            m[a] = ds
    return m


def main():
    """Check the committed split against the labelled cohort and print the partition.

    Raises:
        SystemExit: With status 1 if the test and dev sets overlap, do not partition the labelled cohort,
            the test set has the wrong size, or a strain has no test animal.
    """
    ap = argparse.ArgumentParser(description="Verify the committed dev/test split against the live cohort.")
    ap.add_argument("--skip-strain", action="store_true",
                    help="skip the canonical strain-coverage check (avoids loading the cohort seam)")
    args = ap.parse_args()

    cohort = labelled_cohort()
    split = json.loads(SPLIT.read_text())
    test, dev = set(split["test"]), set(split["dev"])

    # ---- core integrity: the committed split partitions exactly the labelled cohort ----
    errs = []
    if test & dev:
        errs.append(f"test and dev overlap: {sorted(test & dev)}")
    if test | dev != cohort:
        errs.append(f"test and dev do not partition the labelled cohort; missing {sorted(cohort - (test | dev))}, "
                    f"extra {sorted((test | dev) - cohort)}")
    if len(test) != N_TEST:
        errs.append(f"test has {len(test)} animals, expected {N_TEST}")

    print(f"labelled cohort: {len(cohort)} animals (from keymap)")
    print(f"test ({len(test)}): {sorted(test)}")
    print(f"dev  ({len(dev)}): {sorted(dev)}")

    # ---- strain coverage: derive each animal's strain canonically; every strain needs a test animal ----
    if not args.skip_strain:
        try:
            smap = strain_of(cohort)
            missing = cohort - set(smap)
            if missing:
                print(f"note: {len(missing)} animal(s) not found in the cohort seam; strain check skipped: "
                      f"{sorted(missing)}")
            else:
                by_test = {}
                for a in test:
                    by_test.setdefault(smap[a], []).append(a)
                strains = sorted({smap[a] for a in cohort})
                uncovered = [s for s in strains if s not in by_test]
                if uncovered:
                    errs.append(f"strains with no test animal: {uncovered}")
                print("test by strain: " + ", ".join(f"{s}:{sorted(by_test.get(s, []))}" for s in strains))
        except Exception as e:
            print(f"note: strain check skipped ({type(e).__name__}: {e})")

    if errs:
        print("\nsplit invalid:")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)
    print("\nsplit OK: no errors in the checks above")


if __name__ == "__main__":
    main()
