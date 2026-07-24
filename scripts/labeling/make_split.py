#!/usr/bin/env python
"""Verify the committed animal-level train(dev)/test split against the live cohort — it does NOT re-draw.

The split was LOCKED once by a seed-0 draw stratified by STRAIN (recording modality/montage — metadata
fixed before any labels): each strain contributes >=1 test animal, the two extra go to two seeded-random
strains; NO label/prevalence peeking, NO re-rolls. That draw is frozen in ``config/labeling/split.json``.

**Why this script no longer draws.** The original draw did ``rng.sample(animals_of_strain, k)``, so the
*within-strain ordering* of the animal list was load-bearing — it, together with seed 0, fixed exactly which
animals are sealed. That order was hand-listed and is NOT recoverable from any canonical sort of the cohort
(the canonical ``iter_cohort_animals`` enumeration orders animals differently), so re-deriving the list and
re-drawing would silently change the test set. The lock is therefore a committed artifact; this script only
checks it stays consistent, deriving the cohort from canonical sources (no hardcoded animal list):

  * the labelled animals   <- the committed keymap (``config/labeling/keymap.csv``)
  * each animal's strain    <- ``build_cohort_bundle.iter_cohort_animals`` (dataset == strain)

    uv run python scripts/labeling/make_split.py            # verify; exits non-zero on drift
    uv run python scripts/labeling/make_split.py --skip-strain   # fast: partition check only (no cohort load)
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
    """The animals actually in the shipped bundle -> the committed keymap (canonical; no hardcoding)."""
    with open(KEYMAP, newline="") as f:
        return {r["recording"].rsplit("__", 1)[0] for r in csv.DictReader(f)}


def strain_of(animals):
    """animal -> strain (dataset) via the blessed cohort seam the bundle/scorer share (no drift)."""
    import build_cohort_bundle as C  # heavy (pulls the loader stack); only imported when the strain check runs
    logging.disable(logging.INFO)
    m = {}
    for ds, _sc, _cfg, a in C.iter_cohort_animals(C.REAL_STRAINS, limit_per_dataset=None, seed=0):
        if a in animals:
            m[a] = ds
    return m


def main():
    ap = argparse.ArgumentParser(description="Verify the committed dev/test split against the live cohort.")
    ap.add_argument("--skip-strain", action="store_true",
                    help="skip the canonical strain-coverage check (avoids loading the cohort seam)")
    args = ap.parse_args()

    cohort = labelled_cohort()
    split = json.loads(SPLIT.read_text())
    test, dev = set(split["test"]), set(split["dev"])

    # ---- core integrity: the committed lock partitions EXACTLY the labelled cohort ----
    errs = []
    if test & dev:
        errs.append(f"test/dev overlap: {sorted(test & dev)}")
    if test | dev != cohort:
        errs.append(f"partition != labelled cohort; missing={sorted(cohort - (test | dev))} "
                    f"extra={sorted((test | dev) - cohort)}")
    if len(test) != N_TEST:
        errs.append(f"|test|={len(test)} != {N_TEST}")

    print(f"labelled cohort: {len(cohort)} animals (from keymap)")
    print(f"TEST ({len(test)}): {sorted(test)}")
    print(f"DEV  ({len(dev)}): {sorted(dev)}")

    # ---- strain coverage: derive each animal's strain canonically, check every strain has >=1 test ----
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
        print("\nLOCK INVALID:")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)
    print("\nlock OK: committed split is a valid strain-covering partition of the labelled cohort ✓")


if __name__ == "__main__":
    main()
