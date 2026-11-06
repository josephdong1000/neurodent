#!/usr/bin/env python
"""Feed rater-exported CSVs back through the scoring pipeline.

The rater labels windows in the blinded HTML bundle and clicks Export CSV, which downloads
``labels_<bundle>_<rater>.csv`` to their machine. Copy those CSVs onto the cluster, then::

    python scripts/labeling/score_bundle.py joseph=~/labels_mixed_joseph.csv [alice=... bob=...]

Each positional argument is ``rater_id=path``. Without ``--keymap``, the keymap is the nearest
``_unblind/keymap.csv`` above a rater CSV, else the committed ``config/labeling/keymap.csv``. Later builds into
the same output root write ``_unblind-2/``, ``_unblind-3/`` and so on, which are not searched; pass ``--keymap``
for those.

The report prints label statistics, so labels of animals in ``config/labeling/split.json``'s test list are
refused unless ``--include-test`` is given; the test split is read against detectors only by
``score_test.py``.

The script runs the library round trip in order: :func:`ingest` melts the ``label_*`` slot columns to long
form, :func:`unblind` maps each neutral slot to its true channel through the experimenter-side keymap, per
``(recording, window, slot)``, and :func:`consensus` and :func:`interrater` summarize the result. It prints a
report of each step.

A clean run means every labelled cell parsed, every labelled slot mapped back to a real anatomical
channel, and consensus and agreement computed. The script exits non-zero if anything a rater labelled
fails to unblind, so a broken export or a stale keymap cannot pass unnoticed.

With a single rater the consensus is that rater's labels and kappa is undefined, so the run checks parsing
and unblinding only. With two or more raters it also reports Cohen's or Fleiss' kappa, the human agreement
ceiling that detectors are measured against.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from neurodent.results.scoring import consensus, ingest, interrater, unblind  # noqa: E402

CONFIG_KEYMAP = REPO / "config/labeling/keymap.csv"  # committed and permanent; results/ is disposable


def _default_keymap(rater_csvs):
    """Find the unblinding keymap for a set of rater CSVs.

    Walks up from each rater CSV to the nearest ``_unblind/keymap.csv``, then falls back to the committed
    ``config/labeling/keymap.csv``, so ``--keymap`` can usually be omitted and scoring still works after
    ``results/`` is cleared. Keys of later builds in the same output root (``_unblind-2/`` and so on) are not
    searched.

    Args:
        rater_csvs (Iterable[Path]): Rater CSV paths.

    Returns:
        Path | None: The keymap path, or None if none is found.
    """
    for p in rater_csvs:
        for anc in Path(p).resolve().parents:
            cand = anc / "_unblind" / "keymap.csv"
            if cand.exists():
                return cand
    return CONFIG_KEYMAP if CONFIG_KEYMAP.exists() else None


def main(argv=None):
    """Parse arguments, run the round trip, and print the report.

    Args:
        argv (list[str], optional): Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 once the round trip completes.

    Raises:
        ValueError: If a rater CSV cannot be ingested, unblinded or combined: an unrecognized label token, a
            CSV without label columns, a labelled cell with no keymap entry, or a cell labelled twice by one
            rater (see ``ingest``, ``unblind`` and ``consensus``).
        AssertionError: If a cell is still on a neutral slot after unblinding.
        SystemExit: On a bad argument, a missing rater CSV or keymap, or test-split labels without
            ``--include-test``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raters", nargs="+", metavar="rater=csv",
                    help="one or more rater_id=path_to_exported_csv")
    ap.add_argument("--keymap", type=Path, default=None,
                    help="unblinding key from slots to channels (default: _unblind/keymap.csv found above a "
                         "rater CSV, else the committed config/labeling/keymap.csv)")
    ap.add_argument("--rule", default="majority", choices=["majority", "unanimous", "any"],
                    help="consensus rule (default: majority)")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional: write the unblinded long-form labels to this CSV")
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json",
                    help="locked split whose test animals are refused (default: config/labeling/split.json)")
    ap.add_argument("--include-test", action="store_true",
                    help="report on labels of test-split animals too, exposing their statistics")
    args = ap.parse_args(argv)

    manifests = {}
    for spec in args.raters:
        if "=" not in spec:
            ap.error(f"expected rater=csv, got {spec!r}")
        rater, path = spec.split("=", 1)
        p = Path(path).expanduser()
        if not p.exists():
            ap.error(f"{rater}: no such file {p}")
        manifests[rater] = p

    keymap_path = args.keymap or _default_keymap(manifests.values())
    if keymap_path is None or not Path(keymap_path).exists():
        ap.error("could not find keymap.csv; pass it with --keymap (the first build into an output root writes "
                 "_unblind/keymap.csv beside the bundle directory, later ones _unblind-2/ and so on, outside the "
                 "shipped zip; the shipped key is committed at config/labeling/keymap.csv)")
    keymap = pd.read_csv(keymap_path)

    print(f"keymap : {keymap_path}  ({len(keymap)} rows, "
          f"{keymap['recording'].nunique()} recordings)")
    for rater, p in manifests.items():
        print(f"rater  : {rater:12s} {p}")

    # ---- ingest: melt label_<slot> columns to one row per (recording, window, slot, rater) ----
    long_df = ingest(manifests)                       # raises on any unrecognised label token
    if args.split.exists() and not args.include_test:
        test_animals = set(json.loads(args.split.read_text())["test"])
        in_test = sorted({r.rsplit("__", 1)[0] for r in long_df["recording"]} & test_animals)
        if in_test:
            ap.error(f"labels include test-split animals {in_test}; pass the _dev files, or --include-test to "
                     "report on them knowingly")
    labelled = long_df["y"].notna().sum()
    print(f"\ningest : {len(long_df)} cells, {labelled} labelled "
          f"({len(long_df) - labelled} blank/unsure)")

    # ---- unblind: map each neutral slot to its true channel; raises if a labelled cell has no entry ----
    restored = unblind(long_df, keymap)               # this is the round-trip check
    leaked = restored["channel"].astype(str).str.startswith("Ch ").sum()
    assert leaked == 0, f"{leaked} cells still on a neutral slot after unblind (keymap gap)"
    print(f"unblind: {len(restored)} cells mapped to real channels, 0 unmapped labelled cells")

    # channels recovered per recording and the category mix, as a sanity check
    by_rec = (restored.dropna(subset=["y"])
              .groupby("recording")["channel"].nunique().sort_index())
    print("\nchannels recovered per recording (labelled cells):")
    for rec, n in by_rec.items():
        print(f"  {rec:28s} {n} channels")
    print("\nlabel categories:", restored["category"].value_counts().to_dict())

    # ---- consensus and interrater agreement ----
    cons = consensus(restored, rule=args.rule)
    n_reject = int(cons["y_true"].sum())
    print(f"\nconsensus ({args.rule}): {len(cons)} (recording, window, channel) cells, "
          f"{n_reject} reject, {len(cons) - n_reject} keep, "
          f"{int(cons['any_event'].sum())} tagged as real events")
    kap = interrater(restored)
    if kap["metric"] is None:
        print(f"interrater: n/a (needs two or more raters labelling shared cells) over {kap['n_cells']} shared cells")
    else:
        print(f"interrater: {kap['metric']} kappa = {kap['kappa']:.3f} "
              f"over {kap['n_cells']} cells x {kap['n_raters']} raters")

    if args.out:
        restored.to_csv(args.out, index=False)
        print(f"\nwrote unblinded long-form labels to {args.out}")

    print("\nround trip complete: export, ingest, unblind and consensus checked end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
