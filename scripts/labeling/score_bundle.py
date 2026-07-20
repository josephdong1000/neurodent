#!/usr/bin/env python
"""Feed rater-exported CSVs back through the scoring pipeline.

The rater labels windows in the blinded HTML bundle and clicks **Export**, which downloads
``labels_<bundle>_<rater>.csv`` to their machine. Copy those CSVs onto the cluster, then::

    python scripts/labeling/score_bundle.py \
        --keymap results/labeling/mixed/_unblind/keymap.csv \
        joseph=~/labels_mixed_joseph.csv [alice=... bob=...]

Each positional argument is ``rater_id=path``. The script runs the exact library round-trip —
:func:`ingest` (melt ``label_*`` slot columns to long form) -> :func:`unblind` (neutral slot ->
true channel via the experimenter-side keymap, per ``(recording, window, slot)``) -> :func:`consensus`
+ :func:`interrater` — and prints a health report.

Purpose in the DOGFOOD phase: prove the end-to-end round-trip before the full package ships. A green
run means every labelled cell parsed, mapped back to a real anatomical channel with **no** unmapped
slot (unblinding is intact), and consensus/agreement compute. It exits non-zero if anything a rater
labelled fails to unblind, so a broken export or stale keymap can't slip through silently.

With a single rater (dogfood) the "consensus" is just that rater's calls and kappa is undefined; the
value is the parse + unblind verification. With >=2 raters it also reports Cohen/Fleiss kappa — the
human agreement ceiling any detector is measured against.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from neurodent.results.scoring import consensus, ingest, interrater, unblind  # noqa: E402


def _default_keymap(rater_csvs):
    """Best-effort: the keymap sits at ``<bundle_parent>/_unblind/keymap.csv``. Given a rater CSV in,
    or near, a bundle tree, walk up looking for it so ``--keymap`` can usually be omitted."""
    for p in rater_csvs:
        for anc in Path(p).resolve().parents:
            cand = anc / "_unblind" / "keymap.csv"
            if cand.exists():
                return cand
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raters", nargs="+", metavar="rater=csv",
                    help="one or more rater_id=path_to_exported_csv")
    ap.add_argument("--keymap", type=Path, default=None,
                    help="path to _unblind/keymap.csv (default: auto-discover next to a rater CSV)")
    ap.add_argument("--rule", default="majority", choices=["majority", "unanimous", "any"],
                    help="consensus rule (default: majority)")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional: write the unblinded long-form labels to this CSV")
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
        ap.error("could not find keymap.csv; pass it with --keymap "
                 "(it lives at <bundle>/_unblind/keymap.csv, OUTSIDE the shipped zip)")
    keymap = pd.read_csv(keymap_path)

    print(f"keymap : {keymap_path}  ({len(keymap)} rows, "
          f"{keymap['recording'].nunique()} recordings)")
    for rater, p in manifests.items():
        print(f"rater  : {rater:12s} <- {p}")

    # ---- ingest: melt label_<slot> columns to one row per (recording, window, slot, rater) ----
    long_df = ingest(manifests)                       # raises on any unrecognised label token
    labelled = long_df["y"].notna().sum()
    print(f"\ningest : {len(long_df)} cells, {labelled} labelled "
          f"({len(long_df) - labelled} blank/unsure)")

    # ---- unblind: neutral slot -> true channel; raises if a LABELLED cell has no keymap entry ----
    restored = unblind(long_df, keymap)               # this is the round-trip proof
    leaked = restored["channel"].astype(str).str.startswith("Ch ").sum()
    assert leaked == 0, f"{leaked} cells still on a neutral slot after unblind (keymap gap)"
    print(f"unblind: {len(restored)} cells mapped to real channels, 0 unmapped labelled cells  OK")

    # per-recording channel recovery + category mix, as a sanity surface
    by_rec = (restored.dropna(subset=["y"])
              .groupby("recording")["channel"].nunique().sort_index())
    print("\nchannels recovered per recording (labelled cells):")
    for rec, n in by_rec.items():
        print(f"  {rec:28s} {n} channels")
    print("\nlabel categories:", restored["category"].value_counts().to_dict())

    # ---- consensus + interrater ----
    cons = consensus(restored, rule=args.rule)
    n_reject = int(cons["y_true"].sum())
    print(f"\nconsensus ({args.rule}): {len(cons)} (recording,window,channel) cells, "
          f"{n_reject} REJECT / {len(cons) - n_reject} KEEP, "
          f"{int(cons['any_event'].sum())} tagged real-event")
    kap = interrater(restored)
    if kap["metric"] is None:
        print(f"interrater: n/a (single rater; agreement needs >=2) over {kap['n_cells']} shared cells")
    else:
        print(f"interrater: {kap['metric']} kappa = {kap['kappa']:.3f} "
              f"over {kap['n_cells']} cells x {kap['n_raters']} raters")

    if args.out:
        restored.to_csv(args.out, index=False)
        print(f"\nwrote unblinded long-form labels -> {args.out}")

    print("\nround-trip OK — export -> ingest -> unblind -> consensus verified end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
