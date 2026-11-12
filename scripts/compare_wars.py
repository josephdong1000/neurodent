#!/usr/bin/env python3
"""Compare two WAR folders: file hashes first, then the parquet column by column, the json key by key.

Usage:

    python scripts/compare_wars.py A_DIR B_DIR [--key animalday timestamp] [--show 5]

For every file in either folder it prints the size and sha256 on each side. When war.parquet
differs, both tables are read and compared: the columns each has, the rows each has, matched on
the key columns (animalday and timestamp by default) and counted per animalday, then every
common column on the common rows. Values must be exactly equal; NaN equals NaN. When war.json
differs, the keys whose values differ are listed. A large WAR takes several GB to compare, so run
it in a cluster job.

Exit code: 0 when every file is identical, 1 when anything differs, 2 when a folder is missing.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path, block=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def equal(x, y):
    """Exact equality of two stored values: scalars, arrays, lists, tuples or dicts, NaN equal to NaN."""
    if isinstance(x, dict) or isinstance(y, dict):
        return isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys() and all(
            equal(x[k], y[k]) for k in x
        )
    if isinstance(x, (list, tuple, np.ndarray)) or isinstance(y, (list, tuple, np.ndarray)):
        try:
            a, b = np.asarray(x), np.asarray(y)
        except ValueError:  # ragged
            return len(x) == len(y) and all(equal(p, q) for p, q in zip(x, y))
        if a.dtype == object or b.dtype == object:
            return a.shape == b.shape and all(equal(p, q) for p, q in zip(a.ravel(), b.ravel()))
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    try:
        same = x == y
        return bool(same) if not isinstance(same, np.ndarray) else bool(same.all())
    except (TypeError, ValueError):
        return False


def max_abs_difference(x, y):
    try:
        a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if a.shape == b.shape:
            return float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    except (TypeError, ValueError):
        pass
    return None


def json_differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in A" for k in a.keys() - b.keys()]
        out += [f"{path}/{k}: only in B" for k in b.keys() - a.keys()]
        for k in sorted(a.keys() & b.keys(), key=str):
            out += json_differences(a[k], b[k], f"{path}/{k}")
        return out
    return [] if equal(a, b) else [f"{path}: {json.dumps(a, default=str)[:120]} != {json.dumps(b, default=str)[:120]}"]


def compare_parquet(path_a, path_b, key, show):
    import pyarrow.parquet as pq

    a, b = pq.read_table(path_a).to_pandas(), pq.read_table(path_b).to_pandas()
    differs = False
    only_a, only_b = sorted(set(a.columns) - set(b.columns)), sorted(set(b.columns) - set(a.columns))
    if only_a or only_b:
        differs = True
        print(f"  columns only in A: {only_a}\n  columns only in B: {only_b}")
    print(f"  rows: A {len(a)}, B {len(b)}")
    key = [k for k in key if k in a.columns and k in b.columns]
    if key:
        ka, kb = a.set_index(key), b.set_index(key)
        if not ka.index.is_unique or not kb.index.is_unique:
            print(f"  key {key} repeats within a table, so rows are compared by position")
            key = []
    if key:
        in_a, in_b = set(ka.index), set(kb.index)
        for name, extra in (("A", in_a - in_b), ("B", in_b - in_a)):
            if extra:
                differs = True
                by_day = Counter(k[0] if isinstance(k, tuple) else k for k in extra)
                print(f"  rows only in {name}: {len(extra)}, by {key[0]}: {dict(sorted(by_day.items(), key=str))}")
        common = sorted(in_a & in_b, key=lambda k: tuple(map(str, k)) if isinstance(k, tuple) else str(k))
        ka, kb = ka.loc[common], kb.loc[common]
    else:
        n = min(len(a), len(b))
        ka, kb = a.iloc[:n], b.iloc[:n]
    for column in [c for c in a.columns if c in b.columns and c not in key]:
        xs, ys = ka[column].tolist(), kb[column].tolist()
        bad = [i for i, (x, y) in enumerate(zip(xs, ys)) if not equal(x, y)]
        if not bad:
            continue
        differs = True
        diffs = [d for d in (max_abs_difference(xs[i], ys[i]) for i in bad) if d is not None]
        largest = f", largest absolute difference {max(diffs):.6g}" if diffs else ""
        print(f"  column {column}: {len(bad)} of {len(xs)} rows differ{largest}")
        for i in bad[:show]:
            where = ka.index[i] if key else i
            print(f"    row {where}: A {str(xs[i])[:100]} | B {str(ys[i])[:100]}")
    return differs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--key", nargs="*", default=["animalday", "timestamp"], help="columns that identify a row")
    ap.add_argument("--show", type=int, default=5, help="differing rows or keys to print per column")
    args = ap.parse_args(argv)
    for folder in (args.a, args.b):
        if not folder.is_dir():
            print(f"not a folder: {folder}", file=sys.stderr)
            return 2

    names = sorted({p.relative_to(args.a) for p in args.a.rglob("*") if p.is_file()}
                   | {p.relative_to(args.b) for p in args.b.rglob("*") if p.is_file()}, key=str)
    differs = False
    for name in names:
        pa, pb = args.a / name, args.b / name
        if not pa.exists() or not pb.exists():
            differs = True
            print(f"{name}: only in {'B' if not pa.exists() else 'A'}")
            continue
        ha, hb = sha256(pa), sha256(pb)
        if ha == hb:
            print(f"{name}: identical ({pa.stat().st_size} bytes, sha256 {ha[:16]})")
            continue
        differs = True
        print(f"{name}: differs ({pa.stat().st_size} and {pb.stat().st_size} bytes)")
        if name.suffix == ".parquet":
            if not compare_parquet(pa, pb, args.key, args.show):
                print("  the values are all equal; the files differ only in how they are written")
        elif name.suffix == ".json":
            found = json_differences(json.loads(pa.read_text()), json.loads(pb.read_text()))
            print(f"  {len(found)} differing keys" + "".join(f"\n    {d}" for d in found[: args.show * 4]))
    print("identical" if not differs else "different")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
