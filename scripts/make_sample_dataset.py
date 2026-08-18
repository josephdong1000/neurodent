"""Generate the packaged sample dataset from the integration test fixtures.

Trims the ColMajor ``.bin`` recordings to a short excerpt and copies the metadata
and EDF files unchanged, writing the result to ``src/neurodent/data/sample/``.
The originals under ``.tests/`` are left untouched so the Snakemake integration
tests keep their full-length inputs.

Run from the repository root::

    python scripts/make_sample_dataset.py
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / ".tests" / "integration" / "data"
DEST = REPO_ROOT / "src" / "neurodent" / "data" / "sample"
DEFAULT_SECONDS = 60


def _read_meta(csv_path: Path) -> tuple[int, float]:
    """Return ``(n_channels, sampling_rate)`` from a Meta csv."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return len(rows), float(rows[0]["SampleRate"])


def trim_bin(src: Path, dst: Path, csv_path: Path, seconds: int) -> int:
    """Trim a column-major float32 recording to its first ``seconds``.

    Returns the number of samples written. The on-disk layout is column-major:
    every sample of channel 0, then channel 1, and so on. ``read_bin_csv_pair``
    derives the sample count from file size, so the layout must be preserved and
    the metadata needs no change.
    """
    n_channels, sampling_rate = _read_meta(csv_path)
    n_samples = src.stat().st_size // (np.dtype(np.float32).itemsize * n_channels)
    keep = min(n_samples, int(seconds * sampling_rate))

    data = np.memmap(
        src, dtype=np.float32, mode="r", shape=(n_samples, n_channels), order="F"
    )
    np.ascontiguousarray(data[:keep].T).tofile(dst)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=int,
        default=DEFAULT_SECONDS,
        help=f"excerpt length in seconds (default {DEFAULT_SECONDS})",
    )
    args = parser.parse_args()

    if not SOURCE.is_dir():
        raise SystemExit(f"Source fixtures not found: {SOURCE}")

    for animal_dir in sorted(p for p in SOURCE.iterdir() if p.is_dir()):
        out_dir = DEST / animal_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_paths = sorted(animal_dir.glob("*_Meta.csv"))
        if not csv_paths:
            raise SystemExit(f"No Meta csv in {animal_dir}")
        csv_path = csv_paths[0]

        for src in sorted(animal_dir.iterdir()):
            dst = out_dir / src.name
            if src.suffix == ".bin":
                kept = trim_bin(src, dst, csv_path, args.seconds)
                print(f"{src.name}: {kept} samples -> {dst.stat().st_size / 1e6:.2f} MB")
            else:
                shutil.copy2(src, dst)
                print(f"{src.name}: copied")


if __name__ == "__main__":
    main()
