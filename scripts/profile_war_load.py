#!/usr/bin/env python3
"""Profile peak memory of the standardize cycle: load -> reorder/pad -> save.

This mirrors workflow/scripts/standardize_wars.py for one animal, but as a
standalone script that can be wrapped by memray on the HPC.

Pass ``--convert-first`` to convert the legacy JSON-encoded WAR to the new
native format (encoding_version=2) in a subprocess BEFORE the measurement,
so the profiled load goes through the new path (not the legacy JSON path).
Without this flag, the measurement reflects whatever format the on-disk WAR
already has — useful for measuring legacy WARs, but won't show #4's load
side gains for WARs saved by the old code.

Usage (HPC, wrapped by memray):
    sbatch --job-name=profile_war_load --cpus-per-task=2 --mem=80G --time=30:00 \\
      --output=logs/profile_war_load.out --error=logs/profile_war_load.err \\
      --wrap="cd /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Joseph_devtree && \\
        mkdir -p analysis/profiling && \\
        uv run memray run -o analysis/profiling/load_arxrosa-1015_v2.bin \\
          scripts/profile_war_load.py arxrosa-1015 --convert-first"

Inspect afterward (cheap, local or HPC):
    uv run memray flamegraph analysis/profiling/load_arxrosa-1015_v2.bin
    uv run memray stats     analysis/profiling/load_arxrosa-1015_v2.bin
"""
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

from neurodent import visualization
from neurodent.workflow.utils import apply_samples_config, load_samples_config, resolve_samples_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAR_ROOT = REPO_ROOT / "results" / "wars_quality_filtered"
# arx_rosa is a single-file dataset (inline samples_data); resolve it from the dataset config.
SAMPLES_PATH = REPO_ROOT / "config" / "datasets" / "arx_rosa.yaml"
# Matches config/config.yaml standardization.channel_reorder (8 EEG channels).
CHANNEL_REORDER = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"]


def _rss_gb() -> float:
    """Return current process RSS in GB."""
    return psutil.Process().memory_info().rss / 1024**3


def _rss_peak_gb() -> float:
    """Peak RSS since process start (Linux); falls back to current RSS elsewhere."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    except Exception:
        return _rss_gb()


def _report(label: str, t0: float) -> None:
    print(
        f"[{time.monotonic() - t0:6.1f}s] "
        f"RSS={_rss_gb():6.2f}G  peak={_rss_peak_gb():6.2f}G  | {label}",
        flush=True,
    )


def _convert_to_native_subprocess(animal: str, war_root: Path, output_dir: Path) -> None:
    """Convert a legacy JSON-encoded WAR to native format in an isolated subprocess.

    Runs in a child process so its memory footprint (which will spike to
    load the legacy WAR) does NOT show up in the parent process's memray
    profile. The parent then loads the resulting native-format WAR cleanly.
    """
    script = f"""
from pathlib import Path
from neurodent import visualization
from neurodent.workflow.utils import apply_samples_config, load_samples_config, resolve_samples_config

samples_config = resolve_samples_config(load_samples_config({str(SAMPLES_PATH)!r}))
apply_samples_config(samples_config)

src = Path({str(war_root)!r}) / {animal!r}
dst = Path({str(output_dir)!r})
dst.mkdir(parents=True, exist_ok=True)

war = visualization.WindowAnalysisResult.load_parquet_and_json(
    folder_path=src, parquet_name="war.parquet", json_name="war.json"
)
war.save_parquet_and_json(dst)
print(f"[convert] {{dst}}/war.parquet written ({{(dst / 'war.parquet').stat().st_size / 1024**3:.2f}} G)", flush=True)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "animal",
        nargs="?",
        default="arxrosa-1015",
        help="animal id (folder under results/wars_quality_filtered/). Default: arxrosa-1015",
    )
    ap.add_argument(
        "--war-root",
        type=Path,
        default=DEFAULT_WAR_ROOT,
        help=f"Root containing <animal>/war.parquet. Default: {DEFAULT_WAR_ROOT}",
    )
    ap.add_argument(
        "--samples",
        type=Path,
        default=SAMPLES_PATH,
        help=f"dataset/samples config (for channel aliases; inline samples_data or samples_file). Default: {SAMPLES_PATH}",
    )
    ap.add_argument(
        "--convert-first",
        action="store_true",
        help="Convert legacy WAR to native (encoding_version=2) in a subprocess "
        "before measuring, so the measured load goes through the new native path.",
    )
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="Use the streaming reorder_and_pad path (#3 — bounded peak memory) "
        "instead of the eager load→mutate→save path.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for streaming path (rows per pandas batch). Default: 5000.",
    )
    args = ap.parse_args()

    input_war_dir = args.war_root / args.animal
    if not input_war_dir.exists():
        sys.exit(f"WAR folder not found: {input_war_dir}")

    samples_config = resolve_samples_config(load_samples_config(args.samples))
    apply_samples_config(samples_config)

    # If requested, pre-convert legacy WAR to native format in a subprocess so
    # the parent process (which memray traces) sees a fresh, low-memory start.
    if args.convert_first:
        native_root = Path(tempfile.mkdtemp(prefix="profile_native_war_"))
        native_dir = native_root / args.animal
        print(f"[convert] converting {input_war_dir} -> {native_dir} (subprocess)", flush=True)
        _convert_to_native_subprocess(args.animal, args.war_root, native_dir)
        input_war_dir = native_dir
        print(f"[convert] done — measuring load against native WAR", flush=True)

    t0 = time.monotonic()
    _report("script start", t0)

    parquet_size_gb = (input_war_dir / "war.parquet").stat().st_size / 1024**3
    print(f"           input war.parquet on disk: {parquet_size_gb:.2f} G", flush=True)

    if args.streaming:
        # Streaming path: scan_parquet_and_json → reorder → save chain, never materialises
        # the full WAR DataFrame.  This mirrors what standardize_wars.py does.
        with tempfile.TemporaryDirectory(prefix="profile_war_") as tmpdir:
            _report(f"about to stream reorder_and_pad (batch_size={args.batch_size})", t0)
            war = visualization.WindowAnalysisResult.scan_parquet_and_json(input_war_dir, filename="war")
            war.reorder_and_pad_channels(CHANNEL_REORDER, use_abbrevs=True)
            war.save_parquet_and_json(Path(tmpdir), filename="war", batch_size=args.batch_size)
            _report("stream reorder_and_pad done", t0)
    else:
        # Eager path (current production behavior pre-#3).
        _report("about to load_parquet_and_json", t0)
        war = visualization.WindowAnalysisResult.load_parquet_and_json(
            folder_path=input_war_dir, parquet_name="war.parquet", json_name="war.json"
        )
        _report("loaded WAR", t0)

        _report("about to reorder_and_pad_channels", t0)
        war.reorder_and_pad_channels(CHANNEL_REORDER, use_abbrevs=True)
        _report("reorder_and_pad_channels done", t0)

        with tempfile.TemporaryDirectory(prefix="profile_war_") as tmpdir:
            _report("about to save_parquet_and_json", t0)
            war.save_parquet_and_json(Path(tmpdir))
            _report("save done", t0)

        del war

    gc.collect()
    _report("freed + gc.collect", t0)

    print(f"\nPeak RSS: {_rss_peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
