"""Pre-flight a dataset config before spending cluster time on it.

A green ``snakemake --dry-run`` proves only that the DAG resolves. It never opens a file,
and neither does ``validate_only``: ``load_animal_recordings`` returns right after recording
discovery group sizes, with no per-file load. For IQSEC2 that gap was the whole problem, so
this script runs both the cheap structural check and the file-level checks that actually
predict whether a run will survive.

Gates, in increasing cost:

1. ``discovery``   Per animal, run ``load_animal_recordings(validate_only=True)``. Covers
                   patterns, skip_sessions and the manual_datetime key check.
2. ``readable``    Open every discovered rhd with the config's own lro_kwargs. This is the
                   check that would have caught 49 unreadable files blocking 10 of 23
                   animals. Slow, so it is opt-in.

Usage:
    uv run python scripts/iqsec2/validate_config.py iqsec2
    uv run python scripts/iqsec2/validate_config.py iqsec2 --readable --json out.json
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def build_config(dataset):
    """Assemble the merged config exactly as workflow/Snakefile does."""
    from neurodent.workflow.utils import (
        deep_merge_dict,
        expand_animals_config,
        merge_dataset_config,
        resolve_samples_config,
    )
    from neurodent.workflow.utils.config import apply_samples_config

    config = yaml.safe_load((REPO / "config/config.yaml").read_text())
    local = REPO / "config/config.local.yaml"
    if local.exists():
        override = yaml.safe_load(local.read_text())
        if isinstance(override, dict):
            config = deep_merge_dict(config, override)
    config, _ = merge_dataset_config(config, dataset, datasets_dir=str(REPO / "config/datasets"))
    samples_config = expand_animals_config(resolve_samples_config(config))
    apply_samples_config(samples_config)
    return config, samples_config


def gate_discovery(config, samples_config, logger):
    """Run the structural pre-flight per animal."""
    from neurodent.workflow.utils import load_animal_recordings

    results = []
    for animal_id in [a["id"] for a in samples_config["animals"]]:
        record = {"animal": animal_id}
        try:
            summary = load_animal_recordings(
                samples_config, config, [("", animal_id, "")], animal_id,
                logger=logger, validate_only=True,
            )
            record.update(ok=True, **summary)
            print(f"  ok    {animal_id:<20} sessions={summary['n_sessions']:<3} files={summary['n_files']}")
        except Exception as exc:  # noqa: BLE001 - report every animal, do not stop at the first
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
            print(f"  FAIL  {animal_id:<20} {type(exc).__name__}: {str(exc)[:130]}")
        results.append(record)
    return results


def gate_readable(samples_config, logger):
    """Open every discovered rhd with the config's own reader kwargs.

    Deliberately uses each animal's real lro_kwargs rather than a fixed call, so that if the
    config sets ignore_integrity_checks this gate reports what a real run would see. To
    detect newly corrupt files, run it once with the flag stripped and keep the result as a
    baseline: with the flag on, a timestamp discontinuity is by design invisible here.
    """
    from neurodent.loading.discovery import FileDiscoverer
    from neurodent.workflow.utils.discovery import resolve_animal_pattern

    data_root = str(samples_config.get("data_root", ""))
    overrides = samples_config.get("_animal_overrides", {})
    results = []
    for entry in samples_config["animals"]:
        animal_id = entry["id"]
        specs = overrides.get(animal_id, {}).get("sources") or [overrides.get(animal_id, {})]
        failures, seen = [], 0
        for spec in specs:
            pattern = spec.get("pattern") or entry.get("pattern")
            kwargs = dict(spec.get("lro_kwargs") or entry.get("lro_kwargs") or {})
            if not pattern or kwargs.get("extract_func") != "read_intan":
                continue
            resolved = resolve_animal_pattern(pattern, animal_id, data_root=data_root)
            for item in FileDiscoverer(resolved).discover():
                seen += 1
                path = getattr(item, "path", None) or item.paths[0]
                try:
                    from neo.rawio.intanrawio import IntanRawIO

                    reader = IntanRawIO(
                        filename=str(path),
                        ignore_integrity_checks=bool(kwargs.get("ignore_integrity_checks")),
                    )
                    reader.parse_header()
                except Exception as exc:  # noqa: BLE001 - collect, do not stop
                    failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
        if seen:
            mark = "ok   " if not failures else "FAIL "
            print(f"  {mark} {animal_id:<20} {seen - len(failures)}/{seen} readable")
        results.append({"animal": animal_id, "n_files": seen, "failures": failures})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="dataset name under config/datasets/")
    parser.add_argument("--readable", action="store_true",
                        help="also open every discovered rhd (slow; submit via sbatch)")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger("validate")
    logger.setLevel(logging.CRITICAL)

    config, samples_config = build_config(args.dataset)
    animals = samples_config["animals"]
    print(f"dataset={args.dataset}  animals={len(animals)}\n")

    print("gate 1: discovery")
    discovery = gate_discovery(config, samples_config, logger)
    n_bad = sum(1 for r in discovery if not r["ok"])
    n_files = sum(r.get("n_files", 0) for r in discovery)
    print(f"\n  {len(discovery) - n_bad} ok, {n_bad} failed, {n_files} files discovered")

    readable = None
    if args.readable:
        print("\ngate 2: every discovered rhd opens")
        readable = gate_readable(samples_config, logger)
        bad = sum(len(r["failures"]) for r in readable)
        print(f"\n  {bad} unreadable file(s)")
        n_bad += bad

    if args.json:
        args.json.write_text(json.dumps({"discovery": discovery, "readable": readable}, indent=2))
        print(f"\nwrote {args.json}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
