"""Pre-flight a dataset config before spending cluster time on it.

A green ``snakemake --dry-run`` proves only that the DAG resolves. It never opens a file,
and neither does ``validate_only``: ``load_animal_recordings`` returns right after recording
discovery group sizes, with no per-file load. For IQSEC2 that gap was the whole problem, so
this script runs both the cheap structural check and the file-level checks that actually
predict whether a run will survive.

Gates, in increasing cost:

1. ``discovery``   Per animal, run ``load_animal_recordings(validate_only=True)``. Covers
                   patterns, skip_sessions and every manual_datetime check that needs no
                   file contents.
2. ``readable``    Open every file a load would open, with the reader and settings the
                   loader resolved for its source, reading headers only. This is the check
                   that would have caught 49 unreadable rhd files blocking 10 of 23
                   animals. It used to open rhd files only and skip every other reader
                   without a word; it now covers every format and lists every animal.
                   Slow, so it is opt-in.

Nothing here is specific to IQSEC2 any more; it works on any dataset config.

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


def gate_readable(discovery, logger):
    """Open every file a load would open, with the reader and settings the load would use.

    Works from gate 1's per-source records, so the files, the reader and its settings are
    the loader's own resolution rather than a second copy of it. Every animal gets a row; one
    whose gate 1 failed is listed as not checked. A file shared by several animals (one Intan
    file holds four) is opened once.

    A reader failure is classified the way a load would treat it: ``abort`` stops that
    animal's load, ``drop`` loses that one file (a group whose payload is empty or corrupt)
    and the load carries on. Within a source every file must hold the same channels as its
    first file, matched as the loader matches them, and a source that is split must contain
    every channel it is split to.

    It reports what a real run would see with the config's own settings. With
    ``ignore_integrity_checks`` on, a timestamp discontinuity is invisible here by design;
    to detect newly corrupt rhd files, run once with the flag stripped and keep the result as
    a baseline.
    """
    from neurodent.loading.lro_merge import match_channels
    from neurodent.loading.preflight import failure_outcome, open_item, reader_kwargs

    opened = {}
    results = []
    for record in discovery:
        animal_id = record["animal"]
        if not record["ok"]:
            print(f"  SKIP  {animal_id:<20} not checked, because gate 1 failed")
            results.append({"animal": animal_id, "checked": False, "reason": "gate 1 failed"})
            continue
        n_items, aborts, drops, problems = 0, [], [], []
        for source in record["sources"]:
            kwargs = source["lro_kwargs"]
            label = "animal" if source["source"] is None else f"source {source['source']}"
            first = None
            for item in source["items"]:
                n_items += 1
                paths = tuple(item.get_path_list())
                key = (paths, kwargs.get("mode", "si"), str(kwargs.get("extract_func")),
                       repr(sorted(reader_kwargs(kwargs).items())))
                if key not in opened:
                    try:
                        opened[key] = ("ok", open_item(item, kwargs))
                    except Exception as exc:  # noqa: BLE001 - classify, do not stop
                        opened[key] = (failure_outcome(item, kwargs, exc), f"{type(exc).__name__}: {exc}")
                outcome, info = opened[key]
                if outcome != "ok":
                    (aborts if outcome == "abort" else drops).append({"file": paths[0], "error": info})
                    continue
                names = info["channel_names"]
                if first is None:
                    first = (paths[0], names)
                else:
                    try:
                        match_channels(first[1], names, first[0], paths[0])
                    except ValueError as exc:
                        problems.append({"file": paths[0], "error": f"{label}: {exc}"})
                missing = [c for c in (source["split_to"] or []) if c not in names]
                if missing:
                    problems.append({
                        "file": paths[0],
                        "error": f"{label}: split to channels the file does not have: {missing}",
                    })
        failed = bool(aborts or problems)
        note = f", {len(drops)} would be dropped" if drops else ""
        if failed:
            worst = (aborts or problems)[0]
            print(f"  FAIL  {animal_id:<20} {len(aborts)} would stop the load, {len(problems)} "
                  f"inconsistent{note}; first: {Path(worst['file']).name}: {worst['error'][:110]}")
        else:
            print(f"  ok    {animal_id:<20} {n_items - len(drops)}/{n_items} open{note}")
        results.append({"animal": animal_id, "checked": True, "n_files": n_items, "failed": failed,
                        "aborts": aborts, "drops": drops, "problems": problems})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="dataset name under config/datasets/")
    parser.add_argument("--readable", action="store_true",
                        help="also open every file a load would open (slow; submit via sbatch)")
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
        print("\ngate 2: every file a load would open, opened the way it would be")
        readable = gate_readable(discovery, logger)
        n_failed = sum(1 for r in readable if r["checked"] and r["failed"])
        n_skipped = sum(1 for r in readable if not r["checked"])
        n_dropped = sum(len(r.get("drops", [])) for r in readable)
        print(f"\n  {len(readable) - n_failed - n_skipped} ok, {n_failed} failed, "
              f"{n_skipped} not checked; {n_dropped} file(s) would be dropped")
        n_bad += n_failed + n_skipped

    if args.json:
        def _plain(obj):
            return obj.get_path_list() if hasattr(obj, "get_path_list") else str(obj)

        report = {"discovery": discovery, "readable": readable}
        args.json.write_text(json.dumps(report, indent=2, default=_plain))
        print(f"\nwrote {args.json}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
