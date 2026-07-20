#!/usr/bin/env python
"""Build blinded human-labeling image bundles across a cohort defined in ``config/datasets``.

For each animal in the given dataset config(s) this reconstructs the recording(s) exactly as WAR
generation does (via :func:`neurodent.workflow.utils.load_animal_recordings`), draws random windows, and
renders EVERY channel with the rows SHUFFLED and NEUTRALLY LABELLED (``Ch A``, ``Ch B``, ...) so a rater
cannot read anatomy off the label or position and bias their judgement. Pass MULTIPLE datasets (or
``--all``) to mix all strains into ONE bundle: recordings with different channel counts (e.g. 8- vs
10-channel montages) share a UNION of neutral slots, a shorter recording leaves the extra slots blank,
and the rater page shows only the channels that recording has.

The de-scramble ``keymap.csv`` — how the neutral slots map back to true channels — is written to an
``_unblind/`` directory OUTSIDE the bundle, so it is never shipped to raters. On return, feed it to
:func:`neurodent.results.scoring.unblind` to put labels back in order.

Run from the REPO ROOT (dataset ``extract_func`` paths resolve relative to the CWD)::

    # cheap discovery check first -- surfaces stale configs without loading any data (dev box is fine):
    uv run python scripts/labeling/build_cohort_bundle.py --all --dry-run

    # real loads are heavy -- submit on the cluster via the sbatch wrapper (forwards all flags):
    sbatch scripts/labeling/run_cohort_bundle.sbatch --dataset arx_parv --animal 29 --n-per-animal 40
    sbatch scripts/labeling/run_cohort_bundle.sbatch --all --n-per-animal 20   # one mixed cohort bundle
"""
import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

# Import the sibling labeling scripts (render_context.py / build_rater_bundle.py) by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_context as R  # noqa: E402
import build_rater_bundle as B  # noqa: E402

from neurodent import constants  # noqa: E402
from neurodent.core.utils import set_temp_directory, resolve_channel, resolve_channels  # noqa: E402
from neurodent.workflow.utils import (  # noqa: E402
    apply_samples_config,
    enumerate_cohort,
    expand_animals_config,
    load_animal_recordings,
    load_dataset_config,
    resolve_samples_config,
)

log = logging.getLogger("build_cohort_bundle")

REAL_STRAINS = ["arx_parv", "arx_rosa", "sox5_bin", "ap3b2_rhd"]   # the cohort (excludes test fixtures)


def _require_repo_root():
    """Dataset configs reference ``extract_func`` by a CWD-relative path, so refuse to run elsewhere."""
    if not Path("config/config.yaml").exists():
        raise SystemExit(
            "Run from the repo root: 'config/config.yaml' not found in the CWD. Dataset extract_func "
            "paths resolve relative to the working directory."
        )


def _prepare(dataset):
    """Assemble config, expand samples, and install the channel map / ANIMAL_METADATA globals.

    ``apply_samples_config`` must run before any ``load_animal_recordings`` / ``resolve_channels`` call,
    so it lives here. Returns ``(config, samples_config)``.
    """
    config = load_dataset_config(dataset)
    samples_config = expand_animals_config(resolve_samples_config(config))
    set_temp_directory(config["temp_directory"])
    apply_samples_config(samples_config)
    return config, samples_config


def load_cohort_animal(samples_config, config, animal_id, logger=log):
    """Load one animal's recordings EXACTLY as the bundle builder (and WAR generation) do.

    The SINGLE definition of the cohort load path — the config ``channel_subset`` lookup plus
    ``load_animal_recordings`` — so any downstream consumer (rendering, detector scoring) reconstructs
    the same recordings, in the same channel identity/order, that the rater saw. Returns the
    ``AnimalOrganizer`` (its ``.long_recordings`` / ``.animaldays`` are index-aligned to the
    ``"{animal}__{i}"`` recording naming used across the bundle).
    """
    channel_subset = samples_config.get("_animal_channel_subsets", {}).get(animal_id)
    return load_animal_recordings(
        samples_config, config, [("", animal_id, "")], animal_id,
        channel_subset=channel_subset, logger=logger,
    )


def iter_cohort_animals(datasets, *, animal=None, limit_per_dataset=None, seed=0):
    """Yield ``(dataset, samples_config, config, animal_id)`` for every animal in the cohort.

    Prepares each dataset's config/channel-map globals exactly once (via :func:`_prepare`) and applies
    the same seeded ``--limit-per-dataset`` selection (via :func:`_select_animals`). Shared by the
    bundle builder and the detector scorer so both walk the IDENTICAL cohort in the identical order —
    the loader cannot drift between "what was rendered" and "what is scored".
    """
    for ds in datasets:
        config, samples_config = _prepare(ds)   # installs THIS dataset's channel map / ANIMAL_METADATA
        animals = _select_animals(samples_config, animal, limit_per_dataset, seed)
        log.info(f"-- {ds}: {len(animals)} animals")
        for a in animals:
            yield ds, samples_config, config, a


def dryrun_animal(samples_config, config, animal_id):
    """Pre-flight report for one animal; never raises (captures errors into ``note``).

    Uses the loader's OWN validation via ``load_animal_recordings(..., validate_only=True)`` —
    the same discovery + skip_sessions + ``manual_datetimes`` checks the real load runs (which
    raise on any mismatch), WITHOUT loading data. So a clean report guarantees a clean load;
    the dry-run can no longer disagree with the real loader. A cheap config-only
    channel-resolvability check is added for joint animals (channel *presence* in the recording
    needs a load and is left to the real run).
    """
    rec = {"animal": animal_id, "n_files": 0, "n_sessions": 0, "n_datetimes": None,
           "is_joint": False, "channels_ok": None, "dt_ok": True, "note": ""}
    dts = samples_config.get("manual_datetimes", {}).get(animal_id)
    rec["n_datetimes"] = len(dts) if isinstance(dts, (list, dict)) else None
    rec["is_joint"] = animal_id in samples_config.get("_animal_channel_subsets", {})

    # Cheap config-only check: do this animal's channel_subset tokens resolve to canonical
    # abbrevs via the configured channel map?
    if rec["is_joint"]:
        subset = samples_config["_animal_channel_subsets"][animal_id]
        try:
            for c in subset:
                resolve_channel(c)
            rec["channels_ok"] = True
        except ValueError as e:
            rec["channels_ok"] = False
            rec["note"] = f"channel does not resolve: {e}"

    # THE single checkpoint: the real loader's discovery + datetime validation, no data load.
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.WARNING)   # quiet the loader's per-file INFO chatter for the table
    try:
        summary = load_animal_recordings(
            samples_config, config, [("", animal_id, "")], animal_id, validate_only=True,
        )
        rec["n_sessions"] = summary["n_sessions"]
        rec["n_files"] = summary["n_files"]
    except Exception as e:  # exactly what the real load would raise
        rec["dt_ok"] = False
        rec["note"] = (f"{type(e).__name__}: {e} [loading will raise]; " + rec["note"]).strip("; ")
    finally:
        root.setLevel(prev_level)
    return rec


def _print_dryrun(reports):
    hdr = f"{'animal':<16} {'files':>6} {'sess':>5} {'dates':>6} {'joint':>6} {'chans':>6}  note"
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        chans = "-" if r["channels_ok"] is None else ("ok" if r["channels_ok"] else "BAD")
        dates = "-" if r["n_datetimes"] is None else str(r["n_datetimes"])
        print(f"{r['animal']:<16} {r['n_files']:>6} {r['n_sessions']:>5} {dates:>6} "
              f"{'yes' if r['is_joint'] else 'no':>6} {chans:>6}  {r['note']}")


def render_animal(samples_config, config, animal_id, bundle_dir, *, n_per_animal, seed, blind_seed,
                  all_channels):
    """Load one animal and blind-render its windows into the shared mixed bundle. Returns keymap rows.

    ``all_channels`` is the union neutral-slot set (``Ch A..Ch{max}``) so recordings with different
    channel counts share ONE manifest; a shorter recording fills only its own prefix and leaves the
    rest blank. Append is driven by whether the shared manifest already exists.
    """
    genotype = constants.ANIMAL_METADATA.get(animal_id, {}).get("genotype", "Unknown")
    ao = load_cohort_animal(samples_config, config, animal_id)
    lros = ao.long_recordings
    if not lros:
        log.warning(f"{animal_id}: no recordings, skipped")
        return []

    base, rem = divmod(n_per_animal, len(lros))
    keymap_rows = []
    for i, lro in enumerate(lros):
        count = base + (1 if i < rem else 0)
        if count <= 0:
            continue
        true_names = resolve_channels(list(lro.channel_names))
        recording = f"{animal_id}__{i}"
        try:
            # blind_seed makes render_windows blind EACH window independently and return a
            # per-(recording, window, slot) keymap — no display slot can be learned to a true channel
            # across this recording's windows.
            rows, keymap = R.render_lro(
                lro, bundle_dir, n_select=count, recording=recording,
                append=(bundle_dir / R.MANIFEST).exists(), seed=seed,
                blind_seed=blind_seed, true_names=true_names, all_channels=all_channels,
            )
        except ValueError as e:  # e.g. select_random cannot draw enough distinct windows in this session
            log.warning(f"{recording}: skipped ({e})")
            continue
        for k in keymap:
            keymap_rows.append({"recording": recording, "window": int(k["window"]), "slot": k["slot"],
                                "channel": k["channel"], "animal": animal_id, "genotype": genotype})
        log.info(f"{recording}: rendered {len(rows)} window(s), {len(true_names)} ch")
    return keymap_rows


def _write_keymap(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["recording", "window", "slot", "channel", "animal", "genotype"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _dataset_channel_count(dataset):
    """Montage size (number of canonical channels) for a dataset, from its config -- cheap, no load."""
    return len(resolve_samples_config(load_dataset_config(dataset)).get("channels", {}))


def _select_animals(samples_config, animal, limit_per_dataset, seed):
    """This dataset's cohort, optionally capped to a seeded random subset.

    ``limit_per_dataset`` is a DEV/dogfood convenience (e.g. 1 animal per strain for a fast
    mixed smoke-test); a real run leaves it ``None`` and loads every animal.
    """
    animals = [animal] if animal else enumerate_cohort(samples_config)
    if limit_per_dataset and not animal and len(animals) > limit_per_dataset:
        animals = sorted(random.Random(seed).sample(animals, limit_per_dataset))
    return animals


def build_cohort(datasets, out_root, *, animal=None, n_per_animal=40, seed=0, blind_seed=0,
                 dry_run=False, make_bundle=True, limit_per_dataset=None):
    """Render one or more datasets' animals into ONE blinded mixed bundle.

    Recordings of different channel counts (e.g. 8- and 10-channel strains) share the bundle via a
    UNION of neutral slots; a shorter recording leaves the extra slots blank and the rater page shows
    only the channels that recording actually has. Returns a process exit code (0 ok, 1 on any failure).
    """
    _require_repo_root()
    out_root = Path(out_root)

    if dry_run:
        rc = 0
        for ds in datasets:
            config, samples_config = _prepare(ds)
            animals = _select_animals(samples_config, animal, limit_per_dataset, seed)
            print(f"\n## {ds}  ({len(animals)} animals)")
            reports = [dryrun_animal(samples_config, config, a) for a in animals]
            _print_dryrun(reports)
            bad = [r for r in reports if r["n_files"] == 0 or r["channels_ok"] is False or not r["dt_ok"]]
            if bad:
                print(f"  {len(bad)} broken (0 files / unresolved channels / session-datetime mismatch): "
                      f"{[r['animal'] for r in bad]}")
                rc = 1
        return rc

    # Union slot set across all datasets (from each config's channel map; no data load).
    n_slots = max(_dataset_channel_count(ds) for ds in datasets)
    all_channels = R.neutral_labels(n_slots)
    log.info(f"datasets={datasets}  n_slots={n_slots}  n_per_animal={n_per_animal} "
             f"seed={seed} blind_seed={blind_seed}")

    # Write a FRESH sister dir each run (never delete prior output; user cleans up manually).
    # The bundle and its keymap share the same suffix so a run's zip and its de-blind key stay paired
    # and no earlier run's keymap is ever clobbered.
    out_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        sfx = "" if n == 1 else f"-{n}"
        bundle_dir = out_root / f"bundle{sfx}"
        unblind_dir = out_root / f"_unblind{sfx}"
        if not bundle_dir.exists() and not unblind_dir.exists():
            break
        n += 1
    log.info(f"writing fresh bundle dir: {bundle_dir}  (keymap -> {unblind_dir})")

    all_keymap, failures, rendered = [], [], []
    for ds, samples_config, config, a in iter_cohort_animals(
        datasets, animal=animal, limit_per_dataset=limit_per_dataset, seed=seed
    ):
        try:
            kr = render_animal(samples_config, config, a, bundle_dir,
                               n_per_animal=n_per_animal, seed=seed, blind_seed=blind_seed,
                               all_channels=all_channels)
            all_keymap += kr
            if kr:
                rendered.append(a)
        except Exception as e:  # one stale animal must not sink the cohort
            log.error(f"{a}: FAILED -- {type(e).__name__}: {e}")
            failures.append({"dataset": ds, "animal": a, "error": f"{type(e).__name__}: {e}"})

    # Keymap + cohort manifest live OUTSIDE the bundle dir (in the paired _unblind sister) so build()
    # never ships them to raters.
    _write_keymap(unblind_dir / "keymap.csv", all_keymap)
    (unblind_dir / "cohort_manifest.json").write_text(json.dumps({
        "datasets": datasets, "seed": seed, "blind_seed": blind_seed, "n_per_animal": n_per_animal,
        "n_slots": n_slots, "rendered_animals": rendered, "failures": failures,
    }, indent=2, default=str))

    zips = []
    if make_bundle and (bundle_dir / R.MANIFEST).exists():
        name = "_".join(datasets) if len(datasets) <= 2 else f"cohort{len(datasets)}strains"
        zips.append(B.build(bundle_dir, name=name))
    log.info(f"built {len(zips)} bundle(s); keymap -> {unblind_dir / 'keymap.csv'}")
    if failures:
        log.warning(f"{len(failures)} animal(s) failed: {[f['animal'] for f in failures]}")
    return 1 if failures else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", default=None,
                   help="one or more dataset names (config/datasets/<name>.yaml); >1 -> one MIXED bundle")
    p.add_argument("--all", action="store_true",
                   help=f"render all real strains ({' '.join(REAL_STRAINS)}) into one mixed bundle")
    p.add_argument("--animal", default=None, help="single animal id (only valid with exactly one --dataset)")
    p.add_argument("--n-per-animal", type=int, default=40, help="windows drawn per animal (spread across its sessions)")
    p.add_argument("--seed", type=int, default=0, help="window-selection seed (reproducible draw)")
    p.add_argument("--blind-seed", type=int, default=0, help="channel-shuffle seed (reproducible blinding)")
    p.add_argument("--out", default=None, help="output root (default: results/labeling/<dataset>, or .../mixed)")
    p.add_argument("--dry-run", action="store_true", help="discovery-only check; no data load, no render")
    p.add_argument("--no-bundle", action="store_true", help="render + keymap but do not build the zip")
    p.add_argument("--limit-per-dataset", type=int, default=None,
                   help="DEV/dogfood only: cap each dataset to a seeded random N animals (e.g. 1 per "
                        "strain for a fast smoke test). Omit for a real run (loads every animal).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    datasets = list(REAL_STRAINS) if args.all else args.dataset
    if not datasets:
        p.error("provide --dataset <name ...> or --all")
    if args.animal and len(datasets) != 1:
        p.error("--animal requires exactly one --dataset")
    out_root = args.out or (f"results/labeling/{datasets[0]}" if len(datasets) == 1
                            else "results/labeling/mixed")
    return build_cohort(
        datasets, out_root, animal=args.animal, n_per_animal=args.n_per_animal,
        seed=args.seed, blind_seed=args.blind_seed, dry_run=args.dry_run,
        make_bundle=not args.no_bundle, limit_per_dataset=args.limit_per_dataset,
    )


if __name__ == "__main__":
    sys.exit(main())
