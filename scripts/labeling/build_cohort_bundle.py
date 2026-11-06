#!/usr/bin/env python
"""Build blinded human-labeling image bundles across a cohort defined in ``config/datasets``.

For each animal in the given dataset configs, this loads the recordings the way WAR generation does
(:func:`neurodent.workflow.utils.load_animal_recordings`), draws random windows, and renders every channel
with the rows shuffled and labelled neutrally (``Ch A``, ``Ch B``, ...), so a rater cannot read anatomy off
the label or the row position. Passing several datasets (or ``--all``) mixes the strains into one bundle:
recordings with different channel counts (for example 8- and 10-channel montages) share the union of
neutral slots, a shorter recording leaves the extra slots blank, and the rater page shows only the channels
that recording has.

The unblinding key ``keymap.csv``, which maps the neutral slots back to the true channels, is written to an
``_unblind/`` directory outside the bundle, so it is never shipped to raters. When the labels come back,
:func:`neurodent.results.scoring.unblind` uses it to restore the true channels.

Run from the repository root, since dataset ``extract_func`` paths resolve relative to the working
directory::

    # Discovery check that loads no data and surfaces stale configs; fine on a development machine.
    uv run python scripts/labeling/build_cohort_bundle.py --all --dry-run

    # Real loads are heavy; submit them on the cluster through the sbatch wrapper, which forwards all flags.
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

# Import the sibling labeling scripts (render_context.py and build_rater_bundle.py) by path.
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
    """Exit unless the working directory is the repository root.

    Dataset configs reference ``extract_func`` by a path relative to the working directory.

    Raises:
        SystemExit: If ``config/config.yaml`` is not in the working directory.
    """
    if not Path("config/config.yaml").exists():
        raise SystemExit(
            "Run from the repo root: 'config/config.yaml' not found in the CWD. Dataset extract_func "
            "paths resolve relative to the working directory."
        )


def _prepare(dataset):
    """Load a dataset's config and expanded samples, set its temp directory, and install its globals.

    ``apply_samples_config`` installs the channel map and ``ANIMAL_METADATA``. It has to run before any
    ``load_animal_recordings`` or ``resolve_channels`` call, so it runs here.

    Args:
        dataset (str): Dataset name, as in ``config/datasets/<name>.yaml``.

    Returns:
        tuple[dict, dict]: ``(config, samples_config)``.
    """
    config = load_dataset_config(dataset)
    samples_config = expand_animals_config(resolve_samples_config(config))
    set_temp_directory(config["temp_directory"])
    apply_samples_config(samples_config)
    return config, samples_config


def load_cohort_animal(samples_config, config, animal_id, logger=log):
    """Load one animal's recordings the way the bundle builder and WAR generation do.

    This is the one definition of the cohort load path (the configured ``channel_subset`` and
    ``load_animal_recordings``), so every consumer, rendering and detector scoring alike, reconstructs the
    recordings the rater saw, with the same channel identities and order.

    Args:
        samples_config (dict): Expanded samples config from :func:`_prepare`.
        config (dict): Dataset config from :func:`_prepare`.
        animal_id (str): Canonical animal id.
        logger (logging.Logger, optional): Logger for the loader's progress messages. Defaults to this
            module's logger.

    Returns:
        AnimalOrganizer: The loaded animal. Its ``long_recordings`` and ``animaldays`` are index-aligned
        with the ``"{animal}__{i}"`` recording names used across the bundle.
    """
    channel_subset = samples_config.get("_animal_channel_subsets", {}).get(animal_id)
    return load_animal_recordings(
        samples_config, config, [("", animal_id, "")], animal_id,
        channel_subset=channel_subset, logger=logger,
    )


def iter_cohort_animals(datasets, *, animal=None, limit_per_dataset=None, seed=0):
    """Yield ``(dataset, samples_config, config, animal_id)`` for every animal in the cohort.

    Prepares each dataset once with :func:`_prepare` and applies the seeded ``--limit-per-dataset``
    selection with :func:`_select_animals`. The bundle builder and the detector scoring scripts share this
    iterator, so they walk the same cohort in the same order and score what was rendered. A dataset's
    globals stay installed until the iterator moves on to the next dataset.

    Args:
        datasets (list[str]): Dataset names, walked in order.
        animal (str, optional): Yield only this animal id. Defaults to every animal in each dataset.
        limit_per_dataset (int, optional): Cap each dataset to this many animals, drawn with ``seed``.
            Defaults to no cap.
        seed (int, optional): Seed for the ``limit_per_dataset`` draw. Defaults to 0.

    Yields:
        tuple[str, dict, dict, str]: ``(dataset, samples_config, config, animal_id)``.
    """
    for ds in datasets:
        config, samples_config = _prepare(ds)   # installs this dataset's channel map and ANIMAL_METADATA
        animals = _select_animals(samples_config, animal, limit_per_dataset, seed)
        log.info(f"dataset {ds}: {len(animals)} animals")
        for a in animals:
            yield ds, samples_config, config, a


def dryrun_animal(samples_config, config, animal_id):
    """Return a preflight report for one animal without loading recording data.

    Runs the loader's own validation, ``load_animal_recordings(..., validate_only=True)``: discovery,
    ``skip_sessions`` and every ``manual_datetimes`` check that needs no file contents, raising where the
    real load would. Anything that reads a file (the reader, units, channel names, the channel split) is
    only checked by a real load. For an animal with a configured ``channel_subset``, it also checks that
    every subset token resolves through the channel map; whether those channels are in the recording needs
    a real load.

    Args:
        samples_config (dict): Expanded samples config from :func:`_prepare`.
        config (dict): Dataset config from :func:`_prepare`.
        animal_id (str): Canonical animal id.

    Returns:
        dict: ``animal``, ``n_files``, ``n_sessions``, ``n_datetimes``, ``is_joint``, ``channels_ok``,
        ``dt_ok`` and ``note``. Loader errors and unresolved channels are recorded in ``note`` rather than
        raised.
    """
    rec = {"animal": animal_id, "n_files": 0, "n_sessions": 0, "n_datetimes": None,
           "is_joint": False, "channels_ok": None, "dt_ok": True, "note": ""}
    dts = samples_config.get("manual_datetimes", {}).get(animal_id)
    rec["n_datetimes"] = len(dts) if isinstance(dts, (list, dict)) else None
    rec["is_joint"] = animal_id in samples_config.get("_animal_channel_subsets", {})

    # Config-only check: every channel_subset token resolves to a canonical abbreviation through the
    # configured channel map.
    if rec["is_joint"]:
        subset = samples_config["_animal_channel_subsets"][animal_id]
        try:
            for c in subset:
                resolve_channel(c)
            rec["channels_ok"] = True
        except ValueError as e:
            rec["channels_ok"] = False
            rec["note"] = f"channel does not resolve: {e}"

    # The real loader's discovery and datetime validation, without loading data.
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.WARNING)   # keep the loader's per-file info logging out of the table
    try:
        summary = load_animal_recordings(
            samples_config, config, [("", animal_id, "")], animal_id, validate_only=True,
        )
        rec["n_sessions"] = summary["n_sessions"]
        rec["n_files"] = summary["n_files"]
    except Exception as e:  # the error the real load would raise
        rec["dt_ok"] = False
        rec["note"] = (f"{type(e).__name__}: {e} [loading will raise]; " + rec["note"]).strip("; ")
    finally:
        root.setLevel(prev_level)
    return rec


def _print_dryrun(reports):
    """Print dry-run reports from :func:`dryrun_animal` as a table."""
    hdr = f"{'animal':<16} {'files':>6} {'sess':>5} {'dates':>6} {'joint':>6} {'chans':>6}  note"
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        chans = "-" if r["channels_ok"] is None else ("ok" if r["channels_ok"] else "bad")
        dates = "-" if r["n_datetimes"] is None else str(r["n_datetimes"])
        print(f"{r['animal']:<16} {r['n_files']:>6} {r['n_sessions']:>5} {dates:>6} "
              f"{'yes' if r['is_joint'] else 'no':>6} {chans:>6}  {r['note']}")


def render_animal(samples_config, config, animal_id, bundle_dir, *, n_per_animal, seed, blind_seed,
                  all_channels):
    """Load one animal and render its windows, blinded, into the shared bundle.

    Splits ``n_per_animal`` windows across the animal's recordings and appends to the bundle manifest when
    it already exists. A recording for which ``render_lro`` raises ValueError (for example, too few distinct
    windows to draw) is logged and skipped.

    Args:
        samples_config (dict): Expanded samples config from :func:`_prepare`.
        config (dict): Dataset config from :func:`_prepare`.
        animal_id (str): Canonical animal id.
        bundle_dir (Path): Bundle directory shared by every animal in the run.
        n_per_animal (int): Windows to draw for this animal, split across its recordings.
        seed (int): Window-selection seed.
        blind_seed (int): Channel-shuffle seed.
        all_channels (list[str]): The union of neutral slots (``Ch A`` onward), so recordings with
            different channel counts share one manifest. A shorter recording fills only its leading slots
            and leaves the rest blank.

    Returns:
        list[dict]: Keymap rows with keys ``recording``, ``window``, ``slot``, ``channel``, ``animal`` and
        ``genotype``. Empty when the animal has no recordings.
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
            # With blind_seed set, render_windows blinds each window independently and returns a
            # per-(recording, window, slot) keymap, so no display slot can be learned to a true channel
            # across this recording's windows.
            rows, keymap = R.render_lro(
                lro, bundle_dir, n_select=count, recording=recording,
                append=(bundle_dir / R.MANIFEST).exists(), seed=seed,
                blind_seed=blind_seed, true_names=true_names, all_channels=all_channels,
            )
        except ValueError as e:  # for example, select_random cannot draw enough distinct windows here
            log.warning(f"{recording}: skipped ({e})")
            continue
        for k in keymap:
            keymap_rows.append({"recording": recording, "window": int(k["window"]), "slot": k["slot"],
                                "channel": k["channel"], "animal": animal_id, "genotype": genotype})
        log.info(f"{recording}: rendered {len(rows)} window(s), {len(true_names)} ch")
    return keymap_rows


def _write_keymap(path, rows):
    """Write keymap rows to ``path`` as CSV, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["recording", "window", "slot", "channel", "animal", "genotype"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _dataset_channel_count(dataset):
    """Return a dataset's montage size (its number of canonical channels) from its config, without a load."""
    return len(resolve_samples_config(load_dataset_config(dataset)).get("channels", {}))


def _select_animals(samples_config, animal, limit_per_dataset, seed):
    """Return a dataset's animals, optionally capped to a seeded random subset.

    ``limit_per_dataset`` is for development smoke tests (for example one animal per strain for a fast
    mixed bundle); a real run leaves it ``None`` and loads every animal.

    Args:
        samples_config (dict): Expanded samples config from :func:`_prepare`.
        animal (str | None): Return only this animal id when given.
        limit_per_dataset (int | None): Maximum number of animals, drawn with ``seed``. Ignored when
            ``animal`` is given.
        seed (int): Seed for the draw.

    Returns:
        list[str]: Animal ids. A capped draw is sorted.
    """
    animals = [animal] if animal else enumerate_cohort(samples_config)
    if limit_per_dataset and not animal and len(animals) > limit_per_dataset:
        animals = sorted(random.Random(seed).sample(animals, limit_per_dataset))
    return animals


def build_cohort(datasets, out_root, *, animal=None, n_per_animal=40, seed=0, blind_seed=0,
                 dry_run=False, make_bundle=True, limit_per_dataset=None):
    """Render the animals of one or more datasets into one blinded bundle.

    Recordings with different channel counts (for example 8- and 10-channel strains) share the bundle
    through the union of neutral slots; a shorter recording leaves the extra slots blank, and the rater
    page shows only the channels that recording has. Each run writes a new ``bundle`` directory and a
    paired ``_unblind`` directory holding ``keymap.csv`` and ``cohort_manifest.json``; when earlier runs
    exist, both get the next free ``-N`` suffix. An animal that fails is logged and recorded in the cohort
    manifest, and the rest of the cohort still renders.

    Args:
        datasets (list[str]): Dataset names.
        out_root (str | Path): Directory that receives the bundle and unblinding directories.
        animal (str, optional): Only this animal id. Defaults to every animal.
        n_per_animal (int, optional): Windows drawn per animal, split across its recordings. Defaults to 40.
        seed (int, optional): Window-selection seed, also used for the ``limit_per_dataset`` draw.
            Defaults to 0.
        blind_seed (int, optional): Channel-shuffle seed. Defaults to 0.
        dry_run (bool, optional): Print a discovery report per dataset instead of rendering. Defaults to
            False.
        make_bundle (bool, optional): Build the rater zip after rendering. Defaults to True.
        limit_per_dataset (int, optional): Cap each dataset to this many seeded random animals. Defaults
            to no cap.

    Returns:
        int: Process exit code, 1 if any animal failed (in a dry run, if any report shows no files, an
        unresolved channel or a loader validation error), else 0.

    Raises:
        SystemExit: If not run from the repository root.
    """
    _require_repo_root()
    out_root = Path(out_root)

    if dry_run:
        rc = 0
        for ds in datasets:
            config, samples_config = _prepare(ds)
            animals = _select_animals(samples_config, animal, limit_per_dataset, seed)
            print(f"\n{ds}  ({len(animals)} animals)")
            reports = [dryrun_animal(samples_config, config, a) for a in animals]
            _print_dryrun(reports)
            bad = [r for r in reports if r["n_files"] == 0 or r["channels_ok"] is False or not r["dt_ok"]]
            if bad:
                print(f"  {len(bad)} with no files, unresolved channels or a loader validation error: "
                      f"{[r['animal'] for r in bad]}")
                rc = 1
        return rc

    # Union slot set across all datasets (from each config's channel map; no data load).
    n_slots = max(_dataset_channel_count(ds) for ds in datasets)
    all_channels = R.neutral_labels(n_slots)
    log.info(f"datasets={datasets}  n_slots={n_slots}  n_per_animal={n_per_animal} "
             f"seed={seed} blind_seed={blind_seed}")

    # Write new sibling directories each run and never delete earlier output; the user cleans up by hand.
    # The bundle and its keymap directory share one suffix, so a run's zip and its unblinding key stay
    # paired and no earlier run's keymap is overwritten.
    out_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        sfx = "" if n == 1 else f"-{n}"
        bundle_dir = out_root / f"bundle{sfx}"
        unblind_dir = out_root / f"_unblind{sfx}"
        if not bundle_dir.exists() and not unblind_dir.exists():
            break
        n += 1
    log.info(f"writing bundle to {bundle_dir}, keymap to {unblind_dir}")

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
            log.error(f"{a}: failed with {type(e).__name__}: {e}")
            failures.append({"dataset": ds, "animal": a, "error": f"{type(e).__name__}: {e}"})

    # The keymap and cohort manifest live outside the bundle directory, in the paired _unblind directory,
    # so build() never ships them to raters.
    _write_keymap(unblind_dir / "keymap.csv", all_keymap)
    (unblind_dir / "cohort_manifest.json").write_text(json.dumps({
        "datasets": datasets, "seed": seed, "blind_seed": blind_seed, "n_per_animal": n_per_animal,
        "n_slots": n_slots, "rendered_animals": rendered, "failures": failures,
    }, indent=2, default=str))

    zips = []
    if make_bundle and (bundle_dir / R.MANIFEST).exists():
        name = "_".join(datasets) if len(datasets) <= 2 else f"cohort{len(datasets)}strains"
        zips.append(B.build(bundle_dir, name=name))
    log.info(f"built {len(zips)} bundle(s); keymap at {unblind_dir / 'keymap.csv'}")
    if failures:
        log.warning(f"{len(failures)} animal(s) failed: {[f['animal'] for f in failures]}")
    return 1 if failures else 0


def main(argv=None):
    """Parse command-line arguments and build the cohort bundle.

    Args:
        argv (list[str], optional): Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code from :func:`build_cohort`.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", default=None,
                   help="one or more dataset names (config/datasets/<name>.yaml); several render into one "
                        "mixed bundle")
    p.add_argument("--all", action="store_true",
                   help=f"render all real strains ({' '.join(REAL_STRAINS)}) into one mixed bundle")
    p.add_argument("--animal", default=None, help="single animal id (only valid with exactly one --dataset)")
    p.add_argument("--n-per-animal", type=int, default=40, help="windows drawn per animal (spread across its sessions)")
    p.add_argument("--seed", type=int, default=0, help="window-selection seed (reproducible draw)")
    p.add_argument("--blind-seed", type=int, default=0, help="channel-shuffle seed (reproducible blinding)")
    p.add_argument("--out", default=None, help="output root (default: results/labeling/<dataset>, or .../mixed)")
    p.add_argument("--dry-run", action="store_true", help="discovery-only check; no data load, no render")
    p.add_argument("--no-bundle", action="store_true", help="render and write the keymap, but do not build the zip")
    p.add_argument("--limit-per-dataset", type=int, default=None,
                   help="for development smoke tests: cap each dataset to N seeded random animals (for "
                        "example 1 per strain). Omit for a real run, which loads every animal.")
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
