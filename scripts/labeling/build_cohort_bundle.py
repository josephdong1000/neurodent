#!/usr/bin/env python
"""Build blinded human-labeling image bundles across a cohort defined in ``config/datasets``.

For each animal in a dataset config this reconstructs the recording(s) exactly as WAR generation does
(via :func:`neurodent.workflow.utils.load_animal_recordings`), draws random windows, and renders EVERY
channel with the rows SHUFFLED and NEUTRALLY LABELLED (``Ch A``, ``Ch B``, ...) so a rater cannot read
anatomy off the label or position and bias their judgement. Animals that share a channel montage
accumulate into one rater bundle (``build_rater_bundle.build``); different montages get separate
bundles.

The de-scramble ``keymap.csv`` — how the neutral slots map back to true channels — is written to an
``_unblind/`` directory OUTSIDE every bundle, so it is never shipped to raters. On return, feed it to
:func:`neurodent.results.scoring.unblind` to put labels back in order.

Run from the REPO ROOT (dataset ``extract_func`` paths resolve relative to the CWD)::

    # cheap discovery check first -- surfaces stale configs without loading any data (dev box is fine):
    uv run python scripts/labeling/build_cohort_bundle.py --dataset arx_parv --dry-run

    # real loads are heavy (EDF->bin conversion) -- submit on the cluster via the sbatch wrapper,
    # which forwards all flags and writes a detailed slurm-*.out log:
    sbatch scripts/labeling/run_cohort_bundle.sbatch --dataset arx_parv --animal 29 --n-per-animal 40
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import logging
import sys
from pathlib import Path

# Import the sibling labeling scripts (render_context.py / build_rater_bundle.py) by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_context as R  # noqa: E402
import build_rater_bundle as B  # noqa: E402

from neurodent import constants  # noqa: E402
from neurodent.core.utils import set_temp_directory, resolve_channel, resolve_channels  # noqa: E402
from neurodent.loading.discovery import FileDiscoverer  # noqa: E402
from neurodent.workflow.utils import (  # noqa: E402
    apply_samples_config,
    enumerate_cohort,
    expand_animals_config,
    get_discovery_animal_filter,
    load_animal_recordings,
    load_dataset_config,
    resolve_animal_pattern,
    resolve_samples_config,
)

log = logging.getLogger("build_cohort_bundle")


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


def _montage_key(montage):
    """Short, BLINDED, stable directory/bundle name for a channel montage.

    Must NOT contain channel names: the bundle name is baked into the rater HTML (``__BUNDLE__``), so
    anatomy in it would defeat the per-channel blinding. Keyed by channel count + a hash of the sorted
    montage; the human-readable channel list lives only in the experimenter-side cohort manifest.
    """
    digest = hashlib.sha1("|".join(montage).encode()).hexdigest()[:8]
    return f"m{len(montage)}_{digest}"


def _discovery_for(samples_config, config, animal_id):
    """Resolve ``(discovery_pattern, discovery_filter, is_joint)`` for one animal.

    The cheap half of :func:`load_animal_recordings` (no LRO construction / no MNE intermediate); it
    reuses the same ``resolve_animal_pattern`` / ``get_discovery_animal_filter`` the loader uses, so the
    dry-run discovers exactly what the real run would.
    """
    data_root = samples_config.get("data_root", samples_config.get("data_parent_folder", ""))
    analysis_config = config["analysis"]["war_generation"]
    overrides = samples_config.get("_animal_overrides", {}).get(animal_id, {})
    pattern = overrides.get("pattern", analysis_config.get("pattern"))
    if pattern is None:
        raise KeyError("no 'pattern' configured for war_generation")
    is_joint = animal_id in samples_config.get("_animal_channel_subsets", {})
    discovery_pattern = resolve_animal_pattern(pattern, animal_id, data_root=str(data_root))
    filt = get_discovery_animal_filter(animal_id, is_joint, samples_config.get("_animal_groups", {}))
    return discovery_pattern, filt, is_joint


def dryrun_animal(samples_config, config, animal_id):
    """Discovery-only report for one animal; never raises (captures errors into ``note``)."""
    rec = {"animal": animal_id, "n_files": 0, "n_sessions": 0, "n_datetimes": None,
           "is_joint": False, "channels_ok": None, "dt_ok": True, "note": ""}
    # A per-animal manual_datetime must line up with the discovered sessions, or loading raises: a LIST
    # by count, a session-keyed DICT by exact keys.
    dts = samples_config.get("manual_datetimes", {}).get(animal_id)
    rec["n_datetimes"] = len(dts) if isinstance(dts, (list, dict)) else None
    try:
        pattern, filt, is_joint = _discovery_for(samples_config, config, animal_id)
        rec["is_joint"] = is_joint
        items = FileDiscoverer(pattern).discover(animal=filt)
        # Honor skip_sessions (dataset-level + per-animal) the same way load_animal_recordings does,
        # so counts match what a real load sees.
        skip = list(config["analysis"]["war_generation"].get(
            "skip_sessions", config["analysis"]["war_generation"].get("skip_days", [])))
        skip += list(samples_config.get("_animal_overrides", {}).get(animal_id, {}).get("skip_sessions", []))
        if skip:
            items = [it for it in items
                     if not any(fnmatch.fnmatch(it.metadata.get("session", "unknown"), p) for p in skip)]
        rec["n_files"] = len(items)
        sessions = {it.metadata.get("session", "unknown") for it in items}
        rec["n_sessions"] = len(sessions)
        if is_joint:  # only joint animals carry an explicit channel subset we can check without loading
            subset = samples_config["_animal_channel_subsets"][animal_id]
            try:
                for c in subset:
                    resolve_channel(c)
                rec["channels_ok"] = True
            except ValueError as e:
                rec["channels_ok"] = False
                rec["note"] = f"channel does not resolve: {e}"
        if isinstance(dts, list) and rec["n_sessions"] and len(dts) != rec["n_sessions"]:
            rec["dt_ok"] = False
            rec["note"] = (f"sessions({rec['n_sessions']}) != datetimes({len(dts)}) "
                           f"[loading will raise]; " + rec["note"]).strip("; ")
        elif isinstance(dts, dict) and rec["n_sessions"]:
            missing, extra = sorted(sessions - set(dts)), sorted(set(dts) - sessions)
            if missing or extra:
                rec["dt_ok"] = False
                rec["note"] = (f"datetime-dict keys != sessions (missing={missing[:2]}, extra={extra[:2]}) "
                               f"[loading will raise]; " + rec["note"]).strip("; ")
    except Exception as e:  # discovery / pattern errors -> report, do not crash the whole cohort
        rec["note"] = f"{type(e).__name__}: {e}"
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


def render_animal(samples_config, config, animal_id, out_root, *, n_per_animal, seed, blind_seed,
                  bundles):
    """Load one animal, blind-render its windows into the right montage bundle. Returns keymap rows."""
    genotype = constants.ANIMAL_METADATA.get(animal_id, {}).get("genotype", "Unknown")
    channel_subset = samples_config.get("_animal_channel_subsets", {}).get(animal_id)
    ao = load_animal_recordings(
        samples_config, config, [("", animal_id, "")], animal_id,
        channel_subset=channel_subset, logger=log,
    )
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
        montage = tuple(sorted(set(true_names)))
        bundle = bundles.setdefault(montage, {"dir": out_root / _montage_key(montage), "started": False})
        recording = f"{animal_id}__{i}"
        perm, display_names, keymap = R.blind_channels(true_names, f"{blind_seed}:{recording}")
        try:
            rows = R.render_lro(
                lro, bundle["dir"], n_select=count, recording=recording,
                append=bundle["started"], seed=seed,
                channel_perm=perm, display_names=display_names,
            )
        except ValueError as e:  # e.g. select_random cannot draw enough distinct windows in this session
            log.warning(f"{recording}: skipped ({e})")
            continue
        bundle["started"] = True
        for k in keymap:
            keymap_rows.append({"recording": recording, "slot": k["slot"], "channel": k["channel"],
                                "animal": animal_id, "genotype": genotype})
        log.info(f"{recording}: rendered {len(rows)} window(s), {len(true_names)} ch "
                 f"-> {bundle['dir'].name}")
    return keymap_rows


def _write_keymap(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["recording", "slot", "channel", "animal", "genotype"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_cohort(dataset, out_root, *, animal=None, n_per_animal=40, seed=0, blind_seed=0,
                 dry_run=False, make_bundle=True):
    """Enumerate a dataset's animals and build blinded per-montage rater bundles.

    Returns a process exit code: 0 on success, 1 if any animal failed discovery (dry-run) or loading.
    """
    _require_repo_root()
    config, samples_config = _prepare(dataset)
    animals = [animal] if animal else enumerate_cohort(samples_config)
    log.info(f"dataset={dataset}  animals={len(animals)}  n_per_animal={n_per_animal} "
             f"seed={seed} blind_seed={blind_seed}")

    if dry_run:
        reports = [dryrun_animal(samples_config, config, a) for a in animals]
        _print_dryrun(reports)
        bad = [r for r in reports if r["n_files"] == 0 or r["channels_ok"] is False or not r["dt_ok"]]
        if bad:
            print(f"\n{len(bad)} animal(s) look broken (0 files, unresolved channels, or "
                  f"session/datetime mismatch): {[r['animal'] for r in bad]}")
        return 1 if bad else 0

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bundles, all_keymap, failures = {}, [], []
    for a in animals:
        try:
            all_keymap += render_animal(samples_config, config, a, out_root,
                                        n_per_animal=n_per_animal, seed=seed, blind_seed=blind_seed,
                                        bundles=bundles)
        except Exception as e:  # one stale animal must not sink the cohort
            log.error(f"{a}: FAILED -- {type(e).__name__}: {e}")
            failures.append({"animal": a, "error": f"{type(e).__name__}: {e}"})

    # Keymap + cohort manifest live OUTSIDE the bundle dirs so build() never ships them to raters.
    unblind_dir = out_root / "_unblind"
    _write_keymap(unblind_dir / "keymap.csv", all_keymap)
    manifest = {
        "dataset": dataset, "seed": seed, "blind_seed": blind_seed, "n_per_animal": n_per_animal,
        "animals": animals,
        "montages": {_montage_key(m): {"dir": str(b["dir"]), "channels": list(m), "started": b["started"]}
                     for m, b in bundles.items()},
        "failures": failures,
    }
    (unblind_dir / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    zips = []
    if make_bundle:
        for montage, b in bundles.items():
            if b["started"]:
                zips.append(B.build(b["dir"], name=f"{dataset}_{_montage_key(montage)}"))
    log.info(f"built {len(zips)} bundle(s); keymap -> {unblind_dir / 'keymap.csv'}")
    if failures:
        log.warning(f"{len(failures)} animal(s) failed: {[f['animal'] for f in failures]}")
    return 1 if failures else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset name (config/datasets/<name>.yaml stem)")
    p.add_argument("--animal", default=None, help="single animal id to render (default: whole cohort)")
    p.add_argument("--n-per-animal", type=int, default=40, help="windows drawn per animal (spread across its sessions)")
    p.add_argument("--seed", type=int, default=0, help="window-selection seed (reproducible draw)")
    p.add_argument("--blind-seed", type=int, default=0, help="channel-shuffle seed (reproducible blinding)")
    p.add_argument("--out", default=None, help="output root (default: results/labeling/<dataset>)")
    p.add_argument("--dry-run", action="store_true", help="discovery-only check; no data load, no render")
    p.add_argument("--no-bundle", action="store_true", help="render + keymap but do not build the zip(s)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_root = args.out or f"results/labeling/{args.dataset}"
    return build_cohort(
        args.dataset, out_root, animal=args.animal, n_per_animal=args.n_per_animal,
        seed=args.seed, blind_seed=args.blind_seed, dry_run=args.dry_run,
        make_bundle=not args.no_bundle,
    )


if __name__ == "__main__":
    sys.exit(main())
