#!/usr/bin/env python
"""Score artifact detectors at fixed settings against the dev-split human labels.

The rater CSVs go through the same path as ``score_bundle.py`` (``ingest``, ``unblind``, ``consensus``).
Each detector's per (window, channel) decision is then re-derived on the recordings the raters saw, and
precision, recall and F1 are reported at (recording, window, channel) granularity. Thresholds are the
config values and nothing is tuned here; tuning and selection happen in ``select_detectors.py``.

Recordings are loaded through the shared ``build_cohort_bundle.iter_cohort_animals`` and
``load_cohort_animal`` seam, the same ``load_animal_recordings`` call and ``channel_subset`` the bundle
used. Features come from ``AnimalAnalyzer.compute_windowed_analysis(window_s=FRAG_S,
apply_notch_filter=True)``, the WAR-generation path.

Labels name a recording by its July position, ``"{animal}__{i}"``. Positions can move under the current
loader, so recordings are matched to labels through ``config/labeling/recording_map.csv``, written by
``compare_window_hashes.py`` after the sample-hash check. ``--positional-recordings`` skips the map, for
bundles rendered by this code. A labeled window is a ``FRAG_S``-second fragment index (``t_start_s = window
* FRAG_S``); ``score_keep_mask`` matches each labeled cell to the detector row whose start time is within
``FRAG_S/2``, so ``n_uncovered`` should be 0.

This script reads dev labels only: it refuses animals outside ``split.json``'s dev list. The test split is
scored once, by ``score_test.py``.

Feature extraction over whole recordings is heavy; run it on the cluster::

    uv run python scripts/labeling/score_detectors.py --all \
        JD=scripts/labeling/labels/labels_cohort4strains_JD_dev.csv \
        JL=scripts/labeling/labels/labels_cohort4strains_JL_dev.csv \
        Yong=scripts/labeling/labels/labels_cohort4strains_Yong_dev.csv
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cohort_bundle as C  # noqa: E402  (the shared cohort load seam)
import render_context as R  # noqa: E402  (FRAG_S, the one definition of the window length)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.scoring import consensus, ingest, score_keep_mask, score_mask, unblind  # noqa: E402

log = logging.getLogger("score_detectors")

CONFIG_KEYMAP = REPO / "config/labeling/keymap.csv"
CONFIG_SPLIT = REPO / "config/labeling/split.json"
CONFIG_RECORDING_MAP = REPO / "config/labeling/recording_map.csv"

# Files whose content decides every score; their hashes go into each output's provenance.
CODE_FILES = [
    REPO / "src/neurodent/results/autoreject_detector.py",
    REPO / "src/neurodent/results/scoring.py",
    REPO / "scripts/labeling/score_detectors.py",
    REPO / "scripts/labeling/extract_dev_features.py",
    REPO / "scripts/labeling/select_detectors.py",
    REPO / "scripts/labeling/build_cohort_bundle.py",
]

# Fragment detectors at their config/config.yaml fragment_filter_config settings: name: (WAR keep-mask
# accessor, params). Each accessor returns a (W, C) mask over the whole WAR, True meaning keep; it is
# sliced per recording below. LOF works per channel and is opt-in (it needs compute_bad_channels).
FRAGMENT_DETECTORS = {
    "logrms_range": ("get_filter_logrms_range", {"z_range": 3}),
    "high_rms": ("get_filter_high_rms", {"max_rms": 500}),
    "low_rms": ("get_filter_low_rms", {"min_rms": 50}),
    "high_beta": ("get_filter_high_beta", {"max_beta_prop": 0.4}),
}
# Enough features for the four fragment filters: rms drives logrms_range, high_rms and low_rms;
# psdband and psdtotal drive high_beta. Not "all", so extraction stays cheap.
FEATURES = ["rms", "psdband", "psdtotal"]


def available_cpus() -> int:
    """Return the CPUs this process may use: the affinity mask on Linux, the machine's count elsewhere."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def check_split(animals, split_path, which):
    """Refuse labels that come from outside one partition of the locked split.

    Args:
        animals: Animal ids present in the labels.
        split_path: Path to ``split.json``.
        which: ``"dev"`` or ``"test"``.

    Returns:
        The set of animal ids in that partition.

    Raises:
        SystemExit: If any animal is outside the partition.
    """
    split = json.loads(Path(split_path).read_text())
    allowed = set(split[which])
    outside = set(animals) - allowed
    if outside:
        raise SystemExit(
            f"labels include animals outside split['{which}']: {sorted(outside)}. Refusing, since this would "
            f"mix the partitions; pass the _{which} label files."
        )
    return allowed


class RecordingMap:
    """Translate a loaded recording to the label key ``"{animal}__{i}"`` the raters' files use.

    With a map file, a recording is identified by its animalday, falling back to (animal, current
    position) when the map row has no animalday; recordings missing from the map are not scored. Without a
    map (``positional``), the current position is assumed to equal the labeled one, which holds only for
    bundles rendered by this code.

    Args:
        path: ``recording_map.csv`` with columns july_recording, animal, animalday and port_position, or
            None with ``positional``.
        positional: Use positions directly.

    Raises:
        SystemExit: If neither a readable map nor ``positional`` is given.
    """

    def __init__(self, path=None, positional=False):
        self.positional = positional
        self.by_animalday, self.by_position = {}, {}
        if positional:
            return
        if path is None or not Path(path).exists():
            raise SystemExit(
                f"recording map not found: {path}. Run the sample-hash check (hash_labeled_windows.py, then "
                "compare_window_hashes.py) to write it, or pass --positional-recordings for a bundle "
                "rendered by this code."
            )
        rmap = pd.read_csv(path)
        missing = {"july_recording", "animal", "animalday", "port_position"} - set(rmap.columns)
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        for r in rmap.itertuples(index=False):
            if isinstance(r.animalday, str) and r.animalday:
                self.by_animalday[(str(r.animal), r.animalday)] = r.july_recording
            if not pd.isna(r.port_position):
                self.by_position[(str(r.animal), int(r.port_position))] = r.july_recording

    def label(self, animal_id, position, animalday=None):
        """Return the label key of the recording at ``position`` with ``animalday``, or None if unmapped."""
        if self.positional:
            return f"{animal_id}__{position}"
        if animalday is not None and (animal_id, animalday) in self.by_animalday:
            return self.by_animalday[(animal_id, animalday)]
        return self.by_position.get((animal_id, int(position)))


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(inputs, **params):
    """Describe what produced an output: inputs and code by sha256, the git commit, and run parameters.

    Args:
        inputs: ``{name: path}`` of input files (rater CSVs, keymap, split, recording map, features).
        **params: Run parameters to record (rule, raters, seeds, caps, flags).

    Returns:
        A JSON-serializable dict.
    """
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "inputs": {name: {"path": str(p), "sha256": _sha256(p)} for name, p in inputs.items() if p is not None},
        "code": {str(p.relative_to(REPO)): _sha256(p) for p in CODE_FILES if p.exists()},
        "params": params,
    }


def _prf(tn, fp, fn, tp):
    """Return precision, recall and F1 from a pooled confusion, 0 where undefined."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score_animal(war, ao, animal_id, consensus_df, detectors=FRAGMENT_DETECTORS, rmap=None):
    """Score each fragment detector on one animal's WAR against the consensus, per recording.

    ``war.channel_abbrevs`` are the canonical channel names the consensus is keyed on (``score_keep_mask``
    matches by name, so column order does not matter). The recording at position ``i`` is the WAR slice
    with ``animalday == ao.animaldays[i]``, scored with ``grid_times`` set to the slice's fragment starts so
    labeled window ``w`` lines up with detector row ``w``.

    Args:
        war: The animal's WindowAnalysisResult.
        ao: The animal's AnimalOrganizer.
        animal_id: Animal id.
        consensus_df: Output of :func:`build_consensus`.
        detectors: ``{name: (accessor, params)}``.
        rmap: A :class:`RecordingMap`; None uses positions.

    Returns:
        A list of per (recording, detector) result dicts.
    """
    rmap = rmap or RecordingMap(positional=True)
    ch_names = list(war.channel_abbrevs)
    animaldays = war.result["animalday"].to_numpy()
    recs = set(consensus_df["recording"])
    masks = {name: getattr(war, method)(**params) for name, (method, params) in detectors.items()}

    rows = []
    for i, animalday in enumerate(ao.animaldays):
        recording = rmap.label(animal_id, i, animalday)
        if recording is None or recording not in recs:
            continue
        rowsel = animaldays == animalday
        n = int(rowsel.sum())
        if n == 0:
            log.warning(f"{recording}: no WAR fragments for animalday {animalday!r}; skipped")
            continue
        grid_times = np.arange(n) * R.FRAG_S
        for name, keep in masks.items():
            res = score_keep_mask(keep[rowsel], ch_names, consensus_df, recording,
                                  grid_times=grid_times, frag_s=R.FRAG_S, strict=False)
            rows.append({"recording": recording, "animalday": animalday, "detector": name, **res})
    return rows


def autoreject_grids(lro, needed_windows=None, *, max_fit_fragments=adr.MAX_FIT_FRAGMENTS, seed=0, parallel=True):
    """Flag one recording's pool under the 6 named autoreject configurations.

    Fragments come from ``adr.build_fit_epochs``, shared with ``extract_dev_features.py`` so both fit on
    the same fragments for the same cap and seed.

    Returns:
        ``({config: (n_pool, C) bool, True meaning reject}, ch_names, grid_times)``.
    """
    X, ch_names, fs, grid_times = adr.build_fit_epochs(
        lro, needed_windows, fragment_len_s=R.FRAG_S, max_fit_fragments=max_fit_fragments, seed=seed)
    grids = adr.compute_masks(X, fs, parallel=parallel)
    return grids, ch_names, grid_times


def score_animal_autoreject(ao, animal_id, consensus_df, *, max_fit_fragments=adr.MAX_FIT_FRAGMENTS, seed=0,
                            rmap=None):
    """Score the 6 named autoreject configurations per recording against the consensus.

    The detector is fit per recording, so each recording gets its own grids. They mean True for reject and
    go through :func:`score_mask` directly.

    Returns:
        A list of per (recording, detector) result dicts.
    """
    rmap = rmap or RecordingMap(positional=True)
    recs = set(consensus_df["recording"])
    rows = []
    for i, lro in enumerate(ao.long_recordings):
        animalday = ao.animaldays[i] if i < len(ao.animaldays) else None
        recording = rmap.label(animal_id, i, animalday)
        if recording is None or recording not in recs:
            continue
        needed = consensus_df.loc[consensus_df["recording"] == recording, "window"].unique()
        try:
            grids, ch_names, grid_times = autoreject_grids(
                lro, needed, max_fit_fragments=max_fit_fragments, seed=seed)
        except Exception as e:
            log.error(f"{recording}: autoreject failed: {type(e).__name__}: {e}")
            continue
        for name, grid in grids.items():
            res = score_mask(grid, ch_names, consensus_df, recording,
                             grid_times=grid_times, frag_s=R.FRAG_S, strict=False)
            rows.append({"recording": recording, "animalday": animalday, "detector": name, **res})
    return rows


def build_consensus(manifests, keymap_path, rule):
    """Return the consensus ground truth from rater CSVs and the keymap (the ``score_bundle.py`` path)."""
    long_df = ingest(manifests)
    keymap = pd.read_csv(keymap_path)
    restored = unblind(long_df, keymap)
    return consensus(restored, rule=rule)


def parse_raters(ap, specs):
    """Return ``{rater: path}`` from ``rater=csv`` arguments, erroring through ``ap`` on a bad one."""
    manifests = {}
    for spec in specs:
        if "=" not in spec:
            ap.error(f"expected rater=csv, got {spec!r}")
        rater, path = spec.split("=", 1)
        p = Path(path).expanduser()
        if not p.exists():
            ap.error(f"{rater}: no such file {p}")
        manifests[rater] = p
    return manifests


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raters", nargs="+", metavar="rater=csv", help="one or more rater_id=exported_csv (dev files)")
    ap.add_argument("--keymap", type=Path, default=CONFIG_KEYMAP,
                    help="slot to channel unblinding key (default: config/labeling/keymap.csv)")
    ap.add_argument("--split", type=Path, default=CONFIG_SPLIT, help="locked split (default: config/labeling)")
    ap.add_argument("--recording-map", type=Path, default=CONFIG_RECORDING_MAP,
                    help="labeled recording to current recording map from compare_window_hashes.py")
    ap.add_argument("--positional-recordings", action="store_true",
                    help="match recordings by position, only for a bundle rendered by this code")
    ap.add_argument("--dataset", nargs="+", default=None,
                    help="dataset name(s) whose animals to score (config/datasets/<name>.yaml)")
    ap.add_argument("--all", action="store_true",
                    help=f"scan all real strains ({' '.join(C.REAL_STRAINS)})")
    ap.add_argument("--rule", default="majority", choices=["majority", "unanimous", "any"])
    ap.add_argument("--lof", action="store_true",
                    help="also score the LOF bad-channel detector (needs compute_bad_channels; slower)")
    ap.add_argument("--lof-threshold", type=float, default=2.5)
    ap.add_argument("--autoreject", action="store_true",
                    help="also score the 6 named autoreject configurations; builds raw fragments per recording")
    ap.add_argument("--ar-fit-max-fragments", type=int, default=adr.MAX_FIT_FRAGMENTS,
                    help="autoreject fit-pool cap per recording (labeled fragments always included; 0 or "
                         "negative uses the whole recording)")
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="match the bundle's --limit-per-dataset so the same animals are loaded")
    ap.add_argument("--seed", type=int, default=0, help="match the bundle's --seed for --limit selection")
    ap.add_argument("--workers", type=int, default=None,
                    help="dask process workers for feature extraction (default: all cores allocated to the "
                         "job); extraction over whole recordings is the cost")
    ap.add_argument("--chunk-duration-s", type=float, default=3600,
                    help="seconds of data buffered per chunk during dask extraction (matches the pipeline)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write per (recording, detector) rows to this CSV, with a .provenance.json beside it")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    C._require_repo_root()

    datasets = list(C.REAL_STRAINS) if args.all else args.dataset
    if not datasets:
        ap.error("provide --dataset <name ...> or --all")
    manifests = parse_raters(ap, args.raters)
    if not args.keymap.exists():
        ap.error(f"keymap not found: {args.keymap}")

    cons = build_consensus(manifests, args.keymap, args.rule)
    animals_wanted = {r.rsplit("__", 1)[0] for r in cons["recording"].unique()}
    check_split(animals_wanted, args.split, "dev")
    rmap = RecordingMap(None if args.positional_recordings else args.recording_map,
                        positional=args.positional_recordings)
    print(f"consensus: {len(cons)} labelled cells over {cons['recording'].nunique()} recordings "
          f"({len(animals_wanted)} animals); raters={list(manifests)}")

    # Dask spreads feature extraction over the job's cores, as the pipeline does; the serial path takes
    # about 1.6 s per fragment. The dashboard is off.
    n_workers = args.workers or available_cpus()
    log.info(f"starting dask LocalCluster with {n_workers} process workers (dashboard off)")

    all_rows = []
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        # Workers import spikeinterface and neurodent on start (about 15 s) and register gradually. Wait for
        # the full pool, or the first animal's work spreads over only the workers registered so far.
        try:
            client.wait_for_workers(n_workers, timeout=180)
        except Exception as e:
            log.warning(f"only {len(client.scheduler_info()['workers'])}/{n_workers} workers up after "
                        f"180 s ({type(e).__name__}); proceeding with what registered")
        log.info(f"dask cluster up: {len(client.scheduler_info()['workers'])} workers")
        for ds, samples_config, config, animal_id in C.iter_cohort_animals(
            datasets, limit_per_dataset=args.limit_per_dataset, seed=args.seed
        ):
            if animal_id not in animals_wanted:
                continue
            try:
                ao = C.load_cohort_animal(samples_config, config, animal_id)
                az = AnimalAnalyzer(ao)
                if args.lof:
                    az.compute_bad_channels(lof_threshold=args.lof_threshold, lof_chunk_duration_s=60)
                # window_s pinned to the labeling FRAG_S and the notch on, as for the rendered windows.
                war = az.compute_windowed_analysis(
                    FEATURES, window_s=R.FRAG_S, apply_notch_filter=True,
                    multiprocess_mode="dask", chunk_duration_s=args.chunk_duration_s)
                detectors = dict(FRAGMENT_DETECTORS)
                if args.lof:
                    lof_dict = war.get_bad_channels_by_lof_threshold(args.lof_threshold)
                    detectors["lof"] = ("get_filter_reject_channels_by_recording_session",
                                        {"bad_channels_dict": lof_dict})
                all_rows += score_animal(war, ao, animal_id, cons, detectors, rmap=rmap)
                if args.autoreject:
                    cap = args.ar_fit_max_fragments if args.ar_fit_max_fragments > 0 else None
                    all_rows += score_animal_autoreject(ao, animal_id, cons, max_fit_fragments=cap,
                                                        seed=args.seed, rmap=rmap)
                log.info(f"{animal_id}: scored {len(ao.animaldays)} recording(s)")
            except Exception as e:
                log.error(f"{animal_id}: failed: {type(e).__name__}: {e}")

    if not all_rows:
        print("no recordings scored (no overlap between the cohort and the consensus recordings).")
        return 1

    df = pd.DataFrame(all_rows)
    scored = set(df["recording"])
    unscored = sorted(set(cons["recording"]) - scored)
    if unscored:
        print(f"{len(unscored)} labelled recording(s) not scored: {unscored}")
    if args.out:
        df.to_csv(args.out, index=False)
        inputs = {f"labels_{r}": p for r, p in manifests.items()}
        inputs.update(keymap=args.keymap, split=args.split,
                      recording_map=None if args.positional_recordings else args.recording_map)
        prov = provenance(inputs, script="score_detectors", rule=args.rule, raters=sorted(manifests),
                          positional_recordings=args.positional_recordings, lof=args.lof,
                          lof_threshold=args.lof_threshold, autoreject=args.autoreject,
                          ar_fit_max_fragments=args.ar_fit_max_fragments, seed=args.seed)
        Path(str(args.out) + ".provenance.json").write_text(json.dumps(prov, indent=2))
        print(f"wrote per-recording rows to {args.out}")

    # Pooled per-detector metrics from summed confusions (each cell counts once).
    order = list(FRAGMENT_DETECTORS) + (["lof"] if args.lof else []) + \
        (adr.CONFIG_NAMES if args.autoreject else [])
    present = [n for n in order if (df["detector"] == n).any()]
    print(f"\n{'detector':<22}{'n':>7}{'uncov':>7}{'prec':>8}{'recall':>8}{'f1':>8}   confusion[tn fp / fn tp]")
    print("-" * 84)
    for name in present:
        sub = df[df["detector"] == name]
        conf = np.zeros((2, 2), dtype=int)
        for c in sub["confusion"].dropna():
            conf += np.array(c, dtype=int)
        tn, fp, fn, tp = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
        p, r, f1 = _prf(tn, fp, fn, tp)
        n = int(sub["n"].fillna(0).sum())
        uncov = int(sub["n_uncovered"].fillna(0).sum())
        print(f"{name:<22}{n:>7}{uncov:>7}{p:>8.3f}{r:>8.3f}{f1:>8.3f}   [{tn} {fp} / {fn} {tp}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
