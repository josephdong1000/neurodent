#!/usr/bin/env python
"""Hash the samples behind every labeled window, to check that a loader still reads what the raters saw.

The labeled windows were rendered in July by the July loader. ``config/labeling/keymap.csv`` names a
recording only by its position in the animal's recording list (``"{animal}__{i}"``), a window by its 5 s
fragment index, and a channel by its canonical name. If a loader change moves recordings, renames channels
or shifts samples, the labels attach to other data without any error. This script loads every labeled
animal with whichever neurodent tree is imported. For each recording the tree produces, and each labeled
(window, channel) of that animal whose window fits inside it, it writes the sha256 of the raw samples and
of the notch-filtered view the rater saw, with the RMS, mean and first samples, so that a numeric-only
change can be told from different samples. ``compare_window_hashes.py`` joins the outputs of two trees.

Each output row is one labeled cell in one recording. ``status`` is ok when the cell's samples were hashed,
channel_absent or channel_duplicate when no column or several columns carry its canonical channel, and
read_error when the recording could not be opened, the window could not be read or the column could not be
hashed. A read_error row holds the exception in ``error`` and no hash, so ``compare_window_hashes.py`` counts
it as not hashed, while the rows of the animal's other recordings and windows are kept. A recording that cannot
be opened gets a read_error row for each labeled cell whose window fits inside it, or for every labeled
cell of the animal when its length is unknown. ``first_file`` is the recording's first source file. A
recording split out of a multi-animal file has no source item, so its construction arguments are searched
instead, and for an MNE-loaded file they name the intermediate in ``--temp-dir``. A path inside
``--temp-dir`` is written as ``<temp>/`` plus its path relative to that directory (for an intermediate, its
file name), so the July and port outputs agree although their temp directories differ. Every row also
records the code the run used: ``git_commit`` is the working tree's ``git rev-parse HEAD``, ``src_dirty``
whether ``git status --porcelain -- src`` lists anything there, and ``src_tree`` the git object id of ``src``
at that commit (``git rev-parse HEAD:src``). Each is empty when git cannot tell or the working directory is
not the top level of a git checkout or worktree (a plain copy inside another repository would otherwise get
that repository's commit). ``compare_window_hashes.py`` carries them into the recording map. The src tree
names the loader code apart from the commit: it stays the same when only files outside ``src`` change, such
as the committed map, and ``extract_dev_features.py`` checks its own src tree against the map's
``port_src_tree``. The map takes a run's src tree only when src_dirty is False, since with uncommitted changes
under ``src`` the tree is not the code that ran. A port run without a recorded src tree and a clean ``src``
therefore gives a map that ``extract_dev_features.py`` refuses.

``--all-windows`` scans instead of hashing labeled cells: for the animals named with ``--animal`` it hashes
every window of every recording, every column, into rows with status scan. A scan row has the raw sha256,
statistics and first samples of one column in one window, ``channel`` is that column's canonical name, and
``july_recording`` and ``view_sha256`` are empty. A window that cannot be read gives a scan_read_error row
per column, and a recording that cannot be opened one scan_read_error row without window or channel.
``compare_window_hashes.py --port-scan`` looks up different and missing cells there, to tell samples that
moved by whole windows from samples that changed. A scan reads whole recordings: each animal costs a full
load, as in a hash run, plus a read of every sample, and writes one row per window and column, about
600 bytes each, so a day at 1000 Hz with 10 channels gives 172,800 rows, about 100 MB.

The same file runs under the July tree (commit 41353542) and under the port, so it imports only APIs both
trees have and none of its sibling scripts. The working directory selects the tree's dataset configs and
reader paths, and ``PYTHONPATH`` selects its ``neurodent``; the script refuses to run when the imported
``neurodent`` is not inside the working directory. Each run needs its own empty ``--temp-dir``: MNE
intermediates are cached there under the first source file's name, so a cache left by the other tree
would be read back as this tree's samples. The loader's ``cache_policy="force_regenerate"`` cannot replace
the empty directory: in both trees it deletes the intermediate right after opening it, and an EDF
intermediate is reopened on every read. The July bundle itself read intermediates from the configured
``temp_directory``, which may have been written by older code; ``--reuse-temp-dir`` replays a preserved
copy of that cache instead of regenerating it.

By default scoring also reads MNE intermediates from the configured temp directory, which may hold older
files. The scoring runs, ``extract_dev_features.py`` and ``score_detectors.py``, must therefore be given the
port hash run's ``--temp-dir`` through their own ``--temp-dir`` option, so that they read the intermediates
this check validated.

Rows are written to ``<out>.partial``. Each animal's rows are appended and flushed as soon as the animal
finishes, so a run stopped by a time or memory limit keeps every animal it finished. A scan writes its rows
as each recording is read instead, so its ``<out>.partial`` may end inside an animal. When every requested
animal has been tried, the file is renamed to ``<out>``; a file still named ``<out>.partial`` comes from a
run that did not finish. An ``<out>`` left by an earlier run is removed when writing starts. The exit
status is 1 when an animal could not be loaded or a row has status read_error or scan_read_error, and 0
otherwise.

Run on the cluster from the root of each tree. The keymap, manifest and fragment-count table default to
the copies in this script's repository; the keymap and manifest are identical in both trees::

    cd <july-worktree>
    PYTHONPATH=$PWD/src python <port-worktree>/scripts/labeling/hash_labeled_windows.py --tree july \\
        --out <scratch>/hashes_july.csv --temp-dir <scratch>/hash_tmp_july
    cd <port-worktree>
    PYTHONPATH=$PWD/src python scripts/labeling/hash_labeled_windows.py --tree port \\
        --out <scratch>/hashes_port.csv --temp-dir <scratch>/hash_tmp_port

A scan of the port for some animals, with its own empty temp directory::

    cd <port-worktree>
    PYTHONPATH=$PWD/src python scripts/labeling/hash_labeled_windows.py --tree port --all-windows \\
        --animal <id> [<id> ...] --out <scratch>/scan_port.csv --temp-dir <scratch>/scan_tmp_port
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import neurodent
from neurodent.analysis.long_recording_analyzer import LongRecordingAnalyzer
from neurodent.core.utils import resolve_channels
from neurodent.core.utils.paths import set_temp_directory
from neurodent.workflow.utils import (
    apply_samples_config,
    enumerate_cohort,
    expand_animals_config,
    load_animal_recordings,
    load_dataset_config,
    resolve_samples_config,
)

log = logging.getLogger("hash_labeled_windows")

REPO = Path(__file__).resolve().parents[2]
CONFIG_KEYMAP = REPO / "config/labeling/keymap.csv"
CONFIG_MANIFEST = REPO / "config/labeling/cohort_manifest.json"
CONFIG_JULY_N_FRAGMENTS = REPO / "config/labeling/july_n_fragments.csv"

FRAG_S = 5.0    # render_context.FRAG_S: a labeled window is one 5 s fragment
N_HEAD = 4      # leading raw samples written per cell
TEMP_TAG = "<temp>"             # stands for the run's --temp-dir in first_file
PARTIAL_SUFFIX = ".partial"     # the output's suffix until the run finishes
SCAN = "scan"                   # status of a scan row with hashed samples (--all-windows)
SCAN_ERROR = "scan_read_error"  # status of a scan row whose samples could not be read

FIELDS = [
    "tree", "animal", "position", "animalday", "date_string", "first_file", "n_files", "n_fragments", "fs",
    "samples_per_fragment", "july_n_fragments", "july_recording", "window", "channel", "column", "status",
    "error", "start_sample", "n_samples", "dtype", "raw_sha256", "view_sha256", "raw_rms", "raw_mean",
    "raw_ptp", *[f"raw_head_{k}" for k in range(N_HEAD)], "channel_names", "canonical_channels",
    "git_commit", "src_dirty", "src_tree",
]


def check_tree():
    """Return the imported ``neurodent`` package directory after checking it belongs to the working tree.

    Returns:
        Path: The package directory.

    Raises:
        SystemExit: If the package is not inside the working directory, or the working directory has no
            ``config/config.yaml``.
    """
    pkg = Path(neurodent.__file__).resolve().parent
    cwd = Path.cwd().resolve()
    log.info(f"neurodent imported from {pkg}; working directory {cwd}")
    if cwd not in pkg.parents:
        raise SystemExit(
            f"neurodent was imported from {pkg}, which is not inside the working directory {cwd}. Run from "
            "the root of the tree under test with PYTHONPATH=$PWD/src, so that the configs, readers and "
            "loader all come from one tree."
        )
    if not (cwd / "config" / "config.yaml").exists():
        raise SystemExit(f"{cwd} has no config/config.yaml; run from the root of the tree under test.")
    local = cwd / "config" / "config.local.yaml"
    log.info(f"config/config.local.yaml {'is' if local.exists() else 'is not'} present in this tree")
    return pkg


def code_version(tree):
    """Return the git commit of a tree, whether its ``src`` directory differs from it, and the id of that ``src``.

    Only the top level of a git checkout or worktree has a commit. git answers for a plain copy inside another
    repository, such as an archive unpacked in an ignored directory, with that repository's commit, so such a
    tree records none.

    Args:
        tree (str | Path): Root of the tree under test.

    Returns:
        dict[str, str | bool]: ``git_commit``, the output of ``git rev-parse HEAD``; ``src_dirty``, True
        when ``git status --porcelain -- src`` lists a modified or untracked file; and ``src_tree``, the
        output of ``git rev-parse HEAD:src``, the value ``score_detectors.src_state`` records for a scoring
        run. Each is empty when git cannot tell or the tree is not the top level of its repository.
    """
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    unknown = {"git_commit": "", "src_dirty": "", "src_tree": ""}
    top = git("rev-parse", "--show-toplevel")
    if not top:
        return unknown
    if Path(top).resolve() != Path(tree).resolve():
        log.warning(f"{tree} is not the top level of a git checkout but lies inside the repository at {top}, "
                    "whose commit does not name this tree's code")
        return unknown
    commit = git("rev-parse", "HEAD")
    src_tree = git("rev-parse", "HEAD:src")
    status = git("status", "--porcelain", "--", "src")
    return {"git_commit": commit or "", "src_dirty": "" if status is None else bool(status),
            "src_tree": src_tree or ""}


def prepare_temp_dir(path, reuse=False):
    """Point neurodent's intermediates at a temp directory, which must be empty unless ``reuse``.

    Args:
        path (str | Path): The directory; created if missing.
        reuse (bool, optional): Accept files already in the directory, and log them. Defaults to False.

    Returns:
        Path: The resolved directory.

    Raises:
        SystemExit: If the directory holds files and ``reuse`` is False.
    """
    path = Path(path).resolve()
    present = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
    if present and not reuse:
        raise SystemExit(
            f"--temp-dir {path} is not empty. Each run needs its own empty temp directory, because MNE "
            "intermediates cached there by an earlier run, possibly of the other tree, would be read back as "
            "this run's samples. Pass --reuse-temp-dir only to replay a preserved cache on purpose."
        )
    for p in present:
        log.warning(f"reusing {p} ({p.stat().st_size} bytes, modified {p.stat().st_mtime:.0f})")
    set_temp_directory(path)
    return path


def read_keymap_cells(path):
    """Return the labeled cells of each animal.

    Args:
        path (str | Path): ``keymap.csv`` with columns recording, window, channel and animal.

    Returns:
        dict[str, dict[tuple[int, str], set[int]]]: Animal to ``{(window, channel): July positions}``, the
        positions being those of the July recordings that labeled the cell.

    Raises:
        ValueError: If a recording label is malformed or disagrees with the row's animal.
    """
    cells = defaultdict(lambda: defaultdict(set))
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            animal, _, position = row["recording"].rpartition("__")
            if animal != row["animal"] or not position.isdigit():
                raise ValueError(f"{path}: recording {row['recording']!r} does not match animal {row['animal']!r}")
            cells[animal][(int(row["window"]), row["channel"])].add(int(position))
    return {a: dict(c) for a, c in cells.items()}


def read_july_n_fragments(path):
    """Return the n_fragments values that reproduce each July recording's window draw.

    Args:
        path (str | Path | None): ``july_n_fragments.csv`` from ``july_fragment_counts.py``, or None.

    Returns:
        dict[str, list[int]]: Recording label to its matching n_fragments values; empty when the file is
        absent.
    """
    if path is None or not Path(path).exists():
        log.info(f"no July fragment-count table at {path}; fragment counts are logged without it")
        return {}
    with open(path, newline="") as fh:
        return {r["recording"]: [int(n) for n in r["matches"].split(";") if n] for r in csv.DictReader(fh)}


def find_datasets(datasets, animals):
    """Find the one dataset among ``datasets`` whose config lists each animal.

    Args:
        datasets (list[str]): Dataset names from the manifest.
        animals (list[str]): Animal ids to place.

    Returns:
        tuple[dict[str, str], dict[str, tuple[dict, dict]], list[dict]]: Animal to dataset; dataset to
        ``(config, samples_config)``; and one failure record per animal found in no dataset or in several.
    """
    prepared, members = {}, {}
    for ds in datasets:
        config = load_dataset_config(ds)
        samples_config = expand_animals_config(resolve_samples_config(config))
        prepared[ds] = (config, samples_config)
        members[ds] = {str(a) for a in enumerate_cohort(samples_config)}
    owner, failures = {}, []
    for a in animals:
        hits = [ds for ds in datasets if a in members[ds]]
        if len(hits) == 1:
            owner[a] = hits[0]
        else:
            error = f"listed in {len(hits)} of the datasets {datasets}: {hits}"
            log.error(f"{a}: {error}")
            failures.append({"animal": a, "error": error})
    return owner, prepared, failures


def cells_in_recording(cells, animal, position, n_fragments):
    """Return the labeled cells of an animal whose window is inside a recording of ``n_fragments``.

    Every labeled window of the animal is tried at every position, because recordings may have moved.

    Args:
        cells (dict[tuple[int, str], set[int]]): ``{(window, channel): July positions}`` for the animal.
        animal (str): Animal id.
        position (int): Position of the recording in this tree.
        n_fragments (int | None): Number of 5 s fragments in the recording, or None when it is unknown,
            which keeps every labeled window.

    Returns:
        list[tuple[int, str, str]]: ``(window, channel, july_recording)`` sorted by window and channel;
        ``july_recording`` is ``"{animal}__{position}"`` when July labeled the cell at this position, else
        empty.
    """
    out = []
    for (w, ch), positions in sorted(cells.items()):
        if 0 <= w and (n_fragments is None or w < n_fragments):
            out.append((w, ch, f"{animal}__{position}" if position in positions else ""))
    return out


def _provenance_path(rec, depth=0):
    """First file path in a SpikeInterface recording's construction arguments, searched depth first."""
    if rec is None or depth > 32:
        return ""
    kwargs = getattr(rec, "_kwargs", None) or {}
    for key in ("file_path", "file_paths", "folder_path"):
        value = kwargs.get(key)
        if isinstance(value, (list, tuple)):
            if value:
                return str(value[0])
        elif value:
            return str(value)
    for key in ("parent_recording", "recording", "recording_list", "recordings"):
        value = kwargs.get(key)
        for parent in value if isinstance(value, (list, tuple)) else [value]:
            if hasattr(parent, "_kwargs"):
                found = _provenance_path(parent, depth + 1)
                if found:
                    return found
    return ""


def tag_temp_path(path, temp_dir):
    """Return a path inside ``temp_dir`` as ``"<temp>/"`` plus its path there, and any other path unchanged.

    The path is compared both as given and with the symlinks of its directory and of ``temp_dir`` resolved,
    so a temp directory named through a symlink still matches. A path that leaves the temp directory through
    ``..`` is not inside it.

    Args:
        path (str): A file path, possibly empty.
        temp_dir (str | Path | None): The run's temp directory, or None to return every path as it is.

    Returns:
        str: The path to record.
    """
    if not path or temp_dir is None:
        return path
    p, t = Path(path), Path(temp_dir)
    for inner, outer in ((p, t), (p.parent.resolve() / p.name, t.resolve())):
        try:
            rel = inner.relative_to(outer)
        except ValueError:
            continue
        if ".." not in rel.parts:
            return "/".join([TEMP_TAG, *rel.parts])
    return path


def first_source_path(lro, temp_dir=None):
    """Return the first source file of a recording.

    A recording split out of a multi-animal file has no ``item``, so the recording's own construction
    arguments are searched instead; for an MNE-loaded file that finds the intermediate, which is named
    after the first source file. Each run has its own temp directory, so a path inside it is written by
    :func:`tag_temp_path` without the directory, and the value does not depend on the run.

    Args:
        lro: A ``LongRecordingOrganizer``.
        temp_dir (str | Path | None, optional): The run's temp directory, where the loader writes MNE
            intermediates; None returns the path as found. Defaults to None.

    Returns:
        str: The path, or empty if none is recorded.
    """
    item = getattr(lro, "item", None)
    if hasattr(item, "get_path_list"):
        paths = list(item.get_path_list())
    elif isinstance(item, (list, tuple)):
        paths = list(item)
    else:
        paths = [] if item is None else [item]
    path = str(paths[0]) if paths else _provenance_path(getattr(lro, "LongRecording", None))
    return tag_temp_path(path, temp_dir)


def _date_string(lro):
    try:
        return lro.get_date_string()
    except (ValueError, AttributeError) as e:
        log.warning(f"no date string: {e}")
        return ""


def _traces_uV(rec):
    """Traces in microvolts; SpikeInterface renamed ``return_scaled`` to ``return_in_uV`` in 0.103."""
    try:
        return rec.get_traces(return_in_uV=True)
    except TypeError:
        return rec.get_traces(return_scaled=True)


def _sha256_f32(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def _num(x):
    return repr(float(x))


def _error_text(e):
    """An exception as one line of text, for the error column."""
    return " ".join(f"{type(e).__name__}: {e}".split())


def _row(info, window, channel, july_recording, **values):
    """A row holding ``info``, the cell's key and ``values``, with every other column of :data:`FIELDS` empty."""
    row = dict.fromkeys(FIELDS, "")
    row.update(info)
    row.update(july_recording=july_recording, window=window, channel=channel, **values)
    return row


def _raw_values(x):
    """The raw sha256, statistics and first samples of one column of a window, as row values."""
    x64 = x.astype(np.float64)
    out = {"raw_sha256": _sha256_f32(x), "raw_rms": _num(np.sqrt(np.mean(x64 ** 2))), "raw_mean": _num(np.mean(x64)),
           "raw_ptp": _num(np.ptp(x64))}
    for k in range(min(N_HEAD, x64.size)):
        out[f"raw_head_{k}"] = _num(x64[k])
    return out


def _hash_cell(raw, view, canonical, ch):
    """Return the status of one cell and, when exactly one column carries its channel, its hashes.

    Args:
        raw (np.ndarray): The window's raw samples in microvolts, shaped (samples, channels).
        view (np.ndarray): The window's notch-filtered samples, the rater view, of the same shape.
        canonical (list[str]): Canonical channel name of each column.
        ch (str): The cell's canonical channel.

    Returns:
        dict: Values of the status and column columns and, when the status is ok, of the hash, statistic and
        first-sample columns.
    """
    cols = [j for j, c in enumerate(canonical) if c == ch]
    if len(cols) != 1:
        return {"status": "channel_absent" if not cols else "channel_duplicate",
                "column": ";".join(str(j) for j in cols)}
    col = cols[0]
    out = {"status": "ok", "column": col, **_raw_values(raw[:, col])}
    out["view_sha256"] = _sha256_f32(view[:, col])
    return out


def _open_recording(lro, info, temp_dir=None):
    """Store a recording's columns in ``info`` and return what reading its windows needs.

    Each value is stored as soon as it is known, so when opening fails part way, ``info`` keeps what was
    learned for the rows that report the failure.

    Args:
        lro: A ``LongRecordingOrganizer``.
        info (dict): Row values of the recording, updated in place.
        temp_dir (str | Path | None, optional): The run's temp directory, for :func:`first_source_path`.
            Defaults to None.

    Returns:
        tuple: The ``LongRecordingAnalyzer``, the samples per fragment, and the channel names, canonical
        channel names and recording channel ids, each a list of str.
    """
    info["animalday"] = getattr(lro, "animalday", "") or ""
    info["date_string"] = _date_string(lro)
    info["first_file"] = first_source_path(lro, temp_dir)
    info["n_files"] = len(getattr(lro, "file_durations", None) or [])
    lan = LongRecordingAnalyzer(lro, fragment_len_s=FRAG_S, apply_notch_filter=True)
    info["n_fragments"] = int(lan.n_fragments)
    info["fs"] = int(lan.f_s)
    spf = int(lro.LongRecording.time_to_sample_index(FRAG_S))
    info["samples_per_fragment"] = spf
    names = [str(c) for c in lro.channel_names]
    info["channel_names"] = ";".join(names)
    canonical = [str(c) for c in resolve_channels(list(lro.channel_names))]
    info["canonical_channels"] = ";".join(canonical)
    ids = [str(c) for c in lro.LongRecording.get_channel_ids()]
    return lan, spf, names, canonical, ids


def hash_recording(tree, animal, position, lro, cells, july_n_fragments, temp_dir=None):
    """Return one row per labeled cell of ``animal`` whose window fits inside one recording.

    A failure is written as rows with status read_error, the error text and no hash, instead of being
    raised, so it costs only the cells it covers. When the recording cannot be opened, those are the cells
    whose window fits inside it, or every cell of the animal when its length is unknown; when a window
    cannot be read, the cells of that window; when a column cannot be hashed, that one cell.

    Args:
        tree (str): Tree label written to every row.
        animal (str): Animal id.
        position (int): Position of the recording in this tree's ``long_recordings``.
        lro: The ``LongRecordingOrganizer`` at that position.
        cells (dict[tuple[int, str], set[int]]): ``{(window, channel): July positions}`` for the animal.
        july_n_fragments (dict[str, list[int]]): Recording label to the n_fragments values that reproduce
            its July window draw.
        temp_dir (str | Path | None, optional): The run's temp directory, for :func:`first_source_path`.
            Defaults to None.

    Returns:
        list[dict]: Rows with the columns in :data:`FIELDS`.
    """
    label = f"{animal}__{position}"
    july_here = july_n_fragments.get(label, [])
    info = {"tree": tree, "animal": animal, "position": position,
            "july_n_fragments": ";".join(str(n) for n in july_here)}
    try:
        lan, spf, names, canonical, ids = _open_recording(lro, info, temp_dir)
    except Exception as e:  # a recording that cannot be opened costs only the cells it covers
        error = _error_text(e)
        log.exception(f"{label}: could not open the recording, {error}")
        spf = info.get("samples_per_fragment")
        covered = cells_in_recording(cells, animal, position, info.get("n_fragments"))
        return [_row(info, w, ch, july_label, status="read_error", error=error,
                     start_sample="" if spf is None else spf * w)
                for w, ch, july_label in covered]
    n_fragments = info["n_fragments"]

    log.info(f"{label}: animalday {info['animalday']!r}, date {info['date_string']!r}, n_fragments {n_fragments}, "
             f"fs {info['fs']}, samples_per_fragment {spf}, {info['n_files']} file(s), first file "
             f"{info['first_file']!r}")
    log.info(f"{label}: channel_names {names}")
    log.info(f"{label}: canonical channels {canonical}")
    if ids != names:
        log.info(f"{label}: recording channel ids {ids}")
    if july_n_fragments:
        if july_here:
            log.info(f"{label}: n_fragments {n_fragments} in this tree; the July draw at this position is "
                     f"reproduced by n_fragments {info['july_n_fragments']}")
        else:
            log.info(f"{label}: n_fragments {n_fragments} in this tree; no July recording had this position")
        same = sorted(r for r, ns in july_n_fragments.items()
                      if r.rpartition("__")[0] == animal and n_fragments in ns)
        log.info(f"{label}: July recordings of {animal} whose draw n_fragments {n_fragments} reproduces: "
                 f"{same if same else 'none'}")

    by_window = defaultdict(list)
    for w, ch, july_label in cells_in_recording(cells, animal, position, n_fragments):
        by_window[w].append((ch, july_label))

    rows = []
    for w, entries in sorted(by_window.items()):
        window = dict(info, start_sample=spf * w)
        try:
            raw = np.asarray(_traces_uV(lro.get_fragment(FRAG_S, w)))
            view = np.asarray(lan.get_fragment_np(w))
            window.update(n_samples=raw.shape[0], dtype=str(raw.dtype))
        except Exception as e:  # a window that cannot be read costs only its own cells
            error = _error_text(e)
            log.exception(f"{label} window {w}: could not read the samples, {error}")
            rows += [_row(window, w, ch, july_label, status="read_error", error=error) for ch, july_label in entries]
            continue
        for ch, july_label in entries:
            row = _row(window, w, ch, july_label)
            try:
                row.update(_hash_cell(raw, view, canonical, ch))
            except Exception as e:  # a column that cannot be hashed costs only its own cell
                error = _error_text(e)
                log.exception(f"{label} window {w}: could not hash channel {ch}, {error}")
                row.update(status="read_error", error=error)
            if row["status"] in ("channel_absent", "channel_duplicate"):
                log.warning(f"{label} window {w}: channel {ch} occurs {canonical.count(ch)} times in {canonical}")
            rows.append(row)
    return rows


def hash_animal(tree, animal, samples_config, config, cells, july_n_fragments, temp_dir=None):
    """Load one animal the way the July bundle did and hash its labeled cells in every recording.

    A recording that cannot be opened, or a window that cannot be read, gives read_error rows (see
    :func:`hash_recording`), and the rows of the other recordings and windows are kept.

    Args:
        tree (str): Tree label written to every row.
        animal (str): Animal id.
        samples_config (dict): Expanded samples config of the animal's dataset, already applied.
        config (dict): The dataset config.
        cells (dict[tuple[int, str], set[int]]): ``{(window, channel): July positions}`` for the animal.
        july_n_fragments (dict[str, list[int]]): Recording label to its July n_fragments matches.
        temp_dir (str | Path | None, optional): The run's temp directory, for :func:`first_source_path`.
            Defaults to None.

    Returns:
        list[dict]: Rows for every recording of the animal.

    Raises:
        Exception: Whatever ``load_animal_recordings`` raises when the animal cannot be loaded at all.
    """
    ao = load_animal(animal, samples_config, config)
    rows = []
    for position, lro in enumerate(ao.long_recordings):
        rows += hash_recording(tree, animal, position, lro, cells, july_n_fragments, temp_dir=temp_dir)
    return rows


def load_animal(animal, samples_config, config):
    """Load one animal the way the July bundle did: one ``load_animal_recordings`` call with its channel subset.

    Args:
        animal (str): Animal id.
        samples_config (dict): Expanded samples config of the animal's dataset, already applied.
        config (dict): The dataset config.

    Returns:
        The animal's ``AnimalOrganizer``.

    Raises:
        Exception: Whatever ``load_animal_recordings`` raises when the animal cannot be loaded at all.
    """
    channel_subset = samples_config.get("_animal_channel_subsets", {}).get(animal)
    ao = load_animal_recordings(samples_config, config, [("", animal, "")], animal,
                                channel_subset=channel_subset, logger=log)
    log.info(f"{animal}: {len(ao.long_recordings)} recording(s)")
    return ao


def scan_recording(tree, animal, position, lro, july_n_fragments, temp_dir=None):
    """Yield one scan row per window and column of a recording, for ``--all-windows``.

    A scan row holds the raw sha256, statistics and first samples of one column in one window, with status
    scan and no rater-view hash; its channel is the column's canonical name. A failure gives rows with status
    scan_read_error and the error text instead of being raised: when a window cannot be read, one per
    column; when a column cannot be hashed, that one; when the recording cannot be opened, one row with no
    window or channel.

    Args:
        tree (str): Tree label written to every row.
        animal (str): Animal id.
        position (int): Position of the recording in this tree's ``long_recordings``.
        lro: The ``LongRecordingOrganizer`` at that position.
        july_n_fragments (dict[str, list[int]]): Recording label to the n_fragments values that reproduce
            its July window draw.
        temp_dir (str | Path | None, optional): The run's temp directory, for :func:`first_source_path`.
            Defaults to None.

    Yields:
        dict: Rows with the columns in :data:`FIELDS`, window by window.
    """
    label = f"{animal}__{position}"
    info = {"tree": tree, "animal": animal, "position": position,
            "july_n_fragments": ";".join(str(n) for n in july_n_fragments.get(label, []))}
    try:
        _, spf, _, canonical, _ = _open_recording(lro, info, temp_dir)
    except Exception as e:  # a recording that cannot be opened gives one row saying why
        error = _error_text(e)
        log.exception(f"{label}: could not open the recording for the scan, {error}")
        yield _row(info, "", "", "", status=SCAN_ERROR, error=error)
        return
    n_fragments = info["n_fragments"]
    log.info(f"{label}: scanning {n_fragments} windows of {len(canonical)} channels {canonical}, "
             f"{info['n_files']} file(s), date {info['date_string']!r}")

    n_failed = 0
    for w in range(n_fragments):
        window = dict(info, start_sample=spf * w)
        try:
            raw = np.asarray(_traces_uV(lro.get_fragment(FRAG_S, w)))
            window.update(n_samples=raw.shape[0], dtype=str(raw.dtype))
        except Exception as e:  # a window that cannot be read costs only its own rows
            error = _error_text(e)
            if n_failed == 0:
                log.exception(f"{label} window {w}: could not read the samples, {error}")
            else:
                log.error(f"{label} window {w}: could not read the samples, {error}")
            n_failed += 1
            for j, ch in enumerate(canonical):
                yield _row(window, w, ch, "", column=j, status=SCAN_ERROR, error=error)
            continue
        for j, ch in enumerate(canonical):
            try:
                values = _raw_values(raw[:, j])
            except Exception as e:  # a column that cannot be hashed costs only its own row
                error = _error_text(e)
                log.error(f"{label} window {w}: could not hash column {j} ({ch}), {error}")
                yield _row(window, w, ch, "", column=j, status=SCAN_ERROR, error=error)
                continue
            yield _row(window, w, ch, "", column=j, status=SCAN, **values)
    log.info(f"{label}: scanned {n_fragments} windows, {n_failed} of them unreadable")


def scan_animal(tree, animal, samples_config, config, july_n_fragments, temp_dir=None):
    """Load one animal as :func:`hash_animal` does and return the scan rows of each recording.

    The rows are produced as they are iterated, so a caller can write them without holding them all.

    Args:
        tree (str): Tree label written to every row.
        animal (str): Animal id.
        samples_config (dict): Expanded samples config of the animal's dataset, already applied.
        config (dict): The dataset config.
        july_n_fragments (dict[str, list[int]]): Recording label to its July n_fragments matches.
        temp_dir (str | Path | None, optional): The run's temp directory, for :func:`first_source_path`.
            Defaults to None.

    Returns:
        list[Iterator[dict]]: The rows of :func:`scan_recording` for each recording, in position order.

    Raises:
        Exception: Whatever ``load_animal_recordings`` raises when the animal cannot be loaded at all.
    """
    ao = load_animal(animal, samples_config, config)
    return [scan_recording(tree, animal, position, lro, july_n_fragments, temp_dir=temp_dir)
            for position, lro in enumerate(ao.long_recordings)]


def write_scan(writer, fh, tree, animal, samples_config, config, july_n_fragments, counts, temp_dir=None,
               code=None):
    """Scan one animal (:func:`scan_animal`) and write its rows as they are read, flushing after each recording.

    Args:
        writer (csv.DictWriter): Writer of the output.
        fh: The output's file handle.
        tree (str): Tree label written to every row.
        animal (str): Animal id.
        samples_config (dict): Expanded samples config of the animal's dataset, already applied.
        config (dict): The dataset config.
        july_n_fragments (dict[str, list[int]]): Recording label to its July n_fragments matches.
        counts (Counter): Rows written by status, updated as rows are written.
        temp_dir (str | Path | None, optional): The run's temp directory. Defaults to None.
        code (dict | None, optional): Values written to every row, from :func:`code_version`. Defaults to
            None.

    Raises:
        Exception: Whatever ``load_animal_recordings`` raises when the animal cannot be loaded at all.
    """
    for rows in scan_animal(tree, animal, samples_config, config, july_n_fragments, temp_dir=temp_dir):
        for row in rows:
            row.update(code or {})
            writer.writerow(row)
            counts[row["status"]] += 1
        fh.flush()  # the operating system keeps flushed rows when the process is killed


def main(argv=None):
    """Hash the labeled cells of every requested animal and write them as CSV, one animal at a time.

    Rows go to ``<out>.partial``, flushed after each animal, and the file is renamed to ``<out>`` once every
    requested animal has been tried. A file from an earlier run at ``<out>`` is removed when writing starts.
    With ``--all-windows`` the rows are scan rows instead, flushed after each recording.

    Args:
        argv (list[str], optional): Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 when every animal loaded and no row has status read_error or scan_read_error, 1 otherwise.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, help="label written to every row, for example july or port")
    p.add_argument("--keymap", type=Path, default=CONFIG_KEYMAP, help="keymap.csv of the labeled cells")
    p.add_argument("--manifest", type=Path, default=CONFIG_MANIFEST,
                   help="cohort_manifest.json: its datasets are searched for each animal")
    p.add_argument("--july-n-fragments", type=Path, default=CONFIG_JULY_N_FRAGMENTS,
                   help="july_n_fragments.csv, whose counts are logged next to this tree's; skipped if absent")
    p.add_argument("--animal", nargs="+", default=None, help="hash only these animal ids")
    p.add_argument("--out", type=Path, required=True,
                   help="output CSV; it is written as <out>.partial and renamed to <out> when the run finishes")
    p.add_argument("--temp-dir", type=Path, required=True,
                   help="empty temp directory for this run's intermediates, never shared between trees")
    p.add_argument("--reuse-temp-dir", action="store_true",
                   help="accept intermediates already in --temp-dir, for example a preserved copy of the cache "
                        "the July bundle read; every reused file is logged")
    p.add_argument("--all-windows", action="store_true",
                   help="scan the --animal animals: hash every window of every recording, every column, as rows "
                        "with status scan, for compare_window_hashes.py --port-scan. It reads whole recordings, "
                        "about 600 bytes of output per window and column")
    args = p.parse_args(argv)
    if args.all_windows and not args.animal:
        p.error("--all-windows reads every sample of every recording; name the animals to scan with --animal")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    check_tree()
    tree_dir = Path.cwd().resolve()
    code = code_version(tree_dir)
    log.info(f"{tree_dir}: git commit {code['git_commit'] or 'unknown'}, src dirty {code['src_dirty']!r}, "
             f"src tree {code['src_tree'] or 'unknown'}")
    refused = ("a recording map made from these rows as the port hashes has an empty port_src_tree, which "
               "extract_dev_features.py refuses")
    if not code["git_commit"]:
        log.warning(f"no git commit of {tree_dir} is recorded, so the rows name no code version")
    elif code["src_dirty"] is True:
        log.warning(f"src in {tree_dir} differs from commit {code['git_commit']}, so neither that commit nor its "
                    f"src tree is exactly the code that ran, and {refused}")
    elif code["src_dirty"] == "":
        log.warning(f"git could not tell whether src in {tree_dir} differs from commit {code['git_commit']}, so "
                    f"{refused}")
    if not code["src_tree"]:
        log.warning(f"no src tree of {tree_dir} is recorded, so {refused}")
    temp_dir = prepare_temp_dir(args.temp_dir, reuse=args.reuse_temp_dir)
    cells = read_keymap_cells(args.keymap)
    manifest = json.loads(Path(args.manifest).read_text())
    order = [str(a) for a in manifest.get("rendered_animals", [])]
    animals = [a for a in order if a in cells] + sorted(a for a in cells if a not in order)
    if args.animal:
        unknown = sorted(set(args.animal) - set(cells))
        if unknown:
            raise SystemExit(f"--animal {unknown}: not in {args.keymap}")
        animals = [a for a in animals if a in set(args.animal)]
    july_n_fragments = read_july_n_fragments(args.july_n_fragments)

    owner, prepared, failures = find_datasets(list(manifest["datasets"]), animals)

    out = args.out
    partial = out.with_name(out.name + PARTIAL_SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        log.warning(f"removing {out}, left by an earlier run, so that only a run that finishes leaves a file there")
        out.unlink()
    ok, error_status = (SCAN, SCAN_ERROR) if args.all_windows else ("ok", "read_error")
    n_rows = n_hashed = 0
    read_errors = Counter()
    with open(partial, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        fh.flush()
        for ds in manifest["datasets"]:
            members = [a for a in animals if owner.get(a) == ds]
            if not members:
                continue
            config, samples_config = prepared[ds]
            set_temp_directory(temp_dir)
            apply_samples_config(samples_config)
            log.info(f"dataset {ds}: {len(members)} animal(s)")
            for a in members:
                if args.all_windows:
                    counts = Counter()
                    try:
                        write_scan(writer, fh, args.tree, a, samples_config, config, july_n_fragments, counts,
                                   temp_dir=temp_dir, code=code)
                    except Exception as e:  # one animal that fails must not stop the others
                        log.exception(f"{a}: failed, {type(e).__name__}: {e}")
                        failures.append({"animal": a, "error": f"{type(e).__name__}: {e}"})
                    n_rows += sum(counts.values())
                    n_hashed += counts[ok]
                    read_errors[a] = counts[error_status]
                    log.info(f"{a}: wrote {sum(counts.values())} scan rows ({read_errors[a]} read errors) to "
                             f"{partial}")
                    continue
                try:
                    rows = hash_animal(args.tree, a, samples_config, config, cells[a], july_n_fragments,
                                       temp_dir=temp_dir)
                except Exception as e:  # one animal that fails to load must not stop the others
                    log.exception(f"{a}: failed, {type(e).__name__}: {e}")
                    failures.append({"animal": a, "error": f"{type(e).__name__}: {e}"})
                    continue
                for r in rows:
                    r.update(code)
                writer.writerows(rows)
                fh.flush()  # the operating system keeps flushed rows when the process is killed
                n_rows += len(rows)
                n_hashed += sum(1 for r in rows if r["status"] == "ok")
                read_errors[a] = sum(1 for r in rows if r["status"] == "read_error")
                log.info(f"{a}: wrote {len(rows)} rows ({read_errors[a]} read errors) to {partial}")
    os.replace(partial, out)

    n_errors = sum(read_errors.values())
    log.info(f"wrote {n_rows} rows ({n_hashed} hashed, {n_errors} read errors) for "
             f"{len(animals) - len(failures)} of {len(animals)} animal(s) to {out}")
    for a, n in read_errors.items():
        if n:
            log.error(f"{a}: {n} row(s) with status {error_status}; the error column says why")
    for f in failures:
        log.error(f"{f['animal']}: {f['error']}")
    return 1 if failures or n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
