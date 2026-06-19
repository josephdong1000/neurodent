"""Audit arx_rosa manual_datetime against actual recording-start times.

EDF: the authoritative start is the EDF header start (MNE ``info['meas_date']``),
which is what DataWave wrote into the file (NOT the filesystem creation date).
RHD (Intan): the classic header has no wall clock, so the absolute start is the
``YYMMDD_HHMMSS`` token in the filename.

For each group/session we print the config's manual_datetime alongside the real
first-file start and the per-file start sequence (to expose gaps / wrong anchors).

Run on SLURM:
    sbatch --job-name=arxrosa_times --cpus-per-task=2 --mem=8G --time=00:30:00 \\
      --wrap="cd <repo> && uv run python scripts/check_arxrosa_recording_times.py"
"""
import glob
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mne

from neurodent.workflow.utils import load_samples_config, resolve_samples_config

mne.set_log_level("ERROR")

DATASET = "config/datasets/arx_rosa.yaml"


def pattern_to_regex(pattern: str) -> re.Pattern:
    base = pattern.split("/")[-1]
    rx = re.escape(base)
    rx = rx.replace(re.escape("{session}"), r"(?P<session>.*?)")
    rx = rx.replace(re.escape("{index}"), r"(?P<index>\d+)")
    rx = rx.replace(re.escape("\\*"), r".*").replace(re.escape("*"), r".*")
    return re.compile("^" + rx + "$", re.IGNORECASE)


def naive(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0)


def parse_cfg_dt(v):
    if not v:
        return None
    return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")


def edf_start(f):
    raw = mne.io.read_raw_edf(f, preload=False)
    return naive(raw.info.get("meas_date")), raw.n_times / raw.info["sfreq"]


def rhd_start_from_name(name):
    m = re.search(r"_(\d{6})_(\d{6})", name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%y%m%d%H%M%S")


def main():
    cfg = resolve_samples_config(load_samples_config(DATASET))
    data_root = cfg["data_root"]

    groups = {}
    for a in cfg["animals"]:
        groups.setdefault((a["group"], a["pattern"]), a["manual_datetime"])

    for (group, pattern), mdt in groups.items():
        folder = Path(data_root) / group
        is_edf = pattern.lower().endswith(".edf")
        files = []
        for ext in (("*.EDF", "*.edf") if is_edf else ("*.rhd",)):
            files += glob.glob(str(folder / ext))
        files = sorted(set(files))
        rx = pattern_to_regex(pattern)

        print("=" * 90)
        print(f"GROUP {group!r}  [{'EDF' if is_edf else 'RHD'}]  files={len(files)}")
        print(f"  config manual_datetime: {mdt}")

        by_sess = defaultdict(list)
        unmatched = []
        for f in files:
            m = rx.match(Path(f).name)
            if m:
                by_sess[m.group("session")].append((int(m.group("index")), f))
            else:
                unmatched.append(f)

        for sess in sorted(by_sess):
            items = sorted(by_sess[sess])
            cfg_dt = parse_cfg_dt(mdt.get(sess) if isinstance(mdt.get(sess), str) else None)
            print(f"  -- session {sess!r}  config={mdt.get(sess)!r}")
            for i, (idx, f) in enumerate(items):
                try:
                    if is_edf:
                        start, dur = edf_start(f)
                        durs = f"{dur:7.0f}s"
                    else:
                        start, durs = rhd_start_from_name(Path(f).name), "   (rhd)"
                    flag = ""
                    if i == 0 and cfg_dt and start:
                        diff = abs((start - cfg_dt).total_seconds())
                        flag = "  <-- MATCH" if diff <= 60 else f"  <-- MISMATCH (config off by {(start - cfg_dt)})"
                    print(f"       idx{idx}: start={start}  {durs}  {Path(f).name}{flag}")
                except Exception as e:  # noqa: BLE001
                    print(f"       idx{idx}: ERR {type(e).__name__}: {e}  {Path(f).name}")
        for f in unmatched:
            try:
                start = edf_start(f)[0] if is_edf else rhd_start_from_name(Path(f).name)
            except Exception:  # noqa: BLE001
                start = "ERR"
            print(f"  [unmatched-by-pattern] start={start}  {Path(f).name}")
    print("=" * 90)
    print("DONE")


if __name__ == "__main__":
    main()
