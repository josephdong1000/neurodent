"""Automated audit + fix of per-session ``manual_datetime`` from recording headers.

For every animal in a dataset config (no animal can be skipped - it iterates the
config), derive each session's true start:
  - EDF: the EDF header start (MNE ``info['meas_date']``), which DataWave wrote
    into the file (NOT the filesystem creation time). All clips of a session
    share one header time = the SESSION (day) start.
  - RHD (Intan): the ``YYMMDD_HHMMSS`` token in the filename (per-file accurate).

Safety guards - a session is FLAGGED (not auto-written) when it can't be trusted:
  - EDF session whose lowest-index file is not Section/Selection 1 (the header
    gives the day start, not this clip's start).
  - EDF session whose files span more than one recording day (mislabeled files,
    e.g. the arx_rosa '1199' ' ' token spanning Jan 19 + Jan 21).
  - A session whose config value is a LIST (explicit per-file starts for a
    missing-file gap, e.g. the arx_parv '29 32 34' '_0_' session) is shown as
    ``LIST`` in scan and is NEVER overwritten by apply.

Two steps (keep the heavy header reads on SLURM, the write local):
  1. scan  (SLURM): read headers, print report, dump proposals to JSON.
       sbatch --wrap="cd <repo> && uv run python scripts/fix_recording_times.py \\
                      scan --config config/datasets/arx_rosa.yaml --out /tmp/rt.json"
  2. apply (local): rewrite manual_datetime from the JSON, preserving comments.
       uv run python scripts/fix_recording_times.py apply \\
                      --config config/datasets/arx_rosa.yaml --in /tmp/rt.json
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DT_FMT = "%Y-%m-%d %H:%M:%S"


def pattern_to_regex(pattern: str) -> re.Pattern:
    base = pattern.split("/")[-1]
    rx = re.escape(base)
    rx = rx.replace(re.escape("{session}"), r"(?P<session>.*?)")
    rx = rx.replace(re.escape("{index}"), r"(?P<index>\d+)")
    rx = rx.replace(re.escape("*"), r".*")
    return re.compile("^" + rx + "$", re.IGNORECASE)


def edf_start(f: str):
    import mne

    mne.set_log_level("ERROR")
    md = mne.io.read_raw_edf(f, preload=False).info.get("meas_date")
    if md is None:
        return None
    if md.tzinfo is not None:
        md = md.astimezone(timezone.utc).replace(tzinfo=None)
    return md.replace(microsecond=0)


def rhd_start(name: str):
    m = re.search(r"_(\d{6})_(\d{6})", name)
    return datetime.strptime(m.group(1) + m.group(2), "%y%m%d%H%M%S") if m else None


def _samples_data(config_path: str) -> dict:
    from neurodent.workflow.utils import load_samples_config, resolve_samples_config

    return resolve_samples_config(load_samples_config(config_path))


def scan(config_path: str, out_path: str) -> None:
    sd = _samples_data(config_path)
    data_root = sd["data_root"]
    groups = {}
    for a in sd["animals"]:
        groups.setdefault((a["group"], a["pattern"]), a.get("manual_datetime") or {})

    proposals: dict = {}
    for (group, pattern), mdt in groups.items():
        folder = Path(data_root) / group
        is_edf = pattern.lower().endswith(".edf")
        files = []
        for ext in (("*.EDF", "*.edf") if is_edf else ("*.rhd",)):
            files += glob.glob(str(folder / ext))
        rx = pattern_to_regex(pattern)
        by_sess = defaultdict(list)
        for f in sorted(set(files)):
            m = rx.match(Path(f).name)
            if m:
                by_sess[m.group("session")].append((int(m.group("index")), f))

        print("=" * 88)
        print(f"GROUP {group!r}  [{'EDF' if is_edf else 'RHD'}]  sessions={len(by_sess)}")
        gp = {}
        for sess, items in by_sess.items():
            items.sort()
            min_idx, first_f = items[0]
            starts = {(edf_start(f) if is_edf else rhd_start(Path(f).name)) for _, f in items}
            starts.discard(None)
            days = {s.date() for s in starts}
            first_start = edf_start(first_f) if is_edf else rhd_start(Path(first_f).name)
            flag = None
            if is_edf and min_idx != 1:
                flag = f"lowest index is {min_idx} (not 1); header = day-start, not this clip"
            elif is_edf and len(days) > 1:
                flag = f"files span {len(days)} recording days {sorted(map(str, days))}"
            proposed = None if flag else (first_start.strftime(DT_FMT) if first_start else None)
            cur_raw = mdt.get(sess)
            is_list = isinstance(cur_raw, list)
            cur = cur_raw if isinstance(cur_raw, str) else (None if cur_raw is None else str(cur_raw))
            if is_list:
                verdict = "LIST "  # gap-handling per-file starts; apply() preserves it
            elif flag:
                verdict = "FLAG "
            elif cur == proposed:
                verdict = "same "
            else:
                verdict = "WRITE"
            note = f"   ({flag})" if flag else ("   (list preserved)" if is_list else "")
            print(f"  [{verdict}] session {sess!r:>6}: config={cur!r}  ->  {proposed!r}" + note)
            gp[sess] = {"proposed": proposed, "flag": flag, "current": cur, "is_list": is_list}
        proposals[group] = gp

    Path(out_path).write_text(json.dumps(proposals, indent=2))
    print("=" * 88)
    print(f"wrote proposals -> {out_path}")


def apply(config_path: str, in_path: str) -> None:
    from ruamel.yaml import YAML
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ

    from neurodent.workflow.utils import load_samples_config

    proposals = json.loads(Path(in_path).read_text())

    # locate the file that actually holds the animals (inline dataset vs pointer)
    loaded = load_samples_config(config_path)
    if "samples_data" in loaded:
        target, anchor = config_path, ("samples_data", "animals")
    else:
        target, anchor = loaded["samples"]["samples_file"], ("animals",)

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    with open(target) as f:
        doc = yaml.load(f)
    node = doc
    for k in anchor:
        node = node[k]

    n = 0
    for animal in node:
        gp = proposals.get(animal.get("group"))
        if not gp or "manual_datetime" not in animal:
            continue
        for sess, info in gp.items():
            if sess not in animal["manual_datetime"]:
                continue
            # Never overwrite a list-valued session (explicit per-file starts for a
            # missing-file gap, e.g. arx_parv '29 32 34' '_0_') with a single scalar.
            if isinstance(animal["manual_datetime"][sess], list):
                continue
            if info["proposed"] and str(animal["manual_datetime"][sess]) != info["proposed"]:
                animal["manual_datetime"][sess] = DQ(info["proposed"])
                n += 1
    with open(target, "w") as f:
        yaml.dump(doc, f)
    flagged = {g: [s for s, i in gp.items() if i["flag"]] for g, gp in proposals.items()}
    flagged = {g: s for g, s in flagged.items() if s}
    print(f"applied {n} manual_datetime value(s) to {target}")
    if flagged:
        print("FLAGGED (left unchanged - need manual review):")
        for g, s in flagged.items():
            print(f"  {g!r}: sessions {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--config", default="config/datasets/arx_rosa.yaml")
    s.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--config", default="config/datasets/arx_rosa.yaml")
    a.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args.config, args.out)
    else:
        apply(args.config, args.inp)


if __name__ == "__main__":
    main()
