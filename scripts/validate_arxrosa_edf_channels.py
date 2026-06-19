"""Validate ArxRosa EDF channel layout against the corrected datasets/arx_rosa.yaml.

Run on SLURM (not locally). Checks, per REF bank (REF1=E1-8, REF2=E9-16,
REF3=E17-24, REF4=E25-32), which channels carry signal vs. are flat/floating.

Answers two questions raised while correcting config/datasets/arx_rosa.yaml:
  1. 5->8 expansion: do positions 3, 7, 8 (newly added Bar/Vis channels) carry
     real signal within a live bank?
  2. 1015 bank: for the 2-animal 1017/1015 session, is REF1 live (1015's new
     home) and is REF4 flat (where 1015 used to be)? Don says channels 1-16 only.
"""

import re
import sys
import numpy as np
import mne

mne.set_log_level("ERROR")

ROOT = "/mnt/isilon/marsh_single_unit/PythonEEG Data/Arx Rosa"

# (label, path) — one file per session is enough for a layout check.
FILES = [
    ("1017/1015 (2-animal session)",
     f"{ROOT}/Arx Rosa 1017 1015/MARSH 20150224ARXROSA10171015_Selection1.EDF"),
    ("967/968/969/418 (4-animal session)",
     f"{ROOT}/Arx Rosa  967 968 969 418/MARSH 20141125ARXROSATAM967968969418_Selection1.EDF"),
]
if len(sys.argv) > 1:
    FILES = [(p, p) for p in sys.argv[1:]]

CROP_S = 120.0
PAT = re.compile(r"E(\d+)-REF(\d+)")


def analyze(label, path):
    print("=" * 78)
    print(f"{label}\n  {path}")
    try:
        raw = mne.io.read_raw_edf(path, preload=False)
    except Exception as e:  # noqa: BLE001
        print(f"  !! could not open: {type(e).__name__}: {e}")
        return
    sf = raw.info["sfreq"]
    dur = raw.n_times / sf
    raw.crop(tmax=min(CROP_S, dur - 1.0 / sf))
    raw.load_data()
    data = raw.get_data() * 1e6  # -> microvolts
    names = raw.ch_names
    sd = data.std(axis=1)
    print(f"  sfreq={sf:.1f} Hz, dur={dur:.0f}s, n_ch={len(names)}")

    # bucket by REF bank -> {pos: (name, std)}
    banks: dict[int, dict[int, tuple]] = {}
    for nm, s in zip(names, sd):
        m = PAT.search(nm)
        if not m:
            continue
        e, ref = int(m.group(1)), int(m.group(2))
        pos = e - (ref - 1) * 8
        banks.setdefault(ref, {})[pos] = (nm, s)

    flat = np.median(np.sort(sd)[: max(1, len(sd) // 4)])  # typical flat level
    thr = max(flat * 3, 5.0)
    region = {1: "Mot", 2: "Mot", 3: "Bar", 4: "Bar", 5: "Hip", 6: "Hip", 7: "Vis", 8: "Vis"}
    for ref in sorted(banks):
        live_pos = [p for p, (_, s) in banks[ref].items() if s > thr]
        tag = "LIVE" if len(live_pos) >= 4 else ("partial" if live_pos else "FLAT/floating")
        print(f"  -- REF{ref}  [{tag}] live positions: {sorted(live_pos)}")
        for pos in sorted(banks[ref]):
            nm, s = banks[ref][pos]
            mark = "live" if s > thr else "flat"
            print(f"       pos{pos} {region.get(pos,'?'):3s} {nm:14s} std={s:8.1f}uV  {mark}")
    print(f"  (flat~{flat:.1f}uV, live threshold={thr:.1f}uV)")


if __name__ == "__main__":
    for label, path in FILES:
        analyze(label, path)
    print("=" * 78)
    print("DONE")
