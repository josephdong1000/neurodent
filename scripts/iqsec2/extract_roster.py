"""Extract the IQSEC2 roster and session structure from the lab records and the data.

The IQSEC2 config cannot be hand-written. Its adult half spans ten rhd cohorts whose
session start times live in nine thousand filenames, and its port assignments live in a
spreadsheet rather than in the directory names. Both are transcribed here so the config
is generated from the sources and can be regenerated when either changes.

Two authorities, deliberately kept separate:

* ``IQSEC2 Compiled EEG data.xlsx``, sheet ``EEG Reference Sheet``, gives ID, cage,
  allele genotype and the amp-channel list. Cage number maps to port letter (1->A, 2->B,
  3->C, 4->D); the trailing letter in "3B" is the rack, not the port. Folder-name
  genotypes disagree with this sheet and are NOT used.
* The filesystem gives the session keys and their start times.

Sex is derived from the allele, which is the only systematic source: the sheet has no sex
column. Per the legend in the ``Demographic Info`` sheet, anything with a ``y`` is male.

Usage:
    uv run python scripts/iqsec2/extract_roster.py > /tmp/iqsec2_roster.json
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RHD_ROOT = Path("/mnt/isilon/marsh_single_unit/PythonEEG Data/IQSEC2")
BIN_ROOT = Path("/mnt/isilon/marsh_single_unit/PythonEEG Data Bins/IQSEC2")
WORKBOOK = RHD_ROOT / "IQSEC2 Compiled EEG data.xlsx"

CAGE_TO_PORT = {"1": "A", "2": "B", "3": "C", "4": "D"}
# The canonical implant, corroborated three ways: DataWave column order, the sheet's
# Amp channel column, and a montage note repeated in 28 sheet rows.
AMP_TO_REGION = {
    "010": "LVis", "012": "LHip", "014": "LBar", "015": "LMot",
    "016": "RMot", "017": "RBar", "019": "RHip", "021": "RVis",
}
CANONICAL_AMPS = sorted(AMP_TO_REGION)


def read_reference_sheet():
    """Return per-animal records from the EEG Reference Sheet.

    Returns:
        list[dict]: One record per animal row with id, allele, sex, cage, port and the
        recording start/end and died/euth cells, which mark animals that were implanted
        but never recorded.
    """
    import openpyxl
    import warnings

    warnings.filterwarnings("ignore")
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    ws = wb["EEG Reference Sheet"]
    rows = list(ws.iter_rows(values_only=True))

    out = []
    for row in rows[1:]:
        if row[1] is None:
            continue
        animal_id = str(row[1]).strip()
        raw_allele = re.sub(r"\s+", " ", str(row[3]).strip()) if row[3] is not None else ""
        match = re.search(r"\(([^)]*)\)", raw_allele)
        allele = match.group(1) if match else None

        # Anything with a /y is hemizygous male; the two-X forms are female. Nothing
        # else in the workbook records sex.
        if allele is None:
            sex = None
        elif allele.endswith("/y"):
            sex = "Male"
        else:
            sex = "Female"

        cage = str(row[10]).strip() if row[10] is not None else ""
        cage_digit = re.match(r"([1-4])", cage)
        port = CAGE_TO_PORT.get(cage_digit.group(1)) if cage_digit else None

        out.append({
            "id": animal_id,
            "allele": allele,
            "sex": sex,
            "cage": cage,
            "port": port,
            "recording_start": str(row[8]) if row[8] is not None else None,
            "recording_end": str(row[9]) if row[9] is not None else None,
            "died": str(row[6]).strip() if row[6] is not None else None,
        })
    return out


RHD_STAMP = re.compile(r"^(?P<session>.+_(?P<ymd>\d{6}))_(?P<hms>\d{6})\.rhd$")


def scan_rhd_cohorts():
    """Return session structure and start times for every rhd cohort.

    The R1 pattern captures ``{session}`` as ``<prefix>_YYMMDD`` and ``{index}`` as the
    ``HHMMSS``. A session's start time is the earliest index within it. No leading
    wildcard is used, because a greedy one eats the prefix and fuses two distinct hookups
    of the same date into a single session.

    Returns:
        dict: cohort -> {"depth": int, "sessions": {session_key: iso_start}, "n_files": int}
    """
    out = {}
    for cohort_dir in sorted(RHD_ROOT.iterdir()):
        if not cohort_dir.is_dir():
            continue
        files = sorted(cohort_dir.rglob("*.rhd"))
        if not files:
            continue

        sessions = defaultdict(list)
        depths = set()
        for path in files:
            rel = path.relative_to(cohort_dir)
            depths.add(len(rel.parts))
            match = RHD_STAMP.match(path.name)
            if not match:
                sessions["UNPARSED"].append(path.name)
                continue
            sessions[match.group("session")].append(match.group("hms"))

        starts = {}
        for session, stamps in sessions.items():
            if session == "UNPARSED":
                continue
            ymd = session.rsplit("_", 1)[1]
            hms = min(stamps)
            yy, mm, dd = ymd[:2], ymd[2:4], ymd[4:6]
            starts[session] = f"20{yy}-{mm}-{dd} {hms[:2]}:{hms[2:4]}:{hms[4:6]}"

        out[cohort_dir.name] = {
            "depths": sorted(depths),
            "n_files": len(files),
            "n_sessions": len(starts),
            "sessions": dict(sorted(starts.items(), key=lambda kv: kv[1])),
            "unparsed": sessions.get("UNPARSED", []),
            "subdirs": sorted(p.name for p in cohort_dir.iterdir() if p.is_dir()),
        }
    return out


def scan_rhd_ports():
    """Return the amp-channel layout of one rhd file per cohort.

    Ports are not uniform: three cohorts expose 32, 64 or 96 channels rather than 128,
    so a global channel map would name channels that do not exist and raise on load.
    """
    from spikeinterface.extractors import read_intan

    out = {}
    for cohort_dir in sorted(RHD_ROOT.iterdir()):
        if not cohort_dir.is_dir():
            continue
        files = sorted(cohort_dir.rglob("*.rhd"))
        if not files:
            continue
        try:
            rec = read_intan(str(files[0]), stream_id="0")
            ids = [str(c) for c in rec.get_channel_ids()]
            out[cohort_dir.name] = {
                "n_channels": len(ids),
                "ports": sorted({c.split("-")[0] for c in ids}),
                "fs": float(rec.get_sampling_frequency()),
            }
        except Exception as exc:  # noqa: BLE001 - report and continue the census
            out[cohort_dir.name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def main():
    payload = {
        "reference_sheet": read_reference_sheet(),
        "rhd_cohorts": scan_rhd_cohorts(),
        "rhd_ports": scan_rhd_ports(),
        "canonical_amps": CANONICAL_AMPS,
        "amp_to_region": AMP_TO_REGION,
    }
    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
