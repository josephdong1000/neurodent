"""Generate config/datasets/iqsec2.yaml (adult half) from the lab records and the data.

Hand-writing this config is not practical: the ten rhd cohorts carry 39 sessions whose
start times live in 9890 filenames, and the port assignments live in a spreadsheet rather
than in any directory name. Both are read from their sources here so the config can be
regenerated when either changes.

What is transcribed by hand, and why:

* The cohort-to-animal mapping below. The reference sheet gives ID, cage and allele but
  not which cohort directory an animal belongs to, and cohort directory names are not
  parseable (``IQSEC287xIQ102`` is a breeding cross, not animals 87 and 102). Every entry
  is cross-checked against the sheet's port and against the ports the rhd file actually
  exposes; the check runs on every generation and refuses to emit on a mismatch.
* The exclusions, each with its reason.

Everything is read from its source on every run, and there is deliberately no cached roster
file. An earlier revision split the scan into a separate script writing json, which put a
copy of the montage map and of the lab records in a second place where either could
silently go stale, and the two scripts documented different paths for that json so
following both never produced a working pipeline.

Reading the sources directly is also not slow, which is what makes the cache unnecessary:
the whole run is about fifteen seconds. The port census goes through neo's header parser
rather than spikeinterface's read_intan, which reads the same bytes but does not build a
full extractor object only to discard it, and that one choice is the difference between
fifteen minutes and eight seconds.

Usage:
    uv run python scripts/iqsec2/generate_config.py --out config/datasets/iqsec2.yaml

Do NOT redirect stdout into the config. The shell truncates the target before this script
starts, so a refusal to generate would destroy the file the refusal exists to protect.
"""

import argparse
import calendar
import os
import re
import tempfile
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA_ROOT = "/mnt/isilon/marsh_single_unit"
RHD_SUBTREE = "PythonEEG Data/IQSEC2"

# Canonical implant: amp offset -> region. Corroborated three ways (DataWave column
# order, the sheet's Amp channel column, and a montage note repeated in 28 sheet rows).
AMP_TO_REGION = {
    "010": "LVis", "012": "LHip", "014": "LBar", "015": "LMot",
    "016": "RMot", "017": "RBar", "019": "RHip", "021": "RVis",
}
REGION_ORDER = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LVis", "RVis"]

# Cohort -> {animal id: port}. Hand-transcribed, machine-verified (see module docstring).
RHD_COHORTS = {
    "20180917_IQSEC2_6_23": {"6": "C", "23": "D"},
    "IQSEC 56 57 60 61": {"56": "A", "57": "B", "60": "C", "61": "D"},
    "IQSEC 65_66": {"65": "C", "66": "D"},
    "013120_IQSEC2_78_76_Recordings": {"76": "A", "78": "D"},
    "012720_IQSEC2_77": {"77": "C"},
    "011320 IQSEC2_80_81_82_83": {"80": "A", "81": "B", "82": "C", "83": "D"},
    "20200505_IQSEC2_97_98_99": {"97": "A", "98": "B", "99": "C"},
    "20200602_IQSEC2_106_107": {"106": "C", "107": "D"},
    "20200609_IQSEC2_114_117_118": {"114": "A", "118": "C"},
    "IQSEC animal 3": {"3": "C"},
}

# The canonical genotype vocabulary. A populated GENOTYPE_MAP is authoritative and raises
# on any value it does not cover, so every genotype the generator emits must appear here.
# '-/y' is deliberately absent: it occurs once in the sheet, for animal 42, and is a
# transcription error for '(x/y)'. See ALLELE_CORRECTIONS.
GENOTYPE_LEVELS = ["Unknown", "+/y", "x/y", "+/x", "+/+", "x/x"]

# Sheet values corrected before use, each with the evidence for the correction.
ALLELE_CORRECTIONS = {
    # The sheet gives animal 42 '(-/y)', which appears nowhere in the Demographic Info
    # legend and nowhere else in the workbook. The lab's own Control Male analysis sheet
    # lists 42 beside the four confirmed (x/y) control males, and its folder is named
    # '42 IQSEC2 CT Male'. Read as a transcription slip for (x/y).
    "42": "x/y",
}

# Cohort directories that hold rhd files but are not part of the adult study.
PUP_COHORTS = {"Pup EEG"}

BIN_SUBTREE = "PythonEEG Data Bins/IQSEC2"

# DataWave writes region words rather than amp numbers. Matched case-insensitively because
# three cohorts carry a lowercase spelling of the same names.
DATAWAVE_REGION = {
    "l vis ctx": "LVis", "l hipp": "LHip", "l barrel ctx": "LBar", "l motor ctx": "LMot",
    "r motor ctx": "RMot", "r barrel ctx": "RBar", "r hipp": "RHip", "r vis ctx": "RVis",
}

# Adults recorded only by the DataWave rig, so they have no rhd leg at all and are absent
# from RHD_COHORTS. They are included because they are what breaks the date confound: as
# rhd-only, every control male sits in a 35 day window in 2020 with just one of eight
# mutants inside it, and adding these puts controls in early 2019 as well.
#
# cage is the sheet's cage number; the port it implies is cross-checked against the port
# the Meta.csv sidecars actually declare, and a disagreement refuses to generate.
BIN_ADULTS = {
    "51": {"dir": "IQSEC2 51_52/51 IQSEC2 Mut Male", "cage": "3", "nested": True},
    "52": {"dir": "IQSEC2 51_52/52 IQSEC2 Mut Male", "cage": "4"},
    "42": {"dir": "IQSEC 42 43 44 47/42 IQSEC2 CT Male", "cage": "1"},
    "43": {"dir": "IQSEC 42 43 44 47/43 IQSEC2 CT Male", "cage": "2"},
    "44": {"dir": "IQSEC 42 43 44 47/44 IQSEC2 CT Male", "cage": "3"},
}

# Bin adults deliberately not loaded, with the reason.
BIN_EXCLUSIONS = {
    "47": (
        "Animal 47 (IQSEC 42 43 44 47/47 IQSEC2 Het Female) has only 3 live bins and they "
        "do not share a montage: two declare 7 channels, missing D-021 (RVis), and one "
        "declares the full 8. _validate_channel_names compares resolved abbreviations, so "
        "a 7-channel file and an 8-channel file in one animal raise rather than merge. "
        "The pattern language cannot express a per-file exclusion (skip_sessions matches "
        "on {session}, and these share one), so isolating the 8-channel file would leave "
        "the animal with a single recording. Excluded rather than fudged. It is a het "
        "female contributing about 3 hours, so it does not bear on the mutant-versus-"
        "control male contrast these bin animals were added to de-confound."
    ),
}

# Animals whose implant was also recorded by the DataWave rig, producing a second block of
# data in the bin tree that THIS config does not reach. Counted at generation time so the
# note cannot go stale. Not loaded here because the bin half is a different reader and a
# different channel key space; recorded per animal so nobody reads these entries as the
# animal's complete record.
DUAL_TREE_BINS = {
    "80": "011320 IQSEC2_80_81_82_83", "81": "011320 IQSEC2_80_81_82_83",
    "82": "011320 IQSEC2_80_81_82_83", "83": "011320 IQSEC2_80_81_82_83",
    "76": "013120_IQSEC2_78_76_Recordings", "78": "013120_IQSEC2_78_76_Recordings",
    "77": "012720_IQSEC2_77",
}

# Per-animal notes that no automated check can derive.
ANIMAL_NOTES = {
    "81": (
        "Records conflict on this animal, resolved in favour of the reference sheet. The "
        "sheet's genotype column gives IQSEC2(+/y), a mutant male, and that is what is "
        "emitted. Against it: the workbook's four per-genotype analysis sheets are "
        "complete and non-overlapping, and they place 81 in Mutant Female (+/+) while "
        "Mutant Male explicitly omits it, so the (+/+) total of 3 only reconciles that "
        "way. Demographic Info lists 81 under BOTH columns. The tiebreaker is that its "
        "(+/y) entry carries age 36, exactly its age at implant given a 2019-12-02 birth "
        "and a 2020-01-07 implant, while the (+/+) entry carries age 46, matching no "
        "implant date. Genetics agrees: the litter holds five (+/x) hets, implying a "
        "wild-type sire, who cannot produce (+/+) daughters."
    ),
    "82": (
        "Same records conflict as animal 81 and resolved the same way, in favour of the "
        "reference sheet's IQSEC2(+/y). See the note on 81 for the full evidence."
    ),
}

# Every port the hardware exposes but no animal claims, with the reason it is empty. Keyed
# by (cohort, port) rather than by cohort, so "checked and found empty" is distinguishable
# from "never considered". The port guard refuses to generate on any unassigned port that
# is missing from this table.
#
# Live counts below are canonical declared amps (010,012,014,015,016,017,019,021) reading
# under OPEN_CIRCUIT_OHMS in that cohort's own impedance export.
PORT_EXCLUSIONS = {
    ("IQSEC 65_66", "A"): "Empty headstage: 1 of 8 canonical amps live.",
    ("IQSEC 65_66", "B"): (
        "4 of 8 canonical amps live, in a non-canonical set, with no reference-sheet row "
        "and no name anywhere in the workbook. An unidentified animal cannot be given a "
        "genotype, so it is not loaded."
    ),
    ("013120_IQSEC2_78_76_Recordings", "B"): "Empty headstage: 0 of 8 canonical amps live.",
    ("013120_IQSEC2_78_76_Recordings", "C"): "Empty headstage: 0 of 8 canonical amps live.",
    ("012720_IQSEC2_77", "A"): "Empty headstage: 0 of 8 canonical amps live.",
    ("012720_IQSEC2_77", "D"): (
        "A CDKL5 animal, not IQSEC2: these 140 rhd are a joint recording filed identically "
        "under PythonEEG Data/CDKL5. All 8 canonical amps are live, so this is a real "
        "implant belonging to another study."
    ),
    ("20200505_IQSEC2_97_98_99", "D"): "Empty headstage: 0 of 8 canonical amps live.",
    ("20200602_IQSEC2_106_107", "A"): "Empty headstage: 0 of 8 canonical amps live.",
    ("20200602_IQSEC2_106_107", "B"): "Empty headstage: 0 of 8 canonical amps live.",
    ("20200609_IQSEC2_114_117_118", "B"): (
        "Animal 117, not loaded. The reference sheet is the authority here and it gives "
        "117 no Cage, no Amp channel, no Recording Start and no Recording End, so no port "
        "assignment is derivable for it under the cage-to-port rule this config uses "
        "everywhere else; 115, its littermate, is marked Died with the same blank cells. "
        "Recorded honestly: the hardware disagrees. This port carries 7 of 8 live "
        "canonical amps, indistinguishable in quality from PortA which IS loaded, and the "
        "cohort's DataWave sidecars name their four cages Cage 1 (114), Cage 2 (117), "
        "Cage 3 (118) and Cage 4 (Empty), so the acquisition software recorded a Cage 2 "
        "animal named 117 for the full three days. That is a gap in the lab record rather "
        "than evidence the port is empty, and resolving it needs the lab, not more "
        "analysis. Two earlier justifications for this exclusion were withdrawn as false: "
        "that the 59.5 GB file named 117 is a cage video (all four cohort .ddf are the "
        "same kind of file, including those of two loaded animals), and that PortB is "
        "92.4 percent line noise (that figure has no script, output or log behind it)."
    ),
    ("20200609_IQSEC2_114_117_118", "D"): (
        "Empty headstage: 0 of 8 canonical amps live, and the cohort's DataWave sidecar "
        "names this cage 'Empty'."
    ),
}

# Cohorts whose rhd files sit at two depths. No single glob spans them: ** cannot cross a
# separator in this discovery layer, and a pattern list is an AND-pair, not a union. Each
# such cohort gets two sources instead. The nested block is a genuine continuation, not a
# quarantine: it begins one cadence step after the flat block ends, so the two abut.
NESTED_SUBDIRS = {"20200505_IQSEC2_97_98_99": "extra recordings"}

# Impedance records are found by the header string, not by filename. The lab's exports are
# spelled Impedence, Impendence and Impedences, and four carry no impedance word at all, so
# a *Impedence.csv glob finds nothing in five of the ten cohorts, including the only record
# covering animal 60.
IMPEDANCE_SNIFF = "Impedance Magnitude at 500 Hz (ohms)"

# An amplifier input reading at or above this is not connected to an electrode.
#
# Derived from the data rather than assumed. The 011320 cohort was measured twice on the
# same four implants two days apart and both sweeps are on disk, giving a test-retest
# spread of 0.652x to 1.590x (median 0.984x) on 32 declared electrodes. Across the dataset
# the 176 measured declared electrodes run 2.78e4 to 1.77e6, then there is a 5.757x gap,
# then four readings at 1.02e7 to 1.75e7. Only that gap is wider than the instrument's own
# reproducibility, so the cut goes in it. A 1.5e6 cut, which an earlier draft used, sits in
# a 1.276x gap, i.e. inside retest noise, and would have flagged a connected electrode.
OPEN_CIRCUIT_OHMS = 5e6


def _impedance_by_channel(cohort):
    """Return {channel name: magnitude in ohms} for a cohort, or {} if it has no record.

    A cohort may carry more than one sweep (011320 was re-measured for its second hookup).
    Impedance is genuinely re-measured per hookup rather than carried over, so the highest
    reading per channel is taken: a pin open in either hookup is not a usable electrode for
    the whole animal.

    A magnitude of exactly 0.0 means the impedance test was never run, not that the channel
    is perfect. Those are dropped here so a caller sees an absent key rather than a value
    that would pass any "below the threshold" comparison.
    """
    import csv

    worst = {}
    cohort_dir = Path(DATA_ROOT) / RHD_SUBTREE / cohort
    for path in sorted(cohort_dir.rglob("*.csv")):
        try:
            with path.open(errors="replace") as handle:
                if IMPEDANCE_SNIFF not in handle.readline():
                    continue
                handle.seek(0)
                for row in csv.DictReader(handle):
                    name = (row.get("Channel Name") or "").strip()
                    try:
                        ohms = float(row[IMPEDANCE_SNIFF])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not name or ohms == 0.0:
                        continue
                    worst[name] = max(worst.get(name, 0.0), ohms)
        except OSError:
            continue
    return worst


def _open_electrodes(cohort, port):
    """Return (open regions, per-region ohms, measured?) for one animal's declared montage.

    ``measured`` is False when the cohort carries no impedance record at all, or when the
    record covers none of this port's canonical amps. The two cases are reported the same
    way because neither is evidence that the electrodes are sound, and the caller must not
    silently treat an unmeasured animal as a clean one.
    """
    readings = _impedance_by_channel(cohort)
    seen = {}
    for amp, region in AMP_TO_REGION.items():
        ohms = readings.get(f"{port}-{amp}")
        if ohms is not None:
            seen[region] = ohms
    if not seen:
        return [], {}, False
    opens = sorted(
        (r for r, ohms in seen.items() if ohms >= OPEN_CIRCUIT_OHMS),
        key=REGION_ORDER.index,
    )
    return opens, seen, True


WORKBOOK = Path(DATA_ROOT) / RHD_SUBTREE / "IQSEC2 Compiled EEG data.xlsx"

# Cage number to headstage port. The trailing letter in a cage like '3B' is the rack, not
# the port, so only the leading digit is read.
CAGE_TO_PORT = {"1": "A", "2": "B", "3": "C", "4": "D"}

RHD_STAMP = re.compile(r"^(?P<session>.+_(?P<ymd>\d{6}))_(?P<hms>\d{6})\.rhd$")


def read_reference_sheet():
    """Return one record per animal from the workbook's EEG Reference Sheet.

    This sheet is the authority for genotype, sex and port. Folder-name genotypes
    contradict it and are not used anywhere.

    Sex is derived from the allele because the sheet has no sex column and the allele is
    the only systematic source: per the legend in the Demographic Info sheet, anything
    carrying a 'y' is hemizygous male and the two-X forms are female.
    """
    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    rows = list(workbook["EEG Reference Sheet"].iter_rows(values_only=True))

    out = {}
    for row in rows[1:]:
        if row[1] is None:
            continue
        animal_id = str(row[1]).strip()
        raw_allele = re.sub(r"\s+", " ", str(row[3]).strip()) if row[3] is not None else ""
        match = re.search(r"\(([^)]*)\)", raw_allele)
        allele = match.group(1) if match else None
        allele = ALLELE_CORRECTIONS.get(animal_id, allele)

        if allele is None:
            sex = None
        elif allele.endswith("/y"):
            sex = "Male"
        else:
            sex = "Female"

        cage = str(row[10]).strip() if row[10] is not None else ""
        digit = re.match(r"([1-4])", cage)
        out[animal_id] = {
            "id": animal_id,
            "allele": allele,
            "sex": sex,
            "cage": cage,
            "port": CAGE_TO_PORT.get(digit.group(1)) if digit else None,
            # The sheet's own Amp channel list, 1-based and absolute across the four ports.
            # Read only so the generator can check it against AMP_TO_REGION and the
            # cage-to-port rule, both of which are otherwise hand-transcribed.
            "amp_channel": str(row[11]).strip() if len(row) > 11 and row[11] is not None else None,
            "died": str(row[6]).strip() if row[6] is not None else None,
        }
    return out


PROBEINFO_POSITIONAL = re.compile(r"^([ABCD])-(\d{3})$")


def _probeinfo_region(raw):
    """Resolve one ProbeInfo string to a canonical abbreviation, or None.

    ProbeInfo is port-qualified, e.g. 'Intan Input (1)/PortC L Vis Ctx'. The tail is either
    a region word, in any of the case variants the lab's exports use, or a positional amp
    id like 'C-010' for files whose export dropped the region names.
    """
    _, _, tail = raw.partition("/")
    tail = tail.strip()
    if not tail.startswith("Port") or len(tail) < 6:
        return None
    tail = tail[5:].strip()
    positional = PROBEINFO_POSITIONAL.match(tail)
    if positional:
        return AMP_TO_REGION.get(positional.group(2))
    return DATAWAVE_REGION.get(tail.lower())


def _bin_channel_sets(animal_dir):
    """Return {abbrev tuple: [raw ProbeInfo tuples]} over an animal's LIVE bins.

    Zero-byte bins are skipped: the reader raises on them and the loader substitutes a
    0-sample placeholder that later stages drop, so they never reach the merge and their
    sidecars say nothing about the montage.
    """
    import csv

    out = defaultdict(list)
    for binary in sorted((Path(DATA_ROOT) / BIN_SUBTREE / animal_dir).rglob("*_ColMajor.bin")):
        if binary.stat().st_size == 0:
            continue
        sidecar = binary.with_name(binary.name.replace("_ColMajor.bin", "_Meta.csv"))
        if not sidecar.exists():
            continue
        rows = list(csv.DictReader(sidecar.open(errors="replace")))
        raw = tuple(row["ProbeInfo"].strip() for row in rows)
        abbrevs = tuple(_probeinfo_region(name) for name in raw)
        out[abbrevs].append(raw)
    return out


def _sidecars_without_lastedit(animal_dir):
    """Live sidecars whose trailing column has no header name, so LastEdit is unreadable.

    These are stale MATLAB writetable output. DDFBinaryMetadata checks for the column by
    name and falls back to dt_end=None with a warning, which is survivable for metadata but
    leaves that file with no end time to place it on the timeline.
    """
    out = []
    for binary in sorted((Path(DATA_ROOT) / BIN_SUBTREE / animal_dir).rglob("*_ColMajor.bin")):
        if binary.stat().st_size == 0:
            continue
        sidecar = binary.with_name(binary.name.replace("_ColMajor.bin", "_Meta.csv"))
        if sidecar.exists() and "LastEdit" not in sidecar.open(errors="replace").readline():
            out.append(sidecar.name)
    return out


def bin_probeinfo_by_region():
    """Return {abbrev: sorted raw ProbeInfo strings} across every included bin adult.

    Built from the sidecars rather than hand-written. The lab's exports carry three
    spellings of the same montage (title case, lowercase, and positional amp ids) and
    resolve_channel matches exactly with no fallback, so every spelling present in the data
    has to appear in the map or that file fails to resolve.

    Deliberately reads EVERY sidecar, including those whose bin is zero-byte and therefore
    never loads today. Sixty of the 73 bins in the 42/43/44/47 cohort are empty because an
    export was abandoned, their sources still exist as .ddf, and the lab has been asked to
    re-run it. If that happens those files become live, and several of them carry the
    lowercase spelling that no live file currently uses. Listing an alias that nothing
    resolves costs nothing; omitting one that something resolves is a load failure.
    """
    import csv

    out = defaultdict(set)
    for spec in BIN_ADULTS.values():
        for sidecar in sorted((Path(DATA_ROOT) / BIN_SUBTREE / spec["dir"]).rglob("*_Meta.csv")):
            for row in csv.DictReader(sidecar.open(errors="replace")):
                raw = (row.get("ProbeInfo") or "").strip()
                abbrev = _probeinfo_region(raw) if raw else None
                if abbrev:
                    out[abbrev].add(raw)
    return {k: sorted(v) for k, v in out.items()}


def _cohort_files(cohort):
    """Every rhd file in a cohort, at any depth."""
    return sorted((Path(DATA_ROOT) / RHD_SUBTREE / cohort).rglob("*.rhd"))


def scan_ports(cohort):
    """Return the amp-channel layout of a cohort, sampled across all of its sessions.

    Ports are not uniform: three cohorts expose 32, 64 or 96 channels rather than 128, so a
    global channel map would name channels that do not exist and raise on load.

    Sampled at the first, middle and last file of every session rather than at one file per
    cohort. One sample cannot see a port appearing or disappearing partway through, nor a
    sampling rate that changes between sessions, and either would silently invalidate the
    emitted channel_subset.

    Read through neo's header parser rather than spikeinterface's read_intan. Both read the
    same bytes, but read_intan builds a full extractor object that is then thrown away, at
    about 7 seconds per file against 0.06 for the header alone. Over the 125-file sample
    that is the difference between fifteen minutes and eight seconds, and it is the whole
    reason this script can read its sources directly instead of caching them. The header
    parser also has no integrity check to trip over, so the 49 files with short timestamp
    discontinuities need no special handling here.
    """
    from neo.rawio.intanrawio import read_rhd

    by_session = defaultdict(list)
    for path in _cohort_files(cohort):
        match = RHD_STAMP.match(path.name)
        by_session[match.group("session") if match else "UNPARSED"].append(path)

    sample = []
    for files in by_session.values():
        files.sort()
        for index in {0, len(files) // 2, len(files) - 1}:
            sample.append(files[index])

    ports, rates, widths, errors = set(), set(), set(), []
    for path in sample:
        try:
            global_info, channels = read_rhd(str(path), "header-attached")[:2]
            # signal_type 0 is the amplifier stream. The rest are aux inputs and supply
            # voltage, which would inflate the channel count by 16 and invent ports.
            ids = [c["native_channel_name"] for c in channels if c.get("signal_type") == 0]
            ports.update(name.split("-")[0] for name in ids)
            rates.add(float(global_info["sampling_rate"]))
            widths.add(len(ids))
        except Exception as exc:  # noqa: BLE001 - report and continue the census
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    return {
        "n_channels": max(widths) if widths else 0,
        "ports": sorted(ports),
        "fs": sorted(rates),
        "n_sampled": len(sample),
        "varies": {"n_channels": len(widths) > 1, "fs": len(rates) > 1},
        "errors": errors,
    }


def _scan_sessions(cohort, subdir=""):
    """Return {session: {"start", "end", "n_files"}} for one directory level of a cohort.

    Deliberately single-level rather than recursive. The roster scans recursively, which
    pools a nested subdirectory's files into the parent's session minima, so a flat source
    can inherit a start time belonging to a file the flat pattern will never discover. The
    emitted pattern is single-level, so the scan that feeds it must be too.

    The end time is derived from the last file's start stamp plus its size divided by the
    cohort's own bytes-per-second, which is taken from the modal file size over the modal
    inter-file interval. Sizes rather than headers, because two of the sessions that need
    an end time contain files neo refuses to open, so a header-based implementation would
    acquire a dependency on the very problem it is meant to survive.
    """
    stamp = re.compile(r"^(?P<session>.+_(?P<ymd>\d{6}))_(?P<hms>\d{6})\.rhd$")

    def _epoch(session, hms):
        ymd = session.rsplit("_", 1)[1]
        return (
            f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]} {hms[:2]}:{hms[2:4]}:{hms[4:6]}",
            _to_seconds(f"20{ymd[:2]}", ymd[2:4], ymd[4:6], hms[:2], hms[2:4], hms[4:6]),
        )

    found = {}
    for path in sorted((Path(DATA_ROOT) / RHD_SUBTREE / cohort / subdir).glob("*.rhd")):
        match = stamp.match(path.name)
        if not match:
            continue
        iso, secs = _epoch(match.group("session"), match.group("hms"))
        found.setdefault(match.group("session"), []).append((secs, iso, path.stat().st_size))

    # Bytes per second from the cohort's own modal file, so this holds for the 32, 64, 96
    # and 128 channel cohorts alike rather than assuming the 128 channel constant.
    starts = sorted(s for items in found.values() for s, _, _ in items)
    gaps = Counter(b - a for a, b in zip(starts, starts[1:]) if 0 < b - a <= 7200)
    sizes = Counter(sz for items in found.values() for _, _, sz in items)
    bps = None
    if gaps and sizes:
        modal_gap = gaps.most_common(1)[0][0]
        modal_size = sizes.most_common(1)[0][0]
        if modal_gap:
            bps = modal_size / modal_gap

    out = {}
    for session, items in found.items():
        items.sort()
        first_secs, first_iso, _ = items[0]
        last_secs, _, last_size = items[-1]
        tail = (last_size / bps) if bps else 0.0
        out[session] = {
            "start": first_iso,
            "start_secs": first_secs,
            "end_secs": last_secs + tail,
            "n_files": len(items),
        }
    return out


def _to_seconds(year, month, day, hour, minute, second):
    """Epoch seconds for a naive local timestamp, used only for interval arithmetic."""
    return calendar.timegm(
        (int(year), int(month), int(day), int(hour), int(minute), int(second), 0, 0, 0)
    )


def _day_end_secs(died):
    """Epoch seconds for the last instant of the death DATE.

    The sheet records a date with no time of day, so this is the latest moment the animal
    could still have been alive.
    """
    return _to_seconds(died[:4], died[5:7], died[8:10], 23, 59, 59)


# Animals with no reference-sheet row, so their metadata is assumed rather than recorded.
ASSUMED_METADATA = {"3": {"genotype": "x/x", "sex": "Female"}}


def _died_date(row):
    """Return the animal's death date as YYYY-MM-DD, or None if it has none.

    The sheet's Died/Euth column is free text: a real date, or a note like
    "9/27/19-headcap", or "perfused 6/9/20". Only an unambiguous date is used.
    """
    raw = (row or {}).get("died")
    if not raw:
        return None
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if iso:
        return iso.group(0)
    us = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", raw)
    if us:
        month, day, year = us.groups()
        year = int(year) + 2000 if len(year) == 2 else int(year)
        try:
            return datetime(year, int(month), int(day)).strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _sessions_before_death(sessions, died, extents):
    """Split sessions by where they sit relative to the animal's death date.

    Returns (keep, dropped, straddling). A session that begins after the death date is
    dropped: the rig kept running on an empty headstage, and that enters analysis as EEG
    indistinguishable from signal. A session that begins on or before the date but ends
    after it is KEPT and reported, because the sheet records a date with no time of day,
    so how much of it is post-mortem is genuinely unknown between none and all of it.

    An earlier version compared start dates only, which silently kept a full 24 hour
    session for animal 81 under the rationale that these animals were perfused at the end
    of recording. That rationale holds for 82, 83 and 60 and fails for 81, the only animal
    the rule acts on: its retained block runs 22 minutes past the death date, and the three
    sessions around it are one unbroken 43 hour acquisition split by the filename's
    midnight rollover, with inter-session gaps of 1.0 and 2.1 seconds.
    """
    if not died:
        return sessions, [], []
    cutoff = _day_end_secs(died)
    keep, dropped, straddling = {}, [], []
    for key, start in sessions.items():
        span = extents.get(key)
        if span is None:
            # No extent available: fall back to the start-date comparison rather than
            # guessing, and let the caller see it as an ordinary keep.
            (keep.__setitem__(key, start) if start[:10] <= died else dropped.append(key))
            continue
        if span["start_secs"] > cutoff:
            dropped.append(key)
            continue
        keep[key] = start
        if span["end_secs"] > cutoff:
            straddling.append(key)
    return keep, sorted(dropped), sorted(straddling)


def _lro_kwargs_lines(indent):
    """Emit the per-animal lro_kwargs block at the given indent.

    One helper serves both emission sites (the flat animal and the per-source list item)
    so a reader option cannot be added to one and silently missed by the other.

    ignore_integrity_checks is on because 49 of the 9867 rhd files carry short timestamp
    discontinuities that read_intan otherwise refuses to open, which aborts 10 of the 23
    animals before any data is read. Measured across all 49 files: 70 real sample drops,
    of 1 sample (64 of them), 2 samples (3) and 3 samples (3), so 0.5 to 1.5 ms at the
    native 2000 Hz, totalling 0.0395 s across the whole dataset. The flag splices those
    gaps shut rather than splitting the recording. See the header for the residual.
    """
    pad = " " * indent
    return [
        f"{pad}lro_kwargs:",
        f"{pad}  extract_func: \"read_intan\"",
        f"{pad}  stream_id: \"0\"",
        f"{pad}  datetimes_are_start: true",
        f"{pad}  ignore_integrity_checks: true",
    ]


def q(value):
    """Quote a YAML scalar when it would otherwise be misparsed."""
    text = str(value)
    if text != text.strip() or any(c in text for c in ":#{}[],&*?|>'\"%@`") or text == "":
        return "'" + text.replace("'", "''") + "'"
    return text


PORT_BASE = {"A": 0, "B": 32, "C": 64, "D": 96}


def _decode_amp_channels(raw, port):
    """Decode the sheet's Amp channel cell into a set of amp offsets, or None.

    The cell is 1-based and absolute across all four ports, written as a mixed list and
    range, e.g. '(75,77,79-82,84,86)' for a port C animal. Subtracting one for the 1-based
    numbering and the port's 32-channel base yields the offsets used everywhere else, so
    animal 6's cell above decodes to 10,12,14,15,16,17,19,21.

    This is an independent witness: it confirms AMP_TO_REGION and the cage-to-port rule at
    once, using a column no other part of the generator reads. Returns None when the cell
    is absent or unparseable, so a missing cell is not mistaken for a mismatch.
    """
    if not raw or port not in PORT_BASE:
        return None
    # Some rows carry prose in this cell rather than a channel list, e.g. a pup row reading
    # '4 hr Video only-start time10:02AM'. Digits scraped out of prose decode to nonsense
    # and would raise a false mismatch, so any letter disqualifies the cell outright.
    if any(ch.isalpha() for ch in raw):
        return None
    numbers = []
    for chunk in re.findall(r"\d+\s*-\s*\d+|\d+", raw):
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-"))
            if hi < lo or hi - lo > 64:
                return None
            numbers.extend(range(lo, hi + 1))
        else:
            numbers.append(int(chunk))
    if not numbers:
        return None
    base = PORT_BASE[port]
    offsets = {n - 1 - base for n in numbers}
    if any(o < 0 or o > 31 for o in offsets):
        return None
    return {f"{o:03d}" for o in offsets}


def _write_atomic(path, text):
    """Write the config only once every check has passed.

    Earlier revisions printed to stdout and expected the caller to redirect into the
    config. The shell truncates the target before the interpreter starts, so a refusal, a
    missing roster or any other exception left the tracked config at zero bytes, turning a
    protective guard into a destructive one. A sibling temp file plus a rename means a
    failed run leaves the previous config exactly as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the IQSEC2 adult dataset config.")
    parser.add_argument("--out", type=Path, required=True, help="path to write the yaml to")
    args = parser.parse_args(argv)

    if not WORKBOOK.exists():
        raise SystemExit(f"Reference workbook not found: {WORKBOOK}")

    # Read every source directly. Slow (a recursive walk of roughly ten thousand files plus
    # a header sample) but there is no cached copy of any of this to go stale.
    print(f"reading {WORKBOOK.name} ...")
    ref = read_reference_sheet()
    ports_seen = {}
    for cohort in RHD_COHORTS:
        print(f"scanning {cohort} ...")
        ports_seen[cohort] = scan_ports(cohort)

    # Verify every hand-transcribed port against both the sheet and the file itself.
    problems = []
    for cohort, animals in RHD_COHORTS.items():
        census = ports_seen.get(cohort, {})
        # The census reads with ignore_integrity_checks, so anything still failing is a
        # genuinely unreadable file rather than one of the known timestamp discontinuities.
        for failure in census.get("errors", []):
            problems.append(f"{cohort}: unreadable even with ignore_integrity_checks: {failure}")
        if census.get("varies", {}).get("n_channels"):
            problems.append(f"{cohort}: amp channel count varies across sessions")
        if census.get("varies", {}).get("fs"):
            problems.append(f"{cohort}: sampling rate varies across sessions: {census.get('fs')}")
        available = set(census.get("ports", []))
        for animal_id, port in animals.items():
            if port not in available:
                problems.append(f"{cohort}/{animal_id}: port {port} not among {sorted(available)}")
            row = ref.get(animal_id)
            if row is None:
                if animal_id not in ASSUMED_METADATA:
                    problems.append(f"{cohort}/{animal_id}: no reference-sheet row and no assumed metadata")
            elif row["port"] != port:
                problems.append(
                    f"{cohort}/{animal_id}: sheet says port {row['port']}, config says {port}"
                )
            if row is not None:
                amps = _decode_amp_channels(row.get("amp_channel"), port)
                if amps is not None and amps != set(AMP_TO_REGION):
                    problems.append(
                        f"{cohort}/{animal_id}: the sheet's Amp channel cell "
                        f"{row['amp_channel']!r} decodes on port {port} to "
                        f"{sorted(amps)}, not the canonical {sorted(AMP_TO_REGION)}. "
                        f"Either the port is wrong or this animal has a different montage."
                    )
        # Reverse direction. The check above only proves that every port this file CLAIMS
        # is real; it says nothing about a port the hardware exposes that nobody claimed,
        # which is how an animal goes missing without any check noticing.
        for port in sorted(available - set(animals.values())):
            if (cohort, port) not in PORT_EXCLUSIONS:
                problems.append(
                    f"{cohort}/Port{port}: exposed by the hardware, assigned to no animal, "
                    f"and absent from PORT_EXCLUSIONS. Either assign it or record why it "
                    f"is empty."
                )

    # The bin adults. Every ProbeInfo string must resolve, the montage must be consistent
    # across an animal's live files (a 7-channel and an 8-channel file in one animal raise
    # at merge rather than combining), and the port the sidecars declare must match the one
    # the sheet's cage implies.
    for animal_id, spec in BIN_ADULTS.items():
        sets = _bin_channel_sets(spec["dir"])
        if not sets:
            problems.append(f"bin/{animal_id}: no live bins under {spec['dir']}")
            continue
        for abbrevs, raws in sets.items():
            if None in abbrevs:
                unresolved = {r for names in raws for r, a in zip(names, abbrevs) if a is None}
                problems.append(f"bin/{animal_id}: unresolvable ProbeInfo {sorted(unresolved)}")
        distinct = {tuple(sorted(a for a in abbrevs if a)) for abbrevs in sets}
        if len(distinct) > 1:
            problems.append(
                f"bin/{animal_id}: live files declare {len(distinct)} different montages "
                f"{sorted(distinct)}; they would raise at merge rather than combine"
            )
        expected = CAGE_TO_PORT[spec["cage"]]
        declared = {
            name.split("/")[1][4:5]
            for names in (r for raws in sets.values() for r in raws)
            for name in names
            if "/" in name
        }
        if declared != {expected}:
            problems.append(
                f"bin/{animal_id}: sheet cage {spec['cage']} implies port {expected}, "
                f"sidecars declare {sorted(declared)}"
            )
        if animal_id not in ref:
            problems.append(f"bin/{animal_id}: no reference-sheet row")

    # Every cohort on disk must be accounted for. Without this a renamed or newly added
    # cohort directory is silently dropped from the study.
    on_disk = {
        p.name for p in (Path(DATA_ROOT) / RHD_SUBTREE).iterdir()
        if p.is_dir() and any(p.rglob("*.rhd"))
    }
    for name in sorted(on_disk - set(RHD_COHORTS) - PUP_COHORTS):
        problems.append(f"{name}: cohort holds rhd files but is in neither RHD_COHORTS nor PUP_COHORTS")

    # Every genotype that will be emitted must have a GENOTYPE_MAP entry. A populated map
    # is authoritative and raises on any value it does not cover, so an allele appearing in
    # the sheet but not in the literal below would be a load-time failure, not a warning.
    for cohort, animals in RHD_COHORTS.items():
        for animal_id in animals:
            assumed = ASSUMED_METADATA.get(animal_id, {})
            genotype = assumed.get("genotype") or ref.get(animal_id, {}).get("allele") or "Unknown"
            if genotype not in GENOTYPE_LEVELS:
                problems.append(
                    f"{cohort}/{animal_id}: genotype {genotype!r} has no GENOTYPE_MAP entry"
                )

    if problems:
        raise SystemExit("Refusing to generate:\n  " + "\n  ".join(problems))

    out = []
    w = out.append
    w("# IQSEC2 - adult half (Intan rhd cohorts)")
    w("#")
    w("# GENERATED by scripts/iqsec2/generate_config.py. Regenerate rather than hand-editing;")
    w("# the port map comes from 'IQSEC2 Compiled EEG data.xlsx' sheet 'EEG Reference Sheet'")
    w("# and the session start times from the rhd filenames.")
    w("#")
    w("# Ports come from the sheet's Cage column (1->A 2->B 3->C 4->D; the trailing letter")
    w("# in '3B' is the rack, not the port). Folder-name genotypes contradict the sheet and")
    w("# are not used. Positional assignment (folder-name order maps to A,B,C,D) is wrong in")
    w("# six of the ten cohorts because several start at cage 3, and is never used here:")
    w("# 20180917, IQSEC 65_66, 013120, 012720, 20200602 and IQSEC animal 3.")
    w("#")
    w("# lro_kwargs carries ignore_integrity_checks. 49 of the 9867 rhd files hold short")
    w("# timestamp discontinuities that read_intan otherwise refuses to open, which aborts")
    w("# 10 of the 23 animals. Measured across all 49: 70 sample drops of 1 to 3 samples,")
    w("# 0.5 to 1.5 ms each at 2000 Hz, 0.0395 s total for the whole dataset. The flag")
    w("# splices them shut. Two residuals are worth knowing. The splice is not amplitude")
    w("# free: the step at the join is a median 199 uV and a worst case 6921 uV, each")
    w("# contaminating exactly one 5 s window, so at worst 0.055 percent of one animal's")
    w("# windows. And the dataset's real timeline error is elsewhere and much larger:")
    w("# 20200609_..._200609 contains a cross-file timestamp restart of 75.2 s that")
    w("# accumulates to about 109 s of drift, which no per-file check can see because neo")
    w("# validates continuity within a file only.")
    w("#")
    w("# This config covers the adult half only. The pup cohorts need their own config and")
    w("# it does not exist yet: 32 ProbeInfo strings occur in both pup and adult cohorts and")
    w("# would need different abbreviations in each, which set_channel_map refuses. That")
    w("# constraint belongs to the DataWave ProbeInfo key space; every raw name in THIS")
    w("# file is a bare Intan id, so it does not bind here.")
    w("#")
    w("# Caveat on the female arm, recorded because a figure will not show it: IQSEC2_3 is")
    w("# the only x/x animal, so it is the baseline every female difference heatmap is")
    w("# normalized against. It has no reference-sheet row, its genotype and sex are both")
    w("# assumed, its cohort has no impedance record at all, and 25 of its 142 files are")
    w("# spliced. The female arm is 7 het, 1 mutant and that 1 assumed control. The male")
    w("# contrast, 8 mutant against 6 control, is the result that stands on its own.")
    w("")
    w("analysis:")
    w("  war_generation:")
    w("    # Deliberately minimal: the per-animal merge is a shallow update that can override")
    w("    # but never delete, so anything reader-specific here would leak into every animal.")
    w("    pattern: \"{index}.rhd\"  # placeholder; every animal overrides it")
    w("    lro_kwargs:")
    w("      mode: \"si\"")
    w("")
    w("samples_data:")
    w(f"  data_root: {DATA_ROOT}")
    w("")
    w("  # The allele notation IS the genotype vocabulary. Per the legend in the workbook's")
    w("  # 'Demographic Info' sheet, '+' is the MUTANT allele and 'x' is the wildtype X:")
    w("  #   (+/y) mutant male   (+/+) mutant female   (+/x) het female   (x/y) control male")
    w("  # Reading these by the usual convention would label 14 mutants as wildtype.")
    w("  GENOTYPE_MAP:")
    w("    Unknown: [Unknown, unknown, UNKNOWN]")
    for allele in GENOTYPE_LEVELS:
        if allele != "Unknown":
            w(f"    {q(allele)}: [{q(allele)}]")
    w("  SEX_MAP:")
    w("    Unknown: [Unknown, unknown, UNKNOWN]")
    w("    Male: [Male, male, M, m]")
    w("    Female: [Female, female, F, f]")
    w("")
    w("  # Two key spaces, because this dataset has two acquisition systems. The Intan rhd")
    w("  # cohorts expose bare ids like 'C-015'. The DataWave bin cohorts expose")
    w("  # port-prefixed ProbeInfo strings like 'Intan Input (1)/PortC L Motor Ctx', and")
    w("  # the lab's exports spell those three ways: title case, lowercase, and positional")
    w("  # amp ids where the export dropped the region words. resolve_channel matches")
    w("  # exactly with no fallback, so every spelling present in the data must be listed.")
    w("  # The bin entries are generated from the sidecars, not hand-written.")
    w("  channels:")
    bin_names = bin_probeinfo_by_region()
    for region in REGION_ORDER:
        amp = next(a for a, r in AMP_TO_REGION.items() if r == region)
        w(f"    {region}:")
        for port in "ABCD":
            w(f"    - {port}-{amp}")
        for raw in bin_names.get(region, ()):
            w(f"    - {q(raw)}")
    w("")
    w("  # One anchor per port keeps each animal to a single line instead of eight.")
    w("  _ports:")
    for port in "ABCD":
        ids = ", ".join(f"'{port}-{a}'" for a in sorted(AMP_TO_REGION))
        w(f"    port_{port.lower()}: &port_{port.lower()} [{ids}]")
    w("")
    w("  animals:")

    for cohort in RHD_COHORTS:
        n_files = len(_cohort_files(cohort))
        # Scanned here rather than taken from the roster: the roster scans recursively, so
        # for a cohort with a nested subdirectory its session minima pool both depths and a
        # flat source can inherit a start time from a file its pattern never reaches.
        flat_extents = _scan_sessions(cohort)
        sessions_all = {k: v["start"] for k, v in flat_extents.items()}
        w(f"  # --- {cohort}: {n_files} files, {len(sessions_all)} sessions, "
          f"{ports_seen[cohort]['n_channels']} amp channels on "
          f"{','.join(ports_seen[cohort]['ports'])}")
        for (exc_cohort, exc_port), reason in PORT_EXCLUSIONS.items():
            if exc_cohort != cohort:
                continue
            for line in _wrap(f"Port{exc_port} not loaded. {reason}", 84):
                w(f"  #   {line}")
        if cohort == "IQSEC animal 3":
            for line in _wrap(
                "No reference-sheet row: genotype x/x and sex Female are ASSUMED. "
                "20180522_Impedence.xlsx labels this animal's channels as an exact L/R mirror "
                "of all 49 other impedance workbooks, but those labels are typed by hand and "
                "are never read by the pipeline; the rhd header carries only bare ids, so the "
                "standard montage applies. If the implant really was mirrored, every "
                "lateralized result for this animal is inverted and nothing would catch it.",
                84,
            ):
                w(f"  #   {line}")

        for animal_id, port in RHD_COHORTS[cohort].items():
            row = ref.get(animal_id, {})
            died = _died_date(row)
            sessions, dropped, straddling = _sessions_before_death(
                sessions_all, died, flat_extents
            )
            # A nested source is death-filtered on its own scan. An earlier version filtered
            # only the flat arm while emitting skip_sessions at animal level, where it
            # applies to every source, so a post-death nested session kept its datetime and
            # lost its files, which raises on the manual_datetime key check.
            extra = NESTED_SUBDIRS.get(cohort)
            nested_sessions = {}
            if extra:
                nested_extents = _scan_sessions(cohort, extra)
                nested_all = {k: v["start"] for k, v in nested_extents.items()}
                nested_sessions, nested_dropped, nested_straddling = _sessions_before_death(
                    nested_all, died, nested_extents
                )
                dropped = sorted(set(dropped) | set(nested_dropped))
                straddling = sorted(set(straddling) | set(nested_straddling))
            if dropped:
                for line in _wrap(
                    f"Animal {animal_id} died {died} per the reference sheet. "
                    f"{len(dropped)} session(s) starting after that date are excluded: the rig "
                    f"kept recording other animals, so this port is an empty headstage there. "
                    f"Dropped: {', '.join(dropped)}",
                    84,
                ):
                    w(f"  #   {line}")
            for key in straddling:
                span = flat_extents.get(key) or nested_extents[key]
                over = (span["end_secs"] - _day_end_secs(died)) / 60.0
                for line in _wrap(
                    f"KEPT BUT UNVERIFIED: session {key} starts on or before animal "
                    f"{animal_id}'s recorded death date ({died}) and runs about "
                    f"{over:.0f} minutes past the end of it, across {span['n_files']} "
                    f"files. The sheet records a date with no time of day, so the "
                    f"post-mortem fraction of this session is unknown: it is somewhere "
                    f"between none of it and all of it. It is retained rather than "
                    f"dropped because dropping it would discard real data on a guess, "
                    f"but any result leaning on this animal should be checked against it.",
                    84,
                ):
                    w(f"  #   {line}")
            if animal_id in ANIMAL_NOTES:
                for line in _wrap(ANIMAL_NOTES[animal_id], 84):
                    w(f"  #   {line}")
            bin_cohort = DUAL_TREE_BINS.get(animal_id)
            if bin_cohort:
                share = sum(1 for a in DUAL_TREE_BINS.values() if a == bin_cohort)
                live = [
                    p for p in (Path(DATA_ROOT) / BIN_SUBTREE / bin_cohort).rglob("*_ColMajor.bin")
                    if p.stat().st_size > 0
                ]
                # 8 channels, float32, 2000 Hz, per the Meta.csv sidecars.
                hours = sum(p.stat().st_size for p in live) / (8 * 4 * 2000) / 3600 / share
                for line in _wrap(
                    f"INCOMPLETE RECORD: this animal's implant was also recorded by the "
                    f"DataWave rig, leaving roughly {len(live) // share} more bin files "
                    f"and about {hours:.0f} more hours under "
                    f"'{BIN_SUBTREE}/{bin_cohort}' that no pattern in this config reaches. "
                    f"The two legs abut rather than overlap. What is loaded here is real "
                    f"and correctly attributed, but it is not the whole recording, and "
                    f"coverage is uneven across animals for that reason.",
                    84,
                ):
                    w(f"  #   {line}")
            opens, ohms_by_region, measured = _open_electrodes(cohort, port)
            if not measured:
                for line in _wrap(
                    f"Animal {animal_id} has NO impedance record: its cohort carries no "
                    f"impedance export, and the rhd headers report an impedance test "
                    f"frequency of 0 with a magnitude of 0 on every channel, so the test "
                    f"was never run. The montage below is therefore unverified hardware, "
                    f"not verified-good hardware. Do not read the absence of bad_channels "
                    f"here as evidence that these electrodes were sound.",
                    84,
                ):
                    w(f"  #   {line}")
            elif opens:
                detail = ", ".join(
                    f"{region} ({port}-{amp}) {ohms_by_region[region]:.2e} ohm"
                    for amp, region in sorted(AMP_TO_REGION.items())
                    if region in opens
                )
                for line in _wrap(
                    f"Animal {animal_id}: {len(opens)} declared electrode(s) read at or "
                    f"above {OPEN_CIRCUIT_OHMS:.0e} ohm in this cohort's own impedance "
                    f"export and are unconnected amplifier inputs, not electrodes: "
                    f"{detail}. Declared bad below. Nothing else in the pipeline would "
                    f"remove them: the manual channel filter is the only gate on the "
                    f"ep_analysis path, and LOF scores a dead channel around 1.1 against "
                    f"a threshold of 2.5 because its distance metric scales with amplitude.",
                    84,
                ):
                    w(f"  #   {line}")

            assumed = ASSUMED_METADATA.get(animal_id, {})
            genotype = assumed.get("genotype") or row.get("allele") or "Unknown"
            sex = assumed.get("sex") or row.get("sex") or "Unknown"
            w(f"  - id: IQSEC2_{animal_id}")
            w(f"    genotype: {q(genotype)}")
            w(f"    sex: {q(sex)}")
            w(f"    channel_subset: *port_{port.lower()}")
            if opens:
                # Flat list, not the per-session dict form: expand_animals_config turns a
                # list into {"_all": [...]}, which collapses into the session-independent
                # reject list. The dict form instead requires every animalday to be listed
                # and raises on any that is not.
                w("    bad_channels:")
                for region in opens:
                    w(f"    - {region}")
            if dropped:
                # Removing the datetime is not enough: discovery still finds the files and
                # then requires a datetime for every session it found. Skip at discovery.
                w("    skip_sessions:")
                for key in dropped:
                    w(f"    - {q(key)}")
            if extra:
                # This cohort's files sit at two depths, which no single glob can span:
                # ** cannot cross a separator here, and a pattern list is an AND-pair
                # rather than a union. Two sources reach both.
                w("    sources:")
                for label, subpath in (("flat", ""), ("nested", f"/{extra}")):
                    w(f"    # {label}")
                    w(f"    - pattern: \"{{data_root}}/{RHD_SUBTREE}/{cohort}{subpath}/{{session}}_{{index}}.rhd\"")
                    for line in _lro_kwargs_lines(6):
                        w(line)
                    w("      manual_datetime:")
                    subset = nested_sessions if subpath else sessions
                    for key, start in subset.items():
                        w(f"        {q(key)}: {q(start)}")
            else:
                w(f"    pattern: \"{{data_root}}/{RHD_SUBTREE}/{cohort}/{{session}}_{{index}}.rhd\"")
                for line in _lro_kwargs_lines(4):
                    w(line)
                w("    manual_datetime:")
                for key, start in sessions.items():
                    w(f"      {q(key)}: {q(start)}")
    # --- DataWave bin adults ---
    w("  # === DataWave bin cohorts (no rhd leg at all) ===")
    for line in _wrap(
        "These six animals were recorded only by the DataWave rig. They are here because "
        "they are what makes the male genotype contrast interpretable: with the rhd "
        "cohorts alone every control male falls in a 35 day window in 2020 and only one "
        "of eight mutants falls inside it, so genotype and recording date are very nearly "
        "the same variable and any difference between the groups could equally be a "
        "difference between two eras of the rig. These animals put controls in early 2019 "
        "as well, so the two groups interleave. That improves the design; it does not make "
        "it a designed experiment, and recording date is still worth carrying as a "
        "covariate.",
        84,
    ):
        w(f"  #   {line}")
    for animal_id, reason in BIN_EXCLUSIONS.items():
        for line in _wrap(f"Animal {animal_id} not loaded. {reason}", 84):
            w(f"  #   {line}")

    for animal_id, spec in BIN_ADULTS.items():
        row = ref.get(animal_id, {})
        port = CAGE_TO_PORT[spec["cage"]]
        sets = _bin_channel_sets(spec["dir"])
        n_live = sum(len(v) for v in sets.values())
        hours = sum(
            p.stat().st_size for p in (Path(DATA_ROOT) / BIN_SUBTREE / spec["dir"]).rglob("*_ColMajor.bin")
            if p.stat().st_size > 0
        ) / (8 * 4 * 2000) / 3600
        spellings = len({raw for raws in sets.values() for names in raws for raw in names})
        for line in _wrap(
            f"Animal {animal_id}: {n_live} live bins, about {hours:.0f} h, Port{port} from "
            f"sheet cage {spec['cage']} and confirmed by the sidecars. {spellings} distinct "
            f"ProbeInfo spellings of the same 8 electrodes, all mapped above. No "
            f"channel_subset: unlike the rhd cohorts, where one file holds four animals' "
            f"ports, each bin file holds only this animal.",
            84,
        ):
            w(f"  #   {line}")
        stale = _sidecars_without_lastedit(spec["dir"])
        if stale:
            for line in _wrap(
                f"WATCH: {len(stale)} live sidecar(s) here are stale MATLAB writetable "
                f"output whose final column has no name, so LastEdit cannot be read: "
                f"{', '.join(stale)}. These bins carry no timestamps of their own and no "
                f"manual_datetime covers them, so DDFBinaryMetadata sets dt_end to None "
                f"and only warns. Whether the timeline survives a None among its files has "
                f"not been tested, because these are multi-GB files and testing it needs a "
                f"real load on the cluster. If war_generation fails for this animal, look "
                f"here first.",
                84,
            ):
                w(f"  #   {line}")
        w(f"  - id: IQSEC2_{animal_id}")
        w(f"    genotype: {q(row.get('allele') or 'Unknown')}")
        w(f"    sex: {q(row.get('sex') or 'Unknown')}")
        base = f"{{data_root}}/{BIN_SUBTREE}/{spec['dir']}"
        # 51's files sit under Part 1 and Part 2, which serve as the session key. The rest
        # are flat, so their own directory is the session.
        if spec.get("nested"):
            stem = f"{base}/{{session}}/Cage {spec['cage']}-{{index}}"
        else:
            parent, _, leaf = spec["dir"].rpartition("/")
            stem = f"{{data_root}}/{BIN_SUBTREE}/{parent}/{{session}}/Cage {spec['cage']}-{{index}}"
        w("    pattern:")
        w(f"    - \"{stem}_ColMajor.bin\"")
        w(f"    - \"{stem}_Meta.csv\"")
        w("    lro_kwargs:")
        w("      mode: \"si\"")
        w("      extract_func: \"neurodent.readers:read_bin_csv_pair\"")
        w("      multiprocess_mode: \"serial\"")
        # LastEdit in the sidecar is the recording END, verified against the .ddf mtimes.
        w("      datetimes_are_start: false")

    _write_atomic(args.out, "\n".join(out) + "\n")
    print(f"wrote {args.out} ({len(out)} lines)")


def _wrap(text, width):
    words, line, lines = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


if __name__ == "__main__":
    main()
