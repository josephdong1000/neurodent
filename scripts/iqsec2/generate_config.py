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

Usage:
    uv run python scripts/iqsec2/generate_config.py > config/datasets/iqsec2.yaml
"""

import json
import sys
from pathlib import Path

ROSTER = Path("results/iqsec2_roster.json")
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

# Animals present in a cohort's files but deliberately not loaded, with the reason.
EXCLUSIONS = {
    "20200609_IQSEC2_114_117_118": (
        "117 and 115 were implanted but never recorded: the reference sheet gives both "
        "Recording Start, Recording End, Cage and Amp channel as N/A, with 115 marked "
        "Died and 117 perfused 2020-06-09. The 59.5 GB file named 117 is a cage video. "
        "PortB is 92.4 percent line noise at the median across 109 timepoints. PortD "
        "carries no animal."
    ),
    "IQSEC 65_66": (
        "PortB carries 7 connected channels in a non-canonical set "
        "(14,15,16,18,19,20,22 rather than 10,12,14,15,16,17,19,21) with no "
        "reference-sheet row and no name anywhere. Unidentified, so not loaded."
    ),
    "012720_IQSEC2_77": (
        "PortD is a CDKL5 animal: these 140 rhd are a joint recording filed identically "
        "under PythonEEG Data/CDKL5. PortA is an empty headstage. Only PortC is IQSEC2."
    ),
}

# Cohorts whose rhd files sit at two depths. No single glob spans them: ** cannot cross a
# separator in this discovery layer, and a pattern list is an AND-pair, not a union. Each
# such cohort gets two sources instead. The nested block is a genuine continuation, not a
# quarantine: it begins one cadence step after the flat block ends, so the two abut.
NESTED_SUBDIRS = {"20200505_IQSEC2_97_98_99": "extra recordings"}


def _sessions_under(cohort, subdir):
    """Session start times for the files inside a cohort's nested subdirectory."""
    import re
    from pathlib import Path as _P
    stamp = re.compile(r"^(?P<session>.+_(?P<ymd>\d{6}))_(?P<hms>\d{6})\.rhd$")
    found = {}
    for path in sorted((_P(DATA_ROOT) / RHD_SUBTREE / cohort / subdir).glob("*.rhd")):
        m = stamp.match(path.name)
        if not m:
            continue
        found.setdefault(m.group("session"), []).append(m.group("hms"))
    out = {}
    for session, stamps in found.items():
        ymd = session.rsplit("_", 1)[1]
        hms = min(stamps)
        out[session] = f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]} {hms[:2]}:{hms[2:4]}:{hms[4:6]}"
    return out


# Animals with no reference-sheet row, so their metadata is assumed rather than recorded.
ASSUMED_METADATA = {"3": {"genotype": "x/x", "sex": "Female"}}


def _died_date(row):
    """Return the animal's death date as YYYY-MM-DD, or None if it has none.

    The sheet's Died/Euth column is free text: a real date, or a note like
    "9/27/19-headcap", or "perfused 6/9/20". Only an unambiguous date is used.
    """
    import re
    from datetime import datetime

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


def _sessions_before_death(sessions, died):
    """Drop sessions starting strictly AFTER the animal died.

    A session ON the death date is kept: these animals were perfused at the end of
    recording, so the last session and the death share a date. A session after it is
    the rig still running on an empty headstage, which enters analysis as EEG and is
    indistinguishable from signal once it reaches the feature extractor.
    """
    if not died:
        return sessions, []
    keep = {k: v for k, v in sessions.items() if v[:10] <= died}
    dropped = sorted(set(sessions) - set(keep))
    return keep, dropped


def q(value):
    """Quote a YAML scalar when it would otherwise be misparsed."""
    text = str(value)
    if text != text.strip() or any(c in text for c in ":#{}[],&*?|>'\"%@`") or text == "":
        return "'" + text.replace("'", "''") + "'"
    return text


def main():
    data = json.loads(ROSTER.read_text())
    ref = {r["id"]: r for r in data["reference_sheet"]}
    cohorts = data["rhd_cohorts"]
    ports_seen = data["rhd_ports"]

    # Verify every hand-transcribed port against both the sheet and the file itself.
    problems = []
    for cohort, animals in RHD_COHORTS.items():
        available = set(ports_seen.get(cohort, {}).get("ports", []))
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
    if problems:
        raise SystemExit("Refusing to generate, port mapping disagrees with its sources:\n  "
                         + "\n  ".join(problems))

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
    w("# are NOT used. Positional assignment (folder-name order maps to A,B,C,D) is wrong in")
    w("# five cohorts because several start at cage 3, and is never used here.")
    w("#")
    w("# Pups live in a separate config: 32 ProbeInfo strings occur in both pup and adult")
    w("# cohorts and would need different abbreviations in each, which set_channel_map")
    w("# refuses. See config/datasets/iqsec2_pup.yaml.")
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
    for allele in ["+/y", "x/y", "+/x", "+/+", "-/y", "x/x"]:
        w(f"    {q(allele)}: [{q(allele)}]")
    w("  SEX_MAP:")
    w("    Unknown: [Unknown, unknown, UNKNOWN]")
    w("    Male: [Male, male, M, m]")
    w("    Female: [Female, female, F, f]")
    w("")
    w("  # rhd exposes bare Intan ids, so the map is keyed on those. The bin half uses")
    w("  # port-prefixed ProbeInfo strings and is a different key space; it is not present")
    w("  # here because this config covers the rhd cohorts only.")
    w("  channels:")
    for region in REGION_ORDER:
        amp = next(a for a, r in AMP_TO_REGION.items() if r == region)
        w(f"    {region}:")
        for port in "ABCD":
            w(f"    - {port}-{amp}")
    w("")
    w("  # One anchor per port keeps each animal to a single line instead of eight.")
    w("  _ports:")
    for port in "ABCD":
        ids = ", ".join(f"'{port}-{a}'" for a in sorted(AMP_TO_REGION))
        w(f"    port_{port.lower()}: &port_{port.lower()} [{ids}]")
    w("")
    w("  animals:")

    for cohort in RHD_COHORTS:
        info = cohorts[cohort]
        sessions_all = info["sessions"]
        note = EXCLUSIONS.get(cohort)
        w(f"  # --- {cohort}: {info['n_files']} files, {len(sessions_all)} sessions, "
          f"{ports_seen[cohort]['n_channels']} amp channels on "
          f"{','.join(ports_seen[cohort]['ports'])}")
        if note:
            for line in _wrap(note, 84):
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
            sessions, dropped = _sessions_before_death(sessions_all, died)
            if dropped:
                for line in _wrap(
                    f"Animal {animal_id} died {died} per the reference sheet. "
                    f"{len(dropped)} session(s) starting after that date are excluded: the rig "
                    f"kept recording other animals, so this port is an empty headstage there. "
                    f"Dropped: {', '.join(dropped)}",
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
            if dropped:
                # Removing the datetime is not enough: discovery still finds the files and
                # then requires a datetime for every session it found. Skip at discovery.
                w("    skip_sessions:")
                for key in dropped:
                    w(f"    - {q(key)}")
            extra = NESTED_SUBDIRS.get(cohort)
            if extra:
                # This cohort's files sit at two depths, which no single glob can span:
                # ** cannot cross a separator here, and a pattern list is an AND-pair
                # rather than a union. Two sources reach both.
                w("    sources:")
                for label, subpath in (("flat", ""), ("nested", f"/{extra}")):
                    w(f"    # {label}")
                    w(f"    - pattern: \"{{data_root}}/{RHD_SUBTREE}/{cohort}{subpath}/{{session}}_{{index}}.rhd\"")
                    w("      lro_kwargs:")
                    w("        extract_func: \"read_intan\"")
                    w("        stream_id: \"0\"")
                    w("        datetimes_are_start: true")
                    w("      manual_datetime:")
                    subset = sessions if not subpath else _sessions_under(cohort, extra)
                    for key, start in subset.items():
                        w(f"        {q(key)}: {q(start)}")
            else:
                w(f"    pattern: \"{{data_root}}/{RHD_SUBTREE}/{cohort}/{{session}}_{{index}}.rhd\"")
                w("    lro_kwargs:")
                w("      extract_func: \"read_intan\"")
                w("      stream_id: \"0\"")
                w("      datetimes_are_start: true")
                w("    manual_datetime:")
                for key, start in sessions.items():
                    w(f"      {q(key)}: {q(start)}")
    print("\n".join(out))


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
