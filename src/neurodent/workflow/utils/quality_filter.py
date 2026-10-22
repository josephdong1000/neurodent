"""Keep a record of why the quality filter excluded an animal, so no exclusion is silent.

The war_quality_filter checkpoint applies exclusions written in the config: an animal listed in
bad_folder_animalday, or one whose genotype is Unknown while exclude_unknown_genotypes is on. An
excluded animal is left with an empty output directory, and every later step finds no WAR there
and leaves it out. Until this module existed, the only trace was a warning in that animal's own
log file, so a run could finish with fewer animals than the config lists and still look complete.

The checkpoint now writes the reason into the animal's output directory, and the Snakefile reads
it back through quality_filter_outcome, warns once per excluded animal in the main log, and lists
every exclusion when the run ends.
"""

import json
from pathlib import Path

EXCLUSION_RECORD = "EXCLUDED.json"


def write_exclusion_record(directory, animal, reason):
    """Write why an animal was excluded into its quality-filter output directory.

    Args:
        directory (str | Path): The animal's war_quality_filter output directory.
        animal (str): The animal id, as the config names it.
        reason (str): Why the filter excluded it, in words a person can act on.

    Returns:
        Path: The record written.
    """
    path = Path(directory) / EXCLUSION_RECORD
    path.write_text(json.dumps({"animal": animal, "reason": reason}, indent=1) + "\n")
    return path


def clear_exclusion_record(directory):
    """Remove an exclusion record left by an earlier run, for an animal that now passes."""
    (Path(directory) / EXCLUSION_RECORD).unlink(missing_ok=True)


def quality_filter_outcome(directory):
    """Return whether the quality filter kept an animal, and if not, why.

    Args:
        directory (str | Path): The animal's war_quality_filter output directory.

    Returns:
        tuple[bool, str | None]: ``(True, None)`` when the filter copied the WAR through.
            Otherwise ``(False, reason)``, where reason is the recorded one, or says plainly
            that none was recorded. That happens for an animal excluded by a run made before
            exclusion records existed, whose directory is empty.
    """
    directory = Path(directory)
    if (directory / "war.parquet").exists():
        return True, None
    record = directory / EXCLUSION_RECORD
    if record.exists():
        try:
            reason = json.loads(record.read_text()).get("reason")
        except (OSError, ValueError) as error:
            return False, f"its exclusion record could not be read ({error})"
        return False, reason or "its exclusion record gives no reason"
    return False, (
        "no reason was recorded, because an earlier run excluded it before exclusion records "
        "existed; rerun war_quality_filter for this animal to record one"
    )


def format_exclusion_summary(excluded):
    """Render the end-of-run list of excluded animals.

    Args:
        excluded (dict[str, str]): Animal to reason.

    Returns:
        str: One line per animal under a count, or an empty string when nothing was excluded.
    """
    if not excluded:
        return ""
    lines = [
        f"{len(excluded)} animal(s) excluded by war_quality_filter, and so absent from every "
        f"later step:"
    ]
    lines += [f"  {animal}: {reason}" for animal, reason in sorted(excluded.items())]
    return "\n".join(lines)
