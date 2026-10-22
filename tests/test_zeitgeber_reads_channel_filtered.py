"""Zeitgeber must leave out the bad channels the config names, like every other figure.

war_zeitgeber read the fragment-filtered WARs, one step before channel filtering, so electrodes
the config marks as bad were averaged into every zeitgeber figure while EP and relfreq, which read
the manually channel-filtered WARs, left them out. Channel averaging uses nanmean, so reading the
filtered branch drops a rejected channel from the average rather than blanking the window.
"""

from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / "workflow"


def test_no_analysis_rule_reads_wars_from_before_channel_filtering():
    """Zeitgeber read the fragment-filtered WARs, so configured bad channels reached its figures.

    Every analysis branch should read what channel filtering produced. The fragment-filtered
    WARs are an intermediate: written by fragment filtering, read only by channel filtering.
    """
    readers = sorted(
        p.name for p in (WORKFLOW / "rules").glob("*.smk")
        if "wars_fragment_filtered" in p.read_text()
        and p.name not in {"war_fragment_filtering.smk", "war_channel_filtering.smk"}
    )
    if "wars_fragment_filtered" in (WORKFLOW / "Snakefile").read_text():
        readers.append("Snakefile")

    assert readers == [], f"read the fragment-filtered WARs directly: {readers}"
