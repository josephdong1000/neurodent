"""An animal the quality filter excludes must be reported, never silently missing.

war_quality_filter applies exclusions written in the config, and an excluded animal is left with
an empty output directory. Every later step gathers per-animal outputs through helpers in the
Snakefile, and before this change each helper skipped an empty directory without a word, so a run
could end with fewer animals than the config lists and look complete. arx_parv relies on the
filter to drop its Unknown-genotype animals, so this was live there.

The checkpoint now records why in the directory, and one helper in the Snakefile reads it back,
reports it, and is the only way any gatherer decides whether an animal is in.
"""

import re
from pathlib import Path

import pytest

from neurodent.workflow.utils import (
    clear_exclusion_record,
    format_exclusion_summary,
    quality_filter_outcome,
    write_exclusion_record,
)

WORKFLOW = Path(__file__).parent.parent / "workflow"


class TestExclusionRecords:
    def test_kept_animal(self, tmp_path):
        (tmp_path / "war.parquet").write_bytes(b"")

        assert quality_filter_outcome(tmp_path) == (True, None)

    def test_recorded_exclusion_reports_its_reason(self, tmp_path):
        write_exclusion_record(tmp_path, "IQSEC2_3", "genotype is Unknown")

        assert quality_filter_outcome(tmp_path) == (False, "genotype is Unknown")

    def test_empty_directory_without_a_record_says_so(self, tmp_path):
        """An animal excluded by a run made before records existed must not read as a reason."""
        kept, reason = quality_filter_outcome(tmp_path)

        assert kept is False
        assert "no reason was recorded" in reason
        assert "rerun war_quality_filter" in reason

    def test_unreadable_record_is_reported_not_raised(self, tmp_path):
        (tmp_path / "EXCLUDED.json").write_text("{not json")

        kept, reason = quality_filter_outcome(tmp_path)

        assert kept is False
        assert "could not be read" in reason

    def test_parquet_wins_over_a_stale_record(self, tmp_path):
        """If both exist the animal is kept: the WAR is there and later steps will read it."""
        write_exclusion_record(tmp_path, "A1", "old reason")
        (tmp_path / "war.parquet").write_bytes(b"")

        assert quality_filter_outcome(tmp_path) == (True, None)

    def test_clear_removes_a_stale_record_and_tolerates_none(self, tmp_path):
        write_exclusion_record(tmp_path, "A1", "old reason")
        clear_exclusion_record(tmp_path)
        clear_exclusion_record(tmp_path)

        assert not (tmp_path / "EXCLUDED.json").exists()

    def test_summary(self):
        assert format_exclusion_summary({}) == ""
        summary = format_exclusion_summary({"b": "reason two", "a": "reason one"})
        lines = summary.splitlines()
        assert lines[0].startswith("2 animal(s) excluded by war_quality_filter")
        assert lines[1:] == ["  a: reason one", "  b: reason two"]


class TestTheSnakefileCannotSkipSilently:
    """Structural guards, because no test runs the Snakefile's helpers end to end."""

    @pytest.fixture(scope="class")
    def snakefile(self):
        return (WORKFLOW / "Snakefile").read_text()

    def test_only_one_helper_asks_the_checkpoint_whether_an_animal_was_kept(self, snakefile):
        live = [
            line for line in snakefile.splitlines()
            if "checkpoints.war_quality_filter.get" in line and not line.lstrip().startswith("#")
        ]

        assert len(live) == 1, live
        body = snakefile.split("def _passed_quality_filter", 1)[1].split("\ndef ", 1)[0]
        assert "checkpoints.war_quality_filter.get" in body

    def test_exclusions_are_listed_at_the_end_of_every_run(self, snakefile):
        for handler in ("onsuccess", "onerror"):
            block = re.search(rf"^{handler}:\n((?:    .*\n?)+)", snakefile, re.M)
            assert block, f"no {handler} handler"
            assert "_report_quality_filter_exclusions" in block.group(1)

    def test_the_checkpoint_writes_and_clears_its_record(self):
        rule = (WORKFLOW / "rules" / "war_quality_filter.smk").read_text()

        assert rule.count("write_exclusion_record(") == 2, "one per exclusion branch"
        assert "clear_exclusion_record(" in rule
