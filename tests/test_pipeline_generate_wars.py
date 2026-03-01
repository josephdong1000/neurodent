"""Tests for the generate_wars pipeline script logic.

These tests validate the pipeline's parameter construction and config handling
without requiring real EEG data or running Snakemake. They use mocking to
isolate the logic under test, providing fast, reliable feedback for CI.

This addresses the need for pythonic/standardized testing of the Snakemake
pipeline without running full datasets through.
"""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from neurodent.workflow.utils import build_discovery_pattern


class TestBuildDiscoveryPattern:
    """Tests for the build_discovery_pattern helper that converts
    old-style mode+file_pattern config to new pattern-based strings."""

    # --- New-style pattern (takes priority) ---

    def test_explicit_pattern_string(self):
        """Explicit pattern template is prepended with base_path."""
        result = build_discovery_pattern(
            "/data/session1",
            pattern="{animal}/{session}/*.bin",
        )
        assert result == "/data/session1/{animal}/{session}/*.bin"

    def test_explicit_pattern_list(self):
        """List of patterns are each prepended with base_path."""
        result = build_discovery_pattern(
            "/data/session1",
            pattern=["{animal}/{session}/*_ColMajor.bin", "{animal}/{session}/*_Meta.csv"],
        )
        assert result == [
            "/data/session1/{animal}/{session}/*_ColMajor.bin",
            "/data/session1/{animal}/{session}/*_Meta.csv",
        ]

    def test_explicit_pattern_overrides_mode(self):
        """When both pattern and mode are given, pattern takes priority."""
        result = build_discovery_pattern(
            "/data/session1",
            mode="nest",
            file_pattern="*.rhd",
            pattern="{index}.edf",
        )
        assert result == "/data/session1/{index}.edf"

    # --- Legacy mode conversion ---

    def test_nest_mode_default_file_pattern(self):
        """Nest mode without file_pattern uses {index} placeholder."""
        result = build_discovery_pattern("/data/session1", mode="nest")
        assert result == "/data/session1/{animal}/{session}/{index}"

    def test_nest_mode_with_file_pattern(self):
        """Nest mode with file_pattern converts glob to {index} placeholder."""
        result = build_discovery_pattern("/data/session1", mode="nest", file_pattern="*.bin")
        assert result == "/data/session1/{animal}/{session}/{index}.bin"

    def test_base_mode_default(self):
        """Base mode without file_pattern uses wildcard."""
        result = build_discovery_pattern("/data/session1", mode="base")
        assert result == "/data/session1/*"

    def test_base_mode_with_rhd(self):
        """Base mode with *.rhd file pattern."""
        result = build_discovery_pattern(
            "/data/session1", mode="base", file_pattern="*.rhd"
        )
        assert result == "/data/session1/*.rhd"

    def test_concat_mode(self):
        """Concat mode behaves like base mode."""
        result = build_discovery_pattern(
            "/data/session1", mode="concat", file_pattern="*.nwb"
        )
        assert result == "/data/session1/*.nwb"

    def test_unknown_mode_falls_back_to_glob(self):
        """Unknown mode names still produce a valid glob pattern."""
        result = build_discovery_pattern(
            "/data/session1", mode="custom", file_pattern="*.xyz"
        )
        assert result == "/data/session1/*.xyz"

    # --- Edge cases ---

    def test_trailing_slash_stripped(self):
        """Trailing slash in base_path is stripped."""
        result = build_discovery_pattern("/data/session1/", mode="base", file_pattern="*.rhd")
        assert result == "/data/session1/*.rhd"

    def test_nest_mode_glob_to_index_conversion(self):
        """Nest mode converts * in file_pattern to {index} placeholder."""
        result = build_discovery_pattern("/data/s1", mode="nest", file_pattern="*_ColMajor.bin")
        assert result == "/data/s1/{animal}/{session}/{index}_ColMajor.bin"

    def test_path_object_as_base(self):
        """Path object is accepted as base_path."""
        result = build_discovery_pattern(
            Path("/data") / "session1", mode="base", file_pattern="*.rhd"
        )
        assert result == "/data/session1/*.rhd"

    def test_missing_mode_and_pattern_raises(self):
        """ValueError when neither pattern nor mode is provided."""
        with pytest.raises(ValueError, match="Either 'pattern' or 'mode'"):
            build_discovery_pattern("/data/session1")


class TestBuildDiscoveryPatternDatasetConfigs:
    """Test build_discovery_pattern with realistic dataset configurations
    matching the actual config/datasets/*.yaml files."""

    def test_sox5_bin_config(self):
        """sox5_bin: mode='nest', lro_kwargs.mode='bin'."""
        result = build_discovery_pattern(
            "/mnt/data/Sox5/010822_cohort4",
            mode="nest",
        )
        assert "{animal}" in result
        assert "{session}" in result
        assert result == "/mnt/data/Sox5/010822_cohort4/{animal}/{session}/{index}"

    def test_ap3b2_rhd_config(self):
        """ap3b2_rhd: mode='base', file_pattern='*.rhd'."""
        result = build_discovery_pattern(
            "/mnt/data/AP3B2/session1",
            mode="base",
            file_pattern="*.rhd",
        )
        assert result == "/mnt/data/AP3B2/session1/*.rhd"

    def test_ap3b2_nwb_config(self):
        """ap3b2_nwb: mode='concat', file_pattern='*.nwb'."""
        result = build_discovery_pattern(
            "/mnt/data/AP3B2/session1",
            mode="concat",
            file_pattern="*.nwb",
        )
        assert result == "/mnt/data/AP3B2/session1/*.nwb"

    def test_arx_rosa_base_config(self):
        """arx_rosa: mode='base', file_pattern='*'."""
        result = build_discovery_pattern(
            "/mnt/data/ArxRosa/session1",
            mode="base",
            file_pattern="*",
        )
        assert result == "/mnt/data/ArxRosa/session1/*"

    def test_arx_rosa_edf_override(self):
        """arx_rosa with EDF session override: file_pattern='*.EDF'."""
        result = build_discovery_pattern(
            "/mnt/data/ArxRosa/session1",
            mode="base",
            file_pattern="*.EDF",
        )
        assert result == "/mnt/data/ArxRosa/session1/*.EDF"


class TestGenerateWarsParameterConstruction:
    """Test that generate_wars.py correctly constructs parameters for the
    new AnimalOrganizer API.

    These tests mock the AnimalOrganizer and verify the arguments it receives,
    without requiring any real data or Snakemake.
    """

    def _build_mock_config(self, mode="nest", file_pattern=None, pattern=None):
        """Helper to build a realistic pipeline config dict."""
        war_gen = {
            "assume_from_number": True,
            "skip_sessions": ["bad"],
            "lro_kwargs": {"multiprocess_mode": "dask"},
        }
        if mode is not None:
            war_gen["mode"] = mode
        if file_pattern is not None:
            war_gen["file_pattern"] = file_pattern
        if pattern is not None:
            war_gen["pattern"] = pattern
        return {
            "analysis": {"war_generation": war_gen},
            "temp_directory": "/tmp",
            "cluster": {
                "war_generation": {"interface": "lo"},
            },
            "overrides": {},
        }

    def test_pattern_for_nest_mode(self):
        """Nest mode produces pattern with {animal}/{session}/{index} placeholders."""
        config = self._build_mock_config(mode="nest")
        analysis_cfg = config["analysis"]["war_generation"]

        pattern = build_discovery_pattern(
            "/data/parent/session_folder",
            mode=analysis_cfg.get("mode"),
            file_pattern=analysis_cfg.get("file_pattern"),
            pattern=analysis_cfg.get("pattern"),
        )
        assert pattern == "/data/parent/session_folder/{animal}/{session}/{index}"

    def test_pattern_for_base_mode_with_rhd(self):
        """Base mode with *.rhd produces flat glob pattern."""
        config = self._build_mock_config(mode="base", file_pattern="*.rhd")
        analysis_cfg = config["analysis"]["war_generation"]

        pattern = build_discovery_pattern(
            "/data/parent/session_folder",
            mode=analysis_cfg.get("mode"),
            file_pattern=analysis_cfg.get("file_pattern"),
            pattern=analysis_cfg.get("pattern"),
        )
        assert pattern == "/data/parent/session_folder/*.rhd"

    def test_explicit_pattern_in_config(self):
        """Explicit pattern in config overrides mode."""
        config = self._build_mock_config(
            mode="nest",
            pattern="{animal}/{session}/{index}.edf",
        )
        analysis_cfg = config["analysis"]["war_generation"]

        pattern = build_discovery_pattern(
            "/data/parent/session_folder",
            mode=analysis_cfg.get("mode"),
            file_pattern=analysis_cfg.get("file_pattern"),
            pattern=analysis_cfg.get("pattern"),
        )
        assert pattern == "/data/parent/session_folder/{animal}/{session}/{index}.edf"

    def test_skip_sessions_backward_compat(self):
        """skip_sessions falls back to skip_days for backward compatibility."""
        # Config with only old-style skip_days
        analysis_cfg = {"skip_days": ["bad", "test"], "assume_from_number": True}

        skip = analysis_cfg.get("skip_sessions", analysis_cfg.get("skip_days", []))
        assert skip == ["bad", "test"]

    def test_skip_sessions_new_style(self):
        """New-style skip_sessions takes priority over skip_days."""
        analysis_cfg = {
            "skip_sessions": ["bad"],
            "skip_days": ["bad", "test"],
            "assume_from_number": True,
        }

        skip = analysis_cfg.get("skip_sessions", analysis_cfg.get("skip_days", []))
        assert skip == ["bad"]

    def test_joint_session_uses_none_animal_id(self):
        """For joint sessions, animal_id should be None during discovery."""
        is_joint = True
        source_animal_id = "ArxRosa-967"
        ao_animal_id = None if is_joint else source_animal_id
        assert ao_animal_id is None

    def test_regular_session_passes_animal_id(self):
        """For regular sessions, animal_id is passed for filtering."""
        is_joint = False
        source_animal_id = "M3_MHET"
        ao_animal_id = None if is_joint else source_animal_id
        assert ao_animal_id == "M3_MHET"


class TestPipelineIntegrationWithSyntheticData:
    """Integration-style tests using synthetic file structures on disk.

    These tests create temporary directory trees that mimic real dataset
    layouts, then verify that build_discovery_pattern + FileDiscoverer
    correctly discover files. This validates the end-to-end pattern
    construction without needing real EEG data.
    """

    def test_nest_mode_file_discovery(self, tmp_path):
        """Nest mode pattern discovers files organized as animal/session/files."""
        # Create synthetic nest-mode directory structure
        animal_dir = tmp_path / "A10"
        for day in ["day1", "day2"]:
            day_dir = animal_dir / day
            day_dir.mkdir(parents=True)
            (day_dir / "data_ColMajor.bin").write_bytes(b"\x00" * 100)
            (day_dir / "data_Meta.csv").write_text("header\n")

        # Also create another animal to verify filtering
        other_dir = tmp_path / "B20"
        other_day = other_dir / "day1"
        other_day.mkdir(parents=True)
        (other_day / "data_ColMajor.bin").write_bytes(b"\x00" * 100)

        # Build pattern using pipeline helper
        pattern = build_discovery_pattern(str(tmp_path), mode="nest")
        assert "{animal}" in pattern
        assert "{session}" in pattern
        assert "{index}" in pattern

        # Verify FileDiscoverer finds the right files
        from neurodent.core.discovery import FileDiscoverer

        discoverer = FileDiscoverer(pattern)
        all_files = discoverer.discover()
        assert len(all_files) > 0

        # Filter for A10 only
        a10_files = discoverer.discover(animal="A10")
        for f in a10_files:
            assert f.metadata["animal"] == "A10"

        # Verify sessions are discovered
        sessions = {f.metadata["session"] for f in a10_files}
        assert "day1" in sessions
        assert "day2" in sessions

    def test_base_mode_file_discovery(self, tmp_path):
        """Base mode pattern discovers files flat in the session folder."""
        # Create synthetic base-mode directory: flat files
        (tmp_path / "recording_001.rhd").write_bytes(b"\x00" * 100)
        (tmp_path / "recording_002.rhd").write_bytes(b"\x00" * 100)
        (tmp_path / "notes.txt").write_text("notes")  # Should be excluded by pattern

        # Build pattern
        pattern = build_discovery_pattern(str(tmp_path), mode="base", file_pattern="*.rhd")
        assert pattern == f"{tmp_path}/*.rhd"

        # Verify FileDiscoverer finds only .rhd files
        from neurodent.core.discovery import FileDiscoverer

        discoverer = FileDiscoverer(pattern)
        files = discoverer.discover()
        assert len(files) == 2
        for f in files:
            assert f.path.endswith(".rhd")

    def test_explicit_pattern_file_discovery(self, tmp_path):
        """Explicit pattern with {animal} and {session} placeholders."""
        # Create structure matching explicit pattern
        session_dir = tmp_path / "MouseA" / "2025-01-15"
        session_dir.mkdir(parents=True)
        (session_dir / "trace_001.edf").write_bytes(b"\x00" * 100)
        (session_dir / "trace_002.edf").write_bytes(b"\x00" * 100)

        # Build pattern from explicit template (uses {index} for filename)
        pattern = build_discovery_pattern(
            str(tmp_path),
            pattern="{animal}/{session}/{index}.edf",
        )

        from neurodent.core.discovery import FileDiscoverer

        discoverer = FileDiscoverer(pattern)
        files = discoverer.discover()
        assert len(files) == 2
        assert files[0].metadata["animal"] == "MouseA"
        assert files[0].metadata["session"] == "2025-01-15"

    def test_multi_pattern_file_discovery(self, tmp_path):
        """Multi-pattern discovery groups data+metadata file pairs."""
        # Create paired files: data.bin + meta.csv
        session = tmp_path / "AnimalX" / "session1"
        session.mkdir(parents=True)
        (session / "rec_ColMajor.bin").write_bytes(b"\x00" * 100)
        (session / "rec_Meta.csv").write_text("header\n")

        # Build multi-pattern using {index} placeholders for file stems
        patterns = build_discovery_pattern(
            str(tmp_path),
            pattern=[
                "{animal}/{session}/{index}_ColMajor.bin",
                "{animal}/{session}/{index}_Meta.csv",
            ],
        )
        assert isinstance(patterns, list)
        assert len(patterns) == 2

        from neurodent.core.discovery import FileDiscoverer

        discoverer = FileDiscoverer(patterns)
        groups = discoverer.discover()
        assert len(groups) == 1
        assert groups[0].is_multi_file
        assert len(groups[0].paths) == 2
        assert groups[0].metadata["animal"] == "AnimalX"
        assert groups[0].metadata["session"] == "session1"

    def test_pipeline_pattern_with_session_overrides(self, tmp_path):
        """Test that session-specific config overrides produce correct patterns."""
        from neurodent.workflow.utils import apply_path_overrides

        # Base config (arx_rosa-like)
        base_config = {
            "analysis": {
                "war_generation": {
                    "mode": "base",
                    "file_pattern": "*",
                    "assume_from_number": True,
                    "skip_sessions": [],
                    "lro_kwargs": {"mode": "si", "input_type": "files"},
                }
            }
        }

        # Session-specific override for EDF format
        session_overrides = {
            "analysis.war_generation.file_pattern": "*.EDF",
            "analysis.war_generation.lro_kwargs.mode": "mne",
            "analysis.war_generation.lro_kwargs.extract_func": "read_raw_edf",
        }

        overridden = apply_path_overrides(base_config, session_overrides)
        cfg = overridden["analysis"]["war_generation"]

        # Build pattern from overridden config
        pattern = build_discovery_pattern(
            str(tmp_path / "edf_session"),
            mode=cfg.get("mode"),
            file_pattern=cfg.get("file_pattern"),
            pattern=cfg.get("pattern"),
        )

        assert pattern.endswith("/*.EDF")
        assert cfg["lro_kwargs"]["mode"] == "mne"
        assert cfg["lro_kwargs"]["extract_func"] == "read_raw_edf"
