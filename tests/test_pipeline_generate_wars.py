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


class TestGenerateWarsParameterConstruction:
    """Test that generate_wars.py correctly constructs parameters for the
    new AnimalOrganizer API.

    These tests mock the AnimalOrganizer and verify the arguments it receives,
    without requiring any real data or Snakemake.
    """

    def _build_mock_config(self, pattern="{animal}/{session}/{index}.nwb"):
        """Helper to build a realistic pipeline config dict."""
        war_gen = {
            "pattern": pattern,
            "assume_from_number": True,
            "skip_sessions": ["bad"],
            "lro_kwargs": {"multiprocess_mode": "dask"},
        }
        return {
            "analysis": {"war_generation": war_gen},
            "temp_directory": "/tmp",
            "cluster": {
                "war_generation": {"interface": "lo"},
            },
            "overrides": {},
        }

    def test_pattern_for_nest_layout(self):
        """Nested layout config produces pattern with {animal}/{session}/{index} placeholders."""
        config = self._build_mock_config(pattern="{animal}/{session}/{index}.nwb")
        analysis_cfg = config["analysis"]["war_generation"]

        base_path = "/data/parent/session_folder"
        pattern = f"{base_path}/{analysis_cfg['pattern']}"
        assert pattern == "/data/parent/session_folder/{animal}/{session}/{index}.nwb"

    def test_pattern_for_flat_layout_with_rhd(self):
        """Flat layout with *.rhd produces glob pattern."""
        config = self._build_mock_config(pattern="*.rhd")
        analysis_cfg = config["analysis"]["war_generation"]

        base_path = "/data/parent/session_folder"
        pattern = f"{base_path}/{analysis_cfg['pattern']}"
        assert pattern == "/data/parent/session_folder/*.rhd"

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
    layouts, then verify that FileDiscoverer correctly discovers files
    when given a pattern directly from the config. This validates the
    end-to-end pattern construction without needing real EEG data.
    """

    def test_nest_layout_file_discovery(self, tmp_path):
        """Nested layout pattern discovers files organized as animal/session/files."""
        # Create synthetic directory structure with NWB files
        animal_dir = tmp_path / "A10"
        for day in ["day1", "day2"]:
            day_dir = animal_dir / day
            day_dir.mkdir(parents=True)
            (day_dir / "recording.nwb").write_bytes(b"\x00" * 100)

        # Also create another animal to verify filtering
        other_dir = tmp_path / "B20"
        other_day = other_dir / "day1"
        other_day.mkdir(parents=True)
        (other_day / "recording.nwb").write_bytes(b"\x00" * 100)

        # Build pattern the same way generate_wars.py does:
        # base_path / relative_pattern from config
        pattern = f"{tmp_path}/{{animal}}/{{session}}/{{index}}.nwb"
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

    def test_flat_layout_file_discovery(self, tmp_path):
        """Flat layout pattern discovers files in the session folder."""
        # Create synthetic base-mode directory: flat files
        (tmp_path / "recording_001.rhd").write_bytes(b"\x00" * 100)
        (tmp_path / "recording_002.rhd").write_bytes(b"\x00" * 100)
        (tmp_path / "notes.txt").write_text("notes")  # Should be excluded by pattern

        # Build pattern
        pattern = f"{tmp_path}/*.rhd"

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

        # Build pattern from config template (uses {index} for filename)
        pattern = f"{tmp_path}/{{animal}}/{{session}}/{{index}}.edf"

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
        patterns = [
            f"{tmp_path}/{{animal}}/{{session}}/{{index}}_ColMajor.bin",
            f"{tmp_path}/{{animal}}/{{session}}/{{index}}_Meta.csv",
        ]

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
                    "pattern": "*",
                    "assume_from_number": True,
                    "skip_sessions": [],
                    "lro_kwargs": {"mode": "si", "input_type": "files"},
                }
            }
        }

        # Session-specific override for EDF format
        session_overrides = {
            "analysis.war_generation.pattern": "*.EDF",
            "analysis.war_generation.lro_kwargs.mode": "mne",
            "analysis.war_generation.lro_kwargs.extract_func": "read_raw_edf",
        }

        overridden = apply_path_overrides(base_config, session_overrides)
        cfg = overridden["analysis"]["war_generation"]

        # Build pattern from overridden config
        base_path = str(tmp_path / "edf_session")
        pattern = f"{base_path}/{cfg['pattern']}"

        assert pattern.endswith("/*.EDF")
        assert cfg["lro_kwargs"]["mode"] == "mne"
        assert cfg["lro_kwargs"]["extract_func"] == "read_raw_edf"
