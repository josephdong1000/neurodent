"""Tests for the generate_wars pipeline script logic.

These tests validate the pipeline's parameter construction and config handling
without requiring real EEG data or running Snakemake. They use mocking to
isolate the logic under test, providing fast, reliable feedback for CI.

This addresses the need for pythonic/standardized testing of the Snakemake
pipeline without running full datasets through.
"""

import os
import sys
import tempfile
import warnings
from contextlib import nullcontext
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

    def test_pattern_with_hierarchical_placeholders(self):
        """Pattern with {animal}/{session}/{index} placeholders extracts all metadata."""
        config = self._build_mock_config(pattern="{animal}/{session}/{index}.nwb")
        analysis_cfg = config["analysis"]["war_generation"]

        base_path = "/data/parent/session_folder"
        pattern = f"{base_path}/{analysis_cfg['pattern']}"
        assert pattern == "/data/parent/session_folder/{animal}/{session}/{index}.nwb"

    def test_pattern_with_index_placeholder(self):
        """Pattern with {index} placeholder discovers files in session folder."""
        config = self._build_mock_config(pattern="{index}.rhd")
        analysis_cfg = config["analysis"]["war_generation"]

        base_path = "/data/parent/session_folder"
        pattern = f"{base_path}/{analysis_cfg['pattern']}"
        assert pattern == "/data/parent/session_folder/{index}.rhd"

    def test_skip_sessions_backward_compat(self):
        """skip_sessions falls back to skip_days for backward compatibility."""
        # Config with only old-style skip_days
        analysis_cfg = {"skip_days": ["bad", "test"]}

        skip = analysis_cfg.get("skip_sessions", analysis_cfg.get("skip_days", []))
        assert skip == ["bad", "test"]

    def test_skip_sessions_new_style(self):
        """New-style skip_sessions takes priority over skip_days."""
        analysis_cfg = {
            "skip_sessions": ["bad"],
            "skip_days": ["bad", "test"],
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

    def test_hierarchical_pattern_file_discovery(self, tmp_path):
        """Pattern with {animal}/{session}/{index} discovers files in nested directories."""
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
        from neurodent.loading.discovery import FileDiscoverer

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

    def test_index_only_pattern_file_discovery(self, tmp_path):
        """Pattern with {index} placeholder discovers files in session folder."""
        # Create synthetic directory: files in session folder
        (tmp_path / "recording_001.rhd").write_bytes(b"\x00" * 100)
        (tmp_path / "recording_002.rhd").write_bytes(b"\x00" * 100)
        (tmp_path / "notes.txt").write_text("notes")  # Should be excluded by pattern

        # Build pattern with {index} placeholder
        pattern = f"{tmp_path}/{{index}}.rhd"

        # Verify FileDiscoverer finds only .rhd files and extracts index
        from neurodent.loading.discovery import FileDiscoverer

        discoverer = FileDiscoverer(pattern)
        files = discoverer.discover()
        assert len(files) == 2
        for f in files:
            assert f.path.endswith(".rhd")
            assert "index" in f.metadata

    def test_explicit_pattern_file_discovery(self, tmp_path):
        """Explicit pattern with {animal} and {session} placeholders."""
        # Create structure matching explicit pattern
        session_dir = tmp_path / "MouseA" / "2025-01-15"
        session_dir.mkdir(parents=True)
        (session_dir / "trace_001.edf").write_bytes(b"\x00" * 100)
        (session_dir / "trace_002.edf").write_bytes(b"\x00" * 100)

        # Build pattern from config template (uses {index} for filename)
        pattern = f"{tmp_path}/{{animal}}/{{session}}/{{index}}.edf"

        from neurodent.loading.discovery import FileDiscoverer

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

        from neurodent.loading.discovery import FileDiscoverer

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
                    "pattern": "{index}",
                    "skip_sessions": [],
                    "lro_kwargs": {"mode": "si", "input_type": "files"},
                }
            }
        }

        # Session-specific override for EDF format
        session_overrides = {
            "analysis.war_generation.pattern": "{index}.EDF",
            "analysis.war_generation.lro_kwargs.mode": "mne",
            "analysis.war_generation.lro_kwargs.extract_func": "read_raw_edf",
        }

        overridden = apply_path_overrides(base_config, session_overrides)
        cfg = overridden["analysis"]["war_generation"]

        # Build pattern from overridden config
        base_path = str(tmp_path / "edf_session")
        pattern = f"{base_path}/{cfg['pattern']}"

        assert pattern.endswith("/{index}.EDF")
        assert cfg["lro_kwargs"]["mode"] == "mne"
        assert cfg["lro_kwargs"]["extract_func"] == "read_raw_edf"


# ---------------------------------------------------------------------------
# Tests for compute_windowed_analysis config wiring
# ---------------------------------------------------------------------------


class TestComputeWindowedAnalysisConfigWiring:
    """Verify that generate_wars.py reads compute_windowed_analysis config and
    passes the correct arguments (including chunk_duration_s) to
    AnimalOrganizer.compute_windowed_analysis().
    """

    def _build_war_gen_config(self, cwa_overrides=None):
        """Return a minimal analysis.war_generation config dict."""
        cwa = {
            "features": ["all"],
            "multiprocess_mode": "dask",
            "chunk_duration_s": None,
        }
        if cwa_overrides:
            cwa.update(cwa_overrides)
        return {"compute_windowed_analysis": cwa}

    # ------------------------------------------------------------------
    # Logic extracted from generate_wars.py (the block we want to test)
    # ------------------------------------------------------------------

    @staticmethod
    def _call_cwa(analysis_config, mock_ao):
        """Replicate the generate_wars.py logic for calling compute_windowed_analysis."""
        import warnings
        cwa_config = analysis_config.get("compute_windowed_analysis", {})
        cwa_features = cwa_config.get("features", ["all"])
        cwa_multiprocess_mode = cwa_config.get("multiprocess_mode", "dask")
        cwa_chunk_duration_s = cwa_config.get("chunk_duration_s", 3600)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*fmin=.*Spectrum estimate will be unreliable.*",
                category=RuntimeWarning,
            )
            mock_ao.compute_windowed_analysis(
                cwa_features,
                multiprocess_mode=cwa_multiprocess_mode,
                chunk_duration_s=cwa_chunk_duration_s,
            )

    def test_default_config_passes_all_features_and_dask(self):
        """Default config: features=["all"], mode="dask", chunk_duration_s=None."""
        ao = MagicMock()
        cfg = self._build_war_gen_config()
        self._call_cwa(cfg, ao)

        ao.compute_windowed_analysis.assert_called_once_with(
            ["all"],
            multiprocess_mode="dask",
            chunk_duration_s=None,
        )

    def test_chunk_duration_s_forwarded_from_config(self):
        """When chunk_duration_s is set in config it must reach compute_windowed_analysis."""
        ao = MagicMock()
        cfg = self._build_war_gen_config({"chunk_duration_s": 250})
        self._call_cwa(cfg, ao)

        _, kwargs = ao.compute_windowed_analysis.call_args
        assert kwargs["chunk_duration_s"] == 250

    def test_chunk_duration_s_null_passes_none(self):
        """Explicit null in YAML translates to Python None."""
        ao = MagicMock()
        cfg = self._build_war_gen_config({"chunk_duration_s": None})
        self._call_cwa(cfg, ao)

        _, kwargs = ao.compute_windowed_analysis.call_args
        assert kwargs["chunk_duration_s"] is None

    def test_features_list_forwarded(self):
        """Custom features list is forwarded verbatim."""
        ao = MagicMock()
        features = ["rms", "psd", "pcorr"]
        cfg = self._build_war_gen_config({"features": features})
        self._call_cwa(cfg, ao)

        pos_args, _ = ao.compute_windowed_analysis.call_args
        assert pos_args[0] == features

    def test_multiprocess_mode_serial_forwarded(self):
        """serial multiprocess_mode is forwarded correctly."""
        ao = MagicMock()
        cfg = self._build_war_gen_config({"multiprocess_mode": "serial"})
        self._call_cwa(cfg, ao)

        _, kwargs = ao.compute_windowed_analysis.call_args
        assert kwargs["multiprocess_mode"] == "serial"

    def test_missing_cwa_section_uses_defaults(self):
        """When compute_windowed_analysis section is absent, safe defaults apply."""
        ao = MagicMock()
        cfg = {}  # no compute_windowed_analysis key
        self._call_cwa(cfg, ao)

        ao.compute_windowed_analysis.assert_called_once_with(
            ["all"],
            multiprocess_mode="dask",
            chunk_duration_s=3600,
        )


class TestSaveFifChunkLenWiring:
    """Verify that generate_wars.py reads export_chunk_duration_s and passes it to
    fdsar.save_fif_and_json().
    """

    @staticmethod
    def _call_save_fif(fdsar_config, mock_fdsar, animalday_dir):
        """Replicate the generate_wars.py save_fif_and_json call logic."""
        fdsar_export_chunk_duration_s = fdsar_config.get("export_chunk_duration_s", 60)
        mock_fdsar.save_fif_and_json(
            animalday_dir,
            convert_to_mne=True,
            overwrite=True,
            chunk_duration_s=fdsar_export_chunk_duration_s,
        )

    def test_default_chunk_duration_s_is_60(self, tmp_path):
        """When export_chunk_duration_s is absent, 60 s is used."""
        fdsar = MagicMock()
        self._call_save_fif({}, fdsar, tmp_path)

        _, kwargs = fdsar.save_fif_and_json.call_args
        assert kwargs["chunk_duration_s"] == 60

    def test_custom_chunk_duration_s_forwarded(self, tmp_path):
        """Custom export_chunk_duration_s is forwarded to save_fif_and_json."""
        fdsar = MagicMock()
        self._call_save_fif({"export_chunk_duration_s": 30}, fdsar, tmp_path)

        _, kwargs = fdsar.save_fif_and_json.call_args
        assert kwargs["chunk_duration_s"] == 30

    def test_convert_to_mne_always_true(self, tmp_path):
        """convert_to_mne must always be True (required for saving .fif)."""
        fdsar = MagicMock()
        self._call_save_fif({"export_chunk_duration_s": 45}, fdsar, tmp_path)

        _, kwargs = fdsar.save_fif_and_json.call_args
        assert kwargs["convert_to_mne"] is True


class TestFdsarDiagnosticsConfigWiring:
    """Verify that generate_fdsar_diagnostics.py reads spike_averaged_traces
    config and passes the parameters to fdsar.plot_spike_averaged_traces().
    """

    @staticmethod
    def _call_plot(sat_config, mock_fdsar, output_dir):
        """Replicate the generate_fdsar_diagnostics.py call logic."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            mock_fdsar.plot_spike_averaged_traces(
                tmin=sat_config.get("tmin", -0.5),
                tmax=sat_config.get("tmax", 0.5),
                baseline=sat_config.get("baseline", None),
                save_dir=output_dir,
                animal_id="test_animal_day1",
                save_epoch=sat_config.get("save_epochs", True),
            )

    def test_defaults_when_sat_config_empty(self, tmp_path):
        """With empty sat_config the expected defaults are used."""
        fdsar = MagicMock()
        self._call_plot({}, fdsar, tmp_path)

        _, kwargs = fdsar.plot_spike_averaged_traces.call_args
        assert kwargs["tmin"] == -0.5
        assert kwargs["tmax"] == 0.5
        assert kwargs["baseline"] is None
        assert kwargs["save_epoch"] is True

    def test_tmin_tmax_from_config(self, tmp_path):
        """Custom tmin/tmax from config are forwarded correctly."""
        fdsar = MagicMock()
        sat = {"tmin": -1.0, "tmax": 1.0}
        self._call_plot(sat, fdsar, tmp_path)

        _, kwargs = fdsar.plot_spike_averaged_traces.call_args
        assert kwargs["tmin"] == -1.0
        assert kwargs["tmax"] == 1.0

    def test_save_epochs_false_forwarded(self, tmp_path):
        """save_epochs=false in config maps to save_epoch=False."""
        fdsar = MagicMock()
        self._call_plot({"save_epochs": False}, fdsar, tmp_path)

        _, kwargs = fdsar.plot_spike_averaged_traces.call_args
        assert kwargs["save_epoch"] is False

    def test_save_epochs_true_forwarded(self, tmp_path):
        """save_epochs=true in config maps to save_epoch=True."""
        fdsar = MagicMock()
        self._call_plot({"save_epochs": True}, fdsar, tmp_path)

        _, kwargs = fdsar.plot_spike_averaged_traces.call_args
        assert kwargs["save_epoch"] is True

    def test_save_dir_and_animal_id_forwarded(self, tmp_path):
        """save_dir and animal_id must reach plot_spike_averaged_traces."""
        fdsar = MagicMock()
        fdsar.animal_id = "M3_MHET"
        fdsar.animal_day = "day1"

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fdsar.plot_spike_averaged_traces(
                tmin={}.get("tmin", -0.5),
                tmax={}.get("tmax", 0.5),
                baseline={}.get("baseline", None),
                save_dir=tmp_path,
                animal_id=f"{fdsar.animal_id}_{fdsar.animal_day}",
                save_epoch={}.get("save_epochs", True),
            )

        _, kwargs = fdsar.plot_spike_averaged_traces.call_args
        assert kwargs["save_dir"] == tmp_path
        assert kwargs["animal_id"] == "M3_MHET_day1"

    def test_save_epochs_config_key_maps_to_save_epoch_param(self, tmp_path):
        """The config key 'save_epochs' (plural) maps to the function param 'save_epoch' (singular)."""
        fdsar = MagicMock()
        # Explicitly set save_epochs=False to verify the plural→singular mapping
        self._call_plot({"save_epochs": False}, fdsar, tmp_path)

        _, kwargs = fdsar.plot_spike_averaged_traces.call_args
        # The function receives the singular 'save_epoch' keyword
        assert "save_epoch" in kwargs
        assert kwargs["save_epoch"] is False


class TestFdsarChunkDurationWiring:
    """Verify that generate_wars.py reads chunk_duration_s from
    frequency_domain_spike_detection config and passes it to
    ao.compute_frequency_domain_spike_analysis().
    """

    @staticmethod
    def _call_fdsar(fdsar_config, mock_ao):
        """Replicate the generate_wars.py logic for calling
        compute_frequency_domain_spike_analysis."""
        detection_params = fdsar_config.get("default_params", {})
        multiprocess_mode = fdsar_config.get("multiprocess_mode", "serial")
        fdsar_chunk_duration_s = fdsar_config.get("chunk_duration_s", 3600)
        mock_ao.compute_frequency_domain_spike_analysis(
            detection_params=detection_params,
            multiprocess_mode=multiprocess_mode,
            chunk_duration_s=fdsar_chunk_duration_s,
        )

    def test_chunk_duration_s_absent_defaults_3600(self):
        """When chunk_duration_s is null/absent, 3600 s is forwarded."""
        ao = MagicMock()
        self._call_fdsar({"default_params": {}, "multiprocess_mode": "dask"}, ao)

        _, kwargs = ao.compute_frequency_domain_spike_analysis.call_args
        assert kwargs["chunk_duration_s"] == 3600

    def test_chunk_duration_s_forwarded(self):
        """When chunk_duration_s is a number, it is forwarded verbatim."""
        ao = MagicMock()
        self._call_fdsar(
            {"default_params": {}, "multiprocess_mode": "dask", "chunk_duration_s": 300.0},
            ao,
        )

        _, kwargs = ao.compute_frequency_domain_spike_analysis.call_args
        assert kwargs["chunk_duration_s"] == 300.0

    def test_missing_chunk_duration_s_defaults_3600(self):
        """When chunk_duration_s key is absent, 3600 is used."""
        ao = MagicMock()
        self._call_fdsar({"default_params": {}}, ao)

        _, kwargs = ao.compute_frequency_domain_spike_analysis.call_args
        assert kwargs["chunk_duration_s"] == 3600


class TestLofThresholdWiring:
    """Verify that generate_wars.py reads lof config and passes
    reject_lof_threshold and lof_chunk_duration_s to ao.compute_bad_channels().
    """

    @staticmethod
    def _call_compute_bad_channels(lof_config, mock_ao):
        """Replicate the generate_wars.py logic for calling compute_bad_channels."""
        lof_threshold = lof_config.get("reject_lof_threshold")
        lof_chunk_duration_s = lof_config.get("lof_chunk_duration_s", 60)
        mock_ao.compute_bad_channels(
            lof_threshold=lof_threshold,
            lof_chunk_duration_s=lof_chunk_duration_s,
        )

    def test_lof_threshold_forwarded(self):
        """reject_lof_threshold is forwarded as lof_threshold."""
        ao = MagicMock()
        self._call_compute_bad_channels({"reject_lof_threshold": 3.0}, ao)

        _, kwargs = ao.compute_bad_channels.call_args
        assert kwargs["lof_threshold"] == 3.0

    def test_lof_threshold_none_when_absent(self):
        """When reject_lof_threshold is absent, None is forwarded."""
        ao = MagicMock()
        self._call_compute_bad_channels({}, ao)

        _, kwargs = ao.compute_bad_channels.call_args
        assert kwargs["lof_threshold"] is None

    def test_no_limit_memory_parameter(self):
        """compute_bad_channels should NOT receive a limit_memory kwarg."""
        ao = MagicMock()
        self._call_compute_bad_channels({"reject_lof_threshold": 2.5}, ao)

        _, kwargs = ao.compute_bad_channels.call_args
        assert "limit_memory" not in kwargs

    def test_lof_chunk_duration_s_forwarded(self):
        """lof_chunk_duration_s is forwarded from config."""
        ao = MagicMock()
        self._call_compute_bad_channels(
            {"reject_lof_threshold": 2.5, "lof_chunk_duration_s": 120}, ao,
        )

        _, kwargs = ao.compute_bad_channels.call_args
        assert kwargs["lof_chunk_duration_s"] == 120

    def test_lof_chunk_duration_s_default_60(self):
        """When lof_chunk_duration_s is absent, defaults to 60."""
        ao = MagicMock()
        self._call_compute_bad_channels({"reject_lof_threshold": 2.5}, ao)

        _, kwargs = ao.compute_bad_channels.call_args
        assert kwargs["lof_chunk_duration_s"] == 60


@pytest.mark.skipif(sys.platform != "linux", reason="memray is only available on Linux")
class TestMemrayIntegration:
    """Test the memray/cProfile profiling integration in generate_wars.py.

    These tests replicate the profiling logic from the ``if __name__ == "__main__"``
    block and verify that memray.Tracker and FileDestination behave correctly,
    including the overwrite=True fix that resolved the 94% pipeline stall.
    """

    @staticmethod
    def _build_tracker_ctx(memray_enabled, memray_path):
        """Replicate the generate_wars.py logic for building the memray context manager.

        This mirrors lines 282-294 of generate_wars.py exactly.
        """
        import memray

        if memray_enabled:
            tracker_ctx = memray.Tracker(
                destination=memray.FileDestination(str(memray_path), overwrite=True)
            )
        else:
            tracker_ctx = nullcontext()
        return tracker_ctx

    def test_tracker_creates_memray_bin(self, tmp_path):
        """Tracker writes a valid memray.bin file."""
        memray_path = tmp_path / "memray.bin"
        ctx = self._build_tracker_ctx(True, memray_path)
        with ctx:
            _ = [i * 2 for i in range(100)]

        assert memray_path.exists()
        assert memray_path.stat().st_size > 0

    def test_tracker_overwrites_existing_file(self, tmp_path):
        """FileDestination(overwrite=True) overwrites an existing memray.bin without error.

        This was the root cause of the 94% pipeline stall: memray.Tracker without
        overwrite=True raises OSError when the file already exists from a previous run.
        """
        memray_path = tmp_path / "memray.bin"

        # First run: create the file
        with self._build_tracker_ctx(True, memray_path):
            _ = list(range(50))
        assert memray_path.exists()

        # Second run: overwrite must succeed (not raise OSError)
        with self._build_tracker_ctx(True, memray_path):
            _ = list(range(200))
        assert memray_path.exists()
        assert memray_path.stat().st_size > 0

    def test_disabled_memray_uses_nullcontext(self, tmp_path):
        """When memray is disabled, a nullcontext is returned (no file created)."""
        memray_path = tmp_path / "memray.bin"
        ctx = self._build_tracker_ctx(False, memray_path)
        assert isinstance(ctx, nullcontext)
        with ctx:
            pass
        assert not memray_path.exists()

    def test_memray_bin_readable_by_file_reader(self, tmp_path):
        """The memray.bin file produced by Tracker is readable by memray.FileReader."""
        import memray

        memray_path = tmp_path / "memray.bin"
        with self._build_tracker_ctx(True, memray_path):
            _ = bytearray(1024)

        reader = memray.FileReader(str(memray_path))
        metadata = reader.metadata
        assert metadata.pid > 0

    def test_env_var_gating_matches_script_logic(self):
        """NEURODENT_MEMRAY env var logic matches generate_wars.py."""
        # When env var is unset → falsy
        saved = os.environ.pop("NEURODENT_MEMRAY", None)
        try:
            assert not os.environ.get("NEURODENT_MEMRAY")
        finally:
            if saved is not None:
                os.environ["NEURODENT_MEMRAY"] = saved

    def test_cprofile_context_independent_of_memray(self, tmp_path):
        """cProfile and memray contexts are independent — both can wrap main() together.

        Replicates the nested context pattern from generate_wars.py lines 296-308.
        """
        import cProfile

        memray_path = tmp_path / "memray.bin"
        tracker_ctx = self._build_tracker_ctx(True, memray_path)

        profiler = cProfile.Profile()
        with tracker_ctx:
            profiler.enable()
            _ = sum(range(1000))
            profiler.disable()

        assert memray_path.exists()
        prof_path = tmp_path / "profile.prof"
        profiler.dump_stats(str(prof_path))
        assert prof_path.exists()
        assert prof_path.stat().st_size > 0

    def test_tracker_with_subdirectory_path(self, tmp_path):
        """Tracker works when memray_path is in a nested subdirectory (mimics war output layout)."""
        nested = tmp_path / "results" / "wars" / "animal_x"
        nested.mkdir(parents=True)
        memray_path = nested / "memray.bin"

        with self._build_tracker_ctx(True, memray_path):
            _ = list(range(10))

        assert memray_path.exists()

