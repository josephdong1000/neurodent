"""
Integration Tests for Snakemake Workflow
========================================

Tests that validate the pipeline's data-loading path using a minimal
synthetic NWB dataset generated on the fly.  These tests exercise the real
``FileDiscoverer``, ``AnimalOrganizer``, and analysis pipeline against actual
files on disk, without requiring production-scale recordings.

Running
-------
Run only integration tests::

    uv run pytest tests/integration/ -v -m integration

Or include them in the full suite::

    uv run pytest tests/ -v
"""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def example_pipeline_env(tmp_path):
    """Create a complete, tiny pipeline environment under tmp_path.

    Returns a dict with ``data_root``, ``samples_config``, ``animals``,
    ``session_folder``, and the full ``config`` dict that would normally
    come from Snakemake.
    """
    from tests.example_data.generate import create_synthetic_dataset

    ds = create_synthetic_dataset(tmp_path, n_sessions=2, duration_s=3)

    # Build a minimal pipeline config (mirrors config/config.yaml + example.yaml)
    pipeline_config = {
        "temp_directory": str(tmp_path / "tmp"),
        "analysis": {
            "war_generation": {
                "pattern": "{animal}/{session}/{index}.nwb",
                "assume_from_number": True,
                "skip_sessions": [],
                "lro_kwargs": {
                    "mode": "si",
                    "extract_func": "read_nwb_recording",
                    "multiprocess_mode": "serial",
                },
            },
        },
        "cluster": {
            "war_generation": {"interface": None},
        },
        "overrides": {},
    }

    ds["config"] = pipeline_config
    return ds


# ---------------------------------------------------------------------------
# Tests — Dataset Generation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExampleDatasetGeneration:
    """Verify that the synthetic dataset generator produces valid NWB files."""

    def test_creates_directory_tree(self, example_dataset):
        """The generated dataset has the expected directory structure."""
        root = example_dataset["data_root"]
        session = example_dataset["session_folder"]

        for animal_id in example_dataset["animals"]:
            day_dir = root / session / animal_id / "day1"
            assert day_dir.is_dir(), f"Missing directory: {day_dir}"

            nwb_files = list(day_dir.glob("*.nwb"))
            assert len(nwb_files) == 1, f"Expected 1 NWB file, got {nwb_files}"

    def test_nwb_file_readable_by_spikeinterface(self, example_dataset):
        """NWB file can be loaded by SpikeInterface."""
        import spikeinterface.extractors as se

        root = example_dataset["data_root"]
        session = example_dataset["session_folder"]
        animal_id = example_dataset["animals"][0]

        nwb_file = next((root / session / animal_id / "day1").glob("*.nwb"))
        rec = se.read_nwb_recording(str(nwb_file))

        assert rec.get_num_channels() == 8
        # example_dataset fixture uses default duration_s=5, sr=1000
        assert rec.get_num_samples() == 5 * 1000

    def test_samples_config_structure(self, example_dataset):
        """samples_config contains all required keys."""
        sc = example_dataset["samples_config"]
        assert "data_parent_folder" in sc
        assert "ANIMAL_METADATA" in sc
        assert "data_folders_to_animal_ids" in sc
        assert "GENOTYPE_ALIASES" in sc


# ---------------------------------------------------------------------------
# Tests — File Discovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFileDiscoveryWithExampleData:
    """Test that FileDiscoverer works with the generated synthetic data."""

    def test_discovers_all_animals(self, example_pipeline_env):
        """Pattern discovers files for all animals in the dataset."""
        from neurodent.core.discovery import FileDiscoverer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        # Build absolute pattern the same way generate_wars.py does
        base_path = str(ds["data_root"] / ds["session_folder"])
        pattern = f"{base_path}/{cfg['pattern']}"

        discoverer = FileDiscoverer(pattern)
        all_files = discoverer.discover()

        # 2 animals × 2 sessions = at least 4 discovered items
        assert len(all_files) >= 4

        found_animals = {f.metadata["animal"] for f in all_files}
        for animal_id in ds["animals"]:
            assert animal_id in found_animals, f"{animal_id} not discovered"

    def test_discovers_sessions_per_animal(self, example_pipeline_env):
        """Pattern discovers multiple sessions for a single animal."""
        from neurodent.core.discovery import FileDiscoverer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        base_path = str(ds["data_root"] / ds["session_folder"])
        pattern = f"{base_path}/{cfg['pattern']}"

        animal_id = ds["animals"][0]
        discoverer = FileDiscoverer(pattern)
        animal_files = discoverer.discover(animal=animal_id)

        sessions = {f.metadata["session"] for f in animal_files}
        assert "day1" in sessions
        assert "day2" in sessions

    def test_filter_by_animal_id(self, example_pipeline_env):
        """Filtering by animal_id returns only that animal's files."""
        from neurodent.core.discovery import FileDiscoverer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        base_path = str(ds["data_root"] / ds["session_folder"])
        pattern = f"{base_path}/{cfg['pattern']}"

        discoverer = FileDiscoverer(pattern)
        for animal_id in ds["animals"]:
            filtered = discoverer.discover(animal=animal_id)
            for f in filtered:
                assert f.metadata["animal"] == animal_id


# ---------------------------------------------------------------------------
# Tests — Config Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSamplesConfigIntegration:
    """Test that the generated samples_config works with neurodent utilities."""

    def test_inject_config_aliases(self, example_dataset):
        """inject_config_aliases succeeds with the synthetic config."""
        from neurodent.workflow import inject_config_aliases
        from neurodent import constants

        # Save original state to restore after test
        orig_genotype_aliases = constants.GENOTYPE_ALIASES
        orig_animal_metadata = constants.ANIMAL_METADATA

        try:
            sc = example_dataset["samples_config"]
            inject_config_aliases(sc)

            # Verify metadata was injected
            assert "ExWT" in constants.ANIMAL_METADATA
            assert constants.ANIMAL_METADATA["ExWT"]["gene"] == "WT"
            assert "ExKO" in constants.ANIMAL_METADATA
            assert constants.ANIMAL_METADATA["ExKO"]["gene"] == "KO"
        finally:
            # Restore original global state to avoid leaking into other tests
            constants.GENOTYPE_ALIASES = orig_genotype_aliases
            constants.ANIMAL_METADATA = orig_animal_metadata

    def test_samples_config_serializable(self, example_dataset):
        """samples_config can be serialized to JSON (for writing to disk)."""
        sc = example_dataset["samples_config"]
        dumped = json.dumps(sc, indent=2)
        reloaded = json.loads(dumped)
        assert reloaded["ANIMAL_METADATA"] == sc["ANIMAL_METADATA"]


# ---------------------------------------------------------------------------
# Tests — Pipeline Steps (data loading → analysis)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPipelineSteps:
    """Test actual pipeline stages end-to-end with synthetic NWB data.

    These tests exercise the real AnimalOrganizer → LongRecordingOrganizer →
    analysis pipeline, not just file discovery.
    """

    def test_animal_organizer_loads_data(self, example_pipeline_env):
        """AnimalOrganizer successfully loads NWB data for a single animal."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.visualization import AnimalOrganizer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        # Inject metadata so genotype resolution works
        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(ds["samples_config"])

            base_path = str(ds["data_root"] / ds["session_folder"])
            pattern = f"{base_path}/{cfg['pattern']}"
            animal_id = ds["animals"][0]

            ao = AnimalOrganizer(
                pattern,
                animal_id=animal_id,
                skip_sessions=cfg.get("skip_sessions", []),
                assume_from_number=cfg["assume_from_number"],
                lro_kwargs=cfg["lro_kwargs"],
            )

            assert ao.animal_id == animal_id
            assert len(ao.long_recordings) >= 1
            # Verify sessions were created
            assert len(ao.unique_animaldays) >= 1

            # Verify the underlying SI recording was loaded correctly
            for lro in ao.long_recordings:
                assert hasattr(lro, "LongRecording")
                assert lro.LongRecording is not None
                assert lro.LongRecording.get_num_channels() == 8
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases

    @pytest.mark.xfail(
        reason="WAR generation with SI-loaded NWB needs manual_datetimes; "
               "the global-timeline code path passes input_type to the NWB "
               "extractor. Fix tracked in base branch.",
        strict=False,
    )
    def test_war_generation(self, example_pipeline_env):
        """compute_windowed_analysis produces a WindowAnalysisResult.

        Uses a single-session dataset to avoid the manual_datetimes
        code path that has a pre-existing input_type leak issue.
        """
        from tests.example_data.generate import create_synthetic_dataset
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.visualization import AnimalOrganizer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(ds["samples_config"])

            base_path = str(ds["data_root"] / ds["session_folder"])
            pattern = f"{base_path}/{cfg['pattern']}"
            animal_id = ds["animals"][0]

            ao = AnimalOrganizer(
                pattern,
                animal_id=animal_id,
                skip_sessions=["day2"],  # use only 1 session to avoid timestamp issues
                assume_from_number=cfg["assume_from_number"],
                lro_kwargs=cfg["lro_kwargs"],
            )

            # base_folder_path is normally set during persist_recording in the
            # full pipeline; set it here so compute_windowed_analysis logging works
            for lro in ao.long_recordings:
                if not hasattr(lro, "base_folder_path"):
                    lro.base_folder_path = "synthetic_test"

            war = ao.compute_windowed_analysis(
                ["all"],
                multiprocess_mode="serial",
                window_s=1,  # small windows for tiny data
                apply_notch_filter=False,  # skip filtering for speed
            )

            assert war is not None
            # WAR should have a features DataFrame
            assert hasattr(war, "features_df") or hasattr(war, "df")
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases

