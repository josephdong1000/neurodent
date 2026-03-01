"""
Integration Tests for Snakemake Workflow
========================================

Tests that validate the pipeline's data-loading path using a minimal
synthetic dataset generated on the fly.  These tests exercise the real
``FileDiscoverer`` code against actual files on disk, without requiring
production-scale recordings.

Running
-------
Run only integration tests::

    uv run pytest tests/integration/ -v -m integration

Or include them in the full suite::

    uv run pytest tests/ -v
"""

import json
import subprocess
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
                "pattern": "{animal}/{session}/{index}",
                "assume_from_number": True,
                "skip_sessions": [],
                "lro_kwargs": {
                    "mode": "bin",
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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExampleDatasetGeneration:
    """Verify that the synthetic dataset generator produces valid files."""

    def test_creates_directory_tree(self, example_dataset):
        """The generated dataset has the expected nest-mode structure."""
        root = example_dataset["data_root"]
        session = example_dataset["session_folder"]

        for animal_id in example_dataset["animals"]:
            day_dir = root / session / animal_id / "day1"
            assert day_dir.is_dir(), f"Missing directory: {day_dir}"

            bin_files = list(day_dir.glob("*_ColMajor.bin"))
            meta_files = list(day_dir.glob("*_Meta.csv"))
            assert len(bin_files) == 1, f"Expected 1 bin file, got {bin_files}"
            assert len(meta_files) == 1, f"Expected 1 meta file, got {meta_files}"

    def test_bin_file_has_correct_size(self, example_dataset):
        """Binary file size matches n_samples × n_channels × sizeof(float32)."""
        import numpy as np

        root = example_dataset["data_root"]
        session = example_dataset["session_folder"]
        animal_id = example_dataset["animals"][0]

        bin_file = next((root / session / animal_id / "day1").glob("*_ColMajor.bin"))
        file_size = bin_file.stat().st_size

        # example_dataset fixture uses default duration_s=5
        n_samples = 5 * 1000
        n_channels = 8
        expected = n_samples * n_channels * np.dtype(np.float32).itemsize
        assert file_size == expected

    def test_meta_csv_has_correct_channels(self, example_dataset):
        """Meta CSV lists the right number of channels."""
        root = example_dataset["data_root"]
        session = example_dataset["session_folder"]
        animal_id = example_dataset["animals"][0]

        meta_file = next((root / session / animal_id / "day1").glob("*_Meta.csv"))
        lines = meta_file.read_text().strip().splitlines()
        # 1 header + 8 channel rows
        assert len(lines) == 9, f"Expected 9 lines, got {len(lines)}"

    def test_samples_config_structure(self, example_dataset):
        """samples_config contains all required keys."""
        sc = example_dataset["samples_config"]
        assert "data_parent_folder" in sc
        assert "ANIMAL_METADATA" in sc
        assert "data_folders_to_animal_ids" in sc
        assert "GENOTYPE_ALIASES" in sc


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

        # 2 animals × 2 sessions, each session has multiple files (bin + meta)
        # discovered independently as DiscoveredFile objects
        assert len(all_files) >= 4  # at least 2 animals × 2 sessions

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

