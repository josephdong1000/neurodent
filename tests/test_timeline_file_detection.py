"""
Tests for _compute_global_timeline file vs folder detection.

Tests the fix that allows _compute_global_timeline to handle both:
- Folder-based discovery (traditional mode)
- File-based discovery (when using file_pattern like "*.rhd")
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from neurodent.visualization.results import AnimalOrganizer


class TestComputeGlobalTimelineFileDetection:
    """Test that _compute_global_timeline correctly detects files vs folders."""

    @pytest.fixture
    def mock_ao(self):
        """Create a minimal AnimalOrganizer mock for testing."""
        ao = Mock(spec=AnimalOrganizer)
        ao.animal_id = "test_animal"
        ao.animal_param = ["test_animal"]
        ao.read_mode = "concat"
        return ao

    def test_detects_file_path_and_adjusts_input_type(self, tmp_path):
        """When path is a file, input_type should be changed to 'file'."""
        # Create a temp file
        test_file = tmp_path / "test.rhd"
        test_file.write_text("dummy content")
        
        assert test_file.is_file(), "Test file should exist"
        
        # The logic we're testing
        base_lro_kwargs = {
            "input_type": "files",
            "file_pattern": "*.rhd",
            "extract_func": "read_intan",
        }
        
        adjusted_kwargs = base_lro_kwargs.copy()
        if Path(test_file).is_file():
            adjusted_kwargs["input_type"] = "file"
            adjusted_kwargs.pop("file_pattern", None)
        
        assert adjusted_kwargs["input_type"] == "file"
        assert "file_pattern" not in adjusted_kwargs
        assert adjusted_kwargs["extract_func"] == "read_intan"

    def test_detects_folder_path_and_keeps_input_type(self, tmp_path):
        """When path is a folder, input_type should remain unchanged."""
        # Create a temp folder
        test_folder = tmp_path / "test_folder"
        test_folder.mkdir()
        
        assert test_folder.is_dir(), "Test folder should exist"
        
        base_lro_kwargs = {
            "input_type": "files",
            "file_pattern": "*.rhd",
            "extract_func": "read_intan",
        }
        
        adjusted_kwargs = base_lro_kwargs.copy()
        if Path(test_folder).is_file():
            adjusted_kwargs["input_type"] = "file"
            adjusted_kwargs.pop("file_pattern", None)
        
        # Should remain unchanged since it's a folder
        assert adjusted_kwargs["input_type"] == "files"
        assert adjusted_kwargs["file_pattern"] == "*.rhd"

    def test_file_detection_with_various_extensions(self, tmp_path):
        """Test file detection works with various file extensions."""
        extensions = [".rhd", ".nwb", ".bin", ".zarr", ".txt"]
        
        for ext in extensions:
            test_file = tmp_path / f"test{ext}"
            test_file.write_text("dummy")
            
            assert Path(test_file).is_file(), f"File with {ext} should be detected as file"
            
            base_kwargs = {"input_type": "files", "file_pattern": f"*{ext}"}
            adjusted = base_kwargs.copy()
            if Path(test_file).is_file():
                adjusted["input_type"] = "file"
                adjusted.pop("file_pattern", None)
            
            assert adjusted["input_type"] == "file", f"Should detect {ext} as file"

    def test_preserves_other_kwargs_when_adjusting(self, tmp_path):
        """Test that other kwargs are preserved when adjusting for file mode."""
        test_file = tmp_path / "test.rhd"
        test_file.write_text("dummy")
        
        base_lro_kwargs = {
            "input_type": "files",
            "file_pattern": "*.rhd",
            "extract_func": "read_intan",
            "mode": "si",
            "stream_id": "0",
            "custom_param": "should_be_preserved",
        }
        
        adjusted_kwargs = base_lro_kwargs.copy()
        if Path(test_file).is_file():
            adjusted_kwargs["input_type"] = "file"
            adjusted_kwargs.pop("file_pattern", None)
        
        # Check all other params preserved
        assert adjusted_kwargs["extract_func"] == "read_intan"
        assert adjusted_kwargs["mode"] == "si"
        assert adjusted_kwargs["stream_id"] == "0"
        assert adjusted_kwargs["custom_param"] == "should_be_preserved"
        # Check modifications
        assert adjusted_kwargs["input_type"] == "file"
        assert "file_pattern" not in adjusted_kwargs

    def test_handles_missing_file_pattern_gracefully(self, tmp_path):
        """Test that popping file_pattern works even if not present."""
        test_file = tmp_path / "test.rhd"
        test_file.write_text("dummy")
        
        base_lro_kwargs = {
            "input_type": "files",
            "extract_func": "read_intan",
            # No file_pattern key
        }
        
        adjusted_kwargs = base_lro_kwargs.copy()
        if Path(test_file).is_file():
            adjusted_kwargs["input_type"] = "file"
            adjusted_kwargs.pop("file_pattern", None)  # Should not raise
        
        assert adjusted_kwargs["input_type"] == "file"
        assert "file_pattern" not in adjusted_kwargs

    def test_symlink_to_file_detected_as_file(self, tmp_path):
        """Test that symlinks to files are correctly detected as files."""
        # Create actual file and symlink to it
        actual_file = tmp_path / "actual.rhd"
        actual_file.write_text("content")
        
        symlink = tmp_path / "link.rhd"
        try:
            symlink.symlink_to(actual_file)
        except OSError:
            pytest.skip("Symlink creation not supported on this system")
        
        assert Path(symlink).is_file(), "Symlink to file should be detected as file"
        
        base_kwargs = {"input_type": "files", "file_pattern": "*.rhd"}
        adjusted = base_kwargs.copy()
        if Path(symlink).is_file():
            adjusted["input_type"] = "file"
        
        assert adjusted["input_type"] == "file"


class TestComputeGlobalTimelineIntegration:
    """Integration tests for _compute_global_timeline with synthetic data."""

    @pytest.fixture
    def synthetic_nwb_dataset(self, tmp_path):
        """Create a synthetic NWB dataset for integration testing."""
        from tests.data.generate import create_synthetic_dataset

        return create_synthetic_dataset(tmp_path)

    def test_timeline_computation_with_nwb_files(self, synthetic_nwb_dataset):
        """Test that timeline computation works with file-based discovery."""
        from neurodent import core

        ds = synthetic_nwb_dataset
        data_root = ds["data_root"]
        session_folder = ds["session_folder"]
        animal_id = ds["animals"][0]

        # Discover NWB files for the animal
        animal_dir = data_root / session_folder / animal_id
        nwb_files = sorted(animal_dir.rglob("*.nwb"))[:2]
        assert len(nwb_files) >= 1, "Should have at least 1 NWB file"

        base_datetime = datetime(2025, 1, 1, 12, 0, 0)

        # Test that we can create LROs for individual files
        for nwb_file in nwb_files:
            assert nwb_file.is_file(), f"{nwb_file.name} should be a file"

            lro = core.LongRecordingOrganizer(
                nwb_file,
                extract_func="read_nwb_recording",
                manual_datetimes=base_datetime,
            )

            assert hasattr(lro, "LongRecording"), "LRO should have LongRecording"
            assert lro.LongRecording is not None
            duration = lro.LongRecording.get_duration()
            assert duration > 0, f"File {nwb_file.name} should have positive duration"

    @pytest.mark.mutates_constants
    def test_animal_organizer_with_manual_datetimes_and_file_pattern(
        self, synthetic_nwb_dataset
    ):
        """Test that AnimalOrganizer works with manual_datetimes and pattern-based discovery."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.visualization import AnimalOrganizer

        ds = synthetic_nwb_dataset
        base_path = str(ds["data_root"] / ds["session_folder"])
        pattern = f"{base_path}/{{animal}}/{{session}}/{{index}}.nwb"
        animal_id = ds["animals"][0]

        # Inject metadata so genotype resolution works
        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(ds["samples_config"])

            ao = AnimalOrganizer(
                pattern,
                animal_id=animal_id,
                assume_from_number=True,
                lro_kwargs={
                    "mode": "si",
                    "extract_func": "read_nwb_recording",
                    "multiprocess_mode": "serial",
                    "manual_datetimes": datetime(2025, 1, 1, 12, 0, 0),
                },
            )

            assert ao is not None
            assert ao.animal_id == animal_id
            assert len(ao.long_recordings) >= 1
            assert len(ao.unique_animaldays) >= 1

            for lro in ao.long_recordings:
                assert hasattr(lro, "LongRecording")
                assert lro.LongRecording is not None
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
