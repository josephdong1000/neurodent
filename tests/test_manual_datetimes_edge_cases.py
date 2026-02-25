
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from neurodent.visualization import results

class TestManualDatetimesEdgeCases:
    """Test extreme edge cases for manual_datetimes configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        self.animal_id = "A123"

        # Create test folders
        self.folder1 = self.base_path / f"WT_{self.animal_id}_2023-01-15"
        self.folder2 = self.base_path / f"WT_{self.animal_id}_2023-01-16"
        
        for folder in [self.folder1, self.folder2]:
            folder.mkdir(parents=True)
            (folder / "dummy_ColMajor_001.bin").touch()
            (folder / "dummy_Meta_001.json").touch()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_mock_lro(self, folder_name="test"):
        mock_lro = Mock()
        mock_lro.meta = Mock(f_s=1000, n_channels=3)
        mock_lro.base_folder_path = folder_name
        mock_lro.file_durations = [100.0]
        mock_lro.LongRecording = Mock()
        mock_lro.LongRecording.get_duration.return_value = 100.0
        mock_lro.file_end_datetimes = [None]
        mock_lro.channel_names = ["Ch1", "Ch2", "Ch3"]
        return mock_lro

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_mixed_bag_configuration(self, mock_glob, mock_lro_class):
        """
        Test Case 3: The 'Mixed Bag'.
        One animal uses ID-based key, another uses flat folder keys.
        """
        mock_glob.return_value = [str(self.folder1), str(self.folder2)]
        mock_lro_class.return_value = self._create_mock_lro()

        # Config: 
        # - "Start_Animal" uses explicit ID key (ignored by our current A123 run)
        # - "WT_A123_2023-01-15" uses flat folder key (used by fallback)
        mixed_config = {
            "Start_Animal": { "SomeFolder": datetime(2023, 1, 1) }, # Different animal
            f"WT_{self.animal_id}_2023-01-15": datetime(2023, 2, 1, 10, 0), # Flat key for A123
             f"WT_{self.animal_id}_2023-01-16": datetime(2023, 2, 1, 11, 0)
        }

        # Run for A123
        ao = results.AnimalOrganizer(
            base_folder_path=str(self.base_path),
            animal_id=self.animal_id,  # "A123"
            mode="concat",
            lro_kwargs={"manual_datetimes": mixed_config}
        )

        # Should successfully fallback to using the flat folder keys
        assert ao._processed_timestamps[f"WT_{self.animal_id}_2023-01-15"] == datetime(2023, 2, 1, 10, 0)

    @patch("neurodent.visualization.results.core.LongRecordingOrganizer")
    @patch("glob.glob")
    def test_shadowing_trap_error(self, mock_glob, mock_lro_class):
        """
        Test Case 4: The 'Shadowing Trap'.
        Dictionary has both valid Animal ID key and ignored flat folder keys.
        """
        mock_glob.return_value = [str(self.folder1), str(self.folder2)]
        mock_lro_class.return_value = self._create_mock_lro()

        shadowing_config = {
            # 1. The Priority: Found ID key, used exclusively.
            self.animal_id: {
                f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 1, 10, 0)
            },
            
            # 2. The Shadowed Key: Flat folder key.
            # This should be IGNORED because key #1 exists.
            # Thus, folder2 will be considered "missing" from the spec.
            f"WT_{self.animal_id}_2023-01-16": datetime(2023, 1, 1, 11, 0)
        }

        # Should raise ValueError because folder2 is missing from the explicit ID spec
        # and the flat key providing it is ignored.
        with pytest.raises(ValueError) as exc_info:
            results.AnimalOrganizer(
                base_folder_path=str(self.base_path),
                animal_id=self.animal_id,
                mode="concat",
                lro_kwargs={"manual_datetimes": shadowing_config}
            )
        
        error_msg = str(exc_info.value)
        assert "Ambiguous manual_datetimes configuration" in error_msg
        assert "Please nest all folder keys" in error_msg
