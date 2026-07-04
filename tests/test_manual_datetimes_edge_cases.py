import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from neurodent.loading import animal_organizer as results


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

    @patch("neurodent.loading.long_recording_organizer.LongRecordingOrganizer")
    @patch("neurodent.loading.discovery.glob.glob")
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
            "Start_Animal": {"SomeFolder": datetime(2023, 1, 1)},  # Different animal
            f"WT_{self.animal_id}_2023-01-15": datetime(
                2023, 2, 1, 10, 0
            ),  # Flat key for A123
            f"WT_{self.animal_id}_2023-01-16": datetime(2023, 2, 1, 11, 0),
        }

        # Run for A123
        ao = results.AnimalOrganizer(
            pattern=f"{self.base_path}/WT_{{animal}}_{{session}}",
            animal_id=self.animal_id,  # "A123"
            lro_kwargs={"manual_datetimes": mixed_config},
        )

        # Should successfully fallback to using the flat folder keys
        # Keys are now full paths (from _get_item_key)
        assert ao._processed_timestamps[str(self.folder1)] == datetime(
            2023, 2, 1, 10, 0
        )

    @patch("neurodent.loading.long_recording_organizer.LongRecordingOrganizer")
    @patch("neurodent.loading.discovery.glob.glob")
    def test_shadowing_trap_error(self, mock_glob, mock_lro_class):
        """
        Test Case 4: The 'Shadowing Trap'.
        Dictionary has an animal ID key (deprecated, no longer recognized) alongside
        a flat folder key. Since animal-ID-keyed dicts are no longer supported,
        the animal ID key is not recognized as an item or session name, and the
        missing item raises an error.
        """
        mock_glob.return_value = [str(self.folder1), str(self.folder2)]
        mock_lro_class.return_value = self._create_mock_lro()

        shadowing_config = {
            # Animal ID key — no longer recognized as a special key
            self.animal_id: {
                f"WT_{self.animal_id}_2023-01-15": datetime(2023, 1, 1, 10, 0)
            },
            # Flat folder key matching one item
            f"WT_{self.animal_id}_2023-01-16": datetime(2023, 1, 1, 11, 0),
        }

        # Should raise ValueError because folder1's item name is not in the dict
        # (the animal ID key is no longer unwrapped)
        with pytest.raises(ValueError, match="Missing entries in manual_datetimes for items") as exc_info:
            results.AnimalOrganizer(
                pattern=f"{self.base_path}/WT_{{animal}}_{{session}}",
                animal_id=self.animal_id,
                lro_kwargs={"manual_datetimes": shadowing_config},
            )

        # Verify the missing item is the one whose key was only under the animal ID
        assert f"WT_{self.animal_id}_2023-01-15" in str(exc_info.value)
