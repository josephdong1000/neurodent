import pytest
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from neurodent.visualization.results import AnimalOrganizer
from neurodent.core import LongRecordingOrganizer


class TestAnimalOrganizerInitialization:
    @pytest.fixture
    def mock_lro(self):
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["ch1", "ch2"]
        lro.base_folder_path = "/tmp/mock/folder"
        lro.labels = {}
        return lro

    def test_init_initializes_containers(self, tmp_path):
        """Test that standard __init__ initializes all output containers."""
        # Setup minimal valid directory structure
        study_folder = tmp_path / "study_data"
        study_folder.mkdir()
        (study_folder / "test_animal_WT_20240101").mkdir()

        # Mock _create_long_recordings to avoid loading actual data
        with patch.object(AnimalOrganizer, "_create_long_recordings"):
            ao = AnimalOrganizer(
                pattern=f"{str(study_folder)}/{{animal}}_WT_{{session}}",
                animal_id="test_animal",
            )

        self._assert_containers_initialized(ao)

    def test_from_lros_initializes_containers(self, mock_lro):
        """Test that from_lros factory initializes all output containers."""
        ao = AnimalOrganizer.from_lros([mock_lro], animal_id="test_animal")
        self._assert_containers_initialized(ao)

    def test_from_lros_uses_metadata_date(self, mock_lro):
        """Test that from_lros uses LRO metadata for date, ignoring folder path."""
        # Set a misleading path to prove we aren't parsing it
        mock_lro.base_folder_path = "/path/to/Misleading_Feb-02-2023"
        # Set explicit metadata return value
        mock_lro.get_date_string.return_value = "Jan-01-2022"

        ao = AnimalOrganizer.from_lros([mock_lro], animal_id="Animal", genotype="WT")

        # Verify get_date_string was called
        mock_lro.get_date_string.assert_called_once()

        # Verify result used the metadata date, NOT the path date
        expected_animalday = "Animal WT Jan-01-2022"
        assert ao.animaldays[0] == expected_animalday

    def test_from_lros_sets_sex_and_genotype(self, mock_lro):
        """from_lros must honor both genotype AND sex. Regression guard: the
        war_generation script previously passed genotype but not sex, baking
        sex='Unknown' into every WAR (parquet column + JSON sidecar)."""
        ao = AnimalOrganizer.from_lros(
            [mock_lro], animal_id="Animal", genotype="KO", sex="Female"
        )
        assert ao.genotype == "KO"
        assert ao.sex == "Female"

    def test_from_lros_sex_defaults_unknown(self, mock_lro):
        """sex defaults to 'Unknown' when not provided (the default that caused
        the bug when the caller omitted sex)."""
        ao = AnimalOrganizer.from_lros([mock_lro], animal_id="Animal")
        assert ao.sex == "Unknown"

    def test_from_lros_fails_without_metadata(self, mock_lro):
        """Test that from_lros fails hard if metadata date is missing."""
        mock_lro.get_date_string.side_effect = ValueError("No timestamps")

        with pytest.raises(ValueError, match="Could not determine date"):
            AnimalOrganizer.from_lros([mock_lro], animal_id="Animal")

    def test_containers_are_usable(self, mock_lro):
        """Test that initialized containers are mutable and usable."""
        ao = AnimalOrganizer.from_lros([mock_lro], animal_id="test_animal")

        # Verify lists, dicts, and dataframes are mutable/usable
        ao.long_analyzers.append("mock_analyzer")
        ao.bad_channels_dict["session1"] = ["bad_ch"]
        ao.features_df = pd.DataFrame({"a": [1, 2]})

        assert ao.long_analyzers[0] == "mock_analyzer"
        assert ao.bad_channels_dict["session1"] == ["bad_ch"]
        assert not ao.features_df.empty

    def _assert_containers_initialized(self, ao):
        """Helper to assert all expected attributes exist and are in initial state."""
        assert isinstance(ao.long_analyzers, list)
        assert len(ao.long_analyzers) == 0

        assert isinstance(ao.bad_channels_dict, dict)
        assert len(ao.bad_channels_dict) == 0

        assert isinstance(ao.features_df, pd.DataFrame)
        assert ao.features_df.empty

        assert isinstance(ao.features_avg_df, pd.DataFrame)
        assert ao.features_avg_df.empty

        assert ao.spike_analysis_results is None
        assert ao.frequency_domain_spike_analysis_results is None
        assert ao.window_analysis_result is None
