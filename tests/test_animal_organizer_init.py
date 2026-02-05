
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
        with patch.object(AnimalOrganizer, '_create_long_recordings'):
             ao = AnimalOrganizer(
                base_folder_path=str(study_folder), 
                anim_id="test_animal",
                mode="concat" 
            )
        
        self._assert_containers_initialized(ao)

    def test_from_lros_initializes_containers(self, mock_lro):
        """Test that from_lros factory initializes all output containers."""
        ao = AnimalOrganizer.from_lros([mock_lro], animal_id="test_animal")
        self._assert_containers_initialized(ao)

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
