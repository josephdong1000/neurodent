"""
Test AnimalOrganizer.split() and from_lros() functionality.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import with try/except for optional spikeinterface dependency
try:
    import spikeinterface.core as si
    SI_AVAILABLE = True
except ImportError:
    SI_AVAILABLE = False

from neurodent.loading import LongRecordingOrganizer
from neurodent.loading import AnimalOrganizer
from neurodent.analysis import AnimalAnalyzer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_lro():
    """Create a mock LongRecordingOrganizer for testing."""
    lro = MagicMock(spec=LongRecordingOrganizer)
    lro.channel_names = ["Ch0", "Ch1", "Ch2", "Ch3"]
    lro.base_folder_path = Path("/mock/path/day1")
    lro.manual_datetimes = datetime(2023, 1, 1, 12, 0)
    lro.datetimes_are_start = True
    lro.file_end_datetimes = [datetime(2023, 1, 1, 14, 0)]
    lro._is_in_memory = False
    
    # Mock date string
    lro.get_date_string.return_value = "Jan-01-2023"
    
    # Mock the split method
    def mock_split(groups):
        result = {}
        for group_name, channels in groups.items():
            child_lro = MagicMock(spec=LongRecordingOrganizer)
            child_lro.channel_names = channels
            child_lro.base_folder_path = None
            child_lro._is_in_memory = True
            child_lro.manual_datetimes = lro.manual_datetimes
            child_lro.datetimes_are_start = lro.datetimes_are_start
            child_lro.file_end_datetimes = lro.file_end_datetimes
            child_lro.get_date_string.return_value = lro.get_date_string.return_value
            result[group_name] = child_lro
        return result
    
    lro.split = mock_split
    return lro


@pytest.fixture
def mock_multi_day_lros():
    """Create multiple mock LROs simulating multi-day recordings."""
    lros = []
    for i in range(3):
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0", "Ch1", "Ch2", "Ch3"]
        lro.base_folder_path = Path(f"/mock/path/day{i}")
        lro.manual_datetimes = datetime(2023, 1, 1 + i, 12, 0)
        lro.datetimes_are_start = True
        lro._is_in_memory = False
        lro.get_date_string.return_value = f"Jan-{i+1:02d}-2023"
        
        def make_mock_split(parent_lro):
            def mock_split(groups):
                result = {}
                for group_name, channels in groups.items():
                    child_lro = MagicMock(spec=LongRecordingOrganizer)
                    child_lro.channel_names = channels
                    child_lro.base_folder_path = None
                    child_lro._is_in_memory = True
                    child_lro.manual_datetimes = parent_lro.manual_datetimes
                    child_lro.save_recording = MagicMock(return_value=Path("/output"))
                    child_lro.get_date_string.return_value = parent_lro.get_date_string.return_value
                    result[group_name] = child_lro
                return result
            return mock_split
        
        lro.split = make_mock_split(lro)
        lros.append(lro)
    
    return lros


# =============================================================================
# Tests: from_lros()
# =============================================================================

class TestFromLros:
    """Test AnimalOrganizer.from_lros() factory method."""

    def test_from_lros_creates_valid_ao(self, mock_lro):
        """Test that from_lros creates a valid AnimalOrganizer."""
        ao = AnimalOrganizer.from_lros(
            lros=[mock_lro],
            animal_id="TestAnimal",
            genotype="WT",
        )
        
        assert ao.animal_id == "TestAnimal"
        assert ao.genotype == "WT"
        assert len(ao.long_recordings) == 1
        assert ao.long_recordings[0] is mock_lro

    def test_from_lros_inherits_channel_names(self, mock_lro):
        """Test that channel names are inherited from first LRO."""
        ao = AnimalOrganizer.from_lros(
            lros=[mock_lro],
            animal_id="TestAnimal",
        )
        
        assert ao.channel_names == ["Ch0", "Ch1", "Ch2", "Ch3"]

    def test_from_lros_generates_animaldays(self, mock_multi_day_lros):
        """Test that animaldays are generated from LRO metadata (get_date_string)."""
        ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="TestAnimal",
        )
        
        assert len(ao.animaldays) == 3
        # Expected format: "{animal_id} {genotype} {date}"
        # genotype defaults to "Unknown" if not specified
        assert "TestAnimal Unknown Jan-01-2023" in ao.animaldays[0]
        assert "TestAnimal Unknown Jan-02-2023" in ao.animaldays[1]
        assert "TestAnimal Unknown Jan-03-2023" in ao.animaldays[2]

    def test_from_lros_empty_list_raises_error(self):
        """Test that empty LRO list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create AnimalOrganizer from empty"):
            AnimalOrganizer.from_lros(lros=[], animal_id="TestAnimal")

    def test_from_lros_sets_default_genotype(self, mock_lro):
        """Test that default genotype is 'Unknown'."""
        ao = AnimalOrganizer.from_lros(
            lros=[mock_lro],
            animal_id="TestAnimal",
        )
        
        assert ao.genotype == "Unknown"

    def test_analyzer_initializes_dataframe(self, mock_lro):
        """Test that AnimalAnalyzer initializes its empty features DataFrame."""
        ao = AnimalOrganizer.from_lros(
            lros=[mock_lro],
            animal_id="TestAnimal",
        )
        az = AnimalAnalyzer(ao)

        assert hasattr(az, "features_df")
        assert az.features_df.empty

    def test_from_lros_works_for_in_memory_with_metadata(self):
        """Test animalday generation when LRO has no base_folder_path but has metadata."""
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0"]
        lro.base_folder_path = None  # In-memory LRO
        lro.get_date_string.return_value = "Jan-01-2023"
        
        ao = AnimalOrganizer.from_lros(
            lros=[lro],
            animal_id="TestAnimal",
        )
        
        assert ao.animaldays[0] == "TestAnimal Unknown Jan-01-2023"

    def test_from_lros_warns_on_channel_reorder(self, caplog):
        """Test that from_lros warns when channels are in different order."""
        lro1 = MagicMock(spec=LongRecordingOrganizer)
        lro1.channel_names = ["Ch0", "Ch1", "Ch2"]
        lro1.base_folder_path = Path("/mock/day1")
        lro1.get_date_string.return_value = "Jan-01-2023"

        lro2 = MagicMock(spec=LongRecordingOrganizer)
        lro2.channel_names = ["Ch2", "Ch1", "Ch0"]  # Same channels, different order
        lro2.base_folder_path = Path("/mock/day2")
        lro2.get_date_string.return_value = "Jan-02-2023"
        
        import logging
        with caplog.at_level(logging.WARNING):
            ao = AnimalOrganizer.from_lros(
                lros=[lro1, lro2],
                animal_id="TestAnimal",
            )
        
        # Should use first LRO's order
        assert ao.channel_names == ["Ch0", "Ch1", "Ch2"]
        # Should log warning
        assert "different order" in caplog.text

    def test_from_lros_raises_on_channel_mismatch(self):
        """Test that from_lros raises error when channels don't match."""
        lro1 = MagicMock(spec=LongRecordingOrganizer)
        lro1.channel_names = ["Ch0", "Ch1"]
        lro1.base_folder_path = Path("/mock/day1")
        lro1.get_date_string.return_value = "Jan-01-2023"

        lro2 = MagicMock(spec=LongRecordingOrganizer)
        lro2.channel_names = ["Ch0", "Ch2"]  # Different channel set
        lro2.base_folder_path = Path("/mock/day2")
        lro2.get_date_string.return_value = "Jan-02-2023"
        
        with pytest.raises(ValueError, match="inconsistent channel names"):
            AnimalOrganizer.from_lros(
                lros=[lro1, lro2],
                animal_id="TestAnimal",
            )

    def test_from_lros_derives_folder_metadata(self):
        """Test that from_lros derives folder metadata from LROs."""
        lro1 = MagicMock(spec=LongRecordingOrganizer)
        lro1.channel_names = ["Ch0"]
        lro1.base_folder_path = Path("/mock/animal/day1")
        lro1.get_date_string.return_value = "Jan-01-2023"

        lro2 = MagicMock(spec=LongRecordingOrganizer)
        lro2.channel_names = ["Ch0"]
        lro2.base_folder_path = Path("/mock/animal/day2")
        lro2.get_date_string.return_value = "Jan-02-2023"
        
        ao = AnimalOrganizer.from_lros(
            lros=[lro1, lro2],
            animal_id="TestAnimal",
        )
        
        # Verify AO was created successfully with the right number of recordings
        assert len(ao.long_recordings) == 2

    def test_from_lros_merges_duplicate_dates(self):
        """Test that from_lros automatically merges LROs with same date."""
        from datetime import datetime

        # Create 4 mock LROs: 2 for Jan-01, 1 for Jan-02, 1 for Jan-03
        lros = []

        # Two LROs for Jan-01-2023
        for i in range(2):
            lro = MagicMock(spec=LongRecordingOrganizer)
            lro.channel_names = ["Ch0", "Ch1"]
            lro.base_folder_path = Path(f"/mock/session{i}/day1")
            lro.get_date_string.return_value = "Jan-01-2023"
            lro.file_end_datetimes = [datetime(2023, 1, 1, 12 + i, 0)]
            lro.file_durations = [3600.0]

            # Mock merge method
            lro.merge = MagicMock()

            lros.append(lro)

        # One LRO for Jan-02-2023
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0", "Ch1"]
        lro.base_folder_path = Path("/mock/session0/day2")
        lro.get_date_string.return_value = "Jan-02-2023"
        lro.file_end_datetimes = [datetime(2023, 1, 2, 12, 0)]
        lro.file_durations = [3600.0]
        lros.append(lro)

        # One LRO for Jan-03-2023
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0", "Ch1"]
        lro.base_folder_path = Path("/mock/session1/day3")
        lro.get_date_string.return_value = "Jan-03-2023"
        lro.file_end_datetimes = [datetime(2023, 1, 3, 12, 0)]
        lro.file_durations = [3600.0]
        lros.append(lro)

        # Create AnimalOrganizer
        ao = AnimalOrganizer.from_lros(
            lros=lros,
            animal_id="TestAnimal",
            genotype="WT"
        )

        # Verify merge was called for Jan-01 LROs
        assert lros[0].merge.called or lros[1].merge.called, \
            "merge() should be called for duplicate date LROs"

        # Verify only 3 unique dates in final result
        assert len(ao.long_recordings) == 3, \
            f"Expected 3 merged LROs, got {len(ao.long_recordings)}"
        assert len(ao.unique_animaldays) == 3, \
            f"Expected 3 unique animaldays, got {len(ao.unique_animaldays)}"
        assert len(set(ao.unique_animaldays)) == 3, \
            "Animaldays should all be unique"

        # Verify animalday strings
        expected_dates = {"Jan-01-2023", "Jan-02-2023", "Jan-03-2023"}
        actual_dates = {day.split()[-1] for day in ao.unique_animaldays}
        assert actual_dates == expected_dates

    def test_from_lros_merge_incompatible_raises_error(self):
        """Test that incompatible LROs with same date raise clear error."""
        from datetime import datetime

        lros = []

        # LRO 1: Jan-01, channels Ch0, Ch1
        lro1 = MagicMock(spec=LongRecordingOrganizer)
        lro1.channel_names = ["Ch0", "Ch1"]
        lro1.get_date_string.return_value = "Jan-01-2023"
        lro1.file_end_datetimes = [datetime(2023, 1, 1, 12, 0)]
        lro1.file_durations = [3600.0]
        lro1.base_folder_path = Path("/mock/session0/day1")

        # LRO 2: Jan-01, DIFFERENT channels Ch2, Ch3
        lro2 = MagicMock(spec=LongRecordingOrganizer)
        lro2.channel_names = ["Ch2", "Ch3"]  # Incompatible!
        lro2.get_date_string.return_value = "Jan-01-2023"
        lro2.file_end_datetimes = [datetime(2023, 1, 1, 14, 0)]
        lro2.file_durations = [3600.0]
        lro2.base_folder_path = Path("/mock/session1/day1")

        # Mock merge to raise ValueError (mimicking real validation)
        lro1.merge = MagicMock(
            side_effect=ValueError("Channel names mismatch")
        )

        lros = [lro1, lro2]

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="Cannot merge LROs for"):
            AnimalOrganizer.from_lros(
                lros=lros,
                animal_id="TestAnimal",
                genotype="WT"
            )

    def test_consolidate_sessions_with_overlapping_dates(self):
        """
        Test end-to-end consolidation mimicking generate_wars.py workflow
        where multiple session AOs are consolidated via from_lros(), and
        some sessions share dates.
        """
        from datetime import datetime

        # Simulate 2 session AOs:
        # Session 1: Jan-01, Jan-02
        # Session 2: Jan-01 (same day!), Jan-03

        session1_lros = []
        session2_lros = []

        # Session 1, Day 1 (Jan-01)
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0", "Ch1"]
        lro.get_date_string.return_value = "Jan-01-2023"
        lro.file_end_datetimes = [datetime(2023, 1, 1, 8, 0)]  # Morning session
        lro.file_durations = [3600.0]
        lro.base_folder_path = Path("/mock/session1/day1")
        lro.merge = MagicMock()
        session1_lros.append(lro)

        # Session 1, Day 2 (Jan-02)
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0", "Ch1"]
        lro.get_date_string.return_value = "Jan-02-2023"
        lro.file_end_datetimes = [datetime(2023, 1, 2, 12, 0)]
        lro.file_durations = [3600.0]
        lro.base_folder_path = Path("/mock/session1/day2")
        session1_lros.append(lro)

        # Session 2, Day 1 (Jan-01 again!)
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0", "Ch1"]
        lro.get_date_string.return_value = "Jan-01-2023"
        lro.file_end_datetimes = [datetime(2023, 1, 1, 14, 0)]  # Afternoon session
        lro.file_durations = [3600.0]
        lro.base_folder_path = Path("/mock/session2/day1")
        lro.merge = MagicMock()
        session2_lros.append(lro)

        # Session 2, Day 2 (Jan-03)
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["Ch0", "Ch1"]
        lro.get_date_string.return_value = "Jan-03-2023"
        lro.file_end_datetimes = [datetime(2023, 1, 3, 12, 0)]
        lro.file_durations = [3600.0]
        lro.base_folder_path = Path("/mock/session2/day3")
        session2_lros.append(lro)

        # Consolidate all LROs (mimicking generate_wars.py line 142-153)
        all_lros = session1_lros + session2_lros

        ao = AnimalOrganizer.from_lros(
            lros=all_lros,
            animal_id="Animal123",
            genotype="WT"
        )

        # Should have exactly 3 unique dates (Jan-01, Jan-02, Jan-03)
        assert len(ao.long_recordings) == 3
        assert len(ao.unique_animaldays) == 3
        assert len(set(ao.unique_animaldays)) == 3

        # Verify Jan-01 LROs were merged
        # (Check that merge was called on at least one of the Jan-01 LROs)
        jan01_lros = [session1_lros[0], session2_lros[0]]
        merge_called = any(lro.merge.called for lro in jan01_lros)
        assert merge_called, "Jan-01 LROs should have been merged"

        # Verify animaldays
        expected_dates = {"Jan-01-2023", "Jan-02-2023", "Jan-03-2023"}
        actual_dates = {day.split()[-1] for day in ao.unique_animaldays}
        assert actual_dates == expected_dates

    def test_numeric_channel_names_require_explicit_map(self):
        """Channel resolution is exact: numeric names not in the configured map raise loudly
        (no number inference), while an explicitly-mapped raw name resolves."""
        from neurodent.core.utils import resolve_channel

        # Unconfigured numeric name -> loud raise (previously inferred via assume_from_number).
        with pytest.raises(ValueError, match="not in the configured channel map"):
            resolve_channel("0")
        with pytest.raises(ValueError, match="not in the configured channel map"):
            resolve_channel("channel_9")


# =============================================================================
# Tests: split()
# =============================================================================

class TestSplit:
    """Test AnimalOrganizer.split() method."""

    def test_split_creates_child_aos(self, mock_multi_day_lros):
        """Test that split creates child AnimalOrganizers."""
        # Create parent AO using from_lros
        parent_ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="Combined",
            genotype="WT",
        )
        
        splits = parent_ao.split({
            "AnimalA": ["Ch0", "Ch1"],
            "AnimalB": ["Ch2", "Ch3"],
        })
        
        assert "AnimalA" in splits
        assert "AnimalB" in splits
        assert isinstance(splits["AnimalA"], AnimalOrganizer)
        assert isinstance(splits["AnimalB"], AnimalOrganizer)

    def test_split_preserves_day_count(self, mock_multi_day_lros):
        """Test that each child AO has same number of days as parent."""
        parent_ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="Combined",
        )
        
        splits = parent_ao.split({"AnimalA": ["Ch0", "Ch1"]})
        
        assert len(splits["AnimalA"].long_recordings) == 3

    def test_split_inherits_genotype(self, mock_multi_day_lros):
        """Test that child AOs inherit parent genotype."""
        parent_ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="Combined",
            genotype="HET",
        )
        
        splits = parent_ao.split({"AnimalA": ["Ch0", "Ch1"]})
        
        assert splits["AnimalA"].genotype == "HET"

    def test_split_with_output_base(self, mock_multi_day_lros, tmp_path):
        """Test that output_base triggers LRO.save_recording() calls."""
        parent_ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="Combined",
        )

        splits = parent_ao.split(
            groups={"AnimalA": ["Ch0", "Ch1"]},
            output_base=tmp_path,
        )

        # Verify save_recording was called on each child LRO
        for child_lro in splits["AnimalA"].long_recordings:
            child_lro.save_recording.assert_called_once()

    def test_split_with_persist_base_deprecated(self, mock_multi_day_lros, tmp_path):
        """Test that the deprecated persist_base alias still triggers save_recording()."""
        parent_ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="Combined",
        )

        with pytest.warns(DeprecationWarning, match="persist_base"):
            splits = parent_ao.split(
                groups={"AnimalA": ["Ch0", "Ch1"]},
                persist_base=tmp_path,
            )

        # Verify save_recording was called on each child LRO
        for child_lro in splits["AnimalA"].long_recordings:
            child_lro.save_recording.assert_called_once()

    def test_split_no_recordings_raises_error(self):
        """Test that split raises error when no recordings loaded."""
        ao = AnimalOrganizer.from_lros(
            lros=[MagicMock(spec=LongRecordingOrganizer, channel_names=["Ch0"])],
            animal_id="Test",
        )
        ao.long_recordings = []  # Clear recordings
        
        with pytest.raises(ValueError, match="No recordings loaded to split"):
            ao.split({"A": ["Ch0"]})

    def test_split_empty_groups(self, mock_multi_day_lros):
        """Test split with empty groups dict returns empty dict."""
        parent_ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="Combined",
        )
        
        splits = parent_ao.split({})
        
        assert splits == {}

    def test_split_channel_subset(self, mock_multi_day_lros):
        """Test that child AOs have correct channel subset."""
        parent_ao = AnimalOrganizer.from_lros(
            lros=mock_multi_day_lros,
            animal_id="Combined",
        )
        
        splits = parent_ao.split({
            "AnimalA": ["Ch0", "Ch1"],
            "AnimalB": ["Ch2", "Ch3"],
        })
        
        assert splits["AnimalA"].channel_names == ["Ch0", "Ch1"]
        assert splits["AnimalB"].channel_names == ["Ch2", "Ch3"]


# =============================================================================
# Integration Tests (require SpikeInterface)
# =============================================================================

@pytest.mark.skipif(not SI_AVAILABLE, reason="SpikeInterface not available")
class TestSplitIntegration:
    """Integration tests using real SpikeInterface recordings."""

    @pytest.fixture
    def dummy_multi_day_ao(self, tmp_path):
        """Create a real AnimalOrganizer with multiple days."""
        # Create 2 days of recordings
        day_folders = []
        for day in range(2):
            duration_s = 2.0
            sampling_frequency = 1000.0
            num_channels = 4
            num_samples = int(duration_s * sampling_frequency)
            
            traces = np.random.randn(num_samples, num_channels).astype(np.float32)
            recording = si.NumpyRecording(
                traces_list=[traces],
                sampling_frequency=sampling_frequency,
            )
            
            day_folder = tmp_path / f"day{day}"
            recording.save(folder=day_folder, format="binary")
            day_folders.append(day_folder)
        
        # Create LROs manually with different dates for each day
        lros = []
        for day_idx, folder in enumerate(day_folders):
            lro = LongRecordingOrganizer(item=folder,
                
                manual_datetimes=datetime(2023, 1, day_idx + 1, 12, 0),  # Jan-01, Jan-02
            )
            lro.channel_names = ["Ch0", "Ch1", "Ch2", "Ch3"]
            lros.append(lro)
        
        # Create AO from LROs
        ao = AnimalOrganizer.from_lros(
            lros=lros,
            animal_id="TestAnimal",
            genotype="WT",
        )
        
        return ao

    def test_split_real_recordings(self, dummy_multi_day_ao, tmp_path):
        """Test splitting with real SI recordings."""
        splits = dummy_multi_day_ao.split({
            "GroupA": ["Ch0", "Ch1"],
            "GroupB": ["Ch2", "Ch3"],
        })
        
        # Verify splits
        assert len(splits) == 2
        assert splits["GroupA"].channel_names == ["Ch0", "Ch1"]
        assert splits["GroupB"].channel_names == ["Ch2", "Ch3"]
        
        # Verify each child AO has correct number of days
        assert len(splits["GroupA"].long_recordings) == 2
        assert len(splits["GroupB"].long_recordings) == 2

    def test_split_and_save_real_recordings(self, dummy_multi_day_ao, tmp_path):
        """Test splitting and saving real recordings to disk."""
        output_base = tmp_path / "output"

        splits = dummy_multi_day_ao.split(
            groups={"GroupA": ["Ch0", "Ch1"]},
            output_base=output_base,
            format="zarr",
        )

        # Verify output directories created
        group_a_dir = output_base / "GroupA"
        assert group_a_dir.exists()

        # Verify zarr files created
        zarr_files = list(group_a_dir.glob("*.zarr"))
        assert len(zarr_files) == 2

        # Each saved folder carries a NeuRodent sidecar for faithful reload
        from neurodent import constants
        for zf in zarr_files:
            assert (zf / constants.NEURODENT_SIDECAR_NAME).exists()

    def test_split_data_integrity(self, dummy_multi_day_ao):
        """Test that split preserves data integrity."""
        parent_traces = [
            lro.LongRecording.get_traces() 
            for lro in dummy_multi_day_ao.long_recordings
        ]
        
        splits = dummy_multi_day_ao.split({
            "GroupA": ["Ch0", "Ch1"],
        })
        
        child_traces = [
            lro.LongRecording.get_traces()
            for lro in splits["GroupA"].long_recordings
        ]
        
        # Verify first two channels match
        for parent, child in zip(parent_traces, child_traces):
            np.testing.assert_array_almost_equal(
                parent[:, :2],
                child,
                decimal=5
            )
