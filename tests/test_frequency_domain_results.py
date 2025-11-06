"""
Unit tests for neurodent.visualization.frequency_domain_results module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
import warnings

try:
    import spikeinterface.core as si

    SPIKEINTERFACE_AVAILABLE = True
except ImportError:
    si = None
    SPIKEINTERFACE_AVAILABLE = False

import mne

from neurodent.visualization.frequency_domain_results import FrequencyDomainSpikeAnalysisResult
from neurodent import core


@pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
class TestFrequencyDomainSpikeAnalysisResult:
    """Test FrequencyDomainSpikeAnalysisResult class."""

    @pytest.fixture
    def sample_spike_indices(self):
        """Sample spike indices for testing."""
        return [
            np.array([500, 1500, 3000]),  # ch0
            np.array([800, 2200]),  # ch1
            np.array([1000]),  # ch2
            np.array([]),  # ch3 (no spikes)
        ]

    @pytest.fixture
    def sample_mne_raw(self, sample_spike_indices):
        """Create sample MNE RawArray with spike annotations."""
        n_channels = len(sample_spike_indices)
        fs = 1000.0
        duration = 5.0
        n_samples = int(duration * fs)

        info = mne.create_info(ch_names=[f"ch{i}" for i in range(n_channels)], sfreq=fs, ch_types="eeg")
        data = np.random.randn(n_channels, n_samples) * 0.1
        raw = mne.io.RawArray(data, info)

        # Add spike annotations
        onsets = []
        descriptions = []

        for ch_idx, spike_indices in enumerate(sample_spike_indices):
            for spike_idx in spike_indices:
                onsets.append(spike_idx / fs)
                descriptions.append(f"Spike_Ch{ch_idx}")

        if onsets:
            annotations = mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descriptions)
            raw.set_annotations(annotations)

        return raw

    @pytest.fixture
    def detection_params(self):
        """Sample detection parameters."""
        return {
            "bp": (3.0, 40.0),
            "notch": (59.0, 61.0),
            "freq_slices": (10.0, 20.0),
            "sneo_percentile": 99.9,
            "cluster_gap_ms": 80.0,
        }

    @pytest.fixture
    def mock_sorting_analyzer(self):
        """Create mock SortingAnalyzer for testing."""
        mock_sa = MagicMock()
        mock_sa.sorting.get_unit_ids.return_value = ["0"]
        mock_sa.sorting.get_sampling_frequency.return_value = 1000.0
        mock_sa.sorting.get_unit_spike_train.return_value = np.array([500, 1500, 3000])
        mock_sa.recording.get_channel_ids.return_value = ["ch0"]
        return mock_sa

    def test_init_with_result_sas(self, mock_sorting_analyzer, detection_params):
        """Test initialization with result_sas."""
        result_sas = [mock_sorting_analyzer]

        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_sas=result_sas,
            detection_params=detection_params,
            animal_id="test_animal",
            genotype="WT",
            channel_names=["ch0"],
        )

        assert fdsar.result_sas == result_sas
        assert fdsar.result_mne is None
        assert fdsar.detection_params == detection_params
        assert fdsar.animal_id == "test_animal"
        assert fdsar.genotype == "WT"

    def test_init_with_result_mne(self, sample_mne_raw, detection_params):
        """Test initialization with result_mne."""
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_mne=sample_mne_raw,
            detection_params=detection_params,
            animal_id="test_animal",
            genotype="WT",
            channel_names=sample_mne_raw.ch_names,
        )

        assert fdsar.result_sas is None
        assert fdsar.result_mne == sample_mne_raw
        assert fdsar.detection_params == detection_params

    def test_init_both_or_neither_raises_error(self, sample_mne_raw, mock_sorting_analyzer):
        """Test that providing both or neither result types raises error."""
        # Both provided
        with pytest.raises(ValueError, match="Exactly one of result_sas or result_mne must be provided"):
            FrequencyDomainSpikeAnalysisResult(result_sas=[mock_sorting_analyzer], result_mne=sample_mne_raw)

        # Neither provided
        with pytest.raises(ValueError, match="Exactly one of result_sas or result_mne must be provided"):
            FrequencyDomainSpikeAnalysisResult()

    @patch.object(FrequencyDomainSpikeAnalysisResult, "_convert_to_spikeinterface")
    def test_from_detection_results(self, mock_convert, sample_spike_indices, sample_mne_raw, detection_params):
        """Test creation from raw detection results."""
        mock_convert.return_value = [MagicMock()]

        fdsar = FrequencyDomainSpikeAnalysisResult.from_detection_results(
            spike_indices_per_channel=sample_spike_indices,
            mne_raw_with_annotations=sample_mne_raw,
            detection_params=detection_params,
            animal_id="test_animal",
            genotype="WT",
        )

        mock_convert.assert_called_once()
        assert fdsar.spike_indices == sample_spike_indices
        assert fdsar.result_mne == sample_mne_raw
        assert fdsar.detection_params == detection_params

    def test_convert_to_spikeinterface(self, sample_spike_indices, sample_mne_raw):
        """Test conversion to SpikeInterface format."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            result_sas = FrequencyDomainSpikeAnalysisResult._convert_to_spikeinterface(
                sample_spike_indices, sample_mne_raw
            )

        assert len(result_sas) == len(sample_spike_indices)

        # Check each channel
        for ch_idx, sa in enumerate(result_sas):
            assert hasattr(sa, "sorting")
            assert hasattr(sa, "recording")

            # Check units
            unit_ids = sa.sorting.get_unit_ids()
            if len(sample_spike_indices[ch_idx]) > 0:
                assert str(ch_idx) in unit_ids
                spike_train = sa.sorting.get_unit_spike_train(str(ch_idx))
                np.testing.assert_array_equal(spike_train, sample_spike_indices[ch_idx])
            else:
                assert len(unit_ids) == 0

    def test_get_spike_counts_per_channel(self, sample_spike_indices, sample_mne_raw, detection_params):
        """Test spike count extraction."""
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_mne=sample_mne_raw,
            spike_indices=sample_spike_indices,
            detection_params=detection_params,
            channel_names=sample_mne_raw.ch_names,
        )

        counts = fdsar.get_spike_counts_per_channel()
        expected_counts = [len(spikes) for spikes in sample_spike_indices]

        assert counts == expected_counts

    def test_get_spike_counts_from_mne_annotations(self, sample_mne_raw, detection_params):
        """Test spike count extraction from MNE annotations when spike_indices not available."""
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_mne=sample_mne_raw,
            spike_indices=None,  # No direct spike indices
            detection_params=detection_params,
            channel_names=sample_mne_raw.ch_names,
        )

        counts = fdsar.get_spike_counts_per_channel()

        # Should extract from annotations
        assert len(counts) == len(sample_mne_raw.ch_names)
        assert sum(counts) > 0  # Should have some spikes from annotations

    def test_get_total_spike_count(self, sample_spike_indices, sample_mne_raw, detection_params):
        """Test total spike count calculation."""
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_mne=sample_mne_raw,
            spike_indices=sample_spike_indices,
            detection_params=detection_params,
            channel_names=sample_mne_raw.ch_names,
        )

        total = fdsar.get_total_spike_count()
        expected_total = sum(len(spikes) for spikes in sample_spike_indices)

        assert total == expected_total

    def test_save_and_load_fif_and_json(self, sample_spike_indices, sample_mne_raw, detection_params):
        """Test saving and loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Create FDSAR
            fdsar = FrequencyDomainSpikeAnalysisResult(
                result_mne=sample_mne_raw,
                spike_indices=sample_spike_indices,
                detection_params=detection_params,
                animal_id="test_animal",
                genotype="WT",
                animal_day="day1",
                channel_names=sample_mne_raw.ch_names,
            )

            # Save
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fdsar.save_fif_and_json(save_dir)

            # Check files exist (slugify lowercases the filename)
            assert (save_dir / "test_animal-wt-day1-raw.fif").exists()
            assert (save_dir / "test_animal-wt-day1.json").exists()

            # Load
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                loaded_fdsar = FrequencyDomainSpikeAnalysisResult.load_fif_and_json(save_dir)

            # Check loaded data
            assert loaded_fdsar.animal_id == "test_animal"
            assert loaded_fdsar.genotype == "WT"
            assert loaded_fdsar.animal_day == "day1"
            assert loaded_fdsar.detection_params == detection_params
            assert loaded_fdsar.result_mne is not None

    def test_plot_spike_averaged_traces(self, sample_mne_raw, detection_params):
        """Test spike-averaged trace plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            fdsar = FrequencyDomainSpikeAnalysisResult(
                result_mne=sample_mne_raw, detection_params=detection_params, channel_names=sample_mne_raw.ch_names
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # Test plotting with saving
                counts = fdsar.plot_spike_averaged_traces(save_dir=save_dir, animal_id="test_animal", save_epoch=True)

            # Check that counts are returned
            assert isinstance(counts, dict)
            assert len(counts) == len(sample_mne_raw.ch_names)
            # Check that all channel indices are present
            assert set(counts.keys()) == set(range(len(sample_mne_raw.ch_names)))

            # Check that some files were created (if spikes were detected)
            saved_files = list(save_dir.glob("*"))
            if sum(counts.values()) > 0:
                assert len(saved_files) > 0

    def test_convert_to_mne(self, mock_sorting_analyzer, detection_params):
        """Test conversion to MNE format."""
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_sas=[mock_sorting_analyzer], detection_params=detection_params, channel_names=["ch0"]
        )

        # Mock the conversion method
        with patch("neurodent.visualization.results.SpikeAnalysisResult.convert_sas_to_mne") as mock_convert:
            mock_mne = MagicMock()
            mock_convert.return_value = mock_mne

            result = fdsar.convert_to_mne()

            mock_convert.assert_called_once()
            assert result == mock_mne

    def test_str_and_repr(self, sample_spike_indices, sample_mne_raw, detection_params):
        """Test string representations."""
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_mne=sample_mne_raw,
            spike_indices=sample_spike_indices,
            detection_params=detection_params,
            animal_id="test_animal",
            genotype="WT",
            animal_day="day1",
            channel_names=sample_mne_raw.ch_names,
        )

        str_repr = str(fdsar)
        assert "FrequencyDomainSpikeAnalysisResult" in str_repr
        assert "test_animal" in str_repr
        assert "WT" in str_repr
        assert "day1" in str_repr

        assert repr(fdsar) == str_repr


@pytest.mark.unit
class TestFrequencyDomainSpikeAnalysisResultUtils:
    """Test utility methods that don't require SpikeInterface."""

    def test_get_spike_counts_empty(self):
        """Test spike count methods with empty data."""
        # Create minimal FDSAR with no data
        fdsar = FrequencyDomainSpikeAnalysisResult.__new__(FrequencyDomainSpikeAnalysisResult)
        fdsar.spike_indices = []
        fdsar.result_mne = None

        counts = fdsar.get_spike_counts_per_channel()
        assert counts == []

        total = fdsar.get_total_spike_count()
        assert total == 0

    def test_init_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test that channel abbreviations are created when channel_names provided
        fdsar = FrequencyDomainSpikeAnalysisResult.__new__(FrequencyDomainSpikeAnalysisResult)
        fdsar.result_sas = None
        fdsar.result_mne = MagicMock()
        fdsar.spike_indices = []
        fdsar.detection_params = {}
        fdsar.animal_id = None
        fdsar.genotype = None
        fdsar.animal_day = None
        fdsar.bin_folder_name = None
        fdsar.metadata = None
        fdsar.channel_names = ["ch1", "ch2"]
        fdsar.assume_from_number = False

        # Mock the parse function
        with patch("neurodent.core.parse_chname_to_abbrev") as mock_parse:
            mock_parse.return_value = "parsed"

            fdsar.channel_abbrevs = [
                core.parse_chname_to_abbrev(x, assume_from_number=False) for x in fdsar.channel_names
            ]

            assert len(fdsar.channel_abbrevs) == 2
