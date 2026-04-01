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

    def test_plot_spike_averaged_traces_empty_epochs(self, detection_params):
        """Test that plot_spike_averaged_traces handles empty epochs gracefully.

        When spike annotations are at recording edges, the epoch window may
        extend beyond the recording, causing MNE to drop all epochs. The method
        should log a warning and continue instead of raising.
        """
        n_channels = 2
        fs = 1000.0
        duration = 1.0  # Short recording
        n_samples = int(duration * fs)

        info = mne.create_info(ch_names=[f"ch{i}" for i in range(n_channels)], sfreq=fs, ch_types="eeg")
        data = np.random.randn(n_channels, n_samples) * 0.1
        raw = mne.io.RawArray(data, info)

        # Place spikes at the very start (index 0) so epoch window extends
        # before recording start, causing MNE to drop all epochs
        annotations = mne.Annotations(
            onset=[0.0, 0.0],
            duration=[0.0, 0.0],
            description=["Spike_Ch0", "Spike_Ch1"],
        )
        raw.set_annotations(annotations)

        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_mne=raw,
            detection_params=detection_params,
            channel_names=raw.ch_names,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Use a wide window so epochs are dropped (tmin=-0.5 extends before t=0)
            counts = fdsar.plot_spike_averaged_traces(tmin=-0.5, tmax=0.5)

        # Should return without raising
        assert isinstance(counts, dict)
        assert len(counts) == n_channels

    def test_convert_to_mne(self, mock_sorting_analyzer, detection_params):
        """Test conversion to MNE format."""
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_sas=[mock_sorting_analyzer], detection_params=detection_params, channel_names=["ch0"]
        )

        # Mock the conversion method
        with patch("neurodent.visualization.frequency_domain_results.FrequencyDomainSpikeAnalysisResult.convert_sas_to_mne") as mock_convert:
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

    def test_load_fif_and_json_does_not_preload_data(self, sample_spike_indices, sample_mne_raw, detection_params):
        """load_fif_and_json() must open the .fif with preload=False (memory-mapped).

        This is critical for memory efficiency in make_fdsar_diagnostics: loading
        with preload=True pulls the full recording into RAM (~2-7 GB per animalday),
        causing SLURM OOM kills at 20 GB. With preload=False MNE memory-maps the
        file, and mne.Epochs reads only the windowed samples around spikes.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            fdsar = FrequencyDomainSpikeAnalysisResult(
                result_mne=sample_mne_raw,
                spike_indices=sample_spike_indices,
                detection_params=detection_params,
                animal_id="test_animal",
                genotype="WT",
                animal_day="day1",
                channel_names=sample_mne_raw.ch_names,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fdsar.save_fif_and_json(save_dir)
                loaded_fdsar = FrequencyDomainSpikeAnalysisResult.load_fif_and_json(save_dir)

            assert loaded_fdsar.result_mne.preload is False

    def test_plot_spike_averaged_traces_with_memmap_raw(self, sample_mne_raw, detection_params):
        """plot_spike_averaged_traces() works correctly when raw is memory-mapped (preload=False).

        Verifies that mne.Epochs with preload=True correctly loads windowed data
        from a memory-mapped .fif file, producing the same results as preloaded data.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            fdsar_orig = FrequencyDomainSpikeAnalysisResult(
                result_mne=sample_mne_raw,
                detection_params=detection_params,
                channel_names=sample_mne_raw.ch_names,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fdsar_orig.save_fif_and_json(save_dir)

            # Load with preload=False (the new default)
            plot_dir = Path(temp_dir) / "plots"
            plot_dir.mkdir()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                loaded_fdsar = FrequencyDomainSpikeAnalysisResult.load_fif_and_json(save_dir)
                assert loaded_fdsar.result_mne.preload is False
                counts = loaded_fdsar.plot_spike_averaged_traces(
                    save_dir=plot_dir, animal_id="test_animal", save_epoch=False
                )

            assert isinstance(counts, dict)
            assert set(counts.keys()) == set(range(len(sample_mne_raw.ch_names)))


def _make_mock_sorting_analyzer(traces_uv, sfreq, channel_id, spike_samples):
    """Build a mock SortingAnalyzer backed by a real numpy array.

    Args:
        traces_uv: 1-D numpy array of trace values in micro-volts.
        sfreq: Sampling frequency (Hz).
        channel_id: Channel identifier string.
        spike_samples: 1-D numpy array of spike sample indices.
    """
    n_samples = len(traces_uv)
    duration = n_samples / sfreq

    rec = MagicMock()
    rec.get_sampling_frequency.return_value = sfreq
    rec.get_duration.return_value = duration
    rec.get_channel_ids.return_value = np.array([channel_id])

    def _get_traces(start_frame=0, end_frame=None, return_scaled=True):
        end = end_frame if end_frame is not None else n_samples
        return traces_uv[start_frame:end].reshape(-1, 1).copy()

    rec.get_traces = _get_traces

    sorting = MagicMock()
    if len(spike_samples) > 0:
        sorting.get_unit_ids.return_value = ["0"]
        sorting.get_sampling_frequency.return_value = sfreq
        sorting.get_unit_spike_train.return_value = spike_samples
    else:
        sorting.get_unit_ids.return_value = []
        sorting.get_sampling_frequency.return_value = sfreq

    sa = MagicMock()
    sa.recording = rec
    sa.sorting = sorting
    return sa


@pytest.mark.unit
class TestConvertDaskParallelization:
    """Test that Dask multiprocess mode produces the same results as serial."""

    @pytest.fixture
    def generated_sorting_analyzers(self):
        """Generate a list of mock SortingAnalyzers with deterministic data.

        Returns three channels each with 120 s of data at 1000 Hz (long enough
        to exercise the chunking logic at the default chunk_duration_s=60).
        """
        rng = np.random.default_rng(42)
        sfreq = 1000.0
        duration_s = 120.0
        n_samples = int(duration_s * sfreq)

        sas = []
        for ch_idx in range(3):
            traces = rng.standard_normal(n_samples).astype(np.float64) * 100  # µV
            spikes = np.sort(rng.choice(n_samples, size=10, replace=False))
            sa = _make_mock_sorting_analyzer(traces, sfreq, f"ch{ch_idx}", spikes)
            sas.append(sa)
        return sas

    @pytest.fixture
    def single_sorting_analyzer(self):
        """Generate a single mock SortingAnalyzer for convert_sa_to_np tests."""
        rng = np.random.default_rng(99)
        sfreq = 1000.0
        duration_s = 180.0  # 3 chunks at default chunk_duration_s=60
        n_samples = int(duration_s * sfreq)
        traces = rng.standard_normal(n_samples).astype(np.float64) * 100
        spikes = np.sort(rng.choice(n_samples, size=5, replace=False))
        return _make_mock_sorting_analyzer(traces, sfreq, "ch0", spikes)

    # --- convert_sas_to_mne tests ---

    def test_convert_sas_to_mne_serial_equals_dask(self, generated_sorting_analyzers):
        """Serial and Dask modes should produce identical MNE RawArrays."""
        raw_serial = FrequencyDomainSpikeAnalysisResult.convert_sas_to_mne(
            generated_sorting_analyzers, multiprocess_mode="serial"
        )
        raw_dask = FrequencyDomainSpikeAnalysisResult.convert_sas_to_mne(
            generated_sorting_analyzers, multiprocess_mode="dask"
        )

        # Data must be identical
        np.testing.assert_array_equal(raw_serial.get_data(), raw_dask.get_data())
        assert raw_serial.ch_names == raw_dask.ch_names
        assert raw_serial.info["sfreq"] == raw_dask.info["sfreq"]

        # Annotations must match
        assert len(raw_serial.annotations) == len(raw_dask.annotations)
        np.testing.assert_array_almost_equal(
            raw_serial.annotations.onset, raw_dask.annotations.onset
        )

    def test_convert_sas_to_mne_correct_shape(self, generated_sorting_analyzers):
        """Output should have the correct number of channels and samples."""
        raw = FrequencyDomainSpikeAnalysisResult.convert_sas_to_mne(
            generated_sorting_analyzers, multiprocess_mode="dask"
        )
        assert raw.get_data().shape[0] == 3  # 3 channels
        assert raw.get_data().shape[1] == 120_000  # 120 s * 1000 Hz

    def test_convert_sas_to_mne_empty_list(self):
        """Empty input should return None in both modes."""
        assert FrequencyDomainSpikeAnalysisResult.convert_sas_to_mne(
            [], multiprocess_mode="serial"
        ) is None
        assert FrequencyDomainSpikeAnalysisResult.convert_sas_to_mne(
            [], multiprocess_mode="dask"
        ) is None

    # --- convert_sa_to_np tests ---

    def test_convert_sa_to_np_serial_equals_dask(self, single_sorting_analyzer):
        """Serial and Dask modes should produce identical numpy arrays."""
        traces_serial = FrequencyDomainSpikeAnalysisResult.convert_sa_to_np(
            single_sorting_analyzer, multiprocess_mode="serial"
        )
        traces_dask = FrequencyDomainSpikeAnalysisResult.convert_sa_to_np(
            single_sorting_analyzer, multiprocess_mode="dask"
        )
        np.testing.assert_array_equal(traces_serial, traces_dask)

    def test_convert_sa_to_np_correct_length(self, single_sorting_analyzer):
        """Output length should equal total_frames."""
        traces = FrequencyDomainSpikeAnalysisResult.convert_sa_to_np(
            single_sorting_analyzer, multiprocess_mode="dask"
        )
        assert len(traces) == 180_000  # 180 s * 1000 Hz

    def test_convert_sa_to_np_uv_to_v_scaling(self, single_sorting_analyzer):
        """Traces should be scaled from µV to V (×1e-6)."""
        traces = FrequencyDomainSpikeAnalysisResult.convert_sa_to_np(
            single_sorting_analyzer, multiprocess_mode="serial"
        )
        # Raw mock data has std ~100 µV → after scaling std ~1e-4 V
        assert np.abs(traces).max() < 1.0  # definitely in V, not µV

    def test_convert_sa_to_np_multi_channel_raises(self):
        """Should raise ValueError when SortingAnalyzer has more than 1 channel."""
        sa = MagicMock()
        sa.recording.get_channel_ids.return_value = np.array(["ch0", "ch1"])
        with pytest.raises(ValueError, match="Expected SortingAnalyzer to have 1 channel"):
            FrequencyDomainSpikeAnalysisResult.convert_sa_to_np(
                sa, multiprocess_mode="serial"
            )
        with pytest.raises(ValueError, match="Expected SortingAnalyzer to have 1 channel"):
            FrequencyDomainSpikeAnalysisResult.convert_sa_to_np(
                sa, multiprocess_mode="dask"
            )


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
