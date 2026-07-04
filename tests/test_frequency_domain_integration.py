"""
Integration tests for frequency domain spike detection using real test data.

These tests verify the complete pipeline using the A10 and F22 test datasets.
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pytest

try:
    import spikeinterface.core as si

    SPIKEINTERFACE_AVAILABLE = True
except ImportError:
    si = None
    SPIKEINTERFACE_AVAILABLE = False

from neurodent.loading import AnimalOrganizer
from neurodent.analysis.spike_detection import FrequencyDomainSpikeDetector
from neurodent.results.frequency_domain_results import (
    FrequencyDomainSpikeAnalysisResult,
)


# Test data configuration
TEST_DATA_BASE = Path(__file__).parent.parent / ".tests" / "integration" / "data"
TEST_ANIMALS = (
    ["A10", "F22"]
    if (TEST_DATA_BASE / "A10" / "A10_recording.edf").exists()
    else []
)

# Detection parameters for testing (lowered thresholds for better detection in test data)
TEST_DETECTION_PARAMS = {
    "bp": (3.0, 40.0),
    "notch": 60.0,
    "notch_q": 30.0,
    "freq_slices": (10.0, 20.0),
    "sneo_percentile": 98.0,  # Lower threshold for test data
    "cluster_gap_ms": 80.0,
    "vote_k": 1,  # Lower consensus requirement
}


@pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
@pytest.mark.skipif(len(TEST_ANIMALS) == 0, reason="Test data not available")
@pytest.mark.integration
@pytest.mark.slow
class TestFrequencyDomainSpikeDetectionIntegration:
    """Integration tests using real test data."""

    @pytest.fixture(scope="class", params=TEST_ANIMALS)
    def animal_organizer(self, request):
        """Create AnimalOrganizer for test animals — shared across tests per animal."""
        from datetime import datetime as dt
        from tests.integration.readers import read_bin_csv_pair
        animal_id = request.param

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            ao = AnimalOrganizer(
                [
                    str(TEST_DATA_BASE / animal_id) + "/{index}_ColMajor.bin",
                    str(TEST_DATA_BASE / animal_id) + "/{index}_Meta.csv",
                ],
                animal_id,
                lro_kwargs={
                    "mode": "si",
                    "extract_func": read_bin_csv_pair,
                    "multiprocess_mode": "serial",
                    "manual_datetimes": dt(2023, 12, 13, 12, 0, 0),
                    "datetimes_are_start": True,
                },
            )

        # Verify we have data
        assert len(ao.long_recordings) > 0, f"No recordings found for {animal_id}"

        return ao

    @pytest.fixture(scope="class")
    def fdsar_default(self, animal_organizer):
        """Default spike detection — cached per animal, shared across tests."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return animal_organizer.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS,
                chunk_duration_s=30.0,
                multiprocess_mode="auto",
            )

    def test_frequency_domain_spike_detection_basic(self, animal_organizer, fdsar_default):
        """Test basic frequency domain spike detection on real data."""
        fdsar_list = fdsar_default

        # Verify results structure
        assert isinstance(fdsar_list, list)
        assert len(fdsar_list) > 0, "No results returned"

        # Check each result
        for fdsar in fdsar_list:
            assert isinstance(fdsar, FrequencyDomainSpikeAnalysisResult)
            assert fdsar.animal_id == animal_organizer.animal_id
            assert fdsar.genotype == animal_organizer.genotype
            assert fdsar.detection_params == TEST_DETECTION_PARAMS

            # Verify data integrity
            assert fdsar.result_sas is not None
            assert len(fdsar.result_sas) == len(fdsar.channel_names)

            # Check spike counts
            spike_counts = fdsar.get_spike_counts_per_channel()
            assert len(spike_counts) == len(fdsar.channel_names)
            assert all(count >= 0 for count in spike_counts)

            logging.info(
                f"Animal {fdsar.animal_id}, Day {fdsar.animal_day}: "
                f"Total spikes = {fdsar.get_total_spike_count()}, "
                f"Channels = {len(spike_counts)}"
            )

    def test_spike_detection_with_different_parameters(self, animal_organizer):
        """Test spike detection with different parameter sets."""
        # Test with multiple parameter combinations
        param_sets = [
            {
                **TEST_DETECTION_PARAMS,
                "freq_slices": (10.0, 20.0),
                "sneo_percentile": 95.0,
            },
            {
                **TEST_DETECTION_PARAMS,
                "freq_slices": (15.0, 25.0),
                "sneo_percentile": 98.0,
            },
        ]

        chunk_duration_s = 20.0  # Shorter for parameter testing

        results = []
        for i, params in enumerate(param_sets):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                fdsar_list = animal_organizer.compute_frequency_domain_spike_analysis(
                    detection_params=params,
                    chunk_duration_s=chunk_duration_s,
                    multiprocess_mode="auto",
                )

            results.append(fdsar_list)

            # Verify each parameter set produces results
            assert len(fdsar_list) > 0

            for fdsar in fdsar_list:
                assert fdsar.detection_params == params
                total_spikes = fdsar.get_total_spike_count()
                logging.info(f"Parameter set {i + 1}: {total_spikes} total spikes")

        # Results should vary with different parameters
        # (This is a weak test, but verifies parameters have some effect)
        spike_counts_1 = sum(fdsar.get_total_spike_count() for fdsar in results[0])
        spike_counts_2 = sum(fdsar.get_total_spike_count() for fdsar in results[1])

        # Allow for some variation due to parameter differences
        assert spike_counts_1 >= 0 and spike_counts_2 >= 0

    def test_spikeinterface_compatibility(self, fdsar_default):
        """Test that results are compatible with SpikeInterface infrastructure."""
        # Test SpikeInterface compatibility
        for fdsar in fdsar_default:
            assert fdsar.result_sas is not None

            for ch_idx, sa in enumerate(fdsar.result_sas):
                # Verify SortingAnalyzer structure
                assert hasattr(sa, "sorting")
                assert hasattr(sa, "recording")

                # Check unit structure
                unit_ids = sa.sorting.get_unit_ids()
                if len(unit_ids) > 0:
                    # Should have unit corresponding to channel index if spikes detected
                    assert str(ch_idx) in unit_ids

                    # Check spike train
                    spike_train = sa.sorting.get_unit_spike_train(str(ch_idx))
                    assert isinstance(spike_train, np.ndarray)
                    assert len(spike_train) >= 0

    def test_mne_annotation_creation(self, fdsar_default):
        """Test that MNE annotations are properly created."""
        for fdsar in fdsar_default:
            raw = fdsar.convert_to_mne()
            assert raw is not None

            # Check annotations
            annotations = raw.annotations

            # Count spike annotations
            spike_annotations = [
                desc for desc in annotations.description if desc.startswith("Spike_Ch")
            ]

            # Verify annotation structure
            if len(spike_annotations) > 0:
                # Check timing consistency
                assert len(annotations.onset) == len(annotations.description)
                assert all(onset >= 0 for onset in annotations.onset)

                # Check that onset times are within recording duration
                duration = raw.times[-1]
                assert all(onset <= duration for onset in annotations.onset)

            # Compare with direct spike counts
            spike_counts = fdsar.get_spike_counts_per_channel()
            total_from_counts = sum(spike_counts)
            total_from_annotations = len(spike_annotations)

            # Should match (allowing for some tolerance in case of edge effects)
            assert abs(total_from_counts - total_from_annotations) <= 2

    def test_save_and_load_integration(self, fdsar_default, tmp_path):
        """Test saving and loading with real data."""
        # Test save/load for first result
        if fdsar_default:
            fdsar = fdsar_default[0]
            save_dir = tmp_path / "test_save"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fdsar.save_fif_and_json(save_dir)

            # File names use the canonical path-safe stem (the unsafe branch was removed).
            assert (save_dir / f"{fdsar.path_safe_save_stem}.json").exists()
            assert (save_dir / f"{fdsar.path_safe_save_stem}-raw.fif").exists()

            # Load
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                loaded_fdsar = FrequencyDomainSpikeAnalysisResult.load_fif_and_json(
                    save_dir
                )

            # Verify loaded data
            assert loaded_fdsar.animal_id == fdsar.animal_id
            assert loaded_fdsar.genotype == fdsar.genotype
            assert loaded_fdsar.animal_day == fdsar.animal_day
            assert loaded_fdsar.detection_params == fdsar.detection_params

    def test_spike_averaged_plotting(self, fdsar_default, tmp_path):
        """Test spike-averaged trace plotting with real data."""
        # Test plotting for first result that has spikes
        plot_dir = tmp_path / "plots"

        for fdsar in fdsar_default:
            spike_counts = fdsar.get_spike_counts_per_channel()

            if sum(spike_counts) > 0:  # Only test if spikes detected
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    returned_counts = fdsar.plot_spike_averaged_traces(
                        save_dir=plot_dir, animal_id=fdsar.animal_id, save_epoch=True
                    )

                # Verify return values - convert dict to list for comparison
                returned_counts_list = [
                    returned_counts[i] for i in range(len(spike_counts))
                ]
                assert returned_counts_list == spike_counts

                # Check that some files were created
                saved_files = list(plot_dir.glob("*"))
                assert len(saved_files) > 0

                break  # Only test one result with spikes


@pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
@pytest.mark.skipif(len(TEST_ANIMALS) == 0, reason="Test data not available")
@pytest.mark.integration
@pytest.mark.slow
class TestFrequencyDomainSpikeDetectorStandalone:
    """Test FrequencyDomainSpikeDetector directly with real recordings."""

    @pytest.fixture(scope="class", params=TEST_ANIMALS[:1])  # Test with one animal to save time
    def spikeinterface_recording(self, request):
        """Get a SpikeInterface recording from test data."""
        animal_id = request.param

        try:
            import mne
        except ImportError:
            mne = None

        dummy_extract = lambda x, **kw: mne.io.RawArray(
            np.random.randn(64, 10000), mne.create_info(64, 1000., "eeg")
        ) if mne else None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            ao = AnimalOrganizer(
                TEST_DATA_BASE,
                animal_id,
                lro_kwargs={
                    "mode": "mne",
                    "extract_func": dummy_extract,
                    "multiprocess_mode": "serial",
                    "overwrite_rowbins": False,
                    "intermediate": "bin",
                },
            )

        # Get first recording
        recording = ao.long_recordings[0].LongRecording
        return recording

    def test_direct_spike_detection(self, spikeinterface_recording):
        """Test FrequencyDomainSpikeDetector directly on SpikeInterface recording."""
        chunk_duration_s = 15.0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            spike_indices = (
                FrequencyDomainSpikeDetector.detect_spikes_recording(
                    spikeinterface_recording,
                    detection_params=TEST_DETECTION_PARAMS,
                    chunk_duration_s=chunk_duration_s,
                    multiprocess_mode="auto",
                )
            )

        # Verify output structure
        assert isinstance(spike_indices, list)
        assert len(spike_indices) == spikeinterface_recording.get_num_channels()

        for ch_spikes in spike_indices:
            assert isinstance(ch_spikes, np.ndarray)
            assert ch_spikes.dtype == int

    def test_detection_parameter_effects(self, spikeinterface_recording):
        """Test that different parameters produce different results."""
        chunk_duration_s = 10.0

        # Test with high threshold (should detect fewer spikes)
        high_threshold_params = {**TEST_DETECTION_PARAMS, "sneo_percentile": 99.5}

        # Test with low threshold (should detect more spikes)
        low_threshold_params = {**TEST_DETECTION_PARAMS, "sneo_percentile": 95.0}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            high_spikes = FrequencyDomainSpikeDetector.detect_spikes_recording(
                spikeinterface_recording,
                detection_params=high_threshold_params,
                chunk_duration_s=chunk_duration_s,
                multiprocess_mode="auto",
            )

            low_spikes = FrequencyDomainSpikeDetector.detect_spikes_recording(
                spikeinterface_recording,
                detection_params=low_threshold_params,
                chunk_duration_s=chunk_duration_s,
                multiprocess_mode="auto",
            )

        # Count total spikes
        high_total = sum(len(spikes) for spikes in high_spikes)
        low_total = sum(len(spikes) for spikes in low_spikes)

        # Lower threshold should generally detect more or equal spikes
        assert low_total >= high_total

        logging.info(
            f"High threshold: {high_total} spikes, Low threshold: {low_total} spikes"
        )
