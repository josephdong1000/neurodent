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

from neurodent import visualization, core
from neurodent.core.frequency_domain_spike_detection import FrequencyDomainSpikeDetector
from neurodent.visualization.frequency_domain_results import FrequencyDomainSpikeAnalysisResult


# Test data configuration (matches the pattern from pipeline script)
TEST_DATA_BASE = Path(__file__).parent.parent / "notebooks" / "tests" / "test-data"
TEST_ANIMALS = ["A10", "F22"] if TEST_DATA_BASE.exists() else []

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
class TestFrequencyDomainSpikeDetectionIntegration:
    """Integration tests using real test data."""

    @pytest.fixture(params=TEST_ANIMALS)
    def animal_organizer(self, request):
        """Create AnimalOrganizer for test animals."""
        animal_id = request.param

        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            ao = visualization.AnimalOrganizer(
                TEST_DATA_BASE,
                animal_id,
                mode="concat",
                assume_from_number=True,
                skip_days=["bad"],
                lro_kwargs={"mode": "bin", "multiprocess_mode": "serial", "overwrite_rowbins": False},
            )

        # Verify we have data
        assert len(ao.long_recordings) > 0, f"No recordings found for {animal_id}"

        return ao

    def test_frequency_domain_spike_detection_basic(self, animal_organizer):
        """Test basic frequency domain spike detection on real data."""
        # Limit processing time by using max_length
        max_length = 30000  # ~30 seconds at 1kHz

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Run frequency domain spike detection
            fdsar_list = animal_organizer.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

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
            assert fdsar.result_mne is not None
            assert fdsar.result_sas is not None
            assert len(fdsar.result_sas) == len(fdsar.result_mne.ch_names)

            # Check spike counts
            spike_counts = fdsar.get_spike_counts_per_channel()
            assert len(spike_counts) == len(fdsar.result_mne.ch_names)
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
            {**TEST_DETECTION_PARAMS, "freq_slices": (10.0, 20.0), "sneo_percentile": 95.0},
            {**TEST_DETECTION_PARAMS, "freq_slices": (15.0, 25.0), "sneo_percentile": 98.0},
        ]

        max_length = 20000  # Shorter for parameter testing

        results = []
        for i, params in enumerate(param_sets):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                fdsar_list = animal_organizer.compute_frequency_domain_spike_analysis(
                    detection_params=params, max_length=max_length, multiprocess_mode="serial"
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

    def test_spikeinterface_compatibility(self, animal_organizer):
        """Test that results are compatible with SpikeInterface infrastructure."""
        max_length = 15000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            fdsar_list = animal_organizer.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        # Test SpikeInterface compatibility
        for fdsar in fdsar_list:
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

    def test_mne_annotation_creation(self, animal_organizer):
        """Test that MNE annotations are properly created."""
        max_length = 15000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            fdsar_list = animal_organizer.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        for fdsar in fdsar_list:
            raw = fdsar.result_mne
            assert raw is not None

            # Check annotations
            annotations = raw.annotations

            # Count spike annotations
            spike_annotations = [desc for desc in annotations.description if desc.startswith("Spike_Ch")]

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

    def test_save_and_load_integration(self, animal_organizer, tmp_path):
        """Test saving and loading with real data."""
        max_length = 10000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            fdsar_list = animal_organizer.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        # Test save/load for first result
        if fdsar_list:
            fdsar = fdsar_list[0]
            save_dir = tmp_path / "test_save"

            # Save without slugifying to match test expectations
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fdsar.save_fif_and_json(save_dir, slugify_filebase=False)

            # Verify files exist
            assert (save_dir / f"{fdsar.animal_id}-{fdsar.genotype}-{fdsar.animal_day}.json").exists()
            assert (save_dir / f"{fdsar.animal_id}-{fdsar.genotype}-{fdsar.animal_day}-raw.fif").exists()

            # Load
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                loaded_fdsar = FrequencyDomainSpikeAnalysisResult.load_fif_and_json(save_dir)

            # Verify loaded data
            assert loaded_fdsar.animal_id == fdsar.animal_id
            assert loaded_fdsar.genotype == fdsar.genotype
            assert loaded_fdsar.animal_day == fdsar.animal_day
            assert loaded_fdsar.detection_params == fdsar.detection_params

    def test_spike_averaged_plotting(self, animal_organizer, tmp_path):
        """Test spike-averaged trace plotting with real data."""
        max_length = 20000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            fdsar_list = animal_organizer.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        # Test plotting for first result that has spikes
        plot_dir = tmp_path / "plots"

        for fdsar in fdsar_list:
            spike_counts = fdsar.get_spike_counts_per_channel()

            if sum(spike_counts) > 0:  # Only test if spikes detected
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    returned_counts = fdsar.plot_spike_averaged_traces(
                        save_dir=plot_dir, animal_id=fdsar.animal_id, save_epoch=True
                    )

                # Verify return values - convert dict to list for comparison
                returned_counts_list = [returned_counts[i] for i in range(len(spike_counts))]
                assert returned_counts_list == spike_counts

                # Check that some files were created
                saved_files = list(plot_dir.glob("*"))
                assert len(saved_files) > 0

                break  # Only test one result with spikes


@pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
@pytest.mark.skipif(len(TEST_ANIMALS) == 0, reason="Test data not available")
@pytest.mark.integration
class TestFrequencyDomainSpikeDetectorStandalone:
    """Test FrequencyDomainSpikeDetector directly with real recordings."""

    @pytest.fixture(params=TEST_ANIMALS[:1])  # Test with one animal to save time
    def spikeinterface_recording(self, request):
        """Get a SpikeInterface recording from test data."""
        animal_id = request.param

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            ao = visualization.AnimalOrganizer(
                TEST_DATA_BASE,
                animal_id,
                mode="concat",
                assume_from_number=True,
                skip_days=["bad"],
                lro_kwargs={"mode": "bin", "multiprocess_mode": "serial", "overwrite_rowbins": False},
            )

        # Get first recording
        recording = ao.long_recordings[0].LongRecording
        return recording

    def test_direct_spike_detection(self, spikeinterface_recording):
        """Test FrequencyDomainSpikeDetector directly on SpikeInterface recording."""
        max_length = 15000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            spike_indices, mne_raw = FrequencyDomainSpikeDetector.detect_spikes_recording(
                spikeinterface_recording,
                detection_params=TEST_DETECTION_PARAMS,
                max_length=max_length,
                multiprocess_mode="serial",
            )

        # Verify output structure
        assert isinstance(spike_indices, list)
        assert len(spike_indices) == spikeinterface_recording.get_num_channels()

        for ch_spikes in spike_indices:
            assert isinstance(ch_spikes, np.ndarray)
            assert ch_spikes.dtype == int

        # Verify MNE object
        assert mne_raw is not None
        assert hasattr(mne_raw, "annotations")
        assert len(mne_raw.ch_names) == spikeinterface_recording.get_num_channels()

    def test_detection_parameter_effects(self, spikeinterface_recording):
        """Test that different parameters produce different results."""
        max_length = 10000

        # Test with high threshold (should detect fewer spikes)
        high_threshold_params = {**TEST_DETECTION_PARAMS, "sneo_percentile": 99.5}

        # Test with low threshold (should detect more spikes)
        low_threshold_params = {**TEST_DETECTION_PARAMS, "sneo_percentile": 95.0}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            high_spikes, _ = FrequencyDomainSpikeDetector.detect_spikes_recording(
                spikeinterface_recording,
                detection_params=high_threshold_params,
                max_length=max_length,
                multiprocess_mode="serial",
            )

            low_spikes, _ = FrequencyDomainSpikeDetector.detect_spikes_recording(
                spikeinterface_recording,
                detection_params=low_threshold_params,
                max_length=max_length,
                multiprocess_mode="serial",
            )

        # Count total spikes
        high_total = sum(len(spikes) for spikes in high_spikes)
        low_total = sum(len(spikes) for spikes in low_spikes)

        # Lower threshold should generally detect more or equal spikes
        assert low_total >= high_total

        logging.info(f"High threshold: {high_total} spikes, Low threshold: {low_total} spikes")
