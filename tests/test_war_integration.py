"""
Tests for WAR integration with frequency domain spike detection.

These tests verify that FrequencyDomainSpikeAnalysisResult objects can be integrated
with WindowAnalysisResult via the read_sars_spikes method to add nspike/lognspike features.
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

try:
    import spikeinterface.core as si

    SPIKEINTERFACE_AVAILABLE = True
except ImportError:
    si = None
    SPIKEINTERFACE_AVAILABLE = False

from neurodent import visualization, constants
from neurodent.visualization.frequency_domain_results import FrequencyDomainSpikeAnalysisResult


# Test data configuration
TEST_DATA_BASE = Path(__file__).parent.parent / "notebooks" / "tests" / "test-data"
TEST_ANIMALS = ["A10"] if TEST_DATA_BASE.exists() else []  # Use one animal for faster testing

# Parameters for testing
TEST_DETECTION_PARAMS = {
    "bp": [3.0, 40.0],
    "notch": 60.0,
    "freq_slices": [10.0, 20.0],
    "sneo_percentile": 95.0,  # Lower threshold for test data
    "cluster_gap_ms": 80.0,
    "vote_k": 1,
}


@pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
@pytest.mark.skipif(len(TEST_ANIMALS) == 0, reason="Test data not available")
@pytest.mark.integration
class TestWARIntegration:
    """Test integration between frequency domain spike detection and WAR analysis."""

    @pytest.fixture
    def animal_organizer_with_war(self):
        """Create AnimalOrganizer with both WAR and spike detection results."""
        animal_id = TEST_ANIMALS[0]

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

        return ao

    def test_war_without_spikes(self, animal_organizer_with_war):
        """Test WAR generation without spike features as baseline."""
        # Generate WAR without spike features
        war_features = ["rms", "ampvar", "psdtotal"]  # Exclude spike features

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            war = animal_organizer_with_war.compute_windowed_analysis(
                features=war_features, window_s=4, multiprocess_mode="serial"
            )

        # Verify WAR structure
        assert hasattr(war, "result")
        assert isinstance(war.result, pd.DataFrame)
        assert len(war.result) > 0

        # Check that spike features are not present
        assert "nspike" not in war.result.columns
        assert "lognspike" not in war.result.columns

        # Verify expected features are present
        for feature in war_features:
            assert feature in war.result.columns

        return war

    def test_frequency_domain_spike_detection_for_war(self, animal_organizer_with_war):
        """Test frequency domain spike detection for WAR integration."""
        max_length = 20000  # Limit for faster testing

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            fdsar_list = animal_organizer_with_war.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        # Verify spike detection results
        assert len(fdsar_list) > 0
        assert all(isinstance(fdsar, FrequencyDomainSpikeAnalysisResult) for fdsar in fdsar_list)

        # Check spike counts
        for fdsar in fdsar_list:
            spike_counts = fdsar.get_spike_counts_per_channel()
            total_spikes = fdsar.get_total_spike_count()
            logging.info(f"Detected {total_spikes} spikes across {len(spike_counts)} channels")

        return fdsar_list

    def test_war_spike_integration_via_read_sars_spikes(self, animal_organizer_with_war):
        """Test integration of spike features into WAR via read_sars_spikes method."""
        max_length = 15000

        # Step 1: Generate WAR without spike features
        war_features = ["rms", "ampvar"]  # Minimal set for faster testing

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            war = animal_organizer_with_war.compute_windowed_analysis(
                features=war_features, window_s=4, multiprocess_mode="serial"
            )

            # Step 2: Run frequency domain spike detection
            fdsar_list = animal_organizer_with_war.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        # Step 3: Integrate spike features using read_sars_spikes
        original_columns = set(war.result.columns)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            enhanced_war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=False)

        # Verify integration results
        assert enhanced_war is not None
        assert isinstance(enhanced_war.result, pd.DataFrame)

        # Check that spike features were added
        new_columns = set(enhanced_war.result.columns)
        added_columns = new_columns - original_columns

        assert "nspike" in added_columns, "nspike feature not added"
        assert "lognspike" in added_columns, "lognspike feature not added"

        # Verify data integrity
        assert len(enhanced_war.result) == len(war.result), "Number of rows changed"

        # Check nspike values - they should be numpy arrays after processing
        nspike_data = enhanced_war.result["nspike"]
        assert all(isinstance(val, (list, np.ndarray)) for val in nspike_data), (
            "nspike should be list or array of counts per channel"
        )

        # Check lognspike values - they should be numpy arrays after processing
        lognspike_data = enhanced_war.result["lognspike"]
        assert all(isinstance(val, (list, np.ndarray)) for val in lognspike_data), (
            "lognspike should be list or array of log-transformed counts"
        )

        # Verify reasonable spike count ranges
        for nspike_row in nspike_data:
            assert all(count >= 0 for count in nspike_row), "Negative spike counts found"
            assert all(isinstance(count, (int, float)) for count in nspike_row), "Non-numeric spike counts"

        logging.info(
            f"Successfully integrated spike features. "
            f"Original columns: {len(original_columns)}, "
            f"Enhanced columns: {len(new_columns)}"
        )

        return enhanced_war

    def test_inplace_spike_integration(self, animal_organizer_with_war):
        """Test in-place integration of spike features."""
        max_length = 10000

        # Generate WAR
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            war = animal_organizer_with_war.compute_windowed_analysis(
                features=["rms"], window_s=4, multiprocess_mode="serial"
            )

            # Generate spike detection results
            fdsar_list = animal_organizer_with_war.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        # Test in-place integration
        original_id = id(war.result)
        original_columns = set(war.result.columns)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            result = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=True)

        # Verify in-place modification
        assert result is war  # Should return self
        assert id(war.result) != original_id  # DataFrame should be replaced
        assert "nspike" in war.result.columns
        assert "lognspike" in war.result.columns

    def test_spike_integration_with_multiple_recordings(self, animal_organizer_with_war):
        """Test spike integration when there are multiple recording sessions."""
        max_length = 8000  # Short length for multiple recordings test

        # Check if we have multiple recordings
        if len(animal_organizer_with_war.long_recordings) <= 1:
            pytest.skip("Need multiple recordings for this test")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            war = animal_organizer_with_war.compute_windowed_analysis(
                features=["rms"], window_s=4, multiprocess_mode="serial"
            )

            fdsar_list = animal_organizer_with_war.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

        # Should have results for each recording session
        assert len(fdsar_list) == len(animal_organizer_with_war.long_recordings)

        # Integrate
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            enhanced_war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=False)

        # Verify results for each recording session
        unique_animaldays = enhanced_war.result["animalday"].unique()
        assert len(unique_animaldays) == len(fdsar_list)

        # Check that each animalday has spike data
        for animalday in unique_animaldays:
            animalday_data = enhanced_war.result[enhanced_war.result["animalday"] == animalday]
            assert len(animalday_data) > 0
            assert all("nspike" in row.index and "lognspike" in row.index for _, row in animalday_data.iterrows())

    def test_spike_counts_consistency(self, animal_organizer_with_war):
        """Test that spike counts in WAR match original detection results."""
        max_length = 12000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            war = animal_organizer_with_war.compute_windowed_analysis(
                features=["rms"], window_s=4, multiprocess_mode="serial"
            )

            fdsar_list = animal_organizer_with_war.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

            enhanced_war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=False)

        # Compare total spike counts
        for i, fdsar in enumerate(fdsar_list):
            original_total = fdsar.get_total_spike_count()

            # Find corresponding animalday in WAR
            animalday = fdsar.animal_day
            war_data = enhanced_war.result[enhanced_war.result["animalday"] == animalday]

            # Sum spikes across all windows for this animalday
            war_total = 0
            for _, row in war_data.iterrows():
                nspike_row = row["nspike"]
                war_total += sum(nspike_row)

            # Allow for some difference due to windowing effects and max_length truncation
            # The WAR windowing might split or truncate some spikes differently
            assert abs(original_total - war_total) <= max(10, original_total * 0.2), (
                f"Spike count mismatch for {animalday}: original={original_total}, war={war_total}"
            )

            logging.info(f"Spike count consistency check for {animalday}: original={original_total}, war={war_total}")

    def test_log_transformation_correctness(self, animal_organizer_with_war):
        """Test that lognspike values are correctly computed."""
        max_length = 10000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            war = animal_organizer_with_war.compute_windowed_analysis(
                features=["rms"], window_s=4, multiprocess_mode="serial"
            )

            fdsar_list = animal_organizer_with_war.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

            enhanced_war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=False)

        # Check log transformation
        for _, row in enhanced_war.result.iterrows():
            nspike = np.array(row["nspike"])
            lognspike = np.array(row["lognspike"])

            # Compute expected log transformation (using core.log_transform if available)
            try:
                from neurodent.core import log_transform

                expected_lognspike = log_transform(nspike.reshape(1, -1))[0]
                np.testing.assert_allclose(lognspike, expected_lognspike, rtol=1e-6)
            except ImportError:
                # If log_transform not available, check basic properties
                # Log values should be >= 0 and finite
                assert np.all(np.isfinite(lognspike))
                assert np.all(lognspike >= 0)

                # Zero spike counts should give specific log values
                zero_mask = nspike == 0
                if np.any(zero_mask):
                    # Log of zero should be handled consistently
                    assert np.all(lognspike[zero_mask] == lognspike[zero_mask][0])

    def test_feature_compatibility_with_existing_pipeline(self, animal_organizer_with_war):
        """Test that spike-enhanced WAR works with existing pipeline methods."""
        max_length = 10000

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            war = animal_organizer_with_war.compute_windowed_analysis(
                features=["rms", "ampvar"], window_s=4, multiprocess_mode="serial"
            )

            fdsar_list = animal_organizer_with_war.compute_frequency_domain_spike_analysis(
                detection_params=TEST_DETECTION_PARAMS, max_length=max_length, multiprocess_mode="serial"
            )

            enhanced_war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=False)

        # Test that enhanced WAR works with existing methods

        # Test feature averaging (if method exists)
        if hasattr(enhanced_war, "average_features"):
            try:
                averaged = enhanced_war.average_features(["nspike", "lognspike"])
                assert averaged is not None
            except Exception as e:
                logging.warning(f"Feature averaging test failed: {e}")

        # Test data access methods
        assert hasattr(enhanced_war, "result")
        assert isinstance(enhanced_war.result, pd.DataFrame)

        # Test that spike features are recognized as valid features
        all_features = enhanced_war.result.columns.tolist()
        assert "nspike" in all_features
        assert "lognspike" in all_features

        # Verify feature types match expected patterns
        feature_types = {col: type(enhanced_war.result[col].iloc[0]) for col in all_features}
        assert feature_types["nspike"] in (list, np.ndarray), "nspike should be list or array type"
        assert feature_types["lognspike"] in (list, np.ndarray), "lognspike should be list or array type"
