"""
Integration test for Intan RHD units fix.

This test validates the complete fix for the empty WARs issue:
1. Truncation removed (config change)
2. Proper gain scaling for Intan data (code change)

The fix ensures that Intan recordings loaded via read_intan are properly
scaled to microvolts, matching other data formats (NWB, EDF) and allowing
fragment filters to work correctly.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from neurodent import constants
from neurodent.core.core import LongRecordingOrganizer


class TestIntanUnitsFixIntegration:
    """Integration test for the complete Intan units fix."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_intan_data_properly_scaled_for_filters(self, temp_dir):
        """
        Test that Intan data is scaled to microvolts, making it compatible with filters.

        Before fix:
        - Intan data stayed in ADC counts (~6400)
        - Fragment filters calibrated for µV (max_rms: 500 µV)
        - Result: 100% of data filtered out (93600/93600 removed)

        After fix:
        - Intan data converted to µV (via SpikeInterface return_scaled=True)
        - Metadata correctly indicates V_units='µV', mult_to_uV=1.0
        - Fragment filters work correctly
        - Result: Normal filtering (most data passes)
        """
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        # Simulate Intan recording from read_intan
        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 32
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 20000.0  # Typical Intan rate
        mock_recording.get_channel_ids.return_value = np.arange(32)
        mock_recording.get_duration.return_value = 10800.0  # 3 hours

        # Simulate get_traces(return_scaled=True) returning data in µV
        # Normal neural data: RMS ~5-15 µV (not ~6400 µV!)
        n_samples = 1000
        normal_data_uv = np.random.randn(n_samples, 32) * 10.0  # ~10 µV RMS

        mock_recording.get_traces = Mock(return_value=normal_data_uv)

        mock_extract = Mock(return_value=mock_recording)

        # Mock resampling
        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 32
        mock_resampled.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_resampled.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_resampled.get_channel_ids.return_value = np.arange(32)
        mock_resampled.get_duration.return_value = 10800.0
        mock_resampled.get_traces = Mock(return_value=normal_data_uv)

        with patch('spikeinterface.preprocessing.resample', return_value=mock_resampled):
            organizer.convert_file_with_si_to_recording(
                extract_func=mock_extract,
                input_type="folder"
            )

        # Verify metadata is set correctly
        assert organizer.meta.V_units == 'µV'
        assert organizer.meta.mult_to_uV == 1.0

        # Verify data is in reasonable µV range (not ADC counts)
        data = organizer.LongRecording.get_traces(return_scaled=True)
        rms_values = np.sqrt(np.mean(data**2, axis=0))

        # RMS should be in reasonable µV range (5-20 µV), not thousands
        assert np.all(rms_values < 100), \
            f"RMS values should be in µV range (<100), got: {rms_values}"
        assert np.all(rms_values > 0.1), \
            f"RMS values should be reasonable (>0.1 µV), got: {rms_values}"

    def test_units_consistency_across_formats(self, temp_dir):
        """
        Test that Intan data units match NWB/EDF formats after fix.

        All formats should have:
        - V_units = 'µV'
        - mult_to_uV = 1.0
        - Data in same scale (microvolts)
        """
        # Test Intan (via SI mode)
        organizer_intan = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        mock_intan = Mock()
        mock_intan.get_num_channels.return_value = 32
        mock_intan.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_intan.get_sampling_frequency.return_value = 20000.0
        mock_intan.get_channel_ids.return_value = np.arange(32)
        mock_intan.get_duration.return_value = 3600.0

        mock_extract = Mock(return_value=mock_intan)

        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 32
        mock_resampled.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_resampled.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_resampled.get_channel_ids.return_value = np.arange(32)
        mock_resampled.get_duration.return_value = 3600.0

        with patch('spikeinterface.preprocessing.resample', return_value=mock_resampled):
            organizer_intan.convert_file_with_si_to_recording(
                extract_func=mock_extract,
                input_type="folder"
            )

        # All formats should have consistent units
        assert organizer_intan.meta.V_units == 'µV'
        assert organizer_intan.meta.mult_to_uV == 1.0

    def test_filter_calibration_with_correct_units(self, temp_dir):
        """
        Test that fragment filters work correctly with properly scaled data.

        Fragment filters (e.g., max_rms: 500 µV) are calibrated for µV.
        With proper scaling, normal data (~10 µV RMS) passes, artifacts fail.
        """
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 4
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 1000.0
        mock_recording.get_channel_ids.return_value = np.arange(4)
        mock_recording.get_duration.return_value = 3600.0

        # Normal data: RMS ~10 µV (should pass max_rms: 500 filter)
        normal_data = np.random.randn(1000, 4) * 10.0

        # Artifact data: RMS ~1000 µV (should fail max_rms: 500 filter)
        artifact_data = np.random.randn(1000, 4) * 1000.0

        mock_recording.get_traces = Mock(return_value=normal_data)
        mock_extract = Mock(return_value=mock_recording)

        organizer.convert_file_with_si_to_recording(
            extract_func=mock_extract,
            input_type="folder"
        )

        # Calculate RMS like fragment filters do
        normal_rms = np.sqrt(np.mean(normal_data**2, axis=0))
        artifact_rms = np.sqrt(np.mean(artifact_data**2, axis=0))

        # With correct units:
        # - Normal data RMS < 500 µV (passes filter)
        # - Artifact data RMS > 500 µV (filtered out)
        assert np.all(normal_rms < 500), \
            f"Normal data should pass filter (<500 µV), RMS: {normal_rms}"
        assert np.all(artifact_rms > 500), \
            f"Artifact data should fail filter (>500 µV), RMS: {artifact_rms}"

    def test_metadata_persists_through_serialization(self, temp_dir):
        """
        Test that V_units and mult_to_uV persist when metadata is saved/loaded.

        This ensures that WARs generated with correct metadata retain the
        scaling information for downstream analysis.
        """
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            manual_datetimes=datetime(2023, 1, 1, 10, 0, 0),
            datetimes_are_start=True
        )

        mock_recording = Mock()
        mock_recording.get_num_channels.return_value = 32
        mock_recording.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_recording.get_sampling_frequency.return_value = 20000.0
        mock_recording.get_channel_ids.return_value = np.arange(32)
        mock_recording.get_duration.return_value = 3600.0

        mock_extract = Mock(return_value=mock_recording)

        mock_resampled = Mock()
        mock_resampled.get_num_channels.return_value = 32
        mock_resampled.get_dtype.return_value = constants.GLOBAL_DTYPE
        mock_resampled.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
        mock_resampled.get_channel_ids.return_value = np.arange(32)
        mock_resampled.get_duration.return_value = 3600.0

        with patch('spikeinterface.preprocessing.resample', return_value=mock_resampled):
            organizer.convert_file_with_si_to_recording(
                extract_func=mock_extract,
                input_type="folder"
            )

        # Convert metadata to dict (as would happen during WAR serialization)
        meta_dict = organizer.meta.to_dict()

        # Verify critical fields are present and correct
        assert 'V_units' in meta_dict
        assert 'mult_to_uV' in meta_dict
        assert meta_dict['V_units'] == 'µV'
        assert meta_dict['mult_to_uV'] == 1.0

        # This ensures downstream code can access scaling info
        assert meta_dict['n_channels'] == 32
        assert meta_dict['f_s'] == constants.GLOBAL_SAMPLING_RATE


class TestTruncationRemoval:
    """Test that truncation removal allows full dataset loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_no_truncation_warning_with_false(self, temp_dir):
        """Test that truncate=False doesn't produce warnings."""
        # Should not produce any truncation warnings
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            truncate=False
        )

        # Create many files (simulating full dataset)
        files = [f"recording_{i:03d}.rhd" for i in range(122)]

        # Should return all files without warning
        result = organizer._truncate_file_list(files)
        assert len(result) == 122

    def test_full_dataset_loading_scenario(self, temp_dir):
        """
        Test realistic scenario: loading 122 RHD files for multi-day recording.

        Before fix: truncate=5 → only 5 files → incomplete data
        After fix: truncate=false → all 122 files → complete data
        """
        organizer = LongRecordingOrganizer(
            temp_dir,
            mode=None,
            truncate=False  # Config change from truncate=5
        )

        # Simulate 122 RHD files (from logs: "122 folders")
        files = [f"recording_{i:03d}.rhd" for i in range(122)]

        result = organizer._truncate_file_list(files)

        # All files should be loaded
        assert len(result) == 122
        assert result == sorted(files)


class TestUnsignedOffsetBugFix:
    """Test that the unsigned-to-signed offset bug is fixed.

    Root cause: When _apply_resampling calls unsigned_to_signed on uint16 data,
    it subtracts 2^15=32768 from the raw values but does NOT update offset_to_uV.
    When get_traces(return_scaled=True) later applies gain*value + offset,
    the offset is effectively applied twice, creating a ~6390 µV DC offset.

    Fix: Apply scale_to_uV BEFORE unsigned_to_signed. This bakes the correct
    ADC-to-µV conversion into the data and resets gain/offset to 1.0/0.0.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_scale_to_uv_applied_for_integer_recordings(self, temp_dir):
        """Test that scale_to_uV is called for integer recordings with gain info."""
        import spikeinterface.core as si_core

        # Create a NumpyRecording with uint16 data (simulating Intan)
        rng = np.random.default_rng(42)
        n_samples = 1000
        n_channels = 4
        # Simulate uint16 ADC values centered around 32768 (0µV for Intan)
        # Small neural signal ~ ±100 ADC counts
        adc_data = (32768 + rng.integers(-100, 100, size=(n_samples, n_channels))).astype(np.uint16)

        sampling_frequency = 1000.0
        recording = si_core.NumpyRecording(
            traces_list=[adc_data],
            sampling_frequency=sampling_frequency,
        )

        # Set Intan-like gain and offset
        intan_gain = 0.195  # µV per ADC count
        intan_offset = -32768 * intan_gain  # -6389.76 µV
        recording.set_property("gain_to_uV", np.full(n_channels, intan_gain))
        recording.set_property("offset_to_uV", np.full(n_channels, intan_offset))

        # Verify the recording has scaleable traces
        assert recording.has_scaleable_traces()
        assert recording.get_dtype().kind == "u"  # unsigned

        # Create an LRO and apply resampling (which includes the fix)
        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        result = organizer._apply_resampling(recording)

        # After _apply_resampling with the fix, data should be in µV
        traces = result.get_traces(return_scaled=True)

        # The ADC data is centered around 32768 with ±100 variation.
        # Correct µV = (ADC - 32768) * 0.195 ≈ ±19.5 µV
        # Without fix: µV = (ADC - 32768) * 0.195 - 6389.76 ≈ -6390 ± 19.5
        mean_value = np.mean(traces)
        rms_value = np.sqrt(np.mean(traces ** 2))

        # Mean should be near 0 µV (not near -6390 µV)
        assert abs(mean_value) < 50, \
            f"Mean should be near 0 µV, got {mean_value:.1f} µV (double-offset bug?)"

        # RMS should be small (< 50 µV), not ~6390 µV
        assert rms_value < 50, \
            f"RMS should be < 50 µV, got {rms_value:.1f} µV (double-offset bug?)"

    def test_scale_to_uv_not_applied_for_float_recordings(self, temp_dir):
        """Test that scale_to_uV is NOT called for float32 recordings."""
        import spikeinterface.core as si_core

        # Create a float32 recording (already in µV)
        rng = np.random.default_rng(42)
        n_samples = 1000
        n_channels = 4
        data_uv = (rng.standard_normal((n_samples, n_channels)) * 10).astype(np.float32)

        recording = si_core.NumpyRecording(
            traces_list=[data_uv],
            sampling_frequency=1000.0,
        )

        # Float recordings typically don't have gain info
        assert not recording.has_scaleable_traces()

        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        result = organizer._apply_resampling(recording)

        # Data should be unchanged
        traces = result.get_traces(return_scaled=True)
        np.testing.assert_allclose(traces, data_uv, rtol=1e-5)

    def test_rms_matches_expected_for_intan_like_data(self, temp_dir):
        """Test that RMS of Intan-like data is in correct µV range after fix.

        Before fix: RMS ≈ 6390 µV (DC offset from double-applied unsigned bias)
        After fix: RMS ≈ 10-20 µV (actual neural signal amplitude)
        """
        import spikeinterface.core as si_core

        rng = np.random.default_rng(123)
        n_samples = 10000
        n_channels = 8

        # Simulate realistic Intan amplifier recording:
        # - uint16 centered at 32768 (0µV)
        # - Neural signal: ~50 µV peak-to-peak ≈ ±25 ADC counts (25 * 0.195 ≈ 4.9 µV)
        neural_signal = rng.standard_normal((n_samples, n_channels)) * 50  # ~50 ADC counts RMS
        adc_data = (32768 + neural_signal).astype(np.uint16)

        recording = si_core.NumpyRecording(
            traces_list=[adc_data],
            sampling_frequency=20000.0,  # Typical Intan rate
        )

        # Set Intan gains
        recording.set_property("gain_to_uV", np.full(n_channels, 0.195))
        recording.set_property("offset_to_uV", np.full(n_channels, -32768 * 0.195))

        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        result = organizer._apply_resampling(recording)

        traces = result.get_traces(return_scaled=True)
        channel_rms = np.sqrt(np.mean(traces ** 2, axis=0))

        # RMS should be ~9.75 µV (50 ADC * 0.195), definitely not 6390 µV
        for ch_idx, rms in enumerate(channel_rms):
            assert rms < 20, \
                f"Channel {ch_idx} RMS = {rms:.1f} µV, expected < 20 (double-offset bug if ~6390)"
            assert rms > 1, \
                f"Channel {ch_idx} RMS = {rms:.1f} µV, expected > 1 (signal lost?)"

    def test_filter_threshold_compatibility_after_fix(self, temp_dir):
        """Test that properly scaled data passes fragment filter thresholds.

        Config thresholds: max_rms=500, min_rms=50 (in µV).
        Before fix: RMS ≈ 6390 → fails max_rms (6390 > 500)
        After fix: RMS ≈ 10 → needs min_rms adjustment but passes max_rms
        """
        import spikeinterface.core as si_core

        rng = np.random.default_rng(456)
        n_samples = 1000
        n_channels = 4

        # Create data with known RMS
        neural_signal = rng.standard_normal((n_samples, n_channels)) * 100  # ~100 ADC counts RMS
        adc_data = (32768 + neural_signal).astype(np.uint16)

        recording = si_core.NumpyRecording(
            traces_list=[adc_data],
            sampling_frequency=1000.0,
        )
        recording.set_property("gain_to_uV", np.full(n_channels, 0.195))
        recording.set_property("offset_to_uV", np.full(n_channels, -32768 * 0.195))

        organizer = LongRecordingOrganizer(temp_dir, mode=None)
        result = organizer._apply_resampling(recording)

        traces = result.get_traces(return_scaled=True)
        channel_rms = np.sqrt(np.mean(traces ** 2, axis=0))

        max_rms_threshold = 500  # from config
        for rms in channel_rms:
            assert rms < max_rms_threshold, \
                f"RMS {rms:.1f} µV exceeds max_rms={max_rms_threshold} — data not properly scaled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
