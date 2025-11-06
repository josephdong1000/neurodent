"""
Comprehensive tests for the unified caching system.
Consolidates all caching-related tests into a single file.
"""

import json
import os
import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import numpy as np

from neurodent import core

try:
    import spikeinterface

    SPIKEINTERFACE_AVAILABLE = True
except ImportError:
    SPIKEINTERFACE_AVAILABLE = False


class TestUnifiedCachingSystem:
    """Comprehensive tests for the unified caching parameter system."""

    def test_cache_policies_basic(self):
        """Test unified cache policy behavior."""
        cache_path = Path("/fake/cache.edf")
        source_paths = [Path("/fake/source.txt")]

        # Test 'force_regenerate' policy
        result = core.should_use_cache_unified(cache_path, source_paths, "force_regenerate")
        assert result is False

        # Test invalid policy
        with pytest.raises(ValueError, match="Invalid cache_policy"):
            core.should_use_cache_unified(cache_path, source_paths, "invalid")

    def test_cache_policies_with_files(self):
        """Test cache policies with actual files."""
        source_paths = [Path("/fake/source.txt")]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_cache:
            cache_path = Path(tmp_cache.name)

            # 'always' should use cache when it exists
            result = core.should_use_cache_unified(cache_path, source_paths, "always")
            assert result is True

        # Cleanup
        cache_path.unlink(missing_ok=True)

        # 'always' should not use cache when it doesn't exist
        cache_path = Path("/nonexistent/cache.edf")
        result = core.should_use_cache_unified(cache_path, source_paths, "always")
        assert result is False

    def test_cache_policies_auto_timestamps(self):
        """Test 'auto' policy with timestamp comparisons."""
        with (
            tempfile.NamedTemporaryFile(delete=False) as tmp_cache,
            tempfile.NamedTemporaryFile(delete=False) as tmp_source,
        ):
            cache_path = Path(tmp_cache.name)
            source_path = Path(tmp_source.name)

            # Make cache newer
            time.sleep(0.1)
            cache_path.touch()

            result = core.should_use_cache_unified(cache_path, [source_path], "auto")
            assert result is True

            # Make source newer
            time.sleep(0.1)
            source_path.touch()

            result = core.should_use_cache_unified(cache_path, [source_path], "auto")
            assert result is False

        # Cleanup
        cache_path.unlink(missing_ok=True)
        source_path.unlink(missing_ok=True)

    def test_organizer_instantiation(self):
        """Test LongRecordingOrganizer with cache_policy parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Test all cache policies
            policies = ["auto", "always", "force_regenerate"]

            for policy in policies:
                organizer = core.LongRecordingOrganizer(
                    tmpdir_path,
                    mode=None,  # Don't auto-load
                    cache_policy=policy,
                )
                assert organizer is not None

    def test_cache_policy_parameter_signature(self):
        """Test that methods accept cache_policy parameter in their signatures."""
        # Test that the method signatures include cache_policy parameter
        import inspect

        # Check LongRecordingOrganizer.__init__ signature
        init_signature = inspect.signature(core.LongRecordingOrganizer.__init__)
        assert "cache_policy" in init_signature.parameters

        # Check convert_file_with_mne_to_recording signature
        mne_signature = inspect.signature(core.LongRecordingOrganizer.convert_file_with_mne_to_recording)
        assert "cache_policy" in mne_signature.parameters

        # Check convert_rowbins_to_rec signature
        rowbins_signature = inspect.signature(core.LongRecordingOrganizer.convert_rowbins_to_rec)
        assert "cache_policy" in rowbins_signature.parameters

        print("✅ All methods have cache_policy in their signatures")

    def test_cache_status_messages(self):
        """Test cache status message generation."""
        cache_path = Path("/fake/cache.edf")

        message_use = core.get_cache_status_message(cache_path, True)
        assert "Using cached intermediate: cache.edf" in message_use

        message_regen = core.get_cache_status_message(cache_path, False)
        assert "Regenerating intermediate: cache.edf" in message_regen

    def test_legacy_function_still_works(self):
        """Test that the original caching function still works for internal use."""
        # The old function should still be available in utils module for internal use
        cache_path = Path("/fake/cache.edf")
        source_paths = [Path("/fake/source.txt")]

        # Access via utils module
        result = core.utils.should_use_cached_file(cache_path, source_paths, "never")
        assert result is False

    def test_backward_compatibility(self):
        """Test that existing code patterns still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with patch("glob.glob", return_value=[]):
                with patch("neurodent.core.core.LongRecordingOrganizer._validate_timestamps_for_mode"):
                    # Should work with new unified parameter
                    organizer = core.LongRecordingOrganizer(tmpdir_path, mode=None, cache_policy="auto")

                    assert organizer is not None


class TestCachingPerformance:
    """Test caching performance benefits."""

    def test_cache_policy_demonstration(self):
        """Demonstrate different cache policies in action."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a fake source file
            source_file = tmpdir_path / "source.txt"
            source_file.write_text("source data")

            # Create cache file path (but don't create the file yet)
            cache_file = tmpdir_path / "cache.bin"

            # Test 1: 'force_regenerate' policy
            result = core.should_use_cache_unified(cache_file, [source_file], "force_regenerate")
            assert result is False

            # Create the cache file
            cache_file.write_text("cached data")

            # Test 2: 'always' policy with existing cache
            result = core.should_use_cache_unified(cache_file, [source_file], "always")
            assert result is True

            # Test 3: 'auto' policy - cache is newer
            time.sleep(0.1)  # Ensure cache is newer
            cache_file.touch()
            result = core.should_use_cache_unified(cache_file, [source_file], "auto")
            assert result is True

            # Test 4: 'auto' policy - source is newer
            time.sleep(0.1)  # Ensure source is newer
            source_file.touch()
            result = core.should_use_cache_unified(cache_file, [source_file], "auto")
            assert result is False


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_invalid_cache_policy(self):
        """Test that invalid cache policies are rejected."""
        cache_path = Path("/fake/cache.edf")
        source_paths = [Path("/fake/source.txt")]

        with pytest.raises(ValueError, match="Invalid cache_policy"):
            core.should_use_cache_unified(cache_path, source_paths, "invalid_policy")

    def test_cache_policy_typing(self):
        """Test that cache_policy parameter accepts correct types."""
        cache_path = Path("/fake/cache.edf")
        source_paths = [Path("/fake/source.txt")]

        # These should all work without errors
        valid_policies = ["auto", "always", "force_regenerate"]
        for policy in valid_policies:
            try:
                core.should_use_cache_unified(cache_path, source_paths, policy)
            except ValueError as e:
                if "Invalid cache_policy" in str(e):
                    pytest.fail(f"Valid policy '{policy}' was rejected")


# Integration test to ensure everything works together
class TestMNECachingOptimization:
    """Test that MNE conversion properly checks cache before loading/resampling."""

    @pytest.fixture
    def mock_extract_func(self):
        """Create a mock extract function that returns a simple MNE Raw object."""

        def extract_func(filepath, **kwargs):
            # Import mne locally to avoid import issues in tests
            import mne

            # Create a simple synthetic Raw object
            n_channels = 4
            sfreq = 1000  # Will be resampled to GLOBAL_SAMPLING_RATE
            n_samples = 10000
            data = np.random.randn(n_channels, n_samples) * 1e-6  # microvolts

            info = mne.create_info(ch_names=[f"ch{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg")

            raw = mne.io.RawArray(data, info)
            return raw

        return extract_func

    @pytest.mark.unit
    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_mne_cache_prevents_loading_when_cache_exists(self, mock_extract_func):
        """Test that when cache exists, extract_func is not called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a test source file first
            source_file = tmpdir_path / "test.edf"
            source_file.touch()

            # Create cache files AFTER source file to ensure cache is newer
            import time

            time.sleep(0.1)  # Small delay to ensure cache is newer

            # Create cache files with correct naming as used in the actual code
            intermediate_name = f"{tmpdir_path.name}_mne-to-rec"
            cache_file = tmpdir_path / f"{intermediate_name}.edf"
            meta_cache_file = cache_file.with_suffix(cache_file.suffix + ".meta.json")

            # Create a fake EDF cache file by writing minimal data
            cache_file.write_text("fake edf cache content")
            metadata_dict = {
                "metadata_path": None,
                "n_channels": 4,
                "f_s": 1000.0,  # Original sampling rate before resampling
                "V_units": None,
                "mult_to_uV": None,
                "precision": None,
                "dt_end": None,
                "channel_names": ["ch0", "ch1", "ch2", "ch3"],
            }
            with open(meta_cache_file, "w") as f:
                json.dump(metadata_dict, f, indent=2)

            # Mock extract_func to track if it gets called
            mock_func = Mock(side_effect=mock_extract_func)

            # Use manual timestamps to avoid validation errors
            manual_dt = datetime(2023, 1, 1, 12, 0, 0)

            # Mock SpikeInterface read_edf to avoid actual file operations
            with (
                patch("neurodent.core.core.se.read_edf") as mock_read_edf,
                patch("neurodent.core.core.DDFBinaryMetadata.from_json") as mock_from_json,
            ):
                # Mock metadata loading from JSON
                mock_metadata = Mock()
                mock_metadata.n_channels = 4
                mock_metadata.f_s = 1000.0
                mock_metadata.channel_names = ["ch0", "ch1", "ch2", "ch3"]
                mock_metadata.dt_end = None
                mock_from_json.return_value = mock_metadata

                # Mock the SpikeInterface read_edf to return something reasonable
                mock_recording = Mock()
                mock_recording.get_num_channels.return_value = 4
                mock_recording.get_sampling_frequency.return_value = 1000  # Should match metadata
                mock_recording.get_duration.return_value = 10.0
                mock_recording.get_channel_ids.return_value = np.array(["ch0", "ch1", "ch2", "ch3"])
                mock_read_edf.return_value = mock_recording

                lro = core.LongRecordingOrganizer(
                    base_folder_path=tmpdir_path,
                    mode="mne",
                    extract_func=mock_func,
                    input_type="file",
                    file_pattern="test.edf",
                    intermediate="edf",
                    cache_policy="always",  # Force use of cache
                    manual_datetimes=manual_dt,
                    datetimes_are_start=True,
                )

            # Assert that extract_func was NOT called because cache was used
            mock_func.assert_not_called()

            # Assert that read_edf was called (meaning we used the cache)
            assert mock_read_edf.called

            # Assert that metadata was properly loaded from cache
            assert lro.meta is not None
            assert lro.meta.n_channels == 4
            assert lro.meta.f_s == 1000.0  # Should be original sampling rate
            assert lro.meta.channel_names == ["ch0", "ch1", "ch2", "ch3"]

            # Assert that from_json was called (meaning we loaded from cache)
            assert mock_from_json.called

    @pytest.mark.unit
    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_missing_metadata_sidecar_falls_back_to_regenerate(self, mock_extract_func):
        """Test that missing metadata sidecar falls back to regenerating both files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a test source file first
            source_file = tmpdir_path / "test.edf"
            source_file.touch()

            import time

            time.sleep(0.1)

            # Create cache files with correct naming as used in the actual code
            intermediate_name = f"{tmpdir_path.name}_mne-to-rec"
            cache_file = tmpdir_path / f"{intermediate_name}.edf"

            # Create ONLY the EDF cache file, NOT the metadata sidecar
            cache_file.write_text("fake edf cache content")
            # Note: No metadata sidecar file created

            # Mock extract_func to track if it gets called
            mock_func = Mock(side_effect=mock_extract_func)

            # Use manual timestamps to avoid validation errors
            manual_dt = datetime(2023, 1, 1, 12, 0, 0)

            # Mock MNE export and SpikeInterface read to avoid actual file operations
            with (
                patch("neurodent.core.core.mne.export.export_raw") as mock_export,
                patch("neurodent.core.core.se.read_edf") as mock_read_edf,
            ):
                # Mock the SpikeInterface read_edf to return something reasonable
                mock_recording = Mock()
                mock_recording.get_num_channels.return_value = 4
                mock_recording.get_sampling_frequency.return_value = 1000
                mock_recording.get_duration.return_value = 10.0
                mock_recording.get_channel_ids.return_value = np.array(["ch0", "ch1", "ch2", "ch3"])
                mock_read_edf.return_value = mock_recording

                lro = core.LongRecordingOrganizer(
                    base_folder_path=tmpdir_path,
                    mode="mne",
                    extract_func=mock_func,
                    input_type="file",
                    file_pattern="test.edf",
                    intermediate="edf",
                    cache_policy="auto",  # 'auto' should regenerate due to missing metadata
                    manual_datetimes=manual_dt,
                    datetimes_are_start=True,
                )

            # Assert that extract_func WAS called because metadata sidecar was missing
            assert mock_func.call_count >= 1

            # Assert that export_raw was called (meaning we regenerated the cache)
            assert mock_export.called

    @pytest.mark.unit
    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_missing_intermediate_file_falls_back_to_regenerate(self, mock_extract_func):
        """Test that missing intermediate file falls back to regenerating both files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a test source file first
            source_file = tmpdir_path / "test.edf"
            source_file.touch()

            import time

            time.sleep(0.1)

            # Create cache files with correct naming as used in the actual code
            intermediate_name = f"{tmpdir_path.name}_mne-to-rec"
            cache_file = tmpdir_path / f"{intermediate_name}.edf"
            meta_cache_file = cache_file.with_suffix(cache_file.suffix + ".meta.json")

            # Create ONLY the metadata sidecar file, NOT the EDF cache
            metadata_dict = {
                "metadata_path": None,
                "n_channels": 4,
                "f_s": 1000.0,
                "V_units": None,
                "mult_to_uV": None,
                "precision": None,
                "dt_end": None,
                "channel_names": ["ch0", "ch1", "ch2", "ch3"],
            }
            with open(meta_cache_file, "w") as f:
                json.dump(metadata_dict, f, indent=2)
            # Note: No intermediate EDF cache file created

            # Mock extract_func to track if it gets called
            mock_func = Mock(side_effect=mock_extract_func)

            # Use manual timestamps to avoid validation errors
            manual_dt = datetime(2023, 1, 1, 12, 0, 0)

            # Mock MNE export and SpikeInterface read to avoid actual file operations
            with (
                patch("neurodent.core.core.mne.export.export_raw") as mock_export,
                patch("neurodent.core.core.se.read_edf") as mock_read_edf,
            ):
                # Mock the SpikeInterface read_edf to return something reasonable
                mock_recording = Mock()
                mock_recording.get_num_channels.return_value = 4
                mock_recording.get_sampling_frequency.return_value = 1000
                mock_recording.get_duration.return_value = 10.0
                mock_recording.get_channel_ids.return_value = np.array(["ch0", "ch1", "ch2", "ch3"])
                mock_read_edf.return_value = mock_recording

                lro = core.LongRecordingOrganizer(
                    base_folder_path=tmpdir_path,
                    mode="mne",
                    extract_func=mock_func,
                    input_type="file",
                    file_pattern="test.edf",
                    intermediate="edf",
                    cache_policy="auto",  # 'auto' should regenerate due to missing intermediate file
                    manual_datetimes=manual_dt,
                    datetimes_are_start=True,
                )

            # Assert that extract_func WAS called because intermediate file was missing
            assert mock_func.call_count >= 1

            # Assert that export_raw was called (meaning we regenerated the cache)
            assert mock_export.called

    @pytest.mark.unit
    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_invalid_cache_policy_raises_error(self, mock_extract_func):
        """Test that invalid cache policy strings raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a test source file
            source_file = tmpdir_path / "test.edf"
            source_file.touch()

            # Mock extract_func
            mock_func = Mock(side_effect=mock_extract_func)

            # Use manual timestamps to avoid validation errors
            manual_dt = datetime(2023, 1, 1, 12, 0, 0)

            # Test invalid cache policy
            with pytest.raises(ValueError, match="Invalid cache_policy: foo"):
                lro = core.LongRecordingOrganizer(
                    base_folder_path=tmpdir_path,
                    mode="mne",
                    extract_func=mock_func,
                    input_type="file",
                    file_pattern="test.edf",
                    intermediate="edf",
                    cache_policy="foo",  # Invalid cache policy
                    manual_datetimes=manual_dt,
                    datetimes_are_start=True,
                )

    @pytest.mark.unit
    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_mne_cache_calls_loading_when_cache_missing(self, mock_extract_func):
        """Test that when cache doesn't exist, extract_func is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a test source file (but no cache file)
            source_file = tmpdir_path / "test.edf"
            source_file.touch()

            # Mock extract_func to track if it gets called
            mock_func = Mock(side_effect=mock_extract_func)

            # Use manual timestamps to avoid validation errors
            manual_dt = datetime(2023, 1, 1, 12, 0, 0)

            # Mock MNE export and SpikeInterface read to avoid actual file operations
            with (
                patch("neurodent.core.core.mne.export.export_raw") as mock_export,
                patch("neurodent.core.core.se.read_edf") as mock_read_edf,
                patch("neurodent.core.core.spre.resample") as mock_resample,
            ):
                # Mock the SpikeInterface read_edf to return something reasonable
                mock_recording = Mock()
                mock_recording.get_num_channels.return_value = 4
                mock_recording.get_sampling_frequency.return_value = 250
                mock_recording.get_duration.return_value = 10.0
                mock_recording.get_channel_ids.return_value = np.array(["ch0", "ch1", "ch2", "ch3"])
                mock_read_edf.return_value = mock_recording

                # Mock resample to return the same mock_recording (simulating no resampling needed)
                mock_resample.return_value = mock_recording

                lro = core.LongRecordingOrganizer(
                    base_folder_path=tmpdir_path,
                    mode="mne",
                    extract_func=mock_func,
                    input_type="file",
                    file_pattern="test.edf",
                    intermediate="edf",
                    cache_policy="auto",  # Will create cache since it doesn't exist
                    manual_datetimes=manual_dt,
                    datetimes_are_start=True,
                )

            # Assert that extract_func WAS called because cache didn't exist
            # Note: function gets called twice - once for metadata creation, once for data processing
            assert mock_func.call_count >= 1

            # Assert that export_raw was called (meaning we created the cache)
            assert mock_export.called

    @pytest.mark.unit
    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_cache_policy_behaviors_comprehensive(self, mock_extract_func):
        """Test all 4 cache policy behaviors comprehensively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a test source file first
            source_file = tmpdir_path / "test.edf"
            source_file.touch()

            import time

            time.sleep(0.1)

            # Create cache files with correct naming
            intermediate_name = f"{tmpdir_path.name}_mne-to-rec"
            cache_file = tmpdir_path / f"{intermediate_name}.edf"
            meta_cache_file = cache_file.with_suffix(cache_file.suffix + ".meta.json")

            # Create both cache files
            cache_file.write_text("fake edf cache content")
            metadata_dict = {
                "metadata_path": None,
                "n_channels": 4,
                "f_s": 1000.0,
                "V_units": None,
                "mult_to_uV": None,
                "precision": None,
                "dt_end": None,
                "channel_names": ["ch0", "ch1", "ch2", "ch3"],
            }
            with open(meta_cache_file, "w") as f:
                json.dump(metadata_dict, f, indent=2)

            # Manual timestamps
            manual_dt = datetime(2023, 1, 1, 12, 0, 0)

            # Test each cache policy behavior
            policies_and_expected_calls = [
                ("always", False),  # Should NOT call extract_func (uses cache)
                ("force_regenerate", True),  # Should call extract_func (ignores cache)
                ("auto", False),  # Should NOT call extract_func (cache newer than source)
            ]

            for cache_policy, should_call_extract in policies_and_expected_calls:
                # Fresh mock for each test
                mock_func = Mock(side_effect=mock_extract_func)

                # Mock MNE export and SpikeInterface read
                with (
                    patch("neurodent.core.core.mne.export.export_raw") as mock_export,
                    patch("neurodent.core.core.se.read_edf") as mock_read_edf,
                    patch("neurodent.core.core.DDFBinaryMetadata.from_json") as mock_from_json,
                ):
                    # Mock metadata loading from JSON
                    mock_metadata = Mock()
                    mock_metadata.n_channels = 4
                    mock_metadata.f_s = 1000.0
                    mock_metadata.channel_names = ["ch0", "ch1", "ch2", "ch3"]
                    mock_metadata.dt_end = None
                    mock_from_json.return_value = mock_metadata

                    # Mock recording
                    mock_recording = Mock()
                    mock_recording.get_num_channels.return_value = 4
                    mock_recording.get_sampling_frequency.return_value = 1000
                    mock_recording.get_duration.return_value = 10.0
                    mock_recording.get_channel_ids.return_value = np.array(["ch0", "ch1", "ch2", "ch3"])
                    mock_read_edf.return_value = mock_recording

                    lro = core.LongRecordingOrganizer(
                        base_folder_path=tmpdir_path,
                        mode="mne",
                        extract_func=mock_func,
                        input_type="file",
                        file_pattern="test.edf",
                        intermediate="edf",
                        cache_policy=cache_policy,
                        manual_datetimes=manual_dt,
                        datetimes_are_start=True,
                    )

                    if should_call_extract:
                        assert mock_func.call_count >= 1, (
                            f"Cache policy '{cache_policy}' should have called extract_func"
                        )
                        assert mock_export.called, f"Cache policy '{cache_policy}' should have called export_raw"
                    else:
                        assert mock_func.call_count == 0, (
                            f"Cache policy '{cache_policy}' should NOT have called extract_func"
                        )
                        assert mock_from_json.called, f"Cache policy '{cache_policy}' should have loaded from cache"

    @pytest.mark.unit
    @pytest.mark.skipif(not SPIKEINTERFACE_AVAILABLE, reason="SpikeInterface not available")
    def test_force_regenerate_behavior(self, mock_extract_func):
        """Test 'force_regenerate' cache policy behavior.

        'force_regenerate' policy should:
        - Always ignore existing cache files
        - Always regenerate files
        - Always save/create new cache files
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a test source file
            source_file = tmpdir_path / "test.edf"
            source_file.touch()

            # Manual timestamps
            manual_dt = datetime(2023, 1, 1, 12, 0, 0)

            # Fresh mock for the test
            mock_func = Mock(side_effect=mock_extract_func)

            # Mock file operations to track behavior
            with (
                patch("neurodent.core.core.mne.export.export_raw") as mock_export,
                patch("neurodent.core.core.se.read_edf") as mock_read_edf,
                patch("builtins.open", mock_open()) as mock_file_open,
            ):
                # Mock recording
                mock_recording = Mock()
                mock_recording.get_num_channels.return_value = 4
                mock_recording.get_sampling_frequency.return_value = 1000
                mock_recording.get_duration.return_value = 10.0
                mock_recording.get_channel_ids.return_value = np.array(["ch0", "ch1", "ch2", "ch3"])
                mock_read_edf.return_value = mock_recording

                lro = core.LongRecordingOrganizer(
                    base_folder_path=tmpdir_path,
                    mode="mne",
                    extract_func=mock_func,
                    input_type="file",
                    file_pattern="test.edf",
                    intermediate="edf",
                    cache_policy="force_regenerate",
                    manual_datetimes=manual_dt,
                    datetimes_are_start=True,
                )

                # Should call extract_func (ignore any existing cache)
                assert mock_func.call_count >= 1, "force_regenerate policy should call extract_func"

                # Should call export_raw (create intermediate files)
                assert mock_export.called, "force_regenerate policy should call export_raw"


class TestSystemIntegration:
    """Integration tests for the complete unified caching system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow with unified caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Test organizer creation with cache policy
            organizer = core.LongRecordingOrganizer(tmpdir_path, mode=None, cache_policy="auto")

            # Test that unified function is accessible
            assert hasattr(core, "should_use_cache_unified")

            # Test function call
            result = core.should_use_cache_unified(
                tmpdir_path / "test.edf", [tmpdir_path / "source.txt"], "force_regenerate"
            )
            assert result is False


if __name__ == "__main__":
    # Run basic tests when executed directly
    test = TestUnifiedCachingSystem()
    test.test_cache_policies_basic()
    print("✅ Basic unified caching tests passed!")
