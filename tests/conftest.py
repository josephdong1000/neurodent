"""
Pytest configuration and common fixtures for Neurodent tests.
"""

# Set matplotlib backend to non-interactive before any imports
# This prevents Tkinter issues in CI environments (especially Windows)
import matplotlib

matplotlib.use("Agg")

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from neurodent import constants


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def sample_eeg_data() -> np.ndarray:
    """Generate sample EEG data for testing."""
    # Generate 10 seconds of EEG data at 1000 Hz
    duration = 10  # seconds
    fs = constants.GLOBAL_SAMPLING_RATE
    t = np.linspace(0, duration, duration * fs, endpoint=False)

    # Create realistic EEG signal with multiple frequency components
    signal = (
        50 * np.sin(2 * np.pi * 2 * t)  # Delta (2 Hz)
        + 30 * np.sin(2 * np.pi * 6 * t)  # Theta (6 Hz)
        + 20 * np.sin(2 * np.pi * 10 * t)  # Alpha (10 Hz)
        + 15 * np.sin(2 * np.pi * 20 * t)  # Beta (20 Hz)
        + 10 * np.sin(2 * np.pi * 30 * t)  # Gamma (30 Hz)
        + 5 * np.random.randn(len(t))  # Noise
    )

    return signal.astype(constants.GLOBAL_DTYPE)


@pytest.fixture(scope="session")
def sample_multi_channel_eeg_data() -> np.ndarray:
    """Generate multi-channel EEG data for testing."""
    # Generate 5 seconds of multi-channel EEG data
    duration = 5  # seconds
    fs = constants.GLOBAL_SAMPLING_RATE
    n_channels = 8
    t = np.linspace(0, duration, duration * fs, endpoint=False)

    data = np.zeros((n_channels, len(t)))

    for ch in range(n_channels):
        # Different frequency components for each channel
        freq_base = 2 + ch * 2  # Different base frequency per channel
        data[ch] = (
            50 * np.sin(2 * np.pi * freq_base * t)
            + 30 * np.sin(2 * np.pi * (freq_base + 4) * t)
            + 20 * np.sin(2 * np.pi * (freq_base + 8) * t)
            + 10 * np.random.randn(len(t))
        )

    return data.astype(constants.GLOBAL_DTYPE)


@pytest.fixture(scope="session")
def sample_metadata() -> dict:
    """Sample metadata for testing."""
    return {
        "animal": "A10",
        "genotype": "WT",
        "day": "Jan-01-2023",
        "animalday": "A10 WT Jan-01-2023",
        "sampling_rate": constants.GLOBAL_SAMPLING_RATE,
        "n_channels": 8,
        "duration": 10.0,
    }


@pytest.fixture(scope="session")
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing analysis functions."""
    np.random.seed(42)

    data = {
        "animal": ["A10", "A10", "A11", "A11"] * 3,
        "genotype": ["WT", "WT", "KO", "KO"] * 3,
        "channel": ["LAud", "RAud", "LAud", "RAud"] * 3,
        "band": ["delta", "theta", "alpha"] * 4,
        "rms": np.random.randn(12) * 10 + 50,
        "psdtotal": np.random.randn(12) * 5 + 20,
        "cohere": np.random.randn(12) * 0.1 + 0.5,
        "isday": [True, False] * 6,
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def mock_spikeinterface():
    """Mock SpikeInterface objects for testing."""
    mock_recording = Mock()
    mock_recording.get_num_channels.return_value = 8
    mock_recording.get_sampling_frequency.return_value = constants.GLOBAL_SAMPLING_RATE
    mock_recording.get_num_samples.return_value = 10000

    mock_sorting = Mock()
    mock_sorting.get_unit_ids.return_value = [1, 2, 3]
    mock_sorting.get_unit_spike_train.return_value = np.array([100, 200, 300, 400, 500])

    return {
        "recording": mock_recording,
        "sorting": mock_sorting,
    }


@pytest.fixture(scope="session")
def real_spikeinterface_recording():
    """Create a real SpikeInterface recording for proper integration testing."""
    try:
        import spikeinterface as si

        # Create a minimal synthetic recording
        duration = 2.0  # seconds
        sampling_frequency = constants.GLOBAL_SAMPLING_RATE
        n_channels = 8

        # Generate synthetic data
        recording = si.generate_recording(
            num_channels=n_channels, sampling_frequency=sampling_frequency, durations=[duration], seed=42
        )

        return recording

    except ImportError:
        # Fallback if SpikeInterface not available
        return None


@pytest.fixture(scope="session")
def mock_mne():
    """Mock MNE objects for testing."""
    mock_raw = Mock()
    mock_raw.info = Mock()
    mock_raw.info["sfreq"] = constants.GLOBAL_SAMPLING_RATE
    mock_raw.info["nchan"] = 8
    mock_raw.n_times = 10000

    return {
        "raw": mock_raw,
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    yield


@pytest.fixture(scope="session")
def test_file_paths(temp_dir) -> dict:
    """Create test file paths for various file types."""
    paths = {
        "ddf_col": temp_dir / "test_ColMajor_001.bin",
        "ddf_row": temp_dir / "test_RowMajor_001.bin",
        "ddf_meta": temp_dir / "test_Meta_001.json",
        "edf": temp_dir / "test_data.edf",
        "npy": temp_dir / "test_data.npy",
        "csv": temp_dir / "test_data.csv",
    }

    # Create some dummy files
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npy":
            np.save(path, np.random.randn(100, 8))
        elif path.suffix == ".csv":
            pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}).to_csv(path, index=False)
        else:
            path.touch()

    return paths


# ============================================================================
# MOCK FACTORIES - Reusable Mock Creation Patterns
# ============================================================================
# These factory functions reduce duplication across test files by providing
# standardized mock objects with sensible defaults that can be customized.
# Follows pytest best practices for organizing mocks in a central location.
# ============================================================================


@pytest.fixture
def mock_long_recording_organizer():
    """
    Factory fixture for creating LongRecordingOrganizer mocks.

    Returns a callable that creates configured mock LRO instances with
    customizable parameters. This reduces duplication across tests that
    need to mock LongRecordingOrganizer behavior.

    Usage:
        def test_something(mock_long_recording_organizer):
            mock_lro = mock_long_recording_organizer(
                channel_names=['LMot', 'RMot'],
                lof_scores=np.array([1.5, 2.0])
            )
            # Use mock_lro in your test...

    Returns:
        Callable that returns configured Mock objects
    """

    def _create_mock_lro(
        channel_names=None,
        lof_scores=None,
        duration=100.0,
        sampling_rate=1000.0,
        n_channels=None,
        file_end_datetimes=None,
        meta_dict=None,
    ):
        """Create a mock LongRecordingOrganizer with specified parameters."""
        from datetime import datetime

        channel_names = channel_names or ["ch1", "ch2"]
        n_channels = n_channels or len(channel_names)

        mock_lro = Mock()
        mock_lro.channel_names = channel_names
        mock_lro.lof_scores = lof_scores if lof_scores is not None else np.array([1.0] * n_channels)

        # Mock the recording object
        mock_recording = Mock()
        mock_recording.get_duration.return_value = duration
        mock_recording.get_num_samples.return_value = int(duration * sampling_rate)
        mock_recording.get_sampling_frequency.return_value = sampling_rate
        mock_recording.get_num_channels.return_value = n_channels
        mock_lro.LongRecording = mock_recording

        # Mock metadata
        if meta_dict:
            mock_lro.meta = meta_dict
        else:
            mock_lro.meta = Mock(f_s=sampling_rate, n_channels=n_channels)

        # Mock file_end_datetimes
        if file_end_datetimes:
            mock_lro.file_end_datetimes = file_end_datetimes
        else:
            mock_lro.file_end_datetimes = [datetime(2023, 1, 1, 12, 0, 0)]

        # Mock cleanup method
        mock_lro.cleanup_rec = Mock()

        return mock_lro

    return _create_mock_lro


@pytest.fixture
def create_test_directory_structure():
    """
    Factory fixture for creating test directory structures.

    This reduces duplication in tests that need to create temporary
    directories with specific layouts (e.g., for AnimalOrganizer tests).

    Usage:
        def test_something(create_test_directory_structure):
            base_path, dirs = create_test_directory_structure(
                mode="concat",
                animal_id="A123"
            )
            # Use the created structure...

    Returns:
        Callable that creates directory structures and returns paths
    """
    import tempfile
    import shutil
    from pathlib import Path

    created_temps = []

    def _create_structure(mode="concat", animal_id="A123", genotype="WT", include_files=True, base_path=None):
        """
        Create a test directory structure.

        Args:
            mode: Directory mode ("concat", "nest", "noday", "base")
            animal_id: Animal identifier
            genotype: Animal genotype
            include_files: Whether to include test files (for filtering tests)
            base_path: Optional base path (creates temp dir if None)

        Returns:
            tuple: (base_path, dict with 'directories' and 'files' lists)
        """
        if base_path is None:
            temp_dir = tempfile.mkdtemp()
            created_temps.append(temp_dir)
            base_path = Path(temp_dir)
        else:
            base_path = Path(base_path)

        result = {"directories": [], "files": []}

        if mode == "nest":
            # Create animal subfolder structure
            animal_dir = base_path / f"{genotype}_{animal_id}_data"
            animal_dir.mkdir(exist_ok=True)

            day1_dir = animal_dir / f"{genotype}_2023-01-01"
            day2_dir = animal_dir / f"{genotype}_2023-01-02"
            day1_dir.mkdir(exist_ok=True)
            day2_dir.mkdir(exist_ok=True)
            result["directories"].extend([day1_dir, day2_dir])

            if include_files:
                edf_file = animal_dir / "recording.edf"
                bin_file = animal_dir / "data.bin"
                edf_file.touch()
                bin_file.touch()
                result["files"].extend([edf_file, bin_file])

        elif mode in ["concat", "noday"]:
            day1_dir = base_path / f"{genotype}_{animal_id}_2023-01-01"
            day2_dir = base_path / f"{genotype}_{animal_id}_2023-01-02"
            day1_dir.mkdir(exist_ok=True)

            if mode == "concat":
                day2_dir.mkdir(exist_ok=True)
                result["directories"].extend([day1_dir, day2_dir])
            else:
                result["directories"].append(day1_dir)

            if include_files:
                for ext in ["edf", "bin", "json"]:
                    f = base_path / f"{genotype}_{animal_id}_recording.{ext}"
                    f.touch()
                    result["files"].append(f)

        elif mode == "base":
            if include_files:
                for ext in ["edf", "bin"]:
                    f = base_path / f"recording.{ext}"
                    f.touch()
                    result["files"].append(f)

        return base_path, result

    yield _create_structure

    # Cleanup
    for temp_dir in created_temps:
        shutil.rmtree(temp_dir, ignore_errors=True)
