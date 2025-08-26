"""
Pytest configuration and common fixtures for PythonEEG tests.
"""
import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from pythoneeg import constants


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
        50 * np.sin(2 * np.pi * 2 * t) +      # Delta (2 Hz)
        30 * np.sin(2 * np.pi * 6 * t) +      # Theta (6 Hz)
        20 * np.sin(2 * np.pi * 10 * t) +     # Alpha (10 Hz)
        15 * np.sin(2 * np.pi * 20 * t) +     # Beta (20 Hz)
        10 * np.sin(2 * np.pi * 30 * t) +     # Gamma (30 Hz)
        5 * np.random.randn(len(t))           # Noise
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
            50 * np.sin(2 * np.pi * freq_base * t) +
            30 * np.sin(2 * np.pi * (freq_base + 4) * t) +
            20 * np.sin(2 * np.pi * (freq_base + 8) * t) +
            10 * np.random.randn(len(t))
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
        import spikeinterface.preprocessing as spre
        
        # Create a minimal synthetic recording
        duration = 2.0  # seconds
        sampling_frequency = constants.GLOBAL_SAMPLING_RATE
        n_channels = 8
        
        # Generate synthetic data
        recording = si.generate_recording(
            num_channels=n_channels,
            sampling_frequency=sampling_frequency, 
            durations=[duration],
            seed=42
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
    # Set temporary directory for tests
    temp_dir = tempfile.mkdtemp()
    os.environ["TMPDIR"] = temp_dir
    
    yield
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


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