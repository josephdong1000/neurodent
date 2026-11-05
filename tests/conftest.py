"""
Pytest configuration and common fixtures for NeuRodent tests.
"""

# Set matplotlib backend to non-interactive before any imports
# This prevents Tkinter issues in CI environments (especially Windows)
import matplotlib

matplotlib.use("Agg")

import copy
import tempfile
import warnings
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from neurodent import constants


class ConstantsMutatedWarning(UserWarning):
    """Emitted when a test mutates global constants without the ``mutates_constants`` marker."""


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
def setup_test_environment(request):
    """Save and restore mutable module-level constants around each test.

    Uses :func:`copy.deepcopy` so that *in-place* mutations of nested
    containers (e.g. ``constants.CHANNEL_MAP["LMot"].append(...)``) are
    also properly restored, not just full reassignments.

    If constants are mutated during a test that is **not** decorated with
    ``@pytest.mark.mutates_constants``, a :class:`ConstantsMutatedWarning`
    is emitted.  This makes accidental side-effects from production code
    visible in the test output instead of being silently masked.  Tests
    that intentionally call :func:`~neurodent.workflow.utils.apply_samples_config`
    should carry the marker so the warning is suppressed.
    """
    orig = {
        "GENOTYPE_MAP": copy.deepcopy(constants.GENOTYPE_MAP),
        "SEX_MAP": copy.deepcopy(constants.SEX_MAP),
        "CHANNEL_MAP": copy.deepcopy(constants.CHANNEL_MAP),
        "CHANNEL_ABBREVS": copy.deepcopy(constants.CHANNEL_ABBREVS),
        "CHANNEL_ABBREV_BY_RAW": copy.deepcopy(constants.CHANNEL_ABBREV_BY_RAW),
        "DF_SORT_ORDER": copy.deepcopy(constants.DF_SORT_ORDER),
        "ANIMAL_METADATA": copy.deepcopy(constants.ANIMAL_METADATA),
    }

    yield

    changed = [k for k, v in orig.items() if getattr(constants, k) != v]

    # Restore original values so no test can leak state
    for k, v in orig.items():
        setattr(constants, k, v)

    if changed and not request.node.get_closest_marker("mutates_constants"):
        warnings.warn(
            f"Test mutated global constants {changed} without "
            "@pytest.mark.mutates_constants marker. If intentional, "
            "add the marker; otherwise check for side-effects in the "
            "code under test.",
            ConstantsMutatedWarning,
            stacklevel=1,
        )


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


# ============================================================================
# EXAMPLE DATASET FIXTURE
# ============================================================================
# Provides a minimal synthetic EEG dataset on disk for integration testing
# the Snakemake pipeline without requiring real recordings.
# ============================================================================


@pytest.fixture(scope="session")
def example_dataset(tmp_path_factory):
    """Generate a minimal synthetic NWB dataset for pipeline integration tests.

    Creates a directory tree under ``tmp_path`` with two animals
    (ExWT, ExKO), each with one 5-second NWB recording session. The returned
    dict contains the data root path and a ``samples_config`` dict ready
    to be used with ``apply_samples_config`` or pipeline scripts.

    Returns:
        dict with keys ``data_root``, ``samples_config``, ``animals``,
        ``session_folder``.
    """
    from tests.integration.generate import create_synthetic_dataset

    tmp_path = tmp_path_factory.mktemp("dataset")
    return create_synthetic_dataset(tmp_path)


# ============================================================================
# WAR FILE FIXTURES
# ============================================================================
# Small WARs saved to disk for real, and a tracker that counts how many loaded
# WARs are alive at once. For tests of workflow scripts that load whole WARs.
# ============================================================================

WAR_FILE_CHANNELS = ["LMot", "RMot", "LBar"]
"""Channels of every WAR that create_war_files writes. All are in the default CHANNEL_MAP."""


def _war_file_values(rng, nan_channels, *trailing):
    """Random per-channel values of shape (channels, *trailing), NaN on nan_channels."""
    values = rng.normal(1.0, 0.25, (len(WAR_FILE_CHANNELS), *trailing))
    values[nan_channels] = np.nan
    return values


def _war_file_matrix(rng, nan_channels):
    """A random symmetric channel-by-channel matrix, NaN in the rows and columns of nan_channels."""
    values = rng.uniform(0.0, 1.0, (len(WAR_FILE_CHANNELS), len(WAR_FILE_CHANNELS)))
    values = (values + values.T) / 2
    values[nan_channels, :] = np.nan
    values[:, nan_channels] = np.nan
    return values.tolist()


def _war_file_result(animal_id, genotype, sex, rng):
    """One animal's result frame, with every feature in the form a real WAR stores it."""
    day = "Jan-05-2026"
    freqs = np.arange(1.0, 41.0)
    rows = []
    timestamps = pd.date_range("2026-01-05 06:00", periods=48, freq="30min")
    for index, timestamp in enumerate(timestamps):
        # Every value of window 10 is NaN, and LBar is NaN in windows 20 to 22
        if index == 10:
            nan_channels = list(range(len(WAR_FILE_CHANNELS)))
        elif 20 <= index <= 22:
            nan_channels = [2]
        else:
            nan_channels = []

        row = {
            "animalday": f"{animal_id} {genotype} {day}",
            "animal": animal_id,
            "day": day,
            "genotype": genotype,
            "sex": sex,
            "duration": 5.0,
            "endfile": None,
            "timestamp": timestamp,
            "isday": 6 <= timestamp.hour < 18,
        }
        for feature in constants.LINEAR_FEATURES:
            row[feature] = _war_file_values(rng, nan_channels).tolist()
        for feature in constants.LINEAR_2D_FEATURES:
            row[feature] = _war_file_values(rng, nan_channels, 2).tolist()
        for feature in constants.BAND_FEATURES:
            row[feature] = {
                band: _war_file_values(rng, nan_channels).tolist() for band in constants.BAND_NAMES
            }
        for feature in constants.BANDED_MATRIX_FEATURES:
            row[feature] = {
                band: _war_file_matrix(rng, nan_channels) for band in constants.BAND_NAMES
            }
        for feature in constants.SIMPLE_MATRIX_FEATURES:
            row[feature] = _war_file_matrix(rng, nan_channels)
        row["psd"] = (freqs.tolist(), _war_file_values(rng, nan_channels, len(freqs)).T.tolist())
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def create_war_files(tmp_path):
    """
    Factory fixture that writes small synthetic WARs to disk, one folder per animal.

    Each WAR is saved for real with ``save_parquet_and_json``, so loading it goes through
    the production loader. It has the non-feature columns a real WAR has, and every
    feature in its stored form: per-channel lists for linear features, (channel, component)
    pairs for psdslope, dicts of per-channel lists for band features, channel-by-channel
    matrices on their own or in a dict of bands, and the (frequencies, power) pair for psd.
    There are 48 windows, one every 30 minutes for 24 hours from 06:00. Every value of
    window 10 is NaN, and LBar is NaN in windows 20 to 22, as a fragment filter leaves them.

    Usage:
        def test_something(create_war_files):
            ((parquet_path, json_path),) = create_war_files([("a1", "A1", "WT", "Male")])

    Returns:
        Callable taking a list of ``(folder, animal_id, genotype, sex)`` and returning one
        ``(parquet_path, json_path)`` pair per animal, in the same order.
    """
    from neurodent.results import WindowAnalysisResult

    def _create_war_files(animals):
        paths = []
        for seed, (folder, animal_id, genotype, sex) in enumerate(animals):
            result = _war_file_result(animal_id, genotype, sex, np.random.default_rng(seed))
            war = WindowAnalysisResult(
                result=result,
                animal_id=animal_id,
                genotype=genotype,
                sex=sex,
                channel_names=list(WAR_FILE_CHANNELS),
                bad_channels_dict={animalday: [] for animalday in result["animalday"].unique()},
            )
            folder_path = tmp_path / "wars" / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            war.save_parquet_and_json(folder_path, filename="war")
            paths.append((folder_path / "war.parquet", folder_path / "war.json"))
        return paths

    return _create_war_files


@pytest.fixture
def track_war_loads(monkeypatch):
    """
    Record every WAR load, and how many earlier WARs were still alive when it started.

    Wraps the real ``WindowAnalysisResult.load_parquet_and_json``. It keeps a weak
    reference to each WAR it loads, so at the start of each load it can count the earlier
    WARs still alive. For a script that holds one WAR at a time the count is always 0.
    This checks object lifetimes, not memory, so it is deterministic, and it is what
    catches a missing ``del war`` or a ``tqdm`` wrapper around ``iter_wars``.

    Returns:
        Namespace with ``folders`` (the folder name of each load, in order),
        ``alive_at_load`` (earlier WARs alive at the start of each load), and
        ``real_load`` (the untracked loader, for building expected values).
    """
    from neurodent.results import WindowAnalysisResult

    real_load = WindowAnalysisResult.load_parquet_and_json
    tracker = SimpleNamespace(folders=[], alive_at_load=[], real_load=real_load)
    refs = []

    def tracked_load(folder_path, *args, **kwargs):
        tracker.alive_at_load.append(sum(ref() is not None for ref in refs))
        war = real_load(folder_path, *args, **kwargs)
        tracker.folders.append(Path(folder_path).name)
        refs.append(weakref.ref(war))
        return war

    monkeypatch.setattr(WindowAnalysisResult, "load_parquet_and_json", staticmethod(tracked_load))
    return tracker


