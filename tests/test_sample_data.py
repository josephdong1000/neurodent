"""Tests for the sample recordings bundled with the package.

These guard the packaging path: a broken ``package-data`` glob or a ``.gitignore``
rule that swallows the ``.bin``/``.csv`` files would otherwise ship a wheel whose
documentation examples fail at import time for every user.
"""

from pathlib import Path

import numpy as np
import pytest

from neurodent.data import sample_dataset, sample_edf, sample_pattern

ANIMALS = ["A10", "F22"]
EXPECTED_CHANNELS = 10
EXPECTED_SAMPLING_RATE = 1000.0
EXPECTED_SECONDS = 60
EXPECTED_EDF_SECONDS = 5


def test_sample_dataset_exists():
    root = sample_dataset()
    assert root.is_dir(), f"bundled sample directory missing: {root}"
    assert {p.name for p in root.iterdir() if p.is_dir()} == set(ANIMALS)


@pytest.mark.parametrize("animal", ANIMALS)
def test_each_animal_ships_a_complete_file_set(animal):
    """A partial set is the failure mode when package-data globs drift."""
    animal_dir = sample_dataset() / animal
    suffixes = {p.suffix for p in animal_dir.iterdir()}
    assert {".bin", ".csv", ".edf"} <= suffixes, f"{animal} missing files: {suffixes}"


@pytest.mark.parametrize("animal", ANIMALS)
def test_bin_file_is_whole_frames_of_expected_length(animal):
    bin_path = next((sample_dataset() / animal).glob("*_ColMajor.bin"))
    size = bin_path.stat().st_size
    bytes_per_frame = np.dtype(np.float32).itemsize * EXPECTED_CHANNELS
    assert size % bytes_per_frame == 0, "truncated mid-frame"
    n_samples = size // bytes_per_frame
    assert n_samples == EXPECTED_SECONDS * EXPECTED_SAMPLING_RATE


def test_sample_pattern_matches_real_files():
    import glob

    for pattern in sample_pattern():
        for animal in ANIMALS:
            matches = glob.glob(pattern.replace("{animal}", animal))
            assert matches, f"pattern matched nothing: {pattern} ({animal})"


@pytest.mark.parametrize("animal", ANIMALS)
def test_sample_edf(animal):
    assert sample_edf(animal).is_file()


@pytest.mark.parametrize("animal", ANIMALS)
def test_edf_length_matches_what_the_docs_claim(animal):
    """The edf is a short single-file example, not the 60 s excerpt the bin holds.

    Documenting one length for both is the drift this guards against.
    """
    mne = pytest.importorskip("mne")
    raw = mne.io.read_raw_edf(sample_edf(animal), preload=False, verbose="ERROR")
    duration = raw.n_times / raw.info["sfreq"]
    assert duration == pytest.approx(EXPECTED_EDF_SECONDS)


def test_sample_edf_rejects_unknown_animal():
    with pytest.raises(FileNotFoundError):
        sample_edf("NOPE")


@pytest.mark.skipif(
    pytest.importorskip("spikeinterface", reason="spikeinterface required") is None,
    reason="spikeinterface required",
)
@pytest.mark.parametrize("animal", ANIMALS)
def test_bundled_data_loads_through_the_documented_reader(animal):
    """End-to-end check of the path the tutorials actually use."""
    from neurodent.loading.discovery import DiscoveredFile
    from neurodent.readers import read_bin_csv_pair

    animal_dir = sample_dataset() / animal
    paths = [
        str(next(animal_dir.glob("*_ColMajor.bin"))),
        str(next(animal_dir.glob("*_Meta.csv"))),
    ]
    rec = read_bin_csv_pair(DiscoveredFile(paths=paths))

    assert rec.get_num_channels() == EXPECTED_CHANNELS
    assert rec.get_sampling_frequency() == EXPECTED_SAMPLING_RATE
    assert rec.get_total_duration() == pytest.approx(EXPECTED_SECONDS)
    traces = rec.get_traces(start_frame=0, end_frame=1000)
    assert np.isfinite(traces).all()
