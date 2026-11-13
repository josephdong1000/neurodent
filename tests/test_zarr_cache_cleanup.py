"""The temporary zarr cache of the Dask windowed analysis is removed on every exit.

It used to be removed only when the computation succeeded, so every failed or timed-out
war_generation job left its cache in the temp directory. Caches now carry the SLURM job id in
their name, writers remove a partial store when a write fails, compute_windowed_analysis removes
the cache in a finally block, and the remover refuses any path that is not such a cache.
"""

import os
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurodent.core import utils as core_utils
from neurodent.core.utils import (
    cache_fragments_to_zarr,
    remove_zarr_cache,
    stream_fragments_to_zarr,
    stream_recording_to_zarr,
    zarr_cache_path,
)
from neurodent.core.utils.caching import ZARR_CACHE_PATTERN


def _left_behind(folder):
    return sorted(path.name for path in folder.iterdir())


class _Failure(Exception):
    pass


def test_a_cache_name_carries_the_slurm_job(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "25758031")
    path = zarr_cache_path(tmp_path)

    assert os.path.dirname(path) == str(tmp_path)
    assert re.fullmatch(r"temp_slurm25758031_[0-9a-f]{48}\.zarr", os.path.basename(path))
    assert zarr_cache_path(tmp_path) != path


def test_outside_slurm_a_cache_name_carries_the_process(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    name = os.path.basename(zarr_cache_path(tmp_path))

    assert name.startswith(f"temp_pid{os.getpid()}_")
    assert ZARR_CACHE_PATTERN.fullmatch(name)


def test_a_cache_goes_to_the_temp_directory_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert os.path.dirname(zarr_cache_path()) == str(tmp_path)


def test_the_remover_removes_a_cache_and_accepts_one_already_gone(tmp_path):
    path = stream_fragments_to_zarr(
        lambda i: np.full((10, 2), i, np.float32), 4, (10, 2), np.float32, chunk_size=2,
        tmpdir=str(tmp_path),
    )
    assert _left_behind(tmp_path) == [os.path.basename(path)]

    remove_zarr_cache(path)
    remove_zarr_cache(path)
    assert _left_behind(tmp_path) == []


@pytest.mark.parametrize(
    "name",
    [
        "data.zarr",  # a directory someone else made
        "temp_" + "a" * 48 + ".zarr",  # the old cache names, which carried no job
        "temp_slurm1_" + "a" * 47 + ".zarr",
        "temp_slurm1_" + "a" * 48 + ".zarr.bak",
        "temp_slurmX_" + "a" * 48 + ".zarr",
    ],
)
def test_the_remover_refuses_anything_that_is_not_a_cache(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "keep.txt").write_text("data", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to remove"):
        remove_zarr_cache(folder)
    assert (folder / "keep.txt").exists()


@pytest.mark.parametrize("failure", [_Failure, KeyboardInterrupt, SystemExit])
def test_a_failed_stream_leaves_no_partial_store(tmp_path, failure):
    def get_fragment(index):
        if index == 5:
            raise failure("the recording could not be read")
        return np.zeros((10, 2), np.float32)

    with pytest.raises(failure):
        stream_fragments_to_zarr(get_fragment, 8, (10, 2), np.float32, chunk_size=2, tmpdir=str(tmp_path))
    assert _left_behind(tmp_path) == []


def test_a_failed_recording_stream_leaves_no_partial_store(tmp_path):
    recording = MagicMock()
    recording.get_num_channels.return_value = 2
    recording.get_dtype.return_value = np.float32
    calls = []

    def get_traces(start_frame, end_frame, return_scaled):
        calls.append(start_frame)
        if len(calls) == 2:
            raise _Failure("NFS went away")
        return np.zeros((end_frame - start_frame, 2), np.float32)

    recording.get_traces.side_effect = get_traces

    with pytest.raises(_Failure):
        stream_recording_to_zarr(recording, 6, 10, 2, tmpdir=str(tmp_path))
    assert len(calls) == 2
    assert _left_behind(tmp_path) == []


def test_a_failed_bulk_cache_leaves_no_partial_store(tmp_path, monkeypatch):
    import neurodent.core.utils.caching as caching

    real_open = caching.zarr.open

    class FailingArray:
        def __init__(self, array):
            self.array = array

        def __setitem__(self, key, value):
            raise _Failure("disk full")

    monkeypatch.setattr(caching.zarr, "open", lambda *a, **kw: FailingArray(real_open(*a, **kw)))

    with pytest.raises(_Failure):
        cache_fragments_to_zarr(np.zeros((4, 10, 2), np.float32), 4, tmpdir=str(tmp_path))
    assert _left_behind(tmp_path) == []


def _analyzer(n_fragments=6, n_channels=2, window_s=5, fs=1000):
    """An AnimalAnalyzer over one small real recording, with the metadata step mocked."""
    import spikeinterface.core as si

    from neurodent.analysis import AnimalAnalyzer
    from neurodent.loading import AnimalOrganizer

    rng = np.random.default_rng(0)
    traces = rng.normal(0, 50, (n_fragments * window_s * fs, n_channels)).astype(np.float32)
    lrec = MagicMock()
    lrec.display_name = "rec0"
    lrec.LongRecording = si.NumpyRecording([traces], sampling_frequency=fs)
    # A real count wherever the analyzer may compute or format with one, which a bare
    # MagicMock cannot take: the samples in one window, as the organizer gives them
    lrec.get_fragment_num_samples.side_effect = lambda seconds: int(round(seconds * fs))

    ao = AnimalOrganizer.__new__(AnimalOrganizer)
    ao._validate_sampling_rates = MagicMock()
    ao.long_recordings = []
    ao.animaldays = []
    ao.animal_id, ao.genotype, ao.sex = "A1", "WT", "Male"
    ao.channel_names = ["LMot", "RMot", "LBar", "RBar"][:n_channels]
    ao._iter_valid_recordings = MagicMock(side_effect=lambda: iter([(0, lrec)]))

    lan = MagicMock()
    lan.n_fragments, lan.f_s, lan.apply_notch_filter = n_fragments, fs, False
    az = AnimalAnalyzer(ao)
    az._process_fragment_metadata = MagicMock(
        side_effect=lambda idx, lan, window_s: {
            "animalday": "A1 WT Jan-05-2026", "timestamp": float(idx), "animal": "A1",
            "session": "Jan-05-2026", "genotype": "WT", "sex": "Male",
        }
    )
    return az, lan


@pytest.mark.parametrize("chunk_duration_s", [10, None])
@pytest.mark.parametrize("failure", [None, _Failure, SystemExit])
def test_compute_windowed_analysis_removes_its_cache_on_every_exit(
    tmp_path, monkeypatch, chunk_duration_s, failure
):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    az, lan = _analyzer()
    seen = []

    def compute(*tasks):
        seen.append(_left_behind(tmp_path))  # the cache exists while each batch is computed
        if failure is not None:
            raise failure("a worker died, or SLURM sent SIGTERM")
        return [{"rms": 0.0}] * len(tasks)

    with (
        patch("neurodent.analysis.animal_analyzer.LongRecordingAnalyzer", return_value=lan),
        patch("neurodent.analysis.animal_analyzer.dask.compute", side_effect=compute),
        patch("neurodent.analysis.animal_analyzer.validate_timestamps"),
    ):
        if failure is None:
            az.compute_windowed_analysis(
                ["rms"], multiprocess_mode="dask", chunk_duration_s=chunk_duration_s
            )
        else:
            with pytest.raises(failure):
                az.compute_windowed_analysis(
                    ["rms"], multiprocess_mode="dask", chunk_duration_s=chunk_duration_s
                )

    # One cache, there for every batch, which is one zarr chunk of windows
    assert seen and all(left == seen[0] for left in seen)
    assert len(seen[0]) == 1 and ZARR_CACHE_PATTERN.fullmatch(seen[0][0])
    assert _left_behind(tmp_path) == []


def test_the_core_utils_namespace_exposes_the_cache_helpers():
    assert core_utils.zarr_cache_path is zarr_cache_path
    assert core_utils.remove_zarr_cache is remove_zarr_cache
