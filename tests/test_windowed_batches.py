"""compute_windowed_analysis gives the same rows whatever its Dask batch size.

The Dask path used to gather every window of a recording in one dask.compute, which hung
war_generation on a 51,000-window day. It now computes and gathers batch_fragments windows at
a time, one zarr chunk by default, through compute_in_batches.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

si = pytest.importorskip("spikeinterface.core")

from distributed import Client, LocalCluster  # noqa: E402

from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.core import utils as core_utils  # noqa: E402
from neurodent.loading import AnimalOrganizer, LongRecordingOrganizer  # noqa: E402

FS = 1000
WINDOW_S = 5
CHANNELS = ["LMot", "RMot", "LBar", "RBar"]
FEATURES = ["rms", "psdband", "psdslope", "cohere", "pcorr"]
START = datetime.datetime(2020, 1, 1, 8, 0, 0)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))  # the window cache goes under TMPDIR
    with (
        LocalCluster(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None) as cluster,
        Client(cluster) as client,
    ):
        yield client


def _organizer(lengths_s):
    """One animal with one recording a day, of noise and a 7 Hz rhythm."""
    lros = []
    for day, seconds in enumerate(lengths_s):
        rng = np.random.default_rng(day)
        n = int(seconds * FS)
        data = rng.standard_normal((n, len(CHANNELS))) * 20
        data += 40 * np.sin(2 * np.pi * 7 * np.arange(n) / FS)[:, None]
        rec = si.NumpyRecording(data.astype(np.float32), sampling_frequency=float(FS))
        lro = LongRecordingOrganizer(item=None, recording=rec.rename_channels(new_channel_ids=CHANNELS))
        lro.file_durations = [float(seconds)]
        lro.cumulative_file_durations = [float(seconds)]
        lro.manual_datetimes = START + datetime.timedelta(days=day)
        lro.datetimes_are_start = True
        lro.finalize_file_timestamps()
        lros.append(lro)
    return AnimalOrganizer.from_lros(lros, animal_id="T", genotype="WT")


def _assert_same(x, y, where):
    """Exact equality of two stored feature values, which may be dicts, tuples or arrays."""
    if isinstance(x, dict):
        assert isinstance(y, dict) and list(x) == list(y), where
        for key in x:
            _assert_same(x[key], y[key], f"{where}[{key}]")
    elif isinstance(x, (tuple, list)) and not np.isscalar(x):
        assert len(x) == len(y), where
        for i, (a, b) in enumerate(zip(x, y)):
            _assert_same(a, b, f"{where}[{i}]")
    else:
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=where)


def _assert_same_frames(a, b):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for column in a.columns:
        for row in range(len(a)):
            _assert_same(a[column].iloc[row], b[column].iloc[row], f"{column} row {row}")


def _dask_rows(ao, **kwargs):
    return AnimalAnalyzer(ao).compute_windowed_analysis(
        FEATURES, window_s=WINDOW_S, multiprocess_mode="dask", chunk_duration_s=4 * WINDOW_S, **kwargs
    ).result


@pytest.mark.parametrize("batch_fragments", [1, 3, 9, 10**6])
def test_any_batch_size_gives_the_rows_of_the_default(client, batch_fragments):
    ao = _organizer([43, 31])
    default = _dask_rows(ao)

    batched = _dask_rows(ao, batch_fragments=batch_fragments)

    assert len(default) == 8 + 6
    _assert_same_frames(batched, default)


def test_the_default_batch_is_one_zarr_chunk(client, monkeypatch):
    sizes = []
    real = core_utils.compute_in_batches

    def recording(tasks, batch_size, **kwargs):
        sizes.append((len(tasks), batch_size, kwargs))
        return real(tasks, batch_size, **kwargs)

    monkeypatch.setattr(core_utils, "compute_in_batches", recording)
    _dask_rows(_organizer([43, 31]), first_batch_timeout_s=120)

    # chunk_duration_s of 4 windows; one call per recording, with its windows
    assert [(n, size) for n, size, _ in sizes] == [(8, 4), (6, 4)]
    assert all(kw == {"first_timeout_s": 120, "label": "window"} for _, _, kw in sizes)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_mini_real_rows_do_not_depend_on_the_batch_size(client, tmp_path, monkeypatch):
    """A real animal through the pipeline's own loading, at batch sizes 1, 2 and a million."""
    from neurodent.core.utils import set_temp_directory
    from neurodent.workflow import apply_samples_config
    from neurodent.workflow.utils import (
        expand_animals_config,
        load_animal_recordings,
        load_dataset_config,
        resolve_samples_config,
    )

    monkeypatch.chdir(Path(__file__).resolve().parents[1])  # dataset paths are relative
    config = load_dataset_config("mini_real")
    samples_config = expand_animals_config(resolve_samples_config(config))
    apply_samples_config(samples_config)
    set_temp_directory(str(tmp_path))
    ao = load_animal_recordings(samples_config, config, [("", "A10", "")], "A10")

    frames = [
        AnimalAnalyzer(ao).compute_windowed_analysis(
            ["rms", "psdband", "cohere"], multiprocess_mode="dask", batch_fragments=size
        ).result
        for size in (1, 2, 10**6)
    ]

    assert len(frames[0]) > 2
    for frame in frames[1:]:
        _assert_same_frames(frame, frames[0])
    pd.testing.assert_series_equal(frames[0]["timestamp"], frames[2]["timestamp"])
