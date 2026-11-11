"""Serial and Dask analyse the same complete windows.

A recording of n samples with windows of w samples gives floor(n / w) windows, and the samples
after the last of them are not analysed. Before, serial analysed ceil(n / w) windows, the last
one partial with its PSD interpolated, and Dask analysed max(ceil(n / w) - 1, 1), which dropped
a complete window whenever n was an exact multiple of w. IQSEC2 animals 6, 23, 114 and 118 lost
a final window that way on 10 animaldays.
"""

import datetime
import logging

import numpy as np
import pytest

si = pytest.importorskip("spikeinterface.core")

from neurodent import constants  # noqa: E402
from neurodent.analysis import AnimalAnalyzer, LongRecordingAnalyzer  # noqa: E402
from neurodent.loading import AnimalOrganizer, LongRecordingOrganizer  # noqa: E402

FS = 1000
WINDOW_S = 5
WINDOW = WINDOW_S * FS
CHANNELS = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud"]
# Every feature except the spike placeholders, which read_sars_spikes fills in later
FEATURES = [f for f in constants.FEATURES if "nspike" not in f]
START = datetime.datetime(2020, 1, 1, 8, 0, 0)


def _lro(n_samples, n_channels, start, file_durations=None, seed=0):
    """An in-memory recording of noise and a 7 Hz rhythm, dated from ``start``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    data = rng.standard_normal((n_samples, n_channels)) * 20
    data += 40 * np.sin(2 * np.pi * 7 * t)[:, None] * np.linspace(0.5, 1.5, n_channels)
    rec = si.NumpyRecording(data.astype(np.float32), sampling_frequency=float(FS))
    rec = rec.rename_channels(new_channel_ids=CHANNELS[:n_channels])
    lro = LongRecordingOrganizer(item=None, recording=rec)
    file_durations = [float(d) for d in (file_durations or [n_samples / FS])]
    assert sum(file_durations) == pytest.approx(n_samples / FS)
    lro.file_durations = file_durations
    lro.cumulative_file_durations = [float(x) for x in np.cumsum(file_durations)]
    lro.manual_datetimes = start
    lro.datetimes_are_start = True
    lro.finalize_file_timestamps()
    return lro


def _organizer(lengths, n_channels):
    """One animal with one recording per day, of the given lengths in samples.

    The second day's recording is two files, so a file boundary falls inside a window.
    """
    lros = []
    for day, n_samples in enumerate(lengths):
        file_durations = None
        if day == 1:
            first = 1.5 * WINDOW_S
            file_durations = [first, n_samples / FS - first]
        lros.append(
            _lro(n_samples, n_channels, START + datetime.timedelta(days=day), file_durations, seed=day)
        )
    return AnimalOrganizer.from_lros(lros, animal_id="T", genotype="WT")


# Features summed in float32 over the samples. Serial reads each window as the recording
# returns it, column-major here, and Dask reads it back from its zarr cache row-major, so numpy
# adds the samples in a different order and these differ in the last bits (about 1e-6 relative).
# Every other feature is computed in float64 and matches exactly.
FLOAT32_SUMS = {"rms", "ampvar", "logrms", "logampvar"}


def _assert_same(x, y, where, rtol=0.0):
    """Equality of two stored feature values, which may be dicts, tuples or arrays."""
    if isinstance(x, dict):
        assert isinstance(y, dict) and list(x) == list(y), where
        for key in x:
            _assert_same(x[key], y[key], f"{where}[{key}]", rtol)
    elif isinstance(x, tuple):
        assert isinstance(y, tuple) and len(x) == len(y), where
        for i, (a, b) in enumerate(zip(x, y)):
            _assert_same(a, b, f"{where}[{i}]", rtol)
    elif rtol:
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=0, err_msg=where)
    else:
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=where)


def _windowed(ao, mode, apply_notch_filter, features=FEATURES):
    return AnimalAnalyzer(ao).compute_windowed_analysis(
        features,
        window_s=WINDOW_S,
        multiprocess_mode=mode,
        apply_notch_filter=apply_notch_filter,
        chunk_duration_s=2 * WINDOW_S,
    )


@pytest.fixture
def tmpdir_env(tmp_path, monkeypatch):
    """The Dask path writes its window cache under TMPDIR."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("n_channels", [2, 4])
@pytest.mark.parametrize("remainder", [0, 1, WINDOW // 2, WINDOW - 1])
def test_serial_and_dask_give_identical_rows(tmpdir_env, n_channels, remainder):
    """At lengths k·W and k·W + r both modes give k windows, with the same values.

    The notch filter is off, because serial filters each window on its own and Dask filters
    the whole recording, which changes values near the window edges by design. Values match
    exactly except the float32 sums, see FLOAT32_SUMS.
    """
    k = (3, 4)
    ao = _organizer([k[0] * WINDOW + remainder, k[1] * WINDOW + remainder], n_channels)

    serial = _windowed(ao, "serial", apply_notch_filter=False).result
    dask = _windowed(ao, "dask", apply_notch_filter=False).result

    assert len(serial) == len(dask) == sum(k)
    assert list(serial.columns) == list(dask.columns)
    assert serial["animalday"].value_counts().sort_index().tolist() == list(k)
    for column in serial.columns:
        rtol = 1e-5 if column in FLOAT32_SUMS else 0.0
        for row in range(len(serial)):
            _assert_same(serial[column].iloc[row], dask[column].iloc[row], f"{column} row {row}", rtol)


@pytest.mark.parametrize("remainder", [0, 2500])
def test_with_the_notch_filter_both_modes_give_the_same_windows(tmpdir_env, remainder):
    """The window set, times, durations and file ends match whatever the filter does."""
    ao = _organizer([3 * WINDOW + remainder, 2 * WINDOW + remainder], n_channels=3)

    serial = _windowed(ao, "serial", apply_notch_filter=True, features=["rms", "psd"]).result
    dask = _windowed(ao, "dask", apply_notch_filter=True, features=["rms", "psd"]).result

    metadata = ["animalday", "animal", "day", "genotype", "sex", "duration", "endfile", "timestamp", "isday"]
    assert serial[metadata].equals(dask[metadata])
    assert (serial["duration"] == WINDOW_S).all()
    # The second day's first file ends 2.5 s into its second window
    assert serial["endfile"].dropna().tolist() == [2.5]
    expected = [START + datetime.timedelta(seconds=WINDOW_S * i) for i in range(3)]
    expected += [START + datetime.timedelta(days=1, seconds=WINDOW_S * i) for i in range(2)]
    assert serial["timestamp"].tolist() == expected
    # Every window's PSD has the same frequencies; none is interpolated from a partial window
    for result in (serial, dask):
        freqs = [np.asarray(f) for f, _ in result["psd"]]
        assert all(np.array_equal(f, freqs[0]) for f in freqs)


def test_leftover_samples_are_logged(tmpdir_env, caplog):
    ao = _organizer([2 * WINDOW + 1234, 3 * WINDOW], n_channels=2)

    with caplog.at_level(logging.INFO):
        war = _windowed(ao, "serial", apply_notch_filter=False, features=["rms"])

    assert len(war.result) == 5
    assert "The last 1234 samples (1.234 s) are less than a window" in caplog.text
    assert caplog.text.count("are less than a window") == 1


@pytest.mark.parametrize("mode", ["serial", "dask"])
def test_a_recording_shorter_than_a_window_gives_no_rows(tmpdir_env, mode, caplog):
    """Dask used to ask for one full window from it and fail; serial gave one partial row."""
    ao = _organizer([2 * WINDOW, WINDOW - 1, 3 * WINDOW], n_channels=2)
    short_day = ao.long_recordings[1].animalday

    with caplog.at_level(logging.WARNING):
        war = _windowed(ao, mode, apply_notch_filter=False, features=["rms", "psdband"])

    assert len(war.result) == 5
    assert short_day not in set(war.result["animalday"])
    assert f"has {WINDOW - 1} samples, fewer than one {WINDOW_S} s window" in caplog.text


@pytest.mark.parametrize("mode", ["serial", "dask"])
def test_an_animal_with_no_complete_window_raises(tmpdir_env, mode):
    ao = _organizer([WINDOW - 1, WINDOW // 2], n_channels=2)

    with pytest.raises(ValueError, match="No recording of T is as long as one 5 s window"):
        _windowed(ao, mode, apply_notch_filter=False, features=["rms"])


@pytest.mark.parametrize("n_samples, expected", [(3 * WINDOW, 3), (3 * WINDOW + 1, 3), (WINDOW - 1, 0)])
def test_long_recording_analyzer_counts_complete_windows(n_samples, expected):
    lro = _lro(n_samples, 2, START)
    lan = LongRecordingAnalyzer(lro, fragment_len_s=WINDOW_S, apply_notch_filter=False)

    assert lan.n_fragments == expected
    for index in range(expected):
        assert lan.get_fragment_np(index).shape == (WINDOW, 2)
        assert lan.convert_idx_to_timebound(index) == (index * WINDOW_S, (index + 1) * WINDOW_S)
    with pytest.raises(IndexError):
        lan.get_fragment_np(expected)
