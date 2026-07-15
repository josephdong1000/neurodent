"""Cache self-healing must distinguish a corrupt FILE from a bug in our code.

`_get_or_create_intermediate_file` deletes the cache and regenerates when it cannot read it. Catching
broadly there means any bug in the read path -- a bad unit scale, a metadata schema change -- gets
laundered into "the cache was corrupt", which deletes good data and hides the fault permanently: the
regenerated file hits the same bug next run.
"""
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("spikeinterface")

from neurodent.loading import lro_loading
from neurodent.loading.long_recording_organizer import LongRecordingOrganizer
from neurodent.loading.recording_metadata import RecordingMetadata

FS, SECS, N_CH = 250, 10, 4


@pytest.fixture
def source_fif(tmp_path):
    t = np.arange(SECS * FS) / FS
    data_v = np.stack([100e-6 * np.sin(2 * np.pi * 7 * t)] * N_CH)
    info = mne.create_info([f"ch{i}" for i in range(N_CH)], FS, ch_types="eeg")
    path = tmp_path / "src_raw.fif"
    mne.io.RawArray(data_v, info, verbose=False).save(path, overwrite=True, verbose=False)
    return path


def _load(src, cache_dir, cache_policy="auto"):
    return LongRecordingOrganizer(
        src, mode="mne", extract_func=mne.io.read_raw_fif, intermediate="bin",
        intermediate_dir=str(cache_dir), manual_datetimes=datetime(2024, 1, 1),
        cache_policy=cache_policy,
    )


def _cache_files(cache_dir):
    binf = next(Path(cache_dir).glob("*.bin"))
    return binf, binf.with_suffix(binf.suffix + ".meta.json")


def test_truncated_cache_is_detected_and_regenerated(source_fif, tmp_path):
    """read_binary is memmap-backed: it returns a SHORTER recording rather than raising, so a
    truncated cache cannot be caught by a try and has to be measured."""
    cache = tmp_path / "c"
    lro = _load(source_fif, cache)
    n_full = lro.LongRecording.get_num_frames()
    # SpikeInterface's binary reader keeps a file handle open on the cache. On Windows an open
    # handle blocks delete/replace, so a live recording would make the self-healing reload fail
    # with WinError 5/32. Self-heal models a fresh load whose handle is long closed; drop it here.
    del lro
    gc.collect()

    binf, _ = _cache_files(cache)
    intact = binf.read_bytes()
    binf.write_bytes(intact[: len(intact) // 2 - 3])       # truncate, mid-frame

    lro2 = _load(source_fif, cache)                         # must self-heal, not silently shorten
    assert lro2.LongRecording.get_num_frames() == n_full


def test_empty_cache_is_detected_and_regenerated(source_fif, tmp_path):
    cache = tmp_path / "c"
    lro = _load(source_fif, cache)
    n_full = lro.LongRecording.get_num_frames()
    # See truncated-cache test: release the open read handle so the Windows self-heal can
    # delete/replace the cache file instead of failing with WinError 5/32.
    del lro
    gc.collect()

    binf, _ = _cache_files(cache)
    binf.write_bytes(b"")

    lro2 = _load(source_fif, cache)
    assert lro2.LongRecording.get_num_frames() == n_full


def test_corrupt_metadata_is_regenerated(source_fif, tmp_path):
    cache = tmp_path / "c"
    _load(source_fif, cache)
    _, meta = _cache_files(cache)
    meta.write_text("{not json")

    lro = _load(source_fif, cache)                          # JSONDecodeError -> regenerate
    assert lro.LongRecording.get_num_frames() > 0


def test_stale_schema_metadata_is_regenerated(source_fif, tmp_path):
    cache = tmp_path / "c"
    _load(source_fif, cache)
    _, meta = _cache_files(cache)
    meta.write_text(json.dumps({"unexpected_old_field": 1}))

    lro = _load(source_fif, cache)                          # KeyError -> regenerate
    assert lro.LongRecording.get_num_frames() > 0


def test_a_bug_inside_the_read_is_not_laundered_into_a_corrupt_cache(source_fif, tmp_path, monkeypatch):
    """The one that matters. A TypeError from INSIDE the guarded read is a bug in our code, not a
    corrupt file. It must propagate, rather than delete the cache and regenerate -- the regenerated
    file would hit the same bug next run, so the fault would never surface.

    Patched at the reader itself so the fault is raised inside the try, which is what pins the
    exception types down. A fault raised before the try would propagate regardless and prove nothing.
    """
    cache = tmp_path / "c"
    _load(source_fif, cache)                                # populate a good cache
    binf, meta = _cache_files(cache)

    def boom(*a, **k):
        raise TypeError("a bug in our code, not a corrupt file")

    monkeypatch.setattr(lro_loading.se, "read_binary", boom)

    with pytest.raises(TypeError, match="a bug in our code"):
        _load(source_fif, cache)

    assert binf.exists(), "a code bug caused the cache file to be deleted"
    assert meta.exists(), "a code bug caused the metadata sidecar to be deleted"


def test_the_unit_scale_is_computed_outside_the_corruption_handler(source_fif, tmp_path, monkeypatch):
    """_gain_to_uV runs before the try, so a unit-conversion fault cannot be mistaken for a corrupt
    cache and delete good files."""
    cache = tmp_path / "c"
    _load(source_fif, cache)
    binf, meta = _cache_files(cache)

    def boom(*a, **k):
        raise TypeError("mult_to_uV came back as a string")

    monkeypatch.setattr(lro_loading, "_gain_to_uV", boom)

    with pytest.raises(TypeError, match="mult_to_uV"):
        _load(source_fif, cache)
    assert binf.exists() and meta.exists()


def test_cache_policy_always_still_raises_on_a_corrupt_file(source_fif, tmp_path):
    cache = tmp_path / "c"
    _load(source_fif, cache)
    binf, _ = _cache_files(cache)
    binf.write_bytes(b"")

    with pytest.raises(OSError):
        _load(source_fif, cache, cache_policy="always")


def test_bin_intact_check():
    meta = RecordingMetadata(None, n_channels=4, f_s=250.0, dt_end=None, channel_names=list("abcd"))

    class FakePath:
        def __init__(self, size):
            self._size = size

        def stat(self):
            return type("S", (), {"st_size": self._size})()

    assert lro_loading._bin_is_intact(FakePath(4 * 8 * 10), meta)      # 10 whole frames
    assert not lro_loading._bin_is_intact(FakePath(0), meta)           # empty
    assert not lro_loading._bin_is_intact(FakePath(4 * 8 * 10 - 3), meta)   # truncated mid-frame
