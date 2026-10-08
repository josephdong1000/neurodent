"""Tests for how the loader handles a file it cannot read.

Three behaviours are pinned here, because getting any of them wrong is worse than the
crash they replace:

1. By default an unreadable single file still raises. A dataset that silently loses files
   is harder to notice than one that fails.
2. An opted-in skip records what it dropped, so the loss is auditable.
3. A missing or unreadable PATH always raises, opt-in or not. ``OSError`` covers both a
   corrupt payload and a broken mount, and only the first is safe to swallow.

Note which constructor each test uses: ``DiscoveredFile.is_multi_file`` is simply
``paths is not None``, so a one-element ``paths`` tuple still takes the multi-file branch.
``path=`` is what exercises the single-file branch these tests are about.
"""

import numpy as np
import pytest

from neurodent.loading.discovery import DiscoveredFile
from neurodent.loading.long_recording_organizer import LongRecordingOrganizer


def good_reader(path, **kwargs):
    """A reader that returns a real 1-second recording."""
    import spikeinterface as si

    return si.NumpyRecording(
        traces_list=[np.zeros((1000, 2), dtype=np.float32)],
        sampling_frequency=1000.0,
        channel_ids=["A-010", "A-012"],
    )


def corrupt_reader(path, **kwargs):
    """A reader that rejects the payload, as the bin/csv readers do on a zero-byte file."""
    raise ValueError("frame size mismatch: payload is not divisible by the channel count")


@pytest.fixture
def real_file(tmp_path):
    """A path that exists and is readable, so only the reader is at fault."""
    path = tmp_path / "rec.bin"
    path.write_bytes(np.zeros(16, dtype=np.float32).tobytes())
    return path


class TestDefaultIsLoud:
    def test_single_file_reader_failure_raises_by_default(self, real_file):
        with pytest.raises(ValueError, match="frame size mismatch"):
            LongRecordingOrganizer(
                DiscoveredFile(path=str(real_file)),
                mode="si",
                extract_func=corrupt_reader,
                manual_datetimes=None,
            )

    def test_the_flag_is_not_forwarded_to_the_reader(self, real_file):
        """Readers take **kwargs, so an unpopped flag would reach them silently."""
        seen = {}

        def recording_reader(path, **kwargs):
            seen.update(kwargs)
            return good_reader(path)

        LongRecordingOrganizer(
            DiscoveredFile(path=str(real_file)),
            mode="si",
            extract_func=recording_reader,
            skip_unreadable_files=True,
            manual_datetimes=None,
        )
        assert "skip_unreadable_files" not in seen


class TestOptedInSkip:
    def test_skip_produces_a_placeholder_and_a_manifest(self, real_file):
        lro = LongRecordingOrganizer(
            DiscoveredFile(path=str(real_file)),
            mode="si",
            extract_func=corrupt_reader,
            skip_unreadable_files=True,
            manual_datetimes=None,
        )
        assert lro.LongRecording.get_num_samples() == 0
        assert len(lro.skipped_files) == 1
        entry = lro.skipped_files[0]
        assert str(real_file) == entry["path"]
        assert "frame size mismatch" in entry["error"]

    def test_skip_warns_rather_than_dropping_quietly(self, real_file, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            LongRecordingOrganizer(
                DiscoveredFile(path=str(real_file)),
                mode="si",
                extract_func=corrupt_reader,
                skip_unreadable_files=True,
                manual_datetimes=None,
            )
        assert any("extract_func failed" in r.message for r in caplog.records)


class TestEnvironmentErrorsStayLoud:
    """A broken mount or a typo'd pattern must not look like a corrupt recording."""

    def test_stale_nfs_handle_raises_even_when_opted_in(self, real_file):
        """ESTALE is not a FileNotFoundError, and this data lives on NFS."""
        import errno

        def reader(path, **kwargs):
            raise OSError(errno.ESTALE, "Stale file handle", str(path))

        with pytest.raises(OSError):
            LongRecordingOrganizer(
                DiscoveredFile(path=str(real_file)),
                mode="si",
                extract_func=reader,
                skip_unreadable_files=True,
                manual_datetimes=None,
            )

    def test_file_not_found_raises_even_when_opted_in(self, real_file):
        """A FileNotFoundError is re-raised on its type, before any path probing."""

        def reader(path, **kwargs):
            raise FileNotFoundError(path)

        with pytest.raises(FileNotFoundError):
            LongRecordingOrganizer(
                DiscoveredFile(path=str(real_file)),
                mode="si",
                extract_func=reader,
                skip_unreadable_files=True,
                manual_datetimes=None,
            )

    def test_corrupt_payload_on_an_existing_file_is_skippable(self, real_file):
        """The complement: the file is fine, the contents are not, so the skip applies."""
        lro = LongRecordingOrganizer(
            DiscoveredFile(path=str(real_file)),
            mode="si",
            extract_func=corrupt_reader,
            skip_unreadable_files=True,
            manual_datetimes=None,
        )
        assert lro.LongRecording.get_num_samples() == 0


class TestMultiFileBranch:
    """The paired-file branch is guarded without the opt-in, by design.

    The bin/csv readers raise on a zero-byte payload routinely rather than exceptionally,
    and the datasets that ship them carry a per-file datetime, so a dropped file shifts no
    other file's timestamp. Requiring an opt-in there would break sox5 and IQSEC2.
    """

    def test_multi_file_group_skips_without_the_flag(self, real_file):
        sidecar = real_file.with_name("rec_Meta.csv")
        sidecar.write_text("Entity,BinColumn,Label,ProbeInfo,SampleRate\n", encoding="utf-8")
        lro = LongRecordingOrganizer(
            DiscoveredFile(paths=(str(real_file), str(sidecar))),
            mode="si",
            extract_func=corrupt_reader,
            manual_datetimes=None,
        )
        assert lro.LongRecording.get_num_samples() == 0
        assert len(lro.skipped_files) == 2

    def test_multi_file_group_still_raises_on_a_missing_path(self, real_file, tmp_path):
        import errno

        def reader(item, **kwargs):
            raise OSError(errno.ESTALE, "Stale file handle")

        with pytest.raises(OSError, match="Stale file handle"):
            LongRecordingOrganizer(
                DiscoveredFile(paths=(str(real_file), str(tmp_path / "absent.csv"))),
                mode="si",
                extract_func=reader,
                manual_datetimes=None,
            )
