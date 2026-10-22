"""Tests for ``neurodent.readers.read_bin_csv_pair``, focused on channel identity.

The bundled fixtures carry bare channel ids because a prefix that repeats the port letter
already in the id makes every tutorial and config harder to read. Real DataWave exports do
carry that prefix, and one of them is why the reader keys on ``ProbeInfo`` at all: a joint
recording of four animals repeats each region name once per port, so ``Label`` alone is
ambiguous and silently resolves every animal to the same port. These tests cover the
prefixed form directly, with synthetic sidecars, so dropping it from the fixtures does not
drop it from the suite.
"""

import numpy as np
import pytest

from neurodent.loading.discovery import DiscoveredFile
from neurodent.readers import read_bin_csv_pair

HEADER = "Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit"
STAMP = "2023-12-13T11:17:32"


def write_pair(directory, rows, n_samples=100, rate=1000):
    """Write a ColMajor bin and its Meta csv from ``rows`` of (label, probeinfo)."""
    csv_lines = [HEADER]
    for index, (label, probeinfo) in enumerate(rows, start=1):
        csv_lines.append(
            f"{index},{index},{label},{probeinfo},{rate},µV,float32,{STAMP}"
        )
    csv_path = directory / "rec_Meta.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    bin_path = directory / "rec_ColMajor.bin"
    data = np.arange(len(rows) * n_samples, dtype=np.float32)
    bin_path.write_bytes(data.tobytes())
    return DiscoveredFile(paths=(str(bin_path), str(csv_path)))


class TestChannelIdentity:
    def test_probeinfo_is_the_identity_not_label(self, tmp_path):
        """A port-qualified ProbeInfo wins over a bare Label."""
        item = write_pair(tmp_path, [("L Vis Ctx", "Intan Input (1)/PortC L Vis Ctx")])
        rec = read_bin_csv_pair(item)
        assert [str(c) for c in rec.get_channel_ids()] == ["Intan Input (1)/PortC L Vis Ctx"]

    def test_repeated_labels_across_ports_stay_distinct(self, tmp_path):
        """The defect that motivated keying on ProbeInfo.

        A four-animal joint recording repeats every region name once per port. Under Label
        the four copies collide, ``ids_to_indices`` returns a single index, and each
        animal's channel_subset silently selects the same port's data. ProbeInfo keeps them
        distinct, so a per-animal split addresses the right columns.
        """
        rows = [("L Vis Ctx", f"Intan Input (1)/Port{port} L Vis Ctx") for port in "CABD"]
        rec = read_bin_csv_pair(write_pair(tmp_path, rows))
        ids = [str(c) for c in rec.get_channel_ids()]
        assert len(set(ids)) == 4, "port-qualified names must not collapse"
        assert ids == [f"Intan Input (1)/Port{p} L Vis Ctx" for p in "CABD"]
        # Selecting one port must return exactly that port's single channel.
        selected = rec.select_channels(["Intan Input (1)/PortB L Vis Ctx"])
        assert selected.get_num_channels() == 1

    def test_channel_order_follows_the_sidecar(self, tmp_path):
        """Column order is positional, so identity order must match the csv row order."""
        rows = [(f"C-{amp:03d}", f"Intan Input (1)/PortC C-{amp:03d}") for amp in (15, 9, 21)]
        rec = read_bin_csv_pair(write_pair(tmp_path, rows))
        assert [str(c) for c in rec.get_channel_ids()] == [
            "Intan Input (1)/PortC C-015",
            "Intan Input (1)/PortC C-009",
            "Intan Input (1)/PortC C-021",
        ]

    def test_bare_ids_also_work(self, tmp_path):
        """The shortened form the bundled fixtures use."""
        rec = read_bin_csv_pair(write_pair(tmp_path, [("C-009", "C-009"), ("C-010", "C-010")]))
        assert [str(c) for c in rec.get_channel_ids()] == ["C-009", "C-010"]

    def test_blank_probeinfo_raises(self, tmp_path):
        """A blank identity cannot be resolved, so it must fail loudly.

        Without this the channel would be named the empty string and fail much later, at
        channel resolution, with nothing pointing back at the sidecar.
        """
        item = write_pair(tmp_path, [("C-009", "C-009"), ("C-010", "   ")])
        with pytest.raises(ValueError, match="ProbeInfo is blank"):
            read_bin_csv_pair(item)


class TestGuards:
    def test_zero_byte_bin_raises(self, tmp_path):
        """The loader turns this into a 0-sample placeholder that later stages drop.

        sox5 ships over a thousand zero-byte bins and IQSEC2 several hundred, so this path
        is routine rather than exceptional.
        """
        item = write_pair(tmp_path, [("C-009", "C-009")])
        (tmp_path / "rec_ColMajor.bin").write_bytes(b"")
        with pytest.raises(ValueError):
            read_bin_csv_pair(item)

    def test_size_not_divisible_by_channel_count_raises(self, tmp_path):
        item = write_pair(tmp_path, [("C-009", "C-009"), ("C-010", "C-010")])
        (tmp_path / "rec_ColMajor.bin").write_bytes(np.arange(7, dtype=np.float32).tobytes())
        with pytest.raises(ValueError):
            read_bin_csv_pair(item)

    def test_sampling_rate_comes_from_the_sidecar(self, tmp_path):
        item = write_pair(tmp_path, [("C-009", "C-009")], rate=2000)
        assert read_bin_csv_pair(item).get_sampling_frequency() == 2000.0


def _write_sidecar(directory, lines, n_channels=2, n_samples=100):
    """A pair whose sidecar is exactly ``lines`` (header first) and a fitting float32 payload."""
    csv_path = directory / "rec_Meta.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bin_path = directory / "rec_ColMajor.bin"
    bin_path.write_bytes(np.zeros(n_channels * n_samples, dtype=np.float32).tobytes())
    return DiscoveredFile(paths=(str(bin_path), str(csv_path)))


def _rows(both=None, second=None):
    """Two well-formed sidecar rows, with columns replaced on both rows or on the second only."""
    first = {"Entity": "1", "BinColumn": "1", "Label": "L", "ProbeInfo": "A-010", "SampleRate": "1000",
             "Units": "µV", "Precision": "float32", "LastEdit": STAMP, **(both or {})}
    other = {**first, "Entity": "2", "BinColumn": "2", "ProbeInfo": "A-012", **(second or {})}
    return [",".join(first[c] for c in HEADER.split(",")), ",".join(other[c] for c in HEADER.split(","))]


class TestSidecarsTheReaderWouldMisread:
    """These stop the load with SidecarFormatError rather than a ValueError.

    The loader drops a file pair on ValueError and carries on, which suits an empty payload
    but would turn a units or layout mismatch into silently lost or misread data.
    """

    @pytest.mark.parametrize("rows,message", [
        (dict(both={"Units": "mV"}), "Units is 'mV'"),
        (dict(both={"Precision": "float64"}), "Precision is 'float64'"),
        (dict(second={"SampleRate": "2000"}), "SampleRate differs between channels"),
        (dict(second={"Units": "mV"}), "Units differs between channels"),
        (dict(second={"BinColumn": "3"}), "BinColumn in"),
    ])
    def test_a_mismatch_raises(self, tmp_path, rows, message):
        from neurodent.readers import SidecarFormatError

        item = _write_sidecar(tmp_path, [HEADER, *_rows(**rows)])

        with pytest.raises(SidecarFormatError, match=message):
            read_bin_csv_pair(item)

    def test_a_missing_column_raises_instead_of_a_key_error(self, tmp_path):
        """A KeyError was caught by the loader as a corrupt file and the pair dropped."""
        from neurodent.readers import SidecarFormatError

        item = _write_sidecar(tmp_path, ["Entity,Label,SampleRate", "1,L,1000", "2,R,1000"])

        with pytest.raises(SidecarFormatError, match=r"no \['ProbeInfo'\] column"):
            read_bin_csv_pair(item)

    def test_both_micro_signs_and_plain_uv_are_microvolts(self, tmp_path):
        for units in ("µV", "μV", "uV"):
            folder = tmp_path / str(ord(units[0]))
            folder.mkdir()
            rec = read_bin_csv_pair(_write_sidecar(folder, [HEADER, *_rows(both={"Units": units})]))
            assert rec.get_num_channels() == 2

    def test_the_sidecar_is_read_as_utf8_whatever_the_platform_default(self, tmp_path, monkeypatch):
        """Windows opens text as cp1252 by default, which split the micro sign in two.

        Every Windows CI job failed on that. This stands in for Windows by making the default
        cp1252 for the reader's own open() calls.
        """
        import builtins

        from neurodent import readers

        real_open = builtins.open

        def windows_open(file, mode="r", *args, encoding=None, **kwargs):
            if "b" not in mode and encoding is None:
                encoding = "cp1252"
            return real_open(file, mode, *args, encoding=encoding, **kwargs)

        monkeypatch.setattr(readers, "open", windows_open, raising=False)

        rec = read_bin_csv_pair(_write_sidecar(tmp_path, [HEADER, *_rows()]))

        assert rec.get_num_channels() == 2

    def test_a_sidecar_that_is_not_utf8_stops_the_load(self, tmp_path):
        """Its decode error is a ValueError, which the loader would take for a corrupt file and drop."""
        from pathlib import Path

        from neurodent.readers import SidecarFormatError

        item = _write_sidecar(tmp_path, [HEADER, *_rows()])
        csv_path = Path(item.paths[1])
        csv_path.write_bytes(csv_path.read_text(encoding="utf-8").encode("latin-1"))

        with pytest.raises(SidecarFormatError, match="is not UTF-8 text"):
            read_bin_csv_pair(item)

    def test_the_loader_stops_on_it_instead_of_dropping_the_file(self, tmp_path):
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.readers import SidecarFormatError

        item = _write_sidecar(tmp_path, [HEADER, *_rows(both={"Units": "mV"})])

        with pytest.raises(SidecarFormatError):
            LongRecordingOrganizer(
                item, mode="si", extract_func="neurodent.readers:read_bin_csv_pair",
                manual_datetimes=__import__("datetime").datetime(2020, 1, 1),
            )
