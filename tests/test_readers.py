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
