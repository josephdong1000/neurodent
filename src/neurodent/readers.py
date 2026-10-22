"""Custom reader functions for formats SpikeInterface and MNE do not cover.

Readers here are usable either by import or by the ``extract_func`` string form,
e.g. ``extract_func="neurodent.readers:read_bin_csv_pair"``.
"""

import csv
import os

import numpy as np

__all__ = ["SidecarFormatError", "read_bin_csv_pair"]


class SidecarFormatError(RuntimeError):
    """A sidecar describes its payload in a way this reader would misread.

    Deliberately not a ValueError. The loader treats a ValueError from a file pair as a
    corrupt or empty file, drops it and carries on, which is right for a zero-byte payload
    but would turn a units or layout mismatch into silently lost or misread data. This error
    stops the load instead.
    """


def _single_value(rows, column, csv_path):
    """The one value a column holds on every row, or None when the column is absent."""
    if column not in rows[0]:
        return None
    values = {(row.get(column) or "").strip() for row in rows}
    if len(values) != 1:
        raise SidecarFormatError(
            f"{column} differs between channels in {csv_path}: {sorted(values)}. The reader "
            f"reads one {column} for the whole file."
        )
    return values.pop()


def _check_sidecar(rows, csv_path):
    """Refuse a sidecar whose payload description the reader cannot honour.

    Units and Precision are checked only when the sidecar has those columns; every DataWave
    export measured has them, with the values accepted here.
    """
    missing = [c for c in ("ProbeInfo", "SampleRate") if c not in rows[0]]
    if missing:
        raise SidecarFormatError(f"{csv_path} has no {missing} column(s); the reader needs both.")
    rate = _single_value(rows, "SampleRate", csv_path)
    try:
        float(rate)
    except ValueError:
        raise SidecarFormatError(f"SampleRate {rate!r} in {csv_path} is not a number.") from None
    units = _single_value(rows, "Units", csv_path)
    # The micro sign (U+00B5) is what DataWave writes; the Greek mu (U+03BC) looks the same.
    if units is not None and units not in ("\u00b5V", "\u03bcV", "uV"):
        raise SidecarFormatError(
            f"Units is {units!r} in {csv_path}, but the payload is read as microvolts. Rescaling "
            "is not supported, so this file would load at the wrong scale."
        )
    precision = _single_value(rows, "Precision", csv_path)
    if precision is not None and precision.lower() != "float32":
        raise SidecarFormatError(
            f"Precision is {precision!r} in {csv_path}, but the payload is read as float32."
        )
    if "BinColumn" in rows[0]:
        # Counted from 1 in every export measured; counting from 0 is equally consistent.
        columns = [(row.get("BinColumn") or "").strip() for row in rows]
        in_order = [[str(i) for i in range(first, first + len(rows))] for first in (1, 0)]
        if columns not in in_order:
            raise SidecarFormatError(
                f"BinColumn in {csv_path} is {columns}, not consecutive in row order; the "
                "reader takes row i as payload column i, so channels would be mislabelled."
            )


def read_bin_csv_pair(discovered_file, **kwargs):
    """Read paired ColMajor ``.bin`` + Meta ``.csv`` files into a recording.

    Parameters
    ----------
    discovered_file : neurodent.loading.discovery.DiscoveredFile
        Multi-file discovery result containing one ``.bin`` and one ``.csv``.
    **kwargs
        Forwarded from the pipeline (unused).

    Returns
    -------
    spikeinterface.core.NumpyRecording
        Memory-mapped recording with shape ``(n_samples, n_channels)``.

    Raises
    ------
    ValueError
        For a file pair a load should drop and carry on past: no .bin or no .csv, a
        header-only sidecar, a blank ProbeInfo, an empty payload, or a payload whose size
        does not fit the channel count.
    SidecarFormatError
        For a sidecar the reader would misread, which stops the load: one that is not UTF-8
        text, a missing ProbeInfo or SampleRate column, a SampleRate, Units or Precision that differs between channels, Units
        other than microvolts, Precision other than float32, or a BinColumn that does not count
        up by one in row order.
    """
    import spikeinterface.core as si_core

    bin_paths = [p for p in discovered_file.paths if p.endswith(".bin")]
    csv_paths = [p for p in discovered_file.paths if p.endswith(".csv")]

    if not bin_paths:
        raise ValueError(
            f"No .bin file found in discovered paths: {discovered_file.paths}"
        )
    if not csv_paths:
        raise ValueError(
            f"No .csv file found in discovered paths: {discovered_file.paths}"
        )

    bin_path = bin_paths[0]
    csv_path = csv_paths[0]

    # Read as UTF-8 whatever the platform. The default is cp1252 on Windows, which turns the
    # micro sign of 'µV' into two other characters. A sidecar that is not UTF-8 stops the
    # load: its UnicodeDecodeError is a ValueError, which the loader would take for a corrupt
    # file and drop.
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError as exc:
        raise SidecarFormatError(f"{csv_path} is not UTF-8 text: {exc}") from exc

    if not rows:
        raise ValueError(
            f"CSV metadata file has no data rows (header-only): {csv_path}"
        )

    _check_sidecar(rows, csv_path)
    n_channels = len(rows)
    sampling_rate = float(rows[0]["SampleRate"])

    # Channel identity comes from ProbeInfo, not Label. ProbeInfo is port-qualified
    # ("Intan Input (1)/PortC L Vis Ctx") while Label carries only the region name, so
    # Label is ambiguous whenever one export spans several ports: an IQSEC2 joint
    # recording of four pups repeats each of its eight region names once per port, and
    # under Label every animal's channel_subset silently resolves to the same port.
    # ProbeInfo is unique within every sidecar measured (1878 IQSEC2, 9700 sox5) and
    # never blank where Label is populated, and it is what the package's own metadata
    # reader keys on (recording_metadata.py).
    channel_names = [row["ProbeInfo"] for row in rows]
    if any(not name.strip() for name in channel_names):
        raise ValueError(
            f"ProbeInfo is blank for at least one channel in {csv_path}. "
            f"ProbeInfo is the channel identity, so a blank entry cannot be resolved."
        )

    file_size = os.path.getsize(bin_path)
    if file_size == 0:
        raise ValueError(f"Binary file is empty (0 bytes): {bin_path}")

    bytes_per_frame = np.dtype(np.float32).itemsize * n_channels
    remainder = file_size % bytes_per_frame
    if remainder != 0:
        raise ValueError(
            f"Binary file size ({file_size} bytes) is not divisible by the "
            f"expected frame size ({bytes_per_frame} bytes = "
            f"{np.dtype(np.float32).itemsize} bytes/float32 × {n_channels} channels). "
            f"Remainder: {remainder} bytes. "
            f"This usually means either:\n"
            f"  1. The number of channels in the CSV metadata ({n_channels}) "
            f"does not match the binary — check '{csv_path}'.\n"
            f"  2. The binary file is corrupt or was truncated during "
            f"transfer — re-export or re-copy '{bin_path}'.\n"
            f"  3. The binary uses a different dtype (e.g. float64 or int16) "
            f"instead of float32."
        )
    n_samples = file_size // bytes_per_frame
    # The sox5 format stores data in column-major (Fortran) order: for a
    # (n_samples, n_channels) array, all samples of each channel are
    # contiguous in the file before the next channel begins.  Using
    # order='F' ensures np.memmap interprets the byte layout correctly
    # while keeping the mapping virtual (0 bytes loaded until accessed).
    data = np.memmap(
        bin_path, dtype=np.float32, mode="r", shape=(n_samples, n_channels), order="F"
    )

    return si_core.NumpyRecording(
        traces_list=[data],
        sampling_frequency=sampling_rate,
        channel_ids=channel_names,
    )
