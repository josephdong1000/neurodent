"""Custom reader functions for formats SpikeInterface and MNE do not cover.

Readers here are usable either by import or by the ``extract_func`` string form,
e.g. ``extract_func="neurodent.readers:read_bin_csv_pair"``.
"""

import csv
import os

import numpy as np

__all__ = ["read_bin_csv_pair"]


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

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(
            f"CSV metadata file has no data rows (header-only): {csv_path}"
        )

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
