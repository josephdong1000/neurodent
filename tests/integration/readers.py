"""Custom reader functions for test datasets.

These readers are referenced by file path in dataset config files
(e.g. ``extract_func: "tests/integration/readers.py:read_bin_csv_pair"``).
They are **not** part of the neurodent package; they exist solely to
support pipeline testing with committed mini recordings.
"""

import csv
import os

import numpy as np


def read_bin_csv_pair(discovered_file, **kwargs):
    """Read paired ColMajor ``.bin`` + Meta ``.csv`` files into a recording.

    Parameters
    ----------
    discovered_file : neurodent.core.discovery.DiscoveredFile
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
    channel_names = [row["Label"] for row in rows]

    file_size = os.path.getsize(bin_path)
    if file_size == 0:
        raise ValueError(f"Binary file is empty (0 bytes): {bin_path}")

    n_samples = file_size // (np.dtype(np.float32).itemsize * n_channels)
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
