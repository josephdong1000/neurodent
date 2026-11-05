"""Snakemake resource sizes computed from the inputs a rule will read."""

import math
import os
import re

LOAD_BYTES_PER_VALUE = 67
"""Peak resident bytes per stored value while one WAR loads in full. See LOAD_BYTES_PER_WINDOW."""

LOAD_BYTES_PER_WINDOW = 18_200
"""Peak resident bytes per window, on top of LOAD_BYTES_PER_VALUE, while one WAR loads in full.

The two are fitted to WARs of two shapes, each loaded in a fresh process. An IQSEC2 WAR with 8
channels stores 6166 values per window: iqsec2_80 peaked at 46.78 GiB for 714 million values
and 115,806 windows. A synthetic WAR with 2 channels stores 277 values per window: it peaked at
4.89 GB for 34.6 million values and 125,000 windows. With a 0.3 GB interpreter the fit
reproduces both. Per value alone the two shapes cost 70 and 141 bytes, because the lists are
shorter, so a factor taken from one dataset would under-size another with fewer channels.

Parquet file size predicts the load worse still. On IQSEC2 the same peaks are 11.7 to 13.6 times
the file size, and a WAR whose windows are mostly NaN stores under 1 byte per value on disk
(iqsec2_52: 0.91) against about 5 to 6 for the others. A stored value is one leaf of the parquet
file, so the count grows with windows, channels and features alike.
"""

MEMORY_HEADROOM = 1.25
"""Factor on the computed peak, for what the measurements do not capture."""

TIME_SAFETY = 3
"""Factor on the computed run time. Nodes in this cluster have run the same job at half speed."""


def count_war_values(parquet_path):
    """Return the stored values and the windows of one WAR parquet file.

    Reads only the parquet footer, so it is cheap even for a WAR of several GiB.

    Args:
        parquet_path (str | Path): A WAR's .parquet file.

    Returns:
        tuple[int, int]: The number of stored values, summed over every leaf column, and the
        number of windows (rows).

    Raises:
        FileNotFoundError: If the file does not exist, naming that path. During a Snakemake dry
            run, before the inputs exist, Snakemake shows the resource as TBD instead of failing.
    """
    import pyarrow.parquet as pq

    os.stat(parquet_path)  # raise FileNotFoundError with the path as given
    metadata = pq.ParquetFile(parquet_path).metadata
    values = sum(
        metadata.row_group(i).column(j).num_values
        for i in range(metadata.num_row_groups)
        for j in range(metadata.num_columns)
    )
    return values, metadata.num_rows


def war_load_bytes(values, windows):
    """Peak resident bytes while one WAR of this many stored values and windows loads in full."""
    return LOAD_BYTES_PER_VALUE * values + LOAD_BYTES_PER_WINDOW * windows


def _war_input_counts(input):
    paths = input.war_parquet
    # A single path arrives as a plain string, for example while a checkpoint is pending
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    counts = [count_war_values(path) for path in paths]
    largest_load = max((war_load_bytes(values, windows) for values, windows in counts), default=0)
    total_values = sum(values for values, _ in counts)
    total_windows = sum(windows for _, windows in counts)
    return largest_load, total_values, total_windows


def _minutes(duration):
    """Minutes in a duration written as the profile accepts it, such as "2h" or "1h30m"."""
    units = {"s": 1 / 60, "m": 1, "h": 60, "d": 24 * 60}
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([smhd])", str(duration).lower())
    if not parts:
        raise ValueError(f"Cannot read a duration from {duration!r}; write it like '2h' or '90m'")
    return sum(float(value) * units[unit] for value, unit in parts)


def one_war_at_a_time_mem_mb(block):
    """Memory resource for a rule that loads its WAR inputs one at a time, in full.

    Such a rule peaks while it holds the largest WAR plus what it keeps for every window of
    every animal, so the size is computed from the inputs when the job is scheduled:

        mem_mb = base + headroom * (largest war_load_bytes over the input WARs
                                    + mem_bytes_per_window * all windows) / 1e6

    and doubled on each retry. The rule must name its WAR inputs ``war_parquet``.

    Args:
        block (dict): The rule's ``cluster`` config block. ``mem_mb`` is the fixed base, for
            the interpreter and its imports. ``mem_bytes_per_window`` is what the rule's script
            keeps and builds per window, measured for that script.

    Returns:
        Callable taking ``(wildcards, input, attempt)`` and returning MB, for Snakemake's
        ``resources``.
    """
    base_mb = block["mem_mb"]
    bytes_per_window = block["mem_bytes_per_window"]

    def mem_mb(wildcards, input, attempt):
        largest_load, _, total_windows = _war_input_counts(input)
        peak = largest_load + bytes_per_window * total_windows
        return (base_mb + math.ceil(MEMORY_HEADROOM * peak / 1e6)) * 2 ** (attempt - 1)

    return mem_mb


def one_war_at_a_time_time(block):
    """Time resource for a rule that loads its WAR inputs one at a time, in full.

    Loading and feature extraction grow with the stored values, and the rest of the script
    (concatenating, plotting, writing) with the windows, so the limit is computed from the
    inputs when the job is scheduled:

        minutes = base + safety * (seconds_per_value * all stored values
                                   + seconds_per_window * all windows) / 60

    The same limit applies to every attempt. The rule must name its WAR inputs ``war_parquet``.

    Args:
        block (dict): The rule's ``cluster`` config block. ``time`` is the fixed base, written
            as the profile accepts it (for example "30m"). ``seconds_per_value`` and
            ``seconds_per_window`` are the rule script's measured rates.

    Returns:
        Callable taking ``(wildcards, input, attempt)`` and returning a duration such as
        "312m", for Snakemake's ``resources``.
    """
    base_minutes = _minutes(block["time"])
    per_value = block["seconds_per_value"]
    per_window = block["seconds_per_window"]

    def time(wildcards, input, attempt):
        _, total_values, total_windows = _war_input_counts(input)
        seconds = per_value * total_values + per_window * total_windows
        return f"{math.ceil(base_minutes + TIME_SAFETY * seconds / 60)}m"

    return time
