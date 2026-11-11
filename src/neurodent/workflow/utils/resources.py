"""Snakemake resources for each rule, from its cluster config block and the inputs it reads."""

import math
import os
import re

from .logging import increment_memory

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

BASE_BYTES = 1_000_000_000
"""Resident bytes a job holds whatever its input: the interpreter, its imports and fixed state.

Measured on the streaming per-WAR rules of the IQSEC2 run of 22 to 23 October 2026: their sacct
peaks sat 0.56 to 0.77 GB above war_stream_bytes, by a least-squares intercept over each rule's
jobs with at least 10,000 windows. A bare WAR load needs about 0.3 GB on top of war_load_bytes.
"""

MEMORY_HEADROOM = 1.25
"""Factor on the computed peak, for what the measurements do not capture."""

TIME_SAFETY = 3
"""Factor on the computed run time. Nodes in this cluster have run the same job at half speed."""

WAR_LOADING = ("whole", "streamed")
"""How a rule's script reads its WAR inputs: in full, or streamed in row groups and batches."""


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


def row_group_stream_bytes(uncompressed_bytes, values, rows, batch_rows):
    """Peak resident bytes while one parquet row group of a WAR is streamed in batches.

    The parquet reader holds the row group's uncompressed data while the streaming engine
    decodes it ``batch_rows`` rows at a time, and a decoded batch costs what loading that many
    rows of the WAR in full would:

        uncompressed_bytes + war_load_bytes(values in one batch, rows in one batch)

    On the IQSEC2 run this reproduced war_standardize, which streams the generated WARs, each
    one row group: iqsec2_80 (5.45 GB uncompressed, 714 million values, 115,806 windows) peaked
    at 7.85 GB against 7.61 GB from this formula, and iqsec2_44 (2322 windows) at 1.12 GB
    against 1.10 GB. The later rules read WARs written in row groups of 5000 rows: they peaked
    at 2.5 to 3.7 GB for WARs of 13,453 to 115,806 windows, against 2.37 to 2.38 GB from this
    formula plus BASE_BYTES and their state per window.

    Args:
        uncompressed_bytes (int): The row group's ``total_byte_size`` from the parquet footer.
        values (int): Stored values in the row group, over every leaf column.
        rows (int): Windows in the row group.
        batch_rows (int): Rows the streaming engine decodes at a time.

    Returns:
        int: Peak resident bytes, not counting BASE_BYTES.
    """
    if rows <= 0:
        return 0
    batch = min(rows, batch_rows)
    return uncompressed_bytes + war_load_bytes(math.ceil(values * batch / rows), batch)


def war_stream_bytes(parquet_path, batch_rows=None):
    """Peak resident bytes while one WAR is streamed through ``scan_parquet_and_json``.

    The largest :func:`row_group_stream_bytes` over the file's row groups, from the footer
    alone. A WAR written in one row group, as war_generation writes them, costs its whole
    uncompressed size; one written by the streaming engine, in row groups of
    ``STREAM_BATCH_ROWS`` rows, costs about one batch however long it is.

    Args:
        parquet_path (str | Path): A WAR's .parquet file.
        batch_rows (int, optional): Rows decoded at a time. Defaults to
            ``neurodent.results.streaming.STREAM_BATCH_ROWS``, what the streaming save uses.

    Returns:
        int: Peak resident bytes, not counting BASE_BYTES.

    Raises:
        FileNotFoundError: If the file does not exist, naming that path, as count_war_values.
    """
    import pyarrow.parquet as pq

    if batch_rows is None:
        from neurodent.results.streaming import STREAM_BATCH_ROWS

        batch_rows = STREAM_BATCH_ROWS
    os.stat(parquet_path)  # raise FileNotFoundError with the path as given
    metadata = pq.ParquetFile(parquet_path).metadata
    peak = 0
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        values = sum(row_group.column(j).num_values for j in range(row_group.num_columns))
        peak = max(
            peak,
            row_group_stream_bytes(row_group.total_byte_size, values, row_group.num_rows, batch_rows),
        )
    return peak


def _war_parquet_paths(input):
    paths = input.war_parquet
    # A single path arrives as a plain string, for example while a checkpoint is pending
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return list(paths)


def _war_input_counts(input):
    counts = [count_war_values(path) for path in _war_parquet_paths(input)]
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


def war_input_mem_mb(block, war_loading):
    """Memory resource for a rule that reads WARs, sized from their parquet footers.

    The rule must name its WAR inputs ``war_parquet``. When the job is scheduled, its peak is
    estimated from the footers of those WARs:

        peak = BASE_BYTES + held + mem_bytes_per_window * all windows

    where ``held`` is what the script holds of the largest input WAR at a time: the whole WAR
    (war_load_bytes) when ``war_loading`` is "whole", or one row group and one decoded batch
    (war_stream_bytes) when it is "streamed". ``mem_bytes_per_window`` is what the script keeps
    or builds per window on top of that, over all its inputs. The job asks for

        mem_mb = max(block mem_mb, MEMORY_HEADROOM * peak / 1e6)

    doubled on each retry, so the block's ``mem_mb`` is the least it ever asks for. Before,
    the aggregate rules added the block's ``mem_mb`` to the computed peak as the base for the
    interpreter, and every other rule asked for its ``mem_mb`` whatever its input.

    Args:
        block (dict): The rule's ``cluster`` config block, with ``mem_mb`` and
            ``mem_bytes_per_window``.
        war_loading (str): "whole" or "streamed", how the rule's script reads each WAR.

    Returns:
        Callable taking ``(wildcards, input, attempt)`` and returning MB, for Snakemake's
        ``resources``. It raises FileNotFoundError while the inputs do not exist yet, which a
        dry run shows as TBD.

    Raises:
        ValueError: If ``war_loading`` is not one of WAR_LOADING.
        KeyError: If the block has no ``mem_bytes_per_window``.
    """
    if war_loading not in WAR_LOADING:
        raise ValueError(f"war_loading must be one of {WAR_LOADING}, got {war_loading!r}")
    floor_mb = block["mem_mb"]
    if "mem_bytes_per_window" not in block:
        raise KeyError(
            "A rule sized from its input WARs needs mem_bytes_per_window in its cluster block "
            f"(this block has {sorted(block)})"
        )
    bytes_per_window = block["mem_bytes_per_window"]

    def mem_mb(wildcards, input, attempt):
        paths = _war_parquet_paths(input)
        counts = [count_war_values(path) for path in paths]
        if war_loading == "whole":
            held = max((war_load_bytes(values, windows) for values, windows in counts), default=0)
        else:
            held = max((war_stream_bytes(path) for path in paths), default=0)
        peak = BASE_BYTES + held + bytes_per_window * sum(windows for _, windows in counts)
        return max(floor_mb, math.ceil(MEMORY_HEADROOM * peak / 1e6)) * 2 ** (attempt - 1)

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


_TIME_RATES = ("seconds_per_value", "seconds_per_window")


def rule_resources(block, war_loading=None):
    """Return a rule's Snakemake resources, from its block of the ``cluster`` config.

    Every rule of the pipeline takes its resources from this one function::

        resources:
            **rule_resources(config["cluster"]["war_standardize"], war_loading="streamed"),

    - ``nodes`` is the block's ``nodes``.
    - ``time`` is the block's ``time``, the same on every attempt. A block that also gives
      ``seconds_per_value`` and ``seconds_per_window`` gets a limit computed from its input
      WARs, with ``time`` as the fixed part (one_war_at_a_time_time).
    - ``mem_mb`` is the block's ``mem_mb``, doubled on each retry, for a rule whose memory does
      not depend on WARs. A rule that reads WARs passes ``war_loading`` and asks for at least
      that, more when its input WARs need it (war_input_mem_mb).

    Before, each rule wrote out its resources from the block itself, and every rule except
    war_zeitgeber and war_relfreq_plots asked for the block's ``mem_mb`` whatever its input.
    A ``--set-resources`` on the command line replaces what this returns for that rule,
    including the doubling on retry.

    Args:
        block (dict): The rule's block of the ``cluster`` config.
        war_loading (str, optional): How the rule's script reads its ``war_parquet`` inputs:
            "whole" if it loads each WAR in full, one at a time, or "streamed" if it streams
            each through ``WindowAnalysisResult.scan_parquet_and_json``. None, the default, for
            a rule whose memory does not depend on WARs. It describes the script, so the rule
            sets it, not the config.

    Returns:
        dict: Keyword arguments for the rule's ``resources`` directive.

    Raises:
        ValueError: If ``war_loading`` is not one of WAR_LOADING, or the block gives rates for
            input-based sizes and the rule reads no WARs.
    """
    if war_loading is None:
        given = [key for key in ("mem_bytes_per_window", *_TIME_RATES) if key in block]
        if given:
            raise ValueError(
                f"The cluster block gives {given}, which size a rule from its input WARs, but "
                "the rule passes no war_loading. Pass war_loading, or drop those keys."
            )
        mem_mb = increment_memory(block["mem_mb"])
    else:
        mem_mb = war_input_mem_mb(block, war_loading)
    if any(key in block for key in _TIME_RATES):
        time = one_war_at_a_time_time(block)
    else:
        time = block["time"]
    return {"time": time, "mem_mb": mem_mb, "nodes": block["nodes"]}
