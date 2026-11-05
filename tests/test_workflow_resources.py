"""Resource sizes computed from the WARs a rule reads.

The aggregate rules hold one whole WAR at a time plus a frame for every window of every animal.
Their memory and time are computed from the inputs' parquet footers when a job is scheduled,
so the sizes follow each dataset instead of staying fixed to the one they were measured on.
"""

import math
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neurodent.workflow.utils import (
    count_war_values,
    one_war_at_a_time_mem_mb,
    one_war_at_a_time_time,
)
from neurodent.workflow.utils.resources import MEMORY_HEADROOM, TIME_SAFETY, war_load_bytes

TEN_CHANNELS = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]


def _leaves(array):
    """Leaf values in an Arrow array, counted by walking its nesting."""
    if pa.types.is_struct(array.type):
        return sum(_leaves(array.field(i)) for i in range(array.type.num_fields))
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return _leaves(array.flatten())
    return len(array)


def _inputs(paths):
    return SimpleNamespace(war_parquet=[str(parquet_path) for parquet_path, _ in paths])


@pytest.mark.parametrize(
    "values, windows, measured_bytes",
    [
        (714_100_000, 115_806, 49_052_016 * 1024),  # iqsec2_80, 8 channels, fresh process
        (34_625_000, 125_000, 4_776_432 * 1024),  # synthetic, 2 channels, fresh process
    ],
)
def test_load_model_reproduces_both_measured_wars(values, windows, measured_bytes):
    interpreter = 0.3e9
    assert war_load_bytes(values, windows) + interpreter == pytest.approx(measured_bytes, rel=0.02)


def test_count_war_values_counts_every_stored_value_and_window(create_war_files):
    ((parquet_path, _),) = create_war_files(
        [("a1", "A1", "WT", "Male", {"channels": ["LMot", "RMot"], "days": 2})]
    )
    table = pq.read_table(parquet_path)
    stored = sum(_leaves(chunk) for column in table.columns for chunk in column.chunks)

    assert count_war_values(parquet_path) == (stored, 96)


def test_count_grows_with_channels_not_with_how_well_the_file_compresses(create_war_files):
    few, many, dead = create_war_files([
        ("few", "A1", "WT", "Male", {}),
        ("many", "A2", "WT", "Male", {"channels": TEN_CHANNELS}),
        ("dead", "A3", "WT", "Male", {"dead_channel": 0}),
    ])

    assert count_war_values(many[0])[0] > count_war_values(few[0])[0]
    # A channel that is NaN throughout compresses away on disk but is still loaded
    assert count_war_values(dead[0]) == count_war_values(few[0])


def test_memory_follows_the_largest_war_and_every_window(create_war_files):
    paths = create_war_files([
        ("a1", "A1", "WT", "Male", {}),
        ("a2", "A2", "KO", "Female", {"channels": TEN_CHANNELS, "days": 2}),
    ])
    counts = [count_war_values(parquet_path) for parquet_path, _ in paths]
    mem_mb = one_war_at_a_time_mem_mb({"mem_mb": 500, "mem_bytes_per_window": 4000})

    peak = max(war_load_bytes(values, windows) for values, windows in counts) + 4000 * sum(
        windows for _, windows in counts
    )
    expected = 500 + math.ceil(MEMORY_HEADROOM * peak / 1e6)
    assert mem_mb(None, _inputs(paths), 1) == expected
    assert mem_mb(None, _inputs(paths), 2) == 2 * expected


def test_time_follows_every_value_and_window_and_stays_put_on_retry(create_war_files):
    paths = create_war_files([
        ("a1", "A1", "WT", "Male", {}),
        ("a2", "A2", "KO", "Female", {"channels": TEN_CHANNELS, "days": 2}),
    ])
    counts = [count_war_values(parquet_path) for parquet_path, _ in paths]
    time = one_war_at_a_time_time(
        {"time": "30m", "seconds_per_value": 1e-6, "seconds_per_window": 0.01}
    )

    seconds = 1e-6 * sum(values for values, _ in counts) + 0.01 * sum(
        windows for _, windows in counts
    )
    assert time(None, _inputs(paths), 1) == f"{math.ceil(30 + TIME_SAFETY * seconds / 60)}m"
    assert time(None, _inputs(paths), 2) == time(None, _inputs(paths), 1)


@pytest.mark.parametrize(
    "duration, minutes", [("2h", 120), ("90m", 90), ("1h30m", 90), ("1d", 1440), ("45s", 1)]
)
def test_base_time_is_read_in_the_profile_format(create_war_files, duration, minutes):
    paths = create_war_files([("a1", "A1", "WT", "Male", {})])
    time = one_war_at_a_time_time({"time": duration, "seconds_per_value": 0, "seconds_per_window": 0})

    assert time(None, _inputs(paths), 1) == f"{minutes}m"


def test_base_time_without_a_unit_is_refused():
    with pytest.raises(ValueError, match="'2h' or '90m'"):
        one_war_at_a_time_time({"time": "120", "seconds_per_value": 0, "seconds_per_window": 0})


def test_a_single_war_given_as_a_string_is_one_war(create_war_files):
    # Snakemake passes one path as a plain string, for example while a checkpoint is pending
    ((parquet_path, _),) = create_war_files([("a1", "A1", "WT", "Male", {})])
    mem_mb = one_war_at_a_time_mem_mb({"mem_mb": 0, "mem_bytes_per_window": 0})
    values, windows = count_war_values(parquet_path)

    assert mem_mb(None, SimpleNamespace(war_parquet=str(parquet_path)), 1) == math.ceil(
        MEMORY_HEADROOM * war_load_bytes(values, windows) / 1e6
    )


def test_a_missing_war_is_named_so_a_dry_run_shows_tbd(tmp_path):
    # Snakemake shows a resource as TBD, instead of failing, when its function raises
    # FileNotFoundError for a path that is one of the rule's inputs
    missing = str(tmp_path / "a1" / "war.parquet")

    with pytest.raises(FileNotFoundError) as excinfo:
        count_war_values(missing)

    assert excinfo.value.filename == missing
