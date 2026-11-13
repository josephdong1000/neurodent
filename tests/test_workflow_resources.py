"""Resource sizes computed from the WARs a rule reads.

Every rule takes its resources from rule_resources. A rule that reads WARs asks for at least its
block's mem_mb, and more when the parquet footers of its input WARs say it will need it: the
whole WAR for a script that loads it, one row group and one batch for a script that streams it,
plus what the script keeps per window. The sizes follow each dataset instead of staying fixed
to the one they were measured on.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from neurodent.workflow.utils import (
    count_war_values,
    one_war_at_a_time_time,
    rule_resources,
    war_input_mem_mb,
    war_stream_bytes,
)
from neurodent.workflow.utils.resources import (
    BASE_BYTES,
    MEMORY_HEADROOM,
    TIME_SAFETY,
    row_group_stream_bytes,
    war_load_bytes,
)

TEN_CHANNELS = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]
CLUSTER = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "config" / "config.yaml").read_text(encoding="utf-8")
)["cluster"]
MIB = 2**20  # sacct --units=M reports MiB


def _leaves(array):
    """Leaf values in an Arrow array, counted by walking its nesting."""
    if pa.types.is_struct(array.type):
        return sum(_leaves(array.field(i)) for i in range(array.type.num_fields))
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return _leaves(array.flatten())
    return len(array)


def _inputs(paths):
    return SimpleNamespace(war_parquet=[str(parquet_path) for parquet_path, _ in paths])


def _block(**keys):
    return {"time": "1h", "mem_mb": 0, "nodes": 1, "threads": 1, "retries": 0, **keys}


def _expected_mb(floor_mb, held, bytes_per_window, windows):
    peak = BASE_BYTES + held + bytes_per_window * windows
    return max(floor_mb, math.ceil(MEMORY_HEADROOM * peak / 1e6))


def _streamed_copy(parquet_path, folder, batch_rows):
    """The WAR saved again by the streaming engine, in row groups of batch_rows rows."""
    from neurodent.results import WindowAnalysisResult

    war = WindowAnalysisResult.scan_parquet_and_json(Path(parquet_path).parent, filename="war")
    war.save_parquet_and_json(folder, filename="war", batch_size=batch_rows)
    return folder / "war.parquet"


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


# Jobs of the IQSEC2 run of 23 October 2026: sacct MaxRSS against the parquet footer of the job's
# input WAR. For each block, the job with the largest peak and the job that sets its per-window
# rate in config.yaml. A streamed input is given by its largest row group.
MEASURED_JOBS = [
    # block, job, peak MiB, values, windows, largest row group (bytes, values, rows) or None
    ("diagnostic_figures", "filtered iqsec2_83", 46_566.62, 714_059_796, 115_806, None),
    ("diagnostic_figures", "unfiltered iqsec2_83", 45_757.85, 714_059_796, 115_806, None),
    ("war_standardize", "iqsec2_80", 7_487.05, 714_059_796, 115_806, (5_448_742_580, 714_059_796, 115_806)),
    ("war_standardize", "iqsec2_106", 5_101.56, 319_676_270, 51_845, (2_435_683_503, 319_676_270, 51_845)),
    ("war_fragment_filter", "iqsec2_80", 3_487.47, 714_059_796, 115_806, (224_077_466, 30_830_000, 5_000)),
    ("war_fragment_filter", "iqsec2_114", 3_439.71, 320_447_020, 51_970, (223_870_592, 30_830_000, 5_000)),
    ("war_channel_filtering", "lof iqsec2_80", 3_415.09, 714_059_796, 115_806, (213_532_524, 30_830_000, 5_000)),
    ("war_channel_filtering", "lof iqsec2_106", 3_486.29, 319_676_270, 51_845, (223_757_949, 30_830_000, 5_000)),
    ("war_flattening", "manual iqsec2_80", 3_330.01, 714_059_796, 115_806, (213_532_524, 30_830_000, 5_000)),
    ("war_flattening", "manual iqsec2_3", 3_494.36, 313_522_602, 50_847, (218_963_451, 30_830_000, 5_000)),
]


@pytest.mark.parametrize(
    "block, job, peak_mib, values, windows, row_group",
    MEASURED_JOBS,
    ids=[f"{block}-{job}" for block, job, *_ in MEASURED_JOBS],
)
def test_the_configured_rates_cover_this_runs_peaks(block, job, peak_mib, values, windows, row_group):
    if row_group is None:
        held = war_load_bytes(values, windows)
    else:
        held = row_group_stream_bytes(*row_group, batch_rows=5000)
    model = BASE_BYTES + held + CLUSTER[block]["mem_bytes_per_window"] * windows

    # Covered before the headroom, and close enough that the headroom is a margin, not a guess
    assert 0.8 <= peak_mib * MIB / model <= 1.0


def test_the_wars_that_ran_out_of_memory_now_ask_for_enough_at_once(monkeypatch):
    # iqsec2_80 to 83 ran out of memory at 30 GB in diagnostic_figures, then peaked at up to
    # 48.83 GB when retried at 60 GB
    import neurodent.workflow.utils.resources as resources

    mem_mb = war_input_mem_mb(CLUSTER["diagnostic_figures"], "whole")
    for counts in [(714_059_796, 115_806), (665_206_578, 107_883)]:
        monkeypatch.setattr(resources, "count_war_values", lambda path, counts=counts: counts)
        assert mem_mb(None, SimpleNamespace(war_parquet="war.parquet"), 1) > 1.2 * 48_830


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


def test_a_war_streamed_in_one_batch_holds_all_of_it(create_war_files):
    ((parquet_path, _),) = create_war_files([("a1", "A1", "WT", "Male", {"days": 2})])
    metadata = pq.ParquetFile(parquet_path).metadata
    assert metadata.num_row_groups == 1  # as war_generation writes them
    values, windows = count_war_values(parquet_path)

    assert war_stream_bytes(parquet_path, batch_rows=windows) == (
        metadata.row_group(0).total_byte_size + war_load_bytes(values, windows)
    )


def test_streaming_holds_one_batch_of_a_war_in_one_row_group(create_war_files):
    ((parquet_path, _),) = create_war_files([("a1", "A1", "WT", "Male", {"days": 2})])
    metadata = pq.ParquetFile(parquet_path).metadata
    values, windows = count_war_values(parquet_path)

    assert war_stream_bytes(parquet_path, batch_rows=10) == (
        metadata.row_group(0).total_byte_size + war_load_bytes(math.ceil(values * 10 / windows), 10)
    )


def test_streamed_size_follows_the_row_groups_not_the_length(create_war_files, tmp_path):
    short, long = create_war_files([
        ("short", "A1", "WT", "Male", {"days": 2}),
        ("long", "A2", "WT", "Male", {"days": 6}),
    ])
    # In one row group, as war_generation writes them, the whole WAR is held, so the longer WAR
    # costs all of its extra uncompressed bytes
    extra = sum(
        sign * pq.ParquetFile(path).metadata.row_group(0).total_byte_size
        for sign, path in [(1, long[0]), (-1, short[0])]
    )
    assert extra > 0
    assert war_stream_bytes(long[0], batch_rows=10) - war_stream_bytes(
        short[0], batch_rows=10
    ) == pytest.approx(extra, rel=0.01)

    # Written again by the streaming engine, in row groups of 10 rows, a longer WAR costs no more
    short_copy = _streamed_copy(short[0], tmp_path / "short_copy", batch_rows=10)
    long_copy = _streamed_copy(long[0], tmp_path / "long_copy", batch_rows=10)
    assert pq.ParquetFile(long_copy).metadata.num_row_groups > 1
    assert war_stream_bytes(long_copy, batch_rows=10) == pytest.approx(
        war_stream_bytes(short_copy, batch_rows=10), rel=0.1
    )
    assert war_stream_bytes(long_copy, batch_rows=10) < war_stream_bytes(long[0], batch_rows=10)


def test_streamed_size_grows_with_channels(create_war_files):
    few, many = create_war_files([
        ("few", "A1", "WT", "Male", {}),
        ("many", "A2", "WT", "Male", {"channels": TEN_CHANNELS}),
    ])
    assert war_stream_bytes(many[0], batch_rows=10) > war_stream_bytes(few[0], batch_rows=10)


def test_streaming_batch_defaults_to_the_engines(create_war_files, monkeypatch):
    import neurodent.results.streaming as streaming

    ((parquet_path, _),) = create_war_files([("a1", "A1", "WT", "Male", {})])
    monkeypatch.setattr(streaming, "STREAM_BATCH_ROWS", 7)
    assert war_stream_bytes(parquet_path) == war_stream_bytes(parquet_path, batch_rows=7)


def test_whole_memory_follows_the_largest_war_and_every_window(create_war_files):
    paths = create_war_files([
        ("a1", "A1", "WT", "Male", {}),
        ("a2", "A2", "KO", "Female", {"channels": TEN_CHANNELS, "days": 2}),
    ])
    counts = [count_war_values(parquet_path) for parquet_path, _ in paths]
    mem_mb = war_input_mem_mb({"mem_mb": 500, "mem_bytes_per_window": 4000}, "whole")

    held = max(war_load_bytes(values, windows) for values, windows in counts)
    expected = _expected_mb(500, held, 4000, sum(windows for _, windows in counts))
    assert expected > 500
    assert mem_mb(None, _inputs(paths), 1) == expected
    assert mem_mb(None, _inputs(paths), 2) == 2 * expected


def test_streamed_memory_follows_the_streamed_war(create_war_files):
    paths = create_war_files([("a1", "A1", "WT", "Male", {"channels": TEN_CHANNELS, "days": 2})])
    ((parquet_path, _),) = paths
    windows = count_war_values(parquet_path)[1]
    mem_mb = war_input_mem_mb({"mem_mb": 1, "mem_bytes_per_window": 5000}, "streamed")

    expected = _expected_mb(1, war_stream_bytes(parquet_path), 5000, windows)
    assert mem_mb(None, _inputs(paths), 1) == expected
    assert mem_mb(None, _inputs(paths), 3) == 4 * expected


def test_mem_mb_is_the_least_a_job_asks_for(create_war_files):
    paths = create_war_files([("a1", "A1", "WT", "Male", {})])
    for war_loading in ("whole", "streamed"):
        mem_mb = war_input_mem_mb({"mem_mb": 30_000, "mem_bytes_per_window": 0}, war_loading)
        assert mem_mb(None, _inputs(paths), 1) == 30_000
        assert mem_mb(None, _inputs(paths), 2) == 60_000


def test_a_single_war_given_as_a_string_is_one_war(create_war_files):
    # Snakemake passes one path as a plain string, for example while a checkpoint is pending
    ((parquet_path, _),) = create_war_files([("a1", "A1", "WT", "Male", {})])
    mem_mb = war_input_mem_mb({"mem_mb": 0, "mem_bytes_per_window": 0}, "whole")
    values, windows = count_war_values(parquet_path)

    assert mem_mb(None, SimpleNamespace(war_parquet=str(parquet_path)), 1) == _expected_mb(
        0, war_load_bytes(values, windows), 0, windows
    )


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


def test_a_missing_war_is_named_so_a_dry_run_shows_tbd(tmp_path):
    # Snakemake shows a resource as TBD, instead of failing, when its function raises
    # FileNotFoundError for a path that is one of the rule's inputs
    missing = str(tmp_path / "a1" / "war.parquet")

    for count in (count_war_values, war_stream_bytes):
        with pytest.raises(FileNotFoundError) as excinfo:
            count(missing)
        assert excinfo.value.filename == missing


def test_a_rule_that_reads_no_wars_gets_its_block_doubled_on_retry():
    resources = rule_resources(_block(time="2h", mem_mb=4000, nodes=1))

    assert resources["time"] == "2h"
    assert resources["nodes"] == 1
    assert resources["mem_mb"](None, 1) == 4000
    assert resources["mem_mb"](None, 3) == 16000


@pytest.mark.parametrize("war_loading", ["whole", "streamed"])
def test_a_rule_that_reads_wars_is_sized_from_them(create_war_files, war_loading):
    paths = create_war_files([("a1", "A1", "WT", "Male", {"channels": TEN_CHANNELS})])
    block = _block(mem_mb=1, mem_bytes_per_window=2000)
    resources = rule_resources(block, war_loading=war_loading)

    assert resources["time"] == "1h"
    assert resources["mem_mb"](None, _inputs(paths), 2) == war_input_mem_mb(block, war_loading)(
        None, _inputs(paths), 2
    )


def test_time_rates_give_a_computed_limit(create_war_files):
    paths = create_war_files([("a1", "A1", "WT", "Male", {})])
    block = _block(
        time="30m", mem_mb=1, mem_bytes_per_window=0, seconds_per_value=1e-6, seconds_per_window=0.01
    )
    resources = rule_resources(block, war_loading="whole")

    assert resources["time"](None, _inputs(paths), 1) == one_war_at_a_time_time(block)(
        None, _inputs(paths), 1
    )


@pytest.mark.parametrize(
    "keys", [{"mem_bytes_per_window": 0}, {"seconds_per_value": 0, "seconds_per_window": 0}]
)
def test_rates_on_a_rule_that_reads_no_wars_are_refused(keys):
    with pytest.raises(ValueError, match="war_loading"):
        rule_resources(_block(**keys))


def test_a_rule_that_reads_wars_needs_a_per_window_rate():
    with pytest.raises(KeyError, match="mem_bytes_per_window"):
        rule_resources(_block(), war_loading="whole")


def test_an_unknown_war_loading_is_refused():
    with pytest.raises(ValueError, match="whole"):
        rule_resources(_block(mem_bytes_per_window=0), war_loading="lazy")
