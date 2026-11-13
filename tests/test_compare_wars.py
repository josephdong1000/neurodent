"""scripts/compare_wars.py tells identical WARs from WARs that differ, and says where."""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurodent import constants
from neurodent.results import WindowAnalysisResult

SCRIPT = Path(__file__).parent.parent / "scripts" / "compare_wars.py"
DAYS = ["A1 WT Jan-01-2023", "A1 WT Jan-02-2023"]
CHANNELS = ["LMot", "RMot", "LBar"]


@pytest.fixture(scope="module")
def compare_wars():
    spec = importlib.util.spec_from_file_location("compare_wars_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(folder, windows=4, extra_window=False, bad_channels=None, rms_shift=0.0):
    rng = np.random.default_rng(0)
    rows = []
    for d, day in enumerate(DAYS):
        n = windows + (1 if extra_window and d == 1 else 0)
        for w in range(n):
            rms = rng.uniform(100, 200, len(CHANNELS))
            rows.append({
                "animal": "A1", "genotype": "WT", "animalday": day,
                "timestamp": datetime(2023, 1, 1 + d, 8) + timedelta(seconds=5 * w), "duration": 5.0,
                "endfile": np.nan, "rms": list(rms + (rms_shift if (d, w) == (0, 2) else 0.0)),
                "psdband": {b: list(np.full(len(CHANNELS), 0.2)) for b in constants.BAND_NAMES},
            })
    war = WindowAnalysisResult(
        pd.DataFrame(rows), animal_id="A1", genotype="WT", channel_names=list(CHANNELS),
        bad_channels_dict=bad_channels or {day: [] for day in DAYS},
    )
    war.save_parquet_and_json(folder, filename="war", slugify_filename=False)
    return folder


def test_identical_wars(compare_wars, tmp_path, capsys):
    a, b = _save(tmp_path / "a"), _save(tmp_path / "b")

    assert compare_wars.main([str(a), str(b)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "identical"


def test_a_changed_value_is_found_by_column_and_row(compare_wars, tmp_path, capsys):
    a, b = _save(tmp_path / "a"), _save(tmp_path / "b", rms_shift=0.5)

    assert compare_wars.main([str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "column rms: 1 of 8 rows differ, largest absolute difference 0.5" in out
    assert "column psdband" not in out


def test_an_added_window_is_counted_by_animalday(compare_wars, tmp_path, capsys):
    a, b = _save(tmp_path / "a"), _save(tmp_path / "b", extra_window=True)

    assert compare_wars.main([str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "rows: A 8, B 9" in out
    assert f"rows only in B: 1, by animalday: {{'{DAYS[1]}': 1}}" in out
    assert "column rms" not in out, "the common rows are equal"


def test_a_json_difference_names_the_key(compare_wars, tmp_path, capsys):
    a = _save(tmp_path / "a")
    b = _save(tmp_path / "b", bad_channels={DAYS[0]: ["LMot"], DAYS[1]: []})

    assert compare_wars.main([str(a), str(b)]) == 1
    assert f"/bad_channels_dict/{DAYS[0]}" in capsys.readouterr().out


def test_nan_equals_nan_and_nested_values_compare_exactly(compare_wars):
    assert compare_wars.equal(float("nan"), float("nan"))
    assert compare_wars.equal({"a": [1.0, np.nan]}, {"a": np.array([1.0, np.nan])})
    assert not compare_wars.equal({"a": [1.0, 2.0]}, {"a": [1.0, 2.0 + 1e-12]})
    assert not compare_wars.equal((np.arange(3.0), np.ones(2)), (np.arange(3.0), np.ones(3)))


def test_a_missing_folder_is_a_usage_error(compare_wars, tmp_path):
    assert compare_wars.main([str(tmp_path / "nope"), str(tmp_path)]) == 2
