"""Adding a hash to an animal id renames everything named after the animal.

The keys of bad_channels_dict and lof_scores_dict used to keep the old animalday names, so the
WAR found no entry for any renamed animalday and added an empty one, with a warning each (210
in a filtering_comparison run), and the LOF scores were no longer found under their animaldays.
The id was also replaced wherever it occurred in an animalday, not only at its start.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from neurodent.results import WindowAnalysisResult
from neurodent.results.streaming import rename_animalday

CHANNELS = ["LMot", "RMot", "LBar"]


def _war(animal_id, days, n_channels=3, windows=4):
    """A WAR over several days, with LOF scores and bad channels for every animalday.

    Day names contain the animal id again when the id is a digit that also occurs in dates.
    """
    rows, lof, bad = [], {}, {}
    channels = CHANNELS[:n_channels]
    for i, day in enumerate(days):
        animalday = f"{animal_id} WT {day}"
        start = datetime.strptime(day, "%b-%d-%Y") + timedelta(hours=8)
        for w in range(windows):
            rows.append({
                "animal": animal_id, "genotype": "WT", "animalday": animalday, "day": day,
                "timestamp": start + timedelta(seconds=5 * w), "duration": 5.0,
                "rms": list(np.full(n_channels, 10.0 + w)),
            })
        lof[animalday] = {"lof_scores": [1.0 + i] * n_channels, "channel_names": channels}
        bad[animalday] = channels[:1] if i % 2 == 0 else []
    return WindowAnalysisResult(
        pd.DataFrame(rows), animal_id=animal_id, genotype="WT", channel_names=channels,
        bad_channels_dict=bad, lof_scores_dict=lof,
    )


CASES = {
    "letters, two days": ("A10", ["Jan-31-2020", "Feb-01-2020"], 3),
    "a digit that recurs in the dates, three days": ("1", ["Jan-01-2021", "Jan-11-2021", "Nov-11-2021"], 2),
    "one day, one channel": ("F22", ["Mar-05-2022"], 1),
}


@pytest.fixture
def fixed_token(monkeypatch):
    monkeypatch.setattr(secrets, "token_hex", lambda nbytes=None: "beef")
    return "beef"


@pytest.mark.parametrize("case", list(CASES))
def test_eager_hash_renames_the_animaldays_and_both_dicts(case, fixed_token, caplog):
    animal_id, days, n_channels = CASES[case]
    war = _war(animal_id, days, n_channels)
    old_bad, old_lof = dict(war.bad_channels_dict), dict(war.lof_scores_dict)

    with caplog.at_level(logging.WARNING):
        war.add_unique_hash(2)

    new_id = f"{animal_id}_beef"
    expected_days = [f"{new_id} WT {day}" for day in days]
    assert war.animal_id == new_id
    assert (war.result["animal"] == new_id).all()
    assert list(war.result["animalday"].unique()) == expected_days
    assert sorted(war.bad_channels_dict) == sorted(expected_days)
    assert sorted(war.lof_scores_dict) == sorted(expected_days)
    for day, new_day in zip(days, expected_days):
        old_day = f"{animal_id} WT {day}"
        assert war.bad_channels_dict[new_day] == old_bad[old_day]
        assert war.lof_scores_dict[new_day] == old_lof[old_day]
    assert "Added missing animalday" not in caplog.text
    # The LOF scores are still found under each animalday
    assert set(war.get_bad_channels_by_lof_threshold(1.5)) == set(expected_days)


@pytest.mark.parametrize("case", list(CASES))
def test_streaming_hash_renames_alike(case, fixed_token, tmp_path):
    animal_id, days, n_channels = CASES[case]
    _war(animal_id, days, n_channels).save_parquet_and_json(tmp_path / "src", filename="war")

    lazy = WindowAnalysisResult.scan_parquet_and_json(tmp_path / "src", filename="war")
    lazy.add_unique_hash(2)
    lazy.save_parquet_and_json(tmp_path / "lazy", filename="war", batch_size=3)
    eager = _war(animal_id, days, n_channels)
    eager.add_unique_hash(2)

    sidecar = json.loads((tmp_path / "lazy" / "war.json").read_text())
    streamed = WindowAnalysisResult.load_parquet_and_json(tmp_path / "lazy")
    assert sidecar["animal_id"] == eager.animal_id
    assert sidecar["bad_channels_dict"] == eager.bad_channels_dict
    assert sidecar["lof_scores_dict"] == eager.lof_scores_dict
    assert list(streamed.result["animalday"]) == list(eager.result["animalday"])


def test_a_war_aggregated_without_animalday_keeps_none(fixed_token):
    war = _war("A10", ["Jan-31-2020"])
    war.result["isday"] = [True, True, False, False]
    war.aggregate_time_windows(groupby=["isday"])
    assert war.result["animalday"].isna().all()

    war.add_unique_hash(2)

    assert war.animal_id == "A10_beef"
    assert war.result["animalday"].isna().all()


def test_the_suffix_has_two_hex_characters_per_byte():
    for nbytes in (2, 4):
        war = _war("A10", ["Jan-31-2020"])
        war.add_unique_hash(nbytes)
        suffix = war.animal_id.split("_", 1)[1]
        assert len(suffix) == 2 * nbytes
        int(suffix, 16)


class TestRenameAnimalday:
    def test_only_the_leading_id_changes(self):
        assert rename_animalday("1 WT Jan-01-2021", "1", "1_beef") == "1_beef WT Jan-01-2021"

    def test_an_animalday_of_another_animal_raises(self):
        with pytest.raises(ValueError, match="does not start with the animal id"):
            rename_animalday("A11 WT Jan-01-2021", "A10", "A10_beef")


def test_wars_made_without_lof_scores_do_not_share_them(fixed_token):
    """The default lof_scores_dict was one dict for every WAR made without one, and each WAR
    added its animaldays to it, so a later WAR held another animal's keys and renaming them
    with the hash failed."""
    rows = [{"animal": "B2", "genotype": "KO", "animalday": "B2 KO Jan-05-2026",
             "timestamp": datetime(2026, 1, 5, 8), "duration": 5.0, "rms": [1.0, 1.0]}]
    WindowAnalysisResult(pd.DataFrame(rows), animal_id="B2", genotype="KO", channel_names=["LMot", "RMot"])
    rows = [{**rows[0], "animal": "A10", "genotype": "WT", "animalday": "A10 WT Jan-06-2026"}]
    war = WindowAnalysisResult(pd.DataFrame(rows), animal_id="A10", genotype="WT", channel_names=["LMot", "RMot"])

    assert list(war.lof_scores_dict) == ["A10 WT Jan-06-2026"]
    war.add_unique_hash(2)
    assert list(war.lof_scores_dict) == ["A10_beef WT Jan-06-2026"]
