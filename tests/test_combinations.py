"""Tests for the combination evaluation in ``scripts/labeling/select_detectors.py``.

F, L and M are checked cell by cell against the pipeline's own decisions on synthetic WARs: through
``WindowAnalysisResult.apply_filters`` as ``workflow/scripts/filter_wars_fragments.py`` and
``filter_wars_channels.py`` call it (threshold ties, missing scores, high_beta's window rule, the minimum valid
channel rules and the unit each acts on), and through the workflow scripts themselves, run on saved WARs:
``standardize_wars.py``, then ``filter_wars_fragments.py``, then the manual and LOF branches of
``filter_wars_channels.py``, which give today (F or M) and lof_branch (F or L), and each branch alone on the
standardized WAR, which gives M and L. The WARs include recordings padded to their dataset's standardized
channels, a listed channel the recording lacks, and channels of a recording without a labeled cell. The per-cell
columns are built with ``extract_dev_features``' helpers, and the recording records as the extraction writes
them. Also covered: the rule columns against ``scoring.consensus`` on synthetic rater tables, the settings and
canonical channels read from the config, the config files compared with the extraction's record, the
combination masks, the selection and its tie rule, the nested runs, the metrics, the paired intervals and the
breakdowns under every rule, and the nested runs a partition with one dataset or one cv_group cannot support.
Labels are synthetic.
"""

import copy
import importlib.util
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import zscore
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "labeling"))
import extract_dev_features as EDF  # noqa: E402
import select_detectors as SD  # noqa: E402
from neurodent.results import WindowAnalysisResult  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402
from neurodent.results.scoring import LABEL_MAP, consensus  # noqa: E402

pytestmark = pytest.mark.unit

GRID = adr.grid_config_names()
CH = ["LMot", "RMot", "LBar", "RBar", "LHip"]
# Ten standardized channels, of which the synthetic recordings hold the five in CH (PADDED's dataset).
CANON = ["LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
DAYS = ["X1 WT Jan-01-2025", "X1 WT Jan-02-2025", "X1 WT Jan-03-2025"]
FRAGMENT = {"logrms_range": {"z_range": 3}, "high_rms": {"max_rms": 500}, "low_rms": {"min_rms": 50},
            "high_beta": {"max_beta_prop": 0.4}}
SETTINGS = {"fragment_filters": FRAGMENT, "fragment_min_valid_channels": 3,
            "lof": {"reject_lof_threshold": 2.5, "min_valid_channels": 3}, "config_lists": {"min_valid_channels": 3},
            "canonical_channels": {"ds": CH}}
PADDED = {**SETTINGS, "canonical_channels": {"ds": CANON}}
NO_CHANNEL_LIST = {"reject_channels": {"bad_channels": []}}


def _settings(**fragment):
    """SETTINGS with only the given fragment filters."""
    return {**SETTINGS, "fragment_filters": fragment}


def _war(rms, beta=None, total=None, animaldays=None, lof=None):
    """A WAR of (W, C) rms and beta proportions, one row per 5 s window; psdband beta is beta times psdtotal.

    A channel whose psdtotal is 0 gets a beta band of 0, so its proportion is 0/0. psdfrac holds the same beta
    proportion, psdband over psdtotal, which the pipeline's high_beta reads.
    """
    n_rows, n_ch = rms.shape
    beta = np.full(rms.shape, 0.1) if beta is None else beta
    total = np.ones(rms.shape) if total is None else total
    animaldays = animaldays or [DAYS[0]] * n_rows
    band = np.where(total == 0, 0.0, beta * total)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = band / total
    rows = []
    for i, day in enumerate(animaldays):
        bands = {b: (band[i] if b == "beta" else np.full(n_ch, 0.1)).tolist() for b in BANDS}
        rows.append({"animal": "X1", "animalday": day, "genotype": "WT", "duration": 5.0, "isday": True,
                     "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(days=DAYS.index(day), seconds=5 * i),
                     "rms": rms[i].tolist(), "psdtotal": total[i].tolist(), "psdband": bands,
                     "psdfrac": {b: (frac[i] if b == "beta" else np.full(n_ch, 0.1)).tolist() for b in BANDS}})
    lof = {day: {"lof_scores": scores, "channel_names": CH[:n_ch]} for day, scores in (lof or {}).items()}
    return WindowAnalysisResult(pd.DataFrame(rows), animal_id="X1", genotype="WT", channel_names=CH[:n_ch],
                                lof_scores_dict=lof)


def _cells(war, z_unit="war", window_max=np.nanmax):
    """Every (window, channel) cell with the columns extract_dev_features computes, one recording per animalday,
    in the dataset "ds".

    rms, the log RMS z score over the whole WAR and the beta proportions come from the extraction's own
    ``war_filter_scores``. ``z_unit`` "recording" takes the z score within each animalday instead; ``window_max``
    reduces the beta proportions over channels.
    """
    ch = list(war.channel_abbrevs)
    animaldays = war.result["animalday"].to_numpy()
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        rms_all, z_all, beta_all = EDF.war_filter_scores(war)
        rows = []
        for i, day in enumerate(pd.unique(animaldays)):
            sel = np.flatnonzero(animaldays == day)
            z = z_all[sel] if z_unit == "war" else zscore(np.log(rms_all[sel]), axis=0, nan_policy="omit")
            cons = pd.DataFrame([{"recording": f"X1__{i}", "window": w, "channel": c, "t_start_s": 5.0 * w,
                                  "y_true": 0} for w in range(len(sel)) for c in ch])
            cells = EDF.align_cells(rms_all[sel], z, window_max(beta_all[sel], axis=1), EDF.lof_by_abbrev(war, day),
                                    ch, cons, np.arange(len(sel)) * 5.0)
            for cell in cells:
                cell.update(animalday=day, row=int(sel[cell["window"]]))
            rows += cells
    return pd.DataFrame(rows).assign(dataset="ds", config_list=0, config_list_defined=False)


def _records(cells, own=None, lists=None, standardized=CANON):
    """The recording records extract_dev_features writes for these cells' recordings.

    Args:
        cells: Cells with recording and channel.
        own: ``{recording: [channel]}``, the recording's own channels; a recording missing from it has the
            channels its cells are on.
        lists: ``{recording: [listed channel]}``; a recording missing from it has no list.
        standardized: The order the channels are written in.

    Returns:
        ``{"recording_channels", "config_list_by_recording"}``, the listed names a recording lacks under
        ``absent``.
    """
    records = {"recording_channels": {}, "config_list_by_recording": {}}
    for rec, sub in cells.groupby("recording"):
        channels = (own or {}).get(rec) or [c for c in standardized if c in set(sub["channel"])]
        listed = (lists or {}).get(rec)
        records["recording_channels"][rec] = list(channels)
        records["config_list_by_recording"][rec] = {
            "defined": listed is not None, "channels": [c for c in channels if c in (listed or [])],
            "absent": [c for c in (listed or []) if c not in channels]}
    return records


def _with_lists(cells, records):
    """``cells`` with config_list and config_list_defined as the records give them."""
    lists = records["config_list_by_recording"]
    out = cells.copy()
    out["config_list_defined"] = out["recording"].map(lambda r: lists[r]["defined"]).astype(bool)
    out["config_list"] = [int(c in lists[r]["channels"]) for r, c in zip(out["recording"], out["channel"])]
    return out


def _recordings(cells, settings, **records):
    """``recorded_channels`` for the cells, from ``_records(cells, **records)``."""
    return SD.recorded_channels(cells, _records(cells, **records), settings)


def _pad(war):
    """The WAR padded with NaN to CANON, as standardize_wars.py pads a recording that lacks channels."""
    padded = copy.deepcopy(war)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        padded.reorder_and_pad_channels(CANON, use_abbrevs=True, inplace=True)
    return padded


def _pipeline(war, filter_config, min_valid=3):
    """Return the pipeline's (W, C) rejections: the cells ``apply_filters`` turns to NaN (read on psdtotal)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = war.apply_filters(filter_config=filter_config, min_valid_channels=min_valid)
    before = extract_linear_array(war.result["psdtotal"])
    return np.isnan(extract_linear_array(out.result["psdtotal"])) & ~np.isnan(before)


def _at(cells, decisions, war):
    """Return (W, C) decisions at each cell's row and channel."""
    col = {c: j for j, c in enumerate(war.channel_abbrevs)}
    return decisions[cells["row"].to_numpy(), cells["channel"].map(col).to_numpy()].astype(int)


def _cell(cells, row, channel):
    return (cells["row"] == row) & (cells["channel"] == channel)


# F, L and M against the pipeline


def _planted_war():
    rng = np.random.default_rng(0)
    rms = rng.uniform(90, 110, (40, 5))
    rms[0, 0] = 500.0          # equal to max_rms
    rms[1, 0] = 500.5
    rms[2, 1] = 50.0           # equal to min_rms
    rms[3, 1] = 49.5
    rms[4, 1:4] = 600.0        # three rejected, two left: the window rule takes the rest
    rms[5, 2] = np.nan         # missing
    rms[6, 3] = 5.0            # a log RMS outlier
    rms[9, :2] = 600.0         # two rejected, three left: enough
    beta = rng.uniform(0.05, 0.2, (40, 5))
    beta[7, 2] = 0.4           # equal to max_beta_prop
    beta[8, 4] = 0.41          # over it on one channel
    return _war(rms, beta)


@pytest.mark.parametrize("name, cells_rejected, cells_kept", [
    ("high_rms", [(1, "LMot")], [(0, "LMot")]),
    ("low_rms", [(3, "RMot"), (6, "RBar")], [(2, "RMot"), (5, "LBar")]),
    ("logrms_range", [(6, "RBar")], [(5, "LBar")]),
    ("high_beta", [(8, c) for c in CH], [(7, "LBar")]),
])
def test_each_fragment_filter_follows_apply_filters(name, cells_rejected, cells_kept):
    """Strict comparisons, a missing score kept, high_beta's whole window; each filter alone, as configured."""
    war = _planted_war()
    cells = _cells(war)
    settings = _settings(**{name: FRAGMENT[name]})
    f, _ = SD.fragment_reject(cells, settings, _recordings(cells, settings))
    assert f.tolist() == _at(cells, _pipeline(war, {name: FRAGMENT[name]}), war).tolist()
    for row, ch in cells_rejected:
        assert f[_cell(cells, row, ch)].tolist() == [1], (row, ch)
    for row, ch in cells_kept:
        assert f[_cell(cells, row, ch)].tolist() == [0], (row, ch)


def test_fragment_filters_together_follow_apply_filters_with_the_window_rule():
    war = _planted_war()
    cells = _cells(war)
    f, audit = SD.fragment_reject(cells, SETTINGS, _recordings(cells, SETTINGS))
    assert f.tolist() == _at(cells, _pipeline(war, FRAGMENT), war).tolist()
    assert f[(cells["row"] == 4).to_numpy()].tolist() == [1] * 5, "two cells left of five: every channel goes"
    assert f[(cells["row"] == 9).to_numpy()].tolist() == [1, 1, 0, 0, 0], "three left of five are enough"
    assert audit["n_window_rule_only"] == 2 and audit["n_windows_with_channels_without_cells"] == 0
    assert audit["n_kept_by_channels_without_cells"] == 0


@pytest.mark.parametrize("row, dropped, expected, leaning", [
    (4, "LMot", [1, 1, 1, 1], 0),  # one kept cell and the dropped one: two, too few either way
    (9, "LHip", [1, 1, 0, 0], 2),  # two kept cells and the dropped one, kept by the pipeline: three
    (4, "RBar", [0, 1, 1, 0], 2),  # the dropped cell is one the pipeline rejects: the audit names the two kept
])
def test_window_rule_counts_a_cell_without_labels_as_kept(row, dropped, expected, leaning):
    """A window missing a cell (every rater abstained) counts it as kept, as its scores are unknown; the audit
    names the cells the window rule keeps only because of it, the only ones that can differ from the pipeline."""
    war = _planted_war()
    cells = _cells(war)
    kept = cells[~_cell(cells, row, dropped)]
    f, audit = SD.fragment_reject(kept, SETTINGS, _recordings(kept, SETTINGS, own={"X1__0": CH}))
    assert f[(kept["row"] == row).to_numpy()].tolist() == expected
    assert audit["n_windows_with_channels_without_cells"] == 1 and audit["n_kept_by_channels_without_cells"] == leaning
    differ = f != _at(kept, _pipeline(war, FRAGMENT), war)
    assert differ.sum() == (leaning if dropped == "RBar" else 0)


def test_high_beta_takes_the_nan_aware_window_maximum():
    """A 0/0 proportion on one channel does not hide another channel over the threshold.

    extract_dev_features has to take beta_wmax with np.nanmax: np.max returns NaN for the window, which keeps it.
    """
    rms = np.full((6, 5), 100.0)
    beta, total = np.full((6, 5), 0.1), np.ones((6, 5))
    beta[2, 0] = 0.5
    total[2, 1] = 0.0
    war = _war(rms, beta, total)
    settings = _settings(high_beta=FRAGMENT["high_beta"])
    pipeline = _pipeline(war, {"high_beta": FRAGMENT["high_beta"]})
    assert pipeline[2].all() and not np.delete(pipeline, 2, axis=0).any()
    nan_aware = _cells(war)
    f, _ = SD.fragment_reject(nan_aware, settings, _recordings(nan_aware, settings))
    assert f.tolist() == _at(nan_aware, pipeline, war).tolist()
    plain = _cells(war, window_max=np.max)
    assert not SD.fragment_reject(plain, settings, _recordings(plain, settings))[0][(plain["row"] == 2).to_numpy()].any()


def test_log_rms_z_is_taken_over_the_whole_war():
    """The pipeline filters an animal's WAR, every recording together; a z score per recording disagrees."""
    rng = np.random.default_rng(1)
    rms = np.vstack([rng.uniform(95, 105, (30, 5)), rng.uniform(380, 420, (30, 5))])
    rms[45, 0] = 120.0  # low for its own recording, ordinary for the animal
    war = _war(rms, animaldays=[DAYS[0]] * 30 + [DAYS[1]] * 30)
    settings = _settings(logrms_range=FRAGMENT["logrms_range"])
    pipeline = _pipeline(war, {"logrms_range": FRAGMENT["logrms_range"]})
    whole = _cells(war, z_unit="war")
    assert SD.fragment_reject(whole, settings, _recordings(whole, settings))[0].tolist() == _at(whole, pipeline, war).tolist()
    assert not pipeline[45, 0]
    per_recording = _cells(war, z_unit="recording")
    f, _ = SD.fragment_reject(per_recording, settings, _recordings(per_recording, settings))
    assert f[_cell(per_recording, 45, "LMot")].tolist() == [1]


def test_l_follows_the_lof_branch_per_recording():
    """A score equal to the threshold is rejected, each animalday has its own channels, and a recording left with
    two channels is rejected whole."""
    lof = {DAYS[0]: [1.0, 2.5, 2.4999, 1.0, 1.0], DAYS[1]: [3.0, 1.0, 1.0, 1.0, 1.0],
           DAYS[2]: [3.0, 3.0, 3.0, 1.0, 1.0]}
    war = _war(np.full((30, 5), 100.0), animaldays=[d for d in DAYS for _ in range(10)], lof=lof)
    bad = war.get_bad_channels_by_lof_threshold(2.5)
    assert bad == {DAYS[0]: ["RMot"], DAYS[1]: ["LMot"], DAYS[2]: ["LMot", "RMot", "LBar"]}
    pipeline = _pipeline(war, {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": bad}})
    cells = _cells(war)
    lo, audit = SD.lof_reject(cells, SETTINGS, _recordings(cells, SETTINGS))
    assert lo.tolist() == _at(cells, pipeline, war).tolist()
    assert audit["n_rejected"] == 10 + 10 + 50 and audit["n_recording_rule_only"] == 20


def test_l_keeps_a_cell_without_a_score():
    cells = pd.DataFrame({"recording": "r__0", "window": 0, "channel": CH, "dataset": "ds",
                          "lof": [3.0, np.nan, 1.0, 1.0, 1.0], "config_list": 0, "config_list_defined": False})
    lo, audit = SD.lof_reject(cells, SETTINGS, _recordings(cells, SETTINGS))
    assert lo.tolist() == [1, 0, 0, 0, 0] and audit["n_score_missing"] == 1


@pytest.mark.parametrize("form", ["sessions", "every_recording"])
def test_m_follows_the_manual_branch(form):
    """Per-session lists go through reject_channels_by_session; a list-form entry becomes the animal's
    reject_channels, as filter_wars_channels.py does with "_all". A recording left with two channels goes whole."""
    war = _war(np.full((30, 5), 100.0), animaldays=[d for d in DAYS for _ in range(10)])
    if form == "sessions":
        lists = {DAYS[0]: ["LBar"], DAYS[1]: ["LMot", "RMot", "LBar"], DAYS[2]: []}
        config = {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": lists}}
    else:
        lists = {day: ["LMot", "RMot", "LBar"] for day in DAYS}
        config = {"reject_channels": {"bad_channels": lists[DAYS[0]]}, "reject_channels_by_session": {}}
    cells = _cells(war)
    records = _records(cells, lists={f"X1__{i}": lists[day] for i, day in enumerate(DAYS)}, standardized=CH)
    cells = _with_lists(cells, records)
    m, audit = SD.list_reject(cells, SETTINGS, SD.recorded_channels(cells, records, SETTINGS))
    assert m.tolist() == _at(cells, _pipeline(war, config), war).tolist()
    assert audit["n_recording_rule_only"] == (20 if form == "sessions" else 60)
    assert audit["n_defined"] == len(cells), "an empty list is a list"


# A recording with half its dataset's standardized channels: the pipeline filters it padded with NaN


def test_f_counts_the_padded_channels_as_kept():
    """Every fragment filter keeps a NaN cell, so the padded channels count in the window rule."""
    war = _planted_war()
    padded = _pad(war)
    cells = _cells(war)
    f, audit = SD.fragment_reject(cells, PADDED, _recordings(cells, PADDED))
    assert f.tolist() == _at(cells, _pipeline(padded, FRAGMENT), padded).tolist()
    assert f[(cells["row"] == 4).to_numpy()].tolist() == [0, 1, 1, 1, 0], "two kept cells and five padded ones"
    assert audit["n_window_rule_only"] == 0 and audit["n_kept_by_channels_without_cells"] == 0, "padding is exact"
    assert audit["n_windows_with_channels_without_cells"] == 0
    unpadded = SD.fragment_reject(cells, SETTINGS, _recordings(cells, SETTINGS))[0]
    assert unpadded[(cells["row"] == 4).to_numpy()].tolist() == [1] * 5


def test_l_counts_the_padded_channels_as_valid():
    """A padded channel has no LOF score, so the LOF branch never rejects it and the recording rule counts it."""
    war = _war(np.full((10, 5), 100.0), lof={DAYS[0]: [3.0, 3.0, 3.0, 1.0, 1.0]})
    padded = _pad(war)
    bad = padded.get_bad_channels_by_lof_threshold(2.5)
    pipeline = _pipeline(padded, {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": bad}})
    cells = _cells(war)
    lo, audit = SD.lof_reject(cells, PADDED, _recordings(cells, PADDED))
    assert lo.tolist() == _at(cells, pipeline, padded).tolist()
    assert lo.sum() == 30 and audit["n_kept_by_channels_without_cells"] == 0
    assert SD.lof_reject(cells, SETTINGS, _recordings(cells, SETTINGS))[0].sum() == 50, "three of five flagged leave two"


def test_m_counts_listed_channels_the_recording_lacks():
    """The manual branch rejects a listed padded channel, so it counts against the recording rule; the extraction
    writes it under absent."""
    war = _war(np.full((20, 5), 100.0), animaldays=[DAYS[0]] * 10 + [DAYS[1]] * 10)
    padded = _pad(war)
    lists = {DAYS[0]: ["LMot", "RMot", "LBar"],
             DAYS[1]: ["LMot", "LBar", "RBar", "RHip", "LAud", "RAud", "LVis", "RVis"]}
    pipeline = _pipeline(padded, {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": lists}})
    cells = _cells(war)
    records = _records(cells, lists={"X1__0": lists[DAYS[0]], "X1__1": lists[DAYS[1]]})
    assert records["config_list_by_recording"]["X1__1"]["absent"] == lists[DAYS[1]][3:]
    cells = _with_lists(cells, records)
    m, audit = SD.list_reject(cells, PADDED, SD.recorded_channels(cells, records, PADDED))
    assert m.tolist() == _at(cells, pipeline, padded).tolist()
    day = cells["animalday"].to_numpy()
    assert m[day == DAYS[0]].reshape(10, 5).sum(axis=0).tolist() == [10, 10, 10, 0, 0], "seven channels stay valid"
    assert m[day == DAYS[1]].all(), "two standardized channels stay unlisted"
    assert audit["listed_without_cells"] == {"X1__1": lists[DAYS[1]][3:]}
    assert audit["n_recording_rule_only"] == 20
    records["config_list_by_recording"]["X1__1"]["absent"] = []
    assert not SD.list_reject(cells, PADDED, SD.recorded_channels(cells, records, PADDED))[0][day == DAYS[1]].all(), \
        "without the absent entries, seven look valid"


def test_a_channel_without_a_labeled_cell_counts_as_present():
    """A channel of the recording that every rater abstained on in every window has no cell, yet it is the
    recording's own channel: M counts it as listed when the list names it, and F and L count it as valid. The
    pipeline agrees here, where its scores are ordinary; the audits name the channel."""
    lof = {DAYS[0]: [1.0, 1.0, 3.0, 1.0, 1.0]}
    war = _war(np.full((10, 5), 100.0), lof=lof)
    padded = _pad(war)
    full = _cells(war)
    cells = full[full["channel"] != "RBar"]  # every rater abstained on RBar
    own = {"X1__0": CH}
    lists = {"X1__0": ["LMot", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"]}  # five of them padded
    records = _records(cells, own=own, lists=lists)
    assert records["config_list_by_recording"]["X1__0"]["channels"] == ["LMot", "RBar", "LHip"]
    cells = _with_lists(cells, records)
    members, audits = SD.member_masks(cells, PADDED, records)
    config = {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": {DAYS[0]: lists["X1__0"]}}}
    assert members["M"].tolist() == _at(cells, _pipeline(padded, config), padded).tolist()
    assert members["M"].all(), "ten standardized channels, eight listed with RBar: two are too few"
    assert audits["channels_without_cells"] == {"X1__0": ["RBar"]} and audits["padded_channels"] == {"X1__0": CANON[5:]}
    assert audits["M"]["listed_without_cells"] == {"X1__0": ["RBar", "RHip", "LAud", "RAud", "LVis", "RVis"]}
    on_cells = (cells["config_list"] == 1).groupby(cells["channel"]).any().sum()
    assert len(CANON) - on_cells >= 3, "counted from the listed cells alone, the recording would stay"
    # The recording's channels read off its cells instead: RBar looks padded, M is unchanged (a listed
    # standardized channel counts either way) and the audits call RBar padded.
    taken_as_padded = _records(cells, lists=lists)
    assert taken_as_padded["config_list_by_recording"]["X1__0"]["absent"][0] == "RBar"
    m, other = SD.member_masks(cells, PADDED, taken_as_padded)
    assert m["M"].tolist() == members["M"].tolist() and other["channels_without_cells"] == {}
    assert other["padded_channels"] == {"X1__0": ["RBar", *CANON[5:]]}
    bad = padded.get_bad_channels_by_lof_threshold(2.5)
    pipeline_l = _pipeline(padded, {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": bad}})
    assert members["L"].tolist() == _at(cells, pipeline_l, padded).tolist()
    assert members["F"].tolist() == _at(cells, _pipeline(padded, FRAGMENT), padded).tolist()


def test_l_names_the_recordings_kept_by_channels_without_cells():
    """Five standardized channels, LOF flags two; a third channel without a cell keeps the recording, and the
    audit names it."""
    war = _war(np.full((10, 5), 100.0), lof={DAYS[0]: [3.0, 3.0, 1.0, 1.0, 1.0]})
    full = _cells(war)
    cells = full[full["channel"] != "LHip"]
    lo, audit = SD.lof_reject(cells, SETTINGS, _recordings(cells, SETTINGS, own={"X1__0": CH}))
    assert lo.sum() == 20, "three valid channels, LHip among them"
    assert audit["recordings_kept_by_channels_without_cells"] == ["X1__0"]
    assert audit["n_kept_by_channels_without_cells"] == 20


def test_recorded_channels_follow_the_standardized_order_and_the_list():
    cells = _cells(_war(np.full((4, 5), 100.0)))
    records = {"recording_channels": {"X1__0": ["LHip", "LMot", "RMot", "LBar", "RBar"]},
               "config_list_by_recording": {"X1__0": {"defined": True, "channels": ["LHip"],
                                                      "absent": ["RVis", "Nowhere", "LAud"]}}}
    cells = cells.assign(config_list=(cells["channel"] == "LHip").astype(int), config_list_defined=True)
    recordings = SD.recorded_channels(cells, records, PADDED)
    assert recordings == {"X1__0": {"standardized": CANON, "channels": CH, "listed": ["LHip", "LAud", "RVis"],
                                    "defined": True}}, "a listed name outside the standardized channels counts nothing"


@pytest.mark.parametrize("change, match", [
    (lambda r: r["recording_channels"].update(X1__0=CH[:4]), "X1__0: cells on channels outside its recording_channels"),
    (lambda r: r["recording_channels"].update(X1__0=[*CH, "Nowhere"]), "recording_channels names ['Nowhere']"),
    (lambda r: r["recording_channels"].update(X1__0=[*CH, "LMot"]), "not a list of distinct channel names"),
    (lambda r: r.pop("config_list_by_recording"), "provenance lacks ['config_list_by_recording']"),
    (lambda r: r["recording_channels"].clear(), "no recording_channels or config_list_by_recording entry for ['X1__0']"),
    (lambda r: r["config_list_by_recording"]["X1__0"].update(defined=True), "disagree with config_list_by_recording"),
    (lambda r: r["config_list_by_recording"]["X1__0"].update(absent=["LMot"]), "names ['LMot'] while no list applies"),
])
def test_recording_records_that_do_not_describe_the_cells_are_refused(change, match):
    cells = _cells(_planted_war())
    records = _records(cells)
    change(records)
    with pytest.raises(SystemExit, match=re.escape(match)):
        SD.member_masks(cells, PADDED, records)


def test_a_dataset_without_canonical_channels_is_refused():
    cells = _cells(_planted_war())
    with pytest.raises(SystemExit, match="no canonical channels for the datasets"):
        SD.member_masks(cells, {**SETTINGS, "canonical_channels": {"other": CH}}, _records(cells))


# The pipeline's own scripts


SCRIPTS = REPO / "workflow" / "scripts"
CHANNEL_CONFIG = {"manual": {"reject_channels": [], "min_valid_channels": 3},
                  "lof": {"reject_lof_threshold": 2.5, "reject_channels": [], "min_valid_channels": 3}}


def _run_script(name, src, dst, config, samples_config, **params):
    """Run ``workflow/scripts/<name>.py`` on the WAR in ``src``, writing ``dst``, with a stand-in snakemake object."""
    spec = importlib.util.spec_from_file_location(f"pipeline_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded by __name__, so nothing runs
    # The real setup points sys.stdout and sys.stderr at the rule's log files, which would break pytest's capture.
    module.setup_snakemake_logging = lambda _: logging.getLogger(f"pipeline_{name}")
    module.snakemake = SimpleNamespace(
        input=SimpleNamespace(war_parquet=str(src / "war.parquet"), war_json=str(src / "war.json")),
        output=SimpleNamespace(war_parquet=str(dst / "war.parquet")),
        params=SimpleNamespace(config=config, samples_config=samples_config, animal_folder="folder",
                               animal_id="X1", **params),
        wildcards=SimpleNamespace(animal="X1"))
    module.main()
    return dst


def _nan_grid(folder):
    war = WindowAnalysisResult.load_parquet_and_json(folder, filename="war")
    return np.isnan(extract_linear_array(war.result["psdtotal"])), list(war.channel_abbrevs)


def _pipeline_scripts(war, folder, standardized, bad_channels, fragment=FRAGMENT):
    """Run the workflow scripts on ``war`` and return their rejections on the standardized WAR.

    standardize_wars.py pads to ``standardized``; filter_wars_fragments.py gives F; the manual and LOF branches of
    filter_wars_channels.py on its output give today and lof_branch; each branch on the standardized WAR gives M
    and L.

    Returns:
        ``({"F", "today", "lof_branch", "M", "L": (W, C) bool}, standardized channel names)``.
    """
    config = {"analysis": {"standardization": {"channel_reorder": list(standardized), "use_abbrevs": True},
                           "fragment_filter_config": fragment, "channel_filter_config": CHANNEL_CONFIG}}
    samples = {"bad_channels": {"X1": copy.deepcopy(bad_channels)}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        war.save_parquet_and_json(folder / "war", filename="war", slugify_filename=False)
        std = _run_script("standardize_wars", folder / "war", folder / "std", config, samples)
        frag = _run_script("filter_wars_fragments", std, folder / "frag", config, samples)
        runs = {"F": frag}
        for name, src, filter_type in (("today", frag, "manual"), ("lof_branch", frag, "lof"), ("M", std, "manual"),
                                       ("L", std, "lof")):
            # Each run gets its own copy: the manual branch pops "_all" out of the dict it is given.
            runs[name] = _run_script("filter_wars_channels", src, folder / name, config,
                                     {"bad_channels": {"X1": copy.deepcopy(bad_channels)}}, filter_type=filter_type)
        before, channels = _nan_grid(std)
        return {name: _nan_grid(path)[0] & ~before for name, path in runs.items()}, channels


def _at_columns(cells, decisions, channels):
    col = {c: j for j, c in enumerate(channels)}
    return decisions[cells["row"].to_numpy(), cells["channel"].map(col).to_numpy()].astype(int)


def _scripted_war():
    """Two recordings of five channels in a ten-channel dataset: threshold ties, a window the window rule takes
    and one the padded channels keep, high_beta on one channel, and LOF flags that leave one recording whole and
    take the other."""
    rng = np.random.default_rng(7)
    rms = rng.uniform(90, 110, (24, 5))
    rms[0, 0] = 500.0                # a tie with max_rms, kept
    rms[1, 1:5] = 700.0              # four rejected: one kept cell and five padded channels keep the rest
    rms[13, 0] = 49.0                # low_rms
    beta = rng.uniform(0.05, 0.2, (24, 5))
    beta[5, 3] = 0.45                # high_beta takes the window
    lof = {DAYS[0]: [1.0, 2.5, 1.0, 1.0, 1.0],      # a tie with the threshold: RMot flagged
           DAYS[1]: [3.0, 3.0, 3.0, 3.0, 3.0]}      # every own channel flagged; five padded ones remain
    return _war(rms, beta, animaldays=[DAYS[0]] * 12 + [DAYS[1]] * 12, lof=lof)


@pytest.mark.mutates_constants
@pytest.mark.parametrize("entry, lists", [
    ({DAYS[0]: ["RBar", "RHip"], DAYS[1]: ["LMot", "LBar", "RAud", "LVis", "RVis", "LAud", "RHip", "LHip"]},
     {"X1__0": ["RBar", "RHip"], "X1__1": ["LMot", "LBar", "RAud", "LVis", "RVis", "LAud", "RHip", "LHip"]}),
    ({"_all": ["RMot", "RHip"]}, {"X1__0": ["RMot", "RHip"], "X1__1": ["RMot", "RHip"]}),
    ({}, {}),
])
def test_members_and_branches_match_the_workflow_scripts(tmp_path, entry, lists):
    """F, L, M, today (F or M) and lof_branch (F or L) equal what the workflow scripts reject, cell by cell, on a
    padded WAR, with the per-recording lists the extraction resolves: per session, list-form ("_all") and none."""
    war = _scripted_war()
    branches, channels = _pipeline_scripts(war, tmp_path, CANON, entry)
    assert channels == CANON
    full = _cells(war)
    cells = full[~((full["recording"] == "X1__0") & (full["channel"] == "LBar"))]  # no labeled cell on LBar
    records = _records(cells, own={"X1__0": CH, "X1__1": CH}, lists=lists)
    cells = _with_lists(cells, records)
    members, audits = SD.member_masks(cells, PADDED, records)
    preds = SD.combination_preds(members, np.zeros(len(cells), int))
    for name, ours in (("F", members["F"]), ("L", members["L"]), ("M", members["M"]), ("today", preds["today"]),
                       ("lof_branch", preds["lof_branch"])):
        assert ours.tolist() == _at_columns(cells, branches[name], channels).tolist(), name
    row = (cells["row"] == 1).to_numpy()
    assert cells.loc[row, "channel"].tolist() == ["LMot", "RMot", "RBar", "LHip"]
    assert members["F"][row].tolist() == [0, 1, 1, 1], "LMot and five padded channels keep the window"
    assert members["F"][(cells["row"] == 5).to_numpy()].all() and members["L"][(cells["recording"] == "X1__1")].all()
    assert audits["channels_without_cells"] == {"X1__0": ["LBar"]}
    if entry:
        assert members["M"].any() and preds["today"].sum() > members["F"].sum()


@pytest.mark.mutates_constants
def test_m_matches_the_workflow_script_at_the_recording_rule(tmp_path):
    """Lists that leave two standardized channels and three: the first recording goes whole, the second keeps its
    unlisted channels; a listed channel the recording lacks counts, a name outside the standardized ones does
    not."""
    war = _war(np.full((20, 5), 100.0), animaldays=[DAYS[0]] * 10 + [DAYS[1]] * 10)
    entry = {DAYS[0]: ["LMot", "RMot", "LBar", "RHip", "LAud", "RAud", "LVis", "RVis"],
             DAYS[1]: ["LMot", "RMot", "RHip", "LAud", "RAud", "LVis", "RVis"]}
    branches, channels = _pipeline_scripts(war, tmp_path, CANON, entry)
    cells = _cells(war)
    records = _records(cells, lists={"X1__0": entry[DAYS[0]], "X1__1": entry[DAYS[1]]})
    records["config_list_by_recording"]["X1__1"]["absent"].append("Nowhere")  # the pipeline would refuse to resolve it
    cells = _with_lists(cells, records)
    m, audit = SD.list_reject(cells, PADDED, SD.recorded_channels(cells, records, PADDED))
    assert m.tolist() == _at_columns(cells, branches["M"], channels).tolist()
    assert m[(cells["recording"] == "X1__0").to_numpy()].all()
    assert m[(cells["recording"] == "X1__1").to_numpy()].reshape(10, 5).sum(axis=0).tolist() == [10, 10, 0, 0, 0]
    assert audit["n_recording_rule_only"] == 2 * 10


def test_the_window_rule_count_is_the_one_the_workflow_passes():
    src = (REPO / "workflow/scripts/filter_wars_fragments.py").read_text()
    calls = re.findall(r"apply_filters\(filter_config=fragment_filter_config, min_valid_channels=(\d+)\)", src)
    assert calls == [str(SD.FRAGMENT_MIN_VALID_CHANNELS)]


# The consensus rules


def test_category_votes_are_scoring_label_map():
    assert SD.CATEGORY_VOTE.keys() == {category for _, category in LABEL_MAP.values()}
    for vote, category in LABEL_MAP.values():
        assert SD.CATEGORY_VOTE[category] == vote or (np.isnan(vote) and np.isnan(SD.CATEGORY_VOTE[category]))
    assert SD.CATEGORY_VOTE["artifact"] == 1 and SD.CATEGORY_VOTE["event"] == 0 and np.isnan(SD.CATEGORY_VOTE["unsure"])


def _rater_table(seed=0, n_cells=400, raters=("A", "B", "C")):
    """A long rater table as scoring.unblind returns it: every category, blanks, and raters with no row for a cell."""
    rng = np.random.default_rng(seed)
    categories = ["artifact", "clean", "event", "unsure", "unlabeled"]
    rows = []
    for i in range(n_cells):
        for rater in raters:
            if rng.random() < 0.05:
                continue  # this rater has no row for the cell
            category = categories[rng.choice(5, p=[0.3, 0.3, 0.15, 0.15, 0.1])]
            vote = {"artifact": 1, "clean": 0, "event": 0}.get(category, np.nan)
            rows.append({"recording": f"r{i % 7}__0", "window": i // 7, "channel": "LMot" if i % 2 else "RMot",
                         "t_start_s": 5.0 * (i // 7), "rater": rater, "y": vote, "category": category})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rule_columns_equal_scoring_consensus(seed):
    """The cells as the extraction writes them (scoring.consensus under majority, with each rater's category);
    each rule column recomputed from the categories equals scoring.consensus under that rule."""
    long = _rater_table(seed)
    cells = consensus(long, rule="majority").rename(columns={"y_true": "y"})
    labels = SD.rule_labels(cells)
    assert list(labels) == ["majority", "any", "unanimous"]
    keys = ["recording", "window", "channel"]
    for rule in labels:
        expected = cells[keys].merge(consensus(long, rule=rule)[keys + ["y_true"]], on=keys, how="left")["y_true"]
        assert labels[rule].tolist() == expected.astype(int).tolist(), rule
    assert labels["majority"].tolist() == cells["y"].tolist()
    assert (labels["any"] >= labels["majority"]).all() and (labels["majority"] >= labels["unanimous"]).all()
    assert (labels["any"] != labels["majority"]).any() and (labels["unanimous"] != labels["majority"]).any()


def test_rule_columns_refuse_what_they_cannot_vote_on():
    long = _rater_table()
    cells = consensus(long, rule="majority")
    with pytest.raises(SystemExit, match="no rater category columns"):
        SD.rule_labels(cells.drop(columns=[c for c in cells.columns if c.startswith("cat_")]))
    with pytest.raises(SystemExit, match=re.escape("rater categories ['maybe']")):
        SD.rule_labels(cells.assign(cat_A="maybe"))
    with pytest.raises(SystemExit, match="cells hold no rater vote"):
        SD.rule_labels(cells.assign(cat_A="unsure", cat_B="unlabeled", cat_C=None))
    with pytest.raises(SystemExit, match="more than once"):
        SD.rule_labels(pd.concat([cells, cells.head(1)]))


def test_a_y_other_than_the_majority_is_refused():
    cells = consensus(_rater_table(), rule="majority").rename(columns={"y_true": "y"})
    labels = SD.rule_labels(cells)
    SD.require_majority(cells, labels, {})
    any_rule = cells.assign(y=labels["any"])
    with pytest.raises(SystemExit, match="extraction recorded rule 'any'"):
        SD.require_majority(any_rule, labels, {"params": {"rule": "any"}})


# Settings


def _write_config(path, **changes):
    config = {"analysis": {"fragment_filter_config": json.loads(json.dumps(FRAGMENT)),
                           "channel_filter_config": {
                               "manual": {"reject_channels": [], "min_valid_channels": 3},
                               "lof": {"reject_lof_threshold": 2.5, "reject_channels": [], "min_valid_channels": 3}}}}
    for key, value in changes.items():
        section, _, name = key.partition("__")
        target = config["analysis"]["fragment_filter_config" if section == "fragment" else "channel_filter_config"]
        target[name] = value
    path.write_text(yaml.safe_dump(config))
    return path


def _write_dataset(folder, name, channels, **analysis):
    """A dataset config in ``folder/datasets`` with an inline samples channel map and optional analysis keys."""
    (folder / "datasets").mkdir(exist_ok=True)
    config = {"samples_data": {"channels": {c: [f"raw {c}"] for c in channels}, "animals": []}}
    if analysis:
        config["analysis"] = analysis
    (folder / "datasets" / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def test_settings_are_read_from_the_pipeline_config():
    raw = yaml.safe_load((REPO / "config/config.yaml").read_text())["analysis"]
    s = SD.load_settings()
    params = SD.FRAGMENT_FILTER_PARAMS
    assert s["fragment_filters"] == {name: {params[name]: v[params[name]]}
                                     for name, v in raw["fragment_filter_config"].items()}
    assert s["lof"] == {k: raw["channel_filter_config"]["lof"][k] for k in ("reject_lof_threshold", "min_valid_channels")}
    assert s["config_lists"] == {"min_valid_channels": raw["channel_filter_config"]["manual"]["min_valid_channels"]}
    assert s["config"] == {"path": "config/config.yaml", "sha256": SD._sha256(REPO / "config/config.yaml")}
    assert s["canonical_channels"] == {}
    sox5 = yaml.safe_load((REPO / "config/datasets/sox5_bin.yaml").read_text())["samples_data"]["channels"]
    assert SD.load_settings(datasets=["sox5_bin"])["canonical_channels"] == {"sox5_bin": list(sox5)}


def test_canonical_channels_follow_the_standardization_target(tmp_path):
    """The samples channel map's keys in order, unless the merged config sets standardization.channel_reorder;
    the dataset config is merged over config.local.yaml, which is merged over config.yaml."""
    path = _write_config(tmp_path / "config.yaml")
    _write_dataset(tmp_path, "ds1", ["RVis", "LMot", "RMot", "LBar"])
    assert SD.canonical_channels(path, "ds1") == ["RVis", "LMot", "RMot", "LBar"]
    config = yaml.safe_load(path.read_text())
    config["analysis"]["standardization"] = {"channel_reorder": ["LMot", "RMot"], "use_abbrevs": True}
    path.write_text(yaml.safe_dump(config))
    assert SD.canonical_channels(path, "ds1") == ["LMot", "RMot"]
    (tmp_path / "config.local.yaml").write_text(yaml.safe_dump(
        {"analysis": {"standardization": {"channel_reorder": ["LBar", "LMot", "RMot"]}}}))
    assert SD.canonical_channels(path, "ds1") == ["LBar", "LMot", "RMot"]
    _write_dataset(tmp_path, "ds1", ["RVis", "LMot"], standardization={"channel_reorder": ["RVis"]})
    assert SD.canonical_channels(path, "ds1") == ["RVis"]
    assert SD.load_settings(path, datasets=["ds1"])["canonical_channels"] == {"ds1": ["RVis"]}
    with pytest.raises(SystemExit, match="cannot read the channels of dataset ds2"):
        SD.load_settings(path, datasets=["ds2"])
    (tmp_path / "datasets" / "ds3.yaml").write_text(yaml.safe_dump({"samples_data": {"animals": []}}))
    (tmp_path / "config.local.yaml").unlink()
    config["analysis"]["standardization"] = {"use_abbrevs": True}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit, match="ds3 names no channels"):
        SD.canonical_channels(path, "ds3")


def test_settings_take_the_filter_defaults_and_refuse_what_f_cannot_reproduce(tmp_path):
    path = _write_config(tmp_path / "config.yaml", fragment__high_rms={}, fragment__logrms_range=None)
    s = SD.load_settings(path)
    assert s["fragment_filters"]["high_rms"] == {"max_rms": 500} and s["fragment_filters"]["logrms_range"] == {"z_range": 3}
    _write_config(path, fragment__morphological_smoothing={"smoothing_seconds": 300})
    with pytest.raises(SystemExit, match="cannot reproduce"):
        SD.load_settings(path)
    _write_config(path, channel__lof={"reject_lof_threshold": 2.5, "reject_channels": ["LMot"], "min_valid_channels": 3})
    with pytest.raises(SystemExit, match="global channel list"):
        SD.load_settings(path)
    _write_config(path, channel__lof={"reject_channels": [], "min_valid_channels": 3})
    with pytest.raises(SystemExit, match="lacks"):
        SD.load_settings(path)


def test_settings_refuse_overrides_the_pipeline_would_merge(tmp_path):
    path = _write_config(tmp_path / "config.yaml")
    _write_dataset(tmp_path, "ds1", CH, war_generation={})
    assert SD.load_settings(path, datasets=["ds1"])["lof"]["reject_lof_threshold"] == 2.5
    _write_dataset(tmp_path, "ds1", CH, fragment_filter_config={"high_rms": {"max_rms": 900}})
    with pytest.raises(SystemExit, match="ds1.yaml overrides analysis.fragment_filter_config"):
        SD.load_settings(path, datasets=["ds1"])
    (tmp_path / "config.local.yaml").write_text(yaml.safe_dump({"analysis": {"channel_filter_config": {}}}))
    with pytest.raises(SystemExit, match="config.local.yaml overrides"):
        SD.load_settings(path)


def test_settings_differences_name_the_changed_setting(tmp_path):
    _write_dataset(tmp_path, "ds1", CH)
    recorded = SD.load_settings(_write_config(tmp_path / "a.yaml"), datasets=["ds1"])
    same = SD.load_settings(_write_config(tmp_path / "b.yaml"), datasets=["ds1"])
    assert SD.settings_differences(recorded, same) == [], "path and sha256 are not settings"
    moved = SD.load_settings(_write_config(tmp_path / "c.yaml", fragment__high_beta={"max_beta_prop": 0.5}),
                             datasets=["ds1"])
    [diff] = SD.settings_differences(recorded, moved)
    assert diff.startswith("fragment_filters")
    _write_dataset(tmp_path, "ds1", CANON)
    [diff] = SD.settings_differences(recorded, SD.load_settings(tmp_path / "a.yaml", datasets=["ds1"]))
    assert diff.startswith("canonical_channels")
    roundtrip = json.loads(json.dumps(recorded))
    assert SD.settings_differences(roundtrip, recorded) == []


def test_settings_read_the_datasets_the_extraction_scanned():
    cells = pd.DataFrame({"dataset": ["ds2", "ds1"]})
    assert SD.settings_datasets(cells, {"params": {"datasets": ["ds3", "ds1"]}}) == ["ds1", "ds2", "ds3"]
    assert SD.settings_datasets(cells, {}) == ["ds1", "ds2"]


def test_config_problem_compares_the_config_the_extraction_recorded(tmp_path):
    path = _write_config(tmp_path / "config.yaml")
    prov = {"inputs": {"config/config.yaml": {"path": "config/config.yaml", "sha256": SD._sha256(path)}}}
    assert SD.config_problem(path, prov, "dev") is None
    path.write_text(path.read_text() + "# edited\n")
    assert "the dev extraction recorded" in SD.config_problem(path, prov, "dev")
    assert "recorded None" in SD.config_problem(path, {}, "dev")
    assert "not found" in SD.config_problem(tmp_path / "missing.yaml", prov, "dev")


def _recorded(folder, *rels):
    """A provenance recording ``config/<rel>`` with the sha256 of ``folder/<rel>`` for each rel, as the extraction
    records the config files, beside an input outside config/."""
    inputs = {f"config/{rel}": {"path": f"config/{rel}", "sha256": SD._sha256(folder / rel)} for rel in rels}
    return {"inputs": {**inputs, "keymap": {"path": "keymap.csv", "sha256": "not a config file"}}}


def test_config_problem_compares_every_config_file_beside_the_config(tmp_path):
    """The dataset configs, config.local.yaml and samples files the extraction recorded under config/ are compared
    with the files beside --config; a recorded file now gone, and a file the settings read without a record, are
    named too. Inputs outside config/ are not config files."""
    path = _write_config(tmp_path / "config.yaml")
    for name in ("ds1", "ds2", "ds3"):
        _write_dataset(tmp_path, name, CH)
    samples = tmp_path / "samples" / "ds2.json"
    samples.parent.mkdir()
    samples.write_text("{}")
    prov = _recorded(tmp_path, "config.yaml", "datasets/ds1.yaml", "datasets/ds2.yaml", "samples/ds2.json")
    assert SD.config_problem(path, prov, "dev", ["ds1", "ds2"]) is None
    assert SD.config_problem(path, prov, "dev") is None, "a recorded dataset config is compared when no cell reads it"

    ds1 = tmp_path / "datasets" / "ds1.yaml"
    recorded = prov["inputs"]["config/datasets/ds1.yaml"]["sha256"]
    _write_dataset(tmp_path, "ds1", CANON)
    samples.write_text('{"edited": true}')
    problem = SD.config_problem(path, prov, "test", ["ds1", "ds2"])
    assert problem.startswith("the config files differ from those the test features were extracted under")
    assert f"{ds1} has sha256 {SD._sha256(ds1)}, and the test extraction recorded '{recorded}' for " \
           "config/datasets/ds1.yaml" in problem
    assert f"{samples} has sha256" in problem and "for config/samples/ds2.json" in problem
    assert "ds2.yaml" not in problem and "keymap" not in problem
    _write_dataset(tmp_path, "ds1", CH)
    samples.write_text("{}")

    (tmp_path / "datasets" / "ds2.yaml").unlink()
    assert f"{tmp_path / 'datasets' / 'ds2.yaml'} is missing, and the dev extraction recorded " in \
        SD.config_problem(path, prov, "dev", ["ds1"])
    _write_dataset(tmp_path, "ds2", CH)
    assert SD.config_problem(path, prov, "dev", ["ds1", "ds2"]) is None, "the same bytes again"

    assert SD.config_problem(path, prov, "dev", ["ds1", "ds3"]).endswith(
        f"{tmp_path / 'datasets' / 'ds3.yaml'} is read for the settings, and the dev extraction recorded None for "
        "config/datasets/ds3.yaml")
    local = tmp_path / "config.local.yaml"
    local.write_text(yaml.safe_dump({"note": "settings untouched"}))
    assert "recorded None for config/config.local.yaml" in SD.config_problem(path, prov, "dev", ["ds1"])
    assert SD.config_problem(path, _recorded(tmp_path, "config.yaml", "config.local.yaml", "datasets/ds1.yaml",
                                             "datasets/ds2.yaml", "samples/ds2.json"), "dev", ["ds1"]) is None

    other = tmp_path / "pipeline.yaml"  # config/config.yaml is --config itself, whatever its name
    other.write_text(path.read_text())
    local.unlink()
    assert SD.config_problem(other, prov, "dev", ["ds1", "ds2"]) is None
    assert SD.config_input_path(other, "config/config.yaml") == other
    assert SD.config_input_path(other, "config/datasets/ds1.yaml") == ds1


# Columns


def _member_frame(**overrides):
    df = pd.DataFrame({"recording": "r__0", "window": [0, 0], "channel": ["LMot", "RMot"], "dataset": "ds",
                       "rms": 100.0, "logrmsz_abs": 0.1, "beta_wmax": 0.1, "lof": 1.0, "config_list": [0, 1],
                       "config_list_defined": [True, True]})
    return df.assign(**overrides)


@pytest.mark.parametrize("overrides, match", [
    ({"config_list": [0, 2]}, "other than 0 and 1"),
    ({"config_list": [0, np.nan]}, "other than 0 and 1"),
    ({"config_list_defined": [True, None]}, "other than true and false"),
    ({"config_list_defined": [True, False]}, "config_list_defined is false"),
    ({"channel": ["LMot", "LMot"]}, "more than once"),
])
def test_member_columns_are_checked(overrides, match):
    SD.require_member_columns(_member_frame())
    with pytest.raises(SystemExit, match=match):
        SD.require_member_columns(_member_frame(**overrides))
    with pytest.raises(SystemExit, match="lack"):
        SD.require_member_columns(_member_frame().drop(columns="config_list_defined"))


def test_config_list_columns_are_metadata_and_never_ranked():
    df = _member_frame().assign(y=0, **{name: 0 for name in GRID})
    assert SD.classify_columns(df) == GRID
    with pytest.raises(SystemExit, match="unknown role"):
        SD.classify_columns(df.assign(config_list_extra=0))


# Combinations, selection, nested runs


def test_combinations_reject_where_any_member_rejects():
    members = {"F": np.array([1, 0, 0, 0, 0]), "L": np.array([0, 1, 0, 0, 0]), "M": np.array([0, 0, 1, 0, 0])}
    preds = SD.combination_preds(members, np.array([0, 0, 0, 1, 0]))
    assert {k: v.tolist() for k, v in preds.items()} == {
        "today": [1, 0, 1, 0, 0], "replace": [1, 0, 0, 1, 0], "add": [1, 0, 1, 1, 0], "lof_branch": [1, 1, 0, 0, 0],
        "autoreject_alone": [0, 0, 0, 1, 0], "filters_alone": [1, 0, 0, 0, 0], "config_lists_alone": [0, 0, 1, 0, 0]}
    assert list(preds) == ["today", "replace", "add", "lof_branch", "autoreject_alone", "filters_alone",
                           "config_lists_alone"]


def _grid_frame(y, columns):
    df = pd.DataFrame({"y": y})
    for name in GRID:
        df[name] = columns.get(name, np.zeros(len(y), int))
    return df


def test_choose_config_compares_four_decimals_and_breaks_ties_in_grid_order():
    y = np.r_[np.ones(10000, int), np.zeros(10001, int)]
    one_fp = y.copy()
    one_fp[-1] = 1                                   # F1 20000/20001, 1.0 to 4 decimals
    df = _grid_frame(y, {GRID[3]: one_fp, GRID[7]: y})
    assert SD.choose_config(df, GRID, np.zeros(len(y), int)) == (GRID[3], pytest.approx(20000 / 20001), True)
    two_fp = one_fp.copy()
    two_fp[-2] = 1                                   # F1 0.9999 to 4 decimals
    df = _grid_frame(y, {GRID[3]: two_fp, GRID[7]: y})
    assert SD.choose_config(df, GRID, np.zeros(len(y), int)) == (GRID[7], 1.0, False)


def test_choose_config_scores_replace_with_f():
    y = np.array([1, 1, 0, 0, 0, 0])
    f = np.array([1, 0, 0, 0, 0, 0])
    df = _grid_frame(y, {GRID[0]: np.array([1, 0, 1, 1, 1, 1]), GRID[1]: np.array([0, 1, 0, 0, 0, 0])})
    k, f1, tie = SD.choose_config(df, GRID, f)
    assert (k, f1, tie) == (GRID[1], 1.0, False), "F covers the first artifact, so the second completes replace"


def _cohort(seed=0):
    """Four groups in two datasets; GRID[0] predicts y in g1 and g2, GRID[1] in g3 and g4, the rest is noise."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(["g1", "g2", "g3", "g4"], 50)
    df = pd.DataFrame({"y": (rng.random(200) < 0.3).astype(int), "cv_group": groups,
                       "dataset": np.where(np.isin(groups, ["g1", "g2"]), "ds1", "ds2"),
                       "animal": np.char.add("a", groups), "isday": np.tile([True, False], 100),
                       "genotype": np.where(np.isin(groups, ["g1", "g3"]), "WT", "KO"),
                       "config_list": 0, "config_list_defined": np.isin(groups, ["g2", "g3"])})
    noise = (rng.random(200) < 0.5).astype(int)
    first = np.isin(groups, ["g1", "g2"])
    columns = {GRID[0]: np.where(first, df["y"], noise), GRID[1]: np.where(first, noise, df["y"])}
    for name in GRID:
        df[name] = columns.get(name, (rng.random(200) < 0.3).astype(int))
    return df


@pytest.mark.parametrize("outer", ["cv_group", "dataset"])
def test_nested_replace_chooses_k_inside_each_training_part(outer):
    df = _cohort()
    f = (np.random.default_rng(3).random(len(df)) < 0.05).astype(int)
    pred, folds = SD.nested_replace(df, GRID, f, outer)
    assert [fold["held_out"] for fold in folds] == sorted(df[outer].unique())
    for fold in folds:
        held = (df[outer] == fold["held_out"]).to_numpy()
        k, train_f1, tie = SD.choose_config(df[~held], GRID, f[~held])
        assert (fold["autoreject_config"], fold["train_replace_f1"], fold["tie_broken"]) == (k, round(train_f1, 4), tie)
        assert pred[held].tolist() == (f[held] | SD.mask_pred(df[held], k)).tolist()
        assert fold["n"] == held.sum() and fold["tp"] + fold["fn"] == df["y"][held].sum()
    if outer == "cv_group":
        assert {fold["autoreject_config"] for fold in folds} == {GRID[0], GRID[1]}


def test_too_few_names_what_leaving_one_out_lacks():
    df = _cohort()
    assert SD.too_few(df, "cv_group") is None and SD.too_few(df, "dataset", "cv_group") is None
    assert SD.too_few(df.assign(dataset="ds1"), "dataset") == \
        "leaving one dataset out needs at least 2, and the cells hold 1"
    two = df[df["cv_group"].isin(["g1", "g3"])]  # one group in each dataset
    assert SD.too_few(two, "cv_group") is None and SD.too_few(two, "dataset") is None
    assert SD.too_few(two, "cv_group", "cv_group") == ("ranking by leaving one cv_group out needs at least 2 in "
                                                       "each training part, and the part without cv_group g1 holds 1")
    assert SD.too_few(two, "dataset", "cv_group").endswith("the part without dataset ds1 holds 1")


def _scored(df, seed=5):
    """``df`` with random filter and LOF scores, for the diagnostic ranking."""
    rng = np.random.default_rng(seed)
    n = len(df)
    return df.assign(rms=rng.uniform(40, 600, n), logrmsz_abs=rng.uniform(0, 4, n), beta_wmax=rng.uniform(0, 0.6, n),
                     lof=rng.uniform(0.8, 3.0, n))


def test_diagnostic_records_the_runs_the_cells_cannot_support():
    """One dataset: no leave-one-dataset-out; two groups: no ranking inside a training part; one group: no
    ranking at all. Each is recorded with its reason instead of stopping the selection."""
    full = SD.diagnostic(_scored(_cohort()), GRID, boot=50)
    assert set(full["nested_logo"]) == {"f1", "ci95", "pooled", "folds"} and len(full["nested_logo"]["folds"]) == 4
    assert set(full["nested_lodo"]) == {"f1", "pooled", "folds", "per_dataset"} and "not_run" not in full

    one_dataset = SD.diagnostic(_scored(_cohort().assign(dataset="ds1")), GRID, boot=50)
    assert len(one_dataset["order"]) == 43 and len(one_dataset["nested_logo"]["folds"]) == 4
    assert np.isnan(one_dataset["leaderboard"]["high_rms"]["lodo_f1"])
    assert one_dataset["nested_lodo"]["not_run"] == "leaving one dataset out needs at least 2, and the cells hold 1"
    assert np.isnan(one_dataset["nested_lodo"]["f1"]) and one_dataset["nested_lodo"]["folds"] == []

    df = _cohort()
    two_groups = SD.diagnostic(_scored(df[df["cv_group"].isin(["g1", "g3"])]), GRID, boot=50)
    assert len(two_groups["order"]) == 43
    assert two_groups["nested_logo"]["not_run"].endswith("the part without cv_group g1 holds 1")
    assert two_groups["nested_lodo"]["not_run"].endswith("the part without dataset ds1 holds 1")

    one_group = SD.diagnostic(_scored(df[df["cv_group"] == "g1"]), GRID, boot=50)
    assert one_group["not_run"] == "leaving one cv_group out needs at least 2, and the cells hold 1"
    assert one_group["order"] == [] and one_group["leaderboard"] == {} and one_group["ranked_first"] is None


def test_metrics_agree_with_sklearn():
    rng = np.random.default_rng(4)
    y, yhat = (rng.random(300) < 0.3).astype(int), (rng.random(300) < 0.4).astype(int)
    m = SD.metrics(y, yhat)
    assert m["precision"] == pytest.approx(precision_score(y, yhat))
    assert m["recall"] == pytest.approx(recall_score(y, yhat))
    assert m["f1"] == pytest.approx(f1_score(y, yhat))
    assert m["cohen_kappa"] == pytest.approx(cohen_kappa_score(y, yhat))
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n"] == 300 and m["flag_rate"] == pytest.approx(yhat.mean())
    none = SD.metrics(np.zeros(5, int), np.zeros(5, int))
    assert np.isnan(none["precision"]) and np.isnan(none["recall"]) and np.isnan(none["f1"])
    assert np.isnan(none["cohen_kappa"])


def _members(df):
    """F flags nothing, M flags the clean cells of g1 and L those of g2: today and lof_branch each fall short of
    replace in a group."""
    zero = np.zeros(len(df), int)
    clean = (df["y"] == 0).to_numpy()
    return {"F": zero, "M": (clean & (df["cv_group"] == "g1").to_numpy()).astype(int),
            "L": (clean & (df["cv_group"] == "g2").to_numpy()).astype(int)}


def _alongside(df, seed=6):
    """Labels under two other rules: any adds positives to y, unanimous removes some."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    return {"any": np.where(rng.random(len(y)) < 0.2, 1, y), "unanimous": np.where(rng.random(len(y)) < 0.3, 0, y)}


def test_score_combinations_reports_every_breakdown_and_the_paired_differences():
    df = _cohort()
    preds = SD.combination_preds(_members(df), df["y"].to_numpy())
    combos = SD.score_combinations(df, preds, boot=200)
    assert list(combos) == list(SD.COMBINATIONS)
    for name, entry in combos.items():
        assert set(entry) >= {"members", "definition", "cells", "rule", "pooled", "ci95", *SD.BREAKDOWNS}, name
        assert entry["rule"] == "majority" and "alongside" not in entry
        assert set(entry["per_cv_group"]) <= {"g1", "g2", "g3", "g4"}
    assert combos["replace"]["definition"] == "F or A(k)" and combos["today"]["definition"] == "F or M"
    assert combos["lof_branch"]["definition"] == "F or L" and combos["add"]["definition"] == "F or M or A(k)"
    assert set(combos["today"]["per_cv_group"]) == {"g1", "g2", "g3", "g4"}
    assert set(combos["today"]["per_isday"]) == {"True", "False"}
    assert set(combos["today"]["per_genotype"]) == {"WT", "KO"}
    assert set(combos["today"]["per_animal"]) == {"ag1", "ag2", "ag3", "ag4"}
    assert combos["today"]["pooled"]["fp"] == int(_members(df)["M"].sum())
    assert combos["lof_branch"]["pooled"]["fp"] == int(_members(df)["L"].sum())
    lists = combos["config_lists_alone"]
    assert lists["cells"] == "config_list_defined" and lists["pooled"]["n"] == 100
    assert set(lists["per_cv_group"]) == {"g2", "g3"}
    d = combos["replace"]["minus_today"]
    assert d["f1_difference"] == pytest.approx(1.0 - f1_score(df["y"], preds["today"]))
    assert d["ci95"][0] >= 0, "replace is perfect, so no resample favors today"
    assert "minus_today" in combos["add"] and "minus_today" not in combos["today"]
    assert "minus_today" not in combos["lof_branch"]


def test_score_combinations_scores_every_rule_alongside():
    """The alongside entries hold the same metrics, intervals, breakdowns and paired differences as the primary
    ones, each computed against its own rule's labels; the primary metrics stay those of y."""
    df = _cohort()
    preds = SD.combination_preds(_members(df), df["y"].to_numpy())
    others = _alongside(df)
    combos = SD.score_combinations(df, preds, boot=200, alongside=others)
    alone = SD.score_combinations(df, preds, boot=200)
    for name, entry in combos.items():
        assert set(entry["alongside"]) == {"any", "unanimous"}
        assert entry["pooled"] == alone[name]["pooled"] and entry["ci95"] == alone[name]["ci95"]
        for rule, y in others.items():
            scored = entry["alongside"][rule]
            c = SD.combination_cells(df)[name]
            assert scored["pooled"] == SD.metrics(y[c], preds[name][c]), (name, rule)
            assert scored["ci95"] == list(SD.cluster_boot_ci(y[c], preds[name][c], df["cv_group"].to_numpy()[c], n=200))
            assert scored["per_dataset"] == SD.breakdown(df[c], preds[name][c], "dataset", y=y[c])
            assert set(scored) >= {"pooled", "ci95", *SD.BREAKDOWNS}
    groups = df["cv_group"].to_numpy()
    for rule, y in {"majority": df["y"].to_numpy(), **others}.items():
        for name in SD.PAIRED:
            d = (combos[name] if rule == "majority" else combos[name]["alongside"][rule])["minus_today"]
            assert d["f1_difference"] == pytest.approx(f1_score(y, preds[name]) - f1_score(y, preds["today"])), \
                (name, rule)
            assert d["ci95"] == list(SD.paired_delta_ci(y, preds[name], preds["today"], groups, n=200)), (name, rule)
    for rule in others:
        assert combos["replace"]["alongside"][rule]["pooled"]["f1"] != combos["replace"]["pooled"]["f1"]
        for name in SD.PAIRED:
            assert combos[name]["alongside"][rule]["minus_today"]["ci95"] != combos[name]["minus_today"]["ci95"], \
                "each rule's interval is its own, so one taken from majority labels differs"


def test_paired_difference_of_identical_combinations_is_zero():
    df = _cohort()
    members = {"F": df["y"].to_numpy(), "L": np.zeros(len(df), int), "M": np.zeros(len(df), int)}
    combos = SD.score_combinations(df, SD.combination_preds(members, np.zeros(len(df), int)), boot=200)
    d = combos["replace"]["minus_today"]
    assert d["f1_difference"] == 0 and d["ci95"][0] <= 0 <= d["ci95"][1]


def test_intervals_need_two_groups():
    y = np.array([0, 1, 1, 0])
    assert all(np.isnan(SD.cluster_boot_ci(y, y, np.array(["g"] * 4), n=50)))
