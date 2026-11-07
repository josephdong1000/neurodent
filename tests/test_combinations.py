"""Tests for the combination evaluation in ``scripts/labeling/select_detectors.py``.

F, L and M are checked cell by cell against the pipeline's own filter code, ``WindowAnalysisResult.apply_filters``
on a synthetic WAR, with the fragment filters as ``workflow/scripts/filter_wars_fragments.py`` passes them and
the LOF and manual branches as ``filter_wars_channels.py`` builds them: threshold ties, missing scores,
high_beta's window rule, the minimum valid channel rules and the unit each acts on, also on a WAR padded to its
dataset's canonical channels as ``standardize_wars.py`` pads it. The per-cell columns are built with
``extract_dev_features``' helpers. Also covered: the settings and canonical channels read from the config, the
combination masks, the selection and its tie rule, the nested runs, the metrics, the paired intervals and the
breakdowns, and the nested runs a partition with one dataset or one cv_group cannot support. Labels are
synthetic.
"""

import copy
import json
import re
import sys
import warnings
from pathlib import Path

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

pytestmark = pytest.mark.unit

GRID = adr.grid_config_names()
CH = ["LMot", "RMot", "LBar", "RBar", "LHip"]
# Ten canonical channels, of which the synthetic recordings hold the five in CH (PADDED's dataset).
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

    A channel whose psdtotal is 0 gets a beta band of 0, so its proportion is 0/0.
    """
    n_rows, n_ch = rms.shape
    beta = np.full(rms.shape, 0.1) if beta is None else beta
    total = np.ones(rms.shape) if total is None else total
    animaldays = animaldays or [DAYS[0]] * n_rows
    band = np.where(total == 0, 0.0, beta * total)
    rows = []
    for i, day in enumerate(animaldays):
        rows.append({"animal": "X1", "animalday": day, "genotype": "WT", "duration": 5.0, "isday": True,
                     "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(days=DAYS.index(day), seconds=5 * i),
                     "rms": rms[i].tolist(), "psdtotal": total[i].tolist(),
                     "psdband": {b: (band[i] if b == "beta" else np.full(n_ch, 0.1)).tolist() for b in BANDS}})
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
    f, _ = SD.fragment_reject(cells, _settings(**{name: FRAGMENT[name]}))
    assert f.tolist() == _at(cells, _pipeline(war, {name: FRAGMENT[name]}), war).tolist()
    for row, ch in cells_rejected:
        assert f[_cell(cells, row, ch)].tolist() == [1], (row, ch)
    for row, ch in cells_kept:
        assert f[_cell(cells, row, ch)].tolist() == [0], (row, ch)


def test_fragment_filters_together_follow_apply_filters_with_the_window_rule():
    war = _planted_war()
    cells = _cells(war)
    f, audit = SD.fragment_reject(cells, SETTINGS)
    assert f.tolist() == _at(cells, _pipeline(war, FRAGMENT), war).tolist()
    assert f[(cells["row"] == 4).to_numpy()].tolist() == [1] * 5, "two cells left of five: every channel goes"
    assert f[(cells["row"] == 9).to_numpy()].tolist() == [1, 1, 0, 0, 0], "three left of five are enough"
    assert audit["n_window_rule_only"] == 2 and audit["n_window_rule_uncertain"] == 0
    assert audit["n_windows_incomplete"] == 0


@pytest.mark.parametrize("row, channel, uncertain, expected", [
    (4, "LMot", 0, [1, 1, 1, 1]),  # one kept cell left; two even if the missing one was kept
    (9, "LHip", 2, [1, 1, 1, 1]),  # two kept cells left; three, enough, if the missing one was kept
])
def test_window_rule_counts_the_cells_in_the_features(row, channel, uncertain, expected):
    """A window missing a cell (every rater abstained) is counted over the cells it has; the audit names the cells
    whose rejection would go if the missing cell was kept."""
    cells = _cells(_planted_war())
    kept = cells[~_cell(cells, row, channel)]
    f, audit = SD.fragment_reject(kept, SETTINGS)
    assert audit["n_windows_incomplete"] == 1 and audit["n_window_rule_uncertain"] == uncertain
    assert f[(kept["row"] == row).to_numpy()].tolist() == expected


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
    assert SD.fragment_reject(nan_aware, settings)[0].tolist() == _at(nan_aware, pipeline, war).tolist()
    plain = _cells(war, window_max=np.max)
    assert not SD.fragment_reject(plain, settings)[0][(plain["row"] == 2).to_numpy()].any()


def test_log_rms_z_is_taken_over_the_whole_war():
    """The pipeline filters an animal's WAR, every recording together; a z score per recording disagrees."""
    rng = np.random.default_rng(1)
    rms = np.vstack([rng.uniform(95, 105, (30, 5)), rng.uniform(380, 420, (30, 5))])
    rms[45, 0] = 120.0  # low for its own recording, ordinary for the animal
    war = _war(rms, animaldays=[DAYS[0]] * 30 + [DAYS[1]] * 30)
    settings = _settings(logrms_range=FRAGMENT["logrms_range"])
    pipeline = _pipeline(war, {"logrms_range": FRAGMENT["logrms_range"]})
    whole = _cells(war, z_unit="war")
    assert SD.fragment_reject(whole, settings)[0].tolist() == _at(whole, pipeline, war).tolist()
    assert not pipeline[45, 0]
    per_recording = _cells(war, z_unit="recording")
    assert SD.fragment_reject(per_recording, settings)[0][_cell(per_recording, 45, "LMot")].tolist() == [1]


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
    lo, audit = SD.lof_reject(cells, SETTINGS)
    assert lo.tolist() == _at(cells, pipeline, war).tolist()
    assert audit["n_rejected"] == 10 + 10 + 50 and audit["n_recording_rule_only"] == 20


def test_l_keeps_a_cell_without_a_score():
    cells = pd.DataFrame({"recording": "r__0", "window": 0, "channel": CH, "dataset": "ds",
                          "lof": [3.0, np.nan, 1.0, 1.0, 1.0]})
    lo, audit = SD.lof_reject(cells, SETTINGS)
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
    cells["config_list"] = [int(c in lists[d]) for d, c in zip(cells["animalday"], cells["channel"])]
    cells["config_list_defined"] = cells["animalday"].map(lambda d: bool(lists[d]))
    m, audit = SD.list_reject(cells, SETTINGS)
    assert m.tolist() == _at(cells, _pipeline(war, config), war).tolist()
    assert audit["n_recording_rule_only"] == (20 if form == "sessions" else 60)


# A recording with half its dataset's canonical channels: the pipeline filters it padded with NaN


def test_f_counts_the_padded_channels_as_kept():
    """Every fragment filter keeps a NaN cell, so the padded channels count in the window rule."""
    war = _planted_war()
    padded = _pad(war)
    cells = _cells(war)
    f, audit = SD.fragment_reject(cells, PADDED)
    assert f.tolist() == _at(cells, _pipeline(padded, FRAGMENT), padded).tolist()
    assert f[(cells["row"] == 4).to_numpy()].tolist() == [0, 1, 1, 1, 0], "two kept cells and five padded ones"
    assert audit["n_decided_by_absent_channels"] == 2 and audit["n_windows_decided_by_absent_channels"] == 1
    assert audit["n_window_rule_only"] == 0
    assert SD.fragment_reject(cells, SETTINGS)[0][(cells["row"] == 4).to_numpy()].tolist() == [1] * 5


def test_l_counts_the_padded_channels_as_valid():
    """A padded channel has no LOF score, so the LOF branch never rejects it and the recording rule counts it."""
    war = _war(np.full((10, 5), 100.0), lof={DAYS[0]: [3.0, 3.0, 3.0, 1.0, 1.0]})
    padded = _pad(war)
    bad = padded.get_bad_channels_by_lof_threshold(2.5)
    pipeline = _pipeline(padded, {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": bad}})
    cells = _cells(war)
    lo, audit = SD.lof_reject(cells, PADDED)
    assert lo.tolist() == _at(cells, pipeline, padded).tolist()
    assert lo.sum() == 30 and audit["n_decided_by_absent_channels"] == 20
    assert audit["recordings_decided_by_absent_channels"] == ["X1__0"]
    assert SD.lof_reject(cells, SETTINGS)[0].sum() == 50, "three of five flagged leaves two"


def test_m_counts_listed_channels_the_recording_lacks():
    """The manual branch rejects a listed padded channel, so it counts against the recording rule; the extraction
    reports it in config_list_unknown_channels."""
    war = _war(np.full((20, 5), 100.0), animaldays=[DAYS[0]] * 10 + [DAYS[1]] * 10)
    padded = _pad(war)
    lists = {DAYS[0]: ["LMot", "RMot", "LBar"],
             DAYS[1]: ["LMot", "LBar", "RBar", "RHip", "LAud", "RAud", "LVis", "RVis"]}
    pipeline = _pipeline(padded, {**NO_CHANNEL_LIST, "reject_channels_by_session": {"bad_channels_dict": lists}})
    cells = _cells(war)
    cells["config_list"] = [int(c in lists[d]) for d, c in zip(cells["animalday"], cells["channel"])]
    cells["config_list_defined"] = True
    unknown = [{"animal": "X1", "dataset": "ds", "recording": "X1__1", "animalday": DAYS[1], "channel": c,
                "reason": f"{c} is not among the recording's channels"} for c in lists[DAYS[1]][3:]]
    m, audit = SD.list_reject(cells, PADDED, unknown)
    assert m.tolist() == _at(cells, pipeline, padded).tolist()
    day = cells["animalday"].to_numpy()
    assert m[day == DAYS[0]].reshape(10, 5).sum(axis=0).tolist() == [10, 10, 10, 0, 0], "seven channels stay valid"
    assert m[day == DAYS[1]].all(), "two canonical channels stay unlisted"
    assert audit["listed_absent"] == {"X1__1": lists[DAYS[1]][3:]} and audit["unresolved"] == []
    assert audit["n_decided_by_absent_channels"] == 20
    assert audit["recordings_decided_by_absent_channels"] == ["X1__0"], "X1__1 goes whole either way"
    assert not SD.list_reject(cells, PADDED)[0][day == DAYS[1]].all(), "without the report, seven look valid"


def test_listed_absent_reads_the_extraction_report():
    cells = _cells(_war(np.full((4, 5), 100.0)))
    unknown = [{"recording": "X1__0", "channel": "RHip"}, {"recording": "X1__0", "channel": " LAud "},
               {"recording": "X1__0", "channel": "R Vis Ctx", "abbrev": "RVis"},
               {"recording": "X1__0", "channel": "RHip"}, {"recording": "X1__0", "channel": "LMot"},
               {"recording": "X1__0", "channel": "Nowhere"}, {"recording": "X9__0", "channel": "RHip"}]
    absent, unresolved = SD.listed_absent(cells, PADDED, unknown)
    assert absent == {"X1__0": ["RHip", "LAud", "RVis"]}, "once each, and not a channel the recording has"
    assert unresolved == [{"recording": "X1__0", "channel": "Nowhere"}], "a recording without cells is skipped"


def test_member_masks_report_the_padded_channels():
    cells = _cells(_planted_war())
    _, audits = SD.member_masks(cells, PADDED)
    assert audits["absent_channels"] == {"X1__0": CANON[5:]}
    assert SD.member_masks(cells, SETTINGS)[1]["absent_channels"] == {}


def test_cells_outside_the_canonical_channels_are_refused():
    cells = _cells(_planted_war())
    with pytest.raises(SystemExit, match="X1__0 LHip"):
        SD.fragment_reject(cells, {**SETTINGS, "canonical_channels": {"ds": CH[:4]}})
    with pytest.raises(SystemExit, match="no canonical channels for the datasets"):
        SD.lof_reject(cells, {**SETTINGS, "canonical_channels": {"other": CH}})


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


def test_fragment_min_valid_channels_is_the_one_the_workflow_passes():
    src = (REPO / "workflow/scripts/filter_wars_fragments.py").read_text()
    calls = re.findall(r"apply_filters\(filter_config=fragment_filter_config, min_valid_channels=(\d+)\)", src)
    assert calls == [str(SD.FRAGMENT_MIN_VALID_CHANNELS)]


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
        "today": [1, 1, 1, 0, 0], "replace": [1, 0, 0, 1, 0], "add": [1, 1, 1, 1, 0],
        "autoreject_alone": [0, 0, 0, 1, 0], "filters_alone": [1, 0, 0, 0, 0], "config_lists_alone": [0, 0, 1, 0, 0]}


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
    """F flags nothing, L flags the clean cells of g1, M nothing: today is worse than replace in every group."""
    lo = ((df["cv_group"] == "g1") & (df["y"] == 0)).astype(int).to_numpy()
    zero = np.zeros(len(df), int)
    return {"F": zero, "L": lo, "M": zero}


def test_score_combinations_reports_every_breakdown_and_the_paired_differences():
    df = _cohort()
    preds = SD.combination_preds(_members(df), df["y"].to_numpy())
    combos = SD.score_combinations(df, preds, boot=200)
    assert list(combos) == list(SD.COMBINATIONS)
    for name, entry in combos.items():
        assert set(entry) >= {"members", "definition", "cells", "pooled", "ci95", *SD.BREAKDOWNS}, name
        assert set(entry["per_cv_group"]) <= {"g1", "g2", "g3", "g4"}
    assert combos["replace"]["definition"] == "F or A(k)" and combos["today"]["definition"] == "F or L or M"
    assert set(combos["today"]["per_cv_group"]) == {"g1", "g2", "g3", "g4"}
    assert set(combos["today"]["per_isday"]) == {"True", "False"}
    assert set(combos["today"]["per_genotype"]) == {"WT", "KO"}
    assert set(combos["today"]["per_animal"]) == {"ag1", "ag2", "ag3", "ag4"}
    lists = combos["config_lists_alone"]
    assert lists["cells"] == "config_list_defined" and lists["pooled"]["n"] == 100
    assert set(lists["per_cv_group"]) == {"g2", "g3"}
    d = combos["replace"]["minus_today"]
    assert d["f1_difference"] == pytest.approx(1.0 - f1_score(df["y"], preds["today"]))
    assert d["ci95"][0] > 0, "replace is perfect and today is not in g1, so every resample favors replace"
    assert "minus_today" in combos["add"] and "minus_today" not in combos["today"]


def test_paired_difference_of_identical_combinations_is_zero():
    df = _cohort()
    members = {"F": df["y"].to_numpy(), "L": np.zeros(len(df), int), "M": np.zeros(len(df), int)}
    combos = SD.score_combinations(df, SD.combination_preds(members, np.zeros(len(df), int)), boot=200)
    d = combos["replace"]["minus_today"]
    assert d["f1_difference"] == 0 and d["ci95"][0] <= 0 <= d["ci95"][1]


def test_intervals_need_two_groups():
    y = np.array([0, 1, 1, 0])
    assert all(np.isnan(SD.cluster_boot_ci(y, y, np.array(["g"] * 4), n=50)))
