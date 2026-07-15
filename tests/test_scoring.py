"""Tests for neurodent.results.scoring: ground truth and detector scoring (issue #208).

These numbers decide which detector wins, so the failures worth testing for are not crashes but a
score computed over a truth set that silently shrank, or over a mask whose polarity was inverted.
"""
import csv

import numpy as np
import pandas as pd
import pytest

from neurodent.results.scoring import (
    CATEGORIES,
    LABEL_COL_PREFIX,
    consensus,
    ingest,
    interrater,
    score_keep_mask,
    score_mask,
)

CHS = ["ch0", "ch1", "ch2", "ch3"]


def _write_labels(path, recording, cells):
    """cells: {(window, channel): token}. Everything unlisted is `clean`."""
    wins = sorted({w for w, _ in cells}) or [0]
    with open(path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["image", "recording", "window", "t_start_s"]
                     + [f"{LABEL_COL_PREFIX}{c}" for c in CHS])
        for w in wins:
            wtr.writerow([f"{recording}_w{w}.png", recording, w, w * 5.0]
                         + [cells.get((w, c), "clean") for c in CHS])


def _cons(cells, n_win=4):
    return pd.DataFrame([{"recording": "A", "window": w, "channel": c, "t_start_s": w * 5.0,
                          "y_true": int(cells.get((w, c), 0)), "n_raters": 3, "any_event": False}
                         for w in range(n_win) for c in CHS])


# --------------------------------------------------------------------------- ingest / consensus

def test_unknown_label_token_is_refused(tmp_path):
    """An unrecognised token must not become NaN and vanish from consensus."""
    p = tmp_path / "l.csv"
    _write_labels(p, "A", {(0, "ch0"): "bad", (0, "ch1"): "BAD_TYPO"})
    with pytest.raises(ValueError, match="unrecognised label tokens"):
        ingest({"r1": str(p)})


def test_event_is_kept_but_tagged(tmp_path):
    """A seizure looks like an artifact in one isolated window but is the signal. It must be kept
    (y_true=0) and still visible as an event, or the harness cannot show a detector eating real
    brain activity."""
    p = tmp_path / "l.csv"
    _write_labels(p, "A", {(0, "ch0"): "event", (0, "ch1"): "bad"})
    cons = consensus(ingest({"r1": str(p)}))

    ev = cons[(cons.window == 0) & (cons.channel == "ch0")].iloc[0]
    bad = cons[(cons.window == 0) & (cons.channel == "ch1")].iloc[0]
    assert ev.y_true == 0 and ev.any_event
    assert bad.y_true == 1 and not bad.any_event


def test_majority_vote_and_all_unsure_drops_out(tmp_path):
    ps = {}
    for i, tok in enumerate(["bad", "bad", "clean"]):              # 2-1 -> bad
        ps[f"r{i}"] = str(tmp_path / f"r{i}.csv")
        _write_labels(ps[f"r{i}"], "A", {(0, "ch0"): tok, (0, "ch1"): "unsure"})
    long = ingest(ps)
    cons = consensus(long)

    assert cons[(cons.window == 0) & (cons.channel == "ch0")].iloc[0].y_true == 1
    assert cons[(cons.window == 0) & (cons.channel == "ch1")].empty     # nobody knew -> not guessed at
    assert interrater(long)["metric"] == "fleiss"


def test_tie_rejects(tmp_path):
    """Conservative: keeping a suspect window pollutes analysis, dropping a clean one only costs data."""
    ps = {}
    for i, tok in enumerate(["bad", "clean"]):
        ps[f"r{i}"] = str(tmp_path / f"r{i}.csv")
        _write_labels(ps[f"r{i}"], "A", {(0, "ch0"): tok})
    cons = consensus(ingest(ps))
    assert cons[(cons.window == 0) & (cons.channel == "ch0")].iloc[0].y_true == 1


def test_interrater_is_cohen_for_two_raters(tmp_path):
    ps = {}
    for i in range(2):
        ps[f"r{i}"] = str(tmp_path / f"r{i}.csv")
        _write_labels(ps[f"r{i}"], "A", {(0, "ch0"): "bad"})
    assert interrater(ingest(ps))["metric"] == "cohen"


# --------------------------------------------------------------------------- scoring

def test_perfect_detector_scores_one():
    truth = {(1, "ch2"): 1, (3, "ch0"): 1}
    grid = np.zeros((4, 4), dtype=bool)
    grid[1, 2] = grid[3, 0] = True
    r = score_mask(grid, CHS, _cons(truth), "A")
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["n"] == 16 and r["n_uncovered"] == 0


def test_wrong_detector_is_penalised():
    truth = {(1, "ch2"): 1, (3, "ch0"): 1}
    grid = np.zeros((4, 4), dtype=bool)
    grid[1, 2] = True                       # one hit
    grid[0, 1] = True                       # one false alarm; and (3,ch0) is missed
    r = score_mask(grid, CHS, _cons(truth), "A")
    assert r["recall"] == 0.5 and r["precision"] == 0.5 and r["f1"] == 0.5
    assert r["confusion"] == [[13, 1], [1, 1]]


def test_uncovered_cells_are_refused_not_skipped():
    """A detector run on a 2 h crop against labels from later windows. Skipping the uncovered cells
    shrinks the denominator and returns a confident score over whatever happened to line up."""
    cons = _cons({}, n_win=4)
    short = np.zeros((2, 4), dtype=bool)                 # covers windows 0-1 only
    with pytest.raises(ValueError, match="not covered by the detector grid"):
        score_mask(short, CHS, cons, "A")

    r = score_mask(short, CHS, cons, "A", strict=False)  # opt in, and it says what it dropped
    assert r["n"] == 8 and r["n_uncovered"] == 8


def test_unknown_channels_are_refused():
    with pytest.raises(ValueError, match="not covered by the detector grid"):
        score_mask(np.zeros((4, 2), dtype=bool), ["ch0", "ch1"], _cons({}), "A")


# --------------------------------------------------------------------------- absolute-time matching

def test_score_mask_matches_by_absolute_time():
    """With grid_times, a labelled cell is matched to the detector row whose start time equals its
    t_start_s -- not by integer index."""
    truth = {(2, "ch1"): 1}                              # cell at window 2 -> t_start_s = 10.0
    cons = _cons(truth, n_win=4)                          # windows 0..3 at t = 0,5,10,15
    grid = np.zeros((4, len(CHS)), dtype=bool)
    grid[2, CHS.index("ch1")] = True
    r = score_mask(grid, CHS, cons, "A", grid_times=[0.0, 5.0, 10.0, 15.0])
    assert r["f1"] == 1.0 and r["n_uncovered"] == 0


def test_time_match_is_robust_to_a_renumbered_grid_where_index_would_misalign():
    """The whole point: if the detector rows are ordered differently from the window numbering,
    integer-index matching silently misaligns; time matching lands right. Same times (so nothing is
    uncovered), reversed row order."""
    truth = {(2, "ch1"): 1}                              # cell at t = 10
    cons = _cons(truth, n_win=4)                          # windows 0..3 at t = 0,5,10,15
    grid = np.zeros((4, len(CHS)), dtype=bool)
    grid[1, CHS.index("ch1")] = True                     # rows in REVERSE time order -> t=10 is row 1
    assert score_mask(grid, CHS, cons, "A", grid_times=[15.0, 10.0, 5.0, 0.0])["f1"] == 1.0
    # integer-index fallback maps window 2 -> row 2 (t=5, clean) -> wrong
    assert score_mask(grid, CHS, cons, "A")["f1"] < 1.0


def test_time_match_with_no_fragment_in_tolerance_is_uncovered():
    cons = _cons({(0, "ch0"): 1}, n_win=1)               # cell at t = 0
    grid = np.ones((3, len(CHS)), dtype=bool)
    with pytest.raises(ValueError, match="no detector fragment within"):
        score_mask(grid, CHS, cons, "A", grid_times=[100.0, 105.0, 110.0])


# --------------------------------------------------------------------------- the polarity footgun

def test_keep_mask_and_reject_grid_are_inverses():
    """FILTER_REGISTRY masks are True=KEEP; score_mask wants True=REJECT. No runtime check can tell
    them apart, so the inversion must be right."""
    truth = {(1, "ch2"): 1}
    reject = np.zeros((4, 4), dtype=bool)
    reject[1, 2] = True

    assert score_mask(reject, CHS, _cons(truth), "A")["f1"] == 1.0
    assert score_keep_mask(~reject, CHS, _cons(truth), "A")["f1"] == 1.0


def test_passing_a_keep_mask_to_score_mask_scores_terribly():
    """Hand a keep-mask to the reject-scorer and you get a confident, inverted score rather than an
    error. That is why the polarity lives in the function name."""
    truth = {(1, "ch2"): 1}
    keep = np.ones((4, 4), dtype=bool)
    keep[1, 2] = False                                   # a filter's mask: True = keep

    right = score_keep_mask(keep, CHS, _cons(truth), "A")
    wrong = score_mask(keep, CHS, _cons(truth), "A")     # the mistake
    assert right["f1"] == 1.0
    assert wrong["f1"] < 0.2                             # silently, catastrophically wrong


def test_scores_a_real_registered_filter():
    """Against the real registry, so the keep/reject convention is asserted against the actual
    filters rather than against scoring.py's own docstring."""
    from neurodent.results.filters import FILTER_REGISTRY, ChannelInfo, compute_filter_mask

    assert "high_rms" in FILTER_REGISTRY

    n_win = 4
    # rms well under the threshold everywhere except (window 1, ch2), which the filter must reject.
    rms = np.full((n_win, len(CHS)), 50.0)
    rms[1, 2] = 5000.0
    df_stats = pd.DataFrame({"rms": [row.tolist() for row in rms]})

    keep = compute_filter_mask(
        df_stats,
        {"high_rms": {"max_rms": 500}},
        channel_info=ChannelInfo(channel_names=list(CHS), channel_abbrevs=list(CHS)),
        n_windows=n_win,
    )
    assert keep.shape == (n_win, len(CHS))
    assert not keep[1, 2] and keep[0, 0]                 # registry convention: True = KEEP

    truth = {(1, "ch2"): 1}
    assert score_keep_mask(keep, CHS, _cons(truth), "A")["f1"] == 1.0


def test_categories_constant_matches_the_label_map():
    """The rater page offers exactly these four tokens; ingest must understand all of them."""
    from neurodent.results.scoring import LABEL_MAP

    for cat in CATEGORIES:
        assert cat in LABEL_MAP, f"the rater page can emit {cat!r} but ingest cannot read it"
