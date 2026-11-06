"""Tests for the pre-registered evaluation scripts in ``scripts/labeling/``.

Everything runs on synthetic cells and fake labels generated here; no recording and no real label file is
read. Covered: N-rater consensus with rater categories, pairwise agreement, the split guard, the recording
map, cell alignment with metadata, the selection with group folds, nested selection and leave-one-dataset-out,
the event-cell report, and the one-shot test scoring with its refusals.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "labeling"))
import extract_dev_features as EDF  # noqa: E402
import report_event_cells as EV  # noqa: E402
import score_detectors as SDET  # noqa: E402
import score_test as ST  # noqa: E402
import select_detectors as SD  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.scoring import consensus, pairwise_agreement  # noqa: E402

pytestmark = pytest.mark.unit


def _long(rows):
    """Long-form rater table from (recording, window, channel, rater, category) tuples."""
    vote = {"artifact": 1.0, "clean": 0.0, "event": 0.0, "unsure": np.nan}
    df = pd.DataFrame(rows, columns=["recording", "window", "channel", "rater", "category"])
    df["y"] = df["category"].map(vote)
    df["t_start_s"] = df["window"] * 5.0
    return df


def test_three_rater_majority_and_categories():
    df = _long([
        ("r__0", 1, "LMot", "A", "artifact"), ("r__0", 1, "LMot", "B", "artifact"), ("r__0", 1, "LMot", "C", "clean"),
        ("r__0", 2, "LMot", "A", "artifact"), ("r__0", 2, "LMot", "B", "event"), ("r__0", 2, "LMot", "C", "unsure"),
        ("r__0", 3, "LMot", "A", "unsure"), ("r__0", 3, "LMot", "B", "unsure"), ("r__0", 3, "LMot", "C", "unsure"),
    ])
    out = consensus(df, rule="majority").set_index("window")
    assert out.loc[1, "y_true"] == 1 and out.loc[1, "n_raters"] == 3
    assert out.loc[2, "y_true"] == 1 and out.loc[2, "n_raters"] == 2  # one bad, one event: a tie rejects
    assert bool(out.loc[2, "any_event"])
    assert 3 not in out.index  # every rater abstained
    assert out.loc[2, ["cat_A", "cat_B", "cat_C"]].tolist() == ["artifact", "event", "unsure"]
    assert consensus(df, rule="unanimous").set_index("window").loc[2, "y_true"] == 0


def test_pairwise_agreement_counts():
    df = _long([
        ("r__0", w, "LMot", rater, cat)
        for w, cats in enumerate([("artifact", "artifact"), ("clean", "artifact"), ("clean", "clean"),
                                  ("artifact", "clean"), ("unsure", "clean")])
        for rater, cat in zip("AB", cats)
    ])
    row = pairwise_agreement(df).iloc[0]
    assert (row["rater_a"], row["rater_b"], row["n_cells"]) == ("A", "B", 4)
    assert row["a_keep_b_bad"] == 1 and row["a_bad_b_keep"] == 1
    assert row["agreement"] == pytest.approx(0.5)


def test_check_split(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["a", "b"], "test": ["c"]}))
    assert SDET.check_split({"a"}, split, "dev") == {"a", "b"}
    with pytest.raises(SystemExit, match="outside split"):
        SDET.check_split({"a", "c"}, split, "dev")


def test_recording_map(tmp_path):
    assert SDET.RecordingMap(positional=True).label("A1", 2, "A1 WT Jan-01-2025") == "A1__2"
    with pytest.raises(SystemExit, match="recording map not found"):
        SDET.RecordingMap(tmp_path / "missing.csv")
    path = tmp_path / "map.csv"
    pd.DataFrame([
        {"july_recording": "A1__0", "animal": "A1", "animalday": "A1 WT Nov-04-2025", "port_position": 1},
        {"july_recording": "A1__2", "animal": "A1", "animalday": "A1 WT Oct-31-2025", "port_position": 0},
        {"july_recording": "B2__0", "animal": "B2", "animalday": np.nan, "port_position": 0},
    ]).to_csv(path, index=False)
    rmap = SDET.RecordingMap(path)
    assert rmap.label("A1", 0, "A1 WT Oct-31-2025") == "A1__2"  # matched by animalday, not position
    assert rmap.label("A1", 1, "A1 WT Nov-04-2025") == "A1__0"
    assert rmap.label("B2", 0, "B2 KO Jan-01-2025") == "B2__0"  # no animalday in the map: by position
    assert rmap.label("A1", 5, "A1 WT Dec-01-2025") is None  # unmapped


def test_align_cells_carries_metadata():
    cons = pd.DataFrame([{"recording": "a__0", "window": 1, "channel": "RMot", "t_start_s": 5.0, "y_true": 1,
                          "n_raters": 3, "any_event": True, "cat_A": "event"}])
    rows = EDF.align_cells(np.ones((2, 2)), np.zeros((2, 2)), np.zeros(2), {}, ["LMot", "RMot"], cons,
                           np.array([0.0, 5.0]), frag_s=5.0,
                           row_meta={"isday": np.array([True, False])}, cell_cols=["n_raters", "any_event", "cat_A"])
    assert not rows[0]["isday"]
    assert rows[0]["n_raters"] == 3 and rows[0]["cat_A"] == "event" and np.isnan(rows[0]["lof"])


# A synthetic dev and test cohort shaped like extract_dev_features' output.

GRID = adr.grid_config_names()
PERFECT = GRID[5]


def _cells(animals, seed):
    """Cells for (animal, group, dataset) triples; one grid column predicts y exactly, the rest are noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for animal, group, dataset in animals:
        for w in range(40):
            for ch in ("LMot", "RMot"):
                y = int(rng.random() < 0.3)
                cats = ["artifact" if y else "clean", "artifact" if y else "event", "clean" if y else "clean"]
                rows.append({"recording": f"{animal}__0", "window": w, "channel": ch, "t_start_s": 5.0 * w, "y": y,
                             "rms": rng.normal(50, 5) + 100 * y, "logrmsz_abs": abs(rng.normal()),
                             "beta_wmax": rng.random(),
                             "lof": rng.random(), "animal": animal, "animalday": f"{animal} WT Jan-01-2025",
                             "dataset": dataset, "cv_group": group, "genotype": "WT",
                             "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(seconds=5 * w),
                             "isday": bool(w % 2), "n_raters": 3, "any_event": not y,
                             "cat_A": cats[0], "cat_B": cats[1], "cat_C": cats[2]})
    df = pd.DataFrame(rows)
    for name in GRID:
        df[name] = df["y"] if name == PERFECT else (rng.random(len(df)) < 0.3).astype(int)
    return df


DEV = [("d1", "g1", "ds1"), ("d2", "g1", "ds1"), ("d3", "g2", "ds1"), ("d4", "g3", "ds2"), ("d5", "g4", "ds2"),
       ("d6", "g5", "ds3"), ("d7", "g6", "ds3")]
TEST = [("t1", "h1", "ds1"), ("t2", "h2", "ds2"), ("t3", "h3", "ds3")]


def _write_features(tmp_path, name, df, which, **overrides):
    path = tmp_path / f"{name}.parquet"
    df.to_parquet(path)
    params = {"which": which, "rule": "majority", "raters": ["A", "B", "C"], "ar_max_fit_fragments": 5000,
              "seed": 0, "configs": GRID, "positional_recordings": False}
    params.update(overrides)
    prov = {"params": params, "code": {f: "same" for f in ST.MATCHED_CODE}}
    Path(str(path) + ".provenance.json").write_text(json.dumps(prov))
    return path


@pytest.fixture
def cohort(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [a for a, _, _ in DEV], "test": [a for a, _, _ in TEST]}))
    dev = _write_features(tmp_path, "dev", _cells(DEV, 0), "dev")
    test = _write_features(tmp_path, "test", _cells(TEST, 1), "test")
    selection = tmp_path / "dev_selection.json"
    SD.main(["--features", str(dev), "--split", str(split), "--out", str(selection), "--boot", "30"])
    return {"split": split, "dev": dev, "test": test, "selection": selection, "tmp": tmp_path}


def test_selection_picks_the_predictive_column_and_reports_nested(cohort):
    out = json.loads(cohort["selection"].read_text())
    assert out["winner"]["detector"] == PERFECT and out["winner"]["kind"] == "autoreject"
    assert len(out["leaderboard"]) == len(SD.DETECTORS) + len(GRID) == 43
    assert {f["held_out"] for f in out["nested_logo"]["folds"]} == {g for _, g, _ in DEV}
    assert {f["held_out"] for f in out["nested_lodo"]["folds"]} == {"ds1", "ds2", "ds3"}
    assert all(f["pick"] == PERFECT for f in out["nested_logo"]["folds"])
    assert out["nested_logo"]["f1"] == pytest.approx(1.0)
    board = out["leaderboard"][PERFECT]
    assert set(board["per_dataset"]) == {"ds1", "ds2", "ds3"} and set(board["per_group"]) == {g for _, g, _ in DEV}
    assert out["rater_agreement"]["overall"], "pairwise agreement reported"


def test_selection_refuses_test_animals_and_unknown_columns(cohort):
    mixed = pd.concat([pd.read_parquet(cohort["dev"]), pd.read_parquet(cohort["test"]).head(4)])
    path = _write_features(cohort["tmp"], "mixed", mixed, "dev")
    with pytest.raises(SystemExit, match="outside split"):
        SD.main(["--features", str(path), "--split", str(cohort["split"]), "--out", str(cohort["tmp"] / "x.json")])
    extra = pd.read_parquet(cohort["dev"]).assign(mystery=1)
    path = _write_features(cohort["tmp"], "extra", extra, "dev")
    with pytest.raises(SystemExit, match="unknown role"):
        SD.main(["--features", str(path), "--split", str(cohort["split"]), "--out", str(cohort["tmp"] / "y.json")])


def test_score_test_runs_once(cohort):
    out = cohort["tmp"] / "test_score.json"
    args = ["--features", str(cohort["test"]), "--pick", str(cohort["selection"]), "--split", str(cohort["split"]),
            "--out", str(out), "--boot", "30"]
    ST.main(args)
    result = json.loads(out.read_text())
    assert result["pick"]["winner"]["detector"] == PERFECT
    assert result["pooled"]["f1"] == pytest.approx(1.0)
    assert set(result["per_dataset"]) == {"ds1", "ds2", "ds3"}
    with pytest.raises(SystemExit, match="scored once"):
        ST.main(args)


def test_score_test_refusals(cohort):
    base = ["--split", str(cohort["split"]), "--boot", "30"]
    with pytest.raises(SystemExit, match="no frozen pick"):
        ST.main(["--features", str(cohort["test"]), "--pick", str(cohort["tmp"] / "none.json"),
                 "--out", str(cohort["tmp"] / "a.json"), *base])
    other = _write_features(cohort["tmp"], "test_seed1", pd.read_parquet(cohort["test"]), "test", seed=1)
    with pytest.raises(SystemExit, match="produced differently"):
        ST.main(["--features", str(other), "--pick", str(cohort["selection"]), "--out", str(cohort["tmp"] / "b.json"),
                 *base])
    dev_as_test = _write_features(cohort["tmp"], "dev_as_test", pd.read_parquet(cohort["dev"]), "test")
    with pytest.raises(SystemExit, match="outside split"):
        ST.main(["--features", str(dev_as_test), "--pick", str(cohort["selection"]),
                 "--out", str(cohort["tmp"] / "c.json"), *base])


def test_event_report(cohort):
    assert EV.coarse_class(["artifact", "event", "unsure"]) == "bad_and_event"
    assert EV.coarse_class(["artifact", "artifact"]) == "all_bad"
    assert EV.coarse_class(["artifact", "clean"]) == "bad_and_clean"
    assert EV.coarse_class(["event", "clean"]) == "event_no_bad"
    assert EV.coarse_class(["clean", "unsure"]) == "all_clean"
    assert EV.coarse_class(["unsure", "unlabeled"]) == "no_votes"
    out = cohort["tmp"] / "events.csv"
    EV.main(["--features", str(cohort["dev"]), "--selection", str(cohort["selection"]), "--split", str(cohort["split"]),
             "--detectors", PERFECT, "high_rms", "--out", str(out)])
    table = pd.read_csv(out)
    perfect = table[table["detector"] == PERFECT].set_index("coarse_class")
    assert perfect.loc["bad_and_clean", "rejection_rate"] == 1.0  # the y=1 cells: two bad votes, one clean
    assert perfect.loc["event_no_bad", "rejection_rate"] == 0.0
