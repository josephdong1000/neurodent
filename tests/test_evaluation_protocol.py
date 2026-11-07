"""Tests for the pre-registered evaluation scripts in ``scripts/labeling/``.

Everything runs on synthetic cells and fake labels generated here; no recording and no real label file is
read. Covered: N-rater consensus with rater categories, pairwise agreement, the split guard, the recording
map, cell alignment with metadata, the combination selection end to end with its nested runs, diagnostic
ranking and refusals, the event-cell report with the combinations, and the one-shot test scoring with its
output and refusals.
"""

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score

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


# A synthetic dev and test cohort shaped like extract_dev_features' output, scored with a config written here.

GRID = adr.grid_config_names()
PERFECT = GRID[5]       # predicts y on the dev cells, and is noise on the test cells
TEST_PERFECT = GRID[6]  # predicts y on the test cells, and is noise on the dev cells
CHANNELS = ("LMot", "RMot", "LBar", "RBar")
DATASETS = ("ds1", "ds2", "ds3")
CONFIG = {"analysis": {
    "fragment_filter_config": {"logrms_range": {"z_range": 3}, "high_rms": {"max_rms": 500},
                               "low_rms": {"min_rms": 50}, "high_beta": {"max_beta_prop": 0.4}},
    "channel_filter_config": {"manual": {"reject_channels": [], "min_valid_channels": 3},
                              "lof": {"reject_lof_threshold": 2.5, "reject_channels": [], "min_valid_channels": 3}},
}}


def _cells(animals, seed, perfect=PERFECT):
    """Cells for (animal, group, dataset) triples, 40 windows of 4 channels each.

    One grid column (``perfect``) predicts y exactly and the rest are noise. Filter scores stay inside the
    config settings, so F rejects nothing; LOF flags RBar of each dataset's first animal, and the config lists
    name LBar for the ds2 animals, whose lists are defined. Genotypes alternate and isday follows the window.
    """
    rng = np.random.default_rng(seed)
    rows, first = [], set()
    for i, (animal, group, dataset) in enumerate(animals):
        lof = {ch: rng.uniform(0.9, 1.1) for ch in CHANNELS}
        if dataset not in first:
            lof["RBar"], first = 3.0, first | {dataset}
        defined = dataset == "ds2"
        for w in range(40):
            beta = rng.uniform(0.05, 0.3)
            for ch in CHANNELS:
                y = int(rng.random() < 0.3)
                cats = ["artifact" if y else "clean", "artifact" if y else "event", "clean"]
                rows.append({"recording": f"{animal}__0", "window": w, "channel": ch, "t_start_s": 5.0 * w, "y": y,
                             "rms": rng.uniform(80, 200), "logrmsz_abs": abs(rng.normal(0, 0.5)),
                             "beta_wmax": beta, "lof": lof[ch], "animal": animal,
                             "animalday": f"{animal} WT Jan-01-2025", "dataset": dataset, "cv_group": group,
                             "genotype": "WT" if i % 2 else "KO",
                             "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(seconds=5 * w),
                             "isday": bool(w % 2), "n_raters": 3, "any_event": not y,
                             "config_list": int(defined and ch == "LBar"), "config_list_defined": defined,
                             "cat_A": cats[0], "cat_B": cats[1], "cat_C": cats[2]})
    df = pd.DataFrame(rows)
    for name in GRID:
        df[name] = df["y"] if name == perfect else (rng.random(len(df)) < 0.3).astype(int)
    return df


DEV = [("d1", "g1", "ds1"), ("d2", "g1", "ds1"), ("d3", "g2", "ds1"), ("d4", "g3", "ds2"), ("d5", "g4", "ds2"),
       ("d6", "g5", "ds3"), ("d7", "g6", "ds3")]
TEST = [("t1", "h1", "ds1"), ("t2", "h2", "ds2"), ("t3", "h3", "ds2"), ("t4", "h4", "ds3")]


INPUTS = ["keymap", "split", "recording_map", "config/config.yaml", "config/datasets/ds2.yaml"]


def _provenance(which, top=None, **params):
    """Feature provenance as extract_dev_features writes it, with top-level and parameter overrides.

    The label files differ between the partitions, as they do in a real run; every other input is the same.
    """
    p = {"which": which, "rule": "majority", "raters": ["A", "B", "C"], "ar_max_fit_fragments": 5000, "seed": 0,
         "configs": GRID, "positional_recordings": False, "temp_dir": "/scratch/hash_port",
         "datasets": list(DATASETS), "lof_chunk_duration_s": 60, "chunk_duration_s": 3600}
    p.update(params)
    inputs = {**{f"labels_{r}": {"path": f"labels_{r}_{which}.csv", "sha256": f"{r} {which}"} for r in "ABC"},
              **{name: {"path": name, "sha256": f"{name} 0"} for name in INPUTS}}
    prov = {"params": p, "code": {f: "same" for f in ST.matched_code()}, "src_tree": "tree0", "src_dirty": False,
            "inputs": inputs, "config_list_unknown_channels": []}
    prov.update(top or {})
    return prov


def _write_features(tmp_path, name, df, which, top=None, **params):
    """Write the parquet and its provenance, which records the parquet's sha256 unless ``top`` overrides it."""
    path = tmp_path / f"{name}.parquet"
    df.to_parquet(path)
    top = {"outputs": {"features": {"path": str(path), "sha256": SD._sha256(path)}}, **(top or {})}
    Path(str(path) + ".provenance.json").write_text(json.dumps(_provenance(which, top, **params)))
    return path


def _write_datasets(folder, channels=CHANNELS):
    """The dataset configs beside ``folder/config.yaml``, each with an inline samples channel map."""
    (folder / "datasets").mkdir(exist_ok=True)
    for name in DATASETS:
        config = {"samples_data": {"channels": {c: [c] for c in channels}, "animals": []}}
        (folder / "datasets" / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def _write_config(path, **lof):
    config = json.loads(json.dumps(CONFIG))
    config["analysis"]["channel_filter_config"]["lof"].update(lof)
    path.write_text(yaml.safe_dump(config))
    _write_datasets(path.parent)
    return path


def _select_args(cohort, features, out):
    return ["--features", str(features), "--split", str(cohort["split"]), "--config", str(cohort["config"]),
            "--out", str(out), "--boot", "30"]


@pytest.fixture(scope="module")
def _cohort_dir(tmp_path_factory):
    """The split, config, features and dev selection, built once per module."""
    base = tmp_path_factory.mktemp("cohort")
    (base / "split.json").write_text(json.dumps({"dev": [a for a, _, _ in DEV], "test": [a for a, _, _ in TEST]}))
    _write_config(base / "config.yaml")
    _write_features(base, "dev", _cells(DEV, 0), "dev")
    _write_features(base, "test", _cells(TEST, 1, perfect=TEST_PERFECT), "test")
    SD.main(["--features", str(base / "dev.parquet"), "--split", str(base / "split.json"),
             "--config", str(base / "config.yaml"), "--out", str(base / "dev_selection.json"), "--boot", "30"])
    return base


@pytest.fixture
def cohort(_cohort_dir, tmp_path):
    """A fresh copy of the module's cohort files for one test."""
    for path in _cohort_dir.iterdir():
        if path.is_dir():
            shutil.copytree(path, tmp_path / path.name)
        else:
            shutil.copy(path, tmp_path / path.name)
    return {"split": tmp_path / "split.json", "config": tmp_path / "config.yaml", "dev": tmp_path / "dev.parquet",
            "test": tmp_path / "test.parquet", "selection": tmp_path / "dev_selection.json", "tmp": tmp_path}


def _score_args(cohort, features, out, pick=None):
    return ["--features", str(features), "--pick", str(pick or cohort["selection"]), "--split", str(cohort["split"]),
            "--config", str(cohort["config"]), "--out", str(cohort["tmp"] / out), "--boot", "30"]


def _claimed(cohort):
    """The files score_test.py claims that exist in the cohort folder, besides an output."""
    return sorted(p.name for p in cohort["tmp"].glob("*" + ST.SCORED_SUFFIX))


def test_selection_picks_the_column_that_completes_replace(cohort):
    out = json.loads(cohort["selection"].read_text())
    assert set(out) == {"pick", "settings", "member_masks", "combinations", "nested_replace_logo",
                        "nested_replace_lodo", "diagnostic", "rater_agreement", "features", "provenance"}
    assert out["pick"] == {"autoreject_config": PERFECT, "replace_f1": 1.0, "tie_broken": False}
    assert out["settings"]["lof"] == {"reject_lof_threshold": 2.5, "min_valid_channels": 3}
    assert out["settings"]["config"]["sha256"] == SD._sha256(cohort["config"])
    assert out["settings"]["canonical_channels"] == {name: list(CHANNELS) for name in DATASETS}
    assert out["features"]["sha256"] == SD._sha256(cohort["dev"])
    assert {rel: out["provenance"]["code"][rel] for rel in SD.SCORING_CODE} == SD.scoring_code_hashes()
    assert out["member_masks"]["absent_channels"] == {}
    assert out["member_masks"]["F"]["n_rejected"] == 0
    assert out["member_masks"]["L"]["n_rejected"] == 3 * 40  # RBar of d1, d4 and d6
    assert out["member_masks"]["M"]["n_rejected"] == 2 * 40 and out["member_masks"]["M"]["n_defined"] == 2 * 160
    combos = out["combinations"]
    assert list(combos) == list(SD.COMBINATIONS)
    assert combos["replace"]["pooled"]["f1"] == pytest.approx(1.0)
    assert combos["config_lists_alone"]["pooled"]["n"] == 2 * 160  # only the ds2 cells have lists
    assert set(combos["config_lists_alone"]["per_dataset"]) == {"ds2"}
    assert combos["replace"]["minus_today"]["f1_difference"] > 0
    for name, entry in combos.items():
        assert {"per_dataset", "per_cv_group", "per_animal", "per_isday", "per_genotype"} <= set(entry), name
        assert set(entry["pooled"]) >= {"precision", "recall", "f1", "cohen_kappa", "tp", "fp", "fn", "tn",
                                        "flag_rate"}
    assert set(combos["today"]["per_animal"]) == {a for a, _, _ in DEV}
    assert set(combos["today"]["per_isday"]) == {"True", "False"}
    assert set(combos["today"]["per_genotype"]) == {"WT", "KO"}
    logo, lodo = out["nested_replace_logo"], out["nested_replace_lodo"]
    assert {f["held_out"] for f in logo["folds"]} == {g for _, g, _ in DEV}
    assert {f["held_out"] for f in lodo["folds"]} == {"ds1", "ds2", "ds3"}
    assert all(f["autoreject_config"] == PERFECT and f["train_replace_f1"] == 1.0 for f in logo["folds"] + lodo["folds"])
    assert logo["f1"] == pytest.approx(1.0)
    diag = out["diagnostic"]
    assert len(diag["leaderboard"]) == len(SD.DETECTORS) + len(GRID) == 43
    assert diag["order"][0] == PERFECT and "pooled" in diag["leaderboard"]["high_rms"]
    assert "config_list" not in diag["leaderboard"] and "config_list_defined" not in diag["leaderboard"]
    assert out["rater_agreement"]["overall"], "pairwise agreement reported"


def test_selection_records_the_runs_its_cells_cannot_support(cohort):
    """Dev cells from one dataset, or from one cv_group, still give the pick, the combinations, the event cells and
    the test scoring; the leave-one-out runs they cannot support are recorded as not run, with the reason."""
    path = _write_features(cohort["tmp"], "one_dataset", _cells(DEV[:3], 0), "dev")  # ds1: groups g1 and g2
    out = cohort["tmp"] / "one_dataset_selection.json"
    SD.main(_select_args(cohort, path, out))
    result = json.loads(out.read_text())
    assert result["pick"]["autoreject_config"] == PERFECT
    assert [f["held_out"] for f in result["nested_replace_logo"]["folds"]] == ["g1", "g2"]
    lodo = result["nested_replace_lodo"]
    assert lodo["not_run"] == "leaving one dataset out needs at least 2, and the cells hold 1"
    assert np.isnan(lodo["f1"]) and lodo["folds"] == [] and lodo["pooled"] is None
    diag = result["diagnostic"]
    assert len(diag["order"]) == 43 and np.isnan(diag["leaderboard"]["high_rms"]["lodo_f1"])
    assert diag["nested_logo"]["not_run"].endswith("the part without cv_group g1 holds 1")
    assert diag["nested_lodo"]["not_run"] == lodo["not_run"]

    path = _write_features(cohort["tmp"], "one_group", _cells(DEV[:2], 0), "dev")  # d1 and d2 share g1
    out = cohort["tmp"] / "one_group_selection.json"
    SD.main(_select_args(cohort, path, out))
    result = json.loads(out.read_text())
    one_group = "leaving one cv_group out needs at least 2, and the cells hold 1"
    assert result["nested_replace_logo"]["not_run"] == one_group and result["diagnostic"]["not_run"] == one_group
    assert result["diagnostic"]["order"] == [] and result["combinations"]["replace"]["pooled"]["f1"] == 1.0
    assert all(np.isnan(v) for v in result["combinations"]["replace"]["ci95"]), "one group: nothing to resample"
    events = cohort["tmp"] / "one_group_events.csv"
    EV.main(["--features", str(path), "--selection", str(out), "--split", str(cohort["split"]), "--out", str(events)])
    no_lists = [c for c in SD.COMBINATIONS if c != "config_lists_alone"]  # only the ds2 animals have lists
    assert list(pd.unique(pd.read_csv(events)["detector"])) == no_lists
    ST.main(_score_args(cohort, cohort["test"], "one_group_score.json", pick=out))
    scored = json.loads((cohort["tmp"] / "one_group_score.json").read_text())
    assert scored["pick"]["autoreject_config"] == PERFECT and list(scored["combinations"]) == list(SD.COMBINATIONS)


def test_selection_refuses_test_animals_unknown_columns_and_missing_lists(cohort):
    mixed = pd.concat([pd.read_parquet(cohort["dev"]), pd.read_parquet(cohort["test"]).head(4)])
    path = _write_features(cohort["tmp"], "mixed", mixed, "dev")
    with pytest.raises(SystemExit, match="outside split"):
        SD.main(_select_args(cohort, path, cohort["tmp"] / "x.json"))
    extra = pd.read_parquet(cohort["dev"]).assign(mystery=1)
    path = _write_features(cohort["tmp"], "extra", extra, "dev")
    with pytest.raises(SystemExit, match="unknown role"):
        SD.main(_select_args(cohort, path, cohort["tmp"] / "y.json"))
    no_lists = pd.read_parquet(cohort["dev"]).drop(columns=["config_list", "config_list_defined"])
    path = _write_features(cohort["tmp"], "no_lists", no_lists, "dev")
    with pytest.raises(SystemExit, match="lack"):
        SD.main(_select_args(cohort, path, cohort["tmp"] / "z.json"))


@pytest.mark.parametrize("top, which, match", [
    ({"src_dirty": True}, "dev", "dev features: src_dirty is True; src must be committed and clean"),
    ({"src_dirty": None}, "dev", "dev features: src_dirty is None"),
    ({"src_tree": None}, "dev", "dev features: src_tree not recorded"),
    ({}, "test", "dev features: the provenance says which 'test'"),
    ({"outputs": {"features": {"sha256": "f" * 64}}}, "dev", "is not the parquet its provenance describes"),
])
def test_selection_refuses_features_score_test_would_refuse(cohort, top, which, match):
    """Dirty or unrecorded src, another partition, or a provenance that describes another parquet."""
    path = _write_features(cohort["tmp"], "dev_other", pd.read_parquet(cohort["dev"]), which, top)
    out = cohort["tmp"] / "other_selection.json"
    with pytest.raises(SystemExit, match=re.escape(match)):
        SD.main(_select_args(cohort, path, out))
    assert not out.exists()


def test_a_provenance_without_its_parquet_is_refused(cohort):
    """An extraction stopped after writing the provenance and before renaming the parquet into place."""
    dev = _write_features(cohort["tmp"], "dev_gone", pd.read_parquet(cohort["dev"]), "dev")
    test = _write_features(cohort["tmp"], "test_gone", pd.read_parquet(cohort["test"]), "test")
    dev.unlink()
    test.unlink()
    with pytest.raises(SystemExit, match="dev_gone.parquet not found beside its provenance"):
        SD.main(_select_args(cohort, dev, cohort["tmp"] / "gone.json"))
    with pytest.raises(SystemExit, match="test_gone.parquet not found beside its provenance"):
        ST.main(_score_args(cohort, test, "gone_score.json"))
    with pytest.raises(SystemExit, match="dev_gone.parquet not found"):
        EV.main(["--features", str(dev), "--selection", str(cohort["selection"]), "--out", str(cohort["tmp"] / "e.csv")])
    assert not (cohort["tmp"] / "gone.json").exists() and not (cohort["tmp"] / "gone_score.json").exists()
    assert _claimed(cohort) == []


def test_score_test_scores_the_combinations_once(cohort, capsys):
    out = cohort["tmp"] / "test_score.json"
    args = _score_args(cohort, cohort["test"], "test_score.json")
    ST.main(args)
    result = json.loads(out.read_text())
    assert set(result) == {"pick", "settings", "member_masks", "combinations", "event_cells", "rater_agreement",
                           "features", "provenance"}
    assert result["pick"]["autoreject_config"] == PERFECT
    assert result["pick"]["sha256"] == SD._sha256(cohort["selection"])
    assert result["features"]["sha256"] == SD._sha256(cohort["test"])
    combos = result["combinations"]
    assert list(combos) == list(SD.COMBINATIONS)
    # The frozen k is noise on the test cells, and another column would complete replace there.
    test = pd.read_parquet(cohort["test"])
    f, _ = SD.fragment_reject(test, SD.load_settings(cohort["config"], datasets=DATASETS))
    frozen = f1_score(test["y"], f | test[PERFECT])
    assert combos["replace"]["pooled"]["f1"] == pytest.approx(frozen) and frozen < 0.6
    assert combos["autoreject_alone"]["pooled"]["f1"] == pytest.approx(f1_score(test["y"], test[PERFECT]))
    assert SD.choose_config(test, GRID, f)[0] == TEST_PERFECT != PERFECT
    assert set(combos["replace"]["per_dataset"]) == set(DATASETS)
    assert set(combos["replace"]["per_animal"]) == {a for a, _, _ in TEST}
    assert combos["config_lists_alone"]["pooled"]["n"] == 2 * 160  # t2 and t3, the ds2 animals
    assert set(combos["add"]["minus_today"]) == {"f1_difference", "ci95"}
    events = pd.DataFrame(result["event_cells"])
    assert set(events["detector"]) == set(SD.COMBINATIONS)
    assert events.loc[events["detector"] == "config_lists_alone", "n_cells"].sum() == 2 * 160
    assert result["rater_agreement"]["overall"]
    assert "replace minus today" in capsys.readouterr().out
    assert _claimed(cohort) == ["dev_selection.json.test_scored", "test.parquet.test_scored"]
    with pytest.raises(SystemExit, match="scored once"):
        ST.main(args)


def test_score_test_runs_once_per_selection_and_per_test_parquet(cohort):
    """A second run is refused at another output path, with a copy of the selection, or with a copy of the
    test parquet."""
    ST.main(_score_args(cohort, cohort["test"], "first.json"))
    marker = json.loads((cohort["tmp"] / "dev_selection.json.test_scored").read_text())
    assert marker == {"out": str(cohort["tmp"] / "first.json"), "pick": str(cohort["selection"]),
                      "pick_sha256": SD._sha256(cohort["selection"]), "features": str(cohort["test"]),
                      "features_sha256": SD._sha256(cohort["test"])}
    with pytest.raises(SystemExit, match="scored once") as err:
        ST.main(_score_args(cohort, cohort["test"], "second.json"))
    assert "dev_selection.json.test_scored exists" in str(err.value) and "first.json" in str(err.value)
    pick_copy = cohort["tmp"] / "pick_copy.json"
    shutil.copy(cohort["selection"], pick_copy)
    with pytest.raises(SystemExit, match="test.parquet.test_scored exists"):
        ST.main(_score_args(cohort, cohort["test"], "third.json", pick=pick_copy))
    test_copy = _write_features(cohort["tmp"], "test_copy", pd.read_parquet(cohort["test"]), "test")
    with pytest.raises(SystemExit, match="dev_selection.json.test_scored exists"):
        ST.main(_score_args(cohort, test_copy, "fourth.json"))
    assert not any((cohort["tmp"] / name).exists() for name in ("second.json", "third.json", "fourth.json"))


@pytest.mark.parametrize("rel", SD.SCORING_CODE)
def test_score_test_refuses_scoring_code_the_selection_did_not_record(cohort, rel):
    selection = json.loads(cohort["selection"].read_text())
    selection["provenance"]["code"][rel] = "0" * 64
    pick = cohort["tmp"] / "drift.json"
    pick.write_text(json.dumps(selection))
    with pytest.raises(SystemExit, match="other scoring code") as err:
        ST.main(_score_args(cohort, cohort["test"], "q.json", pick=pick))
    assert f"{rel}: the selection recorded sha256 '{'0' * 64}'" in str(err.value)
    assert sum(r in str(err.value) for r in SD.SCORING_CODE) == 1
    del selection["provenance"]
    pick.write_text(json.dumps(selection))
    with pytest.raises(SystemExit, match="other scoring code") as err:
        ST.main(_score_args(cohort, cohort["test"], "q.json", pick=pick))
    assert all(f"{r}: the selection recorded sha256 None" in str(err.value) for r in SD.SCORING_CODE)
    assert not (cohort["tmp"] / "q.json").exists() and _claimed(cohort) == []


def test_score_test_refuses_a_parquet_its_provenance_does_not_describe(cohort):
    test = _write_features(cohort["tmp"], "test_swapped", pd.read_parquet(cohort["test"]), "test",
                           {"outputs": {"features": {"path": "test_swapped.parquet", "sha256": "f" * 64}}})
    with pytest.raises(SystemExit, match="is not the parquet its provenance describes"):
        ST.main(_score_args(cohort, test, "r.json"))
    assert not (cohort["tmp"] / "r.json").exists() and _claimed(cohort) == []


@pytest.mark.parametrize("change, match", [
    (lambda df: df.drop(columns=[c for c in df.columns if c.startswith("cat_")]), "no rater category columns"),
    (lambda df: df.drop(columns="genotype"), "the features lack ['genotype']"),
    (lambda df: df.drop(columns="isday"), "the features lack ['isday']"),
    (lambda df: df.assign(y=np.where(np.arange(len(df)) == 0, 2, df["y"])), "y holds values other than 0 and 1"),
])
def test_score_test_checks_every_column_it_reads_before_claiming(cohort, change, match):
    test = _write_features(cohort["tmp"], "test_cols", change(pd.read_parquet(cohort["test"])), "test")
    with pytest.raises(SystemExit, match=re.escape(match)):
        ST.main(_score_args(cohort, test, "c.json"))
    assert not (cohort["tmp"] / "c.json").exists() and _claimed(cohort) == []


def test_score_test_refuses_a_missing_or_malformed_pick(cohort):
    with pytest.raises(SystemExit, match="no frozen pick"):
        ST.main(_score_args(cohort, cohort["test"], "a.json", pick=cohort["tmp"] / "none.json"))
    bad = cohort["tmp"] / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(SystemExit, match="not a selection JSON"):
        ST.main(_score_args(cohort, cohort["test"], "b.json", pick=bad))
    selection = json.loads(cohort["selection"].read_text())
    for key, value, match in [("pick", {"autoreject_config": "welch/v9"}, "not a grid configuration"),
                              ("settings", {"lof": {}}, "lacks the settings"),
                              ("features", {}, "provenance")]:
        broken = cohort["tmp"] / f"broken_{key}.json"
        broken.write_text(json.dumps({**selection, key: value}))
        with pytest.raises(SystemExit, match=match):
            ST.main(_score_args(cohort, cohort["test"], f"c_{key}.json", pick=broken))
    outputs = ["a.json", "b.json", "c_pick.json", "c_settings.json", "c_features.json"]
    assert not any((cohort["tmp"] / name).exists() for name in outputs), "no refusal claims the output"
    assert _claimed(cohort) == []


def test_score_test_refuses_changed_settings(cohort):
    _write_config(cohort["config"], reject_lof_threshold=3.0)
    with pytest.raises(SystemExit, match="settings differ") as err:
        ST.main(_score_args(cohort, cohort["test"], "s.json"))
    assert "lof: recorded" in str(err.value)
    _write_config(cohort["config"])
    _write_datasets(cohort["tmp"], channels=[*CHANNELS, "LHip"])
    with pytest.raises(SystemExit, match="settings differ") as err:
        ST.main(_score_args(cohort, cohort["test"], "s.json"))
    assert "canonical_channels: recorded" in str(err.value)
    assert not (cohort["tmp"] / "s.json").exists() and _claimed(cohort) == []


@pytest.mark.parametrize("top, params, match", [
    ({"src_tree": "tree1"}, {}, "src_tree: dev tree0, test tree1"),
    ({"src_tree": None}, {}, "src_tree not recorded"),
    ({"src_dirty": True}, {}, "test features: src_dirty is True"),
    ({"src_dirty": None}, {}, "src_dirty is None"),
    ({}, {"seed": 1}, "param seed"),
    ({}, {"temp_dir": "/scratch/other"}, "param temp_dir"),
    ({}, {"datasets": ["ds1"]}, "param datasets"),
    ({}, {"lof_chunk_duration_s": 30}, "param lof_chunk_duration_s: dev 60, test 30"),
    ({}, {"chunk_duration_s": 1800}, "param chunk_duration_s: dev 3600, test 1800"),
    ({}, {"which": "dev"}, "partitions"),
    ({"inputs": {"recording_map": "map1"}}, {}, "input recording_map"),
    ({"inputs": {"config/datasets/ds2.yaml": "edited"}}, {}, "input config/datasets/ds2.yaml"),
    ({"inputs": {"config/config.yaml": "edited"}}, {}, "input config/config.yaml"),
    ({"inputs": {"keymap": "other"}}, {}, "input keymap"),
    ({"inputs": {"config/config.local.yaml": "new"}}, {}, "input config/config.local.yaml: sha256 dev None"),
    ({"code": {}}, {}, "code scripts/labeling/extract_dev_features.py differs"),
])
def test_score_test_refuses_features_built_differently(cohort, top, params, match):
    params, top = dict(params), dict(top)
    which = params.pop("which", "test")
    if "inputs" in top:  # change the sha256 of one input, or add one
        inputs = _provenance("test")["inputs"]
        top["inputs"] = {**inputs, **{k: {"path": k, "sha256": v} for k, v in top["inputs"].items()}}
    test = _write_features(cohort["tmp"], "test_other", pd.read_parquet(cohort["test"]), which, top, **params)
    with pytest.raises(SystemExit, match="produced differently") as err:
        ST.main(_score_args(cohort, test, "p.json"))
    assert match in str(err.value)
    assert not (cohort["tmp"] / "p.json").exists()


def test_matched_code_is_score_detectors_code_files():
    assert ST.matched_code() == [str(p.relative_to(SDET.REPO)) for p in SDET.CODE_FILES]
    assert "scripts/labeling/extract_dev_features.py" in ST.matched_code()


def test_score_test_refuses_dev_animals(cohort):
    dev_as_test = _write_features(cohort["tmp"], "dev_as_test", pd.read_parquet(cohort["dev"]), "test")
    with pytest.raises(SystemExit, match="outside split"):
        ST.main(_score_args(cohort, dev_as_test, "d.json"))


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
    assert list(pd.unique(table["detector"])) == [*SD.COMBINATIONS, PERFECT, "high_rms"]
    for name in ("replace", PERFECT):
        rows = table[table["detector"] == name].set_index("coarse_class")
        assert rows.loc["bad_and_clean", "rejection_rate"] == 1.0  # the y=1 cells: two bad votes, one clean
        assert rows.loc["event_no_bad", "rejection_rate"] == 0.0
    lists = table[table["detector"] == "config_lists_alone"]
    assert lists["n_cells"].sum() == 2 * 160 and lists["n_rejected"].sum() == 2 * 40
    today = table[table["detector"] == "today"]
    assert today["n_cells"].sum() == len(DEV) * 160 and today["n_rejected"].sum() == 3 * 40 + 2 * 40
    other = _write_features(cohort["tmp"], "dev_other", pd.read_parquet(cohort["dev"]).iloc[1:], "dev")
    with pytest.raises(SystemExit, match="not the parquet the selection was made on"):
        EV.main(["--features", str(other), "--selection", str(cohort["selection"]), "--split", str(cohort["split"]),
                 "--out", str(cohort["tmp"] / "other_events.csv")])
