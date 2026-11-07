"""Tests for ``scripts/labeling/extract_dev_features.py`` and the provenance it shares with ``score_detectors.py``.

Unit tests run ``main`` on a synthetic cohort: the loader and the fit-pool cutter are replaced with fakes
built here, the WAR is a real WindowAnalysisResult of synthetic rows, the consensus is synthetic, and the
autoreject grid is the real one on small arrays. They cover the config lists, the filter scores against the
pipeline's own filters, the refusal of unmapped recordings before anything loads, per-animal failures, resume
with byte-identical outputs, the refusal of a mapped recording without cells, what is printed for each split,
the autoreject scores file, the provenance, files that change during a run and outputs an earlier run left,
and the dev and test outputs as ``select_detectors``, ``report_event_cells`` and ``score_test`` read them. A
slow test runs ``main`` end to end on mini_real with fake raters. No real label file is read.
"""

import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
import zlib
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import zscore

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "labeling"))
import build_cohort_bundle as C  # noqa: E402
import extract_dev_features as EDF  # noqa: E402
import score_detectors as SDET  # noqa: E402
from neurodent.results import WindowAnalysisResult  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402

GRID = adr.grid_config_names()
# The fragment filters at their config/config.yaml settings: name: (WAR accessor, params, reject from the cells).
FILTERS = {"logrms_range": ("get_filter_logrms_range", {"z_range": 3}, lambda d: d["logrmsz_abs"] > 3),
           "high_rms": ("get_filter_high_rms", {"max_rms": 500}, lambda d: d["rms"] > 500),
           "low_rms": ("get_filter_low_rms", {"min_rms": 50}, lambda d: d["rms"] < 50),
           "high_beta": ("get_filter_high_beta", {"max_beta_prop": 0.4}, lambda d: d["beta_wmax"] > 0.4)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    """Return the output of a git command run in the repository."""
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def _read_back(war, folder):
    """Return ``war`` as the pipeline's fragment filter reads it: saved to parquet and loaded again."""
    war.save_parquet_and_json(folder, filename="war")
    return WindowAnalysisResult.load_parquet_and_json(folder, filename="war")


def _war_rows(cells, war):
    """Return the WAR row and column of each cell: the window within its animalday's rows, and its channel."""
    animaldays = war.result["animalday"].to_numpy()
    first = {ad: int(np.flatnonzero(animaldays == ad)[0]) for ad in pd.unique(animaldays)}
    rows = (cells["animalday"].map(first) + cells["window"]).to_numpy()
    cols = cells["channel"].map({c: j for j, c in enumerate(war.channel_abbrevs)}).to_numpy()
    return rows, cols


def _check_filters(cells, piped):
    """Assert that one animal's extracted scores give the pipeline's decisions on the WAR it reads back.

    logrmsz_abs must also equal, exactly, the absolute z score the pipeline's logrms_range takes, which runs in
    float64 on the values read back from parquet.

    Returns:
        ``{filter: number of the cells the pipeline rejects}``.
    """
    rows, cols = _war_rows(cells, piped)
    rejected = {}
    for name, (method, params, ours) in FILTERS.items():
        reject = ~getattr(piped, method)(**params)[rows, cols]
        assert (ours(cells).to_numpy() == reject).all(), name
        rejected[name] = int(reject.sum())
    z = zscore(np.log(extract_linear_array(piped.result["rms"])), axis=0, nan_policy="omit")
    assert (cells["logrmsz_abs"].to_numpy() == np.abs(z[rows, cols])).all(), "logrmsz_abs is the pipeline's z"
    return rejected


# --------------------------------------------------------------------------- config lists


@pytest.mark.unit
def test_date_named():
    assert EDF.date_named("M3 MHet Jan-08-2022") == date(2022, 1, 8)
    assert EDF.date_named("cohort 2 group 6 mouse M3 cage3A Het Mar-19-2021") == date(2021, 3, 19)
    assert EDF.date_named("A10_062921") is None
    assert EDF.date_named("") is None


@pytest.mark.unit
def test_list_form_applies_to_every_recording():
    dates = {"a__0": date(2022, 1, 8), "a__1": date(2022, 1, 9)}
    lists, matches, unmatched = EDF.match_config_lists({"_all": ["LMot", "RMot"]}, dates)
    assert lists == {"a__0": ["LMot", "RMot"], "a__1": ["LMot", "RMot"]}
    assert matches == [{"key": "_all", "recording": "a__0"}, {"key": "_all", "recording": "a__1"}]
    assert unmatched == []
    assert EDF.match_config_lists({"_all": []}, dates)[0] == {"a__0": [], "a__1": []}, "an empty list is a list"
    assert EDF.match_config_lists(None, dates) == ({}, [], [])
    assert EDF.match_config_lists({}, dates) == ({}, [], [])


@pytest.mark.unit
def test_dict_form_matches_by_date_and_never_guesses():
    dates = {"a__0": date(2022, 3, 17), "a__1": date(2022, 3, 18), "a__2": date(2022, 3, 19)}
    entry = {
        "OLD MMut Mar-17-2022": ["LAud", "LHip"],   # the id and genotype spellings are not read
        "OLD MMut Mar-19-2022": ["RHip"],
        "OLD MMut Mar-19-2022 copy": ["RAud"],      # no date as its last word
        "OLD MMut Mar-21-2022": ["LAud"],           # no labeled recording that day
        "Other Mar-19-2022": None,                  # an empty YAML entry: a list with no channels
    }
    lists, matches, unmatched = EDF.match_config_lists(entry, dates)
    assert lists == {"a__0": ["LAud", "LHip"], "a__2": ["RHip"]}
    assert "a__1" not in lists, "a recording no key names has no list"
    assert matches == [{"key": "OLD MMut Mar-17-2022", "recording": "a__0"},
                       {"key": "OLD MMut Mar-19-2022", "recording": "a__2"},
                       {"key": "Other Mar-19-2022", "recording": "a__2"}]
    reasons = {u["key"]: u["reason"] for u in unmatched}
    assert reasons == {"OLD MMut Mar-19-2022 copy": "names no date in the form %b-%d-%Y",
                       "OLD MMut Mar-21-2022": "no labeled recording on Mar-21-2022"}

    two = {"b__0": date(2022, 1, 8), "b__1": date(2022, 1, 8)}
    _, matches, unmatched = EDF.match_config_lists({"B Het Jan-08-2022": ["LMot"]}, two)
    assert matches == [] and unmatched == [
        {"key": "B Het Jan-08-2022", "reason": "2 labeled recordings on Jan-08-2022: ['b__0', 'b__1']"}]

    undated = {"c__0": date(2022, 1, 8), "c__1": None}
    lists, _, unmatched = EDF.match_config_lists({"_all": ["LMot"], "C WT Jan-08-2022": ["RMot"]}, undated)
    assert lists == {"c__0": ["LMot"], "c__1": ["LMot"]}, "a list-form entry still applies everywhere"
    assert unmatched == [{"key": "C WT Jan-08-2022", "reason": "labeled recordings with no dated animalday: "
                                                               "['c__1']"}]


@pytest.mark.unit
def test_resolve_list_reports_names_the_recording_lacks():
    def resolve(name):
        known = {"LMot": "LMot", "EEG E9-REF2": "LMot", "RAud": "RAud"}
        if name not in known:
            raise ValueError(name)
        return known[name]

    found, unknown = EDF.resolve_list(["EEG E9-REF2", "RAud", "Bogus"], ["LMot", "RMot"], resolve)
    assert found == {"LMot"}
    assert unknown == [{"channel": "RAud", "abbrev": "RAud", "reason": "RAud is not among the recording's channels"},
                       {"channel": "Bogus", "reason": "not in the dataset's channel map"}]
    _, unknown = EDF.resolve_list(["EEG E9-REF2"], ["RMot"], resolve)
    assert unknown == [{"channel": "EEG E9-REF2", "abbrev": "LMot", "reason": "LMot is not among the recording's "
                                                                              "channels"}], "a raw name's abbreviation"


# --------------------------------------------------------------------------- recording map


@pytest.mark.unit
def test_recording_map_status_reasons_and_position_fallback(tmp_path):
    path = tmp_path / "recording_map.csv"
    pd.DataFrame([
        {"july_recording": "A__0", "animal": "A", "animalday": "A WT Jan-02-2025", "port_position": "1",
         "status": "mapped", "reason": "", "extra": "ignored"},
        {"july_recording": "A__1", "animal": "A", "animalday": "", "port_position": "",
         "status": "unmapped", "reason": "cells differ"},
        {"july_recording": "A__2", "animal": "A", "animalday": "", "port_position": "3",
         "status": "shifted", "reason": "shifted by window offset +1 at port position 3"},
        {"july_recording": "032__0", "animal": "032", "animalday": "", "port_position": "0",
         "status": "mapped", "reason": ""},
    ]).to_csv(path, index=False)
    rmap = SDET.RecordingMap(path)
    assert rmap.label("A", 1, "A WT Jan-02-2025") == "A__0"
    assert rmap.label("A", 0, "A WT Jan-01-2025") is None
    assert rmap.label("A", 1, "A WT Jan-05-2025") is None, "a row with an animalday is not matched by position"
    assert rmap.label("A", 3, "A WT Jan-04-2025") is None, "only mapped rows map"
    assert rmap.label("032", 0, "032 WT Jan-01-2025") == "032__0", "ids keep their leading zeros"
    assert rmap.unmapped(["A__0", "A__1", "A__2", "A__3", "032__0"]) == [
        {"recording": "A__1", "animal": "A", "status": "unmapped", "reason": "cells differ"},
        {"recording": "A__2", "animal": "A", "status": "shifted",
         "reason": "shifted by window offset +1 at port position 3"},
        {"recording": "A__3", "animal": "A", "status": "absent", "reason": "not in the recording map"},
    ]
    assert SDET.RecordingMap(positional=True).unmapped(["A__0"]) == []


# --------------------------------------------------------------------------- provenance


@pytest.mark.unit
def test_code_files_cover_the_package_and_the_feature_scripts():
    import select_detectors as SD

    files = set(SDET.CODE_FILES)
    assert set((REPO / "src/neurodent").rglob("*.py")) <= files
    for rel in ("scripts/labeling/extract_dev_features.py", "scripts/labeling/score_detectors.py",
                "scripts/labeling/build_cohort_bundle.py", "scripts/labeling/render_context.py", "uv.lock"):
        assert REPO / rel in files, rel
    for rel in SD.SCORING_CODE:  # recorded by the selection, so editing one does not invalidate an extraction
        assert REPO / rel not in files, rel
    assert SDET.CODE_FILES[:2] == [REPO / "src/neurodent/results/autoreject_detector.py",
                                   REPO / "src/neurodent/results/scoring.py"], "the listed entries stay first"
    assert len(files) == len(SDET.CODE_FILES), "no file is listed twice"


@pytest.mark.unit
def test_provenance_records_the_src_tree_and_the_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    marker = tmp_path / "input.txt"
    marker.write_text("x")
    inputs = {"marker": marker, **SDET.config_files(["mini_real"])}
    prov = SDET.provenance(inputs, script="test")
    assert {"git_commit", "git_dirty", "src_tree", "src_dirty", "inputs", "code", "params"} <= set(prov)
    assert prov["src_tree"] == _git("rev-parse", "HEAD:src").strip(), "the tree of src at HEAD, not the commit"
    assert prov["src_dirty"] is bool(_git("status", "--porcelain", "--", "src").strip())
    for rel in ("config/config.yaml", "config/datasets/mini_real.yaml"):
        assert prov["inputs"][rel] == {"path": str(REPO / rel), "sha256": _sha(REPO / rel)}
    assert prov["code"]["uv.lock"] == _sha(REPO / "uv.lock")
    assert prov["params"] == {"script": "test"}


@pytest.mark.unit
def test_score_detectors_refuses_another_working_directory(tmp_path, monkeypatch):
    """A copy of the configs in the working directory would load while the provenance hashes the repository's."""
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config/config.yaml", tmp_path / "config/config.yaml")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run from the repository root"):
        SDET.main(["A=labels_A.csv", "--all"])


# --------------------------------------------------------------------------- autoreject evidence


@pytest.mark.unit
def test_ar_cell_evidence_reads_scores_and_cutoffs_from_the_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 50e-6, (10, 3, 2500))
    X[4, 1] *= 40.0
    fits = adr.fit_grid(X, 500)
    cons = pd.DataFrame([{"recording": "a__0", "window": w, "channel": ch, "t_start_s": 5.0 * w}
                         for w in (2, 4) for ch in ("LMot", "RMot")])
    ev = EDF.ar_cell_evidence(fits, ["LMot", "RMot", "LHip"], np.arange(10) * 5.0, cons)
    bits = EDF.ar_cell_bits({n: f.mask for n, f in fits.items()}, ["LMot", "RMot", "LHip"], np.arange(10) * 5.0,
                            cons)
    assert set(ev) == set(bits) == {(2, "LMot"), (2, "RMot"), (4, "LMot"), (4, "RMot")}
    for (w, ch), per_config in ev.items():
        ci = ["LMot", "RMot", "LHip"].index(ch)
        for name, (score, cutoff) in per_config.items():
            assert score == fits[name].scores[w, ci] and cutoff == fits[name].thresholds[ci]
            assert bits[(w, ch)][name] == int(score > cutoff)


# --------------------------------------------------------------------------- main on a synthetic cohort

DS = "mini_real"  # a dataset with a config file, so the config hashes are real; its animals are faked
CH = ["LMot", "RMot", "LHip"]
FS = 500
N_FRAG = 12
WINDOWS = [2, 5, 8]
ANIMALS = {"a1": ("WT", ["Jan-01-2025", "Jan-02-2025"]),
           "a2": ("Het", ["Jan-01-2025", "Jan-02-2025"]),
           "a3": ("WT", ["Jan-03-2025"])}
UNLABELED = {"a3": ["Jan-04-2025"]}  # recordings in the animal's WAR that no rater saw
CANONICAL = {"LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"}


def _animalday(animal, i):
    genotype, days = ANIMALS[animal]
    return f"{animal} {genotype} {(days + UNLABELED.get(animal, []))[i]}"


def _recordings():
    return [(a, i) for a, (_, days) in ANIMALS.items() for i in range(len(days))]


def _consensus():
    rng = np.random.default_rng(1)
    rows = []
    for a, i in _recordings():
        for w in WINDOWS:
            for ch in CH:
                y = int(rng.random() < 0.3)
                rows.append({"recording": f"{a}__{i}", "window": w, "channel": ch, "t_start_s": 5.0 * w,
                             "n_raters": 2, "any_event": False, "y_true": y,
                             "cat_A": "artifact" if y else "clean", "cat_B": "artifact" if y else "event"})
    return pd.DataFrame(rows)


def _fake_animal(animal, nan):
    """Return a fake AnimalOrganizer and a real WindowAnalysisResult for one synthetic animal.

    rms is float32, as in a WAR the loader computes; psdband and psdtotal are float64, as there. Each fragment
    filter rejects some labeled cell:

    - a1: the second recording runs about four times louder than the first, with one window at the first one's
      level; the first recording has a window whose beta proportion is 0/0 on RMot and over 0.4 on LHip.
    - a2: one LMot cell over 500 and one RMot cell under 50; two LHip windows of the first recording are loud
      enough that their log-RMS z score over the whole WAR is over 3, but not over their recording alone.
    - a3: its unlabeled second recording runs three times louder, and one LMot window of the labeled one is
      over 3 in z for that recording alone, but not over the whole WAR.
    """
    genotype, _ = ANIMALS[animal]
    days = ANIMALS[animal][1] + UNLABELED.get(animal, [])
    rng = np.random.default_rng(zlib.crc32(animal.encode()))
    animaldays = [_animalday(animal, i) for i in range(len(days))]
    n = N_FRAG * len(days)
    rms = rng.uniform(80.0, 120.0, (n, len(CH)))
    beta = rng.uniform(0.05, 0.3, (n, len(CH)))
    total = np.full((n, len(CH)), 2.0)
    w0, w1, w2 = WINDOWS
    if animal == "a1":
        rms[N_FRAG:] *= 4.0
        rms[N_FRAG + w1, 0] = 120.0  # low for its own recording, ordinary for the animal
        beta[w2, 2], total[w2, 1] = 0.5, 0.0  # over 0.4 on LHip; 0/0 on RMot
    if animal == "a2":
        rms[w0, 0], rms[N_FRAG + w2, 1] = 650.0, 30.0
        rms[[w0, w2], 2] = 1000.0
    if animal == "a3":
        rms[N_FRAG:] *= 3.0
        rms[w1, 0] = 1000.0
    rms = rms.astype(np.float32)
    band = beta * total
    rows = []
    for k, ad in enumerate(animaldays):
        start = pd.Timestamp(datetime.strptime(ad.split()[-1], "%b-%d-%Y")) + pd.Timedelta(hours=6)
        for w in range(N_FRAG):
            i = k * N_FRAG + w
            rows.append({"animal": animal, "animalday": ad, "genotype": genotype, "duration": 5.0,
                         "isday": bool(w % 2), "timestamp": start + pd.Timedelta(seconds=5 * w),
                         "rms": rms[i], "psdtotal": total[i].tolist(),
                         "psdband": {b: (band[i] if b == "beta" else np.full(len(CH), 0.1)).tolist()
                                     for b in ("delta", "theta", "alpha", "beta", "gamma")}})
    lof = {ad: {"lof_scores": list(rng.uniform(1.0, 3.0, len(CH))), "channel_names": list(CH)} for ad in animaldays}
    war = WindowAnalysisResult(pd.DataFrame(rows), animal_id=animal, genotype=genotype, channel_names=list(CH),
                               lof_scores_dict=lof)
    lros = [SimpleNamespace(name=f"{animal}/{i}", nan=nan) for i in range(len(days))]
    return SimpleNamespace(animaldays=animaldays, long_recordings=lros), war


def _fake_fit_epochs(lro, needed_windows=None, *, fragment_len_s=5.0, max_fit_fragments=None, seed=0, **_):
    rng = np.random.default_rng(zlib.crc32(lro.name.encode()))
    X = rng.normal(0.0, 50e-6, (N_FRAG, len(CH), int(fragment_len_s * FS)))
    X[WINDOWS[0], 1] *= 30.0  # one loud fragment
    if lro.nan:
        X[1, 0, 7] = np.nan
    return adr.FitEpochs(X, list(CH), FS, np.arange(N_FRAG) * fragment_len_s)


def _resolve(name):
    if name not in CANONICAL:
        raise ValueError(f"{name} is not in the channel map")
    return name


def _freeze_code_state(monkeypatch):
    """Pin the code hashes and the src state at their current values, so files edited while a test runs
    cannot change a run's shard key between two runs."""
    code, state = SDET.code_hashes(), SDET.src_state()
    monkeypatch.setattr(SDET, "code_hashes", lambda: dict(code))
    monkeypatch.setattr(SDET, "src_state", lambda: dict(state))
    return code, state


class FakeRun:
    """A synthetic cohort wired into ``EDF.main``; records what was loaded and whether dask started."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.out = tmp_path / "out" / "features.parquet"
        self.scores = self.out.with_name("features.autoreject_scores.parquet")
        self.prov_path = Path(str(self.out) + ".provenance.json")
        self.errors_path = Path(str(self.out) + ".errors.json")
        self.parts = Path(str(self.out) + ".parts")
        self.cons = _consensus()
        self.bad_channels = {"a1": {"_all": ["LMot", "RAud", "Bogus"]},
                             "a2": {"A2 Het Jan-02-2025": ["RMot", "LHip"], "A2 Het Jan-09-2025": ["LMot"]}}
        self.fail, self.nan, self.loaded, self.clusters, self.wars = {}, set(), [], 0, {}
        self.raters = []
        for r in ("A", "B"):
            p = tmp_path / f"labels_{r}.csv"
            p.write_text(f"synthetic {r}\n")
            self.raters.append(f"{r}={p}")
        self.keymap = tmp_path / "keymap.csv"
        self.keymap.write_text("synthetic\n")
        self.write_split("dev")
        self.map_path = tmp_path / "recording_map.csv"
        self.write_map()

        def iter_cohort(datasets, *, animal=None, limit_per_dataset=None, seed=0):
            samples_config = {"bad_channels": self.bad_channels, "_animal_groups": {"a1": "g12", "a2": "g12"}}
            for a in ANIMALS:
                yield DS, samples_config, {}, a

        def load_and_measure(samples_config, config, animal_id, *, lof_chunk_duration_s, chunk_duration_s):
            self.loaded.append(animal_id)
            if animal_id in self.fail:
                raise self.fail[animal_id]
            ao, self.wars[animal_id] = _fake_animal(animal_id, animal_id in self.nan)
            return ao, self.wars[animal_id]

        @contextlib.contextmanager
        def dask_cluster(n_workers):
            self.clusters += 1
            yield None

        monkeypatch.chdir(REPO)
        self.code, self.state = _freeze_code_state(monkeypatch)
        monkeypatch.setattr(C, "iter_cohort_animals", iter_cohort)
        monkeypatch.setattr(EDF, "load_and_measure", load_and_measure)
        monkeypatch.setattr(EDF, "dask_cluster", dask_cluster)
        monkeypatch.setattr(EDF, "build_consensus", lambda manifests, keymap, rule: self.cons.copy())
        monkeypatch.setattr(EDF, "resolve_channel", _resolve)
        monkeypatch.setattr(adr, "build_fit_epochs", _fake_fit_epochs)

    def write_split(self, which):
        self.split = self.tmp / "split.json"
        other = "test" if which == "dev" else "dev"
        self.split.write_text(json.dumps({which: list(ANIMALS), other: []}))

    def write_map(self, unmapped=()):
        rows = []
        for a, i in _recordings():
            rec = f"{a}__{i}"
            if rec in unmapped:
                rows.append({"july_recording": rec, "animal": a, "animalday": "", "port_position": "",
                             "status": "unmapped", "reason": f"{rec} cells differ"})
            else:
                rows.append({"july_recording": rec, "animal": a, "animalday": _animalday(a, i), "port_position": i,
                             "status": "mapped", "reason": ""})
        pd.DataFrame(rows).to_csv(self.map_path, index=False)

    def run(self, *extra, which="dev", positional=True):
        argv = [*self.raters, "--keymap", str(self.keymap), "--split", str(self.split), "--which", which,
                "--datasets", DS, "--workers", "1", "--out", str(self.out), *extra]
        argv += ["--positional-recordings"] if positional else ["--recording-map", str(self.map_path)]
        return EDF.main(argv)

    def frame(self):
        return pd.read_parquet(self.out)

    def provenance(self):
        return json.loads(self.prov_path.read_text())

    def output_bytes(self):
        return self.out.read_bytes(), self.scores.read_bytes()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return FakeRun(tmp_path, monkeypatch)


@pytest.mark.unit
def test_cells_carry_config_lists(fake):
    assert fake.run() == 0
    df = fake.frame()
    assert list(df.columns) == EDF.CELL_COLS + ["cat_A", "cat_B"] + GRID
    assert len(df) == len(fake.cons) and not df[GRID].isna().any().any()
    assert df["config_list"].dtype == np.int64 and df["config_list_defined"].dtype == bool
    a1, a2, a3 = (df[df["animal"] == a] for a in ("a1", "a2", "a3"))
    assert a1["config_list_defined"].all() and (a1["config_list"] == (a1["channel"] == "LMot")).all()
    on_day2 = a2["recording"] == "a2__1"
    assert (a2["config_list_defined"] == on_day2).all()
    assert (a2["config_list"] == (on_day2 & a2["channel"].isin(["RMot", "LHip"]))).all()
    assert not a3["config_list_defined"].any() and not a3["config_list"].any()
    assert set(df["cv_group"]) == {"g12", "a3"} and set(df["dataset"]) == {DS}

    prov = fake.provenance()
    assert prov["config_list_unmatched"] == [{"animal": "a2", "dataset": DS, "key": "A2 Het Jan-09-2025",
                                              "reason": "no labeled recording on Jan-09-2025"}]
    assert {(m["animal"], m["key"], m["recording"], m["animalday"]) for m in prov["config_list_matches"]} == {
        ("a1", "_all", "a1__0", "a1 WT Jan-01-2025"), ("a1", "_all", "a1__1", "a1 WT Jan-02-2025"),
        ("a2", "A2 Het Jan-02-2025", "a2__1", "a2 Het Jan-02-2025")}
    assert prov["config_list_uncovered"] == [{"animal": "a2", "dataset": DS, "recording": "a2__0",
                                              "animalday": "a2 Het Jan-01-2025"}]
    unknown = {(u["recording"], u["channel"], u["reason"]) for u in prov["config_list_unknown_channels"]}
    assert unknown == {(rec, ch, reason) for rec in ("a1__0", "a1__1") for ch, reason in (
        ("RAud", "RAud is not among the recording's channels"), ("Bogus", "not in the dataset's channel map"))}


@pytest.mark.unit
def test_filter_scores_follow_the_pipeline_filters(fake, tmp_path):
    """At the config settings, the extracted scores give the decisions the pipeline's filters take on the WAR it
    reads back from parquet, which holds every recording of the animal, labeled or not."""
    assert fake.run() == 0
    df = fake.frame()
    rejected = dict.fromkeys(FILTERS, 0)
    piped = {}
    for animal, war in fake.wars.items():
        assert extract_linear_array(war.result["rms"]).dtype == np.float32, "rms as the loader computes it"
        piped[animal] = _read_back(war, tmp_path / "wars" / animal)
        for name, n in _check_filters(df[df["animal"] == animal], piped[animal]).items():
            rejected[name] += n
    assert all(rejected.values()), f"every filter must reject some labeled cell: {rejected}"

    # The planted windows tell the whole-WAR z from a z over one recording (a2), or over the labeled recordings
    # only (a3, whose louder second recording is unlabeled).
    cells = df.set_index(["recording", "window", "channel"])
    for rec, w, ch, whole in (("a2__0", WINDOWS[0], "LHip", True), ("a2__0", WINDOWS[2], "LHip", True),
                              ("a3__0", WINDOWS[1], "LMot", False)):
        animal, i = rec.split("__")
        result = piped[animal].result
        rms = extract_linear_array(result["rms"])[(result["animalday"] == _animalday(animal, int(i))).to_numpy()]
        z_recording = zscore(np.log(rms), axis=0, nan_policy="omit")[w, CH.index(ch)]
        assert (cells.loc[(rec, w, ch), "logrmsz_abs"] > 3) == whole and (abs(z_recording) > 3) != whole, rec
    assert cells.loc[("a1__1", WINDOWS[1], "LMot"), "logrmsz_abs"] < 1.0, "z over the whole WAR, not per recording"
    assert cells.loc[("a1__0", WINDOWS[2], "LMot"), "beta_wmax"] == pytest.approx(0.5), "a 0/0 hides nothing"


@pytest.mark.unit
def test_scores_file_holds_the_fit_scores_and_cutoffs(fake):
    assert fake.run() == 0
    df, scores = fake.frame(), pd.read_parquet(fake.scores)
    assert list(scores.columns) == EDF.SCORE_COLS and len(scores) == len(df) * len(GRID)
    assert list(scores["config"][:len(GRID)]) == GRID
    merged = scores.merge(df.melt(id_vars=["recording", "window", "channel"], value_vars=GRID,
                                  var_name="config", value_name="bit"),
                          on=["recording", "window", "channel", "config"], validate="one_to_one")
    assert len(merged) == len(scores)
    assert ((merged["score"] > merged["cutoff"]).astype(int) == merged["bit"]).all()
    assert scores.groupby(["recording", "channel", "config"])["cutoff"].nunique().eq(1).all(), \
        "one learned cutoff per channel, configuration and recording"
    assert fake.provenance()["outputs"]["autoreject_scores"] == {"path": str(fake.scores),
                                                                 "sha256": _sha(fake.scores)}


@pytest.mark.unit
def test_provenance_of_a_run(fake):
    assert fake.run() == 0
    prov = fake.provenance()
    assert prov["src_tree"] == _git("rev-parse", "HEAD:src").strip()
    assert prov["src_dirty"] is bool(_git("status", "--porcelain", "--", "src").strip())
    for a in ANIMALS:
        key = json.loads(EDF.shard_paths(fake.parts, DS, a)["json"].read_text())["key"]
        assert {k: key[k] for k in ("code", "src_tree", "src_dirty")} == {
            k: prov[k] for k in ("code", "src_tree", "src_dirty")}, "the state the shards were keyed under"
    for rel in ("config/config.yaml", f"config/datasets/{DS}.yaml"):
        assert prov["inputs"][rel]["sha256"] == _sha(REPO / rel)
    assert prov["inputs"]["labels_A"]["sha256"] == _sha(fake.tmp / "labels_A.csv")
    assert prov["outputs"]["features"] == {"path": str(fake.out), "sha256": _sha(fake.out)}
    params = prov["params"]
    assert params["which"] == "dev" and params["configs"] == GRID and params["datasets"] == [DS]
    assert params["dropped_recordings"] == [] and prov["dropped_animals"] == [] and prov["resumed_animals"] == []


@pytest.mark.unit
def test_unmapped_recordings_are_refused_before_anything_loads(fake):
    fake.write_map(unmapped={"a2__1"})
    with pytest.raises(SystemExit, match=r"leaves unmapped: \['a2__1'\]"):
        fake.run(positional=False)
    assert fake.loaded == [] and fake.clusters == 0
    assert not fake.out.exists() and not fake.parts.exists()


@pytest.mark.unit
def test_allow_unmapped_skips_animals_without_a_mapped_recording(fake):
    fake.write_map(unmapped={"a1__1", "a3__0"})
    assert fake.run("--allow-unmapped", positional=False) == 0
    assert fake.loaded == ["a1", "a2"], "an animal with no mapped labeled recording is never loaded"
    df = fake.frame()
    assert set(df["recording"]) == {"a1__0", "a2__0", "a2__1"}
    prov = fake.provenance()
    assert prov["dropped_animals"] == ["a3"]
    assert prov["unmapped_recordings"] == [
        {"recording": "a1__1", "animal": "a1", "status": "unmapped", "reason": "a1__1 cells differ"},
        {"recording": "a3__0", "animal": "a3", "status": "unmapped", "reason": "a3__0 cells differ"}]
    assert prov["params"]["dropped_recordings"] == ["a1__1", "a3__0"] and prov["params"]["allow_unmapped"]


@pytest.mark.unit
def test_failed_animals_are_listed_and_no_parquet_is_written(fake, capsys):
    fake.fail["a2"] = RuntimeError("cannot read the session files")
    fake.nan.add("a3")
    assert fake.run() == 1
    assert fake.loaded == ["a1", "a2", "a3"], "the run goes on after a failure"
    assert not fake.out.exists() and not fake.scores.exists() and not fake.prov_path.exists()
    errors = {e["animal"]: e for e in json.loads(fake.errors_path.read_text())}
    assert set(errors) == {"a2", "a3"}
    assert (errors["a2"]["error_type"], errors["a2"]["message"]) == ("RuntimeError", "cannot read the session files")
    assert errors["a3"]["error_type"] == "ValueError" and "NaN" in errors["a3"]["message"], "the detector's refusal"
    assert all(e["dataset"] == DS and "Traceback" in e["traceback"] for e in errors.values())
    assert "2 of 3 animal(s) failed" in capsys.readouterr().err
    assert EDF.shard_paths(fake.parts, DS, "a1")["json"].exists(), "the finished animal is kept"

    fake.fail.clear()
    fake.nan.clear()
    assert fake.run("--resume") == 0
    assert fake.loaded[3:] == ["a2", "a3"] and fake.provenance()["resumed_animals"] == ["a1"]
    assert not fake.errors_path.exists(), "a successful run removes the earlier errors file"


@pytest.mark.unit
def test_resume_reads_matching_shards_and_writes_the_same_bytes(fake):
    assert fake.run() == 0
    fresh = fake.output_bytes()

    assert fake.run("--resume") == 0
    assert fake.loaded == ["a1", "a2", "a3"] and fake.clusters == 1, "nothing reloaded, no dask started"
    assert fake.output_bytes() == fresh
    assert fake.provenance()["resumed_animals"] == ["a1", "a2", "a3"]

    # A shard computed under another key, and one whose file changed, are recomputed.
    a1, a3 = (EDF.shard_paths(fake.parts, DS, a) for a in ("a1", "a3"))
    doc = json.loads(a1["json"].read_text())
    doc["key"]["params"]["seed"] = 99
    a1["json"].write_text(json.dumps(doc))
    a3["cells"].write_bytes(a3["cells"].read_bytes() + b"x")
    assert fake.run("--resume") == 0
    assert fake.loaded[3:] == ["a1", "a3"] and fake.provenance()["resumed_animals"] == ["a2"]
    assert fake.output_bytes() == fresh

    # Without --resume every animal is recomputed, with the same outputs.
    assert fake.run() == 0
    assert fake.loaded[5:] == ["a1", "a2", "a3"] and fake.output_bytes() == fresh

    # Other parameters, or other code, match no shard.
    assert fake.run("--resume", "--ar-max-fit-fragments", "7") == 0
    assert fake.loaded[8:] == ["a1", "a2", "a3"] and fake.provenance()["resumed_animals"] == []
    assert fake.run("--resume") == 0
    assert fake.loaded[11:] == ["a1", "a2", "a3"]
    fake.code["scripts/labeling/extract_dev_features.py"] = "edited"
    assert fake.run("--resume") == 0
    assert fake.loaded[14:] == ["a1", "a2", "a3"] and fake.output_bytes() == fresh


def _stale(path):
    return path.with_name(path.name + ".stale")


@pytest.mark.unit
def test_a_failed_rerun_leaves_no_outputs_at_out(fake):
    """An earlier run's outputs are renamed to .stale when a run starts, so a failed rerun leaves none at --out."""
    assert fake.run() == 0
    first = [p.read_bytes() for p in (fake.out, fake.scores, fake.prov_path)]
    fake.code["scripts/labeling/extract_dev_features.py"] = "edited"
    fake.fail["a2"] = RuntimeError("cannot read the session files")
    assert fake.run() == 1
    assert fake.errors_path.exists()
    assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path))
    assert [_stale(p).read_bytes() for p in (fake.out, fake.scores, fake.prov_path)] == first

    # A run killed during the loop leaves nothing at --out either, and no errors file of the run before it.
    fake.fail["a2"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        fake.run()
    assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path, fake.errors_path))
    assert _stale(fake.errors_path).exists()


@pytest.mark.unit
def test_mapped_recording_without_cells_is_refused(fake):
    assert fake.run(positional=False) == 0
    late = fake.cons["recording"] == "a1__1"
    fake.cons.loc[late, "window"] += 1000  # beyond the recording, so no cell aligns
    fake.cons.loc[late, "t_start_s"] = fake.cons.loc[late, "window"] * 5.0
    for extra in ((), ("--allow-unmapped",)):
        with pytest.raises(SystemExit, match=r"no extracted cells: \['a1__1'\]"):
            fake.run(*extra, positional=False)
        assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path)), "nor the earlier run's"


def _edit(fake, what):
    if what == "code":
        fake.code["src/neurodent/results/autoreject_detector.py"] = "edited"
    elif what == "src_tree":
        fake.state["src_tree"] = "f" * 40
    else:
        (fake.tmp / "labels_A.csv").write_text("edited\n")


@pytest.mark.unit
@pytest.mark.parametrize("what, named", [("code", "code src/neurodent/results/autoreject_detector.py"),
                                         ("src_tree", "src_tree"), ("labels", "inputs labels_A")])
def test_files_changed_during_the_run_are_refused(fake, monkeypatch, what, named):
    """A change while animals are computed stops the run before the shard of the animal it happened during."""
    load = EDF.load_and_measure

    def load_then_edit(*args, **kwargs):
        out = load(*args, **kwargs)
        if fake.loaded == ["a1", "a2"]:
            _edit(fake, what)
        return out

    monkeypatch.setattr(EDF, "load_and_measure", load_then_edit)
    with pytest.raises(SystemExit, match=f"{named} changed during the run"):
        fake.run()
    assert fake.loaded == ["a1", "a2"]
    a1, a2 = (EDF.shard_paths(fake.parts, DS, a)["json"] for a in ("a1", "a2"))
    assert a1.exists() and not a2.exists(), "no shard is keyed under a state its animal did not run under"
    assert not fake.out.exists() and not fake.prov_path.exists()


@pytest.mark.unit
@pytest.mark.parametrize("what, named", [("code", "code src/neurodent/results/autoreject_detector.py"),
                                         ("src_tree", "src_tree")])
def test_files_changed_after_the_last_animal_are_refused(fake, monkeypatch, what, named):
    """The provenance describes the state every shard was keyed under, so a change before it is written refuses."""
    assert fake.run() == 0
    read = EDF.read_shard

    def read_then_edit(paths):
        _edit(fake, what)
        return read(paths)

    monkeypatch.setattr(EDF, "read_shard", read_then_edit)
    with pytest.raises(SystemExit, match=f"{named} changed during the run"):
        fake.run("--resume")
    assert not any(p.exists() for p in (fake.out, fake.scores, fake.prov_path))


@pytest.mark.unit
def test_cell_count_is_printed_for_dev_only(fake, capsys):
    assert fake.run() == 0
    assert f"wrote {len(fake.cons)} cells over 3 animals, 5 recordings, {len(GRID)} autoreject" in \
        capsys.readouterr().out
    fake.write_split("test")
    fake.out = fake.tmp / "out" / "test_features.parquet"
    assert fake.run(which="test") == 0
    out = capsys.readouterr().out
    assert f"wrote 3 animals, 5 recordings, {len(GRID)} autoreject configurations" in out
    assert "cells" not in out


@pytest.mark.unit
def test_run_refuses_another_working_directory(fake, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run from the repository root"):
        fake.run()


@pytest.mark.unit
def test_outputs_go_through_selection_event_cells_and_test_scoring(fake, monkeypatch):
    """The dev and test outputs of main, read by select_detectors, report_event_cells and score_test as written.

    The split lists the three synthetic animals in both partitions, and the test run's consensus flips every
    label. a1's list also names RVis by a raw name, which the selection finds through the extraction's abbrev.
    The cells come from one dataset, so the leave-one-dataset-out runs are recorded as not run.
    """
    import report_event_cells as EV
    import score_test as ST
    import select_detectors as SD

    monkeypatch.setattr(EDF, "resolve_channel", lambda name: "RVis" if name == "raw R Vis" else _resolve(name))
    fake.bad_channels["a1"] = {"_all": ["LMot", "RAud", "Bogus", "raw R Vis"]}
    fake.state.update(src_tree="tree0", src_dirty=False)  # a committed, clean src, as the selection requires
    fake.split.write_text(json.dumps({"dev": list(ANIMALS), "test": list(ANIMALS)}))
    out = fake.tmp / "out"
    dev, selection = out / "dev_features.parquet", out / "dev_selection.json"
    fake.out = dev
    assert fake.run() == 0
    dev_prov = json.loads(Path(f"{dev}.provenance.json").read_text())
    assert "scripts/labeling/select_detectors.py" not in dev_prov["code"]
    assert [k for k in ST.MATCHED_PARAMS if k not in dev_prov["params"]] == [], "every parameter score_test compares"
    assert set(ST.matched_code()) == set(dev_prov["code"]), "the code score_test compares is the code recorded"

    SD.main(["--features", str(dev), "--split", str(fake.split), "--out", str(selection), "--boot", "30"])
    picked = json.loads(selection.read_text())
    k = picked["pick"]["autoreject_config"]
    assert k in GRID and picked["features"]["sha256"] == _sha(dev)
    canonical = picked["settings"]["canonical_channels"][DS]
    assert set(canonical) == CANONICAL and len(canonical) == 10
    masks = picked["member_masks"]
    assert masks["absent_channels"]["a1__0"] == [c for c in canonical if c not in CH]
    assert masks["M"]["listed_absent"] == {"a1__0": ["RAud", "RVis"], "a1__1": ["RAud", "RVis"]}, \
        "RVis by the extraction's abbrev of its raw name"
    assert [u["channel"] for u in masks["M"]["unresolved"]] == ["Bogus", "Bogus"]
    assert masks["M"]["n_defined"] == int(pd.read_parquet(dev)["config_list_defined"].sum())
    assert "not_run" not in picked["nested_replace_logo"]
    assert picked["nested_replace_lodo"]["not_run"] == "leaving one dataset out needs at least 2, and the cells hold 1"

    events = out / "dev_event_cells.csv"
    EV.main(["--features", str(dev), "--selection", str(selection), "--split", str(fake.split), "--out", str(events)])
    assert list(pd.unique(pd.read_csv(events)["detector"]))[:len(SD.COMBINATIONS)] == list(SD.COMBINATIONS)

    fake.cons["y_true"] = 1 - fake.cons["y_true"]
    keymap = fake.keymap.read_text()

    def score(name, features):
        return ["--features", str(features), "--pick", str(selection), "--split", str(fake.split),
                "--out", str(out / name), "--boot", "30"]

    fake.keymap.write_text("edited\n")  # an input the two extractions must share
    fake.out = out / "test_other.parquet"
    assert fake.run(which="test") == 0
    with pytest.raises(SystemExit, match="produced differently") as err:
        ST.main(score("refused.json", fake.out))
    assert "input keymap" in str(err.value) and not (out / "refused.json").exists()

    fake.keymap.write_text(keymap)
    fake.out = out / "test_features.parquet"
    assert fake.run(which="test") == 0
    ST.main(score("test_score.json", fake.out))
    result = json.loads((out / "test_score.json").read_text())
    assert result["pick"]["autoreject_config"] == k and list(result["combinations"]) == list(SD.COMBINATIONS)
    assert result["features"]["sha256"] == _sha(fake.out) and result["settings"] == picked["settings"]
    assert {rel: result["provenance"]["code"][rel] for rel in SD.SCORING_CODE} == SD.scoring_code_hashes()
    assert set(pd.DataFrame(result["event_cells"])["detector"]) == set(SD.COMBINATIONS)
    with pytest.raises(SystemExit, match="scored once"):
        ST.main(score("again.json", fake.out))


# --------------------------------------------------------------------------- end to end on mini_real


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_main_end_to_end_mini_real(tmp_path, monkeypatch):
    """Bundle mini_real, fake three raters, run main twice (the second with --resume) and read the outputs back."""
    pytest.importorskip("spikeinterface")
    monkeypatch.chdir(REPO)
    _freeze_code_state(monkeypatch)
    real_set = C.set_temp_directory
    monkeypatch.setattr(C, "set_temp_directory", lambda p: real_set(tmp_path / "tmp_prepare"))
    real_prepare = C._prepare

    def prepare(ds):
        config, samples_config = real_prepare(ds)
        samples_config["bad_channels"] = {"A10": {"_all": ["LMot", "Nope"]},
                                          "F22": {"F22 KO May-10-2025": ["RHip"], "F22 KO May-11-2025": ["LAud"]}}
        return config, samples_config

    monkeypatch.setattr(C, "_prepare", prepare)
    wars = {}
    measure = EDF.load_and_measure

    def keep_war(samples_config, config, animal_id, **kwargs):
        ao, wars[animal_id] = measure(samples_config, config, animal_id, **kwargs)
        return ao, wars[animal_id]

    monkeypatch.setattr(EDF, "load_and_measure", keep_war)

    assert C.build_cohort(["mini_real"], tmp_path / "bundle_root", n_per_animal=1, seed=0, blind_seed=1,
                          make_bundle=False) == 0
    keymap = next((tmp_path / "bundle_root").rglob("keymap.csv"))
    manifest = pd.read_csv(next((tmp_path / "bundle_root").rglob("manifest.csv")))
    rng = np.random.default_rng(0)
    raters = []
    for r in ("r1", "r2", "r3"):
        filled = manifest.copy()
        for c in (c for c in filled.columns if c.startswith("label_")):
            filled[c] = rng.choice(["bad", "clean", "event"], size=len(filled))
        path = tmp_path / f"fake_{r}.csv"
        filled.to_csv(path, index=False)
        raters.append(f"{r}={path}")
    animals = sorted(set(pd.read_csv(keymap)["animal"].astype(str)))
    assert animals == ["A10", "F22"]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": animals, "test": []}))
    out = tmp_path / "out" / "dev_features.parquet"
    argv = [*raters, "--keymap", str(keymap), "--split", str(split), "--which", "dev", "--positional-recordings",
            "--datasets", "mini_real", "--workers", "1", "--temp-dir", str(tmp_path / "tmp_extract"),
            "--out", str(out)]
    assert EDF.main(argv) == 0

    df = pd.read_parquet(out)
    assert set(df["animal"]) == {"A10", "F22"} and set(df["recording"]) == {"A10__0", "F22__0"}
    assert not df[GRID + ["rms", "logrmsz_abs", "beta_wmax", "lof"]].isna().any().any()
    assert set(wars) == {"A10", "F22"}
    for animal, war in wars.items():  # the loader's float32 rms against the WAR the pipeline reads back
        assert extract_linear_array(war.result["rms"]).dtype == np.float32
        _check_filters(df[df["animal"] == animal], _read_back(war, tmp_path / "wars" / animal))
    assert df["config_list_defined"].all()
    assert (df["config_list"] == np.where(df["animal"] == "A10", df["channel"] == "LMot",
                                          df["channel"] == "RHip")).all()

    scores_path = out.with_name("dev_features.autoreject_scores.parquet")
    scores = pd.read_parquet(scores_path)
    assert len(scores) == len(df) * len(GRID) and not scores[["score", "cutoff"]].isna().any().any()
    bits = df.melt(id_vars=["recording", "window", "channel"], value_vars=GRID, var_name="config", value_name="bit")
    merged = scores.merge(bits, on=["recording", "window", "channel", "config"], validate="one_to_one")
    assert ((merged["score"] > merged["cutoff"]).astype(int) == merged["bit"]).all()

    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["src_tree"] == _git("rev-parse", "HEAD:src").strip()
    assert prov["src_dirty"] is bool(_git("status", "--porcelain", "--", "src").strip())
    for rel in ("config/config.yaml", "config/datasets/mini_real.yaml"):
        assert prov["inputs"][rel]["sha256"] == _sha(REPO / rel)
    assert prov["outputs"]["autoreject_scores"]["sha256"] == _sha(scores_path)
    assert [(u["animal"], u["key"]) for u in prov["config_list_unmatched"]] == [("F22", "F22 KO May-11-2025")]
    assert [(u["animal"], u["channel"]) for u in prov["config_list_unknown_channels"]] == [("A10", "Nope")]
    assert prov["config_list_uncovered"] == []

    first = out.read_bytes(), scores_path.read_bytes()
    assert EDF.main([*argv, "--resume"]) == 0
    assert (out.read_bytes(), scores_path.read_bytes()) == first
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["resumed_animals"] == ["A10", "F22"]
