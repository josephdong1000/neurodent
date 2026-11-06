"""Tests for the detector checks that read no labels: ``rejection_rates.py`` and ``inject_artifacts.py``.

Unit tests cover the artifact injection and its one cell set per pool, the animal draw, the pipeline's quality
filter it applies and the record of the selection, fitting named grid configurations with the detector's
default split seed, chunked scoring and the rate aggregation on synthetic data. A synthetic recording checks
that the scan reads the fragments ``build_fit_epochs`` reads, dates them through the recording's timestamps,
and that the fit seed draws the pool and nothing else. Both scripts are checked to write nothing when there
is nothing to measure. Integration tests run ``rejection_rates`` end to end on a draw of one mini_real animal
and ``inject_artifacts`` on one mini_real animal and on a synthetic pool, with their provenance.
"""

import inspect
import json
import logging
import os
import shlex
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "labeling"))
import inject_artifacts as IA  # noqa: E402
import rejection_rates as RR  # noqa: E402

from neurodent.results import autoreject_detector as adr  # noqa: E402

FS = 250.0
TWO_CONFIGS = ["autoreject/self", "welch/v2/self@nf40_na100"]
CELLS = np.array([[0, 0], [3, 2], [7, 1], [7, 2]])
MINI_REAL_ABBREVS = {"LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"}


@pytest.fixture
def pool():
    """A small clean pool: 12 fragments, 3 channels, 2 s at 250 Hz."""
    return IA.synthetic_pool(12, 3, FS, 2.0, seed=3)


@pytest.fixture
def _at_repo_root(monkeypatch):
    """Run from the repository root, where dataset paths resolve, and restore TMPDIR afterwards.

    Preparing a dataset (``C._prepare``) and ``--temp-dir`` point TMPDIR at a directory for the whole process.
    """
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))


@pytest.fixture
def synthetic_recording(tmp_path):
    """A 152.5 s, 3-channel recording at 250 Hz from 2024-01-01 17:59:00, and its start.

    It holds 30 full 5 s fragments and a ragged 2.5 s one; 12 fragments start before 18:00.
    """
    mne_ = pytest.importorskip("mne")
    pytest.importorskip("spikeinterface")
    from neurodent.loading.long_recording_organizer import LongRecordingOrganizer

    fs, n_samples = 250, int(152.5 * 250)
    data = 50e-6 * np.random.default_rng(0).standard_normal((3, n_samples))
    info = mne_.create_info(["ch0", "ch1", "ch2"], fs, ch_types="eeg")
    src = tmp_path / "src_raw.fif"
    mne_.io.RawArray(data, info, verbose=False).save(src, overwrite=True, verbose=False)
    start = datetime(2024, 1, 1, 17, 59, 0)
    lro = LongRecordingOrganizer(
        src, mode="mne", extract_func=mne_.io.read_raw_fif, intermediate="bin",
        intermediate_dir=str(tmp_path / "im"), manual_datetimes=start, cache_policy="force_regenerate",
    )
    return lro, start


@pytest.fixture
def split_seeds(monkeypatch):
    """Record the split seed of every cutoff search, whether a caller passes it or leaves the default."""
    real = adr.fit_configs
    signature = inspect.signature(real)
    seen = []

    def recording(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.append(bound.arguments["seed"])
        return real(*args, **kwargs)

    monkeypatch.setattr(adr, "fit_configs", recording)
    return seen


def _record_calls(monkeypatch, module, name):
    """Wrap ``module.name`` and return the list that gets each call's ``(kwargs, result)``."""
    real = getattr(module, name)
    calls = []

    def recording(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(module, name, recording)
    return calls


def _tmpdir_at_load(monkeypatch):
    """Record TMPDIR each time an animal's recordings load, and return the list it goes into."""
    real = RR.C.load_cohort_animal
    seen = []

    def load(*args, **kwargs):
        seen.append(os.environ["TMPDIR"])
        return real(*args, **kwargs)

    monkeypatch.setattr(RR.C, "load_cohort_animal", load)
    return seen


class _NoRecordings:
    """An animal whose load found no recordings."""

    long_recordings = []
    animaldays = []


def _cohort_input_names(*datasets):
    """Return the provenance input names of the manifest and the config files the datasets load."""
    names = {"manifest", "base_config", *(f"dataset_config_{ds}" for ds in datasets)}
    if (REPO / "config/config.local.yaml").exists():
        names.add("local_config")
    return names


def _inject_known(pool, kind, amplitude=5.0):
    """Inject ``kind`` at CELLS with spans drawn up front, and check that the input is left alone."""
    ref = IA.reference_rms(pool)
    spans = IA.draw_spans(kind, len(CELLS), pool.shape[2], np.random.default_rng(5))
    before = pool.copy()
    Y = IA.inject(pool, FS, CELLS, kind, amplitude, np.random.default_rng(6), ref_rms=ref, spans=spans)
    assert np.array_equal(pool, before), "inject must return a copy and leave its input unchanged"
    return Y, spans, ref


def _sinusoid_fit(d, t, hz):
    """Least-squares fit of ``a sin + b cos`` at ``hz``; returns ``(fitted, peak amplitude)``."""
    A = np.stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t)], axis=1)
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return A @ coef, float(np.hypot(*coef))


# Injection


@pytest.mark.parametrize("kind", IA.KINDS)
def test_inject_changes_only_the_chosen_cells_within_their_spans(pool, kind):
    Y, spans, _ = _inject_known(pool, kind)
    assert Y.shape == pool.shape and Y.dtype == np.float64
    chosen = np.zeros(pool.shape[:2], dtype=bool)
    chosen[CELLS[:, 0], CELLS[:, 1]] = True
    assert np.array_equal(np.any(Y != pool, axis=2), chosen), "only the chosen cells change"
    for (f, c), (s, e) in zip(CELLS, spans):
        assert np.array_equal(Y[f, c, :s], pool[f, c, :s]) and np.array_equal(Y[f, c, e:], pool[f, c, e:])


def test_clip_holds_its_span_at_the_mean_plus_amplitude_times_rms(pool):
    Y, spans, ref = _inject_known(pool, "clip", 5.0)
    for (f, c), (s, e) in zip(CELLS, spans):
        assert np.ptp(Y[f, c, s:e]) == 0.0, "a clipped span is constant"
        assert Y[f, c, s] == pytest.approx(pool[f, c].mean() + 5.0 * ref[c], rel=1e-12)


def test_flatline_holds_its_span_at_the_fragment_mean_and_ignores_amplitude(pool):
    Y, spans, _ = _inject_known(pool, "flatline", 5.0)
    for (f, c), (s, e) in zip(CELLS, spans):
        assert np.ptp(Y[f, c, s:e]) == 0.0, "a flatlined span is constant"
        assert Y[f, c, s] == pytest.approx(pool[f, c].mean(), rel=1e-12)
    Y20, _, _ = _inject_known(pool, "flatline", 20.0)
    assert np.array_equal(Y, Y20)


def test_step_adds_a_constant_up_to_the_end_of_the_fragment(pool):
    Y, spans, ref = _inject_known(pool, "step", 5.0)
    T = pool.shape[2]
    for (f, c), (s, e) in zip(CELLS, spans):
        assert e == T and 0.1 * T <= T - s <= 0.5 * T
        np.testing.assert_allclose(Y[f, c, s:] - pool[f, c, s:], 5.0 * ref[c], rtol=1e-9)


@pytest.mark.parametrize("kind", ["broadband", "line60"])
def test_broadband_and_line60_add_a_span_with_the_requested_rms(pool, kind):
    Y, spans, ref = _inject_known(pool, kind, 5.0)
    for (f, c), (s, e) in zip(CELLS, spans):
        d = Y[f, c, s:e] - pool[f, c, s:e]
        assert np.sqrt(np.mean(d**2)) == pytest.approx(5.0 * ref[c], rel=1e-9)
        fitted, _ = _sinusoid_fit(d, np.arange(s, e) / FS, IA.LINE_HZ)
        residual = np.sqrt(np.mean((d - fitted) ** 2)) / np.sqrt(np.mean(d**2))
        if kind == "line60":
            assert residual < 1e-9, "line60 adds a pure 60 Hz sinusoid"
        else:
            assert residual > 0.9, "broadband noise is not a 60 Hz sinusoid"


def test_drift_adds_a_half_hertz_sinusoid_of_the_amplitude_over_the_whole_fragment(pool):
    Y, spans, ref = _inject_known(pool, "drift", 5.0)
    T = pool.shape[2]
    for (f, c), (s, e) in zip(CELLS, spans):
        assert (s, e) == (0, T)
        d = Y[f, c] - pool[f, c]
        fitted, peak = _sinusoid_fit(d, np.arange(T) / FS, IA.DRIFT_HZ)
        np.testing.assert_allclose(fitted, d, rtol=0, atol=1e-9 * np.abs(d).max())
        assert peak == pytest.approx(5.0 * ref[c], rel=1e-9)


def test_amplitudes_of_one_kind_differ_only_in_scale(pool):
    """The experiment reseeds every amplitude of a kind alike, so only the scale changes."""
    ref = IA.reference_rms(pool)
    for kind in ("step", "broadband", "line60", "drift"):
        d2 = IA.inject(pool, FS, CELLS, kind, 2.0, np.random.default_rng([0, 1, 3]), ref_rms=ref) - pool
        d10 = IA.inject(pool, FS, CELLS, kind, 10.0, np.random.default_rng([0, 1, 3]), ref_rms=ref) - pool
        np.testing.assert_allclose(d10, 5.0 * d2, rtol=1e-9, atol=1e-15, err_msg=kind)


def test_spans_cover_10_to_50_percent_and_are_seeded():
    T = 1000
    for kind in IA.KINDS:
        spans = IA.draw_spans(kind, 500, T, np.random.default_rng(1))
        assert np.array_equal(spans, IA.draw_spans(kind, 500, T, np.random.default_rng(1))), kind
        if kind == "drift":
            assert (spans == [0, T]).all()
            continue
        lengths = spans[:, 1] - spans[:, 0]
        assert lengths.min() >= 100 and lengths.max() <= 500, kind
        assert (spans[:, 0] >= 0).all() and (spans[:, 1] <= T).all(), kind
        if kind == "step":
            assert (spans[:, 1] == T).all()
        else:
            assert spans[:, 0].min() < 50 and spans[:, 1].max() > 950, f"{kind} spans sit anywhere in the fragment"


@pytest.mark.parametrize(
    "cells, kind, fs, match",
    [
        ([[0, 0], [0, 0]], "clip", FS, "distinct"),
        ([[12, 0]], "clip", FS, "index"),
        ([[0, 3]], "clip", FS, "index"),
        ([[0, 0]], "spike", FS, "unknown kind"),
        ([[0, 0]], "line60", 100.0, "sampling rate"),
    ],
)
def test_inject_refuses_bad_requests(pool, cells, kind, fs, match):
    with pytest.raises(ValueError, match=match):
        IA.inject(pool, fs, cells, kind, 5.0, np.random.default_rng(0))


def test_inject_refuses_spans_that_contradict_the_kind(pool):
    with pytest.raises(ValueError, match="end of the fragment"):
        IA.inject(pool, FS, [[0, 0]], "step", 5.0, np.random.default_rng(0), spans=[[10, 20]])
    with pytest.raises(ValueError, match="whole fragment"):
        IA.inject(pool, FS, [[0, 0]], "drift", 5.0, np.random.default_rng(0), spans=[[10, pool.shape[2]]])
    with pytest.raises(ValueError, match="start < stop"):
        IA.inject(pool, FS, [[0, 0]], "clip", 5.0, np.random.default_rng(0), spans=[[20, 20]])


def test_cells_are_a_seeded_fraction_of_each_channel():
    cells = IA.draw_cells(100, 4, 0.05, np.random.default_rng(0))
    assert np.array_equal(cells, IA.draw_cells(100, 4, 0.05, np.random.default_rng(0))), "the draw is seeded"
    assert not np.array_equal(cells, IA.draw_cells(100, 4, 0.05, np.random.default_rng(1))), "the seed matters"
    assert np.bincount(cells[:, 1], minlength=4).tolist() == [5] * 4, "5 percent of each channel's fragments"
    assert len({tuple(c) for c in cells}) == len(cells), "no cell repeats"
    assert cells[:, 0].min() >= 0 and cells[:, 0].max() < 100
    assert len({tuple(cells[cells[:, 1] == c, 0]) for c in range(4)}) > 1, "each channel takes its own draw"
    assert IA.draw_cells(30, 2, 0.001, np.random.default_rng(0)).shape == (2, 2), "at least one per channel"
    with pytest.raises(ValueError, match="none clean"):
        IA.draw_cells(10, 2, 0.96, np.random.default_rng(0))


def test_every_kind_uses_one_cell_set_whatever_kinds_are_requested(monkeypatch):
    """Kinds are measured on the same cells, and a run split by kind gives the same rows."""
    X = IA.synthetic_pool(40, 3, FS, 2.0, seed=4)
    real = IA.inject
    used = []

    def recording(Xp, fs, cells, kind, *args, **kwargs):
        used.append((kind, np.array(cells)))
        return real(Xp, fs, cells, kind, *args, **kwargs)

    monkeypatch.setattr(IA, "inject", recording)
    configs = ["autoreject/self"]
    whole = pd.DataFrame(IA.run_experiment(X, FS, configs, amplitudes=[2.0, 10.0]))
    cells = used[0][1]
    assert [k for k, _ in used] == [k for k in IA.KINDS for _ in ((0,) if k == "flatline" else (0, 1))]
    assert all(np.array_equal(c, cells) for _, c in used), "every kind and amplitude uses one cell set"
    for kinds in (["line60"], ["step", "clip"], ["drift", "flatline", "broadband"]):
        used.clear()
        part = pd.DataFrame(IA.run_experiment(X, FS, configs, kinds=kinds, amplitudes=[2.0, 10.0]))
        assert used and all(np.array_equal(c, cells) for _, c in used), kinds
        expected = whole[whole["kind"].isin(kinds)].reset_index(drop=True)
        pd.testing.assert_frame_equal(part, expected, check_exact=True, obj=f"rows of {kinds} run alone")


def test_injection_rows_count_flags_on_injected_and_other_cells():
    injected = np.zeros((4, 2), dtype=bool)
    injected[[0, 1], 0] = True
    clean = np.zeros((4, 2), dtype=bool)
    clean[0, 0] = clean[3, 1] = True
    after = np.zeros((4, 2), dtype=bool)
    after[[0, 1], 0] = after[2, 1] = after[3, 1] = after[3, 0] = True
    [row] = IA.injection_rows({"c": clean}, {"c": after}, injected, {"kind": "clip", "amplitude": 5.0})
    assert row == {"kind": "clip", "amplitude": 5.0, "config": "c", "n_injected": 2, "n_other": 6,
                   "rate_injected": 1.0, "rate_injected_clean": 0.5, "rate_other": 0.5,
                   "rate_other_clean": pytest.approx(1 / 6)}


# Animals


def test_draw_is_reproducible_and_leaves_out_labeled_animals():
    cohort = [f"m{i}" for i in range(20)]
    labeled = {"m3", "m7", "m11"}
    drawn = RR.draw_animals(cohort, labeled, 5, seed=0)
    assert drawn == RR.draw_animals(cohort, labeled, 5, seed=0)
    assert drawn == RR.draw_animals(list(reversed(cohort)), labeled, 5, seed=0), "config order does not matter"
    assert drawn == sorted(drawn) and len(set(drawn)) == 5 and not set(drawn) & labeled
    assert any(RR.draw_animals(cohort, labeled, 5, seed=s) != drawn for s in range(1, 5)), "the seed matters"
    assert RR.draw_animals(cohort, labeled, 50, seed=0) == sorted(set(cohort) - labeled)
    assert RR.draw_animals([1, 2, 3, 32], {"32"}, 10, seed=0) == [1, 2, 3], "ids compare as strings"


def test_labeled_set_is_the_manifest_and_covers_the_split():
    labeled = RR.labeled_animals()
    manifest = json.loads((REPO / "config/labeling/cohort_manifest.json").read_text())
    split = json.loads((REPO / "config/labeling/split.json").read_text())
    assert labeled == {str(a) for a in manifest["rendered_animals"]}
    assert set(split["dev"]) | set(split["test"]) <= labeled, "a draw must leave out every evaluation animal"


def _quality_filter_config(exclude_unknown=True, exclude_bad=True):
    """A merged config holding only the pipeline's quality filter settings."""
    return {"samples": {"quality_filter": {"exclude_unknown_genotypes": exclude_unknown,
                                           "exclude_bad_animaldays": exclude_bad}}}


def _toy_dataset(monkeypatch, genotypes, bad=()):
    """Stand a toy dataset in for ``C._prepare``: the given animals and genotypes, both filter settings on."""
    samples_config = {"animals": [{"id": a} for a in genotypes], "bad_folder_animalday": list(bad)}
    monkeypatch.setattr(RR.C, "_prepare", lambda dataset: (_quality_filter_config(), samples_config))
    monkeypatch.setattr(RR.constants, "ANIMAL_METADATA", {a: {"genotype": g} for a, g in genotypes.items()})


def test_quality_filter_reasons_follow_the_pipeline_rules():
    """war_quality_filter.smk compares the genotype with "Unknown" exactly and the id with bad_folder_animalday."""
    genotypes = {"a": "WT", "b": "Unknown", "c": "KO", "d": "UNKNOWN", "e": None}
    listed = {"bad_folder_animalday": ["c"]}

    def reasons(samples_config=listed, **settings):
        return RR.quality_filter_reasons(list(genotypes), genotypes, _quality_filter_config(**settings), samples_config)

    both = reasons()
    assert list(both) == ["b", "c"]
    assert "genotype is Unknown" in both["b"] and "bad_folder_animalday" in both["c"]
    assert list(reasons(exclude_bad=False)) == ["b"] and list(reasons(exclude_unknown=False)) == ["c"]
    assert reasons(exclude_unknown=False, exclude_bad=False) == {}
    assert reasons({}) == {"b": both["b"]}, "a config without bad_folder_animalday lists no one"


def test_draw_takes_only_the_animals_the_quality_filter_keeps(monkeypatch, caplog):
    genotypes = {f"m{i:02d}": ("Unknown" if i % 3 == 0 else "WT") for i in range(12)}
    _toy_dataset(monkeypatch, genotypes, bad=["m01"])
    dropped = RR.prepare_cohort("toy")[2]
    assert dropped.keys() == {"m00", "m03", "m06", "m09", "m01"}
    drop_records = [{"animal": a, "reason": reason} for a, reason in dropped.items()]

    [(ds, everyone)], _ = RR.select_animals(["toy"], draw=100, seed=0, labeled={"m02"})
    assert ds == "toy" and everyone == [(a, "WT") for a in sorted(set(genotypes) - set(dropped) - {"m02"})]
    [(_, drawn)], record = RR.select_animals(["toy", "toy"], draw=3, seed=0, labeled={"m02"})
    kept = [a for a in genotypes if a not in dropped]
    assert [a for a, _ in drawn] == RR.draw_animals(kept, {"m02"}, 3, seed=0), "the filter acts before the draw"
    assert record == {"toy": {"active": list(genotypes), "quality_filter_dropped": drop_records,
                              "labeled_left_out": ["m02"], "candidates": sorted(set(kept) - {"m02"}),
                              "drawn": [a for a, _ in drawn]}}, "a dataset named twice is drawn once"

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rejection_rates"):
        picked, record = RR.select_animals(["toy"], animals=["m03", "m04"], labeled={"m04"}, allow_labeled=True)
    assert picked == [("toy", [("m03", "Unknown"), ("m04", "WT")])], "explicit ids are measured as given"
    assert record == {"toy": {"active": list(genotypes), "quality_filter_dropped": drop_records,
                              "requested": ["m03", "m04"], "labeled": ["m04"]}}
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and "'m03'" in warned[0] and "genotype is Unknown" in warned[0]


@pytest.mark.mutates_constants
def test_preregistered_draw_on_the_shipped_configs_skips_quality_filter_drops(_at_repo_root, monkeypatch):
    """Two animals per labeled dataset and two iqsec2 animals with seed 0, the draw the preregistration fixes."""
    monkeypatch.setattr(RR.C, "set_temp_directory", lambda path: None)  # cluster paths; nothing is loaded here
    datasets = [*RR.C.REAL_STRAINS, "iqsec2"]
    labeled = RR.labeled_animals()
    plan, record = RR.select_animals(datasets, draw=2, seed=0, labeled=labeled)
    assert [ds for ds, _ in plan] == datasets and list(record) == datasets
    for ds, drawn in plan:
        cohort, genotypes, dropped = RR.prepare_cohort(ds)
        assert len(drawn) == 2, ds
        for a, g in drawn:
            assert a in cohort and a not in dropped and g == genotypes[a], (ds, a)
        assert record[ds]["drawn"] == [a for a, _ in drawn], ds
        assert [d["animal"] for d in record[ds]["quality_filter_dropped"]] == list(dropped), ds
        assert record[ds]["candidates"] == RR.draw_candidates([a for a in cohort if a not in dropped], labeled), ds


def test_selection_is_printed_with_genotypes_and_shell_ready_ids():
    plan = [("d1", [("m(1)_group7", "Het"), ("cohort 9 mouse #3 Cage 4A", "WT")]), ("d2", [("400", "KO")]),
            ("d3", [])]
    lines = RR.plan_lines(plan, draw=2, seed=0)
    assert lines[0].startswith("draw of 2 per dataset, seed 0")
    assert lines[1:] == ["  d1: 'm(1)_group7' (genotype Het)", "  d1: 'cohort 9 mouse #3 Cage 4A' (genotype WT)",
                         "  d2: 400 (genotype KO)", "  d3: none"]
    printed = [shlex.split(line.split(": ", 1)[1].rsplit(" (genotype ", 1)[0]) for line in lines[1:4]]
    assert printed == [["m(1)_group7"], ["cohort 9 mouse #3 Cage 4A"], ["400"]], "ids paste into --animal"
    assert RR.plan_lines(plan[1:2]) == ["animals:", "  d2: 400 (genotype KO)"]


@pytest.mark.mutates_constants
def test_animals_are_selected_from_the_config(_at_repo_root):
    assert RR.prepare_cohort("mini_real") == (["A10", "F22"], {"A10": "WT", "F22": "KO"}, {})
    plan, _ = RR.select_animals(["mini_real"], draw=5, seed=0, labeled={"F22"})
    assert plan == [("mini_real", [("A10", "WT")])]
    picked, _ = RR.select_animals(["mini_real"], animals=["F22", "A10", "F22"])
    assert picked == [("mini_real", [("F22", "KO"), ("A10", "WT")])]
    with pytest.raises(ValueError, match="labeled"):
        RR.select_animals(["mini_real"], animals=["A10"], labeled={"A10"})
    allowed, _ = RR.select_animals(["mini_real"], animals=["A10"], labeled={"A10"}, allow_labeled=True)
    assert allowed == [("mini_real", [("A10", "WT")])]
    with pytest.raises(ValueError, match="no active animal"):
        RR.select_animals(["mini_real"], animals=["Z99"])
    with pytest.raises(ValueError, match="exactly one"):
        RR.select_animals(["mini_real", "mini_real"], animals=["A10"])
    with pytest.raises(ValueError, match="positive number"):
        RR.select_animals(["mini_real"])


@pytest.mark.mutates_constants
def test_main_refuses_a_labeled_animal_before_loading(_at_repo_root, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rendered_animals": ["A10"]}))
    out = tmp_path / "rates.csv"
    with pytest.raises(SystemExit) as exc:
        RR.main(["--dataset", "mini_real", "--animal", "A10", "--manifest", str(manifest), "--out", str(out)])
    assert exc.value.code == 2
    assert "labeled" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [manifest], "nothing is written"


@pytest.mark.mutates_constants
def test_main_refuses_an_empty_draw_before_writing(_at_repo_root, tmp_path, capsys):
    """Both mini_real animals are labeled, so the draw is empty and an earlier output stays as it was."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rendered_animals": ["A10", "F22"]}))
    out = tmp_path / "rates.csv"
    out.write_text("earlier results\n")
    with pytest.raises(SystemExit) as exc:
        RR.main(["--dataset", "mini_real", "--draw", "2", "--manifest", str(manifest), "--out", str(out),
                 "--evidence-out", str(tmp_path / "scores.parquet")])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "  mini_real: none" in captured.out.splitlines() and "draw is empty" in captured.err
    assert out.read_text() == "earlier results\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "rates.csv"]


@pytest.mark.mutates_constants
def test_main_writes_nothing_when_no_recording_runs(_at_repo_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(RR.C, "load_cohort_animal", lambda *args, **kwargs: _NoRecordings())
    out = tmp_path / "rates.csv"
    out.write_text("earlier results\n")
    with pytest.raises(SystemExit) as exc:
        RR.main(["--dataset", "mini_real", "--animal", "A10", "--configs", *TWO_CONFIGS, "--out", str(out),
                 "--evidence-out", str(tmp_path / "scores.parquet")])
    assert "has a recording to measure" in str(exc.value.code)
    assert "wrote" not in capsys.readouterr().out
    assert out.read_text() == "earlier results\n" and [p.name for p in tmp_path.iterdir()] == ["rates.csv"]


@pytest.mark.mutates_constants
def test_inject_writes_nothing_when_no_pool_runs(_at_repo_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(IA.C, "load_cohort_animal", lambda *args, **kwargs: _NoRecordings())
    out = tmp_path / "inject.csv"
    out.write_text("earlier results\n")
    with pytest.raises(SystemExit) as exc:
        IA.main(["--dataset", "mini_real", "--animal", "F22", "--configs", *TWO_CONFIGS, "--out", str(out)])
    assert "no pool was measured" in str(exc.value.code)
    assert "wrote" not in capsys.readouterr().out
    assert out.read_text() == "earlier results\n" and [p.name for p in tmp_path.iterdir()] == ["inject.csv"]


# Configurations and scoring


def test_config_names_parse_back_to_the_grid():
    for name in adr.grid_config_names():
        base, spectral = RR.parse_config_name(name)
        assert base in adr.CONFIG_NAMES, name
        if spectral is None:
            assert name == base
        else:
            rebuilt = f"{spectral['representation']}/{base}@nf{spectral['n_frequencies']}_na{spectral['n_averages']}"
            assert rebuilt == name
    assert RR.resolve_config_names(None) == adr.grid_config_names()
    assert RR.resolve_config_names(TWO_CONFIGS + TWO_CONFIGS[:1]) == TWO_CONFIGS
    with pytest.raises(ValueError, match="unknown"):
        RR.resolve_config_names(["v2/self"])  # a base name, not a grid name


def test_fitting_named_configurations_reproduces_the_grid():
    """Named fits equal fit_grid's with split seed 0, the detector default extract_dev_features.py fits with."""
    X = IA.synthetic_pool(40, 3, FS, 5.0, seed=1)
    grid = adr.fit_grid(X, FS, seed=0, parallel=False)
    names = ["autoreject/self+CMR", "welch/v3/self@nf20_na50", "welch/v2/self+CMR@nf80_na200", "autoreject/self"]
    subset = RR.fit_requested(X, FS, names)
    assert list(subset) == names
    for fits in (subset, RR.fit_requested(X, FS, None)):
        for name, fit in fits.items():
            ref = grid[name]
            assert fit.config == name and fit.kind == ref.kind and fit.spectral == ref.spectral, name
            assert np.array_equal(fit.thresholds, ref.thresholds) and np.array_equal(fit.scores, ref.scores), name
            assert (fit.baselines is None) == (ref.baselines is None), name
            if fit.baselines is not None:
                assert np.array_equal(fit.baselines, ref.baselines) and np.array_equal(fit.freqs, ref.freqs), name
    moved = adr.fit_grid(X, FS, seed=1, parallel=False)
    assert any(not np.array_equal(moved[n].thresholds, subset[n].thresholds) for n in names), \
        "another split seed moves the cutoffs, so the comparison above tells the seeds apart"


def test_experiment_fits_reproduce_fit_grid_with_split_seed_0(monkeypatch):
    """Every fit of a default injection run, clean and injected, equals fit_grid on its pool with seed 0."""
    X = IA.synthetic_pool(40, 3, FS, 5.0, seed=1)
    real = RR.fit_requested
    fitted = []

    def recording(Xp, fs, names, **kwargs):
        fits = real(Xp, fs, names, **kwargs)
        fitted.append((np.array(Xp, dtype=float), fits))
        return fits

    monkeypatch.setattr(RR, "fit_requested", recording)
    IA.run_experiment(X, FS, TWO_CONFIGS, kinds=["clip", "drift"], amplitudes=[5.0])
    assert len(fitted) == 3 and np.array_equal(fitted[0][0], X), "the clean pool, then one copy per kind"
    for Xp, fits in fitted:
        grid = adr.fit_grid(Xp, FS, seed=0, parallel=False)
        for name, fit in fits.items():
            assert np.array_equal(fit.thresholds, grid[name].thresholds), name
            assert np.array_equal(fit.scores, grid[name].scores), name


@pytest.mark.parametrize("size", [1, 7, 64])
def test_chunked_scoring_equals_unchunked_apply_fit(size):
    configs = TWO_CONFIGS + ["autoreject/self+CMR", "welch/v3/self+CMR@nf20_na50"]
    fits = RR.fit_requested(IA.synthetic_pool(30, 3, FS, 5.0, seed=1), FS, configs)
    X = IA.synthetic_pool(23, 3, FS, 5.0, seed=2)
    X[4, 1, 100:140] += 2e-3  # a burst, so some fragments are flagged
    chunks = ((np.arange(i, min(i + size, 23)), X[i:i + size]) for i in range(0, 23, size))
    idx, scores = RR.score_chunks(chunks, fits)
    assert np.array_equal(idx, np.arange(23))
    flags = RR.RecordingScan(["a", "b", "c"], fits, 30, idx, scores, []).flags()
    assert any(f.any() for f in flags.values())
    for name, fit in fits.items():
        assert np.array_equal(scores[name], adr.score_fragments(X, fit, parallel=False)), name
        assert np.array_equal(flags[name], adr.apply_fit(X, fit, parallel=False)), name
    with pytest.raises(ValueError, match="no full-length fragments"):
        RR.score_chunks([], fits)


def test_pool_check_passes_on_the_pool_and_catches_other_samples():
    X = IA.synthetic_pool(30, 3, FS, 5.0, seed=1)
    pool_idx = np.arange(0, 30, 2)
    fits = RR.fit_requested(X[pool_idx], FS, TWO_CONFIGS)
    idx, scores = RR.score_chunks([(np.arange(30), X)], fits)
    RR.check_pool_scores(idx, scores, fits, pool_idx)
    Y = X.copy()
    Y[4] *= 1.5  # fragment 4 is in the pool
    idx, scores = RR.score_chunks([(np.arange(30), Y)], fits)
    with pytest.raises(RuntimeError, match="differ"):
        RR.check_pool_scores(idx, scores, fits, pool_idx)
    with pytest.raises(RuntimeError, match="missing"):
        RR.check_pool_scores(idx[:20], {n: s[:20] for n, s in scores.items()}, fits, pool_idx)


# Aggregation and outputs


def test_rates_split_by_light_phase_and_calendar_date():
    # 12 fragments from 17:59:40 (4 before 18:00), then 6 from 23:59:50 (2 before midnight).
    times = [datetime(2025, 5, 10, 17, 59, 40) + timedelta(seconds=5 * i) for i in range(12)]
    times += [datetime(2025, 5, 10, 23, 59, 50) + timedelta(seconds=5 * i) for i in range(6)]
    flags = np.zeros((18, 2), dtype=bool)
    flags[[0, 1, 5, 13, 17], 0] = True
    flags[:, 1] = True
    keys = {"dataset": "d", "animal": "a", "genotype": "WT", "animalday": "a WT May-10-2025"}
    df = pd.DataFrame(RR.rate_rows({"cfg": flags}, ["LMot", "RMot"], times, keys), columns=RR.RATE_COLUMNS)
    assert (df[["dataset", "animal", "genotype", "animalday"]] == pd.Series(keys)).all().all()

    light = df[df["by"] == "isday"].set_index(["isday", "channel"])
    assert df.loc[df["by"] == "isday", "date"].isna().all()
    assert light.loc[(True, "LMot"), ["n_fragments", "n_flagged"]].tolist() == [4, 2]
    assert light.loc[(False, "LMot"), ["n_fragments", "n_flagged"]].tolist() == [14, 3]
    assert light.loc[(False, "LMot"), "rate"] == pytest.approx(3 / 14)
    assert light.loc[(True, "RMot"), "rate"] == 1.0 and light.loc[(False, "RMot"), "rate"] == 1.0

    dated = df[df["by"] == "date"].set_index(["date", "channel"])
    assert df.loc[df["by"] == "date", "isday"].isna().all()
    assert dated.loc[("2025-05-10", "LMot"), ["n_fragments", "n_flagged"]].tolist() == [14, 4]
    assert dated.loc[("2025-05-11", "LMot"), ["n_fragments", "n_flagged"]].tolist() == [4, 1]

    noon = [datetime(2025, 5, 10, 12, 0, 0) + timedelta(seconds=5 * i) for i in range(18)]
    only_day = pd.DataFrame(RR.rate_rows({"cfg": flags}, ["LMot", "RMot"], noon, keys))
    assert only_day.loc[only_day["by"] == "isday", "isday"].tolist() == [True, True], "no rows for an empty group"
    with pytest.raises(ValueError, match="expected"):
        RR.rate_rows({"cfg": flags[:, :1]}, ["LMot", "RMot"], times, keys)


def test_evidence_writer_publishes_only_on_success(tmp_path):
    df = pd.DataFrame({"fragment": [1, 2], "score": [0.5, 1.5]})
    path = tmp_path / "scores.parquet"
    with pytest.raises(RuntimeError, match="stop"):
        with RR.EvidenceWriter(path) as failed:
            failed.write(df)
            raise RuntimeError("stop")
    assert list(tmp_path.iterdir()) == [] and not failed.written, "no partial file is left behind"
    with RR.EvidenceWriter(path) as empty:
        pass
    assert list(tmp_path.iterdir()) == [] and not empty.written, "without a table nothing is written"
    with RR.EvidenceWriter(path) as writer:
        writer.write(df)
        writer.write(df)
    assert writer.written and pd.read_parquet(path)["score"].tolist() == [0.5, 1.5, 0.5, 1.5]
    with RR.EvidenceWriter(None) as nowhere:
        nowhere.write(df)
    assert not nowhere.written
    assert RR.thresholds_path(tmp_path / "rates.csv") == tmp_path / "rates_thresholds.csv"
    assert RR.provenance_path(tmp_path / "rates.csv") == tmp_path / "rates.csv.provenance.json"


def test_scan_reads_the_fragments_build_fit_epochs_reads(synthetic_recording):
    """On a synthetic recording with a ragged end, the chunked scan reads the pool's fragments exactly."""
    lro, start = synthetic_recording
    pool = adr.build_fit_epochs(lro, None, fragment_len_s=5.0, max_fit_fragments=None)
    channels, fs_scan, chunks = RR.fragment_chunks(lro, fragment_len_s=5.0, chunk_fragments=7)
    parts = list(chunks)
    assert [len(i) for i, _ in parts] == [7, 7, 7, 7, 2]
    idx = np.concatenate([i for i, _ in parts])
    assert np.array_equal(idx, np.arange(30)), "the ragged final fragment is skipped"
    assert np.array_equal(np.concatenate([x for _, x in parts]), pool.X)
    assert channels == pool.channels and fs_scan == pool.fs

    times = RR.fragment_times(lro, idx, 5.0)
    assert times[0] == start and times[12] == datetime(2024, 1, 1, 18, 0, 0)

    scan = RR.scan_recording(lro, TWO_CONFIGS, fragment_len_s=5.0, max_fit_fragments=12, chunk_fragments=7)
    assert scan.n_fit_fragments == 12 and np.array_equal(scan.indices, np.arange(30))
    rates = pd.DataFrame(RR.rate_rows(scan.flags(), scan.channels, scan.timestamps, {}))
    light = rates[rates["by"] == "isday"].groupby("isday")["n_fragments"].first()
    assert light[True] == 12 and light[False] == 18

    evidence = RR.evidence_frame(scan, {"dataset": "d", "animal": "a", "animalday": "x"}, 5.0)
    assert len(evidence) == 30 * 3
    row = evidence[(evidence["fragment"] == 8) & (evidence["channel"] == scan.channels[1])]
    assert row["t_start_s"].item() == 40.0 and row["timestamp"].item() == start + timedelta(seconds=40)
    for name in TWO_CONFIGS:
        assert row[name].item() == scan.scores[name][8, 1]


def test_fit_seed_draws_the_pool_and_leaves_the_split_seed_at_the_default(synthetic_recording):
    """Each scan's cutoffs are fit_grid's with split seed 0 on the pool its fit seed draws."""
    lro, _ = synthetic_recording
    pools = {}
    for fit_seed in (0, 7):
        scan = RR.scan_recording(lro, TWO_CONFIGS, fragment_len_s=5.0, max_fit_fragments=12, fit_seed=fit_seed,
                                 chunk_fragments=30)
        pool = adr.build_fit_epochs(lro, None, fragment_len_s=5.0, max_fit_fragments=12, seed=fit_seed)
        grid = adr.fit_grid(pool.X, pool.fs, seed=0, parallel=False)
        assert scan.n_fit_fragments == pool.X.shape[0], fit_seed
        for name in TWO_CONFIGS:
            assert np.array_equal(scan.fits[name].thresholds, grid[name].thresholds), (fit_seed, name)
            assert np.array_equal(scan.fits[name].scores, grid[name].scores), (fit_seed, name)
        pools[fit_seed] = pool
    assert not np.array_equal(pools[0].grid_times, pools[7].grid_times), "the fit seed draws the pool"
    moved = adr.fit_grid(pools[7].X, pools[7].fs, seed=7, parallel=False)
    assert any(not np.array_equal(moved[n].thresholds, scan.fits[n].thresholds) for n in TWO_CONFIGS), \
        "a split seed of 7 moves the cutoffs, so the comparison above tells the seeds apart"


# End to end


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_rejection_rates_end_to_end_mini_real(_at_repo_root, tmp_path, monkeypatch, capsys):
    """Rates, cutoffs, per-fragment scores and provenance for a draw of one mini_real animal, two configurations.

    F22 is listed as labeled, so the draw takes A10: one 120.36 s recording at 1000 Hz starting 2025-05-10
    10:00:00, 24 full 5 s fragments, all in daytime on one date, and a ragged final fragment that is skipped.
    """
    pytest.importorskip("spikeinterface")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rendered_animals": ["F22"]}))
    pools = _record_calls(monkeypatch, adr, "build_fit_epochs")
    tmpdirs = _tmpdir_at_load(monkeypatch)
    out, evidence_path, temp_dir = tmp_path / "rates.csv", tmp_path / "scores.parquet", tmp_path / "intermediates"
    rc = RR.main(["--dataset", "mini_real", "--draw", "1", "--seed", "0", "--manifest", str(manifest),
                  "--configs", *TWO_CONFIGS, "--out", str(out), "--evidence-out", str(evidence_path),
                  "--chunk-fragments", "10", "--temp-dir", str(temp_dir)])
    assert rc == 0
    printed = capsys.readouterr().out.splitlines()
    assert "  mini_real: A10 (genotype WT)" in printed and f"wrote per-fragment scores to {evidence_path}" in printed
    assert tmpdirs == [str(temp_dir)], "--temp-dir is set after the dataset is prepared and before loading"

    rates = pd.read_csv(out, float_precision="round_trip")
    thresholds = pd.read_csv(RR.thresholds_path(out), float_precision="round_trip")
    evidence = pd.read_parquet(evidence_path)
    assert list(rates.columns) == RR.RATE_COLUMNS and list(thresholds.columns) == RR.THRESHOLD_COLUMNS
    keys = rates[["dataset", "animal", "genotype", "animalday"]].drop_duplicates().values.tolist()
    assert keys == [["mini_real", "A10", "WT", "A10 WT May-10-2025"]]
    assert set(rates["config"]) == set(TWO_CONFIGS) and set(rates["channel"]) == MINI_REAL_ABBREVS

    light = rates[rates["by"] == "isday"]
    assert len(light) == 2 * 10 and light["isday"].astype(str).eq("True").all()
    assert (light["n_fragments"] == 24).all()
    assert (light["rate"] == light["n_flagged"] / light["n_fragments"]).all()
    dated = rates[rates["by"] == "date"]
    assert len(dated) == 2 * 10 and (dated["date"] == "2025-05-10").all() and (dated["n_fragments"] == 24).all()

    assert len(thresholds) == 2 * 10 and (thresholds["n_fit_fragments"] == 24).all()
    assert len(evidence) == 24 * 10 and sorted(evidence["fragment"].unique()) == list(range(24))
    assert evidence["isday"].all()
    for config in TWO_CONFIGS:
        cut = thresholds[thresholds["config"] == config].set_index("channel")["threshold"]
        flagged = (evidence[config] > evidence["channel"].map(cut)).groupby(evidence["channel"]).sum()
        counted = light[light["config"] == config].set_index("channel")["n_flagged"]
        assert flagged.sort_index().tolist() == counted.sort_index().tolist(), config

    # The default run's cutoffs are fit_grid's with split seed 0 on the pool build_fit_epochs drew with seed 0.
    [(pool_kwargs, pool)] = pools
    assert pool_kwargs["seed"] == 0
    grid = adr.fit_grid(pool.X, pool.fs, seed=0, parallel=False)
    for config in TWO_CONFIGS:
        cut = thresholds[thresholds["config"] == config].set_index("channel")["threshold"]
        assert cut.loc[list(pool.channels)].tolist() == grid[config].thresholds.tolist(), config

    prov = json.loads(RR.provenance_path(out).read_text())
    assert set(prov["inputs"]) == _cohort_input_names("mini_real") | {"script_rejection_rates"}
    assert prov["inputs"]["manifest"]["path"] == str(manifest.resolve())
    assert prov["inputs"]["dataset_config_mini_real"]["path"] == str(REPO / "config/datasets/mini_real.yaml")
    params = prov["params"]
    assert params["script"] == "rejection_rates" and params["config_names"] == TWO_CONFIGS
    assert (params["draw"], params["seed"], params["fit_seed"]) == (1, 0, 0)
    assert params["out"] == str(out) and params["temp_dir"] == str(temp_dir)
    assert params["selection"] == {"mini_real": {"active": ["A10", "F22"], "quality_filter_dropped": [],
                                                 "labeled_left_out": ["F22"], "candidates": ["A10"],
                                                 "drawn": ["A10"]}}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_inject_artifacts_mini_real_end_to_end(_at_repo_root, tmp_path, monkeypatch, split_seeds):
    """The real-data path: F22's one recording (93 full fragments x 10 channels), subsampled with --fit-seed."""
    pytest.importorskip("spikeinterface")
    pools = _record_calls(monkeypatch, adr, "build_fit_epochs")
    tmpdirs = _tmpdir_at_load(monkeypatch)
    out, temp_dir = tmp_path / "inject.csv", tmp_path / "intermediates"
    rc = IA.main(["--dataset", "mini_real", "--animal", "F22", "--configs", *TWO_CONFIGS, "--kinds", "clip",
                  "flatline", "--amplitudes", "20", "--max-fit-fragments", "40", "--fit-seed", "3",
                  "--temp-dir", str(temp_dir), "--out", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    keys = df[["dataset", "animal", "genotype", "animalday"]].drop_duplicates().values.tolist()
    assert keys == [["mini_real", "F22", "KO", "F22 KO May-10-2025"]]
    assert sorted(df["kind"].unique()) == ["clip", "flatline"] and len(df) == 2 * len(TWO_CONFIGS)

    [(pool_kwargs, pool)] = pools
    assert (pool_kwargs["seed"], pool_kwargs["max_fit_fragments"]) == (3, 40), "--fit-seed draws the pool"
    assert split_seeds and set(split_seeds) == {0}, "and the cutoff search keeps the detector's split seed"
    n_pool = pool.X.shape[0]
    assert n_pool in (39, 40), "a subsample of 40, less a ragged final fragment if it was drawn"
    assert (df["n_injected"] == 2 * 10).all(), "5 percent of the pool on each of 10 channels"
    assert (df["n_injected"] + df["n_other"] == n_pool * 10).all()
    assert tmpdirs == [str(temp_dir)], "--temp-dir is set after the dataset is prepared and before loading"

    prov = json.loads(RR.provenance_path(out).read_text())
    assert set(prov["inputs"]) == _cohort_input_names("mini_real") | {"script_inject_artifacts",
                                                                      "script_rejection_rates"}
    params = prov["params"]
    assert params["script"] == "inject_artifacts" and params["config_names"] == TWO_CONFIGS
    assert (params["fit_seed"], params["max_fit_fragments"], params["temp_dir"]) == (3, 40, str(temp_dir))
    assert params["selection"] == {"mini_real": {"active": ["A10", "F22"], "quality_filter_dropped": [],
                                                 "requested": ["F22"], "labeled": []}}


@pytest.mark.integration
def test_inject_artifacts_synthetic_end_to_end(tmp_path):
    out = tmp_path / "inject.csv"
    common = ["--synthetic", "--configs", *TWO_CONFIGS, "--n-fragments", "60", "--n-channels", "3", "--amplitudes",
              "5", "20", "--seed", "1"]
    assert IA.main([*common, "--out", str(out)]) == 0
    df = pd.read_csv(out, float_precision="round_trip")
    assert list(df.columns) == IA.ROW_COLUMNS
    sizes = df.groupby(["config", "kind"]).size()
    assert len(df) == len(TWO_CONFIGS) * (5 * 2 + 1), "five kinds at two amplitudes, flatline once"
    assert (sizes.xs("flatline", level="kind") == 1).all()
    assert df.loc[df["kind"] == "flatline", "amplitude"].isna().all()
    assert (df["n_injected"] == 3 * 3).all(), "5 percent of 60 fragments on each of 3 channels"
    assert (df["n_injected"] + df["n_other"] == 60 * 3).all()
    rate_cols = ["rate_injected", "rate_injected_clean", "rate_other", "rate_other_clean"]
    assert df[rate_cols].ge(0).all().all() and df[rate_cols].le(1).all().all()
    # The clean fit and the cells are shared by every kind and amplitude, so the clean rates repeat across them.
    assert (df.groupby("config")[["rate_injected_clean", "rate_other_clean"]].nunique() == 1).all().all()

    prov = json.loads(RR.provenance_path(out).read_text())
    assert set(prov["inputs"]) == {"script_inject_artifacts", "script_rejection_rates"}, "no data files are read"
    assert prov["params"]["synthetic"] is True and prov["params"]["selection"] is None
    assert prov["params"]["amplitudes"] == [5.0, 20.0] and prov["params"]["seed"] == 1

    split = tmp_path / "inject_split.csv"
    assert IA.main([*common, "--kinds", "line60", "step", "--out", str(split)]) == 0
    part = pd.read_csv(split, float_precision="round_trip")
    whole = df[df["kind"].isin(["step", "line60"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(part, whole, check_exact=True, obj="a run split by kind")


@pytest.mark.parametrize("extra", [["--fit-seed", "3"], ["--temp-dir", "x"], ["--max-fit-fragments", "10"],
                                   ["--allow-labeled"], ["--manifest", "m.json"]])
def test_synthetic_refuses_real_data_options(tmp_path, extra):
    """Options that only shape a real-data pool are refused with --synthetic instead of being ignored."""
    with pytest.raises(SystemExit):
        IA.main(["--synthetic", *extra, "--out", str(tmp_path / "x.csv")])


@pytest.mark.parametrize("extra", [["--n-fragments", "50"], ["--n-channels", "2"], ["--fs", "500"]])
def test_dataset_refuses_synthetic_options(tmp_path, extra):
    """Options that only shape a synthetic pool are refused with --dataset."""
    with pytest.raises(SystemExit):
        IA.main(["--dataset", "mini_real", "--animal", "F22", *extra, "--out", str(tmp_path / "x.csv")])


def test_cell_draw_and_synthetic_pool_use_different_streams():
    """The cell draw does not reuse the stream that generated the synthetic pool's noise."""
    assert not np.array_equal(np.random.default_rng([0, 2]).random(8), np.random.default_rng(0).random(8))
    assert np.array_equal(np.random.default_rng([0, 0]).random(8), np.random.default_rng(0).random(8))
