"""F, L and M against the pipeline's own decisions on the mini_real recordings.

The feature columns come from a WAR built as ``extract_dev_features.py`` builds it: the cohort load seam,
``compute_bad_channels`` for the LOF scores, and ``score_detectors.FEATURES`` on 5 s windows through the dask
path, with psdfrac added so that high_beta reads it as it does on a pipeline WAR. Every (window, channel) cell
is aligned with the script's ``war_filter_scores``, ``align_cells`` and ``lof_by_abbrev``. The log RMS z score
is taken over the animal's whole WAR and the window's beta maximum ignores NaN, as the pipeline's filters do.

The pipeline's decisions come from its own code on the same WAR: saved, standardized with
``reorder_and_pad_channels`` as ``standardize_wars.py`` does, then filtered with
``apply_filters(fragment_filter_config, min_valid_channels=3)`` as ``filter_wars_fragments.py`` does, or through
the LOF branch of ``filter_wars_channels.py`` (``get_bad_channels_by_lof_threshold``, then
``reject_channels_by_session``). A cell is rejected where the filtered WAR holds NaN and the standardized one a
value. LOF scores one value per channel per recording, keyed by animalday, and the branch rejects per
animalday: the unit of the ``lof`` column. M is compared with the manual branch of ``filter_wars_channels.py``
itself, run on the standardized WAR and on the fragment-filtered one with a per-recording list for each animal,
which also gives today (F or M). The recording records are those the extraction writes. Both mini_real
recordings hold all ten of the dataset's canonical channels, so standardization pads nothing here;
tests/test_combinations.py covers a padded recording.
"""

import copy
import importlib.util
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import zscore

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "labeling"))
import build_cohort_bundle as C  # noqa: E402
import extract_dev_features as EDF  # noqa: E402
import render_context as R  # noqa: E402
import score_detectors as SDET  # noqa: E402
import select_detectors as SD  # noqa: E402
from neurodent import constants  # noqa: E402
from neurodent.analysis import AnimalAnalyzer  # noqa: E402
from neurodent.core.utils.paths import set_temp_directory  # noqa: E402
from neurodent.results import WindowAnalysisResult  # noqa: E402
from neurodent.results.feature_utils import extract_linear_array  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.mutates_constants]

ANIMALS = ("A10", "F22")
DATASET = "mini_real"
# The constants tests/conftest.py restores after each test; the module fixture restores them after the module.
GUARDED = ("GENOTYPE_MAP", "SEX_MAP", "CHANNEL_MAP", "CHANNEL_ABBREVS", "CHANNEL_ABBREV_BY_RAW", "DF_SORT_ORDER",
           "ANIMAL_METADATA")


def _settings(**overrides):
    """The config's F, L and M settings with mini_real's canonical channels, and any override."""
    return {**SD.load_settings(datasets=[DATASET]), **overrides}


def _cells(war, animal, z_unit="war"):
    """Every (window, channel) cell with the columns extract_dev_features computes, one recording per animalday.

    rms, the log RMS z score over the whole WAR (in float64, as the pipeline reads the WAR back from parquet)
    and the beta proportions come from the extraction's own ``war_filter_scores``. ``z_unit`` "recording" takes
    the z score within each animalday instead.
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
            cons = pd.DataFrame([{"recording": f"{animal}__{i}", "window": w, "channel": c, "t_start_s": R.FRAG_S * w,
                                  "y_true": 0} for w in range(len(sel)) for c in ch])
            cells = EDF.align_cells(rms_all[sel], z, np.nanmax(beta_all[sel], axis=1), EDF.lof_by_abbrev(war, day),
                                    ch, cons, np.arange(len(sel)) * R.FRAG_S)
            for cell in cells:
                cell.update(animalday=day, row=int(sel[cell["window"]]))
            rows += cells
    return pd.DataFrame(rows).assign(dataset=DATASET, config_list=0, config_list_defined=False)


def _records(cells, war, settings, lists=None):
    """The recording records the extraction writes: each recording's channels in the standardized order, and its
    list (``{recording: [channel]}``; a recording missing from it has no list)."""
    standardized = settings["canonical_channels"][DATASET]
    own = [c for c in standardized if c in war.channel_abbrevs]
    records = {"recording_channels": {}, "config_list_by_recording": {}}
    for rec in pd.unique(cells["recording"]):
        listed = (lists or {}).get(rec)
        records["recording_channels"][rec] = own
        records["config_list_by_recording"][rec] = {"defined": listed is not None,
                                                    "channels": [c for c in own if c in (listed or [])],
                                                    "absent": [c for c in (listed or []) if c not in own]}
    return records


def _recordings(cells, war, settings):
    return SD.recorded_channels(cells, _records(cells, war, settings), settings)


def _standardize(war, folder):
    """Save the WAR and standardize it as standardize_wars.py does; return the standardized folder."""
    war.save_parquet_and_json(folder / "war", filename="war", slugify_filename=False)
    lazy = WindowAnalysisResult.scan_parquet_and_json(folder / "war", filename="war")
    lazy.reorder_and_pad_channels(None, use_abbrevs=True)
    lazy.save_parquet_and_json(folder / "std", filename="war")
    return folder / "std"


def _pipeline(std, out, filter_config, min_valid=3):
    """Return the pipeline's rejections, WAR rows by channel, from apply_filters on the standardized WAR."""
    lazy = WindowAnalysisResult.scan_parquet_and_json(std, filename="war")
    lazy.apply_filters(filter_config=filter_config, min_valid_channels=min_valid)
    lazy.save_parquet_and_json(out, filename="war")
    before = WindowAnalysisResult.load_parquet_and_json(std, filename="war")
    after = WindowAnalysisResult.load_parquet_and_json(out, filename="war")
    b, a = (extract_linear_array(w.result["psdtotal"]) for w in (before, after))
    return pd.DataFrame(np.isnan(a) & ~np.isnan(b), columns=before.channel_names)


def _lof_branch(std, out, threshold, min_valid):
    """Return the LOF branch's rejections and bad channels, as filter_wars_channels.py builds them."""
    bad = WindowAnalysisResult.scan_parquet_and_json(std, filename="war").get_bad_channels_by_lof_threshold(threshold)
    config = {"reject_channels": {"bad_channels": []}, "reject_channels_by_session": {"bad_channels_dict": bad}}
    return _pipeline(std, out, config, min_valid), bad


def _at(cells, decisions):
    """Return the decisions at each cell's WAR row and channel."""
    return decisions.to_numpy()[cells["row"].to_numpy(), decisions.columns.get_indexer(cells["channel"])].astype(int)


def _run_script(name, src, dst, config, samples_config, animal, **params):
    """Run ``workflow/scripts/<name>.py`` on the WAR in ``src``, writing ``dst``, with a stand-in snakemake object."""
    spec = importlib.util.spec_from_file_location(f"pipeline_{name}", REPO / "workflow" / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded by __name__, so nothing runs
    module.setup_snakemake_logging = lambda _: logging.getLogger(f"pipeline_{name}")  # keeps pytest's capture
    module.snakemake = SimpleNamespace(
        input=SimpleNamespace(war_parquet=str(src / "war.parquet"), war_json=str(src / "war.json")),
        output=SimpleNamespace(war_parquet=str(dst / "war.parquet")),
        params=SimpleNamespace(config=config, samples_config=samples_config, animal_folder="folder",
                               animal_id=animal, **params),
        wildcards=SimpleNamespace(animal=animal))
    module.main()
    return dst


def _rejected(std, out):
    """Return the cells NaN in the WAR at ``out`` and not in the standardized one, WAR rows by channel."""
    before = WindowAnalysisResult.load_parquet_and_json(std, filename="war")
    after = WindowAnalysisResult.load_parquet_and_json(out, filename="war")
    b, a = (extract_linear_array(w.result["psdtotal"]) for w in (before, after))
    return pd.DataFrame(np.isnan(a) & ~np.isnan(b), columns=before.channel_names)


@pytest.fixture(scope="module")
def mini_real(tmp_path_factory):
    """Both mini_real animals: the WAR, its standardized folder and its cells."""
    pytest.importorskip("spikeinterface")
    saved = {k: copy.deepcopy(getattr(constants, k)) for k in GUARDED}
    tmp = tmp_path_factory.mktemp("mini_real")
    try:
        with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mp.chdir(REPO)
            mp.setenv("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))
            config, samples_config = C._prepare("mini_real")
            set_temp_directory(tmp / "intermediates")
            out = {}
            for animal in ANIMALS:
                ao = C.load_cohort_animal(samples_config, config, animal)
                az = AnimalAnalyzer(ao)
                az.compute_bad_channels(lof_threshold=1.0, lof_chunk_duration_s=60)
                war = az.compute_windowed_analysis([*SDET.FEATURES, "psdfrac"], window_s=R.FRAG_S,
                                                   apply_notch_filter=True, multiprocess_mode="dask",
                                                   chunk_duration_s=3600)
                out[animal] = {"war": war, "std": _standardize(war, tmp / animal), "cells": _cells(war, animal)}
            yield out
    finally:
        for k, v in saved.items():
            setattr(constants, k, v)


def test_f_matches_the_pipeline_at_the_config_settings(mini_real, tmp_path):
    settings = _settings()
    fragment = yaml.safe_load(SD.CONFIG_PATH.read_text())["analysis"]["fragment_filter_config"]
    n_rejected = 0
    for animal, d in mini_real.items():
        f, audit = SD.fragment_reject(d["cells"], settings, _recordings(d["cells"], d["war"], settings))
        assert f.tolist() == _at(d["cells"], _pipeline(d["std"], tmp_path / animal, fragment)).tolist(), animal
        assert audit["n_windows_with_channels_without_cells"] == 0
        n_rejected += int(f.sum())
        _, audits = SD.member_masks(d["cells"], settings, _records(d["cells"], d["war"], settings))
        assert audits["padded_channels"] == {} and audits["channels_without_cells"] == {}, "nothing padded"
    assert n_rejected > 0, "the comparison covers rejections"


def test_f_matches_the_pipeline_at_ties_and_through_the_window_rule(mini_real, tmp_path):
    """Thresholds set to values that cells hold, and high enough a share of rejections that windows run short.

    High_beta keeps its config setting: the pipeline reads psdfrac, band over the sum of bands, and the features
    psdband over psdtotal, which agree to rounding but not to the last bit a tie needs.
    """
    cells = mini_real["F22"]["cells"]
    rms = np.sort(cells["rms"].to_numpy())
    z = np.sort(cells["logrmsz_abs"].dropna().to_numpy())
    fragment = {"high_rms": {"max_rms": float(rms[len(rms) // 2])}, "low_rms": {"min_rms": float(rms[len(rms) // 20])},
                "logrms_range": {"z_range": float(z[-25])}, "high_beta": {"max_beta_prop": 0.4}}
    settings = _settings(fragment_filters=fragment)
    f, audit = SD.fragment_reject(cells, settings, _recordings(cells, mini_real["F22"]["war"], settings))
    assert f.tolist() == _at(cells, _pipeline(mini_real["F22"]["std"], tmp_path / "ties", fragment)).tolist()
    for col, value in (("rms", rms[len(rms) // 2]), ("rms", rms[len(rms) // 20]), ("logrmsz_abs", z[-25])):
        assert (cells[col] == value).any(), "a cell sits on the threshold"
    assert audit["n_window_rule_only"] > 0, "the window rule rejects cells the filters keep"


def test_l_matches_the_pipeline_lof_branch_at_the_config_threshold(mini_real, tmp_path):
    settings = _settings()
    lof = settings["lof"]
    flagged = {}
    for animal, d in mini_real.items():
        lo, _ = SD.lof_reject(d["cells"], settings, _recordings(d["cells"], d["war"], settings))
        pipeline, bad = _lof_branch(d["std"], tmp_path / animal, lof["reject_lof_threshold"], lof["min_valid_channels"])
        assert lo.tolist() == _at(d["cells"], pipeline).tolist(), animal
        flagged[animal] = sorted(d["cells"].loc[lo.astype(bool), "channel"].unique())
        assert len(flagged[animal]) == sum(len(v) for v in bad.values())
    assert flagged["F22"], "the comparison covers flagged channels"


def test_l_matches_the_pipeline_at_a_tie_and_through_the_recording_rule(mini_real, tmp_path):
    d = mini_real["F22"]
    [scores] = [v["lof_scores"] for v in d["war"].lof_scores_dict.values()]
    scores = sorted(scores)
    for name, threshold, n_flagged in (("tie", scores[-3], 3), ("rule", scores[1], 9)):
        settings = _settings(lof={"reject_lof_threshold": threshold, "min_valid_channels": 3})
        lo, audit = SD.lof_reject(d["cells"], settings, _recordings(d["cells"], d["war"], settings))
        pipeline, bad = _lof_branch(d["std"], tmp_path / name, threshold, 3)
        assert lo.tolist() == _at(d["cells"], pipeline).tolist(), name
        assert sum(len(v) for v in bad.values()) == n_flagged, name
    assert lo.all() and audit["n_recording_rule_only"] == len(d["cells"]) // 10, "one channel left: all of it goes"


def test_log_rms_z_runs_over_the_animal_war(mini_real, tmp_path):
    """A10 and F22 as two recordings of one animal: the pipeline z-scores log RMS over both.

    A z score within each animalday disagrees with the pipeline here; the z score over the whole WAR agrees.
    """
    first = WindowAnalysisResult.load_parquet_and_json(mini_real["A10"]["std"], filename="war")
    second = WindowAnalysisResult.load_parquet_and_json(mini_real["F22"]["std"], filename="war")
    [f22_day] = second.lof_scores_dict
    day = "A10 WT May-11-2025"
    moved = second.result.assign(animal="A10", genotype=first.genotype, animalday=day,
                                 timestamp=pd.to_datetime(second.result["timestamp"]) + pd.Timedelta(days=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        war = WindowAnalysisResult(pd.concat([first.result, moved], ignore_index=True), animal_id="A10",
                                   genotype=first.genotype, sex=first.sex, channel_names=list(first.channel_names),
                                   lof_scores_dict={**first.lof_scores_dict, day: second.lof_scores_dict[f22_day]})
    war.save_parquet_and_json(tmp_path / "two", filename="war", slugify_filename=False)
    fragment = {"logrms_range": yaml.safe_load(SD.CONFIG_PATH.read_text())["analysis"]["fragment_filter_config"]
                ["logrms_range"]}
    settings = _settings(fragment_filters=fragment)
    pipeline = _pipeline(tmp_path / "two", tmp_path / "two_filtered", fragment)
    whole, per_recording = _cells(war, "A10", "war"), _cells(war, "A10", "recording")
    assert SD.fragment_reject(whole, settings, _recordings(whole, war, settings))[0].tolist() == _at(whole, pipeline).tolist()
    assert (SD.fragment_reject(per_recording, settings, _recordings(per_recording, war, settings))[0].tolist()
            != _at(per_recording, pipeline).tolist())


def test_m_and_today_match_the_pipeline_manual_branch(mini_real, tmp_path):
    """filter_wars_channels.py's manual branch, run on the standardized WAR (M) and on the fragment-filtered one
    (today, F or M), with a per-recording list for each animal: two channels for F22 per session, and eight for
    A10 as a list-form entry, which leaves two valid channels and takes the recording whole."""
    config = yaml.safe_load(SD.CONFIG_PATH.read_text())
    settings = _settings()
    standardized = settings["canonical_channels"][DATASET]
    for animal, d in mini_real.items():
        days = list(pd.unique(d["war"].result["animalday"]))
        names = ["LMot", "RHip"] if animal == "F22" else standardized[:8]
        entry = {day: list(names) for day in days} if animal == "F22" else {"_all": list(names)}
        out = tmp_path / animal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frag = _run_script("filter_wars_fragments", d["std"], out / "frag", config, {}, animal)
            alone = _run_script("filter_wars_channels", d["std"], out / "manual_alone", config,
                                {"bad_channels": {animal: copy.deepcopy(entry)}}, animal, filter_type="manual")
            branch = _run_script("filter_wars_channels", frag, out / "manual", config,
                                 {"bad_channels": {animal: copy.deepcopy(entry)}}, animal, filter_type="manual")
        lists = {f"{animal}__{i}": list(names) for i in range(len(days))}
        records = _records(d["cells"], d["war"], settings, lists)
        cells = d["cells"].assign(config_list=d["cells"]["channel"].isin(names).astype(int), config_list_defined=True)
        members, audits = SD.member_masks(cells, settings, records)
        today = SD.combination_preds(members, np.zeros(len(cells), int))["today"]
        assert members["M"].tolist() == _at(cells, _rejected(d["std"], alone)).tolist(), animal
        assert today.tolist() == _at(cells, _rejected(d["std"], branch)).tolist(), animal
        assert members["F"].tolist() == _at(cells, _rejected(d["std"], frag)).tolist(), animal
        if animal == "A10":
            assert members["M"].all() and audits["M"]["n_recording_rule_only"] == int((~cells["channel"].isin(names)).sum())
        else:
            assert members["M"].sum() == int(cells["channel"].isin(names).sum()) and audits["M"]["n_recording_rule_only"] == 0
