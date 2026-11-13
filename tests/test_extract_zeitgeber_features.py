"""extract_zeitgeber_features loads each WAR once, one at a time, and its output does not change.

The script used to load the WARs in a pool of 15, which on IQSEC2 peaked at 270 of the 293 GiB
the rule had. It now loads them one at a time through iter_wars, in sorted animal order. These
tests pin that each WAR is loaded once, in that order, that no WAR is still alive when the next
one loads, and that the output does not depend on the order of the inputs.
"""

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from neurodent.results.zeitgeber import _extract_zeitgeber_frame

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "workflow" / "scripts" / "extract_zeitgeber_features.py"

# Folder, animal id, genotype, sex, and how each WAR departs from the default 3-channel day.
# Together they cover the ways datasets differ: channel count, recording days, a single
# window, features a WAR lacks, a channel dead throughout, genotypes outside the default plot
# order, and more genotypes than the fixed palette. Input order is not sorted, and the folders
# are not the ids, as on disk, where folder iqsec2_80 holds IQSEC2_80. A5 lacks features the
# zeitgeber config does not ask for; lacking one it asks for raises (see below).
ANIMALS = [
    ("a2", "A2", "KO", "Female", {}),
    ("a1", "A1", "WT", "Male", {"channels": [
        "LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis",
    ]}),
    ("a3", "A3", "x/y", "Female", {"channels": ["LMot", "RMot"], "days": 3}),
    ("a6", "A6", "HET", "Male", {"windows": 1}),
    ("a5", "A5", "WT", "Female", {"drop": ["nspike", "cohere", "psd"]}),
    ("a4", "A4", "KO", "Male", {"dead_channel": 0}),
]

SAMPLES_CONFIG = {
    "ANIMAL_METADATA": [
        {"id": "A1", "sex": "Male", "genotype": "WT"},
        {"id": "A2", "sex": "Female", "genotype": "KO"},
        {"id": "A3", "sex": "Female", "genotype": "x/y"},
        {"id": "A4", "sex": "Male", "genotype": "KO"},
        {"id": "A5", "sex": "Female", "genotype": "WT"},
        {"id": "A6", "sex": "Male", "genotype": "HET"},
    ]
}


def _shipped_zeitgeber_config():
    with open(REPO / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["analysis"]["zeitgeber"]


@pytest.fixture
def zeitgeber(monkeypatch):
    spec = importlib.util.spec_from_file_location("extract_zeitgeber_features_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded by __name__, so nothing runs
    # The real setup points sys.stdout and sys.stderr at the rule's log files and resets the
    # root logger, which would break pytest's output capture for the rest of the session
    monkeypatch.setattr(module, "setup_snakemake_logging", lambda _: logging.getLogger(__name__))
    return module


def _run_main(zeitgeber, paths, output, **settings):
    zeitgeber.snakemake = SimpleNamespace(
        input=SimpleNamespace(
            war_parquet=[str(parquet_path) for parquet_path, _ in paths],
            war_json=[str(json_path) for _, json_path in paths],
        ),
        output=SimpleNamespace(zeitgeber_features=str(output)),
        params=SimpleNamespace(
            config={"analysis": {"zeitgeber": {**_shipped_zeitgeber_config(), **settings}}},
            samples_config=SAMPLES_CONFIG,
        ),
        threads=1,
    )
    zeitgeber.main()
    return pd.read_pickle(output)


@pytest.mark.mutates_constants
def test_main_loads_each_war_once_in_sorted_order(
    zeitgeber, create_war_files, track_war_loads, tmp_path
):
    result = _run_main(zeitgeber, create_war_files(ANIMALS), tmp_path / "zeitgeber_features.pkl")

    folders = sorted(folder for folder, *_ in ANIMALS)
    assert track_war_loads.folders == folders, "one load per animal, in sorted order"
    assert track_war_loads.alive_at_load == [0] * len(ANIMALS), "no WAR may outlive the next load"
    assert sorted(result["animal"].unique()) == folders, "animals are named by folder"


@pytest.mark.mutates_constants
def test_output_does_not_depend_on_input_order(zeitgeber, create_war_files, tmp_path):
    paths = create_war_files(ANIMALS)

    forward = _run_main(zeitgeber, paths, tmp_path / "forward.pkl")
    backward = _run_main(zeitgeber, paths[::-1], tmp_path / "backward.pkl")

    pd.testing.assert_frame_equal(forward, backward, check_exact=True)
    assert (tmp_path / "forward.pkl").read_bytes() == (tmp_path / "backward.pkl").read_bytes()


@pytest.mark.mutates_constants
def test_every_configured_feature_reaches_the_output(zeitgeber, create_war_files, tmp_path):
    """psdslope's two columns used to be left out, by a naming rule that kept psdslope whole."""
    from neurodent.core.utils import channel_averaged_columns

    result = _run_main(zeitgeber, create_war_files(ANIMALS), tmp_path / "zeitgeber_features.pkl")

    columns = [c for f in _shipped_zeitgeber_config()["features"] for c in channel_averaged_columns(f)]
    assert {"psdslope_slope", "psdslope_intercept"} <= set(columns)
    assert set(result.columns) == (
        {"animal", "genotype", "sex", "zt_minutes", "daynight"}
        | set(columns)
        | {f"{c}_nobase" for c in columns}
    )
    assert not result[columns].isna().all().any()


@pytest.mark.mutates_constants
@pytest.mark.parametrize("dropped", [["lognspike"], ["psdslope"], ["zcohere", "zimcoh"]])
def test_a_war_without_a_configured_feature_raises(zeitgeber, create_war_files, tmp_path, dropped):
    """A missing column used to be left out of the output without a word."""
    animals = [a for a in ANIMALS if a[0] != "a5"] + [("a5", "A5", "WT", "Female", {"drop": dropped})]

    with pytest.raises(RuntimeError, match="Failed to process animal a5") as excinfo:
        _run_main(zeitgeber, create_war_files(animals), tmp_path / "zeitgeber_features.pkl")

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert dropped[0] in str(excinfo.value.__cause__)


def test_bins_match_averaging_every_animal_together(create_war_files):
    """Averaging each animal's windows into bins on its own gives the bins that averaging all
    animals' windows together gave, as the script did before."""
    from neurodent import constants
    from neurodent.results import WindowAnalysisResult
    from neurodent.results.zeitgeber import ZeitgeberAnalysisResult, aggregate_zeitgeber_bins

    features = ["logrms", "logpsdband", "psdslope", "zpcorr"]
    animals = [ANIMALS[0], ANIMALS[2], ANIMALS[5]]
    config = {"baseline_hours": 12, "animal_metadata": {
        a[1]: {"genotype": a[2], "sex": a[3]} for a in animals
    }}
    frames, bins = [], []
    for (folder, *_), (parquet, _) in zip(animals, create_war_files(animals)):
        war = WindowAnalysisResult.load_parquet_and_json(parquet.parent)
        frame = ZeitgeberAnalysisResult(war, **config).get_channel_averaged_result(features=features)
        frame["animal"] = folder
        frames.append(frame)
        bins.append(aggregate_zeitgeber_bins(frame, features))

    together = pd.concat(frames, ignore_index=True)
    columns = [c for c in together.columns if c in set(bins[0].columns) - {"animal", "genotype", "sex", "zt_minutes", "daynight"}]
    expected = together.groupby(["animal", "genotype", "sex", "zt_minutes", "daynight"])[columns].mean().reset_index()

    pd.testing.assert_frame_equal(pd.concat(bins, ignore_index=True), expected, check_exact=True)


@pytest.mark.mutates_constants
def test_the_shipped_settings_are_the_ones_read(zeitgeber):
    """interval_minutes was time_aggregation_minutes, which nothing read, and baseline_hours and
    exclude_from_baseline sat under zeitgeber_plots, where nothing read them either."""
    from neurodent.results.zeitgeber import _zeitgeber_settings

    settings = _shipped_zeitgeber_config()
    assert set(settings) - {"features"} <= set(_zeitgeber_settings())
    assert {"interval_minutes", "baseline_hours", "exclude_from_baseline"} <= set(settings)


@pytest.mark.mutates_constants
@pytest.mark.parametrize("interval", [30, 120])
def test_interval_minutes_sets_the_bins(zeitgeber, create_war_files, tmp_path, interval):
    result = _run_main(
        zeitgeber, create_war_files(ANIMALS[:2]), tmp_path / "z.pkl", interval_minutes=interval
    )

    assert set(result["zt_minutes"] % interval) == {0}
    assert result.groupby("animal")["zt_minutes"].nunique().max() == 1440 // interval


@pytest.mark.mutates_constants
def test_baseline_hours_and_exclude_from_baseline_are_read(zeitgeber, create_war_files, tmp_path):
    paths = create_war_files(ANIMALS[:2])

    twelve = _run_main(zeitgeber, paths, tmp_path / "twelve.pkl", baseline_hours=12)
    six = _run_main(
        zeitgeber, paths, tmp_path / "six.pkl", baseline_hours=6, exclude_from_baseline=["logrms"]
    )

    assert "logrms_nobase" in twelve.columns and "logrms_nobase" not in six.columns
    assert not np.allclose(twelve["logampvar_nobase"], six["logampvar_nobase"])
    pd.testing.assert_series_equal(twelve["logampvar"], six["logampvar"])


def test_extraction_errors_name_the_animal():
    class UnreadableWar:
        def get_channel_averaged_result(self, **kwargs):
            raise ValueError("no timestamp column")

    with pytest.raises(RuntimeError, match="Failed to process animal a1") as excinfo:
        _extract_zeitgeber_frame(UnreadableWar(), "a1", ["logrms"], {})

    assert isinstance(excinfo.value.__cause__, ValueError)
