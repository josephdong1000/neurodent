"""generate_relfreq_plots loads each WAR once, one at a time, and writes what it wrote before.

The script used to load every WAR once per feature, 15 at a time in a pool, which ran the
IQSEC2 run out of memory at 150 and then 300 GB. It now takes every feature from a single
load per animal. These tests pin what that must not change: each feature's frame equals the
frame of a fresh one-feature load, each CSV equals what the old per-feature path wrote, and
no WAR is still alive when the next one loads.
"""

import importlib.util
import logging
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from neurodent import constants
from neurodent.results import WindowAnalysisResult

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "workflow" / "scripts" / "generate_relfreq_plots.py"

# Folder, animal id, genotype, sex, and how each WAR departs from the default 3-channel day.
# Together they cover the ways datasets differ: channel count, recording days, a single
# window, features a WAR lacks, a channel dead throughout, genotypes outside the default plot
# order, and more genotypes than the fixed palette. Input order is not sorted.
ANIMALS = [
    ("a2", "A2", "KO", "Female", {}),
    ("a1", "A1", "WT", "Male", {"channels": [
        "LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis",
    ]}),
    ("a3", "A3", "x/y", "Female", {"channels": ["LMot", "RMot"], "days": 3}),
    ("a6", "A6", "HET", "Male", {"windows": 1}),
    ("a5", "A5", "WT", "Female", {"drop": ["nspike", "lognspike", "psdslope"]}),
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


def _shipped_relfreq_features():
    with open(REPO / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["analysis"]["relfreq_plots"]["features"]


@pytest.fixture
def relfreq(monkeypatch):
    spec = importlib.util.spec_from_file_location("generate_relfreq_plots_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded by __name__, so nothing runs
    # The real setup points sys.stdout and sys.stderr at the rule's log files and resets the
    # root logger, which would break pytest's output capture for the rest of the session
    monkeypatch.setattr(module, "setup_snakemake_logging", lambda _: logging.getLogger(__name__))
    return module


def _old_extraction(load, parquet_path, feature):
    """What the old per-feature path returned for one animal: a fresh load, then one feature."""
    war = load(folder_path=parquet_path.parent, parquet_name=parquet_path.name, json_name="war.json")
    df = war.get_channel_averaged_result(features=[feature])
    df["animal"] = war.animal_id
    df["genotype"] = war.genotype
    if "isday" not in df.columns and "isday" in war.result.columns:
        df["isday"] = war.result["isday"].values[: len(df)]
    return df


def _old_csv(relfreq, load, parquet_paths, feature, path):
    """Write the CSV the old per-feature path wrote, one fresh load per animal, and return its bytes.

    It is written to a file with the script's own call, so line endings are the platform's in
    both files. Reading a file back as text turns Windows line endings into newlines, so the
    comparison is of bytes.
    """
    combined = pd.concat(
        [_old_extraction(load, parquet_path, feature) for parquet_path in parquet_paths],
        ignore_index=True,
    )
    if feature in set(constants.BAND_FEATURES) | set(constants.BANDED_MATRIX_FEATURES):
        combined = relfreq.melt_banded_feature(combined, feature)
    df_weighted = relfreq.add_animal_weights(
        relfreq.process_feature_dataframe(combined, SAMPLES_CONFIG)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df_weighted.to_csv(path, index=False)
    return path.read_bytes()


def test_one_load_gives_the_frames_of_one_load_per_feature(relfreq, create_war_files):
    features = _shipped_relfreq_features()
    load = WindowAnalysisResult.load_parquet_and_json

    for parquet_path, _ in create_war_files(ANIMALS):
        war = load(folder_path=parquet_path.parent, parquet_name="war.parquet", json_name="war.json")
        before = pickle.dumps(war.result)

        frames = relfreq.extract_features_from_war(war, features)

        assert pickle.dumps(war.result) == before, "extraction must leave the WAR unchanged"
        assert list(frames) == features
        for feature in features:
            pd.testing.assert_frame_equal(
                frames[feature], _old_extraction(load, parquet_path, feature), check_exact=True
            )


@pytest.mark.mutates_constants
def test_main_loads_each_war_once_and_writes_what_the_old_path_wrote(
    relfreq, create_war_files, track_war_loads, tmp_path
):
    features = _shipped_relfreq_features()
    paths = create_war_files(ANIMALS)
    relfreq.snakemake = SimpleNamespace(
        input=SimpleNamespace(
            war_parquet=[str(parquet_path) for parquet_path, _ in paths],
            war_json=[str(json_path) for _, json_path in paths],
        ),
        output=SimpleNamespace(
            figure_dir=str(tmp_path / "figures"), data_dir=str(tmp_path / "data")
        ),
        params=SimpleNamespace(
            config={
                "analysis": {
                    "relfreq_plots": {
                        "features": features,
                        "figure_format": "png",
                        "data_format": "csv",
                        "dpi": 10,
                    }
                }
            },
            samples_config=SAMPLES_CONFIG,
        ),
        threads=1,
    )

    relfreq.main()

    assert track_war_loads.folders == [folder for folder, *_ in ANIMALS], "one load per animal, in input order"
    assert track_war_loads.alive_at_load == [0] * len(ANIMALS), "no WAR may outlive the next load"

    for feature in features:
        written = (tmp_path / "data" / f"{feature}_relfreq.csv").read_bytes()
        expected = _old_csv(
            relfreq,
            track_war_loads.real_load,
            [parquet_path for parquet_path, _ in paths],
            feature,
            tmp_path / "old_path" / f"{feature}_relfreq.csv",
        )
        assert written == expected, f"{feature}_relfreq.csv differs from the old path's"

    banded = set(constants.BAND_FEATURES) | set(constants.BANDED_MATRIX_FEATURES)
    expected_pngs = []
    for feature in features:
        if feature in banded:
            expected_pngs += [f"{feature}_relfreq_{band}.png" for band in constants.BAND_NAMES]
            expected_pngs.append(f"{feature}_relfreq_byband.png")
        else:
            expected_pngs.append(f"{feature}_relfreq.png")
    assert sorted(path.name for path in (tmp_path / "figures").iterdir()) == sorted(expected_pngs)
