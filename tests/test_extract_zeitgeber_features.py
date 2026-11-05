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
# are not the ids, as on disk, where folder iqsec2_80 holds IQSEC2_80.
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


def _shipped_zeitgeber_config():
    with open(REPO / "config" / "config.yaml") as f:
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


def _run_main(zeitgeber, paths, output):
    zeitgeber.snakemake = SimpleNamespace(
        input=SimpleNamespace(
            war_parquet=[str(parquet_path) for parquet_path, _ in paths],
            war_json=[str(json_path) for _, json_path in paths],
        ),
        output=SimpleNamespace(zeitgeber_features=str(output)),
        params=SimpleNamespace(
            config={"analysis": {"zeitgeber": _shipped_zeitgeber_config()}},
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


def test_extraction_errors_name_the_animal():
    class UnreadableWar:
        def get_channel_averaged_result(self, **kwargs):
            raise ValueError("no timestamp column")

    with pytest.raises(RuntimeError, match="Failed to process animal a1") as excinfo:
        _extract_zeitgeber_frame(UnreadableWar(), "a1", ["logrms"], {})

    assert isinstance(excinfo.value.__cause__, ValueError)
