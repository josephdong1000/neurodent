"""The config schema refuses zeitgeber keys the pipeline does not read.

The config used to say time_aggregation_minutes where the code read interval_minutes, and to
put baseline_hours and exclude_from_baseline under zeitgeber_plots, where nothing read them, so
none of the three had any effect. Unknown keys were dropped without a word.
"""

import copy
from pathlib import Path

import jsonschema
import pytest
import yaml

from neurodent.results.zeitgeber import _zeitgeber_settings
from neurodent.workflow.utils.config import merge_dataset_config

REPO = Path(__file__).parent.parent
DATASETS = sorted(p.stem for p in (REPO / "config" / "datasets").glob("*.yaml"))


def _schema():
    with open(REPO / "workflow" / "schemas" / "config.schema.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _config(dataset="example"):
    with open(REPO / "config" / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    merged, _ = merge_dataset_config(config, dataset, datasets_dir=REPO / "config" / "datasets")
    return merged


def _validate(config):
    jsonschema.Draft202012Validator(_schema()).validate(config)


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_shipped_dataset_validates(dataset):
    _validate(_config(dataset))


@pytest.mark.parametrize(
    "block, key, value",
    [
        ("zeitgeber", "time_aggregation_minutes", 60),
        ("zeitgeber", "figsize", [10, 20]),
        ("zeitgeber_plots", "baseline_hours", 12),
        ("zeitgeber_plots", "exclude_from_baseline", []),
        ("zeitgeber_plots", "features", ["logrms"]),
    ],
)
def test_a_key_nothing_reads_is_refused(block, key, value):
    config = _config()
    config["analysis"][block][key] = value

    with pytest.raises(jsonschema.ValidationError, match=key):
        _validate(config)


@pytest.mark.parametrize(
    "key, value",
    [("zeitgeber_shift_hours", 7), ("baseline_window", "night"), ("baseline_window", [0, 6]), ("interval_minutes", 30)],
)
def test_every_setting_the_pipeline_reads_is_accepted(key, value):
    config = _config()
    config["analysis"]["zeitgeber"][key] = value

    _validate(config)


def test_the_schema_accepts_exactly_the_settings_zeitgeber_reads():
    """The pipeline passes the block, less features, to ZeitgeberAnalysisResult, so the
    schema's keys are its settings, without the ones the script sets itself."""
    allowed = set(_schema()["properties"]["analysis"]["properties"]["zeitgeber"]["properties"])
    set_by_the_script = {"shift_for_48h", "animal_metadata"}
    deprecated = {"genotype_pattern", "sex_mapper", "genotype_aliases"}
    assert allowed - {"features"} == set(_zeitgeber_settings()) - set_by_the_script - deprecated


def test_a_bad_value_is_refused():
    config = copy.deepcopy(_config())
    config["analysis"]["zeitgeber"]["baseline_window"] = "dusk"

    with pytest.raises(jsonschema.ValidationError):
        _validate(config)
