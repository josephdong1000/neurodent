"""
Snakemake Pipeline Unit Tests
=============================

Spot-checks for helper functions and configuration used by the
Snakemake pipeline (``workflow/Snakefile``).

These live under ``.tests/unit/`` following the Snakemake Workflow
Catalog layout:

- ``tests/``              – unit tests of package modules
- ``tests/integration/``  – integration tests of package modules
- ``.tests/unit/``        – unit tests of snakemake functions
- ``.tests/integration/`` – integration tests of full snakemake pipeline

Running
-------
::

    uv run pytest .tests/ -v

"""

import copy
from pathlib import Path

import jsonschema
import pytest
import yaml

from neurodent.workflow.utils import format_config_value, increment_memory
from neurodent.workflow.utils.config import load_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Tests: format_config_value
# ---------------------------------------------------------------------------


class TestFormatConfigValue:
    """Verify the Snakefile config pretty-printer."""

    def test_string(self):
        assert format_config_value("hello") == '"hello"'

    def test_none(self):
        assert format_config_value(None) == "null"

    def test_integer(self):
        assert format_config_value(42) == "42"

    def test_empty_dict(self):
        assert format_config_value({}) == "{}"

    def test_empty_list(self):
        assert format_config_value([]) == "[]"

    def test_flat_dict(self):
        result = format_config_value({"key": "val"}, indent=0)
        assert 'key: "val"' in result

    def test_nested_dict(self):
        result = format_config_value({"outer": {"inner": 1}}, indent=0)
        assert "outer:" in result
        assert "inner: 1" in result

    def test_list_values(self):
        result = format_config_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_list_of_strings(self):
        result = format_config_value(["a", "b"])
        assert result == "['a', 'b']"


# ---------------------------------------------------------------------------
# Tests: increment_memory
# ---------------------------------------------------------------------------


class TestIncrementMemory:
    """Verify the Snakefile retry-memory helper."""

    def test_first_attempt(self):
        mem = increment_memory(4000)
        assert mem(None, 1) == 4000

    def test_second_attempt_doubles(self):
        mem = increment_memory(4000)
        assert mem(None, 2) == 8000

    def test_third_attempt_quadruples(self):
        mem = increment_memory(4000)
        assert mem(None, 3) == 16000

    def test_different_base(self):
        mem = increment_memory(1024)
        assert mem(None, 1) == 1024
        assert mem(None, 2) == 2048


# ---------------------------------------------------------------------------
# Tests: config schema is well-formed YAML
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Smoke-test that pipeline configuration files are valid YAML."""

    def test_schema_loads(self):
        schema_path = REPO_ROOT / "workflow" / "schemas" / "config.schema.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_main_config_loads(self):
        config_path = REPO_ROOT / "config" / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    @pytest.mark.parametrize(
        "dataset",
        [p.stem for p in sorted((REPO_ROOT / "config" / "datasets").glob("*.yaml"))],
    )
    def test_dataset_config_loads(self, dataset):
        path = REPO_ROOT / "config" / "datasets" / f"{dataset}.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict), (
            f"Dataset config '{dataset}' should be a non-empty YAML mapping"
        )


def _schema():
    return yaml.safe_load(
        (REPO_ROOT / "workflow" / "schemas" / "config.schema.yaml").read_text(encoding="utf-8")
    )


def _merged_config(dataset, tmp_path):
    return load_dataset_config(
        dataset,
        config_path=REPO_ROOT / "config" / "config.yaml",
        local_path=tmp_path / "no_local_override.yaml",
        datasets_dir=REPO_ROOT / "config" / "datasets",
    )


def _errors(config):
    schema = _schema()
    validator = jsonschema.validators.validator_for(schema)
    validator.check_schema(schema)
    return [error.message for error in validator(schema).iter_errors(config)]


class TestClusterSchema:
    """The schema lists the keys of every cluster block and refuses anything else."""

    @pytest.mark.parametrize(
        "dataset",
        [p.stem for p in sorted((REPO_ROOT / "config" / "datasets").glob("*.yaml"))],
    )
    def test_every_dataset_config_validates(self, dataset, tmp_path):
        assert _errors(_merged_config(dataset, tmp_path)) == []

    @pytest.mark.parametrize("block", ["split_joint_recordings", "frequency_domain_spike_detection"])
    def test_the_unused_blocks_are_gone_and_refused(self, block, tmp_path):
        config = _merged_config("example", tmp_path)
        assert block not in config["cluster"]
        config["cluster"][block] = copy.deepcopy(config["cluster"]["ep_figures"])
        assert any(block in message for message in _errors(config))

    @pytest.mark.parametrize(
        "block, key",
        [
            ("diagnostic_figures", "interface"),  # only war_generation starts a Dask cluster
            ("ep_figures", "mem_bytes_per_window"),  # its rule reads no WARs
            ("war_standardize", "seconds_per_value"),  # its time is not computed
            ("war_generation", "mem_mb_floor"),  # a typo
        ],
    )
    def test_a_key_its_rules_do_not_read_is_refused(self, block, key, tmp_path):
        config = _merged_config("example", tmp_path)
        config["cluster"][block][key] = 1
        assert any(key in message for message in _errors(config))

    @pytest.mark.parametrize(
        "block, key",
        [("war_zeitgeber", "seconds_per_window"), ("war_flattening", "mem_bytes_per_window"),
         ("notebook", "retries"), ("war_generation", "interface")],
    )
    def test_a_key_its_rules_need_is_required(self, block, key, tmp_path):
        config = _merged_config("example", tmp_path)
        del config["cluster"][block][key]
        assert any(key in message for message in _errors(config))
