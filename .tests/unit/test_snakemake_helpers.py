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

from pathlib import Path

import pytest
import yaml

from neurodent.workflow.utils import format_config_value, increment_memory

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
