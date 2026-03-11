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

import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers copied from workflow/Snakefile so they can be tested without
# importing the Snakefile itself (which requires Snakemake runtime).
# Any change to these functions in the Snakefile should be reflected here.
# ---------------------------------------------------------------------------


def format_config_value(value, indent=4):
    """Format a config value for display (handles nested dicts, lists, etc.)."""
    spaces = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for k, v in value.items():
            formatted_val = format_config_value(v, indent + 2)
            if "\n" in formatted_val:
                lines.append(f"{spaces}{k}:")
                lines.append(formatted_val)
            else:
                lines.append(f"{spaces}{k}: {formatted_val}")
        return "\n".join(lines)
    elif isinstance(value, list):
        if not value:
            return "[]"
        return f"[{', '.join(repr(v) for v in value)}]"
    elif isinstance(value, str):
        return f'"{value}"'
    elif value is None:
        return "null"
    else:
        return str(value)


def increment_memory(base_memory):
    """Return a callable ``mem(wildcards, attempt)`` that doubles on each retry."""
    def mem(wildcards, attempt):
        return base_memory * (2 ** (attempt - 1))
    return mem


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

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        assert cfg is None or isinstance(cfg, dict)
