"""Pytest configuration for Snakemake integration tests."""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so that `tests.*` helper modules
# (e.g. tests.data.generate) are importable from this directory.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Re-export shared fixtures from tests/conftest.py so they are available here.
from tests.conftest import example_dataset  # noqa: E402, F401
