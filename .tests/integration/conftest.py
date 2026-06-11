"""Session-scoped sandbox for snakemake integration tests.

Snakemake rules write outputs to relative ``results/...`` paths.  When the
integration test invokes snakemake with ``cwd=repo_root``, these outputs
clobber the user's real pipeline outputs in a shared dev workspace.

The :func:`snakemake_sandbox` fixture below gives each test session a
private cwd whose read-only repo contents are mirrored via symlinks and
whose mutable outputs (``results/``, ``logs/``, ``.snakemake/``) land in
a tmp dir that pytest cleans up.  Production pipeline runs from the repo
root are never touched.
"""
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]

# Read-only top-level entries that snakemake/uv need access to. Mutable
# directories — `results/`, `logs/`, `.snakemake/`, `.venv/` — are NOT
# listed; they land fresh inside the sandbox as snakemake writes there.
_READ_ONLY_TOP_LEVEL = (
    "workflow",
    "config",
    "src",
    ".tests",
    "tests",       # snakemake config references `tests/integration/readers.py`
    "scripts",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
)


@pytest.fixture(scope="session")
def snakemake_sandbox(tmp_path_factory):
    """Per-session sandbox cwd for invoking snakemake.

    Read-only repo contents (workflow/, config/, src/, etc.) are symlinked
    in.  Mutable directories (`results/`, `logs/`, `.snakemake/`) are NOT
    symlinked — they land naturally as snakemake writes to relative paths
    from this cwd, and pytest cleans the whole sandbox after the session.
    """
    sandbox = tmp_path_factory.mktemp("neurodent_pipeline_sandbox")
    for name in _READ_ONLY_TOP_LEVEL:
        src = _REPO_ROOT / name
        if src.exists():
            (sandbox / name).symlink_to(src)
    return sandbox
