"""
Snakemake Dry-Run Integration Test
===================================

Validates Snakemake DAG construction by invoking ``snakemake --dryrun``
as a subprocess.  This is the Snakemake Workflow Catalog-recommended
integration test: it exercises the full workflow definition — config
loading, rule imports, wildcard resolution — without running any jobs.

Running
-------
::

    uv run pytest .tests/integration/ -v -m integration

"""

import os
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Snakemake Dry-Run Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSnakemakeDryRun:
    """Validate Snakemake DAG construction via ``--dryrun`` for test datasets.

    These tests invoke ``snakemake --dryrun`` as a subprocess to ensure
    that the Snakefile, dataset configs, and sample JSONs are all
    consistent and produce a valid execution plan.  No data is actually
    processed — only the DAG is built and validated.

    This is the recommended way to smoke-test the Snakemake pipeline in
    CI: it catches config typos, missing keys, broken imports, and
    invalid wildcard resolution without the cost of running the full
    pipeline.
    """

    @staticmethod
    def _run_snakemake_dryrun(dataset: str, targets: list[str] | None = None):
        """Run ``snakemake --dryrun`` for *dataset* and return the result.

        Args:
            dataset: Name of the dataset config (e.g. ``"example"``).
            targets: Optional list of target rules/files.  When ``None``,
                only ``make_war`` for the first animal is requested so the
                dry-run stays fast.

        Returns:
            ``subprocess.CompletedProcess`` — caller should check
            ``result.returncode``.
        """
        cmd = [
            "uv", "run", "snakemake",
            "--snakefile", "workflow/Snakefile",
            "--dryrun",
            "--ignore-incomplete",
            "--cores", "1",
            "--quiet",
        ]
        if targets:
            cmd.extend(targets)

        env = {
            **os.environ,
            "NEURODENT_DATASET": dataset,
        }

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            timeout=60,
        )

    def test_example_dataset_dryrun(self):
        """``snakemake --dryrun`` succeeds for the *example* dataset."""
        result = self._run_snakemake_dryrun("example")
        assert result.returncode == 0, (
            f"Snakemake dry-run failed for 'example' dataset.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_mini_real_dataset_dryrun(self):
        """``snakemake --dryrun`` succeeds for the *mini_real* dataset."""
        result = self._run_snakemake_dryrun("mini_real")
        assert result.returncode == 0, (
            f"Snakemake dry-run failed for 'mini_real' dataset.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
