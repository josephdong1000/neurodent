"""
Snakemake Integration Tests
============================

Two levels of validation for the Snakemake pipeline:

1. **Dry-run** (``TestSnakemakeDryRun``) — validates DAG construction by
   invoking ``snakemake --dryrun``.  Catches config typos, missing keys,
   broken imports, and invalid wildcard resolution without executing jobs.

2. **Full pipeline run** (``TestSnakemakePipelineRun``) — actually executes
   the ``make_war`` rule against the committed mini-real dataset in
   ``.tests/integration/data/``.  Verifies that the pipeline produces
   the expected WAR output files end-to-end.

Input data, configuration specifications, and execution are all contained
in ``.tests/integration/`` per Snakemake Workflow Catalog requirements:
https://snakemake.github.io/snakemake-workflow-catalog/?rules=true

Running
-------
::

    uv run pytest .tests/integration/ -v -m integration

"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Repository root (two levels above this file: .tests/integration/ → root)
_REPO_ROOT = Path(__file__).resolve().parents[2]


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
            cwd=str(_REPO_ROOT),
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


# ---------------------------------------------------------------------------
# Full Pipeline Run Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestSnakemakePipelineRun:
    """Execute the Snakemake pipeline end-to-end against mini-real data.

    Unlike the dry-run tests above, these tests actually invoke Snakemake
    jobs against the committed mini-real dataset stored in
    ``.tests/integration/data/``.  They verify that the pipeline runs to
    completion and produces the expected output artifacts.

    Only the ``make_war`` rule is exercised so that the test stays fast
    while still covering the most critical data-loading and analysis path.
    """

    # Animal slug produced by Django's slugify("A10") → "a10"
    ANIMAL = "a10"
    DATASET = "mini_real"

    @classmethod
    def _run_snakemake(
        cls,
        targets: list[str],
        extra_args: list[str] | None = None,
        timeout: int = 600,
    ) -> subprocess.CompletedProcess:
        """Invoke ``snakemake`` as a subprocess for *targets*.

        Args:
            targets: List of output file targets to build.
            extra_args: Additional flags inserted before the target list.
            timeout: Subprocess timeout in seconds.

        Returns:
            ``subprocess.CompletedProcess`` — caller should check
            ``result.returncode``.
        """
        cmd = [
            "uv", "run", "snakemake",
            "--snakefile", "workflow/Snakefile",
            "--cores", "all",
            "--rerun-incomplete",
        ]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(["--"] + targets)

        env = {
            **os.environ,
            "NEURODENT_DATASET": cls.DATASET,
        }

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=env,
            timeout=timeout,
        )

    def _cleanup(self) -> None:
        """Remove output directories created by the test.

        The ``make_war`` rule outputs three artifacts that all need cleanup:
        - ``results/wars/{animal}/``    — WAR pickle + JSON
        - ``results/fdsars/{animal}/``  — FDSAR spike detection directory
        - ``logs/war_generation/``      — Snakemake job logs
        """
        for subdir in (
            f"results/wars/{self.ANIMAL}",
            f"results/fdsars/{self.ANIMAL}",
            "logs/war_generation",
        ):
            shutil.rmtree(_REPO_ROOT / subdir, ignore_errors=True)

    def test_make_war_produces_outputs(self):
        """``make_war`` runs to completion and produces WAR ``.pkl`` and ``.json``."""
        war_pkl = _REPO_ROOT / f"results/wars/{self.ANIMAL}/war.pkl"
        war_json = _REPO_ROOT / f"results/wars/{self.ANIMAL}/war.json"

        self._cleanup()
        try:
            result = self._run_snakemake(
                [f"results/wars/{self.ANIMAL}/war.pkl"],
                extra_args=["--forcerun", "make_war"],
            )
            assert result.returncode == 0, (
                f"Snakemake run failed for '{self.DATASET}' dataset, "
                f"animal '{self.ANIMAL}'.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            assert war_pkl.exists(), f"Expected WAR pickle not created: {war_pkl}"
            assert war_json.exists(), f"Expected WAR JSON not created: {war_json}"
        finally:
            self._cleanup()
