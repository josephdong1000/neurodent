"""
Snakemake Integration Tests
============================

Two levels of validation for the Snakemake pipeline:

1. **Dry-run** (``TestSnakemakeDryRun``) — validates DAG construction by
   invoking ``snakemake --dryrun``.  Catches config typos, missing keys,
   broken imports, and invalid wildcard resolution without executing jobs.

2. **Full pipeline run** (``TestSnakemakePipelineRun``) — actually executes
   the ``war_generation`` rule against the committed mini-real dataset in
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
                only ``war_generation`` for the first animal is requested so the
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

    Only the ``war_generation`` rule is exercised so that the test stays fast
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
        extra_env: dict[str, str] | None = None,
        timeout: int = 600,
    ) -> subprocess.CompletedProcess:
        """Invoke ``snakemake`` as a subprocess for *targets*.

        Args:
            targets: List of output file targets to build.
            extra_args: Additional flags inserted before the target list.
            extra_env: Additional environment variables to set (e.g.
                ``{"NEURODENT_MEMRAY": "1"}``).
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
        if extra_env:
            env.update(extra_env)

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

        The ``war_generation`` rule outputs three artifacts that all need cleanup:
        - ``results/wars/{animal}/``    — WAR parquet + JSON
        - ``results/fdsars/{animal}/``  — FDSAR spike detection directory
        - ``logs/war_generation/``      — Snakemake job logs
        """
        for subdir in (
            f"results/wars/{self.ANIMAL}",
            f"results/fdsars/{self.ANIMAL}",
            f"results/wars_quality_filtered/{self.ANIMAL}",
            f"results/wars_standardized/{self.ANIMAL}",
            f"results/wars_fragment_filtered/{self.ANIMAL}",
            "logs/war_generation",
            "logs/war_quality_filter",
            "logs/war_standardize",
            "logs/war_fragment_filtering",
        ):
            shutil.rmtree(_REPO_ROOT / subdir, ignore_errors=True)

    def test_war_generation_produces_outputs(self):
        """``war_generation`` runs to completion and produces WAR ``.parquet`` and ``.json``."""
        war_parquet = _REPO_ROOT / f"results/wars/{self.ANIMAL}/war.parquet"
        war_json = _REPO_ROOT / f"results/wars/{self.ANIMAL}/war.json"

        self._cleanup()
        try:
            result = self._run_snakemake(
                [f"results/wars/{self.ANIMAL}/war.parquet"],
                extra_args=["--forcerun", "war_generation"],
            )
            assert result.returncode == 0, (
                f"Snakemake run failed for '{self.DATASET}' dataset, "
                f"animal '{self.ANIMAL}'.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            assert war_parquet.exists(), f"Expected WAR parquet not created: {war_parquet}"
            assert war_json.exists(), f"Expected WAR JSON not created: {war_json}"
            # Pickle should no longer be produced
            war_pkl = _REPO_ROOT / f"results/wars/{self.ANIMAL}/war.pkl"
            assert not war_pkl.exists(), (
                f"Pickle file should no longer be produced: {war_pkl}"
            )
        finally:
            self._cleanup()

    def test_war_generation_with_memray(self):
        """``war_generation`` with NEURODENT_MEMRAY=1 produces ``memray.bin`` alongside WAR outputs."""
        war_parquet = _REPO_ROOT / f"results/wars/{self.ANIMAL}/war.parquet"
        war_json = _REPO_ROOT / f"results/wars/{self.ANIMAL}/war.json"
        memray_bin = _REPO_ROOT / f"results/wars/{self.ANIMAL}/memray.bin"

        self._cleanup()
        try:
            result = self._run_snakemake(
                [f"results/wars/{self.ANIMAL}/war.parquet"],
                extra_args=["--forcerun", "war_generation"],
                extra_env={"NEURODENT_MEMRAY": "1"},
            )
            assert result.returncode == 0, (
                f"Snakemake run with memray failed for '{self.DATASET}' dataset, "
                f"animal '{self.ANIMAL}'.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            # WAR outputs still produced
            assert war_parquet.exists(), f"Expected WAR parquet not created: {war_parquet}"
            assert war_json.exists(), f"Expected WAR JSON not created: {war_json}"
            # memray.bin produced and non-empty
            assert memray_bin.exists(), f"Expected memray.bin not created: {memray_bin}"
            assert memray_bin.stat().st_size > 0, "memray.bin is empty"
        finally:
            # Also clean up memray.bin
            if memray_bin.exists():
                memray_bin.unlink()
            self._cleanup()

    def test_downstream_chain_produces_outputs(self):
        """Exercise the WAR post-generation chain end-to-end.

        Runs ``war_generation`` → ``war_quality_filter`` (checkpoint) →
        ``war_standardize`` → ``war_fragment_filtering`` (checkpoint) and
        asserts that each stage produces the expected ``war.parquet`` +
        ``war.json`` artifacts and never produces a legacy ``war.pkl``.

        This is the guard that would have caught the orphaned-parquet bug:
        if any downstream rule reverts to declaring ``war.pkl`` as its
        tracked output, the missing-file assertions will fire.
        """
        stage_dirs = {
            "quality_filtered": _REPO_ROOT / f"results/wars_quality_filtered/{self.ANIMAL}",
            "standardized": _REPO_ROOT / f"results/wars_standardized/{self.ANIMAL}",
            "fragment_filtered": _REPO_ROOT / f"results/wars_fragment_filtered/{self.ANIMAL}",
        }
        fragment_parquet = stage_dirs["fragment_filtered"] / "war.parquet"
        fragment_json = stage_dirs["fragment_filtered"] / "war.json"

        self._cleanup()
        try:
            result = self._run_snakemake(
                [f"results/wars_fragment_filtered/{self.ANIMAL}/war.parquet"],
                extra_args=["--forcerun", "war_generation"],
            )
            assert result.returncode == 0, (
                f"Snakemake downstream chain run failed for '{self.DATASET}' dataset, "
                f"animal '{self.ANIMAL}'.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

            # Final output: fragment-filtered WAR
            assert fragment_parquet.exists(), (
                f"Expected fragment-filtered WAR parquet not created: {fragment_parquet}"
            )
            assert fragment_json.exists(), (
                f"Expected fragment-filtered WAR JSON not created: {fragment_json}"
            )

            # Every intermediate stage must produce parquet + json, never pkl.
            for stage_name, stage_dir in stage_dirs.items():
                assert stage_dir.exists(), (
                    f"Stage '{stage_name}' directory missing: {stage_dir}"
                )
                assert (stage_dir / "war.parquet").exists(), (
                    f"Stage '{stage_name}' missing war.parquet at {stage_dir}"
                )
                assert (stage_dir / "war.json").exists(), (
                    f"Stage '{stage_name}' missing war.json at {stage_dir}"
                )
                assert not (stage_dir / "war.pkl").exists(), (
                    f"Stage '{stage_name}' should not produce war.pkl at {stage_dir}"
                )
        finally:
            self._cleanup()
