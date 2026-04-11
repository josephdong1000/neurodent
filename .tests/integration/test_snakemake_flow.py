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

    Tests range from a narrow ``war_generation``-only smoke test (for fast
    feedback on the most critical data-loading and analysis path) up to the
    authoritative ``test_full_pipeline_all_target`` which runs every rule
    wired into the default ``all`` target.
    """

    # Animal slugs produced by Django's slugify() on the mini_real sample IDs.
    # A10 → "a10", F22 → "f22"
    ANIMAL = "a10"
    ANIMALS = ("a10", "f22")
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
        if targets:
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
        """Remove output directories created by the pipeline tests.

        This is a superset that wipes every per-animal and experiment-level
        results/logs directory the full Snakemake DAG can produce, so it's
        safe to call from both the narrower ``test_war_generation_*`` /
        ``test_downstream_chain_*`` tests and from the full-pipeline test.
        """
        # Per-animal results directories (one per animal slug).
        per_animal_results = (
            "results/wars",
            "results/fdsars",
            "results/wars_quality_filtered",
            "results/wars_standardized",
            "results/wars_fragment_filtered",
            "results/wars_channel_filtered_manual",
            "results/wars_channel_filtered_lof",
            "results/wars_flattened",
            "results/wars_flattened_manual",
            "results/wars_flattened_lof",
            "results/fdsar_diagnostics",
            "results/diagnostic_figures_unfiltered",
            "results/diagnostic_figures_filtered",
        )
        for base in per_animal_results:
            for animal in self.ANIMALS:
                shutil.rmtree(_REPO_ROOT / base / animal, ignore_errors=True)

        # Experiment-level results directories (one per run, not per-animal).
        experiment_level_results = (
            "results/wars_zeitgeber",
            "results/zeitgeber_plots",
            "results/relfreq_plots",
            "results/ep_figures",
            "results/ep_heatmaps",
            "results/lof_evaluation",
            "results/filtering_comparison_plots",
        )
        for subdir in experiment_level_results:
            shutil.rmtree(_REPO_ROOT / subdir, ignore_errors=True)

        # Per-rule log directories.
        log_dirs = (
            "logs/war_generation",
            "logs/war_quality_filter",
            "logs/war_standardize",
            "logs/war_fragment_filtering",
            "logs/war_channel_filtering_manual",
            "logs/war_channel_filtering_lof",
            "logs/war_flattening",
            "logs/war_flattening_manual",
            "logs/war_flattening_lof",
            "logs/fdsar_diagnostics",
            "logs/diagnostic_figures_unfiltered",
            "logs/diagnostic_figures_filtered",
            "logs/war_zeitgeber",
            "logs/zeitgeber_plots",
            "logs/war_relfreq_plots",
            "logs/ep_figures",
            "logs/ep_heatmaps",
            "logs/lof_evaluation",
            "logs/filtering_comparison",
        )
        for subdir in log_dirs:
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

    def test_full_pipeline_all_target(self):
        """Run the default ``all`` target end-to-end on mini_real.

        Exercises every rule currently wired into ``workflow/Snakefile``:
        ``war_generation``, ``fdsar_diagnostics``, ``war_quality_filter``,
        ``war_standardize``, ``war_fragment_filtering``,
        ``war_channel_filtering_{manual,lof}``,
        ``diagnostic_figures_{unfiltered,filtered}``,
        ``war_flattening{,_manual,_lof}``, ``war_zeitgeber``,
        ``zeitgeber_plots``, ``war_relfreq_plots``, ``ep_figures``,
        ``ep_heatmaps``, ``lof_evaluation``, ``filtering_comparison``.

        Snakemake itself enforces that every output declared on ``rule all``
        was produced — a zero return code from snakemake IS the assertion
        that every rule ran successfully. Any future rename, input-wiring
        break, or script import regression that affects any downstream rule
        will fail here immediately.

        This is the authoritative full-DAG coverage test. It is expensive
        (full pipeline over two animals) so it is gated by
        ``@pytest.mark.slow`` and should be run before merging pipeline
        changes.
        """
        self._cleanup()
        try:
            result = self._run_snakemake(
                targets=[],
                extra_args=["--forcerun", "war_generation"],
                timeout=1800,  # 30 min headroom; real wall-clock should be < 5 min
            )
            assert result.returncode == 0, (
                f"Full-pipeline 'all' target failed for '{self.DATASET}' dataset.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

            # Pickle-regression guard: the parquet migration removed all
            # ``war.pkl`` writes, so no rule should ever produce one again.
            # Scope to the mini_real animals we just (re)generated so that
            # stale pickles from unrelated datasets in the same working tree
            # don't trip the assertion.
            war_bases = (
                "results/wars",
                "results/wars_quality_filtered",
                "results/wars_standardized",
                "results/wars_fragment_filtered",
                "results/wars_channel_filtered_manual",
                "results/wars_channel_filtered_lof",
                "results/wars_flattened",
                "results/wars_flattened_manual",
                "results/wars_flattened_lof",
            )
            pkl_files = [
                _REPO_ROOT / base / animal / "war.pkl"
                for base in war_bases
                for animal in self.ANIMALS
                if (_REPO_ROOT / base / animal / "war.pkl").exists()
            ]
            assert not pkl_files, (
                f"Legacy war.pkl files produced by the pipeline: {pkl_files}"
            )
        finally:
            self._cleanup()
