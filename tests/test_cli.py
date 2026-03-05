"""
Tests for the neurodent CLI.

Covers the ``init-pipeline`` sub-command which copies the bundled Snakemake
pipeline files to a target directory.
"""

import shutil
from pathlib import Path

import pytest

from neurodent.cli import build_parser, cmd_init_pipeline


class TestBuildParser:
    """Tests for the argument parser."""

    def test_parser_has_init_pipeline_command(self):
        parser = build_parser()
        # Parse init-pipeline with default args
        args = parser.parse_args(["init-pipeline"])
        assert args.command == "init-pipeline"

    def test_init_pipeline_default_target_is_dot(self):
        parser = build_parser()
        args = parser.parse_args(["init-pipeline"])
        assert args.target == "."

    def test_init_pipeline_custom_target(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(["init-pipeline", str(tmp_path)])
        assert args.target == str(tmp_path)

    def test_init_pipeline_overwrite_flag(self):
        parser = build_parser()
        args = parser.parse_args(["init-pipeline", "--overwrite"])
        assert args.overwrite is True

    def test_init_pipeline_overwrite_defaults_false(self):
        parser = build_parser()
        args = parser.parse_args(["init-pipeline"])
        assert args.overwrite is False

    def test_no_subcommand_has_no_func(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestInitPipeline:
    """Tests for cmd_init_pipeline."""

    @pytest.fixture()
    def target(self, tmp_path):
        return tmp_path / "pipeline"

    def _run(self, tmp_path, extra_args=None):
        """Helper: parse args and invoke cmd_init_pipeline."""
        parser = build_parser()
        argv = ["init-pipeline", str(tmp_path)]
        if extra_args:
            argv += extra_args
        args = parser.parse_args(argv)
        return cmd_init_pipeline(args)

    def test_creates_snakefile(self, tmp_path):
        rc = self._run(tmp_path)
        assert rc == 0
        assert (tmp_path / "Snakefile").exists()

    def test_creates_workflow_directory(self, tmp_path):
        rc = self._run(tmp_path)
        assert rc == 0
        assert (tmp_path / "workflow").is_dir()

    def test_creates_config_directory(self, tmp_path):
        rc = self._run(tmp_path)
        assert rc == 0
        assert (tmp_path / "config").is_dir()

    def test_config_yaml_present(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "config" / "config.yaml").exists()

    def test_workflow_scripts_present(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "workflow" / "scripts").is_dir()

    def test_workflow_rules_present(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "workflow" / "rules").is_dir()

    def test_skip_existing_without_overwrite(self, tmp_path):
        # First run populates the target
        self._run(tmp_path)
        snakefile = tmp_path / "Snakefile"
        snakefile.write_text("# modified")

        # Second run without --overwrite should leave the file unchanged
        rc = self._run(tmp_path)
        assert rc == 0
        assert snakefile.read_text() == "# modified"

    def test_overwrite_replaces_existing(self, tmp_path):
        # Populate target with dummy content
        self._run(tmp_path)
        snakefile = tmp_path / "Snakefile"
        snakefile.write_text("# modified")

        # Second run with --overwrite should replace the file
        rc = self._run(tmp_path, ["--overwrite"])
        assert rc == 0
        assert snakefile.read_text() != "# modified"

    def test_overwrite_replaces_existing_directory(self, tmp_path):
        # Populate target
        self._run(tmp_path)
        workflow_dir = tmp_path / "workflow"
        extra_file = workflow_dir / "EXTRA.txt"
        extra_file.write_text("extra")

        # With --overwrite, the directory is replaced (extra file gone)
        rc = self._run(tmp_path, ["--overwrite"])
        assert rc == 0
        assert not extra_file.exists()

    def test_returns_zero_exit_code_on_success(self, tmp_path):
        rc = self._run(tmp_path)
        assert rc == 0

    def test_custom_target_directory_created_content(self, tmp_path):
        """Files should land in the specified target, not CWD."""
        rc = self._run(tmp_path)
        assert rc == 0
        expected = {tmp_path / "Snakefile", tmp_path / "workflow", tmp_path / "config"}
        for p in expected:
            assert p.exists(), f"Expected {p} to exist"
