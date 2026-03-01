"""Tests for neurodent.workflow utilities."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neurodent.workflow import setup_snakemake_logging, load_wars
from neurodent.workflow.utils import apply_path_overrides


class TestSetupSnakemakeLogging:
    """Tests for setup_snakemake_logging function."""

    def test_creates_log_file(self, tmp_path):
        """Test that logging creates the specified log file."""
        log_file = tmp_path / "test.log"

        # Mock snakemake object
        mock_snakemake = MagicMock()
        mock_snakemake.log = [str(log_file)]

        logger = setup_snakemake_logging(mock_snakemake)

        # Log something
        logger.info("Test message")

        # Flush to ensure write
        logging.shutdown()

        # Verify log file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_returns_logger(self, tmp_path):
        """Test that function returns a logger instance."""
        log_file = tmp_path / "test.log"
        mock_snakemake = MagicMock()
        mock_snakemake.log = [str(log_file)]

        result = setup_snakemake_logging(mock_snakemake)

        assert isinstance(result, logging.Logger)


class TestLoadWars:
    """Tests for load_wars function."""

    def test_load_wars_with_explicit_json_paths(self, tmp_path):
        """Test loading WARs with explicit json paths."""
        # Create mock WAR
        mock_war = MagicMock()
        mock_war.animal_id = "test_animal"

        with patch("neurodent.visualization.WindowAnalysisResult") as mock_war_class:
            mock_war_class.load_pickle_and_json.return_value = mock_war

            pkl_paths = [tmp_path / "war1.pkl", tmp_path / "war2.pkl"]
            json_paths = [tmp_path / "war1.json", tmp_path / "war2.json"]

            result = load_wars(pkl_paths, json_paths)

            assert len(result) == 2
            assert mock_war_class.load_pickle_and_json.call_count == 2

    def test_load_wars_auto_json_paths(self, tmp_path):
        """Test loading WARs with auto-detected json paths."""
        mock_war = MagicMock()

        with patch("neurodent.visualization.WindowAnalysisResult") as mock_war_class:
            mock_war_class.load_pickle_and_json.return_value = mock_war

            pkl_paths = [tmp_path / "animal1" / "war.pkl"]

            result = load_wars(pkl_paths)

            assert len(result) == 1
            # Verify it auto-derived the json path
            call_args = mock_war_class.load_pickle_and_json.call_args
            assert call_args.kwargs["json_name"] == "war.json"

    def test_load_wars_mismatched_lengths(self, tmp_path):
        """Test that mismatched path lengths raise ValueError."""
        pkl_paths = [tmp_path / "war1.pkl", tmp_path / "war2.pkl"]
        json_paths = [tmp_path / "war1.json"]  # Only one json path

        with pytest.raises(ValueError, match="must have the same length"):
            load_wars(pkl_paths, json_paths)

    def test_load_wars_empty_list(self):
        """Test that empty list raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No WARs were successfully loaded"):
            load_wars([])


class TestApplyPathOverrides:
    """Tests for apply_path_overrides function."""

    def test_simple_override(self):
        """Test simple single-level override."""
        config = {"key": "original"}
        overrides = {"key": "overridden"}
        result = apply_path_overrides(config, overrides)
        assert result["key"] == "overridden"

    def test_nested_override(self):
        """Test multi-level nested override."""
        config = {"level1": {"level2": {"level3": "original"}}}
        overrides = {"level1.level2.level3": "overridden"}
        result = apply_path_overrides(config, overrides)
        assert result["level1"]["level2"]["level3"] == "overridden"

    def test_create_new_keys(self):
        """Test creating new keys that don't exist."""
        config = {"existing": "value"}
        overrides = {"new.nested.key": "new_value"}
        result = apply_path_overrides(config, overrides)
        assert result["new"]["nested"]["key"] == "new_value"
        assert result["existing"] == "value"  # Original preserved

    def test_multiple_overrides(self):
        """Test applying multiple overrides."""
        config = {"a": 1, "b": {"c": 2}}
        overrides = {
            "a": 10,
            "b.c": 20,
            "b.d": 30,
            "e.f": 40
        }
        result = apply_path_overrides(config, overrides)
        assert result == {
            "a": 10,
            "b": {"c": 20, "d": 30},
            "e": {"f": 40}
        }

    def test_does_not_mutate_input(self):
        """Test that original config is not mutated."""
        config = {"key": {"nested": "original"}}
        overrides = {"key.nested": "modified"}
        result = apply_path_overrides(config, overrides)
        assert config["key"]["nested"] == "original"  # Unchanged
        assert result["key"]["nested"] == "modified"

    def test_empty_overrides(self):
        """Test that empty overrides returns deep copy."""
        config = {"key": {"nested": "value"}}
        result = apply_path_overrides(config, {})
        assert result == config
        assert result is not config  # Different object

    def test_override_with_dict(self):
        """Test overriding with dict value."""
        config = {"key": "scalar"}
        overrides = {"key": {"nested": "dict"}}
        result = apply_path_overrides(config, overrides)
        assert result["key"] == {"nested": "dict"}

    def test_override_with_list(self):
        """Test overriding with list value."""
        config = {"key": [1, 2, 3]}
        overrides = {"key": [4, 5, 6]}
        result = apply_path_overrides(config, overrides)
        assert result["key"] == [4, 5, 6]

    def test_error_on_empty_path(self):
        """Test that empty path raises ValueError."""
        config = {"key": "value"}
        overrides = {"": "invalid"}
        with pytest.raises(ValueError, match="Override path cannot be empty"):
            apply_path_overrides(config, overrides)

    def test_error_on_non_dict_intermediate(self):
        """Test error when intermediate value is not a dict."""
        config = {"key": "scalar_value"}
        overrides = {"key.nested": "will_fail"}
        with pytest.raises(KeyError, match="intermediate key 'key' is str, not dict"):
            apply_path_overrides(config, overrides)

    def test_real_world_example(self):
        """Test with realistic neurodent config structure."""
        config = {
            "analysis": {
                "war_generation": {
                    "mode": "base",
                    "lro_kwargs": {
                        "mode": "si",
                        "input_type": "files"
                    }
                }
            }
        }
        overrides = {
            "analysis.war_generation.file_pattern": "*.EDF",
            "analysis.war_generation.lro_kwargs.extract_func": "read_edf"
        }
        result = apply_path_overrides(config, overrides)

        assert result["analysis"]["war_generation"]["mode"] == "base"
        assert result["analysis"]["war_generation"]["file_pattern"] == "*.EDF"
        assert result["analysis"]["war_generation"]["lro_kwargs"]["mode"] == "si"
        assert result["analysis"]["war_generation"]["lro_kwargs"]["extract_func"] == "read_edf"


# ---------------------------------------------------------------------------
# inject_config_aliases edge cases
# ---------------------------------------------------------------------------
from neurodent.workflow.utils import inject_config_aliases


class TestInjectConfigAliases:
    """Tests for inject_config_aliases covering alias injection."""

    def test_genotype_aliases_set(self):
        from neurodent import constants

        orig = getattr(constants, "GENOTYPE_ALIASES", None)
        try:
            inject_config_aliases({"GENOTYPE_ALIASES": {"wt": "WT"}})
            assert constants.GENOTYPE_ALIASES == {"wt": "WT"}
        finally:
            if orig is not None:
                constants.GENOTYPE_ALIASES = orig

    def test_chname_aliases_set(self):
        from neurodent import constants

        orig = getattr(constants, "CHNAME_ALIASES", None)
        try:
            inject_config_aliases({"CHNAME_ALIASES": {"motor": "mot"}})
            assert constants.CHNAME_ALIASES == {"motor": "mot"}
        finally:
            if orig is not None:
                constants.CHNAME_ALIASES = orig

    def test_lr_aliases_set(self):
        from neurodent import constants

        orig = getattr(constants, "LR_ALIASES", None)
        try:
            inject_config_aliases({"LR_ALIASES": {"Left": "L"}})
            assert constants.LR_ALIASES == {"Left": "L"}
        finally:
            if orig is not None:
                constants.LR_ALIASES = orig

    def test_empty_config_no_error(self):
        inject_config_aliases({})
