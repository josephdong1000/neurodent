"""Tests for neurodent.workflow utilities."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neurodent.workflow import setup_snakemake_logging, load_wars


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
