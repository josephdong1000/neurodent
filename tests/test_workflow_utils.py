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


@pytest.mark.mutates_constants
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


# ---------------------------------------------------------------------------
# expand_animals_config tests
# ---------------------------------------------------------------------------
from neurodent.workflow.utils import expand_animals_config


class TestExpandAnimalsConfig:
    """Tests for expand_animals_config function."""

    def test_no_animals_key_returns_copy(self):
        """Config without 'animals' key is returned as a deep copy."""
        original = {"data_root": "/data", "ANIMAL_METADATA": [{"id": "X"}]}
        result = expand_animals_config(original)
        assert result == original
        assert result is not original

    def test_data_root_is_canonical(self):
        """'data_root' remains as 'data_root' (canonical key)."""
        cfg = {"data_root": "/my/root", "animals": [{"id": "A", "gene": "WT", "sex": "M"}]}
        result = expand_animals_config(cfg)
        assert result["data_root"] == "/my/root"
        assert "data_parent_folder" not in result

    def test_data_parent_folder_migrated_to_data_root(self):
        """Legacy 'data_parent_folder' is migrated to 'data_root'."""
        cfg = {
            "data_parent_folder": "/legacy/path",
            "animals": [{"id": "A", "gene": "WT", "sex": "M"}],
        }
        result = expand_animals_config(cfg)
        assert result["data_root"] == "/legacy/path"
        assert "data_parent_folder" not in result

    def test_data_root_takes_precedence_over_legacy(self):
        """If both data_root and data_parent_folder exist, data_root wins."""
        cfg = {
            "data_root": "/new",
            "data_parent_folder": "/legacy",
            "animals": [{"id": "A", "gene": "WT", "sex": "M"}],
        }
        result = expand_animals_config(cfg)
        assert result["data_root"] == "/new"

    def test_builds_animal_metadata(self):
        """ANIMAL_METADATA list is built from the animals entries."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M"},
                {"id": "F22", "gene": "KO", "sex": "F"},
            ],
        }
        result = expand_animals_config(cfg)
        meta_ids = {e["id"] for e in result["ANIMAL_METADATA"]}
        assert meta_ids == {"A10", "F22"}

        a10 = next(e for e in result["ANIMAL_METADATA"] if e["id"] == "A10")
        assert a10["gene"] == "WT"
        assert a10["sex"] == "M"

    def test_metadata_excludes_override_keys(self):
        """Override keys (pattern, lro_kwargs, etc.) are NOT in ANIMAL_METADATA."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {
                    "id": "X1", "gene": "WT", "sex": "M",
                    "pattern": "custom/{index}.nwb",
                    "lro_kwargs": {"mode": "si"},
                    "manual_datetime": "2025-01-01 10:00:00",
                    "day_parse_kwargs": {"date_patterns": []},
                },
            ],
        }
        result = expand_animals_config(cfg)
        meta = result["ANIMAL_METADATA"][0]
        assert "pattern" not in meta
        assert "lro_kwargs" not in meta
        assert "manual_datetime" not in meta
        assert "day_parse_kwargs" not in meta
        assert meta["id"] == "X1"
        assert meta["gene"] == "WT"

    def test_no_data_folders_to_animal_ids_generated(self):
        """Unified format does NOT produce data_folders_to_animal_ids."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M"},
            ],
        }
        result = expand_animals_config(cfg)
        assert "data_folders_to_animal_ids" not in result

    def test_builds_manual_datetimes(self):
        """manual_datetimes is built from animals' manual_datetime field."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M", "manual_datetime": "2025-01-01 10:00:00"},
                {"id": "F22", "gene": "KO", "sex": "F"},
            ],
        }
        result = expand_animals_config(cfg)
        assert result["manual_datetimes"] == {"A10": "2025-01-01 10:00:00"}

    def test_auto_generates_genotype_aliases(self):
        """GENOTYPE_ALIASES is auto-generated from gene field."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M"},
                {"id": "B5", "gene": "WT", "sex": "M"},
                {"id": "F22", "gene": "KO", "sex": "F"},
            ],
        }
        result = expand_animals_config(cfg)
        ga = result["GENOTYPE_ALIASES"]
        assert set(ga["WT"]) == {"A10", "B5"}
        assert ga["KO"] == ["F22"]

    def test_explicit_genotype_aliases_preserved(self):
        """Explicit GENOTYPE_ALIASES in config is not overwritten."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M"},
            ],
            "GENOTYPE_ALIASES": {"MyCustom": ["A10"]},
        }
        result = expand_animals_config(cfg)
        assert result["GENOTYPE_ALIASES"] == {"MyCustom": ["A10"]}

    def test_builds_animal_overrides(self):
        """_animal_overrides dict is built from per-animal override fields."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {
                    "id": "X1", "gene": "WT", "sex": "M",
                    "pattern": "{data_root}/custom/{animal}_{index}.rhd",
                    "lro_kwargs": {"mode": "si"},
                    "day_parse_kwargs": {"date_patterns": [["\\d{6}", "%y%m%d"]]},
                },
                {"id": "A10", "gene": "WT", "sex": "M"},
            ],
        }
        result = expand_animals_config(cfg)
        assert "X1" in result["_animal_overrides"]
        assert result["_animal_overrides"]["X1"]["pattern"] == "{data_root}/custom/{animal}_{index}.rhd"
        assert result["_animal_overrides"]["X1"]["lro_kwargs"] == {"mode": "si"}
        assert "A10" not in result["_animal_overrides"]

    def test_no_animal_overrides_when_none_specified(self):
        """_animal_overrides is absent when no animals have overrides."""
        cfg = {
            "data_root": "/data",
            "animals": [{"id": "A10", "gene": "WT", "sex": "M"}],
        }
        result = expand_animals_config(cfg)
        assert "_animal_overrides" not in result

    def test_does_not_mutate_input(self):
        """Original config dict is not mutated."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M"},
            ],
        }
        original_animals = cfg["animals"][0].copy()
        expand_animals_config(cfg)
        assert cfg["animals"][0] == original_animals
        assert "data_root" in cfg

    def test_preserves_existing_manual_datetimes(self):
        """Existing manual_datetimes entries are preserved."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M", "manual_datetime": "2025-01-01"},
            ],
            "manual_datetimes": {"LegacyAnimal": "2020-01-01"},
        }
        result = expand_animals_config(cfg)
        assert result["manual_datetimes"]["LegacyAnimal"] == "2020-01-01"
        assert result["manual_datetimes"]["A10"] == "2025-01-01"

    def test_preserves_existing_animal_metadata(self):
        """Existing ANIMAL_METADATA entries are not duplicated."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M"},
            ],
            "ANIMAL_METADATA": [
                {"id": "A10", "gene": "KO", "sex": "F"},
            ],
        }
        result = expand_animals_config(cfg)
        # Existing entry is kept, not overwritten
        a10_entries = [e for e in result["ANIMAL_METADATA"] if e["id"] == "A10"]
        assert len(a10_entries) == 1
        assert a10_entries[0]["gene"] == "KO"

    def test_full_example_config(self):
        """Test with a realistic config matching the issue's example."""
        cfg = {
            "data_root": "/mnt/data/project",
            "animals": [
                {"id": "AM3", "gene": "WT", "sex": "Male"},
                {"id": "AM5", "gene": "Het", "sex": "Male"},
                {
                    "id": "AP3B2homo-240-M", "gene": "HOMO", "sex": "Male",
                    "pattern": "{data_root}/PortA-*PortB-*/{animal}*_ColMajor_{index}.rhd",
                    "manual_datetime": "2025-11-27 15:39:05",
                    "lro_kwargs": {"mode": "si"},
                },
            ],
            "LR_ALIASES": {"L": ["0"], "R": ["1"]},
        }
        result = expand_animals_config(cfg)

        # data_root stays as data_root
        assert result["data_root"] == "/mnt/data/project"

        # ANIMAL_METADATA built
        meta_ids = {e["id"] for e in result["ANIMAL_METADATA"]}
        assert meta_ids == {"AM3", "AM5", "AP3B2homo-240-M"}

        # No data_folders_to_animal_ids
        assert "data_folders_to_animal_ids" not in result

        # manual_datetimes built
        assert result["manual_datetimes"]["AP3B2homo-240-M"] == "2025-11-27 15:39:05"
        assert "AM3" not in result["manual_datetimes"]

        # GENOTYPE_ALIASES auto-generated
        ga = result["GENOTYPE_ALIASES"]
        assert "AM3" in ga["WT"]
        assert "AM5" in ga["Het"]
        assert "AP3B2homo-240-M" in ga["HOMO"]

        # _animal_overrides built
        ov = result["_animal_overrides"]
        assert "AP3B2homo-240-M" in ov
        assert ov["AP3B2homo-240-M"]["pattern"] == "{data_root}/PortA-*PortB-*/{animal}*_ColMajor_{index}.rhd"
        assert ov["AP3B2homo-240-M"]["lro_kwargs"] == {"mode": "si"}
        assert "AM3" not in ov

        # LR_ALIASES preserved
        assert result["LR_ALIASES"] == {"L": ["0"], "R": ["1"]}

    def test_legacy_data_parent_folder_without_animals(self):
        """Legacy config with data_parent_folder (no animals list) gets migrated."""
        cfg = {"data_parent_folder": "/legacy/path"}
        result = expand_animals_config(cfg)
        assert result["data_root"] == "/legacy/path"
        assert "data_parent_folder" not in result

    def test_bad_channels_list_format(self):
        """List-format bad_channels are stored under _all sentinel key."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M", "bad_channels": ["LHip", "RHip"]},
            ],
        }
        result = expand_animals_config(cfg)
        assert result["bad_channels"]["A10"] == {"_all": ["LHip", "RHip"]}

    def test_bad_channels_dict_format(self):
        """Dict-format bad_channels (per-session) are stored as-is."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {
                    "id": "A10", "gene": "WT", "sex": "M",
                    "bad_channels": {
                        "A10 WT Jan-08-2022": ["LHip"],
                        "A10 WT Jan-09-2022": ["RMot"],
                    },
                },
            ],
        }
        result = expand_animals_config(cfg)
        assert result["bad_channels"]["A10"] == {
            "A10 WT Jan-08-2022": ["LHip"],
            "A10 WT Jan-09-2022": ["RMot"],
        }

    def test_bad_channels_not_in_metadata(self):
        """bad_channels is excluded from ANIMAL_METADATA entries."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M", "bad_channels": ["LHip"]},
            ],
        }
        result = expand_animals_config(cfg)
        meta = result["ANIMAL_METADATA"][0]
        assert "bad_channels" not in meta

    def test_bad_channels_preserves_existing(self):
        """Existing top-level bad_channels entries are preserved."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M", "bad_channels": ["LHip"]},
            ],
            "bad_channels": {"legacy_key X1": {"Session1": ["RAud"]}},
        }
        result = expand_animals_config(cfg)
        assert result["bad_channels"]["legacy_key X1"] == {"Session1": ["RAud"]}
        assert result["bad_channels"]["A10"] == {"_all": ["LHip"]}

    def test_bad_channels_empty_when_none_specified(self):
        """bad_channels dict is empty when no animals have bad_channels."""
        cfg = {
            "data_root": "/data",
            "animals": [{"id": "A10", "gene": "WT", "sex": "M"}],
        }
        result = expand_animals_config(cfg)
        assert result["bad_channels"] == {}

    def test_bad_channels_does_not_mutate_input(self):
        """Original config dict is not mutated by bad_channels expansion."""
        cfg = {
            "data_root": "/data",
            "animals": [
                {"id": "A10", "gene": "WT", "sex": "M", "bad_channels": ["LHip"]},
            ],
        }
        original_bc = cfg["animals"][0]["bad_channels"].copy()
        expand_animals_config(cfg)
        assert cfg["animals"][0]["bad_channels"] == original_bc
        assert "bad_channels" not in cfg  # top-level not added to original
