"""Tests for Snakemake workflow utility functions.

Tests the deep_merge_dict function used in Snakefile for dataset configuration merging, and
analysis_config, the copy of the config that every rule passes to its job as a param.
"""

import copy
from pathlib import Path

import pytest
from neurodent.workflow.utils import analysis_config, deep_merge_dict
from neurodent.workflow.utils.config import EXECUTION_ONLY_KEYS, load_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = sorted(p.stem for p in (REPO_ROOT / "config" / "datasets").glob("*.yaml"))


class TestDeepMergeDict:
    """Test suite for deep_merge_dict function."""

    def test_basic_merge(self):
        """Test basic dictionary merge without nesting."""
        base = {"a": 1, "b": 2}
        override = {"c": 3}
        result = deep_merge_dict(base, override)

        assert result == {"a": 1, "b": 2, "c": 3}
        # Ensure original dicts not modified
        assert base == {"a": 1, "b": 2}
        assert override == {"c": 3}

    def test_override_value(self):
        """Test that override values replace base values."""
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = deep_merge_dict(base, override)

        assert result == {"a": 1, "b": 99}

    def test_nested_dict_merge(self):
        """Test merging nested dictionaries."""
        base = {"level1": {"a": 1, "b": 2}}
        override = {"level1": {"b": 99, "c": 3}}
        result = deep_merge_dict(base, override)

        expected = {"level1": {"a": 1, "b": 99, "c": 3}}
        assert result == expected

    def test_deep_nested_merge(self):
        """Test merging deeply nested dictionaries (3+ levels)."""
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}, "x": 10}}}
        override = {"l1": {"l2": {"l3": {"b": 99, "c": 3}, "y": 20}}}
        result = deep_merge_dict(base, override)

        expected = {"l1": {"l2": {"l3": {"a": 1, "b": 99, "c": 3}, "x": 10, "y": 20}}}
        assert result == expected

    def test_empty_base(self):
        """Test merging into empty base dictionary."""
        base = {}
        override = {"a": 1, "b": 2}
        result = deep_merge_dict(base, override)

        assert result == {"a": 1, "b": 2}

    def test_empty_override(self):
        """Test merging empty override dictionary."""
        base = {"a": 1, "b": 2}
        override = {}
        result = deep_merge_dict(base, override)

        assert result == {"a": 1, "b": 2}

    def test_both_empty(self):
        """Test merging two empty dictionaries."""
        result = deep_merge_dict({}, {})
        assert result == {}

    def test_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge_dict(base, override)

        # Lists should be replaced, not concatenated
        assert result == {"items": [4, 5]}

    def test_none_values(self):
        """Test handling of None values."""
        base = {"a": 1, "b": None}
        override = {"b": 2, "c": None}
        result = deep_merge_dict(base, override)

        assert result == {"a": 1, "b": 2, "c": None}

    def test_mixed_types_override(self):
        """Test overriding dict with non-dict value."""
        base = {"config": {"nested": {"value": 1}}}
        override = {"config": "simple_string"}
        result = deep_merge_dict(base, override)

        # Override should replace entire nested dict with string
        assert result == {"config": "simple_string"}

    def test_non_dict_to_dict(self):
        """Test overriding non-dict with dict."""
        base = {"config": "simple_string"}
        override = {"config": {"nested": {"value": 1}}}
        result = deep_merge_dict(base, override)

        assert result == {"config": {"nested": {"value": 1}}}

    def test_real_world_config_merge(self):
        """Test realistic configuration merge scenario (similar to Snakefile usage)."""
        # Main config
        base = {
            "temp_directory": "/tmp",
            "samples": {
                "quality_filter": {
                    "exclude_unknown_genotypes": True,
                    "exclude_bad_animaldays": True,
                }
            },
            "analysis": {
                "war_generation": {
                    "skip_sessions": ["*bad*"],
                    "lro_kwargs": {
                        "multiprocess_mode": "dask",
                    },
                }
            },
        }

        # Dataset config (ap3b2_rhd)
        override = {
            "samples": {"samples_file": "config/samples_jess_rhd.json"},
            "analysis": {
                "war_generation": {
                    "pattern": "{index}.rhd",
                    "lro_kwargs": {
                        "extract_func": "read_intan",
                        "mode": "si",
                        "stream_id": "0",
                    },
                }
            },
        }

        result = deep_merge_dict(base, override)

        # Verify key merges
        assert result["temp_directory"] == "/tmp"  # preserved from base
        assert (
            result["samples"]["samples_file"] == "config/samples_jess_rhd.json"
        )  # from override
        assert (
            result["samples"]["quality_filter"]["exclude_unknown_genotypes"] is True
        )  # preserved
        assert result["analysis"]["war_generation"]["pattern"] == "{index}.rhd"  # from override
        assert result["analysis"]["war_generation"]["skip_sessions"] == ["*bad*"]  # preserved
        assert (
            result["analysis"]["war_generation"]["lro_kwargs"]["multiprocess_mode"]
            == "dask"
        )  # preserved
        assert (
            result["analysis"]["war_generation"]["lro_kwargs"]["extract_func"]
            == "read_intan"
        )  # from override
        assert (
            result["analysis"]["war_generation"]["lro_kwargs"]["mode"] == "si"
        )  # from override

    def test_no_mutation_of_inputs(self):
        """Test that input dictionaries are not mutated."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}

        base_copy = {"a": {"b": 1}}
        override_copy = {"a": {"c": 2}}

        result = deep_merge_dict(base, override)

        # Ensure inputs unchanged
        assert base == base_copy
        assert override == override_copy
        # But result is different
        assert result == {"a": {"b": 1, "c": 2}}


class TestTruncateAnimalsSlicing:
    """Test the truncate_animals config reading and slicing logic used in Snakefile."""

    def _apply_truncate_animals(self, config, animals):
        """Replicate the Snakefile slicing logic."""
        truncate_animals = config.get("samples", {}).get("truncate_animals", None)
        if truncate_animals is not None:
            return animals[:truncate_animals]
        return animals

    def test_truncate_animals_limits_list(self):
        """When truncate_animals=N, only the first N animals are kept."""
        config = {"samples": {"truncate_animals": 2}}
        animals = ["a", "b", "c", "d", "e"]
        result = self._apply_truncate_animals(config, animals)
        assert result == ["a", "b"]

    def test_truncate_animals_null_keeps_all(self):
        """When truncate_animals=null (None), all animals are kept."""
        config = {"samples": {"truncate_animals": None}}
        animals = ["a", "b", "c"]
        result = self._apply_truncate_animals(config, animals)
        assert result == ["a", "b", "c"]

    def test_truncate_animals_missing_keeps_all(self):
        """When truncate_animals key is absent, all animals are kept."""
        config = {"samples": {}}
        animals = ["a", "b", "c"]
        result = self._apply_truncate_animals(config, animals)
        assert result == ["a", "b", "c"]

    def test_truncate_animals_samples_missing_keeps_all(self):
        """When samples section is absent, all animals are kept."""
        config = {}
        animals = ["a", "b", "c"]
        result = self._apply_truncate_animals(config, animals)
        assert result == ["a", "b", "c"]

    def test_truncate_animals_larger_than_list(self):
        """When truncate_animals > len(animals), all animals are kept."""
        config = {"samples": {"truncate_animals": 10}}
        animals = ["a", "b"]
        result = self._apply_truncate_animals(config, animals)
        assert result == ["a", "b"]

    def test_truncate_animals_one(self):
        """When truncate_animals=1, only the first animal is kept."""
        config = {"samples": {"truncate_animals": 1}}
        animals = ["x", "y", "z"]
        result = self._apply_truncate_animals(config, animals)
        assert result == ["x"]

    def test_truncate_animals_merges_via_deep_merge(self):
        """truncate_animals set in an override config is correctly merged."""
        base_config = {"samples": {"quality_filter": {"exclude_unknown_genotypes": True}}}
        override_config = {"samples": {"truncate_animals": 3}}
        merged = deep_merge_dict(base_config, override_config)

        animals = ["a", "b", "c", "d", "e"]
        result = self._apply_truncate_animals(merged, animals)
        assert result == ["a", "b", "c"]
        # Ensure existing samples keys are preserved
        assert merged["samples"]["quality_filter"]["exclude_unknown_genotypes"] is True


class TestAnalysisConfig:
    """analysis_config leaves out what decides how and where a job runs.

    Snakemake compares each job's params by their repr, so a key in the analysis config that
    only sizes or places the job would rerun every job whenever it changed.
    """

    @staticmethod
    def _config():
        return {
            "active_dataset": "sox5_bin",
            "temp_directory": "/scratch/user",
            "samples": {"truncate_animals": None},
            "analysis": {"zeitgeber": {"features": ["logrms"]}},
            "cluster": {"war_generation": {"time": "12h", "mem_mb": 80_000}},
            "samples_data": {"animals": [{"id": "A1"}]},
        }

    def test_leaves_out_the_execution_only_keys(self):
        result = analysis_config(self._config())
        assert set(EXECUTION_ONLY_KEYS) == {"cluster", "temp_directory", "active_dataset"}
        assert not set(EXECUTION_ONLY_KEYS) & set(result)

    def test_keeps_every_other_key_in_its_order(self):
        config = self._config()
        result = analysis_config(config)
        assert list(result) == [key for key in config if key not in EXECUTION_ONLY_KEYS]
        assert result == {key: config[key] for key in result}

    def test_is_a_copy(self):
        config = self._config()
        before = copy.deepcopy(config)
        result = analysis_config(config)
        result["analysis"]["zeitgeber"]["features"].append("lognspike")
        assert config == before

    def test_a_config_without_execution_keys_passes_through(self):
        config = {"analysis": {"a": 1}}
        assert analysis_config(config) == config

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_editing_job_sizes_does_not_change_a_jobs_params(self, dataset, tmp_path):
        config = load_dataset_config(
            dataset,
            config_path=REPO_ROOT / "config" / "config.yaml",
            local_path=tmp_path / "no_local_override.yaml",
            datasets_dir=REPO_ROOT / "config" / "datasets",
        )
        edited = copy.deepcopy(config)
        for block in edited["cluster"].values():
            block["mem_mb"] = 2 * block["mem_mb"]
            block["time"] = "48h"
        edited["temp_directory"] = "/somewhere/else"
        edited["active_dataset"] = "another_dataset"

        assert repr(analysis_config(edited)) == repr(analysis_config(config))

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_editing_an_analysis_setting_still_changes_them(self, dataset, tmp_path):
        config = load_dataset_config(
            dataset,
            config_path=REPO_ROOT / "config" / "config.yaml",
            local_path=tmp_path / "no_local_override.yaml",
            datasets_dir=REPO_ROOT / "config" / "datasets",
        )
        edited = copy.deepcopy(config)
        edited["analysis"]["fragment_filter_config"]["high_rms"]["max_rms"] += 1

        assert repr(analysis_config(edited)) != repr(analysis_config(config))
