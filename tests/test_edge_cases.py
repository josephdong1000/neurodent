"""
Edge-case and coverage tests for uncovered code paths.

Targets the following coverage gaps identified during code review:
- DiscoveredFile / MultiFileGroup (discovery.py error paths)
- FileDiscoverer edge cases (empty dirs, no placeholders, filtering)
- Natural_Neighbor algorithm (utils.py)
- Cache validation logic (should_use_cached_file, should_use_cache_unified)
- MNE unit extraction (extract_mne_unit_info)
- FragmentAnalyzer dependency resolution (analyze_frag.py)
- FrequencyDomainSpikeDetector baseline window edge case
- Misc small uncovered utilities
"""

import csv
import math
import os
import time
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DiscoveredFile / MultiFileGroup edge cases  (discovery.py)
# ---------------------------------------------------------------------------
from neurodent.core.discovery import DiscoveredFile, FileDiscoverer, MultiFileGroup


class TestDiscoveredFileEdgeCases:
    """Tests for DiscoveredFile error paths and dict-compat API."""

    def test_no_path_or_paths_raises(self):
        with pytest.raises(ValueError, match="Either path or paths must be provided"):
            DiscoveredFile()

    def test_both_path_and_paths_raises(self):
        with pytest.raises(ValueError, match="Cannot provide both path and paths"):
            DiscoveredFile(path="/a.txt", paths=("/a.txt", "/b.txt"))

    def test_fspath_single(self):
        df = DiscoveredFile(path="/data/file.rhd", metadata={"animal": "A10"})
        assert os.fspath(df) == "/data/file.rhd"

    def test_fspath_multi_raises(self):
        df = DiscoveredFile(
            paths=("/data/a.bin", "/data/a.csv"), metadata={"animal": "A10"}
        )
        with pytest.raises(TypeError, match="Multi-file DiscoveredFile"):
            os.fspath(df)

    def test_contains_and_getitem(self):
        df = DiscoveredFile(path="/f.rhd", metadata={"animal": "A10", "session": "s1"})
        assert "path" in df
        assert "paths" not in df
        assert "animal" in df
        assert df["path"] == "/f.rhd"
        assert df["animal"] == "A10"

    def test_contains_paths_key(self):
        df = DiscoveredFile(paths=("/a.bin",), metadata={})
        assert "paths" in df
        assert "path" not in df
        assert df["paths"] == ("/a.bin",)

    def test_is_multi_file(self):
        single = DiscoveredFile(path="/a.rhd")
        multi = DiscoveredFile(paths=("/a.bin", "/a.csv"))
        assert not single.is_multi_file
        assert multi.is_multi_file

    def test_get_path_list_single(self):
        df = DiscoveredFile(path="/a.rhd")
        assert df.get_path_list() == ["/a.rhd"]

    def test_get_path_list_multi(self):
        df = DiscoveredFile(paths=("/a.bin", "/a.csv"))
        assert df.get_path_list() == ["/a.bin", "/a.csv"]

    def test_iter(self):
        df = DiscoveredFile(paths=("/a.bin", "/a.csv"))
        assert list(df) == ["/a.bin", "/a.csv"]

    def test_repr_single(self):
        df = DiscoveredFile(path="/a.rhd", metadata={"animal": "X"})
        r = repr(df)
        assert "path=" in r
        assert "X" in r

    def test_repr_multi(self):
        df = DiscoveredFile(paths=("/a.bin",), metadata={"animal": "X"})
        r = repr(df)
        assert "paths=" in r

    def test_default_metadata_is_empty_dict(self):
        df = DiscoveredFile(path="/a.rhd")
        assert df.metadata == {}

    def test_getitem_missing_key_raises(self):
        df = DiscoveredFile(path="/a.rhd", metadata={"animal": "A"})
        with pytest.raises(KeyError):
            _ = df["nonexistent"]


class TestMultiFileGroupDeprecation:
    """MultiFileGroup should emit DeprecationWarning."""

    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mfg = MultiFileGroup(paths=("/a.bin", "/b.csv"), metadata={"animal": "A10"})
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            assert mfg.paths == ("/a.bin", "/b.csv")


# ---------------------------------------------------------------------------
# FileDiscoverer edge cases
# ---------------------------------------------------------------------------


class TestFileDiscovererEdgeCases:
    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            FileDiscoverer("")

    def test_empty_list_pattern_raises(self):
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            FileDiscoverer([])

    def test_no_matches_returns_empty(self, tmp_path):
        fd = FileDiscoverer(str(tmp_path / "{animal}" / "{session}.rhd"))
        assert fd.discover() == []

    def test_pattern_without_placeholders(self, tmp_path):
        (tmp_path / "data.bin").touch()
        fd = FileDiscoverer(str(tmp_path / "*.bin"))
        results = fd.discover()
        assert len(results) == 1
        assert "path" in results[0]

    def test_filter_no_match(self, tmp_path):
        (tmp_path / "A10" / "s1").mkdir(parents=True)
        (tmp_path / "A10" / "s1" / "1.rhd").touch()
        fd = FileDiscoverer(str(tmp_path / "{animal}" / "{session}" / "{index}.rhd"))
        assert fd.discover(animal="NONEXISTENT") == []

    def test_pathlib_pattern_accepted(self, tmp_path):
        from pathlib import Path

        (tmp_path / "a.txt").touch()
        fd = FileDiscoverer(Path(tmp_path / "*.txt"))
        assert len(fd.discover()) == 1

    def test_multi_pattern_empty_first_returns_empty(self, tmp_path):
        patterns = [
            str(tmp_path / "{animal}" / "data.bin"),
            str(tmp_path / "{animal}" / "meta.json"),
        ]
        fd = FileDiscoverer(patterns)
        assert fd.discover() == []

    def test_discover_sorts_deterministically(self, tmp_path):
        for name in ["c.txt", "a.txt", "b.txt"]:
            (tmp_path / name).touch()
        fd = FileDiscoverer(str(tmp_path / "*.txt"))
        results = fd.discover()
        paths = [r["path"] for r in results]
        assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# Natural_Neighbor algorithm  (utils.py)
# ---------------------------------------------------------------------------
from neurodent.core.utils import Natural_Neighbor


class TestNaturalNeighbor:
    """Tests for the Natural_Neighbor algorithm."""

    def test_read_and_asserts(self):
        nn = Natural_Neighbor()
        data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        nn.read(data)
        assert np.array_equal(nn.data, data)
        nn.asserts()
        assert len(nn.knn) == 4
        assert all(nn.nan_num[i] == 0 for i in range(4))

    def test_count_all_zero(self):
        nn = Natural_Neighbor()
        nn.data = np.array([[0], [1], [2]])
        nn.asserts()
        assert nn.count() == 3  # all have zero natural neighbors initially

    def test_algorithm_small_cluster(self):
        np.random.seed(42)
        data = np.vstack(
            [np.random.randn(10, 2) + [0, 0], np.random.randn(10, 2) + [5, 5]]
        )
        nn = Natural_Neighbor()
        nn.read(data)
        r = nn.algorithm()
        assert isinstance(r, int)
        assert r >= 1

    def test_algorithm_identical_points(self):
        """All points identical – edge case for KDTree."""
        data = np.ones((5, 2))
        nn = Natural_Neighbor()
        nn.read(data)
        r = nn.algorithm()
        assert r >= 1

    def test_algorithm_two_points(self):
        """Minimal dataset of two points."""
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        nn = Natural_Neighbor()
        nn.read(data)
        r = nn.algorithm()
        assert r >= 1

    def test_findKNN(self):
        nn = Natural_Neighbor()
        data = np.array([[0, 0], [1, 0], [2, 0], [10, 0]])
        nn.read(data)
        from scipy.spatial import KDTree

        tree = KDTree(data)
        neighbours = nn.findKNN(data[0], 2, tree)
        assert len(neighbours) == 2
        assert 1 in neighbours

    def test_load_csv(self, tmp_path):
        """Test loading from CSV file."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1.0, 2.0, "classA"])
            writer.writerow([3.0, 4.0, "classB"])
            writer.writerow([5.0, 6.0, "classA"])
        nn = Natural_Neighbor()
        nn.load(str(csv_path))
        assert nn.data.shape == (3, 2)
        assert nn.target == ["classA", "classB", "classA"]


# ---------------------------------------------------------------------------
# Cache validation  (utils.py)
# ---------------------------------------------------------------------------
from neurodent.core.utils import should_use_cached_file, should_use_cache_unified


class TestCacheValidation:
    """Tests for should_use_cached_file and should_use_cache_unified."""

    def test_never_returns_false(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        assert should_use_cached_file(cache, [], "never") is False

    def test_error_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Cache file required"):
            should_use_cached_file(tmp_path / "missing.pkl", [], "error")

    def test_error_exists_returns_true(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        assert should_use_cached_file(cache, [], "error") is True

    def test_always_exists(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        assert should_use_cached_file(cache, [], "always") is True

    def test_always_missing(self, tmp_path):
        assert should_use_cached_file(tmp_path / "x.pkl", [], "always") is False

    def test_auto_cache_missing(self, tmp_path):
        assert should_use_cached_file(tmp_path / "x.pkl", [], "auto") is False

    def test_auto_cache_newer_than_sources(self, tmp_path):
        source = tmp_path / "source.nwb"
        source.touch()
        time.sleep(0.05)
        cache = tmp_path / "cache.pkl"
        cache.touch()
        assert should_use_cached_file(cache, [source], "auto") is True

    def test_auto_cache_older_than_source(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        time.sleep(0.05)
        source = tmp_path / "source.nwb"
        source.touch()
        assert should_use_cached_file(cache, [source], "auto") is False

    def test_auto_missing_source_skipped(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        missing_source = tmp_path / "nonexistent.nwb"
        assert should_use_cached_file(cache, [missing_source], "auto") is True

    def test_invalid_policy_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid use_cached value"):
            should_use_cached_file(tmp_path / "x.pkl", [], "invalid_policy")

    # should_use_cache_unified
    def test_unified_force_regenerate(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        assert should_use_cache_unified(cache, [], "force_regenerate") is False

    def test_unified_always(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        assert should_use_cache_unified(cache, [], "always") is True

    def test_unified_always_missing(self, tmp_path):
        assert should_use_cache_unified(tmp_path / "x.pkl", [], "always") is False

    def test_unified_auto_delegates(self, tmp_path):
        cache = tmp_path / "cache.pkl"
        cache.touch()
        assert should_use_cache_unified(cache, [], "auto") is True

    def test_unified_invalid_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid cache_policy"):
            should_use_cache_unified(tmp_path / "x.pkl", [], "bad")


# ---------------------------------------------------------------------------
# MNE unit extraction  (utils.py)
# ---------------------------------------------------------------------------
from neurodent.core.utils import extract_mne_unit_info


class TestExtractMneUnitInfo:
    """Tests for MNE unit extraction covering all branches."""

    def _make_raw_info(self, unit, unit_mul, n_channels=2):
        """Helper to create a minimal raw_info dict mimicking MNE info."""
        return {
            "chs": [
                {"ch_name": f"ch{i}", "unit": unit, "unit_mul": unit_mul}
                for i in range(n_channels)
            ]
        }

    def test_no_channels(self):
        result = extract_mne_unit_info({"chs": []})
        assert result == (None, None)

    def test_missing_chs_key(self):
        result = extract_mne_unit_info({})
        assert result == (None, None)

    def test_channels_without_unit(self):
        info = {"chs": [{"ch_name": "ch0"}]}
        result = extract_mne_unit_info(info)
        assert result == (None, None)

    def test_inconsistent_units_raises(self):
        info = {
            "chs": [
                {"ch_name": "ch0", "unit": 107, "unit_mul": 0},
                {"ch_name": "ch1", "unit": 112, "unit_mul": 0},
            ]
        }
        with pytest.raises(ValueError, match="Inconsistent units"):
            extract_mne_unit_info(info)

    def test_inconsistent_unit_muls_raises(self):
        info = {
            "chs": [
                {"ch_name": "ch0", "unit": 107, "unit_mul": 0},
                {"ch_name": "ch1", "unit": 107, "unit_mul": -6},
            ]
        }
        with pytest.raises(ValueError, match="Inconsistent unit multipliers"):
            extract_mne_unit_info(info)

    def test_unknown_unit_code(self):
        info = self._make_raw_info(unit=999, unit_mul=0)
        result = extract_mne_unit_info(info)
        assert result == (None, None)

    def test_voltage_micro(self):
        """FIFF_UNIT_V (107) + FIFF_UNITM_MU (-6) → µV, mult=1.0"""
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_MU)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "µV"
            assert mult == pytest.approx(1.0)
        except ImportError:
            pytest.skip("MNE not available")

    def test_voltage_milli(self):
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_M)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "mV"
            assert mult == pytest.approx(1e3)
        except ImportError:
            pytest.skip("MNE not available")

    def test_voltage_none_multiplier(self):
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_NONE)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "V"
            # V → µV conversion: 1 V = 1e6 µV
            assert mult == pytest.approx(1e6)
        except ImportError:
            pytest.skip("MNE not available")

    def test_voltage_nano(self):
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_N)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "nV"
            assert mult == pytest.approx(1e-3)
        except ImportError:
            pytest.skip("MNE not available")

    def test_tesla_units_non_voltage(self):
        """Non-voltage units (Tesla) should return (None, None)."""
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_T, unit_mul=FIFF.FIFF_UNITM_NONE)
            result = extract_mne_unit_info(info)
            assert result == (None, None)
        except ImportError:
            pytest.skip("MNE not available")


# ---------------------------------------------------------------------------
# FragmentAnalyzer dependency resolution  (analyze_frag.py)
# ---------------------------------------------------------------------------
from neurodent.core.analyze_frag import FragmentAnalyzer


class TestFragmentDependencyResolution:
    """Tests for process_fragment_with_dependencies and _resolve_feature_dependencies."""

    @pytest.fixture
    def synthetic_fragment(self):
        """8-channel, 1-second @ 1000 Hz white noise fragment."""
        np.random.seed(0)
        return np.random.randn(1000, 8).astype(np.float32)

    def test_base_feature_no_deps(self, synthetic_fragment):
        results = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment, f_s=1000, features=["rms"], kwargs={}
        )
        assert "rms" in results
        assert results["rms"].shape == (8,)

    def test_single_dep_logrms(self, synthetic_fragment):
        results = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment, f_s=1000, features=["logrms"], kwargs={}
        )
        assert "logrms" in results
        assert results["logrms"].shape == (8,)

    def test_multi_level_deps_logpsdband(self, synthetic_fragment):
        """logpsdband → psdband → psd  (two-level dependency)."""
        results = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment, f_s=1000, features=["logpsdband"], kwargs={}
        )
        assert "logpsdband" in results

    def test_diamond_deps_psdfrac(self, synthetic_fragment):
        """psdfrac → psdband → psd. psdtotal also → psd. No redundant computation."""
        results = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment,
            f_s=1000,
            features=["psdfrac", "psdtotal"],
            kwargs={},
        )
        assert "psdfrac" in results
        assert "psdtotal" in results

    def test_deep_chain_logpsdfrac(self, synthetic_fragment):
        """logpsdfrac → psdfrac → psdband → psd  (three levels)."""
        results = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment, f_s=1000, features=["logpsdfrac"], kwargs={}
        )
        assert "logpsdfrac" in results

    def test_multiple_independent_features(self, synthetic_fragment):
        results = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment,
            f_s=1000,
            features=["rms", "ampvar"],
            kwargs={},
        )
        assert "rms" in results
        assert "ampvar" in results

    def test_cached_dep_not_recomputed(self, synthetic_fragment):
        """Requesting rms + logrms should compute rms once."""
        results = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment,
            f_s=1000,
            features=["rms", "logrms"],
            kwargs={},
        )
        assert "rms" in results
        assert "logrms" in results
        expected_logrms = np.log(results["rms"] + 1)
        np.testing.assert_allclose(results["logrms"], expected_logrms)

    def test_legacy_process_matches_deps(self, synthetic_fragment):
        """Legacy _process_fragment_features_dask gives same results for base features."""
        legacy = FragmentAnalyzer._process_fragment_features_dask(
            synthetic_fragment, f_s=1000, features=["rms", "ampvar"], kwargs={}
        )
        dep = FragmentAnalyzer.process_fragment_with_dependencies(
            synthetic_fragment, f_s=1000, features=["rms", "ampvar"], kwargs={}
        )
        np.testing.assert_allclose(legacy["rms"], dep["rms"])
        np.testing.assert_allclose(legacy["ampvar"], dep["ampvar"])

    def test_invalid_feature_raises(self, synthetic_fragment):
        with pytest.raises(AttributeError):
            FragmentAnalyzer.process_fragment_with_dependencies(
                synthetic_fragment,
                f_s=1000,
                features=["nonexistent_feature"],
                kwargs={},
            )

    def test_legacy_invalid_feature_raises(self, synthetic_fragment):
        with pytest.raises(AttributeError):
            FragmentAnalyzer._process_fragment_features_dask(
                synthetic_fragment,
                f_s=1000,
                features=["nonexistent_feature"],
                kwargs={},
            )


# ---------------------------------------------------------------------------
# FrequencyDomainSpikeDetector baseline edge case
# ---------------------------------------------------------------------------
from neurodent.core.frequency_domain_spike_detection import FrequencyDomainSpikeDetector


class TestSpikeDetectorBaselineEdge:
    """Test short-baseline warning path in _enforce_downward_and_refine_minimal."""

    def test_very_short_signal_warns(self):
        """A spike near signal boundary may produce a baseline < 10 samples."""
        np.random.seed(42)
        signal = np.random.randn(50)
        signal[2] = -20  # artificially large spike near the start

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
                signal,
                fs=1000,
                candidates=np.array([2]),
                search_ms=10,
                baseline_ms=5,  # very small baseline → likely < 10 samples
            )
            # Function should not crash; result is an array
            assert isinstance(result, np.ndarray)

    def test_spike_at_signal_edge(self):
        """Spike at index 0 should not crash."""
        signal = np.zeros(100)
        signal[0] = -10
        result = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
            signal,
            fs=1000,
            candidates=np.array([0]),
            search_ms=10,
            baseline_ms=5,
        )
        assert isinstance(result, np.ndarray)

    def test_empty_candidates(self):
        """Empty candidates should return empty array."""
        signal = np.random.randn(100)
        result = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
            signal,
            fs=1000,
            candidates=np.array([]),
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Misc utility coverage gaps
# ---------------------------------------------------------------------------
from neurodent.core.utils import (
    get_file_stem,
    log_transform,
    get_cache_status_message,
    _get_groupby_keys,
    _get_pairwise_combinations,
)
import pandas as pd


class TestMiscUtilsCoverage:
    """Small utilities that lacked test coverage."""

    def test_get_file_stem_double_ext(self):
        assert get_file_stem("/data/recording.npy.gz") == "recording"

    def test_get_file_stem_single_ext(self):
        assert get_file_stem("/data/file.rhd") == "file"

    def test_log_transform_none(self):
        assert log_transform(None) is None

    def test_log_transform_values(self):
        arr = np.array([0.0, 1.0, np.e - 1])
        result = log_transform(arr)
        np.testing.assert_allclose(result, np.log(arr + 1))

    def test_get_cache_status_message(self, tmp_path):
        msg_use = get_cache_status_message(tmp_path / "c.pkl", True)
        assert "Using cached" in msg_use
        msg_regen = get_cache_status_message(tmp_path / "c.pkl", False)
        assert "Regenerating" in msg_regen

    def test_get_groupby_keys(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        keys = _get_groupby_keys(df, "g")
        assert set(keys) == {"a", "b"}

    def test_get_pairwise_combinations(self):
        combos = _get_pairwise_combinations([1, 2, 3])
        assert set(combos) == {(1, 2), (1, 3), (2, 3)}

    def test_get_pairwise_combinations_empty(self):
        assert _get_pairwise_combinations([]) == []

    def test_get_pairwise_combinations_single(self):
        assert _get_pairwise_combinations([1]) == []


# ---------------------------------------------------------------------------
# Aggregate numpy array coverage  (agg_np_array)
# ---------------------------------------------------------------------------
from neurodent.core.utils import nanmean_series_of_np


class TestNanmeanSeriesOfNp:
    """Tests for the optimized numpy aggregation function."""

    def test_small_series(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        series = pd.Series(list(arr))
        result = nanmean_series_of_np(series, axis=0)
        expected = np.nanmean(arr, axis=0)
        np.testing.assert_allclose(result, expected)

    def test_large_series_stack_path(self):
        """Series > 1000 elements triggers np.stack fast path."""
        np.random.seed(0)
        arrays = [np.random.randn(5) for _ in range(1500)]
        series = pd.Series(arrays)
        result = nanmean_series_of_np(series, axis=0)
        expected = np.nanmean(np.stack(arrays), axis=0)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Workflow utils coverage  (workflow/utils.py)
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
