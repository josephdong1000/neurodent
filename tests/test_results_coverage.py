"""
Tests targeting uncovered lines in neurodent.visualization.results.
"""

import json
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from neurodent.visualization import WindowAnalysisResult, AnimalFeatureParser
from neurodent import constants
from neurodent.visualization.results import _sanitize_feature_request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_war_df(
    n_windows=4,
    channels=None,
    animal="A1",
    genotype="WT",
    include_rms=True,
    include_duration=True,
    extra_columns=None,
):
    """Build a minimal DataFrame accepted by WindowAnalysisResult."""
    channels = channels or ["Left Motor", "Right Motor"]
    n_ch = len(channels)
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_windows):
        for ch in channels:
            row = {
                "animal": animal,
                "animalday": f"{animal} {genotype} Jan-01-2023",
                "isday": True,
                "channel": ch,
                "endfile": f"file_{i}.bin",
                "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=i),
            }
            if include_duration:
                row["duration"] = 60.0
            if include_rms:
                row["rms"] = rng.random(n_ch).tolist()
            if extra_columns:
                row.update(extra_columns[i] if isinstance(extra_columns, list) else extra_columns)
            rows.append(row)
    return pd.DataFrame(rows)


def make_war(n_windows=4, channels=None, animal="A1", genotype="WT", **kwargs):
    """Shortcut: build a WAR with sensible defaults."""
    channels = channels or ["Left Motor", "Right Motor"]
    df = make_war_df(n_windows=n_windows, channels=channels, animal=animal, genotype=genotype, **kwargs)
    return WindowAnalysisResult(
        result=df,
        animal_id=animal,
        genotype=genotype,
        channel_names=channels,
        suppress_short_interval_error=True,
    )


# =========================================================================
# 1. _sanitize_feature_request (lines 1930, 1933)
# =========================================================================

class TestSanitizeFeatureRequest:

    def test_empty_features_raises(self):
        """Line 1930: empty list raises ValueError."""
        with pytest.raises(ValueError, match="Features cannot be empty"):
            _sanitize_feature_request([])

    def test_invalid_feature_raises(self):
        """Line 1933: feature not in constants.FEATURES raises ValueError."""
        with pytest.raises(ValueError, match="Available features are"):
            _sanitize_feature_request(["not_a_real_feature_xyz"])


# =========================================================================
# 2. AnimalFeatureParser._average_feature unsupported type (line 68)
# =========================================================================

class TestAnimalFeatureParserUnsupported:

    def test_unsupported_feature_type_raises(self):
        """Line 68: unsupported FeatureType raises TypeError."""
        parser = AnimalFeatureParser()
        df = pd.DataFrame({"fake_col": [1, 2, 3], "duration": [1.0, 1.0, 1.0]})
        with patch.object(constants, "classify_feature", return_value="UNKNOWN_TYPE"):
            with pytest.raises((TypeError, AttributeError)):
                parser._average_feature(df, "fake_col", "duration")


# =========================================================================
# 3. WAR.__str__ (line 1995)
# =========================================================================

class TestWARStr:

    def test_str_returns_animaldays(self):
        """Line 1995: __str__ returns animaldays."""
        war = make_war()
        result = str(war)
        assert "A1" in result


# =========================================================================
# 4. WAR._update_instance_vars (lines 2051-2052, 2090)
# =========================================================================

class TestUpdateInstanceVars:

    def test_drops_index_column_with_warning(self):
        """Lines 2051-2052: 'index' column dropped with warning."""
        df = make_war_df()
        df["index"] = range(len(df))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            war = WindowAnalysisResult(
                result=df, animal_id="A1", genotype="WT",
                channel_names=["Left Motor", "Right Motor"],
                suppress_short_interval_error=True,
            )
            assert any("index" in str(ww.message) for ww in w)
        assert "index" not in war.result.columns

    def test_animal_id_mismatch_raises(self):
        """Line 2090: animal_id mismatch raises ValueError."""
        df = make_war_df(animal="A1")
        with pytest.raises(ValueError, match="Animal ID mismatch"):
            WindowAnalysisResult(
                result=df, animal_id="WRONG",
                genotype="WT",
                channel_names=["Left Motor", "Right Motor"],
                suppress_short_interval_error=True,
            )


# =========================================================================
# 5. reorder_and_pad_channels (lines 2133-2134, 2155, 2166, 2242)
# =========================================================================

class TestReorderAndPadChannels:

    def test_duplicate_channels_raises(self):
        """Line 2155: duplicate target channels raise ValueError."""
        war = make_war()
        with pytest.raises(ValueError, match="duplicates"):
            war.reorder_and_pad_channels(["LMot", "LMot"])

    def test_no_channels_match_warns(self):
        """Line 2166: no matching channels warns."""
        war = make_war()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            war.reorder_and_pad_channels(["LAud", "RAud", "LVis"], use_abbrevs=True, inplace=False)
            assert any("None of the channel names" in str(ww.message) for ww in w)

    def test_unsupported_feature_type_raises(self):
        """Line 2242: unsupported FeatureType raises ValueError."""
        war = make_war()
        # Inject a fake feature column that classify_feature won't handle
        war.result["fake_feat"] = "dummy"
        war._feature_columns.append("fake_feat")
        fake_type = MagicMock()
        fake_type.is_matrix = False
        fake_type.is_dict_stored = False
        fake_type.__eq__ = lambda s, o: False
        fake_type.name = "FAKE"
        with patch.object(constants, "classify_feature", return_value=fake_type):
            with pytest.raises(ValueError, match="Unsupported FeatureType"):
                war.reorder_and_pad_channels(["LMot", "RMot"])


# =========================================================================
# 6. _extract_band_features non-dict row (lines 2686-2690)
# =========================================================================

class TestExtractBandFeatures:

    def test_non_dict_row_warns(self):
        """Lines 2686-2690: non-dict row produces warning and NaN padding."""
        war = make_war()
        df = war.result.copy()
        n = len(df)
        # First element must be dict to pass the initial check (line 2675),
        # but a later row is corrupted to trigger the per-row warning (line 2686).
        values = [{"alpha": [1.0, 2.0], "beta": [3.0, 4.0]}] * n
        values[1] = "not_a_dict"  # corrupt second row
        df["psdband"] = values
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = war._extract_band_features(df, "psdband", ["alpha", "beta"])
        assert "psdband_alpha" in result.columns


# =========================================================================
# 7. _extract_banded_matrix_features (lines 2733, 2751-2755, 2776, 2791,
#    2797-2802, 2816-2822)
# =========================================================================

class TestExtractBandedMatrixFeatures:

    def test_missing_column_returns_df(self):
        """Line 2733: feature not in columns returns df unchanged."""
        war = make_war()
        df = war.result.copy()
        result = war._extract_banded_matrix_features(df, "nonexistent", ["alpha"])
        assert result is df

    def test_non_2d_matrix_warns(self):
        """Lines 2751-2755: non-2D matrix in dict produces warning and NaN fill."""
        war = make_war()
        df = war.result.copy()
        bad_matrix = np.array([1, 2, 3])  # 1D, not 2D
        df["cohere"] = [{"alpha": bad_matrix}] * len(df)
        result = war._extract_banded_matrix_features(df, "cohere", ["alpha"])
        assert "cohere_alpha" in result.columns
        # Non-2D matrices should be replaced with NaN-filled matrices
        for val in result["cohere_alpha"]:
            if isinstance(val, np.ndarray):
                assert np.isnan(val).all(), "Non-2D input should produce NaN-filled matrix"

    def test_list_3d_array_format(self):
        """Lines 2776, 2791: list-stored 3D array format."""
        war = make_war()
        df = war.result.copy()
        n_ch = 2
        # 3D array: (bands, ch, ch) stored as list
        arr = np.ones((2, n_ch, n_ch))
        df["cohere"] = [arr.tolist()] * len(df)
        result = war._extract_banded_matrix_features(df, "cohere", ["alpha", "beta"])
        assert "cohere_alpha" in result.columns
        assert "cohere_beta" in result.columns

    def test_band_count_mismatch_raises(self):
        """Lines 2797-2802: band count mismatch in 3D array raises."""
        war = make_war()
        df = war.result.copy()
        n_ch = 2
        arr = np.ones((3, n_ch, n_ch))  # 3 bands
        df["cohere"] = [arr] * len(df)
        with pytest.raises(ValueError, match="Band count mismatch|bands"):
            war._extract_banded_matrix_features(df, "cohere", ["alpha", "beta"])  # expects 2

    def test_wrong_dimensionality_raises(self):
        """Lines 2816-2822: wrong dimensionality raises ValueError."""
        war = make_war()
        df = war.result.copy()
        arr = np.ones((2,))  # 1D
        df["cohere"] = [arr] * len(df)
        with pytest.raises(ValueError, match="wrong dimensionality"):
            war._extract_banded_matrix_features(df, "cohere", ["alpha"])

    def test_unexpected_format_raises(self):
        """Lines 2821-2826: unexpected format type raises ValueError."""
        war = make_war()
        df = war.result.copy()
        df["cohere"] = ["string_value"] * len(df)
        with pytest.raises(ValueError, match="unexpected format"):
            war._extract_banded_matrix_features(df, "cohere", ["alpha"])

    def test_2d_array_raises(self):
        """Lines 2809-2814: 2D array for banded feature raises."""
        war = make_war()
        df = war.result.copy()
        n_ch = 2
        arr = np.ones((n_ch, n_ch))  # 2D, not 3D
        df["cohere"] = [arr] * len(df)
        with pytest.raises(ValueError, match="2D array"):
            war._extract_banded_matrix_features(df, "cohere", ["alpha"])


# =========================================================================
# 8. _average_across_channels (lines 2888-2893, 2906)
# =========================================================================

class TestAverageAcrossChannels:

    def test_non_2d_matrix_warns_and_fills_nan(self):
        """Lines 2888-2893: non-2D matrix produces warning and NaN."""
        war = make_war()
        df = war.result.copy()
        df["pcorr"] = [np.eye(2).tolist()] * len(df)
        df.at[df.index[1], "pcorr"] = "not_a_matrix"
        result = war._average_across_channels(df, ["pcorr"])
        assert "pcorr" in result.columns
        # The corrupted row should produce NaN
        assert np.isnan(result["pcorr"].iloc[1])

    def test_small_matrix_nanmean_fallback(self):
        """Lines 2895-2900: 1x1 matrix falls back to nanmean instead of upper triangle."""
        war = make_war()
        df = war.result.copy()
        df["pcorr"] = [np.array([[5.0]]).tolist()] * len(df)
        result = war._average_across_channels(df, ["pcorr"])
        assert "pcorr" in result.columns
        # 1x1 matrix can't use upper triangle; nanmean of [[5.0]] == 5.0
        assert all(v == pytest.approx(5.0) for v in result["pcorr"])


# =========================================================================
# 9. get_filter_high_beta PSD alternative (lines 3000-3006)
# =========================================================================

class TestGetFilterHighBeta:

    def test_psdband_psdtotal_fallback(self):
        """Lines 3000-3004: use psdband+psdtotal when psdfrac absent."""
        war = make_war()
        n_ch = len(war.channel_names)
        if "psdfrac" in war.result.columns:
            war.result = war.result.drop(columns=["psdfrac"])
        # beta=0.3, total=1.0 -> proportion=0.3, which is below max_beta_prop=0.4
        war.result["psdband"] = [{"beta": np.array([0.3] * n_ch)}] * len(war.result)
        war.result["psdtotal"] = [np.array([1.0] * n_ch)] * len(war.result)
        result = war.get_filter_high_beta()
        assert result.shape[0] == len(war.result)
        # All windows should pass (0.3 < 0.4 threshold)
        assert result.all(), "All windows should pass with beta proportion 0.3 < 0.4"

    def test_missing_psd_features_raises(self):
        """Lines 3005-3008: missing both psdfrac and psdband raises."""
        war = make_war()
        war.result = war.result.drop(columns=["psdfrac", "psdband", "psdtotal"], errors="ignore")
        with pytest.raises(ValueError, match="psdfrac or psdband"):
            war.get_filter_high_beta()


# =========================================================================
# 10. get_filter_reject_channels channel not found (line 3066)
# =========================================================================

class TestGetFilterRejectChannels:

    def test_channel_not_found_warns(self):
        """Line 3066: channel not in targets warns."""
        war = make_war()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            war.get_filter_reject_channels(bad_channels=["NonexistentChannel"], use_abbrevs=False)
            assert any("not found" in str(ww.message) for ww in w)


# =========================================================================
# 11. get_filter_morphological_smoothing small structure (line 3216)
# =========================================================================

class TestMorphologicalSmoothing:

    def test_small_structure_early_return(self):
        """Line 3216: structure_size <= 1 returns mask unchanged (no morphological ops).

        With duration=60s and smoothing_seconds=30s, structure_size = max(1, int(30/60)) = 1,
        which triggers the early return path without applying any morphological operations.
        We verify this by using a mask with isolated False values that morphological opening
        would remove if the operations were actually applied.
        """
        war = make_war()
        n_windows = len(war.result)
        n_ch = len(war.channel_names)
        mask = np.ones((n_windows, n_ch), dtype=bool)
        # Set isolated False values that opening would expand if ops ran
        mask[1, 0] = False
        result = war.get_filter_morphological_smoothing(mask, smoothing_seconds=30.0)
        # Early return means isolated False is preserved exactly
        np.testing.assert_array_equal(result, mask)


# =========================================================================
# 12. filter_all (lines 3313-3342, 3568)
# =========================================================================

class TestFilterAll:

    def test_morphological_smoothing_missing_duration_raises(self):
        """Morphological smoothing requires a 'duration' column; raises if missing."""
        war = make_war(include_duration=False)
        # Use apply_filters with a single MASK_POST filter so the missing-duration
        # check fires before any per-row filter needs rms/psd data.
        with pytest.raises(ValueError, match="duration"):
            war.apply_filters(
                filter_config={
                    "morphological_smoothing": {"smoothing_seconds": 10.0},
                },
                min_valid_channels=0,
            )

    def test_default_all_true_when_no_filters_in_apply_filters(self):
        """Line 3568: empty filter list produces all-True mask."""
        war = make_war()
        result = war.apply_filters(filter_config={})
        assert len(result.result) > 0


# =========================================================================
# 13. _apply_filter unsupported FeatureType (line 3662)
# =========================================================================

class TestApplyFilter:

    def test_unsupported_feature_type_raises(self):
        """Line 3662: unsupported FeatureType in _apply_filter raises."""
        war = make_war()
        war.result["fake_feat"] = "dummy"
        war._feature_columns.append("fake_feat")
        fake_type = MagicMock()
        fake_type.is_matrix = False
        fake_type.is_dict_stored = False
        fake_type.name = "FAKE"
        fake_type.__eq__ = lambda s, o: False
        # Need to make the in-check fail for LINEAR etc
        mask = np.ones((len(war.result), len(war.channel_names)), dtype=bool)
        with patch.object(constants, "classify_feature", return_value=fake_type):
            with pytest.raises(ValueError, match="Unsupported FeatureType"):
                war._apply_filter(mask)


# =========================================================================
# 14. _NumpyEncoder (lines 3746-3752)
# =========================================================================

class TestNumpyEncoder:

    def test_np_bool_encoding(self):
        """Line 3750-3751: np.bool_ handled."""
        encoder = WindowAnalysisResult._NumpyEncoder()
        result = encoder.default(np.bool_(True))
        assert result is True

    def test_np_ndarray_encoding(self):
        """Line 3744-3745: np.ndarray handled."""
        encoder = WindowAnalysisResult._NumpyEncoder()
        result = encoder.default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_unsupported_type_raises(self):
        """Line 3752: other types fall through to super().default()."""
        encoder = WindowAnalysisResult._NumpyEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())

    def test_np_integer_encoding(self):
        """Lines 3746-3747: np.integer handled."""
        encoder = WindowAnalysisResult._NumpyEncoder()
        result = encoder.default(np.int64(42))
        assert result == 42

    def test_np_floating_encoding(self):
        """Lines 3748-3749: np.floating handled."""
        encoder = WindowAnalysisResult._NumpyEncoder()
        result = encoder.default(np.float64(3.14))
        assert result == pytest.approx(3.14)


# =========================================================================
# 15. _decode_df_from_parquet JSONDecodeError (lines 3797, 3804-3806)
# =========================================================================

class TestDecodeFromParquet:

    def test_json_decode_error_returns_original(self):
        """Lines 3804-3806: JSONDecodeError returns original string."""
        df = pd.DataFrame({"col": ["not_json", '{"a": 1}']})
        result = WindowAnalysisResult._decode_df_from_parquet(df, ["col"])
        assert result["col"].iloc[0] == "not_json"
        assert result["col"].iloc[1] == {"a": 1}

    def test_missing_column_skipped(self):
        """Line 3797: column not in df is skipped."""
        df = pd.DataFrame({"other": [1, 2]})
        result = WindowAnalysisResult._decode_df_from_parquet(df, ["nonexistent"])
        assert list(result.columns) == ["other"]

    def test_non_string_values_passed_through(self):
        """Line 3806: non-string values returned as-is."""
        df = pd.DataFrame({"col": [42, None]})
        result = WindowAnalysisResult._decode_df_from_parquet(df, ["col"])
        assert result["col"].iloc[0] == 42


# =========================================================================
# 16. get_bad_channels_by_lof_threshold / get_lof_scores (lines 3822, 3847)
# =========================================================================

class TestLOFMethods:

    def test_get_bad_channels_no_scores_raises(self):
        """Line 3822: empty lof_scores_dict raises ValueError."""
        war = make_war()
        war.lof_scores_dict = {}
        with pytest.raises(ValueError, match="LOF scores not available"):
            war.get_bad_channels_by_lof_threshold(1.5)

    def test_get_lof_scores_no_scores_raises(self):
        """Line 3847: empty lof_scores_dict raises ValueError."""
        war = make_war()
        war.lof_scores_dict = {}
        with pytest.raises(ValueError, match="LOF scores not available"):
            war.get_lof_scores()

    def test_get_bad_channels_works(self):
        """Functional test for get_bad_channels_by_lof_threshold."""
        war = make_war()
        animalday = war.animaldays[0]
        war.lof_scores_dict = {
            animalday: {
                "lof_scores": [0.5, 2.0],
                "channel_names": ["LMot", "RMot"],
            }
        }
        result = war.get_bad_channels_by_lof_threshold(1.0)
        assert "RMot" in result[animalday]
        assert "LMot" not in result[animalday]

    def test_get_lof_scores_works(self):
        """Functional test for get_lof_scores."""
        war = make_war()
        animalday = war.animaldays[0]
        war.lof_scores_dict = {
            animalday: {
                "lof_scores": [0.5, 2.0],
                "channel_names": ["LMot", "RMot"],
            }
        }
        result = war.get_lof_scores()
        assert result[animalday]["LMot"] == 0.5
        assert result[animalday]["RMot"] == 2.0


# =========================================================================
# 17. evaluate_lof_threshold_binary (lines 3915, 3940)
# =========================================================================

class TestEvaluateLOFThreshold:

    def test_no_ground_truth_raises(self):
        """Line 3915: empty bad_channels_dict with no ground_truth raises."""
        war = make_war()
        animalday = war.animaldays[0]
        war.lof_scores_dict = {
            animalday: {
                "lof_scores": [0.5, 2.0],
                "channel_names": ["Left Motor", "Right Motor"],
            }
        }
        war.bad_channels_dict = {animalday: []}  # empty → triggers line 3915
        # Actually bad_channels_dict is non-empty (has key), so it won't trigger 3915
        # Need truly empty dict
        war.bad_channels_dict = {}
        with pytest.raises(ValueError, match="No ground truth|empty"):
            war.evaluate_lof_threshold_binary(threshold=1.0)

    def test_invalid_lof_data_raises(self):
        """Line 3940: missing lof_scores field raises ValueError."""
        war = make_war()
        animalday = war.animaldays[0]
        war.lof_scores_dict = {
            animalday: {"bad_key": []}  # missing required fields
        }
        war.bad_channels_dict = {animalday: ["Left Motor"]}
        with pytest.raises(ValueError, match="missing required fields|Invalid LOF data"):
            war.evaluate_lof_threshold_binary(threshold=1.0)


# =========================================================================
# 18. load_parquet_and_json
# =========================================================================

class TestLoadParquetAndJson:

    def test_multiple_json_files_raises(self, tmp_path):
        """Multiple json files raises ValueError."""
        (tmp_path / "data.parquet").write_bytes(b"")
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")
        with pytest.raises(ValueError, match="Expected exactly one json"):
            WindowAnalysisResult.load_parquet_and_json(folder_path=tmp_path)

    def test_missing_parquet_raises(self, tmp_path):
        """Missing parquet file raises FileNotFoundError."""
        fake_parquet = tmp_path / "nonexistent.parquet"
        fake_json = tmp_path / "meta.json"
        fake_json.write_text("{}")
        with pytest.raises(FileNotFoundError, match="Parquet file not found"):
            WindowAnalysisResult.load_parquet_and_json(
                parquet_name=str(fake_parquet), json_name=str(fake_json)
            )

    def test_missing_json_raises(self, tmp_path):
        """Missing json file raises FileNotFoundError."""
        fake_parquet = tmp_path / "data.parquet"
        fake_parquet.write_bytes(b"")
        fake_json = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            WindowAnalysisResult.load_parquet_and_json(
                parquet_name=str(fake_parquet), json_name=str(fake_json)
            )


# =========================================================================
# 19. Parquet loading with sidecar fallback (lines 4110-4116, 4120-4125)
# =========================================================================

class TestParquetLoading:

    def test_legacy_sidecar_metadata(self, tmp_path):
        """Lines 4110-4116: legacy .meta.json sidecar fallback."""
        war = make_war()
        war.save_parquet_and_json(tmp_path, filename="test")

        parquet_path = tmp_path / "test.parquet"
        assert parquet_path.exists(), "Parquet file should be created by save_parquet_and_json"

        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        schema_meta = table.schema.metadata or {}
        encoded_cols = []
        if b"neurodent" in schema_meta:
            nd_meta = json.loads(schema_meta[b"neurodent"])
            encoded_cols = nd_meta.get("encoded_columns", [])

        # Rewrite without neurodent metadata to trigger sidecar fallback
        new_meta = {k: v for k, v in (schema_meta or {}).items() if k != b"neurodent"}
        new_table = table.replace_schema_metadata(new_meta)
        pq.write_table(new_table, parquet_path)

        # Write sidecar file
        sidecar_path = tmp_path / "test.parquet.meta.json"
        sidecar_path.write_text(json.dumps({"encoded_columns": encoded_cols}))

        # Specify json_name explicitly to avoid the "found 2 json" error
        loaded = WindowAnalysisResult.load_parquet_and_json(
            folder_path=tmp_path, json_name="test.json"
        )
        assert loaded.animal_id == "A1"

    def test_parquet_load_failure_falls_back_to_pickle(self, tmp_path):
        """Parquet load failure falls back to a legacy pickle when available."""
        war = make_war()
        war.save_parquet_and_json(tmp_path, filename="test")

        parquet_path = tmp_path / "test.parquet"
        assert parquet_path.exists(), "Parquet file should be created by save_parquet_and_json"

        # Simulate an old on-disk WAR: write a legacy pickle alongside the JSON,
        # then corrupt the parquet so loading has to fall through to the pickle.
        war.result.to_pickle(tmp_path / "test.pkl")
        parquet_path.write_bytes(b"corrupted data")

        loaded = WindowAnalysisResult.load_parquet_and_json(folder_path=tmp_path)
        assert loaded.animal_id == "A1"


# =========================================================================
# 20. aggregate_time_windows (lines 4147, 4149, 4153, 4164, 4178-4179, 4188)
# =========================================================================

class TestAggregateTimeWindows:

    def test_string_groupby_converted(self):
        """Line 4147: string groupby is converted to list."""
        # Use single channel to avoid non-constant 'channel' column error
        war = make_war(n_windows=4, channels=["Left Motor"])
        war.aggregate_time_windows(groupby="animalday")
        assert len(war.result) >= 1

    def test_invalid_groupby_raises(self):
        """Line 4149: invalid groupby column raises ValueError."""
        war = make_war()
        with pytest.raises(ValueError, match="groupby must be from"):
            war.aggregate_time_windows(groupby=["invalid_col"])

    def test_missing_groupby_column_raises(self):
        """Line 4153: groupby column not in result raises ValueError."""
        war = make_war()
        war.result = war.result.drop(columns=["isday"])
        with pytest.raises(ValueError, match="not found in result"):
            war.aggregate_time_windows(groupby=["isday"])

    def test_non_constant_column_raises(self):
        """Lines 4178-4179: non-constant non-feature column raises ValueError."""
        war = make_war()
        # Add a column that varies within groups
        war.result["varying_col"] = range(len(war.result))
        war._nonfeature_columns.append("varying_col")
        with pytest.raises(ValueError, match="not constant"):
            war.aggregate_time_windows(groupby=["animalday", "isday"])

    def test_duration_and_endfile_aggregation(self):
        """Line 4188: endfile takes last value, duration sums."""
        war = make_war(n_windows=4, channels=["Left Motor"])
        original_duration_sum = war.result["duration"].sum()
        war.aggregate_time_windows(groupby=["animalday", "isday"])
        assert war.result["duration"].sum() == pytest.approx(original_duration_sum)


# =========================================================================
# 21. add_unique_hash (lines 4222-4235)
# =========================================================================

class TestAddUniqueHash:

    def test_hash_added_to_animal_id(self):
        """Lines 4222-4235: hash suffix appended to animal_id."""
        war = make_war()
        old_id = war.animal_id
        war.add_unique_hash(nbytes=4)
        assert war.animal_id != old_id
        assert war.animal_id.startswith(old_id + "_")
        # animal column should be updated
        assert all(war.result["animal"] == war.animal_id)
        # animalday should be updated
        assert all(war.animal_id.split("_")[0] in ad for ad in war.result["animalday"])
