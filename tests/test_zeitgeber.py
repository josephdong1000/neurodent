import pytest
import pandas as pd
import numpy as np
import datetime
from neurodent.core import zeitgeber


# Sample data fixtures
@pytest.fixture
def sample_features_df():
    # Create a synthetic dataframe with some features
    # 2 animals, 2 genotypes, 24 hours of data

    data = []
    animals = ["anim1", "anim2"]
    genotypes = ["WT", "KO"]
    sexes = ["Male", "Female"]

    start_time = datetime.datetime(2023, 1, 1, 0, 0)

    for i, (anim, geno, sx) in enumerate(zip(animals, genotypes, sexes)):
        for hour in range(24):
            # Clock time: 0-23
            timestamp = start_time + datetime.timedelta(hours=hour)

            # Simple feature: sine wave peaking at noon (12:00)
            # Clock 12:00 = ZT6.
            # Let's make it peak at ZT12 (18:00 clock)

            # Value 1: constant
            val1 = 10.0

            # Value 2: varying
            val2 = 10.0 + 5.0 * np.sin(2 * np.pi * hour / 24)

            # Add some NaN values specifically for anim2 at 2am (120 min)
            if anim == "anim2" and hour == 2:
                val1 = np.nan
                val2 = np.nan

            data.append(
                {
                    "timestamp": timestamp,
                    "animal": anim,
                    "genotype": geno,
                    "sex": sx,
                    "feature_const": val1,
                    "feature_wave": val2,
                }
            )

    return pd.DataFrame(data)


def test_add_zeitgeber_time_columns(sample_features_df):
    df = zeitgeber.add_zeitgeber_time_columns(sample_features_df)

    assert "zt_minutes" in df.columns
    assert "hour" in df.columns
    assert "minute" in df.columns

    # Check conversion
    # 00:00 -> 0 min
    row0 = df[df["timestamp"].dt.hour == 0].iloc[0]
    assert row0["zt_minutes"] == 0

    # 02:00 -> 120 min
    row2 = df[df["timestamp"].dt.hour == 2].iloc[0]
    assert row2["zt_minutes"] == 120


def test_subtract_zeitgeber_baseline(sample_features_df):
    # First add ZT time
    df = zeitgeber.add_zeitgeber_time_columns(sample_features_df)

    # Shift to ZT (ZT0 = 6am = 360 min)
    df["zt_minutes"] = (df["zt_minutes"] - 360) % 1440

    # Baseline: first 12 hours (ZT0-ZT12)
    # ZT0-12 corresponds to Clock 6:00-18:00

    processed = zeitgeber.subtract_zeitgeber_baseline(df, baseline_hours=12)

    assert "feature_const_nobase" in processed.columns
    assert "feature_wave_nobase" in processed.columns

    # For feature_const (value 10), baseline should be 10, so nobase should be 0 (ignoring NaNs)
    # We need to check non-NaN values
    valid_rows = processed.dropna()
    assert np.allclose(valid_rows["feature_const_nobase"], 0.0)


def test_transform_time_axis(sample_features_df):
    df = zeitgeber.add_zeitgeber_time_columns(sample_features_df)

    # Test with ZT shift (-6 hours)
    # Clock 00:00 (0 min) -> ZT18 (1080 min)
    # Clock 06:00 (360 min) -> ZT0 (0 min)

    processed = zeitgeber.transform_time_axis(
        df, time_range=(0, 24), shift=-6
    )

    row_6am = processed[processed["timestamp"].dt.hour == 6].iloc[0]
    assert row_6am["zt_minutes"] == 0

    row_0am = processed[processed["timestamp"].dt.hour == 0].iloc[0]
    assert row_0am["zt_minutes"] == 1080  # 18 * 60


def test_transform_time_axis_edge_cases():
    """Test edge cases for transform_time_axis."""
    # Create simple test data
    df = pd.DataFrame({
        "zt_minutes": [0, 360, 720, 1080],  # 0, 6, 12, 18 hours
        "genotype": ["WT", "WT", "KO", "KO"],
        "sex": ["Male", "Male", "Female", "Female"],
        "value": [1, 2, 3, 4],
    })

    # Test 1: Invalid time_range (start >= end)
    with pytest.raises(ValueError, match="must be less than end"):
        zeitgeber.transform_time_axis(df, time_range=(24, 24))
    
    with pytest.raises(ValueError, match="must be less than end"):
        zeitgeber.transform_time_axis(df, time_range=(48, 24))

    # Test 2: Positive shift (moves times later)
    result = zeitgeber.transform_time_axis(df, time_range=(0, 24), shift=6)
    # 0:00 + 6h = 6:00 (360 min)
    assert result.iloc[0]["zt_minutes"] == 360
    # 18:00 + 6h = 24:00 = 0:00 (wraps around)
    assert result[result["value"] == 4].iloc[0]["zt_minutes"] == 0

    # Test 3: transform_time_axis NEVER duplicates rows after the refactor —
    # multi-day expansion moved to expand_zt_axis().  The time_range param
    # is a no-op kept for backward compat.
    result_48h = zeitgeber.transform_time_axis(df, time_range=(0, 48), shift=0)
    assert len(result_48h) == len(df), (
        "transform_time_axis must not duplicate rows; use expand_zt_axis instead"
    )
    assert result_48h["zt_minutes"].max() < 1440

    # Test 4: same for any time_range — no duplication.
    result_72h = zeitgeber.transform_time_axis(df, time_range=(0, 72), shift=0)
    assert len(result_72h) == len(df)
    assert result_72h["zt_minutes"].max() < 1440

    # Test 5: No expansion (0-24)
    result_24h = zeitgeber.transform_time_axis(df, time_range=(0, 24), shift=0)
    assert len(result_24h) == len(df)
    assert result_24h["zt_minutes"].max() < 1440

    # Test 6: Metadata carried through (sex/genotype provided in input)
    assert "sex" in result_24h.columns
    assert "genotype" in result_24h.columns
    assert result_24h.iloc[0]["sex"] == "Male"
    assert result_24h.iloc[0]["genotype"] == "WT"


def test_nan_handling_legacy_issue(sample_features_df):
    """
    Test specifically for the issue found in debug_zeitgeber_nans.py
    where specific timepoints might have NaNs.
    """
    df = zeitgeber.add_zeitgeber_time_columns(sample_features_df)

    # Check anim2 at 2am (120 min)
    anim2_2am = df[(df["animal"] == "anim2") & (df["zt_minutes"] == 120)]
    assert len(anim2_2am) == 1
    assert np.isnan(anim2_2am.iloc[0]["feature_const"])

    # Ensure processing doesn't crash with NaNs
    processed = zeitgeber.run_zeitgeber_pipeline(df)

    # The NaN should propagate or be handled gracefully
    anim2_processed_row = processed[
        (processed["animal"] == "anim2")
        & (processed["zt_minutes"] == (120 - 360) % 1440)
    ]
    # ZT shift: 2am (120) - 6am (360) = -240 = 1200 (ZT20).
    # Post-refactor the pipeline no longer duplicates rows for a 48h view,
    # so we expect exactly the original row count per (animal, zt_minutes).

    # Check ZT20
    zt20_rows = processed[
        (processed["animal"] == "anim2") & (processed["zt_minutes"] == 1200)
    ]
    assert len(zt20_rows) >= 1

    # Should still satisfy processing requirements (metadata carried, etc)
    assert "sex" in processed.columns
    assert "genotype" in processed.columns


def test_grouped_baseline_correction():
    # Create data for 2 animals with different baselines
    # Animal 1: Baseline 10
    # Animal 2: Baseline 20

    rows = []
    # ZT0-ZT12 (0-720 min) is baseline
    for minute in range(0, 1440, 60):
        # Animal 1
        rows.append(
            {
                "zt_minutes": minute,
                "animal": "anim1",
                "val": 10.0 if minute <= 720 else 15.0,  # Jump to 15 after baseline
            }
        )
        # Animal 2
        rows.append(
            {
                "zt_minutes": minute,
                "animal": "anim2",
                "val": 20.0 if minute <= 720 else 25.0,  # Jump to 25 after baseline
            }
        )

    df = pd.DataFrame(rows)

    # Baseline correct using 12 hours
    processed = zeitgeber.subtract_zeitgeber_baseline(df, baseline_hours=12)

    # Check animal 1 results
    a1 = processed[processed["animal"] == "anim1"]
    # Baseline period should be 0 (10-10)
    assert np.allclose(a1[a1["zt_minutes"] <= 720]["val_nobase"], 0.0)
    # Post-baseline should be 5 (15-10)
    assert np.allclose(a1[a1["zt_minutes"] > 720]["val_nobase"], 5.0)

    # Check animal 2 results
    a2 = processed[processed["animal"] == "anim2"]
    # Baseline period should be 0 (20-20)
    assert np.allclose(a2[a2["zt_minutes"] <= 720]["val_nobase"], 0.0)
    # Post-baseline should be 5 (25-20)
    assert np.allclose(a2[a2["zt_minutes"] > 720]["val_nobase"], 5.0)


def test_baseline_exclusions():
    # Create simple dataframe
    df = pd.DataFrame(
        {
            "zt_minutes": [0, 600, 1200],  # ZT0, ZT10, ZT20
            "animal": ["a", "a", "a"],
            "feature_inc": [10, 10, 20],  # Should be corrected
            "feature_excl": [100, 100, 200],  # Should be excluded
        }
    )

    processed = zeitgeber.subtract_zeitgeber_baseline(
        df, baseline_hours=12, exclude_from_baseline=["feature_excl"]
    )

    # Check included feature
    assert "feature_inc_nobase" in processed.columns
    # Baseline (0, 600) is 10. val at 1200 is 20. 20-10=10.
    assert processed.iloc[2]["feature_inc_nobase"] == 10.0

    # Check excluded feature
    assert "feature_excl_nobase" not in processed.columns


def test_full_pipeline_via_run_zeitgeber_pipeline(sample_features_df):
    # Rename of test_metadata_enrichment_and_sort
    df = sample_features_df.copy()
    # sample_features_df provides genotype + sex; the pipeline carries them through.

    # We need to ensure zt_minutes exists before pipeline if we haven't stripped it
    # But sample_features_df creates a Raw DF? No, looking at lines 9-50, it makes cols:
    # timestamp, animal, genotype, feature_const, feature_wave.
    # It DOES NOT calculate zt_minutes.
    # So we MUST call add_zeitgeber_time_columns first for the pipeline to work,
    # OR the pipeline should handle it.
    # Our run_zeitgeber_pipeline docstring says: "Required columns: ... - 'zt_minutes'"
    # So we must call add_zeitgeber_time_columns first.

    df = zeitgeber.add_zeitgeber_time_columns(df)

    processed = zeitgeber.run_zeitgeber_pipeline(df)

    # 1. Check Metadata carried through
    assert "sex" in processed.columns
    assert "genotype" in processed.columns
    assert processed.iloc[0]["sex"] == "Male"
    assert processed.iloc[0]["genotype"] == "WT"

    # Check ZT Shift (Clock 0 -> ZT 18 (-6h) = 1080 min)
    # mock_zt_dataframe already has zt_minutes from add_zeitgeber_time_columns (0 at 00:00).
    # Processed should have zt_minutes=1080.  Post-refactor the data layer
    # no longer duplicates rows for a 48h view (that's the plotter's job),
    # so the max zt_minutes stays in the 24h range.
    assert 1080 in processed["zt_minutes"].values
    assert processed["zt_minutes"].max() < 1440

    # Check Baseline Subtraction
    assert "feature_const_nobase" in processed.columns

    # Check daynight label was added by shift_to_zeitgeber_reference.
    assert "daynight" in processed.columns
    assert set(processed["daynight"].unique()).issubset({"Day", "Night"})


def test_variable_intervals():
    # Test custom interval binning
    rows = []
    start_time = datetime.datetime(2023, 1, 1, 0, 0)
    for minute in range(0, 120, 10):  # 0, 10, ... 110
        rows.append({"timestamp": start_time + datetime.timedelta(minutes=minute)})
    df = pd.DataFrame(rows)

    # 1. Default (60 min)
    res_60 = zeitgeber.add_zeitgeber_time_columns(df.copy(), interval_minutes=60)
    assert res_60.iloc[1]["zt_minutes"] == 0  # 10 min
    assert res_60.iloc[5]["zt_minutes"] == 60  # 50 min

    # 2. 30 min interval
    res_30 = zeitgeber.add_zeitgeber_time_columns(df.copy(), interval_minutes=30)
    assert res_30.iloc[1]["zt_minutes"] == 0  # 10 min
    assert res_30.iloc[2]["zt_minutes"] == 30  # 20 min

    # 3. Edge Cases
    # 1 min interval (Valid)
    res_1 = zeitgeber.add_zeitgeber_time_columns(df.copy(), interval_minutes=1)
    assert res_1.iloc[1]["zt_minutes"] == 10

    # 1440 min interval (Valid - 24h bin)
    res_1440 = zeitgeber.add_zeitgeber_time_columns(df.copy(), interval_minutes=1440)
    assert res_1440.iloc[5]["zt_minutes"] == 0  # 50 min -> 0

    # 720 min interval (Valid - 12h bin)
    res_720 = zeitgeber.add_zeitgeber_time_columns(df.copy(), interval_minutes=720)
    assert res_720.iloc[5]["zt_minutes"] == 0

    # 4. Invalid Intervals (Primes and non-divisors)
    primes_and_oddities = [7, 11, 13, 17, 19, 23, 29, 31, 100, 500]
    for p in primes_and_oddities:
        if 1440 % p != 0:
            with pytest.raises(ValueError):
                zeitgeber.add_zeitgeber_time_columns(df.copy(), interval_minutes=p)


def test_flexible_baseline_windows():
    # Construct data covering 24 hours (ZT0-ZT24)
    # ZT0-6: 10
    # ZT6-12: 20
    # ZT12-18: 30
    # ZT18-24: 40

    rows = []
    for h in range(24):  # hours 0-23
        val = 10.0
        if h >= 6:
            val = 20.0
        if h >= 12:
            val = 30.0
        if h >= 18:
            val = 40.0

        rows.append({"zt_minutes": h * 60, "val": val, "animal": "anim1"})
    df = pd.DataFrame(rows)

    # 1. Custom Range (ZT6-12) => Mean=20
    res_range = zeitgeber.subtract_zeitgeber_baseline(df, baseline_window=(6, 12))
    assert np.isclose(
        res_range[res_range["zt_minutes"] == 360].iloc[0]["val_nobase"], 0.0
    )

    # 2. "day" alias (ZT0-12) => Mean=15
    res_day = zeitgeber.subtract_zeitgeber_baseline(df, baseline_window="day")
    assert np.isclose(
        res_day[res_day["zt_minutes"] == 0].iloc[0]["val_nobase"], -5.0
    )

    # 3. "night" alias (ZT12-24) => Mean=35
    res_night = zeitgeber.subtract_zeitgeber_baseline(df, baseline_window="night")
    assert np.isclose(
        res_night[res_night["zt_minutes"] == 720].iloc[0]["val_nobase"], -5.0
    )


def test_get_expanded_feature_names():
    """Test get_expanded_feature_names with various feature types."""
    from neurodent import constants
    
    # Test 1: Band features expand to per-band columns
    band_features = ["psdband", "logpsdband"]
    expanded = zeitgeber.get_expanded_feature_names(band_features)
    
    # Should have len(band_features) * len(BAND_NAMES) items
    expected_count = len(band_features) * len(constants.BAND_NAMES)
    assert len(expanded) == expected_count
    
    # Check specific expansions
    assert "psdband_delta" in expanded
    assert "psdband_theta" in expanded
    assert "logpsdband_gamma" in expanded
    
    # Test 2: BANDED matrix features (cohere, zcohere, imcoh, zimcoh) expand to per-band columns
    banded_matrix_features = ["cohere", "zcohere", "imcoh", "zimcoh"]
    expanded_banded = zeitgeber.get_expanded_feature_names(banded_matrix_features)
    assert "cohere_delta" in expanded_banded
    assert "zcohere_theta" in expanded_banded
    assert "imcoh_alpha" in expanded_banded
    assert "zimcoh_gamma" in expanded_banded
    # Should NOT contain the base name without band suffix
    assert "cohere" not in expanded_banded
    assert "zcohere" not in expanded_banded
    
    # Test 3: SIMPLE matrix features (pcorr, zpcorr) should NOT expand - stay as single column
    simple_matrix_features = ["pcorr", "zpcorr"]
    expanded_simple = zeitgeber.get_expanded_feature_names(simple_matrix_features)
    # Should keep as-is (no band expansion)
    assert expanded_simple == ["pcorr", "zpcorr"]
    # Should NOT have band-expanded versions
    assert "pcorr_delta" not in expanded_simple
    assert "zpcorr_theta" not in expanded_simple
    
    # Test 4: Linear features stay as-is
    linear_features = ["rms", "ampvar", "psdtotal"]
    expanded_linear = zeitgeber.get_expanded_feature_names(linear_features)
    assert expanded_linear == linear_features
    
    # Test 5: Mixed feature types including both banded and simple matrix features
    mixed = ["rms", "psdband", "cohere", "zpcorr"]
    expanded_mixed = zeitgeber.get_expanded_feature_names(mixed)
    
    # rms stays as-is (1), psdband expands (5), cohere expands (5), zpcorr stays as-is (1) = 12
    assert len(expanded_mixed) == 12
    assert "rms" in expanded_mixed
    assert "psdband_delta" in expanded_mixed
    assert "cohere_gamma" in expanded_mixed
    assert "zpcorr" in expanded_mixed  # Simple matrix feature - NOT expanded
    assert "zpcorr_delta" not in expanded_mixed  # Should NOT be expanded
    
    # Test 6: Unknown features pass through
    unknown = ["my_custom_feature"]
    expanded_unknown = zeitgeber.get_expanded_feature_names(unknown)
    assert expanded_unknown == ["my_custom_feature"]
    
    # Test 7: Verify constants are correctly defined
    assert "pcorr" in constants.SIMPLE_MATRIX_FEATURES
    assert "zpcorr" in constants.SIMPLE_MATRIX_FEATURES
    assert "cohere" in constants.BANDED_MATRIX_FEATURES
    assert "zcohere" in constants.BANDED_MATRIX_FEATURES


def test_add_zeitgeber_time_columns_empty_df():
    """Test add_zeitgeber_time_columns with empty/None dataframe."""
    # Empty dataframe
    empty_df = pd.DataFrame()
    result = zeitgeber.add_zeitgeber_time_columns(empty_df)
    assert result.empty
    
    # None dataframe
    result_none = zeitgeber.add_zeitgeber_time_columns(None)
    assert result_none is None


def test_subtract_baseline_no_group_cols():
    """Test baseline subtraction without grouping columns (no animal/sex/gene)."""
    df = pd.DataFrame({
        "zt_minutes": [0, 360, 720, 1080],
        "feature": [10.0, 10.0, 20.0, 20.0],
    })
    
    # Baseline first 12 hours: mean of [10, 10, 20] = 13.33
    result = zeitgeber.subtract_zeitgeber_baseline(df, baseline_hours=12)
    
    assert "feature_nobase" in result.columns
    # Values should be corrected by the global baseline mean
    assert not result["feature_nobase"].isna().all()


def test_subtract_baseline_empty_window():
    """Test baseline subtraction when baseline window has no data (ungrouped)."""
    df = pd.DataFrame({
        "zt_minutes": [720, 1080, 1320],  # All after ZT12
        # No group columns (animal/sex/gene) - tests the ungrouped branch
        "feature": [10.0, 20.0, 30.0],
    })
    
    # Baseline window ZT0-6 (0-360 min) - no data in this range
    result = zeitgeber.subtract_zeitgeber_baseline(df, baseline_window=(0, 6))
    
    # Should have NaN for nobase column since no baseline data
    assert "feature_nobase" in result.columns
    assert result["feature_nobase"].isna().all()


def test_subtract_baseline_empty_df():
    """Test baseline subtraction with empty dataframe."""
    empty_df = pd.DataFrame({"zt_minutes": [], "feature": []})
    result = zeitgeber.subtract_zeitgeber_baseline(empty_df)
    assert result.empty


def test_subtract_baseline_invalid_window_string():
    """Test baseline subtraction with invalid window alias."""
    df = pd.DataFrame({
        "zt_minutes": [0, 60, 120],
        "feature": [1, 2, 3],
    })
    
    with pytest.raises(ValueError, match="Unknown baseline_window alias"):
        zeitgeber.subtract_zeitgeber_baseline(df, baseline_window="invalid")


def test_subtract_baseline_invalid_window_type():
    """Test baseline subtraction with invalid window type."""
    df = pd.DataFrame({
        "zt_minutes": [0, 60, 120],
        "feature": [1, 2, 3],
    })
    
    with pytest.raises(ValueError, match="must be 'day', 'night', or a"):
        zeitgeber.subtract_zeitgeber_baseline(df, baseline_window=123)


def test_enrich_genotype_metadata_empty_df():
    """Test new metadata.enrich_metadata with empty dataframe."""
    from neurodent.core import metadata
    
    empty_df = pd.DataFrame({"animal": []})
    animal_meta = {"M1": {"sex": "Male", "gene": "WT"}}
    
    # Should handle empty df gracefully
    result = metadata.enrich_metadata(empty_df, animal_meta)
    assert len(result) == 0


def test_enrich_metadata_basic():
    """Test new metadata.enrich_metadata basic functionality."""
    from neurodent.core import metadata
    
    df = pd.DataFrame({
        "animal": ["M1", "F1"],
        "value": [10, 20],
    })
    animal_meta = {
        "M1": {"sex": "Male", "gene": "WT"},
        "F1": {"sex": "Female", "gene": "Mut"},
    }
    
    result = metadata.enrich_metadata(df, animal_meta)
    assert "sex" in result.columns
    assert "genotype" in result.columns
    assert result.iloc[0]["sex"] == "Male"
    assert result.iloc[1]["genotype"] == "Mut"


def test_zar_get_grouprows_result():
    """Test ZeitgeberAnalysisResult.get_grouprows_result."""
    from unittest.mock import MagicMock
    
    mock_war = MagicMock()
    mock_war.get_grouprows_result.return_value = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01 06:00", periods=3, freq="1h"),
        "genotype": ["WT", "WT", "WT"],
        "sex": ["Male", "Male", "Male"],
        "feature": [1, 2, 3],
    })
    
    zar = zeitgeber.ZeitgeberAnalysisResult(mock_war, baseline_hours=2)
    result = zar.get_grouprows_result()
    
    assert "zt_minutes" in result.columns
    assert "sex" in result.columns
    mock_war.get_grouprows_result.assert_called_once()


def test_zar_get_groupavg_result():
    """Test ZeitgeberAnalysisResult.get_groupavg_result."""
    from unittest.mock import MagicMock
    
    mock_war = MagicMock()
    mock_war.get_groupavg_result.return_value = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01 06:00", periods=3, freq="1h"),
        "genotype": ["WT", "WT", "WT"],
        "sex": ["Male", "Male", "Male"],
        "feature": [1, 2, 3],
    })
    
    zar = zeitgeber.ZeitgeberAnalysisResult(mock_war, baseline_hours=2)
    result = zar.get_groupavg_result()
    
    assert "zt_minutes" in result.columns
    mock_war.get_groupavg_result.assert_called_once()


def test_zar_empty_df():
    """Test ZeitgeberAnalysisResult with empty dataframe."""
    from unittest.mock import MagicMock
    
    mock_war = MagicMock()
    mock_war.get_result.return_value = pd.DataFrame()
    
    zar = zeitgeber.ZeitgeberAnalysisResult(mock_war)
    result = zar.get_result()
    
    assert result.empty


def test_deprecated_enrich_genotype_with_aliases():
    """Test deprecated enrich_genotype_metadata with genotype_aliases."""
    import warnings
    
    df = pd.DataFrame({
        "animal": ["M1", "F1"],
        "value": [10, 20],
    })
    
    genotype_aliases = {
        "MWT": ["M1"],
        "FMut": ["F1"],
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = zeitgeber.enrich_genotype_metadata(
            df, 
            genotype_aliases=genotype_aliases
        )
        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
    
    # Should still work via conversion to new format
    assert "sex" in result.columns
    assert "genotype" in result.columns
    assert result.iloc[0]["sex"] == "Male"
    assert result.iloc[0]["genotype"] == "WT"


def test_baseline_grouped_empty_window(caplog):
    """Test baseline subtraction with grouped data and empty baseline window."""
    import logging
    
    # Data where baseline window (0-360) has no data for group "b"
    df = pd.DataFrame({
        "zt_minutes": [0, 60, 720, 780],  # a has data in baseline, b does not
        "animal": ["a", "a", "b", "b"],
        "feature": [10.0, 10.0, 20.0, 20.0],
    })
    
    with caplog.at_level(logging.WARNING):
        result = zeitgeber.subtract_zeitgeber_baseline(df, baseline_window=(0, 6))
    
    # Group "a" should have valid baseline subtraction
    a_rows = result[result["animal"] == "a"]
    assert not a_rows["feature_nobase"].isna().all()
    
    # Group "b" baseline was calculated from time 720-780 which is outside 0-360
    # So group "b" should have NaN for nobase
    b_rows = result[result["animal"] == "b"]
    assert b_rows["feature_nobase"].isna().all()


def test_zar_ignores_extra_config_kwargs():
    """Test that ZeitgeberAnalysisResult ignores unknown config kwargs.
    
    Regression test for: run_zeitgeber_pipeline() got an unexpected keyword argument 'features'
    """
    from unittest.mock import MagicMock
    
    mock_war = MagicMock()
    mock_war.get_result.return_value = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01 06:00", periods=3, freq="1h"),
        "genotype": ["WT", "WT", "WT"],
        "sex": ["Male", "Male", "Male"],
        "feature": [1, 2, 3],
    })

    # Pass extra kwargs that are NOT valid for run_zeitgeber_pipeline
    zar = zeitgeber.ZeitgeberAnalysisResult(
        mock_war,
        features=['logpsdband'],  # Invalid - should be ignored
        unknown_key=42,           # Invalid - should be ignored
        baseline_hours=2,         # Valid - should be used
    )
    
    # Should not raise TypeError about unexpected keyword argument
    result = zar.get_result()
    
    # Verify pipeline was applied (has sex/gene columns from enrichment)
    assert "sex" in result.columns
    assert "zt_minutes" in result.columns

