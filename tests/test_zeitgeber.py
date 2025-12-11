
import pytest
import pandas as pd
import numpy as np
import datetime
from neurodent.analysis import zeitgeber

# Sample data fixtures
@pytest.fixture
def sample_features_df():
    # Create a synthetic dataframe with some features
    # 2 animals, 2 genotypes, 24 hours of data
    
    data = []
    animals = ["anim1", "anim2"]
    genotypes = ["M_WT", "F_KO"]
    
    start_time = datetime.datetime(2023, 1, 1, 0, 0)
    
    for i, (anim, geno) in enumerate(zip(animals, genotypes)):
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
                
            data.append({
                "timestamp": timestamp,
                "animal": anim,
                "genotype": geno,
                "feature_const": val1,
                "feature_wave": val2
            })
            
    return pd.DataFrame(data)

def test_convert_to_zeitgeber_time(sample_features_df):
    df = zeitgeber.convert_to_zeitgeber_time(sample_features_df)
    
    assert "total_minutes" in df.columns
    assert "hour" in df.columns
    assert "minute" in df.columns
    
    # Check conversion
    # 00:00 -> 0 min
    row0 = df[df["timestamp"].dt.hour == 0].iloc[0]
    assert row0["total_minutes"] == 0
    
    # 02:00 -> 120 min
    row2 = df[df["timestamp"].dt.hour == 2].iloc[0]
    assert row2["total_minutes"] == 120

def test_baseline_correct_features(sample_features_df):
    # First add ZT time
    df = zeitgeber.convert_to_zeitgeber_time(sample_features_df)
    
    # Shift to ZT (ZT0 = 6am = 360 min)
    df["total_minutes"] = (df["total_minutes"] - 360) % 1440
    
    # Baseline: first 12 hours (ZT0-ZT12)
    # ZT0-12 corresponds to Clock 6:00-18:00
    
    processed = zeitgeber.baseline_correct_features(df, baseline_hours=12)
    
    assert "feature_const_nobase" in processed.columns
    assert "feature_wave_nobase" in processed.columns
    
    # For feature_const (value 10), baseline should be 10, so nobase should be 0 (ignoring NaNs)
    # We need to check non-NaN values
    valid_rows = processed.dropna()
    assert np.allclose(valid_rows["feature_const_nobase"], 0.0)

def test_prepare_plot_data(sample_features_df):
    df = zeitgeber.convert_to_zeitgeber_time(sample_features_df)
    
    # Test with ZT shift
    # Clock 00:00 (0 min) -> ZT18 (1080 min)
    # Clock 06:00 (360 min) -> ZT0 (0 min)
    
    processed = zeitgeber.prepare_plot_data(df, shift_for_48h=False, perform_zt_shift=True)
    
    row_6am = processed[processed["timestamp"].dt.hour == 6].iloc[0]
    assert row_6am["total_minutes"] == 0
    
    row_0am = processed[processed["timestamp"].dt.hour == 0].iloc[0]
    assert row_0am["total_minutes"] == 1080 # 18 * 60

def test_nan_handling_legacy_issue(sample_features_df):
    """
    Test specifically for the issue found in debug_zeitgeber_nans.py
    where specific timepoints might have NaNs.
    """
    df = zeitgeber.convert_to_zeitgeber_time(sample_features_df)
    
    # Check anim2 at 2am (120 min)
    anim2_2am = df[(df["animal"] == "anim2") & (df["total_minutes"] == 120)]
    assert len(anim2_2am) == 1
    assert np.isnan(anim2_2am.iloc[0]["feature_const"])
    
    # Ensure processing doesn't crash with NaNs
    processed = zeitgeber.process_zeitgeber_data(df)
    
    # The NaN should propagate or be handled gracefully
    anim2_processed_row = processed[(processed["animal"] == "anim2") & (processed["total_minutes"] == (120 - 360)%1440)]
    # Note: process_zeitgeber_data does 48h expansion, so we might find 2 rows
    # And ZT shift: 2am (120) - 6am (360) = -240 = 1200 (ZT20)
    
    # Check ZT20
    zt20_rows = processed[(processed["animal"] == "anim2") & (processed["total_minutes"] == 1200)]
    assert len(zt20_rows) >= 1
    
    # Should still satisfy processing requirements (metadata added, etc)
    assert "sex" in processed.columns
    assert "gene" in processed.columns

