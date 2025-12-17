import pandas as pd
import numpy as np
import logging
from neurodent.core import (
    ZeitgeberAnalysisResult,
    transform_time_axis,
    get_expanded_feature_names
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_verify_pipeline():
    logger.info("Starting pipeline verification...")

    # Mock Data Creation
    # Create 48 hours of data for 2 animals
    dates = pd.date_range("2023-01-01 00:00", periods=48, freq="1H")
    
    data = []
    for animal in ["A1", "A2"]:
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "genotype": "M_WT" if animal == "A1" else "F_Mut",
                "feature1": np.sin(i / 24 * 2 * np.pi) + 10, # Sine wave
                "animal": animal,
                "animalday": f"{animal}_Day1",
                "day": "Day1",
                "duration": 3600,
                "endfile": "file.bin",
                "isday": True
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Created mock dataframe with shape: {df.shape}")

    # --- Mimic extract_zeitgeber_features.py logic with Class Wrapper ---
    logger.info("--- Testing Extraction Logic with Class Wrapper ---")

    # Mock WAR object to be wrapped
    class MockWAR:
        def __init__(self, data):
            self.data = data
        def get_channel_averaged_result(self, features=None):
            return self.data.copy()

    # Pipeline config
    pipeline_config = {
        "baseline_hours": 12,
        "zeitgeber_shift_hours": 6,
        "shift_for_48h": False
    }

    # Wrap with ZeitgeberAnalysisResult
    war = MockWAR(df)
    zar = ZeitgeberAnalysisResult(war, **pipeline_config)
    
    # Get result (should trigger pipeline)
    df_processed = zar.get_channel_averaged_result()
    
    # Verify Pipeline Steps
    
    # 1. Metadata
    assert "sex" in df_processed.columns
    assert "gene" in df_processed.columns
    logger.info("Metadata enrichment passed")

    # 2. Time columns
    assert "total_minutes" in df_processed.columns
    logger.info("Time columns addition passed")

    # 3. ZT Shift
    row_6am = df_processed[df_processed["timestamp"].dt.hour == 6].iloc[0]
    assert row_6am["total_minutes"] == 0
    logger.info("ZT shift passed")

    # 4. Baseline Subtraction
    assert "feature1_nobase" in df_processed.columns
    logger.info("Baseline subtraction passed")
    
    # 5. Check NO 48h expansion (shift_for_48h=False)
    assert len(df_processed) == len(df)
    logger.info("No 48h expansion passed")

    # Aggregation
    feature_cols = ["feature1", "feature1_nobase"]
    numeric_feature_cols = df_processed[feature_cols].select_dtypes(include=[int, float]).columns.tolist()
    agg_dict = {feature: "mean" for feature in numeric_feature_cols}
    
    group_cols = ["animal", "genotype", "sex", "gene", "total_minutes"]
    df_agg = df_processed.groupby(group_cols).agg(agg_dict).reset_index()
    logger.info("Aggregation passed")

    # --- Mimic generate_zeitgeber_plots.py logic ---
    logger.info("--- Testing Plotting Logic ---")

    # Transform time axis (48h expansion)
    df_plot = transform_time_axis(
        df_agg, 
        time_range=(0, 48), 
        shift=0
    )
    
    # Verify Metadata Helpers
    logger.info("--- Testing Metadata Helpers ---")
    # Mock config features
    config_features = ["feature1"]
    expanded_features = get_expanded_feature_names(config_features)
    assert "feature1" in expanded_features
    
    # Test expansion logic with known constants (mocking if needed, but we can just test the function)
    # We'll test with a fake band feature if we can, or just trust the function works as tested by unit tests
    # But here we want to verify the pipeline integration.
    
    # Check that our df_processed has the right columns identified
    expanded_features_nobase = [f"{f}_nobase" for f in expanded_features]
    expected_features = set(expanded_features + expanded_features_nobase)
    feature_cols = [col for col in df_processed.columns if col in expected_features]
    
    assert "feature1" in feature_cols
    assert "feature1_nobase" in feature_cols
    assert "total_minutes" not in feature_cols
    
    logger.info("Metadata helpers passed")
    
    # Check expansion
    # Original unique total_minutes should be 24 (hourly bins)
    # Expanded should have 48 unique values (0-23h and 24-47h equivalent)
    # Wait, prepare_plot_data adds 1440 to the copy.
    # So we should have values > 1440.
    assert df_plot["total_minutes"].max() >= 1440
    assert len(df_plot) == 2 * len(df_agg)
    logger.info("Plot preparation passed")

    logger.info("ALL CHECKS PASSED")
