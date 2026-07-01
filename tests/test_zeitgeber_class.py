import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from neurodent.core import zeitgeber


# Mock WindowAnalysisResult
class MockWAR:
    def __init__(self, data, animal_id="Animal_1", channel_names=None):
        self.data = data
        self.animal_id = animal_id
        self.channel_names = channel_names or ["Ch1", "Ch2"]
        self.channel_abbrevs = self.channel_names  # Simplified
        self.fs = 1000

    def get_result(self, features=None, exclude=None, allow_missing=False):
        # Return a copy to simulate data retrieval
        return self.data.copy()

    def get_grouprows_result(self, features=None, multiindex=None):
        # Simplified mock behavior: just return data, maybe group it if we wanted strict mocking
        return self.data.copy()

    def get_groupavg_result(self, features=None, groupby=None):
        # Simplified: return data
        return self.data.copy()

    def get_channel_averaged_result(self, features=None):
        return self.data.copy()


@pytest.fixture
def mock_war_data():
    # Create valid initial data with timestamps
    dates = pd.date_range("2023-01-01 06:00", periods=5, freq="1h")  # 6am to 10am
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "genotype": ["WT"] * 5,
            "sex": ["Male"] * 5,
            "feature1": [1, 2, 3, 4, 5],
            # 'zt_minutes' typically not present in raw WAR output, added by ZAR
        }
    )
    return MockWAR(df)


def test_zar_initialization(mock_war_data):
    """Test ZAR initialization and attribute delegation."""
    zar = zeitgeber.ZeitgeberAnalysisResult(mock_war_data, baseline_hours=2)

    assert zar.animal_id == "Animal_1"
    assert zar.channel_names == ["Ch1", "Ch2"]
    # Check config stored
    assert zar.config["baseline_hours"] == 2


def test_zar_get_result_pipeline(mock_war_data):
    """Test standard get_result interception runs pipeline."""
    zar = zeitgeber.ZeitgeberAnalysisResult(mock_war_data, baseline_hours=2)

    # Act
    df_res = zar.get_result()

    # Assert
    # 1. Total minutes added (via add_zeitgeber_time_columns injection)
    assert "zt_minutes" in df_res.columns
    # 2. Metadata enriched
    assert "sex" in df_res.columns
    assert df_res.iloc[0]["sex"] == "Male"
    # 3. ZT Shift (Default 6h shift: 6am -> ZT0)
    # 6:00 is minute 360. 360 - 6*60 = 0.
    assert df_res.iloc[0]["zt_minutes"] == 0

    # 4. Baseline Correction (baseline_hours=2)
    # Baseline window: ZT 0-2 (first 2 points: 1, 2). Mean = 1.5.
    # Feature1 uncorrected: 1, 2, 3, 4, 5
    # feature1_nobase: 1-1.5=-0.5, 2-1.5=0.5, 3-1.5=1.5, ...
    assert "feature1_nobase" in df_res.columns
    assert df_res.iloc[0]["feature1_nobase"] == -0.5


def test_zar_getattr_fallback(mock_war_data):
    """Test that methods not intercepted are passed to WAR."""
    # Add a custom method to mock
    mock_war_data.custom_method = MagicMock(return_value="called")

    zar = zeitgeber.ZeitgeberAnalysisResult(mock_war_data)

    assert zar.custom_method() == "called"
    mock_war_data.custom_method.assert_called_once()


def test_zar_get_channel_averaged_result(mock_war_data):
    """Test channel averaged result interception."""
    zar = zeitgeber.ZeitgeberAnalysisResult(mock_war_data)
    df = zar.get_channel_averaged_result()

    assert "zt_minutes" in df.columns
    assert "sex" in df.columns
    # daynight is added by shift_to_zeitgeber_reference inside the pipeline.
    assert "daynight" in df.columns
    assert set(df["daynight"].unique()).issubset({"Day", "Night"})


def test_zar_missing_timestamp_fallback(mock_war_data):
    """Test behavior if timestamp is missing from raw data."""
    # Data without timestamp
    df_no_time = pd.DataFrame({"genotype": ["WT"], "sex": ["Male"], "feature1": [1]})
    war_bad = MockWAR(df_no_time)

    zar = zeitgeber.ZeitgeberAnalysisResult(war_bad)
    df_res = zar.get_result()

    # Should run pipeline, but add_zeitgeber_time_columns effectively skips
    # And pipeline steps relying on zt_minutes (shift, baseline) should skip or act robustly
    assert "sex" in df_res.columns  # Metadata still runs
    assert "zt_minutes" not in df_res.columns
    assert "feature1_nobase" not in df_res.columns  # Baseline needs zt_minutes
