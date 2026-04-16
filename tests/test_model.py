"""
Unit tests for model feature engineering and prediction logic.

Run with:
    pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Feature Engineering Tests
# =============================================================================

class TestFeatureEngineering:
    """Test engineer_features() from train_model.py."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([{
            "trans_date_trans_time": "2019-01-02 01:06:00",
            "cc_num": "2703192139880685",
            "merchant": "fraud_Rippin",
            "category": "grocery_pos",
            "amt": 281.06,
            "first": "Jennifer",
            "last": "Banks",
            "gender": "F",
            "street": "561 Perry Cove",
            "city": "Moravian Falls",
            "state": "NC",
            "zip": 28654,
            "lat": 36.0788,
            "long": -81.1781,
            "city_pop": 3495,
            "job": "Psychologist",
            "dob": "9/3/88",
            "trans_num": "abc123",
            "unix_time": 1325376018,
            "merch_lat": 36.011,
            "merch_long": -82.048,
            "is_fraud": 1,
        }])

    def test_extracts_hour(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert "hour" in df.columns
        assert df["hour"].iloc[0] == 1

    def test_extracts_day_of_week(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert "day_of_week" in df.columns
        assert 0 <= df["day_of_week"].iloc[0] <= 6

    def test_is_night_at_1am(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert df["is_night"].iloc[0] == 1

    def test_is_not_night_at_noon(self, sample_df):
        from scripts.train_model import engineer_features
        df = sample_df.copy()
        df["trans_date_trans_time"] = "2019-01-02 12:00:00"
        result, _ = engineer_features(df)
        assert result["is_night"].iloc[0] == 0

    def test_is_weekend(self, sample_df):
        from scripts.train_model import engineer_features
        # 2019-01-05 is Saturday
        df = sample_df.copy()
        df["trans_date_trans_time"] = "2019-01-05 12:00:00"
        result, _ = engineer_features(df)
        assert result["is_weekend"].iloc[0] == 1

    def test_is_not_weekend(self, sample_df):
        from scripts.train_model import engineer_features
        # 2019-01-02 is Wednesday
        df, _ = engineer_features(sample_df.copy())
        assert df["is_weekend"].iloc[0] == 0

    def test_age_computed(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert "age" in df.columns
        assert df["age"].iloc[0] > 0

    def test_distance_computed(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert "distance" in df.columns
        assert df["distance"].iloc[0] >= 0

    def test_gender_enc_female(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert df["gender_enc"].iloc[0] == 0  # F = 0

    def test_gender_enc_male(self, sample_df):
        from scripts.train_model import engineer_features
        df = sample_df.copy()
        df["gender"] = "M"
        result, _ = engineer_features(df)
        assert result["gender_enc"].iloc[0] == 1  # M = 1

    def test_category_encoded(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert "category_enc" in df.columns
        assert isinstance(df["category_enc"].iloc[0], (int, np.integer))

    def test_amt_vs_category_mean(self, sample_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(sample_df.copy())
        assert "amt_vs_category_mean" in df.columns
        assert df["amt_vs_category_mean"].iloc[0] > 0

    def test_test_set_uses_train_encoders(self, sample_df):
        """Encoders fitted on train must transform test without errors."""
        from scripts.train_model import engineer_features
        _, label_encoders = engineer_features(sample_df.copy(), fit=True)
        result, _ = engineer_features(sample_df.copy(),
                                      label_encoders=label_encoders, fit=False)
        assert "category_enc" in result.columns
        assert "state_enc" in result.columns


# =============================================================================
# Haversine Distance Tests
# =============================================================================

class TestHaversine:

    def test_same_point_is_zero(self):
        from scripts.train_model import haversine
        d = haversine(36.0, -81.0, 36.0, -81.0)
        assert d < 0.01

    def test_known_distance(self):
        """NYC to LA ~3940 km."""
        from scripts.train_model import haversine
        d = haversine(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < d < 4000

    def test_non_negative(self):
        from scripts.train_model import haversine
        d = haversine(10.0, 20.0, -10.0, -20.0)
        assert d >= 0


# =============================================================================
# Risk Level Tests
# =============================================================================

class TestRiskLevel:

    def test_low(self):
        from app.model import get_risk_level
        assert get_risk_level(0.0)  == "LOW"
        assert get_risk_level(0.29) == "LOW"

    def test_medium(self):
        from app.model import get_risk_level
        assert get_risk_level(0.3)  == "MEDIUM"
        assert get_risk_level(0.49) == "MEDIUM"

    def test_high(self):
        from app.model import get_risk_level
        assert get_risk_level(0.5)  == "HIGH"
        assert get_risk_level(0.79) == "HIGH"

    def test_critical(self):
        from app.model import get_risk_level
        assert get_risk_level(0.8)  == "CRITICAL"
        assert get_risk_level(1.0)  == "CRITICAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])