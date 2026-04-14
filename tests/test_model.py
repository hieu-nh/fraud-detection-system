"""
Unit tests for model preprocessing and prediction logic.
"""

import pytest
import numpy as np
import pandas as pd


# =============================================================================
# Test feature engineering logic
# =============================================================================

class TestFeatureEngineering:
    """Test data preprocessing functions from train_model.py"""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([{
            "Transaction_ID": 431438.0,
            "Customer_ID": 24239.0,
            "Transaction_Amount (in Million)": 6.0,
            "Transaction_Time": "10:54",
            "Transaction_Date": "2025-03-08",
            "Transaction_Type": "POS",
            "Merchant_ID": 97028.0,
            "Merchant_Category": "ATM",
            "Transaction_Location": "Singapore",
            "Customer_Home_Location": "Lahore",
            "Distance_From_Home": 466.0,
            "Device_ID": 363229.0,
            "IP_Address": "231.92.159.84",
            "Card_Type": "Credit",
            "Account_Balance (in Million)": 30.0,
            "Daily_Transaction_Count": 4.0,
            "Weekly_Transaction_Count": 17.0,
            "Avg_Transaction_Amount (in Million)": 2.0,
            "Max_Transaction_Last_24h (in Million)": 4.0,
            "Is_International_Transaction": "Yes",
            "Is_New_Merchant": "Yes",
            "Failed_Transaction_Count": 0.0,
            "Unusual_Time_Transaction": "No",
            "Previous_Fraud_Count": 1.0,
            "Fraud_Label": "Normal",
        }])

    def test_engineer_features_adds_hour(self, sample_df):
        from scripts.train_model import engineer_features
        result = engineer_features(sample_df.copy())
        assert "Transaction_Hour" in result.columns
        assert result["Transaction_Hour"].iloc[0] == 10

    def test_engineer_features_adds_dayofweek(self, sample_df):
        from scripts.train_model import engineer_features
        result = engineer_features(sample_df.copy())
        assert "Transaction_DayOfWeek" in result.columns
        assert 0 <= result["Transaction_DayOfWeek"].iloc[0] <= 6

    def test_engineer_features_is_same_location(self, sample_df):
        from scripts.train_model import engineer_features
        result = engineer_features(sample_df.copy())
        # Singapore != Lahore → should be 0
        assert result["Is_Same_Location"].iloc[0] == 0

    def test_engineer_features_same_location(self, sample_df):
        from scripts.train_model import engineer_features
        df = sample_df.copy()
        df["Transaction_Location"] = "Lahore"
        result = engineer_features(df)
        assert result["Is_Same_Location"].iloc[0] == 1

    def test_engineer_features_encodes_yes_no(self, sample_df):
        from scripts.train_model import engineer_features
        result = engineer_features(sample_df.copy())
        assert result["Is_International_Transaction"].iloc[0] == 1
        assert result["Unusual_Time_Transaction"].iloc[0] == 0

    def test_engineer_features_encodes_target(self, sample_df):
        from scripts.train_model import engineer_features
        result = engineer_features(sample_df.copy())
        assert result["Fraud_Label"].iloc[0] == 0

    def test_engineer_features_fraud_label(self, sample_df):
        from scripts.train_model import engineer_features
        df = sample_df.copy()
        df["Fraud_Label"] = "Fraud"
        result = engineer_features(df)
        assert result["Fraud_Label"].iloc[0] == 1

    def test_engineer_features_amount_to_balance_ratio(self, sample_df):
        from scripts.train_model import engineer_features
        result = engineer_features(sample_df.copy())
        expected = 6.0 / 30.0
        assert abs(result["Amount_To_Balance_Ratio"].iloc[0] - expected) < 0.001

    def test_engineer_features_drops_id_columns(self, sample_df):
        from scripts.train_model import engineer_features, load_data
        result = engineer_features(sample_df.copy())
        assert "Transaction_Time" not in result.columns
        assert "Transaction_Date" not in result.columns


# =============================================================================
# Test risk level logic
# =============================================================================

class TestRiskLevel:

    def test_low_risk(self):
        from app.model import get_risk_level
        assert get_risk_level(0.1) == "LOW"
        assert get_risk_level(0.0) == "LOW"
        assert get_risk_level(0.29) == "LOW"

    def test_medium_risk(self):
        from app.model import get_risk_level
        assert get_risk_level(0.3) == "MEDIUM"
        assert get_risk_level(0.49) == "MEDIUM"

    def test_high_risk(self):
        from app.model import get_risk_level
        assert get_risk_level(0.5) == "HIGH"
        assert get_risk_level(0.79) == "HIGH"

    def test_critical_risk(self):
        from app.model import get_risk_level
        assert get_risk_level(0.8) == "CRITICAL"
        assert get_risk_level(1.0) == "CRITICAL"


# =============================================================================
# Test data quality
# =============================================================================

class TestDataQuality:

    @pytest.fixture
    def raw_df(self):
        return pd.read_csv("data/raw/FraudShield_Banking_Data.csv")

    def test_dataset_has_expected_columns(self, raw_df):
        expected = [
            "Transaction_Amount (in Million)", "Transaction_Time",
            "Transaction_Date", "Fraud_Label",
        ]
        for col in expected:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_fraud_label_only_two_values(self, raw_df):
        valid = {"Fraud", "Normal"}
        actual = set(raw_df["Fraud_Label"].dropna().unique())
        assert actual == valid

    def test_transaction_amount_positive(self, raw_df):
        amounts = raw_df["Transaction_Amount (in Million)"].dropna()
        assert (amounts > 0).all(), "Transaction amounts should be positive"

    def test_fraud_rate_reasonable(self, raw_df):
        fraud_rate = (raw_df["Fraud_Label"] == "Fraud").mean()
        assert 0.01 < fraud_rate < 0.30, f"Unexpected fraud rate: {fraud_rate}"

    def test_no_excessive_nulls(self, raw_df):
        null_pct = raw_df.isnull().mean()
        for col, pct in null_pct.items():
            assert pct < 0.05, f"Column {col} has {pct:.1%} nulls"

    def test_dataset_minimum_size(self, raw_df):
        assert len(raw_df) >= 1000, "Dataset too small"

    def test_distance_non_negative(self, raw_df):
        distances = raw_df["Distance_From_Home"].dropna()
        assert (distances >= 0).all(), "Distance should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])