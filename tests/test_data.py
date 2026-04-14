"""
Data Quality Tests for FraudShield Dataset.

Tests validate:
- Dataset schema and column presence
- Data types and value ranges
- Class distribution
- Missing value thresholds
- Feature consistency

Run with:
    pytest tests/test_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/FraudShield_Banking_Data.csv")


@pytest.fixture(scope="module")
def raw_df():
    """Load raw dataset once for all tests."""
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    """Load engineered dataset."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.train_model import load_data, engineer_features
    df = load_data()
    return engineer_features(df)


# =============================================================================
# Schema Tests
# =============================================================================

class TestSchema:
    """Validate dataset has expected columns and types."""

    REQUIRED_COLUMNS = [
        "Transaction_Amount (in Million)",
        "Transaction_Time",
        "Transaction_Date",
        "Transaction_Type",
        "Merchant_Category",
        "Transaction_Location",
        "Customer_Home_Location",
        "Distance_From_Home",
        "Card_Type",
        "Account_Balance (in Million)",
        "Daily_Transaction_Count",
        "Weekly_Transaction_Count",
        "Avg_Transaction_Amount (in Million)",
        "Max_Transaction_Last_24h (in Million)",
        "Is_International_Transaction",
        "Is_New_Merchant",
        "Failed_Transaction_Count",
        "Unusual_Time_Transaction",
        "Previous_Fraud_Count",
        "Fraud_Label",
    ]

    def test_required_columns_exist(self, raw_df):
        """All required columns must be present."""
        for col in self.REQUIRED_COLUMNS:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_no_extra_unexpected_columns(self, raw_df):
        """Dataset should have a reasonable number of columns."""
        assert len(raw_df.columns) <= 30, f"Too many columns: {len(raw_df.columns)}"

    def test_fraud_label_column_exists(self, raw_df):
        assert "Fraud_Label" in raw_df.columns


# =============================================================================
# Value Range Tests
# =============================================================================

class TestValueRanges:
    """Validate value ranges for numerical features."""

    def test_transaction_amount_positive(self, raw_df):
        amounts = raw_df["Transaction_Amount (in Million)"].dropna()
        assert (amounts > 0).all(), "Transaction amounts must be positive"

    def test_account_balance_non_negative(self, raw_df):
        balances = raw_df["Account_Balance (in Million)"].dropna()
        assert (balances >= 0).all(), "Account balance must be non-negative"

    def test_distance_non_negative(self, raw_df):
        distances = raw_df["Distance_From_Home"].dropna()
        assert (distances >= 0).all(), "Distance must be non-negative"

    def test_daily_count_non_negative(self, raw_df):
        counts = raw_df["Daily_Transaction_Count"].dropna()
        assert (counts >= 0).all()

    def test_weekly_count_non_negative(self, raw_df):
        counts = raw_df["Weekly_Transaction_Count"].dropna()
        assert (counts >= 0).all()

    def test_failed_count_non_negative(self, raw_df):
        counts = raw_df["Failed_Transaction_Count"].dropna()
        assert (counts >= 0).all()

    def test_previous_fraud_count_non_negative(self, raw_df):
        counts = raw_df["Previous_Fraud_Count"].dropna()
        assert (counts >= 0).all()

    def test_transaction_time_format(self, raw_df):
        """Transaction time should be in HH:MM format."""
        times = raw_df["Transaction_Time"].dropna()
        valid = times.str.match(r"^\d{1,2}:\d{2}$")
        assert valid.mean() > 0.95, "Most transaction times should match HH:MM format"

    def test_transaction_date_parseable(self, raw_df):
        """Transaction dates should be parseable."""
        dates = pd.to_datetime(raw_df["Transaction_Date"], errors="coerce")
        valid_rate = dates.notna().mean()
        assert valid_rate > 0.95, f"Only {valid_rate:.1%} dates are valid"

    def test_weekly_count_gte_daily_count(self, raw_df):
        """Weekly count should be >= daily count."""
        df = raw_df.dropna(subset=["Daily_Transaction_Count", "Weekly_Transaction_Count"])
        valid = (df["Weekly_Transaction_Count"] >= df["Daily_Transaction_Count"]).mean()
        assert valid > 0.95, f"Weekly count should be >= daily count in most rows"


# =============================================================================
# Class Distribution Tests
# =============================================================================

class TestClassDistribution:
    """Validate fraud label distribution."""

    def test_fraud_label_only_two_values(self, raw_df):
        valid_values = {"Fraud", "Normal"}
        actual = set(raw_df["Fraud_Label"].dropna().unique())
        assert actual == valid_values, f"Unexpected values: {actual}"

    def test_fraud_rate_in_reasonable_range(self, raw_df):
        fraud_rate = (raw_df["Fraud_Label"] == "Fraud").mean()
        assert 0.01 < fraud_rate < 0.50, f"Unexpected fraud rate: {fraud_rate:.2%}"

    def test_normal_class_majority(self, raw_df):
        """Normal transactions should be majority class."""
        normal_rate = (raw_df["Fraud_Label"] == "Normal").mean()
        assert normal_rate > 0.5, "Normal transactions should be majority"

    def test_both_classes_have_sufficient_samples(self, raw_df):
        counts = raw_df["Fraud_Label"].value_counts()
        assert counts.get("Fraud", 0) >= 100, "Need at least 100 fraud samples"
        assert counts.get("Normal", 0) >= 100, "Need at least 100 normal samples"


# =============================================================================
# Missing Value Tests
# =============================================================================

class TestMissingValues:
    """Validate missing values are within acceptable thresholds."""

    def test_fraud_label_low_nulls(self, raw_df):
        null_rate = raw_df["Fraud_Label"].isna().mean()
        assert null_rate < 0.01, f"Fraud_Label has {null_rate:.1%} nulls — too high"

    def test_no_column_has_excessive_nulls(self, raw_df):
        """No column should have more than 5% missing values."""
        null_pct = raw_df.isnull().mean()
        for col, pct in null_pct.items():
            assert pct < 0.05, f"Column '{col}' has {pct:.1%} nulls (> 5% threshold)"

    def test_transaction_amount_low_nulls(self, raw_df):
        null_rate = raw_df["Transaction_Amount (in Million)"].isna().mean()
        assert null_rate < 0.01


# =============================================================================
# Categorical Value Tests
# =============================================================================

class TestCategoricalValues:
    """Validate categorical columns have expected values."""

    def test_card_type_valid_values(self, raw_df):
        valid = {"Credit", "Debit"}
        actual = set(raw_df["Card_Type"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected card types: {actual - valid}"

    def test_is_international_valid_values(self, raw_df):
        valid = {"Yes", "No"}
        actual = set(raw_df["Is_International_Transaction"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected values: {actual - valid}"

    def test_is_new_merchant_valid_values(self, raw_df):
        valid = {"Yes", "No"}
        actual = set(raw_df["Is_New_Merchant"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected values: {actual - valid}"

    def test_unusual_time_valid_values(self, raw_df):
        valid = {"Yes", "No"}
        actual = set(raw_df["Unusual_Time_Transaction"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected values: {actual - valid}"

    def test_transaction_type_has_known_values(self, raw_df):
        known = {"POS", "ATM", "Online", "Transfer"}
        actual = set(raw_df["Transaction_Type"].dropna().unique())
        overlap = actual & known
        assert len(overlap) > 0, f"No known transaction types found: {actual}"


# =============================================================================
# Dataset Size Tests
# =============================================================================

class TestDatasetSize:

    def test_minimum_rows(self, raw_df):
        assert len(raw_df) >= 1000, f"Dataset too small: {len(raw_df)} rows"

    def test_expected_row_count(self, raw_df):
        assert len(raw_df) >= 10000, f"Expected at least 10K rows, got {len(raw_df)}"

    def test_minimum_columns(self, raw_df):
        assert len(raw_df.columns) >= 10, "Dataset has too few columns"


# =============================================================================
# Engineered Feature Tests
# =============================================================================

class TestEngineeredFeatures:
    """Validate features after engineering."""

    def test_transaction_hour_range(self, clean_df):
        hours = clean_df["Transaction_Hour"].dropna()
        assert (hours >= 0).all() and (hours <= 23).all(), "Hour must be 0-23"

    def test_day_of_week_range(self, clean_df):
        days = clean_df["Transaction_DayOfWeek"].dropna()
        assert (days >= 0).all() and (days <= 6).all(), "Day of week must be 0-6"

    def test_is_same_location_binary(self, clean_df):
        values = clean_df["Is_Same_Location"].dropna().unique()
        assert set(values).issubset({0, 1}), "Is_Same_Location must be 0 or 1"

    def test_fraud_label_binary(self, clean_df):
        values = clean_df["Fraud_Label"].dropna().unique()
        assert set(values).issubset({0, 1}), "Fraud_Label must be 0 or 1 after encoding"

    def test_amount_to_balance_ratio_non_negative(self, clean_df):
        ratios = clean_df["Amount_To_Balance_Ratio"].dropna()
        assert (ratios >= 0).all(), "Amount/Balance ratio must be non-negative"

    def test_no_inf_values_after_engineering(self, clean_df):
        numeric = clean_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values found after feature engineering"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])