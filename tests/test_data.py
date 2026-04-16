"""
Data Quality Tests for Credit Card Fraud Detection Dataset.

Run with:
    pytest tests/test_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_PATH = Path("data/raw/fraudTrain.csv")
TEST_PATH  = Path("data/raw/fraudTest.csv")


@pytest.fixture(scope="module")
def train_df():
    if not TRAIN_PATH.exists():
        pytest.skip(f"Dataset not found at {TRAIN_PATH}")
    return pd.read_csv(TRAIN_PATH, index_col=0)


@pytest.fixture(scope="module")
def test_df():
    if not TEST_PATH.exists():
        pytest.skip(f"Dataset not found at {TEST_PATH}")
    return pd.read_csv(TEST_PATH, index_col=0)


# =============================================================================
# Schema Tests
# =============================================================================

class TestSchema:

    REQUIRED_COLUMNS = [
        "trans_date_trans_time", "amt", "category", "gender",
        "city_pop", "dob", "lat", "long", "merch_lat", "merch_long",
        "state", "is_fraud",
    ]

    def test_train_has_required_columns(self, train_df):
        for col in self.REQUIRED_COLUMNS:
            assert col in train_df.columns, f"Missing column in train: {col}"

    def test_test_has_required_columns(self, test_df):
        for col in self.REQUIRED_COLUMNS:
            assert col in test_df.columns, f"Missing column in test: {col}"

    def test_train_and_test_same_columns(self, train_df, test_df):
        assert set(train_df.columns) == set(test_df.columns)


# =============================================================================
# Target Tests
# =============================================================================

class TestTarget:

    def test_train_target_binary(self, train_df):
        assert set(train_df["is_fraud"].unique()).issubset({0, 1})

    def test_test_target_binary(self, test_df):
        assert set(test_df["is_fraud"].unique()).issubset({0, 1})

    def test_train_fraud_rate_reasonable(self, train_df):
        rate = train_df["is_fraud"].mean()
        assert 0.001 < rate < 0.10, f"Unexpected fraud rate: {rate:.4f}"

    def test_test_fraud_rate_reasonable(self, test_df):
        rate = test_df["is_fraud"].mean()
        assert 0.001 < rate < 0.10, f"Unexpected fraud rate: {rate:.4f}"

    def test_both_classes_in_train(self, train_df):
        counts = train_df["is_fraud"].value_counts()
        assert counts.get(0, 0) > 0
        assert counts.get(1, 0) > 0

    def test_sufficient_fraud_samples(self, train_df):
        assert train_df["is_fraud"].sum() >= 100


# =============================================================================
# Value Range Tests
# =============================================================================

class TestValueRanges:

    def test_amt_positive(self, train_df):
        amounts = train_df["amt"].dropna()
        assert (amounts > 0).all(), "Transaction amounts must be positive"

    def test_city_pop_non_negative(self, train_df):
        assert (train_df["city_pop"].dropna() >= 0).all()

    def test_gender_valid_values(self, train_df):
        valid = {"M", "F"}
        actual = set(train_df["gender"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected gender values: {actual}"

    def test_category_known_values(self, train_df):
        known = {
            "misc_net", "grocery_pos", "entertainment", "gas_transport",
            "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
            "food_dining", "personal_care", "health_fitness",
            "travel", "kids_pets", "home"
        }
        actual = set(train_df["category"].dropna().unique())
        overlap = actual & known
        assert len(overlap) > 0, "No known categories found"

    def test_trans_date_parseable(self, train_df):
        dates = pd.to_datetime(train_df["trans_date_trans_time"],
                               dayfirst=False, errors="coerce")
        valid_rate = dates.notna().mean()
        assert valid_rate > 0.95, f"Only {valid_rate:.1%} dates are valid"

    def test_dob_parseable(self, train_df):
        # Try dayfirst=True first, then fallback to mixed format
        dobs = pd.to_datetime(train_df["dob"], dayfirst=True, errors="coerce")
        if dobs.notna().mean() < 0.95:
            dobs = pd.to_datetime(train_df["dob"], format="mixed", errors="coerce")
        valid_rate = dobs.notna().mean()
        assert valid_rate > 0.90, f"Only {valid_rate:.1%} dob values are parseable"


# =============================================================================
# Missing Value Tests
# =============================================================================

class TestMissingValues:

    def test_no_nulls_in_target(self, train_df):
        assert train_df["is_fraud"].isna().sum() == 0

    def test_no_nulls_in_amt(self, train_df):
        assert train_df["amt"].isna().sum() == 0

    def test_no_column_excessive_nulls(self, train_df):
        null_pct = train_df.isnull().mean()
        for col, pct in null_pct.items():
            assert pct < 0.05, f"Column '{col}' has {pct:.1%} nulls"


# =============================================================================
# Dataset Size Tests
# =============================================================================

class TestDatasetSize:

    def test_train_minimum_size(self, train_df):
        assert len(train_df) >= 10000, f"Train too small: {len(train_df)}"

    def test_test_minimum_size(self, test_df):
        assert len(test_df) >= 1000, f"Test too small: {len(test_df)}"

    def test_train_larger_than_test(self, train_df, test_df):
        assert len(train_df) > len(test_df)


# =============================================================================
# Engineered Feature Tests
# =============================================================================

class TestEngineeredFeatures:

    @pytest.fixture(scope="class")
    def engineered_df(self, train_df):
        from scripts.train_model import engineer_features
        df, _ = engineer_features(train_df.head(1000).copy())
        return df

    def test_hour_range(self, engineered_df):
        hours = engineered_df["hour"].dropna()
        assert (hours >= 0).all() and (hours <= 23).all()

    def test_day_of_week_range(self, engineered_df):
        days = engineered_df["day_of_week"].dropna()
        assert (days >= 0).all() and (days <= 6).all()

    def test_is_night_binary(self, engineered_df):
        assert set(engineered_df["is_night"].unique()).issubset({0, 1})

    def test_is_weekend_binary(self, engineered_df):
        assert set(engineered_df["is_weekend"].unique()).issubset({0, 1})

    def test_age_positive(self, engineered_df):
        assert (engineered_df["age"].dropna() > 0).all()

    def test_distance_non_negative(self, engineered_df):
        assert (engineered_df["distance"].dropna() >= 0).all()

    def test_amt_vs_category_mean_positive(self, engineered_df):
        assert (engineered_df["amt_vs_category_mean"].dropna() > 0).all()

    def test_no_inf_values(self, engineered_df):
        numeric = engineered_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])