"""
Integration tests for FraudShield Credit Card Fraud Detection API.

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# --- Test transactions ---

NORMAL_TRANSACTION = {
    "trans_date_trans_time": "2019-01-01 14:30:00",
    "amt": 4.97,
    "category": "misc_net",
    "gender": "F",
    "city_pop": 3495,
    "dob": "9/3/88",
    "lat": 36.0788,
    "long": -81.1781,
    "merch_lat": 36.011,
    "merch_long": -82.048,
    "state": "NC",
}

FRAUD_TRANSACTION = {
    "trans_date_trans_time": "2019-01-02 01:06:00",
    "amt": 281.06,
    "category": "grocery_pos",
    "gender": "M",
    "city_pop": 885,
    "dob": "15/9/88",
    "lat": 35.9946,
    "long": -81.7266,
    "merch_lat": 36.430,
    "merch_long": -81.179,
    "state": "NC",
}


@pytest.fixture(autouse=True)
def mock_model():
    """Mock the model for all tests — no .pkl file needed."""
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.predict.return_value = (False, 0.12, "LOW", 8.5)
    mock.get_info.return_value = {
        "model_version":   "1.0.0",
        "model_type":      "GBM_oversampled",
        "is_loaded":       True,
        "features_count":  13,
        "fraud_threshold": 0.851,
    }
    with patch("app.main.model", mock):
        yield mock


# =============================================================================
# Health & Info
# =============================================================================

class TestHealthEndpoint:

    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_response_format(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data

    def test_healthy_when_model_loaded(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_unhealthy_when_model_not_loaded(self):
        with patch("app.main.model", None):
            data = client.get("/health").json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestRootEndpoint:

    def test_returns_200(self):
        assert client.get("/").status_code == 200

    def test_contains_required_fields(self):
        data = client.get("/").json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "metrics" in data


class TestMetricsEndpoint:

    def test_returns_200(self):
        assert client.get("/metrics").status_code == 200

    def test_prometheus_format(self):
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")

    def test_contains_http_requests(self):
        client.get("/health")
        assert "http_requests_total" in client.get("/metrics").text


# =============================================================================
# Prediction Endpoint
# =============================================================================

class TestPredictEndpoint:

    def test_normal_transaction_returns_200(self):
        assert client.post("/predict", json=NORMAL_TRANSACTION).status_code == 200

    def test_response_format(self):
        data = client.post("/predict", json=NORMAL_TRANSACTION).json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_probability_in_range(self):
        data = client.post("/predict", json=NORMAL_TRANSACTION).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_risk_level_valid(self):
        data = client.post("/predict", json=NORMAL_TRANSACTION).json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_is_fraud_boolean(self):
        data = client.post("/predict", json=NORMAL_TRANSACTION).json()
        assert isinstance(data["is_fraud"], bool)

    def test_fraud_transaction_detected(self, mock_model):
        mock_model.predict.return_value = (True, 0.93, "CRITICAL", 9.1)
        data = client.post("/predict", json=FRAUD_TRANSACTION).json()
        assert data["is_fraud"] is True
        assert data["risk_level"] == "CRITICAL"
        assert data["fraud_probability"] > 0.5

    def test_missing_amt_returns_422(self):
        bad = {k: v for k, v in NORMAL_TRANSACTION.items() if k != "amt"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_missing_category_returns_422(self):
        bad = {k: v for k, v in NORMAL_TRANSACTION.items() if k != "category"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_negative_amt_returns_422(self):
        bad = {**NORMAL_TRANSACTION, "amt": -10.0}
        assert client.post("/predict", json=bad).status_code == 422

    def test_empty_body_returns_422(self):
        assert client.post("/predict", json={}).status_code == 422

    def test_model_not_loaded_returns_503(self):
        with patch("app.main.model", None):
            assert client.post("/predict", json=NORMAL_TRANSACTION).status_code == 503


# =============================================================================
# Model Info
# =============================================================================

class TestModelInfoEndpoint:

    def test_returns_200(self):
        assert client.get("/model/info").status_code == 200

    def test_response_format(self):
        data = client.get("/model/info").json()
        assert "model_version" in data
        assert "model_type" in data
        assert "is_loaded" in data
        assert "features_count" in data
        assert "fraud_threshold" in data

    def test_threshold_in_valid_range(self):
        data = client.get("/model/info").json()
        assert 0.0 < data["fraud_threshold"] < 1.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:

    def test_very_large_amount(self):
        tx = {**NORMAL_TRANSACTION, "amt": 9999.99}
        assert client.post("/predict", json=tx).status_code == 200

    def test_midnight_transaction(self):
        tx = {**NORMAL_TRANSACTION, "trans_date_trans_time": "2019-01-01 00:00:00"}
        assert client.post("/predict", json=tx).status_code == 200

    def test_3am_transaction(self):
        tx = {**NORMAL_TRANSACTION, "trans_date_trans_time": "2019-01-01 03:00:00"}
        assert client.post("/predict", json=tx).status_code == 200

    def test_zero_city_pop(self):
        tx = {**NORMAL_TRANSACTION, "city_pop": 0}
        assert client.post("/predict", json=tx).status_code == 200

    def test_all_categories(self):
        categories = [
            "misc_net", "grocery_pos", "entertainment", "gas_transport",
            "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
            "food_dining", "personal_care", "health_fitness", "travel",
            "kids_pets", "home"
        ]
        for cat in categories:
            tx = {**NORMAL_TRANSACTION, "category": cat}
            assert client.post("/predict", json=tx).status_code == 200

    def test_both_genders(self):
        for gender in ["M", "F"]:
            tx = {**NORMAL_TRANSACTION, "gender": gender}
            assert client.post("/predict", json=tx).status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])