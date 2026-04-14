"""
Integration tests for FraudShield Detection API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_TRANSACTION = {
    "transaction_amount": 6.0,
    "transaction_time": "10:54",
    "transaction_date": "2025-03-08",
    "transaction_type": "POS",
    "merchant_category": "Retail",
    "transaction_location": "Singapore",
    "customer_home_location": "Lahore",
    "distance_from_home": 466.0,
    "card_type": "Credit",
    "account_balance": 30.0,
    "daily_transaction_count": 4,
    "weekly_transaction_count": 17,
    "avg_transaction_amount": 2.0,
    "max_transaction_last_24h": 4.0,
    "is_international_transaction": True,
    "is_new_merchant": True,
    "failed_transaction_count": 0,
    "unusual_time_transaction": False,
    "previous_fraud_count": 1,
}

HIGH_RISK_TRANSACTION = {
    "transaction_amount": 50.0,
    "transaction_time": "03:22",
    "transaction_date": "2025-03-08",
    "transaction_type": "Online",
    "merchant_category": "Electronics",
    "transaction_location": "Tokyo",
    "customer_home_location": "Lagos",
    "distance_from_home": 12000.0,
    "card_type": "Credit",
    "account_balance": 2.0,
    "daily_transaction_count": 15,
    "weekly_transaction_count": 50,
    "avg_transaction_amount": 1.0,
    "max_transaction_last_24h": 50.0,
    "is_international_transaction": True,
    "is_new_merchant": True,
    "failed_transaction_count": 5,
    "unusual_time_transaction": True,
    "previous_fraud_count": 1,
}


@pytest.fixture(autouse=True)
def mock_model():
    """Mock the fraud detection model for all tests."""
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.predict.return_value = (False, 0.12, "LOW", 8.5)
    mock.get_info.return_value = {
        "model_version": "1.0.0",
        "model_type": "XGBoost",
        "is_loaded": True,
        "features_count": 21,
        "fraud_threshold": 0.5,
    }
    with patch("app.main.model", mock):
        yield mock


# =============================================================================
# Health & Info Endpoints
# =============================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data

    def test_health_model_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_when_model_not_loaded(self):
        with patch("app.main.model", None):
            response = client.get("/health")
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestRootEndpoint:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_required_fields(self):
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "metrics" in data


class TestMetricsEndpoint:

    def test_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self):
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")

    def test_metrics_contains_http_requests(self):
        client.get("/health")
        response = client.get("/metrics")
        assert "http_requests_total" in response.text


# =============================================================================
# Prediction Endpoint
# =============================================================================

class TestPredictEndpoint:

    def test_predict_valid_transaction_returns_200(self):
        response = client.post("/predict", json=VALID_TRANSACTION)
        assert response.status_code == 200

    def test_predict_response_format(self):
        response = client.post("/predict", json=VALID_TRANSACTION)
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_predict_probability_in_range(self):
        response = client.post("/predict", json=VALID_TRANSACTION)
        data = response.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_risk_level_valid(self):
        response = client.post("/predict", json=VALID_TRANSACTION)
        data = response.json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_predict_is_fraud_boolean(self):
        response = client.post("/predict", json=VALID_TRANSACTION)
        data = response.json()
        assert isinstance(data["is_fraud"], bool)

    def test_predict_fraud_transaction(self, mock_model):
        mock_model.predict.return_value = (True, 0.92, "CRITICAL", 9.1)
        response = client.post("/predict", json=HIGH_RISK_TRANSACTION)
        data = response.json()
        assert data["is_fraud"] is True
        assert data["risk_level"] == "CRITICAL"
        assert data["fraud_probability"] > 0.5

    def test_predict_missing_required_field_returns_422(self):
        incomplete = {k: v for k, v in VALID_TRANSACTION.items() if k != "transaction_amount"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_negative_amount_returns_422(self):
        bad = {**VALID_TRANSACTION, "transaction_amount": -1.0}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_model_not_loaded_returns_503(self):
        with patch("app.main.model", None):
            response = client.post("/predict", json=VALID_TRANSACTION)
            assert response.status_code == 503


# =============================================================================
# Model Info Endpoint
# =============================================================================

class TestModelInfoEndpoint:

    def test_model_info_returns_200(self):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_format(self):
        response = client.get("/model/info")
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "is_loaded" in data
        assert "features_count" in data
        assert "fraud_threshold" in data

    def test_model_info_threshold_in_range(self):
        response = client.get("/model/info")
        data = response.json()
        assert 0.0 < data["fraud_threshold"] < 1.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:

    def test_predict_zero_balance(self):
        tx = {**VALID_TRANSACTION, "account_balance": 0.0}
        response = client.post("/predict", json=tx)
        assert response.status_code == 200

    def test_predict_very_high_amount(self):
        tx = {**VALID_TRANSACTION, "transaction_amount": 999.0}
        response = client.post("/predict", json=tx)
        assert response.status_code == 200

    def test_predict_zero_distance(self):
        tx = {**VALID_TRANSACTION, "distance_from_home": 0.0}
        response = client.post("/predict", json=tx)
        assert response.status_code == 200

    def test_predict_same_location(self):
        tx = {
            **VALID_TRANSACTION,
            "transaction_location": "Singapore",
            "customer_home_location": "Singapore",
            "is_international_transaction": False,
        }
        response = client.post("/predict", json=tx)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])