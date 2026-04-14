# FraudShield — Real-Time Fraud Detection System

![CI](https://github.com/your-org/fraud-detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)

An end-to-end ML system for real-time financial transaction fraud detection. Built with XGBoost, FastAPI, MLflow, Prometheus, and Grafana.

## Features

- 🔍 **Real-time fraud scoring** with probability and risk level (LOW/MEDIUM/HIGH/CRITICAL)
- 🤖 **XGBoost model** trained on 50,000 transactions with SMOTE for class imbalance
- 📊 **MLflow experiment tracking** for model comparison and reproducibility
- 📈 **Prometheus + Grafana** monitoring with fraud-specific metrics and dashboards
- 🧠 **SHAP explainability** — understand why a transaction was flagged
- ⚖️ **Fairness analysis** across card type, transaction type, and geography
- 🐳 **Fully containerized** with Docker Compose (API + MLflow + Prometheus + Grafana)
- 🔄 **GitHub Actions CI/CD** with lint, tests, and Docker build on every push

## Project Structure

```
fraud-detection/
├── app/
│   ├── main.py             # FastAPI application
│   ├── model.py            # Model wrapper + preprocessing
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── metrics.py          # Prometheus metrics definitions
│   ├── middleware.py       # HTTP metrics middleware
│   └── config.py           # Configuration
├── scripts/
│   ├── train_model.py      # Training + MLflow tracking
│   ├── responsible_ai.py   # SHAP + fairness analysis
│   └── load_test.py        # Load testing
├── tests/
│   ├── test_api.py         # API integration tests
│   └── test_model.py       # Unit + data quality tests
├── data/raw/               # Raw dataset
├── models/                 # Saved model files
├── prometheus/             # Prometheus config + alert rules
├── grafana/                # Grafana dashboards + provisioning
├── .github/workflows/      # GitHub Actions CI/CD
├── docker-compose.yml
├── Dockerfile
├── ARCHITECTURE.md
└── CONTRIBUTING.md
```

## Quick Start

### Prerequisites
- Python 3.10+
- Docker + Docker Compose v2

### 1. Clone and install

```bash
git clone <your-repo-url>
cd fraud-detection

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add dataset

Place the dataset file at:
```
data/raw/FraudShield_Banking_Data.csv
```

### 3. Start MLflow and train model

```bash
# Start MLflow tracking server
docker compose up -d mlflow

# Train model (tracks experiments to MLflow)
python scripts/train_model.py
```

View experiments at **http://localhost:5000**

### 4. Start full stack

```bash
docker compose up --build -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | — |
| API Docs | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin/admin |

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "model_loaded": true, "model_version": "1.0.0"}
```

### Predict Fraud

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 45.0,
    "transaction_time": "03:15",
    "transaction_date": "2025-03-08",
    "transaction_type": "Online",
    "merchant_category": "Electronics",
    "transaction_location": "Tokyo",
    "customer_home_location": "Lagos",
    "distance_from_home": 13000.0,
    "card_type": "Credit",
    "account_balance": 1.5,
    "daily_transaction_count": 20,
    "weekly_transaction_count": 60,
    "avg_transaction_amount": 1.0,
    "max_transaction_last_24h": 45.0,
    "is_international_transaction": true,
    "is_new_merchant": true,
    "failed_transaction_count": 4,
    "unusual_time_transaction": true,
    "previous_fraud_count": 1
  }'
```
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9231,
  "risk_level": "CRITICAL",
  "model_version": "1.0.0",
  "latency_ms": 8.3
}
```

### Python Example

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "transaction_amount": 2.0,
    "transaction_time": "14:30",
    "transaction_date": "2025-03-08",
    "transaction_type": "POS",
    "merchant_category": "Retail",
    "transaction_location": "Singapore",
    "customer_home_location": "Singapore",
    "distance_from_home": 5.0,
    "card_type": "Credit",
    "account_balance": 50.0,
    "daily_transaction_count": 2,
    "weekly_transaction_count": 8,
    "avg_transaction_amount": 2.5,
    "max_transaction_last_24h": 3.0,
    "is_international_transaction": False,
    "is_new_merchant": False,
    "failed_transaction_count": 0,
    "unusual_time_transaction": False,
    "previous_fraud_count": 0,
})
print(response.json())
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=html

# API tests only
pytest tests/test_api.py -v

# Data quality tests only
pytest tests/test_model.py::TestDataQuality -v
```

## Load Testing

```bash
# Constant load
python scripts/load_test.py --duration 60 --workers 10

# Spike test
python scripts/load_test.py --spike
```

## Responsible AI

```bash
# Run SHAP + fairness analysis (model must be trained first)
python scripts/responsible_ai.py
```

Generates:
- `reports/shap_importance.png` — Top features driving fraud predictions
- `reports/shap_beeswarm.png` — Feature impact distribution
- `reports/fairness_analysis.png` — Recall comparison across groups

## Monitoring

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_predictions_total` | Counter | Predictions by result (fraud/normal) |
| `fraud_detection_rate` | Gauge | Rolling fraud rate |
| `fraud_probability_distribution` | Histogram | Score distribution |
| `high_risk_transactions_total` | Counter | HIGH/CRITICAL risk count |
| `fraud_prediction_duration_seconds` | Histogram | Inference latency |
| `http_requests_total` | Counter | API request count |

### Alert Rules (8 total)

- `ModelNotLoaded` — Critical when model is down
- `HighFraudRate` — Fraud rate > 15%
- `HighPredictionLatency` — P95 > 100ms
- `HighRiskTransactionSpike` — Spike in HIGH/CRITICAL transactions
- `PredictionErrors` — Error rate > 0.1/s
- `HighErrorRate` — HTTP 5xx > 10%
- `HighRequestLatency` — HTTP P95 > 1s
- `APIDown` — API unreachable

## Docker Commands

```bash
# Start all services
docker compose up -d

# Rebuild after code changes
docker compose up --build -d

# Train model inside container
docker compose exec api python scripts/train_model.py

# View API logs
docker compose logs -f api

# Stop everything
docker compose down
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not loaded` (503) | Run `python scripts/train_model.py` first |
| Grafana no data | Check `localhost:9090/targets` — API must be UP |
| Docker build fails | Use `python:3.10` (not `slim`) in Dockerfile |
| numpy compatibility error | Ensure `numpy<2` in requirements.txt |
| MLflow connection error | Run `docker compose up -d mlflow` first |