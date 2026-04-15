# FraudShield — Real-Time Fraud Detection System

![CI](https://github.com/your-org/fraud-detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Model](https://img.shields.io/badge/model-GBM%20%7C%20RandomForest-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)

An end-to-end ML system for real-time financial transaction fraud detection. Built with scikit-learn, FastAPI, MLflow, Prometheus, and Grafana.

## Features

- 🔍 **Real-time fraud scoring** with probability and risk level (LOW/MEDIUM/HIGH/CRITICAL)
- 🤖 **GBM / RandomForest** with manual oversampling and `class_weight='balanced'`
- 🎯 **Threshold tuning** — finds optimal F1 threshold automatically
- 🔬 **Model experimentation** — compare 3 models + hyperparameter tuning with RandomizedSearch
- 📊 **MLflow experiment tracking** — local (`mlruns/`) or Docker server
- 📈 **Prometheus + Grafana** monitoring with fraud-specific metrics and dashboards
- 🧠 **SHAP explainability** — understand why a transaction was flagged
- ⚖️ **Fairness analysis** across card type, transaction type, and geography
- 🐳 **Fully containerized** with Docker Compose (API + MLflow + Prometheus + Grafana)
- 🔄 **GitHub Actions CI/CD** with lint, tests, and Docker build on every push

## Project Structure

```
fraud-detection/
├── app/
│   ├── main.py               # FastAPI application
│   ├── model.py              # Model wrapper + feature engineering
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── metrics.py            # Prometheus metrics definitions
│   ├── middleware.py         # HTTP metrics middleware
│   └── config.py             # Configuration
├── scripts/
│   ├── preprocess.py         # (Optional) Save processed data to disk
│   ├── train_model.py        # Compare, tune & save best model + MLflow tracking
│   ├── evaluate_model.py     # Evaluate model + generate plots
│   ├── responsible_ai.py     # SHAP + fairness analysis
│   └── load_test.py          # Load testing
├── tests/
│   ├── test_api.py           # API integration tests
│   ├── test_model.py         # Model unit tests
│   └── test_data.py          # Data quality tests
├── data/
│   └── raw/                  # Raw dataset (CSV)
├── models/                   # Saved model files (auto-generated)
├── reports/                  # Evaluation plots (auto-generated)
├── prometheus/               # Prometheus config + alert rules
├── grafana/                  # Grafana dashboards + provisioning
├── .github/workflows/        # GitHub Actions CI/CD
├── docker-compose.yml
├── Dockerfile
├── ARCHITECTURE.md
└── CONTRIBUTING.md
```

## Prerequisites

- Python 3.11
- Docker + Docker Compose

## Installation

```bash
git clone <your-repo-url>
cd fraud-detection

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ML Pipeline

Place dataset at `data/raw/FraudShield_Banking_Data.csv`, then:

```bash
# Full pipeline: compare 3 models → tune best → save (default)
python scripts/train_model.py

# Skip tuning, save best from comparison directly (faster)
python scripts/train_model.py --skip-tune

# Custom tuning iterations
python scripts/train_model.py --n_iter 50
```

**What it does internally:**
1. Compare 3 models (LogisticRegression, RandomForest, GBM) — pick best by PR-AUC
2. Tune best model with RandomizedSearch
3. Find optimal threshold (max F1)
4. Save `models/fraud_model.pkl` + `models/pipeline.pkl`

**View MLflow results:**
```bash
mlflow ui --backend-store-uri mlruns --port 5000
# Open http://localhost:5000
```

```bash
# Evaluate
python scripts/evaluate_model.py
# → Saves reports/confusion_matrix.png, roc_curve.png, etc.

# Responsible AI
python scripts/responsible_ai.py
# → Saves reports/shap_importance.png, fairness_analysis.png
```

## Deployment

### Start full stack

```bash
docker-compose up --build -d
```

| Service    | URL                        | Credentials |
|------------|----------------------------|-------------|
| API        | http://localhost:8000      | —           |
| API Docs   | http://localhost:8000/docs | —           |
| MLflow     | http://localhost:5000      | —           |
| Prometheus | http://localhost:9090      | —           |
| Grafana    | http://localhost:3000      | admin/admin |

### Train model inside container

```bash
docker-compose exec api python scripts/train_model.py
```

### Common Docker commands

```bash
docker-compose up -d              # Start all services
docker-compose up --build -d      # Rebuild and start
docker-compose down               # Stop all services
docker-compose logs -f api        # View API logs
docker-compose ps                 # Check service status
docker-compose restart api        # Restart API only
```

## Prediction Flow

```
API Request (JSON)
        ↓
app/model.py._preprocess()
  ├── Encode categoricals using label_encoders (from pipeline.pkl)
  │   e.g. "POS" → 2, "Credit" → 0, "Electronics" → 3
  ├── Compute risk_score, amount_vs_avg, is_night, is_weekend
  └── Order features correctly (19 features)
        ↓
numpy array (19 features)
        ↓
fraud_model.pkl.predict_proba()
        ↓
fraud_probability = 0.87
        ↓
probability >= best_threshold (from pipeline.pkl)
        ↓
is_fraud = True → risk_level = "CRITICAL"
        ↓
JSON Response
```

> **Note:** `pipeline.pkl` and `fraud_model.pkl` must always be trained together.
> Never mix a new `fraud_model.pkl` with an old `pipeline.pkl` — LabelEncoder mappings may differ.

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
  "fraud_probability": 0.8741,
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

# By file
pytest tests/test_api.py -v      # API integration tests
pytest tests/test_model.py -v    # Model unit tests
pytest tests/test_data.py -v     # Data quality tests
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
python scripts/responsible_ai.py
```

Generates in `reports/`:
- `shap_importance.png` — Top features driving fraud predictions
- `shap_beeswarm.png` — Feature impact distribution
- `fairness_analysis.png` — Recall comparison across groups

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

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not loaded` (503) | Run `python scripts/train_model.py` first |
| MLflow 403 error | Use local: `mlflow ui --backend-store-uri mlruns --port 5000` |
| Grafana no data | Check `localhost:9090/targets` — API must be UP |
| Docker build fails | Use `python:3.10` (not `slim`) in Dockerfile |
| numpy compatibility error | Ensure `numpy<2` in requirements.txt |
| `experiment.py` slow | Reduce `--n_iter` to 10 for faster tuning |