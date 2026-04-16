# FraudShield — Real-Time Credit Card Fraud Detection

![CI](https://github.com/your-org/fraud-detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Model](https://img.shields.io/badge/model-GBM-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)

An end-to-end ML system for real-time credit card fraud detection. Built with scikit-learn GBM, FastAPI, MLflow, Prometheus, and Grafana.

**Model Performance:**
- PR-AUC: **0.821** | F1: **0.784** | Precision: **0.847** | Recall: **0.730**

## Features

- 🔍 **Real-time fraud scoring** with probability and risk level (LOW/MEDIUM/HIGH/CRITICAL)
- 🤖 **GBM model** with manual oversampling — PR-AUC 0.82 on 555K test transactions
- 🎯 **Threshold tuning** — optimal F1 threshold found automatically
- 🔬 **Model comparison** — LogisticRegression, RandomForest, GBM benchmarked
- 📊 **MLflow experiment tracking** — local (`mlruns/`)
- 📈 **Prometheus + Grafana** monitoring with fraud-specific dashboards
- 🧠 **SHAP explainability** + **Fairness analysis** (gender, category, state)
- 🐳 **Fully containerized** with Docker Compose
- 🔄 **GitHub Actions CI/CD**

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
│   ├── train_model.py        # Compare, tune & save best model
│   ├── evaluate_model.py     # Evaluation plots
│   ├── responsible_ai.py     # SHAP + fairness analysis
│   └── load_test.py          # Load testing
├── tests/
│   ├── test_api.py           # API integration tests
│   ├── test_model.py         # Model unit tests
│   └── test_data.py          # Data quality tests
├── data/
│   └── raw/                  # fraudTrain.csv + fraudTest.csv
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

## Dataset

Download from Kaggle:
**https://www.kaggle.com/datasets/kartik2112/fraud-detection/data**

Place both files at:
```
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

Dataset info:
- **Train:** ~1.3M transactions | Fraud rate: 0.58%
- **Test:**  ~555K transactions | Fraud rate: 0.39%
- **Features:** transaction amount, merchant category, location coordinates, cardholder demographics
- **Target:** `is_fraud` (0 = normal, 1 = fraud)

## ML Pipeline

```bash
# Train model (compare 3 models → save best by PR-AUC)
python scripts/train_model.py

# View MLflow results
mlflow ui --backend-store-uri mlruns --port 5000
# Open http://localhost:5000

# Evaluate
python scripts/evaluate_model.py
# → Saves reports/confusion_matrix.png, roc_curve.png, etc.

# Responsible AI
python scripts/responsible_ai.py
# → Saves reports/shap_importance.png, fairness_analysis.png
```

> No separate preprocess step — feature engineering runs automatically inside `train_model.py`.

## Prediction Flow

```
API Request (JSON)
        ↓
app/model.py._preprocess()
  ├── Extract hour, day_of_week, month, is_night, is_weekend from trans_date_trans_time
  ├── Calculate age from dob
  ├── Calculate distance (haversine: cardholder lat/long → merchant lat/long)
  ├── Compute amt_vs_category_mean ratio
  ├── LabelEncode category and state (using encoders from pipeline.pkl)
  └── Encode gender (M=1, F=0)
        ↓
numpy array (13 features)
        ↓
fraud_model.pkl.predict_proba()
        ↓
fraud_probability = 0.92
        ↓
probability >= best_threshold (from pipeline.pkl → 0.851)
        ↓
is_fraud = True → risk_level = "CRITICAL"
        ↓
JSON Response
```

> `pipeline.pkl` and `fraud_model.pkl` must always be trained together.

## Deployment

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

```bash
# Train model inside container
docker-compose exec api python scripts/train_model.py

docker-compose up -d              # Start all
docker-compose down               # Stop all
docker-compose logs -f api        # View logs
docker-compose ps                 # Check status
```

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
    "trans_date_trans_time": "2019-06-21 12:14:00",
    "amt": 1500.00,
    "category": "shopping_net",
    "gender": "M",
    "city_pop": 333497,
    "dob": "19/3/68",
    "lat": 33.986,
    "long": -81.200,
    "merch_lat": 34.421,
    "merch_long": -82.611,
    "state": "SC"
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
    "trans_date_trans_time": "2019-06-21 14:30:00",
    "amt": 29.84,
    "category": "personal_care",
    "gender": "F",
    "city_pop": 302,
    "dob": "17/1/90",
    "lat": 40.320,
    "long": -110.436,
    "merch_lat": 39.450,
    "merch_long": -109.960,
    "state": "UT"
})
print(response.json())
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=app --cov-report=html
```

## Load Testing

```bash
python scripts/load_test.py --duration 60 --workers 10
python scripts/load_test.py --spike
```

## Responsible AI

```bash
python scripts/responsible_ai.py
```

Generates in `reports/`:
- `shap_importance.png` — Top features driving fraud predictions
- `shap_beeswarm.png` — Feature impact distribution
- `fairness_analysis.png` — Recall by gender, category, state

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
| Dataset not found | Download from Kaggle link above, place in `data/raw/` |
| MLflow 403 error | Use local: `mlflow ui --backend-store-uri mlruns --port 5000` |
| Grafana no data | Check `localhost:9090/targets` — API must be UP |
| Docker build fails | Use `python:3.10` (not `slim`) in Dockerfile |
| numpy compatibility | Ensure `numpy<2` in requirements.txt |