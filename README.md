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

### Test Cases

**🔴 Fraud — late night, high amount**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "state": "NC"
  }'
```

**🔴 Fraud — 3am, unusual amount**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-01-02 03:05:00",
    "amt": 276.31,
    "category": "grocery_pos",
    "gender": "F",
    "city_pop": 1595797,
    "dob": "28/10/60",
    "lat": 29.44,
    "long": -98.459,
    "merch_lat": 29.273,
    "merch_long": -98.836,
    "state": "TX"
  }'
```

**🟢 Normal — daytime, small amount**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "state": "NC"
  }'
```

**🟢 Normal — business hours grocery**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-01-01 12:14:00",
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
  }'
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
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=html

# By file
pytest tests/test_api.py -v      # API integration tests (mock model)
pytest tests/test_model.py -v    # Feature engineering + risk level unit tests
pytest tests/test_data.py -v     # Data quality tests (requires dataset in data/raw/)
```

### Test Coverage

| File | What it tests |
|------|--------------|
| `test_api.py` | All endpoints (health, predict, model/info, metrics), validation errors (422), 503 when model not loaded, all 14 categories, both genders |
| `test_model.py` | `engineer_features()` — hour, is_night, is_weekend, age, distance, gender_enc, amt_vs_category_mean; haversine distance; risk level (LOW/MEDIUM/HIGH/CRITICAL) |
| `test_data.py` | Schema, target binary, fraud rate range, no excessive nulls, date/dob parseable, engineered feature validity |

> `test_api.py` uses mock model — no `.pkl` file needed. `test_data.py` requires `data/raw/fraudTrain.csv` and `fraudTest.csv`.

### Sample Test Cases

| Case | Transaction | Expected |
|------|------------|----------|
| Normal | $4.97 misc_net, 2pm NC | `is_fraud=false`, LOW |
| Fraud | $281.06 grocery, 1am NC | `is_fraud=true`, HIGH/CRITICAL |
| Edge | $9999.99 any category | 200 OK |
| Edge | 3am transaction | 200 OK |
| Invalid | Missing `amt` field | 422 |
| Invalid | Negative `amt` | 422 |

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

### Grafana Dashboard (`http://localhost:3000`)

The dashboard has 12 panels:

| Panel | Type | Query | What it shows |
|-------|------|-------|---------------|
| **Model Status** | Stat | `ml_model_loaded` | Green = model loaded, Red = model down |
| **Fraud Rate (Rolling)** | Gauge | `fraud_detection_rate * 100` | % of recent 1000 predictions flagged as fraud. Threshold: yellow >5%, red >15% |
| **Prediction Rate** | Stat | `rate(fraud_predictions_total[5m])` | How many predictions per second |
| **HIGH/CRITICAL Alerts** | Stat | `rate(high_risk_transactions_total[5m])` | Rate of high-risk transactions being detected |
| **API Error Rate** | Stat | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100` | % of requests returning 5xx errors |
| **P95 Prediction Latency** | Stat | `histogram_quantile(0.95, rate(fraud_prediction_duration_seconds_bucket[5m])) * 1000` | 95th percentile model inference time in ms |
| **Predictions Over Time** | Time series | `rate(fraud_predictions_total[5m])` by result | Normal vs Fraud prediction volume over time |
| **Fraud Probability Distribution** | Time series | `histogram_quantile(0.50/0.95/0.99, ...)` | Spread of fraud probability scores — P50, P95, P99 |
| **Request Rate by Endpoint** | Time series | `rate(http_requests_total[5m])` | Traffic per endpoint (predict, health, metrics) |
| **API Latency P50/P95/P99** | Time series | `histogram_quantile(...)` | End-to-end HTTP response time percentiles |
| **High Risk Transactions Over Time** | Time series | `rate(high_risk_transactions_total[5m])` by risk_level | HIGH vs CRITICAL risk transactions over time |
| **Status Code Distribution** | Pie chart | `sum by (status) (rate(http_requests_total[5m]))` | Breakdown of 200, 422, 503 responses |

### Alert Rules (8 total)

View active alerts at `http://localhost:9090/alerts`

| Alert | Condition | Severity | Meaning |
|-------|-----------|----------|---------|
| `ModelNotLoaded` | `ml_model_loaded == 0` for 30s | Critical | Model failed to load — all predictions returning 503 |
| `HighFraudRate` | `fraud_detection_rate > 0.15` for 2m | Critical | More than 15% of recent transactions flagged as fraud — possible attack |
| `HighPredictionLatency` | P95 > 100ms for 2m | Warning | Model inference too slow |
| `HighRiskTransactionSpike` | `rate(high_risk_transactions_total[5m]) > 5` for 1m | Warning | Sudden spike in HIGH/CRITICAL risk detections |
| `PredictionErrors` | `rate(fraud_prediction_errors_total[5m]) > 0.1` for 1m | Critical | Model throwing errors on valid requests |
| `HighErrorRate` | HTTP 5xx > 10% for 1m | Critical | Too many server errors |
| `HighRequestLatency` | HTTP P95 > 1s for 2m | Warning | API responding too slowly |
| `APIDown` | `up{job="fraudshield-api"} == 0` for 30s | Critical | API container unreachable |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not loaded` (503) | Run `python scripts/train_model.py` first |
| Dataset not found | Download from Kaggle link above, place in `data/raw/` |
| Grafana no data | Check `localhost:9090/targets` — API must be UP |
| Docker build fails | Use `python:3.10` (not `slim`) in Dockerfile |
| numpy compatibility | Ensure `numpy<2` in requirements.txt |