# System Architecture — FraudShield Detection System

## Overview

FraudShield is an end-to-end ML system for real-time credit card fraud detection. The system ingests transaction data, scores it with a GBM model, and exposes results via a REST API with full observability.

**Dataset:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
- Train: ~1.3M transactions | Fraud rate: 0.58%
- Test: ~555K transactions | Fraud rate: 0.39%

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Applications                   │
│              (Mobile App / Banking System)               │
└────────────────────────┬────────────────────────────────┘
                         │ POST /predict
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FraudShield API                         │
│                  (FastAPI + Uvicorn)                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  /predict   │  │   /health    │  │   /metrics    │  │
│  │  /model/info│  │              │  │  (Prometheus) │  │
│  └──────┬──────┘  └──────────────┘  └───────────────┘  │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │          Metrics Middleware (Prometheus)         │   │
│  └──────┬──────────────────────────────────────────┘   │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │   FraudDetectionModel                            │   │
│  │   ┌───────────────┐  ┌────────────────────────┐  │   │
│  │   │  Preprocessor │  │    GBM Classifier      │  │   │
│  │   │  (pipeline.pkl│→ │  (fraud_model.pkl)     │  │   │
│  │   └───────────────┘  └────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │ scrape /metrics
         ▼
┌─────────────────┐
│   Prometheus    │
│   (metrics DB)  │
└────────┬────────┘
         │ query
         ▼
┌─────────────────┐
│     Grafana     │
│  (dashboards)   │
└─────────────────┘
```

## Component Responsibilities

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| REST API | FastAPI + Uvicorn | Serve predictions, health check, metrics endpoint |
| ML Model | GBM (scikit-learn) | Binary fraud classification |
| Preprocessing | LabelEncoder + feature engineering | Time parsing, distance, ratios, encoding |
| Metrics | Prometheus Client | Expose business & system metrics |
| Monitoring | Prometheus + Grafana | Collect, store, visualize metrics, alerting |
| Containerization | Docker + Compose | Package and orchestrate all services |
| CI/CD | GitHub Actions | Lint, test, build on every push |

## Data Flow

```
API Request (JSON — 11 fields)
      │
      ▼
Pydantic Validation  ←── 422 if invalid
      │
      ▼
Feature Engineering (app/model.py._preprocess)
  • Extract hour, day_of_week, month, is_night, is_weekend
    from trans_date_trans_time
  • Calculate age from dob
  • Calculate distance (haversine: cardholder → merchant lat/long)
  • Compute amt_vs_category_mean ratio
  • LabelEncode category, state (using encoders from pipeline.pkl)
  • Encode gender (M=1, F=0)
      │
      ▼
numpy array (13 features)
      │
      ▼
GBM.predict_proba()
      │
      ▼
fraud_probability = 0.92
      │
      ▼
probability >= best_threshold (0.851 from pipeline.pkl)
      │
      ▼
Risk Classification
  • < 0.3  → LOW
  • 0.3-0.5 → MEDIUM
  • 0.5-0.8 → HIGH
  • > 0.8   → CRITICAL
      │
      ▼
JSON Response { is_fraud, fraud_probability, risk_level, latency_ms }
```

## ML Pipeline

```
fraudTrain.csv + fraudTest.csv
  │
  ├── engineer_features()
  │     • Parse datetime → hour, day_of_week, month, is_night, is_weekend
  │     • Parse dob → age
  │     • Haversine(lat, long, merch_lat, merch_long) → distance
  │     • amt / category_mean → amt_vs_category_mean
  │     • LabelEncoder → category_enc, state_enc
  │     • gender → gender_enc (M=1, F=0)
  │
  ├── compare_models()
  │     • LogisticRegression (class_weight='balanced')
  │     • RandomForest (class_weight='balanced')
  │     • GBM (manual fraud × 10 oversampling)  ← best PR-AUC 0.82
  │
  ├── find_best_threshold()
  │     → threshold = 0.851 (maximizes F1 on test set)
  │
  └── pickle.dump()
        → fraud_model.pkl  (trained GBM)
        → pipeline.pkl     (label_encoders + threshold + feature_names)
```

## Model Performance

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Threshold |
|-------|---------|--------|----|-----------|--------|-----------|
| GBM_oversampled | 0.9935 | **0.8208** | 0.7841 | 0.8473 | 0.7296 | 0.851 |
| RandomForest_balanced | 0.9899 | 0.7461 | 0.7166 | 0.8072 | 0.6443 | 0.931 |
| LogisticRegression_balanced | 0.9300 | 0.2139 | 0.4277 | 0.3191 | 0.6485 | 0.871 |

> **Why PR-AUC?** With 0.58% fraud rate, accuracy is misleading — a model predicting everything Normal scores 99.4% accuracy while catching zero fraud. PR-AUC is the correct metric for highly imbalanced datasets.

## Technology Stack Justification

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML Model | GBM | Best PR-AUC (0.82) on this dataset, handles class imbalance, fast inference |
| Class imbalance | Manual fraud × 10 oversampling | Simple, effective, avoids SMOTE artifacts on real transaction data |
| Threshold | 0.851 (tuned) | Maximizes F1 — balances precision (0.85) and recall (0.73) |
| API Framework | FastAPI | Async support, auto Swagger docs, Pydantic validation, high performance |
| Monitoring | Prometheus + Grafana | Industry standard, rich PromQL queries, 12 dashboard panels |
| Containerization | Docker | Reproducibility across environments, easy scaling |

## Trade-off Analysis

| Concern | Trade-off |
|---------|-----------|
| **Recall vs Precision** | Threshold 0.851 gives Precision 0.85 / Recall 0.73 — prioritizes fewer false alerts over catching every fraud |
| **GBM vs Neural Network** | GBM: faster training, interpretable via feature importance, <5ms inference. Neural nets would need more data and GPU |
| **Manual oversampling vs SMOTE** | Manual fraud × 10 is simpler and avoids synthetic sample artifacts on real transaction patterns |
| **Local MLflow vs Docker MLflow** | MLflow runs locally (`mlruns/`) — keeps setup simple, no extra container needed |

## Ports & Services

| Service | Port | URL |
|---------|------|-----|
| FraudShield API | 8000 | http://localhost:8000 |
| API Docs (Swagger) | 8000 | http://localhost:8000/docs |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |