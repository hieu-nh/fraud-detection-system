# System Architecture — FraudShield Detection System

## Overview

FraudShield is an end-to-end ML system for real-time financial transaction fraud detection. The system ingests transaction data, scores it with an XGBoost model, and exposes results via a REST API with full observability.

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
│  │   │  Preprocessor │  │   XGBoost Classifier   │  │   │
│  │   │  (Pipeline)   │→ │   (fraud_model.pkl)    │  │   │
│  │   └───────────────┘  └────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │ scrape /metrics                │ artifacts
         ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│   Prometheus    │             │     MLflow      │
│   (metrics DB)  │             │  (experiments)  │
└────────┬────────┘             └─────────────────┘
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
| ML Model | XGBoost | Binary fraud classification |
| Preprocessing | scikit-learn Pipeline | Feature engineering, scaling, encoding |
| Metrics | Prometheus Client | Expose business & system metrics |
| Monitoring | Prometheus + Grafana | Collect, store, visualize metrics |
| Experiment Tracking | MLflow | Track training runs, parameters, metrics, artifacts |
| Containerization | Docker + Compose | Package and orchestrate all services |
| CI/CD | GitHub Actions | Lint, test, build on every push |

## Data Flow

```
Raw Transaction
      │
      ▼
POST /predict  ←── JSON payload (19 fields)
      │
      ▼
Pydantic Validation  ←── 422 if invalid
      │
      ▼
Feature Engineering
  • Extract hour, day-of-week from datetime
  • Compute distance ratio, amount ratios
  • Encode Yes/No → 1/0
  • One-hot encode categorical fields
      │
      ▼
StandardScaler (numerical features)
OneHotEncoder (categorical features)
      │
      ▼
XGBoost.predict_proba()
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
Raw CSV
  │
  ├── load_data()         → Drop nulls, drop ID columns
  ├── engineer_features() → Time parsing, ratio features, binary encoding
  ├── train_test_split()  → 80/20, stratified by Fraud_Label
  ├── SMOTE               → Oversample minority class (fraud ~4.8%)
  ├── StandardScaler      → Normalize numerical features
  ├── OneHotEncoder       → Encode categorical features
  ├── XGBClassifier       → Train with scale_pos_weight
  └── pickle.dump()       → Save model + pipeline
```

## Technology Stack Justification

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML Model | XGBoost | Best performance on tabular data, handles class imbalance, fast inference (<1ms) |
| Class imbalance | SMOTE + scale_pos_weight | Fraud is 4.8% — need both training-time and algorithm-level balancing |
| API Framework | FastAPI | Async support, auto Swagger docs, Pydantic validation, high performance |
| Monitoring | Prometheus + Grafana | Industry standard, works well with Python, rich querying with PromQL |
| Experiment Tracking | MLflow | Open source, integrates with XGBoost natively, self-hosted |
| Containerization | Docker | Reproducibility across environments, easy scaling |

## Trade-off Analysis

| Concern | Trade-off |
|---------|-----------|
| **Latency vs Accuracy** | XGBoost achieves <1ms inference. Deep learning would improve accuracy but add 10-100ms latency — unacceptable for real-time fraud detection |
| **Recall vs Precision** | Tuned for high recall (catch more fraud) at cost of more false positives. False negatives (missed fraud) are more costly than false positives (blocked legitimate transactions) |
| **SMOTE vs Class Weights** | SMOTE creates synthetic fraud samples — better generalization. Class weights alone may not create enough diversity in decision boundaries |
| **Self-hosted MLflow vs Cloud** | Self-hosted in Docker keeps all data on-premise — important for financial transaction data privacy |

## Ports & Services

| Service | Port | URL |
|---------|------|-----|
| FraudShield API | 8000 | http://localhost:8000 |
| API Docs (Swagger) | 8000 | http://localhost:8000/docs |
| MLflow UI | 5000 | http://localhost:5000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |