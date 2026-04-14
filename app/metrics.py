"""
Prometheus metrics for Fraud Detection API.
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# HTTP metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Fraud Detection specific metrics
PREDICTION_COUNT = Counter(
    'fraud_predictions_total',
    'Total predictions made',
    ['model_version', 'result']  # result: fraud / normal
)

PREDICTION_LATENCY = Histogram(
    'fraud_prediction_duration_seconds',
    'Time to make a fraud prediction',
    ['model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

FRAUD_PROBABILITY = Histogram(
    'fraud_probability_distribution',
    'Distribution of fraud probability scores',
    ['model_version'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

FRAUD_RATE = Gauge(
    'fraud_detection_rate',
    'Current fraud rate (rolling)'
)

HIGH_RISK_COUNT = Counter(
    'high_risk_transactions_total',
    'Total HIGH/CRITICAL risk transactions detected',
    ['risk_level']
)

PREDICTION_ERRORS = Counter(
    'fraud_prediction_errors_total',
    'Total prediction errors',
    ['error_type']
)

# Model status
MODEL_LOADED = Gauge(
    'ml_model_loaded',
    'Whether the ML model is loaded (1) or not (0)'
)

MODEL_INFO = Info(
    'ml_model',
    'Information about the loaded ML model'
)

MODEL_LAST_RELOAD = Gauge(
    'ml_model_last_reload_timestamp',
    'Unix timestamp of last model reload'
)