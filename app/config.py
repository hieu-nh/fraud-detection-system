"""
Configuration settings for the Fraud Detection API.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "fraud_model.pkl"))
PIPELINE_PATH = os.getenv("PIPELINE_PATH", str(BASE_DIR / "models" / "pipeline.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# API settings
API_TITLE = "FraudShield Detection API"
API_DESCRIPTION = "Real-time financial transaction fraud detection using Gradient Boosting (GBM)"
API_VERSION = "1.0.0"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Monitoring
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

# Fraud threshold
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))