"""
Fraud Detection Model Wrapper.
"""

import pickle
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from app.config import MODEL_PATH, PIPELINE_PATH, MODEL_VERSION, FRAUD_THRESHOLD
from app.metrics import (
    PREDICTION_COUNT, PREDICTION_LATENCY, FRAUD_PROBABILITY,
    FRAUD_RATE, HIGH_RISK_COUNT, PREDICTION_ERRORS,
    MODEL_LOADED, MODEL_INFO, MODEL_LAST_RELOAD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rolling fraud rate tracking
_recent_predictions = []
_MAX_RECENT = 1000


def get_risk_level(probability: float) -> str:
    """Convert fraud probability to risk level."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.5:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


class FraudDetectionModel:
    """Wrapper for the XGBoost fraud detection model."""

    def __init__(self, model_path: str = MODEL_PATH, pipeline_path: str = PIPELINE_PATH):
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.model = None
        self.pipeline = None
        self.version = MODEL_VERSION
        self.feature_names = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model and preprocessing pipeline from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.pipeline_path, "rb") as f:
                pipeline_data = pickle.load(f)
                self.pipeline = pipeline_data["pipeline"]
                self.feature_names = pipeline_data["feature_names"]

            logger.info(f"Model loaded from {self.model_path}")
            logger.info(f"Pipeline loaded with {len(self.feature_names)} features")

            if MODEL_LOADED is not None:
                MODEL_LOADED.set(1)
            if MODEL_LAST_RELOAD is not None:
                MODEL_LAST_RELOAD.set(time.time())
            if MODEL_INFO is not None:
                MODEL_INFO.info({
                    'version': self.version,
                    'type': 'XGBoost',
                    'path': str(self.model_path)
                })

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)
            raise

    def _preprocess(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess raw input into model features."""
        # Parse time features
        hour = int(data["transaction_time"].split(":")[0])
        date = pd.to_datetime(data["transaction_date"])

        row = {
            "Transaction_Amount (in Million)": data["transaction_amount"],
            "Transaction_Hour": hour,
            "Transaction_DayOfWeek": date.dayofweek,
            "Transaction_Month": date.month,
            "Transaction_Type": data["transaction_type"],
            "Merchant_Category": data["merchant_category"],
            "Is_Same_Location": int(data["transaction_location"] == data["customer_home_location"]),
            "Distance_From_Home": data["distance_from_home"],
            "Card_Type": data["card_type"],
            "Account_Balance (in Million)": data["account_balance"],
            "Daily_Transaction_Count": data["daily_transaction_count"],
            "Weekly_Transaction_Count": data["weekly_transaction_count"],
            "Avg_Transaction_Amount (in Million)": data["avg_transaction_amount"],
            "Max_Transaction_Last_24h (in Million)": data["max_transaction_last_24h"],
            "Is_International_Transaction": int(data["is_international_transaction"]),
            "Is_New_Merchant": int(data["is_new_merchant"]),
            "Failed_Transaction_Count": data["failed_transaction_count"],
            "Unusual_Time_Transaction": int(data["unusual_time_transaction"]),
            "Previous_Fraud_Count": data["previous_fraud_count"],
            "Amount_To_Balance_Ratio": (
                data["transaction_amount"] / data["account_balance"]
                if data["account_balance"] > 0 else 0
            ),
            "Amount_To_Avg_Ratio": (
                data["transaction_amount"] / data["avg_transaction_amount"]
                if data["avg_transaction_amount"] > 0 else 0
            ),
        }

        df = pd.DataFrame([row])
        return df

    def predict(self, data: Dict[str, Any]) -> Tuple[bool, float, str, float]:
        """
        Predict fraud for a transaction.

        Returns:
            Tuple of (is_fraud, probability, risk_level, latency_ms)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            df = self._preprocess(data)
            X = self.pipeline.transform(df)
            proba = self.model.predict_proba(X)[0][1]
            is_fraud = proba >= FRAUD_THRESHOLD
            risk_level = get_risk_level(proba)
            latency_ms = (time.time() - start_time) * 1000

            # Record metrics
            result = "fraud" if is_fraud else "normal"
            PREDICTION_COUNT.labels(model_version=self.version, result=result).inc()
            PREDICTION_LATENCY.labels(model_version=self.version).observe(time.time() - start_time)
            FRAUD_PROBABILITY.labels(model_version=self.version).observe(proba)

            if risk_level in ("HIGH", "CRITICAL"):
                HIGH_RISK_COUNT.labels(risk_level=risk_level).inc()

            # Update rolling fraud rate
            global _recent_predictions
            _recent_predictions.append(int(is_fraud))
            if len(_recent_predictions) > _MAX_RECENT:
                _recent_predictions.pop(0)
            if _recent_predictions:
                FRAUD_RATE.set(sum(_recent_predictions) / len(_recent_predictions))

            return is_fraud, round(float(proba), 4), risk_level, round(latency_ms, 3)

        except Exception as e:
            PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
            logger.error(f"Prediction error: {e}")
            raise

    def is_loaded(self) -> bool:
        return self.model is not None and self.pipeline is not None

    def get_info(self) -> dict:
        return {
            "model_version": self.version,
            "model_type": "XGBoost",
            "is_loaded": self.is_loaded(),
            "features_count": len(self.feature_names) if self.feature_names else 0,
            "fraud_threshold": FRAUD_THRESHOLD,
        }


_model_instance: Optional[FraudDetectionModel] = None


def get_model() -> FraudDetectionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = FraudDetectionModel()
    return _model_instance