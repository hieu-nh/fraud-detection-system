"""
Fraud Detection Model Wrapper.
Matches fraudshield-analysis.ipynb feature engineering.
"""

import pickle
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from app.config import MODEL_PATH, PIPELINE_PATH, MODEL_VERSION, FRAUD_THRESHOLD
from app.metrics import (
    PREDICTION_COUNT, PREDICTION_LATENCY, FRAUD_PROBABILITY,
    FRAUD_RATE, HIGH_RISK_COUNT, PREDICTION_ERRORS,
    MODEL_LOADED, MODEL_INFO, MODEL_LAST_RELOAD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_recent_predictions = []
_MAX_RECENT = 1000


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.5:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


class FraudDetectionModel:
    """Wrapper for the fraud detection model."""

    def __init__(self, model_path: str = MODEL_PATH, pipeline_path: str = PIPELINE_PATH):
        self.model_path    = model_path
        self.pipeline_path = pipeline_path
        self.model         = None
        self.label_encoders = {}
        self.feature_names = None
        self.best_threshold = FRAUD_THRESHOLD
        self.model_name    = "unknown"
        self.version       = MODEL_VERSION
        self._load_model()

    def _load_model(self) -> None:
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.pipeline_path, "rb") as f:
                pipeline_data = pickle.load(f)
                self.label_encoders  = pipeline_data.get("label_encoders", {})
                self.feature_names   = pipeline_data.get("feature_names", [])
                self.best_threshold  = pipeline_data.get("best_threshold", FRAUD_THRESHOLD)
                self.model_name      = pipeline_data.get("model_name", "unknown")

            logger.info(f"Model loaded: {self.model_name} | Threshold: {self.best_threshold:.3f}")

            if MODEL_LOADED is not None:   MODEL_LOADED.set(1)
            if MODEL_LAST_RELOAD is not None: MODEL_LAST_RELOAD.set(time.time())
            if MODEL_INFO is not None:
                MODEL_INFO.info({'version': self.version, 'type': self.model_name,
                                 'path': str(self.model_path)})

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            if MODEL_LOADED is not None: MODEL_LOADED.set(0)
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            if MODEL_LOADED is not None: MODEL_LOADED.set(0)
            raise

    def _preprocess(self, data: Dict[str, Any]) -> np.ndarray:
        """Feature engineering matching fraudshield-analysis.ipynb."""
        from sklearn.preprocessing import LabelEncoder

        hour     = int(data["transaction_time"].split(":")[0])
        date     = pd.to_datetime(data["transaction_date"])
        dow      = date.dayofweek
        is_weekend = int(dow >= 5)
        is_night   = int(hour >= 22 or hour <= 5)

        is_intl    = int(data["is_international_transaction"])
        is_new_m   = int(data["is_new_merchant"])
        unusual    = int(data["unusual_time_transaction"])
        failed     = data["failed_transaction_count"]

        risk_score = round(
            is_intl  * 0.30 +
            is_new_m * 0.20 +
            unusual  * 0.25 +
            (1.0 if failed > 0 else 0.0) * 0.15 +
            is_night * 0.10,
            3
        )

        avg_amt = data["avg_transaction_amount"]
        amount_vs_avg = (data["transaction_amount"] / avg_amt) if avg_amt > 0 else 1.0

        # LabelEncoder for categoricals
        def encode(col_key, value):
            le = self.label_encoders.get(col_key)
            if le is None:
                return 0
            try:
                return int(le.transform([value])[0])
            except ValueError:
                return 0

        row = {
            'Transaction_Amount (in Million)':       data["transaction_amount"],
            'Distance_From_Home':                    data["distance_from_home"],
            'Account_Balance (in Million)':          data["account_balance"],
            'Daily_Transaction_Count':               data["daily_transaction_count"],
            'Weekly_Transaction_Count':              data["weekly_transaction_count"],
            'Avg_Transaction_Amount (in Million)':   data["avg_transaction_amount"],
            'Max_Transaction_Last_24h (in Million)': data["max_transaction_last_24h"],
            'Failed_Transaction_Count':              data["failed_transaction_count"],
            'Previous_Fraud_Count':                  data["previous_fraud_count"],
            'Is_International_Transaction':          float(is_intl),
            'Is_New_Merchant':                       float(is_new_m),
            'Unusual_Time_Transaction':              float(unusual),
            'Transaction_Type_enc':                  encode("Transaction_Type", data["transaction_type"]),
            'Merchant_Category_enc':                 encode("Merchant_Category", data["merchant_category"]),
            'Card_Type_enc':                         encode("Card_Type", data["card_type"]),
            'risk_score':                            risk_score,
            'amount_vs_avg':                         amount_vs_avg,
            'is_night':                              is_night,
            'is_weekend':                            is_weekend,
        }

        df = pd.DataFrame([row])
        if self.feature_names:
            available = [f for f in self.feature_names if f in df.columns]
            df = df[available]

        return df.fillna(0).values

    def predict(self, data: Dict[str, Any]) -> Tuple[bool, float, str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        try:
            X = self._preprocess(data)
            proba    = self.model.predict_proba(X)[0][1]
            threshold = self.best_threshold
            is_fraud = proba >= threshold
            risk_level = get_risk_level(proba)
            latency_ms = (time.time() - start_time) * 1000

            # Metrics
            result = "fraud" if is_fraud else "normal"
            if PREDICTION_COUNT is not None:
                PREDICTION_COUNT.labels(model_version=self.version, result=result).inc()
            if PREDICTION_LATENCY is not None:
                PREDICTION_LATENCY.labels(model_version=self.version).observe(time.time() - start_time)
            if FRAUD_PROBABILITY is not None:
                FRAUD_PROBABILITY.labels(model_version=self.version).observe(proba)
            if risk_level in ("HIGH", "CRITICAL") and HIGH_RISK_COUNT is not None:
                HIGH_RISK_COUNT.labels(risk_level=risk_level).inc()

            global _recent_predictions
            _recent_predictions.append(int(is_fraud))
            if len(_recent_predictions) > _MAX_RECENT:
                _recent_predictions.pop(0)
            if _recent_predictions and FRAUD_RATE is not None:
                FRAUD_RATE.set(sum(_recent_predictions) / len(_recent_predictions))

            return is_fraud, round(float(proba), 4), risk_level, round(latency_ms, 3)

        except Exception as e:
            if PREDICTION_ERRORS is not None:
                PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
            logger.error(f"Prediction error: {e}")
            raise

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "model_version":  self.version,
            "model_type":     self.model_name,
            "is_loaded":      self.is_loaded(),
            "features_count": len(self.feature_names) if self.feature_names else 0,
            "fraud_threshold": self.best_threshold,
        }


_model_instance: Optional[FraudDetectionModel] = None


def get_model() -> FraudDetectionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = FraudDetectionModel()
    return _model_instance