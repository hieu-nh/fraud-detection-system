"""
Fraud Detection Model Wrapper.
Compatible with fraudTrain/fraudTest credit card dataset.
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


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return float(R * 2 * np.arcsin(np.sqrt(a)))


class FraudDetectionModel:
    """Wrapper for the credit card fraud detection model."""

    def __init__(self, model_path: str = MODEL_PATH, pipeline_path: str = PIPELINE_PATH):
        self.model_path    = model_path
        self.pipeline_path = pipeline_path
        self.model         = None
        self.label_encoders = {}
        self.cat_mean      = {}
        self.feature_names = []
        self.best_threshold = FRAUD_THRESHOLD
        self.model_name    = "unknown"
        self.scaler        = None
        self.version       = MODEL_VERSION
        self._load_model()

    def _load_model(self) -> None:
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.pipeline_path, "rb") as f:
                pipeline_data        = pickle.load(f)
                self.label_encoders  = pipeline_data.get("label_encoders", {})
                self.cat_mean        = self.label_encoders.get("cat_mean", {})
                self.feature_names   = pipeline_data.get("feature_names", [])
                self.best_threshold  = pipeline_data.get("best_threshold", FRAUD_THRESHOLD)
                self.model_name      = pipeline_data.get("model_name", "unknown")
                self.scaler          = pipeline_data.get("scaler", None)

            logger.info(f"Model loaded: {self.model_name} | Threshold: {self.best_threshold:.3f}")

            if MODEL_LOADED is not None:      MODEL_LOADED.set(1)
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
        """Feature engineering matching train_model.py exactly."""

        # Datetime features
        dt = pd.to_datetime(data["trans_date_trans_time"], dayfirst=False, errors='coerce')
        hour        = dt.hour if pd.notna(dt) else 0
        day_of_week = dt.dayofweek if pd.notna(dt) else 0
        month       = dt.month if pd.notna(dt) else 1
        is_night    = int(hour >= 22 or hour <= 5)
        is_weekend  = int(day_of_week >= 5)

        # Age from dob
        dob = pd.to_datetime(data.get("dob", ""), dayfirst=True, errors='coerce')
        if pd.notna(dt) and pd.notna(dob):
            age = (dt - dob).days / 365.25
        else:
            age = 40.0  # fallback median

        # Distance
        distance = haversine(
            float(data.get("lat", 0)),
            float(data.get("long", 0)),
            float(data.get("merch_lat", 0)),
            float(data.get("merch_long", 0)),
        )

        # Amount vs category mean
        amt = float(data.get("amt", 0))
        cat = data.get("category", "")
        cat_mean_val = self.cat_mean.get(cat, 1.0) or 1.0
        amt_vs_cat_mean = amt / cat_mean_val

        # Gender
        gender_enc = 1 if data.get("gender", "F") == "M" else 0

        # LabelEncode category and state
        def encode(key, value):
            le = self.label_encoders.get(key)
            if le is None:
                return 0
            try:
                return int(le.transform([value])[0])
            except ValueError:
                return 0

        category_enc = encode("category", cat)
        state_enc    = encode("state", data.get("state", ""))

        row = {
            'amt':                   amt,
            'city_pop':              float(data.get("city_pop", 0)),
            'age':                   age,
            'hour':                  hour,
            'day_of_week':           day_of_week,
            'month':                 month,
            'is_night':              is_night,
            'is_weekend':            is_weekend,
            'distance':              distance,
            'amt_vs_category_mean':  amt_vs_cat_mean,
            'category_enc':          category_enc,
            'gender_enc':            gender_enc,
            'state_enc':             state_enc,
        }

        df = pd.DataFrame([row])
        if self.feature_names:
            available = [f for f in self.feature_names if f in df.columns]
            df = df[available]

        X = df.fillna(0).values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def predict(self, data: Dict[str, Any]) -> Tuple[bool, float, str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        try:
            X        = self._preprocess(data)
            proba    = float(self.model.predict_proba(X)[0][1])
            is_fraud = proba >= self.best_threshold
            risk_level = get_risk_level(proba)
            latency_ms = (time.time() - start_time) * 1000

            # Prometheus metrics
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

            return is_fraud, round(proba, 4), risk_level, round(latency_ms, 3)

        except Exception as e:
            if PREDICTION_ERRORS is not None:
                PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
            logger.error(f"Prediction error: {e}")
            raise

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "model_version":   self.version,
            "model_type":      self.model_name,
            "is_loaded":       self.is_loaded(),
            "features_count":  len(self.feature_names),
            "fraud_threshold": self.best_threshold,
        }


_model_instance: Optional[FraudDetectionModel] = None


def get_model() -> FraudDetectionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = FraudDetectionModel()
    return _model_instance