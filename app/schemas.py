"""
Pydantic schemas for Credit Card Fraud Detection API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """Input schema for fraud prediction."""

    trans_date_trans_time: str = Field(
        ..., description="Transaction datetime", examples=["2019-01-01 00:00:00"]
    )
    amt: float = Field(
        ..., gt=0, description="Transaction amount in USD", examples=[107.23]
    )
    category: str = Field(
        ..., description="Merchant category", examples=["grocery_pos"]
    )
    gender: str = Field(
        ..., description="Cardholder gender: M or F", examples=["F"]
    )
    city_pop: int = Field(
        ..., ge=0, description="Population of cardholder's city", examples=[149]
    )
    dob: str = Field(
        ..., description="Date of birth (DD/MM/YY or YYYY-MM-DD)", examples=["21/6/78"]
    )
    lat: float = Field(
        ..., description="Cardholder latitude", examples=[48.8876]
    )
    long: float = Field(
        ..., description="Cardholder longitude", examples=[-118.1864]
    )
    merch_lat: float = Field(
        ..., description="Merchant latitude", examples=[49.159]
    )
    merch_long: float = Field(
        ..., description="Merchant longitude", examples=[-118.186]
    )
    state: str = Field(
        ..., description="Cardholder's state", examples=["WA"]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "trans_date_trans_time": "2019-01-01 12:30:00",
                "amt": 107.23,
                "category": "grocery_pos",
                "gender": "F",
                "city_pop": 149,
                "dob": "21/6/78",
                "lat": 48.8876,
                "long": -118.1864,
                "merch_lat": 49.159,
                "merch_long": -118.186,
                "state": "WA"
            }
        }
    }


class FraudPredictionResponse(BaseModel):
    """Response schema for fraud prediction."""

    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    model_version: str
    latency_ms: float

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: str

    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_version: str
    model_type: str
    is_loaded: bool
    features_count: int
    fraud_threshold: float

    model_config = {"protected_namespaces": ()}