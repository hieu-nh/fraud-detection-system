"""
Pydantic schemas for Fraud Detection API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class TransactionType(str, Enum):
    POS = "POS"
    ATM = "ATM"
    ONLINE = "Online"
    TRANSFER = "Transfer"


class CardType(str, Enum):
    CREDIT = "Credit"
    DEBIT = "Debit"


class TransactionRequest(BaseModel):
    """Input schema for fraud prediction."""

    transaction_amount: float = Field(..., gt=0, description="Transaction amount in million", examples=[6.0])
    transaction_time: str = Field(..., description="Time in HH:MM format", examples=["10:54"])
    transaction_date: str = Field(..., description="Date in YYYY-MM-DD format", examples=["2025-03-08"])
    transaction_type: str = Field(..., description="POS, ATM, Online, Transfer", examples=["POS"])
    merchant_category: str = Field(..., description="Merchant category", examples=["Retail"])
    transaction_location: str = Field(..., description="City of transaction", examples=["Singapore"])
    customer_home_location: str = Field(..., description="Customer home city", examples=["Lahore"])
    distance_from_home: float = Field(..., ge=0, description="Distance from home in km", examples=[466.0])
    card_type: str = Field(..., description="Credit or Debit", examples=["Credit"])
    account_balance: float = Field(..., ge=0, description="Account balance in million", examples=[30.0])
    daily_transaction_count: int = Field(..., ge=0, examples=[4])
    weekly_transaction_count: int = Field(..., ge=0, examples=[17])
    avg_transaction_amount: float = Field(..., ge=0, description="Average transaction amount in million", examples=[2.0])
    max_transaction_last_24h: float = Field(..., ge=0, examples=[4.0])
    is_international_transaction: bool = Field(..., examples=[True])
    is_new_merchant: bool = Field(..., examples=[True])
    failed_transaction_count: int = Field(..., ge=0, examples=[0])
    unusual_time_transaction: bool = Field(..., examples=[False])
    previous_fraud_count: int = Field(..., ge=0, examples=[1])

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_amount": 6.0,
                "transaction_time": "10:54",
                "transaction_date": "2025-03-08",
                "transaction_type": "POS",
                "merchant_category": "Retail",
                "transaction_location": "Singapore",
                "customer_home_location": "Lahore",
                "distance_from_home": 466.0,
                "card_type": "Credit",
                "account_balance": 30.0,
                "daily_transaction_count": 4,
                "weekly_transaction_count": 17,
                "avg_transaction_amount": 2.0,
                "max_transaction_last_24h": 4.0,
                "is_international_transaction": True,
                "is_new_merchant": True,
                "failed_transaction_count": 0,
                "unusual_time_transaction": False,
                "previous_fraud_count": 1
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