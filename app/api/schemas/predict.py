from pydantic import BaseModel, Field
from typing import Union, Optional

class TransactionInput(BaseModel):
    """
    Input schema for a single transaction.
    """
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    Time: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.36,
                "V2": -0.35,
                "V3": 1.68,
                "V4": 0.45,
                "V5": -0.12,
                "V6": -0.89,
                "V7": -0.21,
                "V8": 0.08,
                "V9": -0.23,
                "V10": 0.07,
                "V11": 0.23,
                "V12": -0.34,
                "V13": 0.11,
                "V14": -0.56,
                "V15": 0.22,
                "V16": -0.09,
                "V17": 0.14,
                "V18": -0.03,
                "V19": 0.01,
                "V20": 0.08,
                "V21": -0.01,
                "V22": -0.02,
                "V23": 0.01,
                "V24": 0.09,
                "V25": 0.03,
                "V26": -0.01,
                "V27": 0.01,
                "V28": 0.02,
                "Amount": 149.62,
                "Time": 0.0
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction"""
    is_fraud: int = Field(..., description = "0 = legitimate, 1 = Fraud")
    probability: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    threshold: float = Field(default=0.5, description="Classification threshold used")


class BatchTransactionInput(BaseModel):
    """Input schema for batch predictions."""
    transactions: list[TransactionInput]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: list[PredictionResponse]
    total_transactions: int
    fraud_count: int
    legitimate_count: int