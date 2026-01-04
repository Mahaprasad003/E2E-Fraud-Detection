# FastAPI Setup Guide for Fraud Detection App

This guide walks you through completing the FastAPI application in your `app/` folder. It's based on your **actual codebase**—your existing `src/modeling/predict.py`, `src/config.py`, and the folder structure you've already created.

---

## Your Current Folder Structure (Already Created)

```
app/
├── __init__.py                     ✅ EXISTS
├── main.py                         ✅ EXISTS (needs minor fix)
├── api/
│   ├── __init__.py                 ✅ EXISTS
│   ├── dependencies/
│   │   └── __init__.py             ✅ EXISTS
│   ├── endpoints/
│   │   └── __init__.py             ✅ EXISTS (need to add predict.py, health.py)
│   └── schemas/
│       └── __init__.py             ✅ EXISTS (need to add predict.py)
├── cache/
│   └── __init__.py                 ✅ EXISTS
├── core/
│   ├── __init__.py                 ✅ EXISTS
│   └── config.py                   ✅ EXISTS
├── middleware/
│   └── __init__.py                 ✅ EXISTS
├── tests/
│   └── __init__.py                 ✅ EXISTS (need to add test files)
└── utils/
    ├── __init__.py                 ✅ EXISTS
    └── helpers.py                  ✅ EXISTS (needs update)
```

---

## What Your Existing Code Provides

### From `src/modeling/predict.py`:
- **`TransactionData`** - Pydantic model with fields: `V1-V28`, `Amount`, `Time`
- **`load_model_and_scaler()`** - Loads model from `models/production/xgb_model.joblib` and scaler from `models/production/scaler.joblib`
- **`predict_fraud(data)`** - Takes dict, DataFrame, or list of dicts → returns numpy array of fraud probabilities (0-1)
- **`predict_fraud_class(data, threshold=0.5)`** - Returns predicted classes (0 or 1)

### From `src/config.py`:
- **`Config.MODEL_PATH`** = `models/production/xgb_model.joblib`
- **`Config.SCALER_PATH`** = `models/production/scaler.joblib`
- **`Config.PREDICTION_THRESHOLD`** = `0.5`

---

## Step 1: Fix `app/utils/helpers.py`

Your current file has wrong imports. Replace it with code that uses your **actual** `src/modeling/predict.py` functions:

```python
"""Helper utilities for the FastAPI app."""
import sys
from pathlib import Path

# Add src to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.predict import (
    load_model_and_scaler,
    predict_fraud,
    predict_fraud_class,
    TransactionData,
    ModelConfig
)

# Re-export for easy access
__all__ = [
    "load_model_and_scaler",
    "predict_fraud", 
    "predict_fraud_class",
    "TransactionData",
    "ModelConfig"
]
```

---

## Step 2: Create `app/api/dependencies/model_loader.py`

This loads your model and scaler **once** at startup (not per request):

```python
"""Load ML model and scaler at startup for performance."""
from app.utils.helpers import load_model_and_scaler

# Load model and scaler once when the module is imported
model, scaler = load_model_and_scaler()

def get_model():
    """Dependency to get the loaded model."""
    return model

def get_scaler():
    """Dependency to get the loaded scaler."""
    return scaler
```

---

## Step 3: Create `app/api/schemas/predict.py`

This defines request/response schemas. Use the **exact features** from your `src/modeling/predict.py`:

```python
"""Pydantic schemas for prediction endpoints."""
from pydantic import BaseModel, Field
from typing import Union, Optional

class TransactionInput(BaseModel):
    """
    Input schema for a single transaction.
    Based on your TransactionData in src/modeling/predict.py.
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
    Time: Optional[float] = None  # Optional, will be dropped during preprocessing

    class Config:
        schema_extra = {
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


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""
    is_fraud: int = Field(..., description="0 = legitimate, 1 = fraud")
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
```

---

## Step 4: Create `app/api/endpoints/health.py`

Simple health check endpoint:

```python
"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}


@router.get("/ready")
def readiness_check():
    """Check if the API is ready to serve requests (model loaded)."""
    try:
        from app.api.dependencies.model_loader import model, scaler
        if model is not None and scaler is not None:
            return {"status": "ready", "model_loaded": True}
        return {"status": "not_ready", "model_loaded": False}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
```

---

## Step 5: Create `app/api/endpoints/predict.py`

The main prediction endpoint using your **actual** `predict_fraud` function:

```python
"""Prediction endpoints for fraud detection."""
from fastapi import APIRouter, HTTPException
from app.api.schemas.predict import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionInput,
    BatchPredictionResponse
)
from app.utils.helpers import predict_fraud, predict_fraud_class, ModelConfig

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict_single(transaction: TransactionInput):
    """
    Predict if a single transaction is fraudulent.
    
    Returns fraud probability and classification.
    """
    try:
        # Convert Pydantic model to dict
        data = transaction.dict()
        
        # Get probability using your existing predict_fraud function
        probabilities = predict_fraud(data)
        probability = float(probabilities[0])
        
        # Classify using threshold from your Config
        threshold = ModelConfig.PREDICTION_THRESHOLD
        is_fraud = 1 if probability >= threshold else 0
        
        return PredictionResponse(
            is_fraud=is_fraud,
            probability=probability,
            threshold=threshold
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchTransactionInput):
    """
    Predict fraud for multiple transactions at once.
    
    More efficient for bulk predictions.
    """
    try:
        # Convert list of Pydantic models to list of dicts
        data = [t.dict() for t in batch.transactions]
        
        # Get probabilities for all transactions
        probabilities = predict_fraud(data)
        threshold = ModelConfig.PREDICTION_THRESHOLD
        
        # Build response
        predictions = []
        fraud_count = 0
        for prob in probabilities:
            prob_float = float(prob)
            is_fraud = 1 if prob_float >= threshold else 0
            fraud_count += is_fraud
            predictions.append(PredictionResponse(
                is_fraud=is_fraud,
                probability=prob_float,
                threshold=threshold
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(predictions),
            fraud_count=fraud_count,
            legitimate_count=len(predictions) - fraud_count
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
```

---

## Step 6: Update `app/api/endpoints/__init__.py`

Make the routers importable:

```python
"""API endpoints."""
from app.api.endpoints import predict, health

__all__ = ["predict", "health"]
```

---

## Step 7: Your `app/main.py` is Already Correct!

Your current `main.py` works. Just note the routes will be:
- `GET /` → Welcome message
- `GET /api/health` → Health check
- `GET /api/ready` → Readiness check  
- `POST /api/predict` → Single prediction
- `POST /api/predict/batch` → Batch predictions

---

## Step 8: Create `app/tests/test_predict.py`

Test file using your actual feature structure:

```python
"""Tests for prediction endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample transaction data matching your TransactionData schema
SAMPLE_TRANSACTION = {
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


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Fraud Detection API"}


def test_health():
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready():
    """Test readiness check endpoint."""
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_single():
    """Test single transaction prediction."""
    response = client.post("/api/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert "threshold" in data
    assert data["is_fraud"] in [0, 1]
    assert 0 <= data["probability"] <= 1


def test_predict_batch():
    """Test batch prediction."""
    batch = {"transactions": [SAMPLE_TRANSACTION, SAMPLE_TRANSACTION]}
    response = client.post("/api/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert len(data["predictions"]) == 2


def test_predict_invalid_amount():
    """Test that negative amount is rejected."""
    invalid = SAMPLE_TRANSACTION.copy()
    invalid["Amount"] = -100  # Invalid: must be positive
    response = client.post("/api/predict", json=invalid)
    assert response.status_code == 422  # Validation error
```

---

## Step 9: Run the API

From your project root directory:

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Set PYTHONPATH so imports work
export PYTHONPATH=$PWD

# Run the API with auto-reload for development
uvicorn app.main:app --reload
```

Then visit:
- **Swagger Docs**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/api/health
- **Root**: http://127.0.0.1:8000/

---

## Step 10: Run Tests

```bash
# From project root with PYTHONPATH set
export PYTHONPATH=$PWD
pytest app/tests/ -v
```

---

## Summary: Files You Need to Create/Update

| File | Action |
|------|--------|
| `app/utils/helpers.py` | **UPDATE** - fix imports to use actual `src/modeling/predict.py` |
| `app/api/dependencies/model_loader.py` | **CREATE** - load model at startup |
| `app/api/schemas/predict.py` | **CREATE** - request/response schemas |
| `app/api/endpoints/health.py` | **CREATE** - health check endpoints |
| `app/api/endpoints/predict.py` | **CREATE** - prediction endpoints |
| `app/api/endpoints/__init__.py` | **UPDATE** - export routers |
| `app/tests/test_predict.py` | **CREATE** - API tests |

---

## Troubleshooting

**Import errors?**
```bash
export PYTHONPATH=$PWD
```

**Model not found?**
Ensure `models/production/xgb_model.joblib` and `scaler.joblib` exist. If not, run your training notebook.

**Pydantic errors?**
You may need to install `pydantic[email]` or update pydantic: `uv add pydantic`

**Port in use?**
Use a different port: `uvicorn app.main:app --reload --port 8001`