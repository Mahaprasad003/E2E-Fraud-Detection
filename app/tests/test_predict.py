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