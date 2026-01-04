from fastapi import APIRouter, HTTPException
from app.api.schemas import (
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