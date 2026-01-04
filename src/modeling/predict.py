import pandas as pd
import joblib
from pathlib import Path
import numpy as np
from typing import Union, Dict, Any, List
from pydantic import BaseModel, Field, validator
from src.config import Config
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TransactionData(BaseModel):
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
    Time: Union[float, None] = None  # Optional, will be dropped

    @validator('Amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class ModelConfig:
    """Configuration class for model paths and settings."""
    MODEL_PATH: Path = Config.MODEL_PATH
    SCALER_PATH: Path = Config.SCALER_PATH
    PREDICTION_THRESHOLD: float = Config.PREDICTION_THRESHOLD

def load_model_and_scaler() -> tuple[Any, Any]:
    """Load the trained XGBoost model and scaler.
    
    Returns:
        Tuple of (model, scaler)
        
    Raises:
        FileNotFoundError: If model or scaler files are not found
    """
    if not ModelConfig.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {ModelConfig.MODEL_PATH}. Please run the training script first.")
    if not ModelConfig.SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at {ModelConfig.SCALER_PATH}. Please run the training script first.")
    
    model = joblib.load(ModelConfig.MODEL_PATH)
    scaler = joblib.load(ModelConfig.SCALER_PATH)
    
    return model, scaler

def preprocess_data(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        data: pandas DataFrame with transaction features
        scaler: fitted RobustScaler
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying original
    df = data.copy()
    
    # Drop Time column if present
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    
    # Scale Amount column
    if 'Amount' in df.columns:
        df['Amount'] = scaler.transform(df[['Amount']])
    
    return df

def predict_fraud(data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> np.ndarray:
    """
    Predict fraud probability for transaction data.
    
    Args:
        data: Transaction data as dict, DataFrame, or list of dicts
        
    Returns:
        numpy array of fraud probabilities (0-1)
        
    Raises:
        ValueError: If input data is invalid
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Convert input to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        # Validate each transaction
        validated_data = []
        for item in data:
            transaction = TransactionData(**item)
            validated_data.append(transaction.dict())
        data = pd.DataFrame(validated_data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a dict, list of dicts, or pandas DataFrame")
    
    # Preprocess data
    processed_data = preprocess_data(data, scaler)
    
    # Make predictions (probabilities for class 1 - fraud)
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    return probabilities

def predict_fraud_class(data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]], 
                       threshold: float = ModelConfig.PREDICTION_THRESHOLD) -> np.ndarray:
    """
    Predict fraud class (0 or 1) for transaction data.
    
    Args:
        data: Transaction data as dict, DataFrame, or list of dicts
        threshold: probability threshold for classification
        
    Returns:
        numpy array of predicted classes (0=legitimate, 1=fraud)
    """
    probabilities = predict_fraud(data)
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions