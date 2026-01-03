"""Configuration settings for the fraud detection project."""
from pathlib import Path

class Config:
    """Central configuration for paths and settings."""
    
    # Data paths
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CREDITCARD_CSV = RAW_DATA_DIR / "creditcard.csv"
    
    # Model paths
    MODELS_DIR = Path("models")
    PRODUCTION_DIR = MODELS_DIR / "production"
    MODEL_PATH = PRODUCTION_DIR / "xgb_model.joblib"
    SCALER_PATH = PRODUCTION_DIR / "scaler.joblib"
    
    # Config paths
    CONFIG_DIR = Path("src/config")
    PARAMS_YAML = CONFIG_DIR / "params.yaml"
    
    # Model settings
    PREDICTION_THRESHOLD = 0.5
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2  # From train split
    
    # MLflow settings
    EXPERIMENT_NAME = "Fraud_Detection_Optimized"
    
    # Optuna settings
    N_TRIALS = 20