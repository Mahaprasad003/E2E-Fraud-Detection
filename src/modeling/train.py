import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from preprocessing import scale_data
from dataloader import load_data
from utils.params import load_params
from pathlib import Path
import joblib
import xgboost as xgb
from typing import Any, Tuple
# from config import MODEL_CONFIG

def train_model() -> xgb.XGBClassifier:
    data_path = Path("data/raw/creditcard.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run the data download script first.")
    X, y = load_data(str(data_path))
    
    # Load parameters
    params = load_params()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params['random_state'], stratify=y)
    print("Data split into training and testing sets.")
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    # Load best parameters from tuning
    best_params = params['models']['xgboost']
    best_params.update({
        'scale_pos_weight': scale_pos_weight, 'random_state': params['random_state'],
        'n_jobs': -1, 'tree_method': 'hist', 'objective': 'binary:logistic',
        'eval_metric': 'aucpr', 'n_estimators': 1000
    })
    
    # Train XGBoost model with best parameters
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_scaled, y_train)
    print("Model training completed.")
    
    # Save trained model
    model_path = Path("models/production/xgb_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"Test AUPRC: {auprc:.4f}")
    
    return model

if __name__ == "__main__":
    train_model()
