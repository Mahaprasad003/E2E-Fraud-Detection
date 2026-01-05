import pandas as pd
from sklearn.model_selection import train_test_split
from utils.params import params
from src.dataloader import load_data
from src.preprocessing import scale_data
from pathlib import Path
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
import mlflow
import dagshub
from dotenv import load_dotenv
import os
import yaml
from typing import Any, Dict
import numpy as np

# Load environment variables
load_dotenv()

def tune_model() -> None:
    # splitting the data
    data_path = Path("data/raw/creditcard.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run the data download script first.")
    X, y = load_data(str(data_path))
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=params["random_state"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,  # 0.2 * 0.8 = 0.16 of original data
    stratify=y_train,
    random_state=params["random_state"]
    )
    print(f"Data split completed:")
    print(f"   Training samples:   {X_train.shape[0]:,} (64% of total)")
    print(f"   Validation samples: {X_val.shape[0]:,} (16% of total)")
    print(f"   Test samples:       {X_test.shape[0]:,} (20% of total)")
    print(f"   Positive samples in train: {sum(y_train==1):,} ({100*sum(y_train==1)/len(y_train):.4f}%)")
    print(f"   Positive samples in val:   {sum(y_val==1):,} ({100*sum(y_val==1)/len(y_val):.4f}%)")
    print(f"   Positive samples in test:  {sum(y_test==1):,} ({100*sum(y_test==1)/len(y_test):.4f}%)")

    # Scale the data
    print("\nScaling data...")
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)
    print("Data scaling completed")

    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    RANDOM_STATE = params["random_state"]

    # Initialize MLflow tracking
    repo_owner = os.getenv('DAGSHUB_USERNAME')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')
    
    if not repo_owner or not repo_name:
        print("DAGSHUB_USERNAME and DAGSHUB_REPO_NAME not found in .env file")
        print("Running without MLflow tracking - parameters will still be saved locally")
        mlflow_available = False
    else:
        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            mlflow.set_experiment("xgboost-tuning")
            mlflow_available = True
            print("✅ MLflow tracking initialized")
        except Exception as e:
            print(f"Failed to initialize MLflow tracking: {e}")
            print("Running without MLflow tracking - parameters will still be saved locally")
            mlflow_available = False

    def objective_xgb(trial: optuna.Trial) -> float:
        """Optuna objective function for XGBoost"""
        config_params = params  # Rename to avoid conflict with trial params
        trial_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': scale_pos_weight,
            'random_state': config_params["random_state"],
            'n_jobs': -1,
            'tree_method': 'hist', # Use hist for faster training similar to LightGBM
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr'
        }

        if mlflow_available:
            with mlflow.start_run(nested=True, run_name=f"XGB_Trial_{trial.number}"):
                model = XGBClassifier(**trial_params)
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
                y_proba = model.predict_proba(X_val_scaled)[:, 1]
                score = average_precision_score(y_val, y_proba)

                mlflow.log_params(trial_params)
                mlflow.log_metric("val_auprc", score)
        else:
            model = XGBClassifier(**trial_params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            score = average_precision_score(y_val, y_proba)

        return score
    
    print("\nStarting XGBoost Optuna tuning")
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(objective_xgb, n_trials=20)
    print(f"\nBest XGBoost AUPRC (Val): {study_xgb.best_value:.4f}")
    
    # Save best parameters to params.yaml
    print("\nSaving best parameters to params.yaml...")
    best_params = study_xgb.best_params
    
    # Load current params
    config_path = Path("src/config/params.yaml")
    with open(config_path) as f:
        current_params = yaml.safe_load(f)
    
    # Ensure models section exists and is a dict
    if 'models' not in current_params or current_params['models'] is None:
        current_params['models'] = {}
    
    # Update XGBoost parameters
    current_params['models']['xgboost'] = best_params
    
    # Save back to file
    with open(config_path, 'w') as f:
        yaml.dump(current_params, f, default_flow_style=False, sort_keys=False)
    
    print("Best parameters saved to params.yaml")
    print(f"Best parameters: {best_params}")


if __name__ == '__main__':
    tune_model()
