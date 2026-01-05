"""
Stratified K-Fold Cross-Validation for XGBoost fraud detection model.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import scale_data
from dataloader import load_data
from utils.params import load_params


def run_cross_validation(
    num_boost_round: int = 5000,
    nfold: int = 5,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50
) -> dict:
    
    params = load_params()
    random_state = params['random_state']
    xgb_params = params['models']['xgboost'].copy()
    
    data_path = Path("data/raw/creditcard.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    X, y = load_data(str(data_path))
    
    test_size = params.get('preprocessing', {}).get('test_size', 0.2)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    X_train_scaled, X_test_scaled = scale_data(X_train_full, X_test)
    
    scale_pos_weight = sum(y_train_full == 0) / sum(y_train_full == 1)
    
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_full)
    
    xgb_cv_params = xgb_params.copy()
    xgb_cv_params.pop('n_estimators', None)
    xgb_cv_params.pop('n_jobs', None)
    
    xgb_cv_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'nthread': -1,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist'
    })
    
    print(f"Running {nfold}-Fold Stratified CV (early_stopping={early_stopping_rounds})")
    
    cv_results = xgb.cv(
        params=xgb_cv_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        stratified=True,
        metrics='aucpr',
        early_stopping_rounds=early_stopping_rounds,
        seed=random_state,
        verbose_eval=verbose_eval
    )
    
    best_iteration = len(cv_results)
    best_score = cv_results['test-aucpr-mean'].iloc[-1]
    best_std = cv_results['test-aucpr-std'].iloc[-1]
    train_score = cv_results['train-aucpr-mean'].iloc[-1]
    train_std = cv_results['train-aucpr-std'].iloc[-1]
    
    print(f"\nOptimal trees: {best_iteration}")
    print(f"Test AUPRC:  {best_score:.4f} +/- {best_std:.4f}")
    print(f"Train AUPRC: {train_score:.4f} +/- {train_std:.4f}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'XGBoost',
        'cv_folds': nfold,
        'stratified': True,
        'random_state': random_state,
        'parameters': xgb_cv_params,
        'results': {
            'optimal_trees': best_iteration,
            'test_auprc_mean': float(best_score),
            'test_auprc_std': float(best_std),
            'train_auprc_mean': float(train_score),
            'train_auprc_std': float(train_std),
        },
        'cv_history': {
            'test_auprc_mean': cv_results['test-aucpr-mean'].tolist(),
            'test_auprc_std': cv_results['test-aucpr-std'].tolist(),
            'train_auprc_mean': cv_results['train-aucpr-mean'].tolist(),
            'train_auprc_std': cv_results['train-aucpr-std'].tolist()
        }
    }
    
    save_results(results, cv_results)
    
    return results


def save_results(results: dict, cv_results_df: pd.DataFrame) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = reports_dir / f"cv_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    csv_path = reports_dir / f"cv_history_{timestamp}.csv"
    cv_results_df.to_csv(csv_path, index=True)
    
    latest_json = reports_dir / "cv_results_latest.json"
    with open(latest_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    latest_csv = reports_dir / "cv_history_latest.csv"
    cv_results_df.to_csv(latest_csv, index=True)
    
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    results = run_cross_validation()
    print(f"\nRecommended n_estimators: {results['results']['optimal_trees']}")
