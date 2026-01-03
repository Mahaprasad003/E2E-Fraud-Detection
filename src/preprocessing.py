from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path

def scale_data(X_train, *datasets):
    """
    Scales only the 'Amount' column using RobustScaler, fits it on X_train,
    transforms all provided datasets, saves the scaler, and returns the transformed datasets.

    Parameters:
    X_train (pd.DataFrame): The training data to fit the scaler.
    *datasets: Variable number of additional datasets to transform (e.g., X_val, X_test).

    Returns:
    tuple: Datasets with only 'Amount' column scaled, in the same order as provided.
    """
    import pandas as pd

    # Fit scaler on training Amount column only
    scaler = RobustScaler()
    scaler.fit(X_train[['Amount']])

    # Function to scale Amount column for a dataset
    def scale_amount_column(X):
        X_scaled = X.copy()
        X_scaled['Amount'] = scaler.transform(X[['Amount']])
        return X_scaled

    # Scale all datasets
    scaled_datasets = [scale_amount_column(X_train)]
    for dataset in datasets:
        scaled_datasets.append(scale_amount_column(dataset))

    model_dir = Path("models/production")
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return tuple(scaled_datasets)