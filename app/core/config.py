import os
from pydantic import BaseSettings
import yaml

with open("src/config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

class Settings(BaseSettings):
    app_name: str = "Fraud Detection API"
    model_path: str = "models/production/xgb_model.joblib"
    scaler_path: str = "models/production/scaler.joblib"
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()