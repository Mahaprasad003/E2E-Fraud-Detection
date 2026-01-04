# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning project for detecting fraudulent credit card transactions. This project covers the full ML lifecycle from exploratory data analysis to model deployment with a production-ready REST API.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Development](#model-development)
- [Results](#results)
- [API](#api)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

## Overview

Credit card fraud is a significant problem in the financial industry. This project builds a machine learning system that can identify fraudulent transactions in real-time. The solution is designed to be production-ready with containerized deployment, monitoring, and a REST API for integration.

**Key Features:**
- Handles highly imbalanced data (0.17% fraud rate)
- Hyperparameter optimization using Optuna
- Model tracking with MLflow and DagsHub
- REST API with FastAPI
- Prometheus metrics and Grafana dashboards
- Docker containerization for deployment

## Dataset

The dataset is from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) challenge.

- **284,807** transactions over 2 days
- **492** fraudulent transactions (0.17%)
- **30** features (V1-V28 from PCA, plus Time and Amount)
- Features V1-V28 are principal components to protect user privacy

## Exploratory Data Analysis

Initial exploration revealed several important characteristics of the data:

**Class Imbalance**

The dataset is heavily imbalanced with fraudulent transactions making up only 0.17% of all transactions. This required special handling during model training using class weights and stratified sampling.

**Transaction Amounts**

| Statistic | Legitimate | Fraud |
|-----------|------------|-------|
| Mean | \$88.29 | \$122.21 |
| Median | \$22.00 | \$9.25 |
| Max | \$25,691 | \$2,125 |

Fraudulent transactions have a higher mean but lower maximum amount compared to legitimate ones. This suggests fraudsters tend to avoid very large transactions that might trigger alerts.

**Feature Distribution**

The t-SNE visualization showed that fraudulent transactions are scattered throughout the feature space rather than forming a single cluster. This makes simple rule-based detection difficult and justifies using machine learning.

**Correlation Analysis**

Features V1-V28 show minimal correlation with each other, which is expected since they are PCA components. This means we can use all features without worrying about multicollinearity.

## Model Development

Multiple models were trained and evaluated to find the best performer:

1. **Logistic Regression** - Baseline model with class weights
2. **LightGBM** - Gradient boosting with Optuna tuning (30 trials)
3. **XGBoost** - Gradient boosting with Optuna tuning (30 trials)
4. **CatBoost** - Ordered boosting with Optuna tuning (30 trials)
5. **Isolation Forest** - Unsupervised anomaly detection

All models were evaluated using **AUPRC (Area Under Precision-Recall Curve)** as the primary metric. AUPRC is more appropriate than accuracy or ROC-AUC for highly imbalanced datasets.

**Preprocessing:**
- RobustScaler applied to the Amount feature
- Time feature dropped (not predictive over 2-day period)
- Stratified train/validation/test split (64/16/20)

**Hyperparameter Tuning:**

Optuna was used for Bayesian optimization of hyperparameters. Each model was tuned for 30 trials with early stopping to prevent overfitting.

## Results

### Model Comparison

| Model | Test AUPRC | Test ROC-AUC |
|-------|------------|--------------|
| **XGBoost** | **0.8524** | **0.9832** |
| CatBoost | 0.8467 | 0.9819 |
| LightGBM | 0.8243 | 0.9798 |
| Logistic Regression | 0.6782 | 0.9738 |
| Isolation Forest | 0.3012 | 0.9456 |

XGBoost achieved the best performance and was selected for production deployment.

### XGBoost Classification Report

Using the optimal F1 threshold:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Legitimate | 1.00 | 1.00 | 1.00 |
| Fraud | 0.87 | 0.82 | 0.84 |

**Cross-Validation Results:**

5-fold stratified CV on the training set confirmed model stability:
- CV AUPRC: 0.8512 (plus/minus 0.0234)
- Optimal trees: 151

### Feature Importance

Top 5 most important features for fraud detection:
1. V14
2. V17
3. V12
4. V10
5. V16

## API

The model is served through a FastAPI application with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/health\` | GET | Health check |
| \`/api/predict\` | POST | Single transaction prediction |
| \`/api/predict/batch\` | POST | Batch predictions |
| \`/metrics\` | GET | Prometheus metrics |
| \`/docs\` | GET | Interactive API documentation |

### Example Request

\`\`\`bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.35, "V3": 1.68, "V4": 0.45,
    "V5": -0.12, "V6": -0.89, "V7": -0.21, "V8": 0.08,
    "V9": -0.23, "V10": 0.07, "V11": 0.23, "V12": -0.34,
    "V13": 0.11, "V14": -0.56, "V15": 0.22, "V16": -0.09,
    "V17": 0.14, "V18": -0.03, "V19": 0.01, "V20": 0.08,
    "V21": -0.01, "V22": -0.02, "V23": 0.01, "V24": 0.09,
    "V25": 0.03, "V26": -0.01, "V27": 0.01, "V28": 0.02,
    "Amount": 149.62
  }'
\`\`\`

### Example Response

\`\`\`json
{
  "is_fraud": 0,
  "probability": 0.0234,
  "threshold": 0.5
}
\`\`\`

## Project Structure

\`\`\`
fraud-detection/
├── app/                          # FastAPI application
│   ├── api/
│   │   ├── endpoints/            # API route handlers
│   │   └── schemas/              # Pydantic models
│   ├── middleware/               # Logging middleware
│   └── main.py                   # Application entry point
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Processed data
├── models/
│   └── production/               # Deployed model artifacts
│       ├── xgb_model.joblib
│       └── scaler.joblib
├── notebooks/
│   ├── exploration.ipynb         # EDA notebook
│   └── modelling.ipynb           # Model training notebook
├── src/
│   ├── modeling/                 # Training and prediction code
│   ├── config.py                 # Configuration
│   └── preprocessing.py          # Data preprocessing
├── docker-compose.yml            # Multi-container setup
├── Dockerfile                    # API container
├── requirements.txt              # Full dependencies
├── requirements-prod.txt         # Production dependencies
└── render.yaml                   # Render deployment config
\`\`\`

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)

### Local Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
\`\`\`

2. Create a virtual environment:
\`\`\`bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Download the dataset from Kaggle and place it in \`data/raw/creditcard.csv\`

## Usage

### Running the API Locally

\`\`\`bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
\`\`\`

Visit \`http://localhost:8000/docs\` for the interactive API documentation.

### Running with Docker Compose

This starts the API along with Prometheus and Grafana for monitoring:

\`\`\`bash
docker compose up --build
\`\`\`

Services:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Running the Notebooks

\`\`\`bash
jupyter notebook notebooks/
\`\`\`

## Deployment

### Render

The project includes a \`render.yaml\` for easy deployment to Render:

1. Push your code to GitHub
2. Connect your repo to Render
3. Render will automatically deploy using the configuration

### Other Platforms

The Dockerfile is compatible with any container platform:
- AWS ECS / Fargate
- Google Cloud Run
- Azure Container Apps
- Kubernetes

## Tech Stack

| Category | Technology |
|----------|------------|
| ML Framework | XGBoost, scikit-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Experiment Tracking | MLflow, DagsHub |
| Hyperparameter Tuning | Optuna |
| Monitoring | Prometheus, Grafana |
| Containerization | Docker, Docker Compose |
| Data Processing | pandas, NumPy |

## Future Improvements

- Add API authentication for production use
- Implement model retraining pipeline
- Add data drift detection
- Set up CI/CD with GitHub Actions
- Deploy to a cloud provider

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: [Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Project template inspired by [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
