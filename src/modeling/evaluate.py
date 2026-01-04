import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)
import joblib
from pathlib import Path
from config import Config
from dataloader import load_data
from sklearn.model_selection import train_test_split
from utils.params import load_params
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def evaluate_model() -> None:
    """
    Evaluate the trained model and create visualizations.

    Loads the saved model, prepares test data, calculates metrics,
    and saves plots to reports/figures/.
    """
    print("Starting model evaluation...")

    # Create output directory
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load model and scaler
    print("Loading model and scaler...")
    try:
        model = joblib.load(Config.MODEL_PATH)
        scaler = joblib.load(Config.SCALER_PATH)
        print("Model and scaler loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        return

    # Load and prepare test data
    print("Preparing test data...")
    params = load_params()

    # Load data
    data_path = Path("data/raw/creditcard.csv")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    X, y = load_data(str(data_path))

    # Split data (same as training)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=params['random_state'],
        stratify=y
    )

    # Scale test data using loaded scaler
    X_test_scaled = X_test.copy()
    if 'Amount' in X_test_scaled.columns:
        X_test_scaled['Amount'] = scaler.transform(X_test_scaled[['Amount']])

    print(f"📈 Test set: {len(X_test)} samples")

    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= Config.PREDICTION_THRESHOLD).astype(int)

    # Calculate metrics
    print("Calculating metrics...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Create plots
    print("Creating visualizations...")

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Classification Report Heatmap
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame(class_report).transpose()
    # Remove support column for visualization
    metrics_df = metrics_df.drop('support', axis=1, errors='ignore')

    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
                linewidths=0.5, cbar=True)
    plt.title('Classification Report Metrics')
    plt.tight_layout()
    plt.savefig(figures_dir / 'classification_report.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Legitimate', bins=50, density=True)
    plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Fraud', bins=50, density=True)
    plt.axvline(x=Config.PREDICTION_THRESHOLD, color='red', linestyle='--',
                label=f'Threshold ({Config.PREDICTION_THRESHOLD})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    print(" Evaluation Summary:")
    print(f"   • ROC-AUC: {roc_auc:.4f}")
    print(f"   • PR-AUC: {pr_auc:.4f}")
    print(f"   • Precision (Fraud): {class_report['1']['precision']:.4f}")
    print(f"   • Recall (Fraud): {class_report['1']['recall']:.4f}")
    print(f"   • F1-Score (Fraud): {class_report['1']['f1-score']:.4f}")

    print(f"Plots saved to {figures_dir}/")
    print("Evaluation complete!")

if __name__ == "__main__":
    evaluate_model()