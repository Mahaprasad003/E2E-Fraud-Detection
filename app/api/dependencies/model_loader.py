from app.utils.helpers import load_model_and_scaler

# Load model and scaler once when the module is imported
model, scaler = load_model_and_scaler()

def get_model():
    """Dependency to get the loaded model."""
    return model

def get_scaler():
    """Dependency to get the loaded scaler."""
    return scaler
