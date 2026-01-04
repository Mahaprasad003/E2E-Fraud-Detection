from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}


@router.get("/ready")
def readiness_check():
    """Check if the API is ready to serve requests (model loaded)."""
    try:
        from app.api.dependencies.model_loader import model, scaler
        if model is not None and scaler is not None:
            return {"status": "ready", "model_loaded": True}
        return {"status": "not_ready", "model_loaded": False}
    except Exception as e:
        return {"status": "error", "detail": str(e)}