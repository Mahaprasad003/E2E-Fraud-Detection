from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.endpoints import predict, health
from app.middleware.logging import LoggingMiddleware

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fradulent transactions",
    version="1.0.0"
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

app.include_router(predict.router, prefix="/api", tags=["predictions"])
app.include_router(health.router, prefix="/api", tags=["health"])

@app.get("/")
def root():
    return {"message": "Welcome to the Fraud Detection API"}
