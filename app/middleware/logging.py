"""Logging middleware for request/response tracking."""
import time
import logging
from uuid import uuid4
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logger
logger = logging.getLogger("fraud_detection_api")
logger.setLevel(logging.INFO)

# Console handler with formatting
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests and outgoing responses."""

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid4())[:8]
        
        # Log request
        logger.info(
            f"[{request_id}] → {request.method} {request.url.path} "
            f"| Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Time the request
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            logger.info(
                f"[{request_id}] ← {response.status_code} "
                f"| Duration: {duration_ms:.2f}ms"
            )
            
            # Add request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] ✗ Error: {str(e)} "
                f"| Duration: {duration_ms:.2f}ms"
            )
            raise
