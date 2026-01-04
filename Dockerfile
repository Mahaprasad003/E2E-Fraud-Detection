# Fraud Detection API Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (production only)
COPY requirements-prod.txt ./

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
