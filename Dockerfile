FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# CACHE LAYER 1: Python dependencies
# ============================================================

# Install only the lightweight dependencies needed for API serving
RUN pip install --no-cache-dir \
    fastapi==0.138.2 \
    pydantic==2.12.5 \
    joblib==1.5.3 \
    scikit-learn==1.9.0 \
    uvicorn==0.52.3 \
    prometheus-fastapi-instrumentator

# ============================================================
# CACHE LAYER 2: Application files
# ============================================================

COPY api/ ./api/
COPY models/ ./models/

# ============================================================

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
