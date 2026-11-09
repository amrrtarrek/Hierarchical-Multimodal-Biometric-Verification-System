# ==============================================================================
# Dockerfile: Hierarchical Multimodal Biometric Verification System
# Member 1 Workstream: Face Modality Lead (Branch: feature/face)
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies required by OpenCV & GUI rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache layer: install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source code, models, and test suites
COPY . .

# Default container entrypoint runs the full biometric unit test suite
CMD ["python", "-m", "pytest", "tests", "-v"]

