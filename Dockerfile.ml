# Use Ubuntu base image instead of CUDA
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy requirements
COPY requirements.txt .
COPY src/ml/requirements.txt ./src/ml/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r src/ml/requirements.txt

# Copy the ML code
COPY src/ml ./src/ml
COPY config ./config

# Create necessary directories
RUN mkdir -p /app/data /app/weights /app/logs /app/mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose necessary ports
EXPOSE 8001
EXPOSE 8888
EXPOSE 6006

# Run ML service
CMD ["python", "-m", "src.ml.applications.inference"] 