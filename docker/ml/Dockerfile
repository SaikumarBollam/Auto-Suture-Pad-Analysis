# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

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
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Expose necessary ports
EXPOSE 8001
EXPOSE 8888
EXPOSE 6006

# Run ML service
CMD ["python", "-m", "src.ml.applications.inference"] 