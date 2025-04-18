# Build stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY src/ml/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ml .

# Create necessary directories
RUN mkdir -p /app/data /app/weights /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; torch.cuda.is_available()"

# Run inference service
CMD ["python", "-m", "core.inference"] 