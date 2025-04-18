from fastapi import APIRouter, Depends
from typing import Dict, Any
import torch
import redis
import mlflow
from minio import Minio
import os

router = APIRouter()

def get_redis_client():
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))

def get_mlflow_client():
    return mlflow.tracking.MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))

def get_minio_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False
    )

@router.get("/")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy"}

@router.get("/gpu")
async def gpu_check() -> Dict[str, Any]:
    """Check GPU availability and status."""
    return {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@router.get("/redis")
async def redis_check(redis_client: redis.Redis = Depends(get_redis_client)) -> Dict[str, Any]:
    """Check Redis connection."""
    try:
        redis_client.ping()
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/mlflow")
async def mlflow_check(mlflow_client = Depends(get_mlflow_client)) -> Dict[str, Any]:
    """Check MLflow connection."""
    try:
        mlflow_client.list_experiments()
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/minio")
async def minio_check(minio_client: Minio = Depends(get_minio_client)) -> Dict[str, Any]:
    """Check MinIO connection."""
    try:
        minio_client.list_buckets()
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/full")
async def full_health_check(
    redis_client: redis.Redis = Depends(get_redis_client),
    mlflow_client = Depends(get_mlflow_client),
    minio_client: Minio = Depends(get_minio_client)
) -> Dict[str, Any]:
    """Comprehensive health check of all services."""
    return {
        "api": {"status": "healthy"},
        "gpu": await gpu_check(),
        "redis": await redis_check(redis_client),
        "mlflow": await mlflow_check(mlflow_client),
        "minio": await minio_check(minio_client)
    } 