from fastapi import Depends, HTTPException, status
from functools import lru_cache
import os
from typing import Optional, Generator
import torch
import logging
from pathlib import Path

from ..ml.core.inference.infer import InferencePipeline
from ..ml.models.model import get_model
from ..ml.core.config.config import Config
from ...config import load_config

logger = logging.getLogger(__name__)

@lru_cache()
def get_model_config() -> dict:
    """Load and cache model configuration."""
    try:
        config_path = os.getenv('MODEL_CONFIG_PATH', 'config/model_config.yaml')
        return load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
        raise HTTPException(status_code=500, detail="Model configuration error")

@lru_cache()
def get_inference_config() -> dict:
    """Load and cache inference configuration."""
    try:
        config_path = os.getenv('INFERENCE_CONFIG_PATH', 'config/inference_config.yaml')
        return load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load inference config: {e}")
        raise HTTPException(status_code=500, detail="Inference configuration error")

@lru_cache()
def get_model(config: dict = Depends(get_model_config)) -> torch.nn.Module:
    """Get and cache the model instance."""
    try:
        model = get_model(config)
        weights_path = os.getenv('MODEL_WEIGHTS_PATH', 'weights/best_model.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        model.load_weights(weights_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Model loading error")

@lru_cache()
def get_inference_pipeline() -> Generator[InferencePipeline, None, None]:
    try:
        config = Config()
        model = get_model(config)
        pipeline = InferencePipeline(model, config)
        yield pipeline
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize inference pipeline: {str(e)}"
        )

def get_device() -> str:
    """Get the device to use for inference."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_batch_size() -> int:
    """Get the batch size for inference."""
    return int(os.getenv('INFERENCE_BATCH_SIZE', '1')) 