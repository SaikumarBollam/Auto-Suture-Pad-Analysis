from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import os
from datetime import timedelta

class Settings(BaseSettings):
    """API Settings loaded from environment variables."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Suture Analysis API"
    API_KEY: str = os.getenv("API_KEY", "your-api-key-here")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Settings
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "weights/latest.pth")
    CONFIG_PATH: str = os.getenv("CONFIG_PATH", "config/model_config.yaml")
    MODEL_UPDATE_CHECK_INTERVAL: int = 3600  # Check for model updates every hour
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]  # In production, replace with specific origins
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100  # Requests per minute
    RATE_LIMIT_WINDOW: int = 60  # Window in seconds
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "api.log")
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = os.getenv("ENABLE_PERFORMANCE_MONITORING", "True").lower() == "true"
    PERFORMANCE_LOG_INTERVAL: int = 300  # Log performance metrics every 5 minutes
    
    # Preprocessing Settings
    PREPROCESSING_OPTIONS: Dict[str, bool] = {
        "grayscale": False,
        "gaussian_blur": False,
        "contour_extraction": False,
        "edge_detection": False
    }
    
    # Model Hyperparameters
    HYPERPARAMETERS: Dict[str, Any] = {
        "learning_rate": 0.005,
        "weight_decay": 0.0001,
        "iou_threshold": 0.25,
        "epochs": 100,
        "batch_size": 16
    }
    
    # Image Settings
    IMAGE_SIZE: tuple = (640, 640)
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/tiff"]
    
    # Performance Metrics
    PERFORMANCE_METRICS: List[str] = [
        "precision",
        "recall",
        "mAP",
        "f1_score"
    ]
    
    # Health Check Settings
    HEALTH_CHECK_INTERVAL: int = 300  # Check health every 5 minutes
    MODEL_HEALTH_TIMEOUT: int = 30  # Timeout for model health check in seconds
    
    class Config:
        case_sensitive = True

settings = Settings() 