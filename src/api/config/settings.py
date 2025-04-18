from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Model Settings
    MODEL_PATH: str = "weights/yolov8n.pt"
    MODEL_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]  # TODO: Configure properly for production
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings() 