import logging
import logging.handlers
import os
from ..config import settings

def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    if settings.LOG_FILE:
        os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)
    
    # Create formatters
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if LOG_FILE is specified
    if settings.LOG_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    
    return logger

# Initialize logger
logger = setup_logging() 