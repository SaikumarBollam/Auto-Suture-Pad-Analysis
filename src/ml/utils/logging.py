"""Logging configuration for the suture detection package."""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_str: Log message format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 