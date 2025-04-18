"""Logging utilities for ML pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(name: str,
                 level: str = "INFO",
                 log_file: Optional[Path] = None,
                 format_str: Optional[str] = None) -> logging.Logger:
    """Setup logging for a module.
    
    Args:
        name: Name of the logger
        level: Logging level
        log_file: Optional path to log file
        format_str: Optional format string for log messages
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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