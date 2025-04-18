from fastapi import HTTPException
from typing import Any, Dict, Optional

class ModelError(Exception):
    """Base exception for model-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class ConfigError(Exception):
    """Exception for configuration-related errors."""
    def __init__(self, message: str, config_path: Optional[str] = None):
        super().__init__(message)
        self.config_path = config_path

class ValidationError(Exception):
    """Exception for validation-related errors."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field

def handle_model_error(error: ModelError) -> HTTPException:
    """Convert ModelError to HTTPException."""
    return HTTPException(
        status_code=500,
        detail={
            "error": "Model Error",
            "message": str(error),
            "details": error.details
        }
    )

def handle_config_error(error: ConfigError) -> HTTPException:
    """Convert ConfigError to HTTPException."""
    return HTTPException(
        status_code=500,
        detail={
            "error": "Configuration Error",
            "message": str(error),
            "config_path": error.config_path
        }
    )

def handle_validation_error(error: ValidationError) -> HTTPException:
    """Convert ValidationError to HTTPException."""
    return HTTPException(
        status_code=400,
        detail={
            "error": "Validation Error",
            "message": str(error),
            "field": error.field
        }
    )

def handle_generic_error(error: Exception) -> HTTPException:
    """Convert generic Exception to HTTPException."""
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal Server Error",
            "message": str(error)
        }
    ) 