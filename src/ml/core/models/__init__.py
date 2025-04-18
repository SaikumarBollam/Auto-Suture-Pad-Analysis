from typing import Dict, Any
from .base import BaseModel
from .yolo import YOLOModel

def get_model(config: Dict[str, Any]) -> BaseModel:
    """Get model instance based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        BaseModel: Model instance
        
    Raises:
        ValueError: If model type is invalid
    """
    model_type = config.get('type', 'yolo')
    
    if model_type == 'yolo':
        return YOLOModel(config)
    else:
        raise ValueError(f"Invalid model type: {model_type}") 