import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ultralytics import YOLO
import logging

from .base import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YOLOModel(BaseModel):
    """YOLO model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the YOLO model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Initialize YOLO model
        try:
            self.yolo = YOLO(f"yolov12{config['model_size']}.pt")
            logger.info(f"Initialized YOLOv12-{config['model_size']} model")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}")
            raise RuntimeError(f"Failed to initialize YOLO model: {str(e)}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.yolo(x)
        
    def predict(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Make predictions with the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional prediction parameters
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Update prediction parameters
        predict_config = self.config.get('validation', {})
        predict_config.update(kwargs)
        
        # Make predictions
        results = self.yolo.predict(x, **predict_config)
        
        # Process results
        predictions = []
        for result in results:
            prediction = {
                'boxes': result.boxes.xyxy.cpu().numpy(),
                'scores': result.boxes.conf.cpu().numpy(),
                'labels': result.boxes.cls.cpu().numpy()
            }
            predictions.append(prediction)
            
        return {'predictions': predictions}
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Training step of the model.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            torch.Tensor: Loss tensor
        """
        # Forward pass
        results = self.yolo(x)
        
        # Calculate loss
        loss = sum(result.loss for result in results)
        
        return loss
        
    def save_weights(self, path: str) -> None:
        """Save model weights.
        
        Args:
            path: Path to save weights
        """
        try:
            self.yolo.save(path)
            logger.info(f"YOLO model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save YOLO model: {str(e)}")
            raise RuntimeError(f"Failed to save YOLO model: {str(e)}")
            
    def load_weights(self, path: str) -> None:
        """Load model weights.
        
        Args:
            path: Path to load weights from
        """
        try:
            self.yolo = YOLO(path)
            logger.info(f"YOLO model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}") 