import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """Base class for all models in the project."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Forward method must be implemented by subclass")
        
    def predict(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Make predictions with the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional prediction parameters
            
        Returns:
            Dict[str, Any]: Prediction results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Predict method must be implemented by subclass")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training step of the model.
        
        Args:
            batch: Dictionary containing input and target tensors
            
        Returns:
            torch.Tensor: Loss tensor
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Train step method must be implemented by subclass")
        
    def val_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Validation step of the model.
        
        Args:
            batch: Dictionary containing input and target tensors
            
        Returns:
            torch.Tensor: Loss tensor
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Validation step method must be implemented by subclass")
        
    def save_weights(self, path: str) -> None:
        """Save model weights.
        
        Args:
            path: Path to save weights
            
        Raises:
            RuntimeError: If saving fails
        """
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Model weights saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model weights: {str(e)}")
            raise RuntimeError(f"Failed to save model weights: {str(e)}")
            
    def load_weights(self, path: str) -> None:
        """Load model weights.
        
        Args:
            path: Path to load weights from
            
        Raises:
            RuntimeError: If loading fails
        """
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            logger.info(f"Model weights loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {str(e)}")
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
            
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration
        """
        return self.config 