import torch
from typing import Dict, Any, Optional
from .transforms import get_transforms

class DataProcessing:
    """Common data processing utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processing.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        transforms = get_transforms(config)
        self.train_transform = transforms['train']
        self.val_transform = transforms['val']
        
    def process_train(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process training data.
        
        Args:
            data: Training data
            
        Returns:
            Dict[str, torch.Tensor]: Processed training data
        """
        data['image'] = self.train_transform(data['image'])
        return data
        
    def process_val(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process validation data.
        
        Args:
            data: Validation data
            
        Returns:
            Dict[str, torch.Tensor]: Processed validation data
        """
        data['image'] = self.val_transform(data['image'])
        return data
        
    def process_inference(self, data: torch.Tensor) -> torch.Tensor:
        """Process inference data.
        
        Args:
            data: Inference data
            
        Returns:
            torch.Tensor: Processed inference data
        """
        return self.val_transform(data) 