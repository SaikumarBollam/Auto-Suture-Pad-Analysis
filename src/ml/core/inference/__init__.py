"""Suture analysis inference package."""

import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from ..models.model import get_model
from ..utils.visualization import Visualizer
from .infer import SutureDetector

class InferencePipeline:
    """Pipeline for model inference and prediction visualization."""
    
    def __init__(self, model_type: str = "yolo", model_path: Optional[str] = None):
        """Initialize the inference pipeline.
        
        Args:
            model_type (str): Type of model to use
            model_path (str, optional): Path to model weights
        """
        self.model = get_model(model_type)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """Make predictions on a single image.
        
        Args:
            image (torch.Tensor): Input image tensor
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        with torch.no_grad():
            image = image.to(self.device)
            if isinstance(self.model, YOLOSutureModel):
                results = self.model.predict(image)
                return {
                    'boxes': results['boxes'].cpu().numpy(),
                    'scores': results['scores'].cpu().numpy(),
                    'labels': results['labels'].cpu().numpy()
                }
            else:
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                return {
                    'class': predicted.cpu().numpy(),
                    'probabilities': torch.softmax(outputs, dim=1).cpu().numpy()
                }
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         class_names: Optional[List[str]] = None) -> None:
        """Process all images in a directory and save visualizations.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save visualizations
            class_names (List[str], optional): Names of classes for visualization
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = Visualizer(str(output_dir))
        
        for image_path in input_dir.glob('*.jpg'):
            # Load and preprocess image
            image = self._load_image(image_path)
            
            # Make prediction
            results = self.predict(image)
            
            # Visualize and save
            visualizer.visualize_prediction(
                image_path=image_path,
                predictions=results,
                class_names=class_names
            )
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess an image.
        
        Args:
            image_path (Path): Path to the image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # TODO: Implement proper image loading and preprocessing
        # This is a placeholder implementation
        image = torch.randn(1, 3, 224, 224)  # Replace with actual image loading
        return image

__all__ = ['InferencePipeline', 'SutureDetector'] 