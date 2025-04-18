import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SutureModel(nn.Module):
    """Base class for suture analysis models."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        logger.info("Initializing base suture model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save model weights.
        
        Args:
            path: Path to save model weights
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Saved model weights to {path}")
        
    def load(self, path: str) -> None:
        """Load model weights.
        
        Args:
            path: Path to model weights
        """
        try:
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {path}")
        except Exception as e:
            logger.error(f"Failed to load model weights from {path}: {str(e)}")
            raise

class YOLOv12SutureModel(SutureModel):
    """YOLOv12-based suture detection and classification model."""
    def __init__(
        self,
        model_size: str = 'l',  # 'n', 's', 'm', 'l', 'x'
        pretrained: bool = True,
        num_classes: int = 2,
        **kwargs
    ):
        """Initialize YOLOv12 model.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes
            
        Raises:
            ValueError: If model_size is invalid
            RuntimeError: If model initialization fails
        """
        super().__init__(num_classes=num_classes)
        self.model_size = model_size
        self.pretrained = pretrained
        
        # Validate model size
        if model_size not in ['n', 's', 'm', 'l', 'x']:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of: n, s, m, l, x")
            
        # Validate number of classes
        if num_classes < 1:
            raise ValueError(f"Invalid number of classes: {num_classes}. Must be at least 1")
            
        logger.info(f"Initializing YOLOv12 model (size: {model_size}, classes: {num_classes})")
        
        try:
            # Initialize YOLOv12 model
            self.model = YOLO(f'yolov12{model_size}.pt' if pretrained else f'yolov12{model_size}.yaml')
            
            # Modify detection head for custom number of classes
            self.model.model.nc = num_classes
            logger.info(f"Set number of classes to {num_classes}")
            
            # Set model to evaluation mode
            self.eval()
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv12 model: {str(e)}")
            raise RuntimeError(f"Failed to initialize YOLOv12 model: {str(e)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        try:
            # Run inference
            results = self.model(x)
            
            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes.xyxy  # [N, 4]
                if boxes is not None:
                    confs = result.boxes.conf  # [N]
                    classes = result.boxes.cls  # [N]
                    detections.append(torch.cat([boxes, confs.unsqueeze(1), classes.unsqueeze(1)], dim=1))
                else:
                    detections.append(torch.empty((0, 6), device=x.device))
                    
            return detections
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise RuntimeError(f"Forward pass failed: {str(e)}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions on input data."""
        return self.forward(x)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step.
        
        Args:
            batch: Dictionary containing:
                - images: Input images [B, C, H, W]
                - targets: Target boxes and labels [B, N, 5] where 5 is [class, x1, y1, x2, y2]
                
        Returns:
            Dict[str, float]: Training metrics
            
        Raises:
            RuntimeError: If training step fails
        """
        try:
            self.train()
            
            # Forward pass
            loss_dict = self.model(batch['images'], batch['targets'])
            
            # Extract losses
            losses = {
                'loss': loss_dict['loss'].item(),
                'box_loss': loss_dict['box_loss'].item(),
                'cls_loss': loss_dict['cls_loss'].item(),
                'dfl_loss': loss_dict['dfl_loss'].item()
            }
            
            self.eval()
            return losses
            
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            raise RuntimeError(f"Training step failed: {str(e)}")

    def train(
        self,
        train_data: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        **kwargs
    ) -> Dict:
        """Train the model.
        
        Args:
            train_data: Path to training data configuration file
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
            **kwargs: Additional training arguments
            
        Returns:
            Dict: Training results
        """
        results = self.model.train(
            data=train_data,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            **kwargs
        )
        return results

class CNNSutureModel(SutureModel):
    """CNN-based suture analysis model."""
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = 'resnet50',
        pretrained: bool = True
    ):
        super().__init__(num_classes=num_classes)
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

def get_model(
    model_type: str = 'yolo',
    model_size: str = 'l',
    num_classes: int = 2,
    pretrained: bool = True
) -> SutureModel:
    """Factory function to create a suture model.
    
    Args:
        model_type: Type of model to create ('yolo' or 'cnn')
        model_size: Size of YOLO model ('n', 's', 'm', 'l', 'x')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        SutureModel: Initialized model
    """
    logger.info(f"Creating model (type: {model_type}, size: {model_size}, classes: {num_classes})")
    
    if model_type.lower() == 'yolo':
        return YOLOv12SutureModel(
            model_size=model_size,
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif model_type.lower() == 'cnn':
        return CNNSutureModel(
            num_classes=num_classes,
            **kwargs
        )
    else:
        logger.error(f"Invalid model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}") 