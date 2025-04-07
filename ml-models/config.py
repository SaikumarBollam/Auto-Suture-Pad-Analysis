"""Configuration management for suture analysis."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the suture analysis project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If config_path is provided but doesn't exist
            ValueError: If configuration validation fails
        """
        self.config_path = config_path
        if config_path is not None:
            self.load_config(config_path)
        else:
            self.config = self._get_default_config()
            
        # Validate configuration
        self._validate_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'model': {
                'type': 'yolo',
                'model_size': 'l',
                'input_size': (640, 640),
                'num_classes': 2,
                'pretrained': True,
                'weights_path': 'weights/yolov12l.pt'
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'plots': True
            },
            'data': {
                'train_path': 'data/train',
                'val_path': 'data/val',
                'test_path': 'data/test',
                'cache': False,
                'image_weights': False,
                'rect': False,
                'single_cls': False,
                'pad': 0.0,
                'min_items': 0,
                'close_mosaic': 10
            },
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_det': 300,
                'plots': True
            },
            'deployment': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'half': False,
                'dnn': False,
                'workers': 8
            }
        }
        
    def _validate_config(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        try:
            # Validate model configuration
            model_config = self.config['model']
            if model_config['type'] not in ['yolo']:
                raise ValueError(f"Invalid model type: {model_config['type']}")
                
            if model_config['model_size'] not in ['n', 's', 'm', 'l', 'x']:
                raise ValueError(f"Invalid model size: {model_config['model_size']}")
                
            if not isinstance(model_config['input_size'], tuple) or len(model_config['input_size']) != 2:
                raise ValueError("Input size must be a tuple of length 2")
                
            if model_config['input_size'][0] % 32 != 0 or model_config['input_size'][1] % 32 != 0:
                raise ValueError("Input size must be divisible by 32")
                
            if model_config['num_classes'] < 1:
                raise ValueError("Number of classes must be at least 1")
                
            # Validate training configuration
            train_config = self.config['training']
            if train_config['epochs'] < 1:
                raise ValueError("Number of epochs must be at least 1")
                
            if train_config['batch_size'] < 1:
                raise ValueError("Batch size must be at least 1")
                
            if train_config['learning_rate'] <= 0:
                raise ValueError("Learning rate must be positive")
                
            # Validate data configuration
            data_config = self.config['data']
            for path in ['train_path', 'val_path', 'test_path']:
                if not Path(data_config[path]).exists():
                    logger.warning(f"{path} does not exist: {data_config[path]}")
                    
            # Validate validation configuration
            val_config = self.config['validation']
            if not 0 <= val_config['conf_threshold'] <= 1:
                raise ValueError("Confidence threshold must be between 0 and 1")
                
            if not 0 <= val_config['iou_threshold'] <= 1:
                raise ValueError("IoU threshold must be between 0 and 1")
                
            # Validate deployment configuration
            deploy_config = self.config['deployment']
            if deploy_config['device'] not in ['cpu', 'cuda']:
                raise ValueError("Device must be either 'cpu' or 'cuda'")
                
            if deploy_config['workers'] < 1:
                raise ValueError("Number of workers must be at least 1")
                
            logger.info("Configuration validation successful")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise ValueError(f"Configuration validation failed: {str(e)}")
        
    def load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If config_path doesn't exist
            ValueError: If configuration file is invalid
        """
        try:
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise ValueError(f"Failed to load configuration: {str(e)}")
            
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
            
        Raises:
            ValueError: If no configuration path is specified
        """
        try:
            if config_path is None:
                config_path = self.config_path
            if config_path is None:
                raise ValueError("No configuration path specified")
                
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
                
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise ValueError(f"Failed to save configuration: {str(e)}")
            
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration
        """
        return self.config['model']
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration.
        
        Returns:
            Dict[str, Any]: Training configuration
        """
        return self.config['training']
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration.
        
        Returns:
            Dict[str, Any]: Data configuration
        """
        return self.config['data']
        
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration.
        
        Returns:
            Dict[str, Any]: Validation configuration
        """
        return self.config['validation']
        
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration.
        
        Returns:
            Dict[str, Any]: Deployment configuration
        """
        return self.config['deployment']