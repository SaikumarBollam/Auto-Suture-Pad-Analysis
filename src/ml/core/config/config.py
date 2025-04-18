"""Generic configuration management for ML pipelines."""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, TypeVar, Generic
from dataclasses import dataclass, field
from ..utils.logging import setup_logging

logger = setup_logging(__name__)

T = TypeVar('T')

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    # Required fields (no default values)
    name: str
    environment: str
    device: str
    
    # Optional fields (with default values)
    seed: int = 42
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create BaseConfig from dictionary."""
        try:
            return cls(
                name=data["name"],
                environment=data.get("environment", "dev"),
                device=data.get("device", "cpu"),
                seed=data.get("seed", 42),
                debug=data.get("debug", False)
            )
        except KeyError as e:
            raise ConfigError(f"Missing required base configuration: {e}")

@dataclass
class ModelConfig:
    """Model configuration settings."""
    # Required fields (no default values)
    type: str
    input_size: List[int]
    num_classes: int
    
    # Optional fields (with default values)
    pretrained: bool = True
    weights_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        try:
            return cls(
                type=data["type"],
                input_size=data["input_size"],
                num_classes=data["num_classes"],
                pretrained=data.get("pretrained", True),
                weights_path=Path(data["weights_path"]) if "weights_path" in data else None
            )
        except KeyError as e:
            raise ConfigError(f"Missing required model configuration: {e}")

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    # Required fields (no default values)
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_split: float
    checkpoint_frequency: int
    early_stopping: Dict[str, Any]
    learning_rate_scheduler: Dict[str, Any]
    num_workers: int
    pin_memory: bool
    checkpoint_dir: Path
    log_dir: Path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create TrainingConfig from dictionary."""
        try:
            return cls(
                batch_size=data["batch_size"],
                epochs=data["epochs"],
                learning_rate=data["learning_rate"],
                weight_decay=data["weight_decay"],
                validation_split=data["validation_split"],
                checkpoint_frequency=data["checkpoint_frequency"],
                early_stopping=data["early_stopping"],
                learning_rate_scheduler=data["learning_rate_scheduler"],
                num_workers=data["num_workers"],
                pin_memory=data["pin_memory"],
                checkpoint_dir=Path(data["checkpoint_dir"]),
                log_dir=Path(data["log_dir"])
            )
        except KeyError as e:
            raise ConfigError(f"Missing required training configuration: {e}")

@dataclass
class DataConfig:
    """Data configuration settings."""
    # Required fields (no default values)
    train_path: Path
    val_path: Path
    test_path: Path
    image_size: List[int]
    augment: bool
    normalize: bool
    cache_images: bool
    dataset: Dict[str, Any]
    loader: Dict[str, Any]
    preprocessing: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Create DataConfig from dictionary."""
        try:
            return cls(
                train_path=Path(data["train_path"]),
                val_path=Path(data["val_path"]),
                test_path=Path(data["test_path"]),
                image_size=data["image_size"],
                augment=data["augment"],
                normalize=data["normalize"],
                cache_images=data["cache_images"],
                dataset=data["dataset"],
                loader=data["loader"],
                preprocessing=data["preprocessing"]
            )
        except KeyError as e:
            raise ConfigError(f"Missing required data configuration: {e}")

@dataclass
class TaskConfig:
    """Task-specific configuration settings."""
    # Required fields (no default values)
    task_type: str
    task_params: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """Create TaskConfig from dictionary."""
        try:
            return cls(
                task_type=data["task_type"],
                task_params=data["task_params"]
            )
        except KeyError as e:
            raise ConfigError(f"Missing required task configuration: {e}")

class Config:
    """Main configuration class that composes all other configs."""
    
    def __init__(self, 
                 base: BaseConfig,
                 model: ModelConfig,
                 training: TrainingConfig,
                 data: DataConfig,
                 task: Optional[TaskConfig] = None):
        """Initialize configuration."""
        self.base = base
        self.model = model
        self.training = training
        self.data = data
        self.task = task
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        try:
            Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(self.training.log_dir).mkdir(parents=True, exist_ok=True)
            Path(self.data.train_path).mkdir(parents=True, exist_ok=True)
            Path(self.data.val_path).mkdir(parents=True, exist_ok=True)
            Path(self.data.test_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigError(f"Failed to create required directories: {e}")
    
    @classmethod
    def from_yaml(cls, 
                 base_config_path: Path,
                 model_config_path: Path, 
                 training_config_path: Path, 
                 data_config_path: Path,
                 task_config_path: Optional[Path] = None,
                 environment: str = None) -> "Config":
        """Load configuration from YAML files."""
        try:
            # Determine environment
            if environment is None:
                environment = os.getenv("ML_ENV", "dev")
            
            # Load configurations
            base_config = cls._load_yaml(base_config_path, environment)
            model_config = cls._load_yaml(model_config_path, environment)
            training_config = cls._load_yaml(training_config_path, environment)
            data_config = cls._load_yaml(data_config_path, environment)
            
            # Load task config if provided
            task_config = None
            if task_config_path:
                task_config = cls._load_yaml(task_config_path, environment)
            
            # Create config instances
            base = BaseConfig.from_dict(base_config)
            model = ModelConfig.from_dict(model_config)
            training = TrainingConfig.from_dict(training_config)
            data = DataConfig.from_dict(data_config)
            task = TaskConfig.from_dict(task_config) if task_config else None
            
            return cls(base=base, model=model, training=training, data=data, task=task)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigError(f"Configuration loading failed: {e}")
    
    @staticmethod
    def _load_yaml(path: Path, environment: str) -> Dict[str, Any]:
        """Load and process YAML file with environment overrides."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        # Apply environment-specific overrides if they exist
        if environment in config:
            base_config = config.copy()
            base_config.update(config[environment])
            return base_config
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration settings.
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate model configuration
            assert self.model.num_classes > 0, "Number of classes must be positive"
            assert all(s > 0 for s in self.model.input_size), "Input size must be positive"
            
            # Validate training configuration
            assert self.training.batch_size > 0, "Batch size must be positive"
            assert self.training.epochs > 0, "Number of epochs must be positive"
            assert 0 < self.training.validation_split < 1, "Validation split must be between 0 and 1"
            assert self.training.checkpoint_frequency > 0, "Checkpoint frequency must be positive"
            
            # Validate data configuration
            assert all(s > 0 for s in self.data.image_size), "Image size must be positive"
            assert self.data.train_path.exists(), f"Training path {self.data.train_path} does not exist"
            assert self.data.val_path.exists(), f"Validation path {self.data.val_path} does not exist"
            
            return True
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dict containing configuration settings
        """
        config_dict = {
            "base": self.base.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
        }
        
        if self.task:
            config_dict["task"] = self.task.__dict__
        
        return config_dict
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(environment={self.base.environment}, model={self.model.__dict__}, training={self.training.__dict__}, data={self.data.__dict__})"