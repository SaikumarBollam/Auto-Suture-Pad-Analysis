"""Task-specific configuration for suture detection."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
from .config import BaseConfig, ConfigError

@dataclass
class SutureModelConfig:
    """Suture detection specific model configuration."""
    size: str
    anchors: List[List[int]]
    backbone: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SutureModelConfig":
        """Create SutureModelConfig from dictionary."""
        try:
            return cls(
                size=data["size"],
                anchors=data["anchors"],
                backbone=data["backbone"]
            )
        except KeyError as e:
            raise ConfigError(f"Missing required suture model configuration: {e}")

@dataclass
class SutureDataConfig:
    """Suture detection specific data configuration."""
    class_names: List[str]
    class_ids: Dict[str, int]
    format: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SutureDataConfig":
        """Create SutureDataConfig from dictionary."""
        try:
            return cls(
                class_names=data["class_names"],
                class_ids=data["class_ids"],
                format=data["format"]
            )
        except KeyError as e:
            raise ConfigError(f"Missing required suture data configuration: {e}")

@dataclass
class SutureConfig:
    """Suture detection specific configuration."""
    model: SutureModelConfig
    data: SutureDataConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SutureConfig":
        """Create SutureConfig from dictionary."""
        try:
            return cls(
                model=SutureModelConfig.from_dict(data["model"]),
                data=SutureDataConfig.from_dict(data["data"])
            )
        except KeyError as e:
            raise ConfigError(f"Missing required suture configuration: {e}")

class SutureDetectionConfig(Config[SutureConfig]):
    """Configuration for suture detection task."""
    
    def validate(self) -> bool:
        """Validate suture detection specific configuration."""
        if not super().validate():
            return False
        
        try:
            # Validate suture-specific model configuration
            assert self.task_specific.model.size in ["n", "s", "m", "l", "x"], "Invalid model size"
            assert len(self.task_specific.model.anchors) > 0, "Anchors must not be empty"
            
            # Validate suture-specific data configuration
            assert len(self.task_specific.data.class_names) > 0, "Class names must not be empty"
            assert len(self.task_specific.data.class_ids) > 0, "Class IDs must not be empty"
            assert self.task_specific.data.format in ["yolo", "coco", "pascal_voc"], "Invalid data format"
            
            return True
        except AssertionError as e:
            logger.error(f"Suture configuration validation failed: {e}")
            return False 