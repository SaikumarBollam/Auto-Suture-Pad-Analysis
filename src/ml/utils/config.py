"""Configuration management utilities."""

import os
from pathlib import Path
import yaml
from typing import Dict, Any, Optional, List, Tuple

class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: str):
        """Initialize config manager.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.config.get('ml', {}).get('model', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.config.get('ml', {}).get('preprocessing', {})
    
    def get_measurement_config(self) -> Dict[str, Any]:
        """Get measurement-specific configuration."""
        return self.config.get('measurements', {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis parameters configuration."""
        return self.config.get('analysis', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config.get('visualization', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_reference_points(self) -> Optional[List[Tuple[float, float]]]:
        """Get calibration reference points if configured."""
        measurement_config = self.get_measurement_config()
        calibration = measurement_config.get('pixel_mm_calibration', {})
        
        if calibration.get('enabled', False):
            points = calibration.get('reference_points', [])
            if len(points) == 2:
                return [(p[0], p[1]) for p in points]
        return None
    
    def get_class_names(self) -> Dict[str, List[str]]:
        """Get configured class names by type."""
        analysis_config = self.get_analysis_config()
        return {
            'suture': analysis_config.get('suture_classes', []),
            'knot': analysis_config.get('knot_classes', []),
            'incision': analysis_config.get('incision_classes', [])
        }
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get quality assessment thresholds."""
        analysis_config = self.get_analysis_config()
        return analysis_config.get('quality_thresholds', {})
    
    def get_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get visualization colors."""
        vis_config = self.get_visualization_config()
        return vis_config.get('colors', {})
    
    def save_config(self, config: Dict[str, Any]):
        """Save updated configuration.
        
        Args:
            config: Configuration dictionary to save
        """
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)