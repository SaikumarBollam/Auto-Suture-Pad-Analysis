"""Logging configuration and utilities."""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

class SutureLogger:
    """Logger for suture analysis results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize logger with configuration.
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self.log_dir = Path(config.get('save_path', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file handler
        log_file = self.log_dir / f"suture_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Set up logger
        self.logger = logging.getLogger('suture_analysis')
        self.logger.setLevel(config.get('level', 'INFO'))
        self.logger.addHandler(file_handler)
        
        # Create measurement log file
        if config.get('save_measurements', True):
            self.measurement_file = self.log_dir / 'measurements.jsonl'
        else:
            self.measurement_file = None
            
    def log_detection(self, image_name: str, boxes: np.ndarray, 
                     classes: list, scores: list):
        """Log detection results.
        
        Args:
            image_name: Name of the processed image
            boxes: Detected bounding boxes
            classes: Detected class names
            scores: Detection confidence scores
        """
        self.logger.info(f"Processing image: {image_name}")
        self.logger.info(f"Found {len(boxes)} detections")
        
        for box, cls, score in zip(boxes, classes, scores):
            self.logger.debug(
                f"Detection: class={cls}, score={score:.3f}, "
                f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
            )
            
    def log_measurements(self, image_name: str, measurements: Dict[str, Any]):
        """Log measurement results.
        
        Args:
            image_name: Name of the processed image
            measurements: Dictionary of measurements
        """
        self.logger.info(f"Measurements for {image_name}:")
        for key, value in measurements.items():
            if isinstance(value, (np.ndarray, list)):
                self.logger.info(f"{key}: mean={np.mean(value):.2f}, std={np.std(value):.2f}")
            else:
                self.logger.info(f"{key}: {value}")
                
        if self.measurement_file:
            # Append to JSONL file
            with open(self.measurement_file, 'a') as f:
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'image': image_name,
                    'measurements': measurements
                }
                json.dump(record, f)
                f.write('\n')
                
    def save_visualization(self, image_name: str, image: np.ndarray):
        """Save visualization image if enabled.
        
        Args:
            image_name: Name of the processed image
            image: Visualization image array
        """
        if self.config.get('save_images', True):
            vis_dir = self.log_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            # Remove file extension and add timestamp
            base_name = Path(image_name).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = vis_dir / f"{base_name}_{timestamp}.png"
            
            cv2.imwrite(str(output_path), image)
            self.logger.info(f"Saved visualization to {output_path}")
            
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log error with context.
        
        Args:
            error: Exception that occurred
            context: Optional context dictionary
        """
        if context:
            self.logger.error(f"Error: {str(error)}, Context: {context}")
        else:
            self.logger.error(f"Error: {str(error)}")
            
    def close(self):
        """Close logger and handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)