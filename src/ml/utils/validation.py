import os
from pathlib import Path
from typing import Dict, Any, List, Set
from PIL import Image
from .logging import setup_logging

logger = setup_logging(__name__)

class DataValidation:
    """Common data validation utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data validation.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        
    def validate_image_file(self, image_path: str) -> bool:
        """Validate single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            bool: True if image is valid
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return False
                
            # Check file format
            if Path(image_path).suffix.lower() not in self.supported_formats:
                logger.error(f"Unsupported format: {image_path}")
                return False
                
            # Check if image can be opened
            with Image.open(image_path) as img:
                img.verify()
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {str(e)}")
            return False
            
    def validate_label_file(self, label_path: Path) -> bool:
        """Validate label file format.
        
        Args:
            label_path: Path to label file
            
        Returns:
            bool: True if label is valid
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                # Check if line is empty
                if not line.strip():
                    continue
                    
                # Split line into components
                parts = line.strip().split()
                
                # Check number of components
                if len(parts) != 5:
                    return False
                    
                # Check if all components are numbers
                try:
                    [float(x) for x in parts]
                except ValueError:
                    return False
                    
                # Check if values are in valid range
                class_id, x, y, w, h = map(float, parts)
                if not (0 <= class_id < self.config['num_classes']):
                    return False
                if not all(0 <= v <= 1 for v in [x, y, w, h]):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating label {label_path}: {str(e)}")
            return False
            
    def validate_dataset(self, data_dir: str) -> Dict[str, Any]:
        """Validate entire dataset.
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': [],
            'missing_labels': [],
            'corrupt_images': [],
            'format_errors': []
        }
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
            
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(list(Path(data_dir).rglob(f"*{ext}")))
            
        results['total_images'] = len(image_files)
        
        # Validate each image
        for img_path in image_files:
            try:
                # Check file format
                if img_path.suffix.lower() not in self.supported_formats:
                    results['format_errors'].append(str(img_path))
                    continue
                    
                # Check if image can be opened
                if not self.validate_image_file(str(img_path)):
                    results['corrupt_images'].append(str(img_path))
                    continue
                    
                # Check if label exists
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    results['missing_labels'].append(str(img_path))
                    logger.warning(f"Missing label for image: {img_path}")
                    continue
                    
                # Validate label format
                if not self.validate_label_file(label_path):
                    results['invalid_images'].append(str(img_path))
                    logger.warning(f"Invalid label format: {label_path}")
                    continue
                    
                results['valid_images'] += 1
                
            except Exception as e:
                logger.error(f"Error validating {img_path}: {str(e)}")
                results['invalid_images'].append(str(img_path))
                
        return results 