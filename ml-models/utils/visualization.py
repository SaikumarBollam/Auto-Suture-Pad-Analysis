import cv2
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

class Visualizer:
    def __init__(self, output_dir: str):
        """Initialize the visualizer with an output directory.
        
        Args:
            output_dir (str): Directory to save visualized images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_predictions(self, 
                            image_path: str, 
                            predictions: List[Tuple[float, float, float, float, float, float]],
                            class_names: Optional[List[str]] = None) -> np.ndarray:
        """Visualize predictions on a single image.
        
        Args:
            image_path (str): Path to the input image
            predictions (List[Tuple]): List of predictions in YOLO format (class, x_center, y_center, width, height, confidence)
            class_names (List[str], optional): List of class names for labeling
            
        Returns:
            np.ndarray: Annotated image
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        img_h, img_w = img.shape[:2]
        
        for pred in predictions:
            cls, x_center, y_center, width, height, conf = pred
            
            # Convert YOLO format to bounding box coordinates
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            if class_names:
                label = f"{class_names[int(cls)]}: {conf:.2f}"
            else:
                label = f"Class {int(cls)}: {conf:.2f}"
                
            # Draw label
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img
    
    def save_visualization(self, img: np.ndarray, filename: str) -> None:
        """Save the visualized image.
        
        Args:
            img (np.ndarray): Annotated image
            filename (str): Name of the output file
        """
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), img)
        
    def process_directory(self, 
                         image_dir: str, 
                         predictions_dir: str,
                         class_names: Optional[List[str]] = None) -> None:
        """Process all images in a directory with their corresponding predictions.
        
        Args:
            image_dir (str): Directory containing input images
            predictions_dir (str): Directory containing prediction files
            class_names (List[str], optional): List of class names for labeling
        """
        image_dir = Path(image_dir)
        predictions_dir = Path(predictions_dir)
        
        for img_file in image_dir.glob("*.jpg"):
            pred_file = predictions_dir / f"{img_file.stem}.txt"
            
            if not pred_file.exists():
                continue
                
            # Read predictions
            predictions = []
            with open(pred_file, 'r') as f:
                for line in f:
                    predictions.append(tuple(map(float, line.strip().split())))
                    
            # Visualize and save
            img = self.visualize_predictions(str(img_file), predictions, class_names)
            self.save_visualization(img, img_file.name) 