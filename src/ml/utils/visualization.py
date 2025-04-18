"""Visualization utilities for suture analysis.

This module provides functionality for visualizing suture angles, measurements,
and other analysis results.
"""

import cv2
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from datetime import datetime
from math import atan2, degrees
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..core.inference import Measurement, MeasurementType

@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    colors: Dict[str, Tuple[int, int, int]] = None
    line_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 2
    measurement_offset: int = 20
    angle_offset: int = 30
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'stitch': (0, 255, 0),      # Green
                'incision': (255, 0, 0),    # Blue
                'knot': (0, 0, 255),        # Red
                'tail': (255, 255, 0),      # Cyan
                'measurement': (255, 255, 255),  # White
                'angle': (255, 0, 255),     # Magenta
                'scale': (0, 255, 255)      # Yellow
            }

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

class SutureVisualizer(Visualizer):
    """Visualizer specialized for suture measurements."""
    
    def __init__(self, output_dir: str, config: Optional[VisualizationConfig] = None):
        """Initialize the suture visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            config: Visualization configuration
        """
        super().__init__(output_dir)
        self.config = config or VisualizationConfig()
        
    def visualize_measurements(self, 
                             image: np.ndarray,
                             measurements: List[Measurement],
                             detections: List[Dict[str, Any]]) -> np.ndarray:
        """Visualize all suture measurements on the image.
        
        Args:
            image: Input image
            measurements: List of measurements
            detections: List of detections
            
        Returns:
            np.ndarray: Annotated image
        """
        img = image.copy()
        
        # Draw detections
        for det in detections:
            self._draw_detection(img, det)
            
        # Draw measurements
        for measurement in measurements:
            self._draw_measurement(img, measurement, detections)
            
        return img
        
    def _draw_detection(self, img: np.ndarray, detection: Dict[str, Any]) -> None:
        """Draw a single detection on the image.
        
        Args:
            img: Image to draw on
            detection: Detection to draw
        """
        color = self.config.colors[detection['class']]
        bbox = detection['bbox']
        
        # Draw bounding box
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), color, self.config.line_thickness)
                     
        # Draw label
        label = f"{detection['class']}: {detection['confidence']:.2f}"
        cv2.putText(img, label, (int(bbox[0]), int(bbox[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, color, self.config.font_thickness)
                   
        # Draw oriented bounding box if available
        if 'obb' in detection and detection['obb'] is not None:
            obb = detection['obb']
            for i in range(4):
                cv2.line(img, tuple(obb[i].astype(int)), 
                        tuple(obb[(i + 1) % 4].astype(int)), color, self.config.line_thickness)
                        
    def _draw_measurement(self, 
                         img: np.ndarray, 
                         measurement: Measurement,
                         detections: List[Dict[str, Any]]) -> None:
        """Draw a single measurement on the image.
        
        Args:
            img: Image to draw on
            measurement: Measurement to draw
            detections: List of detections for reference
        """
        color = self.config.colors['measurement']
        
        if measurement.type == MeasurementType.ALPHA:
            # Draw angle measurement
            self._draw_angle_measurement(img, measurement, detections)
        else:
            # Draw distance measurement
            self._draw_distance_measurement(img, measurement, detections)
            
    def _draw_angle_measurement(self,
                               img: np.ndarray,
                               measurement: Measurement,
                               detections: List[Dict[str, Any]]) -> None:
        """Draw angle measurement.
        
        Args:
            img: Image to draw on
            measurement: Angle measurement
            detections: List of detections
        """
        # Find corresponding stitch
        stitch = next((d for d in detections if d['class'] == 'stitch'), None)
        if stitch is None or 'obb' not in stitch:
            return
            
        # Get angle arc points
        center = np.mean(stitch['obb'], axis=0)
        radius = 50
        start_angle = 0
        end_angle = measurement.value
        
        # Draw angle arc
        cv2.ellipse(img, tuple(center.astype(int)), (radius, radius), 0,
                    start_angle, end_angle, self.config.colors['angle'], self.config.line_thickness)
                    
        # Draw angle label
        label = f"{measurement.value:.1f}°"
        cv2.putText(img, label, tuple((center + [radius + 10, 0]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   self.config.colors['angle'], self.config.font_thickness)
                   
    def _draw_distance_measurement(self,
                                  img: np.ndarray,
                                  measurement: Measurement,
                                  detections: List[Dict[str, Any]]) -> None:
        """Draw distance measurement.
        
        Args:
            img: Image to draw on
            measurement: Distance measurement
            detections: List of detections
        """
        # Get relevant detections based on measurement type
        if measurement.type in [MeasurementType.L1, MeasurementType.R1]:
            stitch = next((d for d in detections if d['class'] == 'stitch'), None)
            incision = next((d for d in detections if d['class'] == 'incision'), None)
            if stitch is None or incision is None:
                return
                
            # Get points
            if measurement.type == MeasurementType.L1:
                point1 = np.array([stitch['bbox'][0], (stitch['bbox'][1] + stitch['bbox'][3]) / 2])
            else:
                point1 = np.array([stitch['bbox'][2], (stitch['bbox'][1] + stitch['bbox'][3]) / 2])
                
            point2 = np.mean(incision['obb'], axis=0)
            
        elif measurement.type in [MeasurementType.T1A, MeasurementType.T1B]:
            tail = next((d for d in detections if d['class'] == 'tail'), None)
            if tail is None:
                return
                
            point1 = tail['points'][0]
            point2 = tail['points'][-1]
            
        else:
            return
            
        # Draw measurement line
        cv2.line(img, tuple(point1.astype(int)), tuple(point2.astype(int)),
                self.config.colors['measurement'], self.config.line_thickness)
                
        # Draw measurement label
        label = f"{measurement.value:.1f}mm"
        mid_point = (point1 + point2) / 2
        cv2.putText(img, label, tuple(mid_point.astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   self.config.colors['measurement'], self.config.font_thickness)
                   
    def generate_report(self,
                       image: np.ndarray,
                       measurements: List[Measurement],
                       detections: List[Dict[str, Any]],
                       output_path: str) -> None:
        """Generate a comprehensive PDF report.
        
        Args:
            image: Input image
            measurements: List of measurements
            detections: List of detections
            output_path: Path to save the report
        """
        # Create PDF
        with PdfPages(output_path) as pdf:
            # Create figure
            fig = plt.figure(figsize=(11, 8.5))
            
            # Plot 1: Original image with measurements
            plt.subplot(2, 1, 1)
            img_with_measurements = self.visualize_measurements(image, measurements, detections)
            plt.imshow(cv2.cvtColor(img_with_measurements, cv2.COLOR_BGR2RGB))
            plt.title('Suture Measurements Visualization')
            plt.axis('off')
            
            # Plot 2: Measurement statistics
            plt.subplot(2, 1, 2)
            self._plot_measurement_statistics(measurements)
            
            # Save page
            pdf.savefig(fig)
            plt.close()
            
            # Create summary page
            fig = plt.figure(figsize=(11, 8.5))
            self._plot_summary_statistics(measurements)
            pdf.savefig(fig)
            plt.close()
            
    def _plot_measurement_statistics(self, measurements: List[Measurement]) -> None:
        """Plot measurement statistics.
        
        Args:
            measurements: List of measurements
        """
        # Group measurements by type
        grouped = {}
        for m in measurements:
            if m.type not in grouped:
                grouped[m.type] = []
            grouped[m.type].append(m.value)
            
        # Create box plots
        plt.boxplot([grouped[t] for t in grouped.keys()],
                   labels=[t.value for t in grouped.keys()])
        plt.title('Measurement Statistics')
        plt.ylabel('Value (mm)')
        plt.xticks(rotation=45)
        
    def _plot_summary_statistics(self, measurements: List[Measurement]) -> None:
        """Plot summary statistics.
        
        Args:
            measurements: List of measurements
        """
        # Calculate statistics
        stats = {}
        for m_type in MeasurementType:
            type_measurements = [m for m in measurements if m.type == m_type]
            if type_measurements:
                values = [m.value for m in type_measurements]
                stats[m_type.value] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'within_standard': sum(1 for m in type_measurements if m.is_within_standard)
                }
                
        # Create table
        plt.table(cellText=[[f"{stats[t]['mean']:.2f} ± {stats[t]['std']:.2f}",
                            f"{stats[t]['min']:.2f}",
                            f"{stats[t]['max']:.2f}",
                            f"{stats[t]['within_standard']}/{len([m for m in measurements if m.type.value == t])}"]
                           for t in stats.keys()],
                 colLabels=['Mean ± Std', 'Min', 'Max', 'Within Standard'],
                 rowLabels=list(stats.keys()),
                 loc='center')
                 
        plt.title('Measurement Summary')
        plt.axis('off')
        
    def save_measurements(self,
                         measurements: List[Measurement],
                         output_path: str) -> None:
        """Save measurements to JSON file.
        
        Args:
            measurements: List of measurements
            output_path: Path to save the JSON file
        """
        # Convert measurements to dictionary
        data = {
            'timestamp': datetime.now().isoformat(),
            'measurements': [{
                'type': m.type.value,
                'value': m.value,
                'unit': m.unit,
                'standard_mean': m.standard_mean,
                'standard_deviation': m.standard_deviation,
                'is_within_standard': m.is_within_standard
            } for m in measurements]
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

def calculate_angle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int
) -> float:
    """Calculate the angle between two points relative to vertical.
    
    Args:
        x1, y1: First point coordinates (normalized)
        x2, y2: Second point coordinates (normalized)
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        Angle in degrees relative to vertical
    """
    x1, y1 = x1 * image_width, y1 * image_height
    x2, y2 = x2 * image_width, y2 * image_height
    dx = x2 - x1
    dy = y2 - y1
    angle_radians = atan2(dy, dx)
    angle_degrees = abs(degrees(angle_radians) - 90)
    return angle_degrees

def visualize_suture_angles(
    image_path: str,
    label_path: str,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """Visualize suture angles on an image.
    
    Args:
        image_path: Path to the input image
        label_path: Path to the label file
        output_path: Optional path to save the visualization
        
    Returns:
        Tuple of (visualized image, list of (class_id, angle) pairs)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from path {image_path}")

    image_height, image_width, _ = image.shape
    suture_classes = {4, 5, 6, 7}  # Class IDs for sutures
    angles = []

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])
            if class_id in suture_classes:
                x1, y1, x2, y2, *_ = map(float, values[1:])
                angle = calculate_angle(x1, y1, x2, y2, image_width, image_height)
                angles.append((class_id, angle))
                
                # Convert normalized coordinates to pixel coordinates
                x1, y1 = int(x1 * image_width), int(y1 * image_height)
                x2, y2 = int(x2 * image_width), int(y2 * image_height)
                
                # Draw line and label
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 10)
                cv2.putText(
                    image,
                    f"Class {class_id}",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    if output_path:
        cv2.imwrite(output_path, image)

    return image, angles

def display_angle_visualization(
    image_path: str,
    label_path: str,
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """Display suture angle visualization using matplotlib.
    
    Args:
        image_path: Path to the input image
        label_path: Path to the label file
        figsize: Figure size for display
    """
    image, angles = visualize_suture_angles(image_path, label_path)
    
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    for i, (class_id, angle) in enumerate(angles, start=1):
        print(f"Suture {i} (Class {class_id}): Angle to vertical = {angle:.2f} degrees")

def batch_visualize_suture_angles(
    input_dir: str,
    output_dir: str,
    detections: Dict[str, List[Tuple[int, float, np.ndarray]]],
    class_names: Dict[int, str],
    confidence_threshold: float = 0.5,
    num_workers: int = 4
) -> Dict[str, str]:
    """Visualize suture angles for multiple images.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save visualized images
        detections: Dictionary mapping image names to detection lists
        class_names: Dictionary mapping class IDs to names
        confidence_threshold: Minimum confidence to display
        num_workers: Number of worker threads
        
    Returns:
        Dictionary mapping image names to paths of visualized images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualized_paths = {}
    
    def process_image(img_name: str) -> Optional[str]:
        img_path = input_dir / img_name
        if not img_path.exists():
            return None
            
        img_detections = detections.get(img_name, [])
        if not img_detections:
            return None
            
        output_path = output_dir / f"{img_name}_visualized.jpg"
        visualize_suture_angles(
            str(img_path),
            img_detections,
            class_names,
            str(output_path),
            confidence_threshold
        )
        return str(output_path)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            img_name: executor.submit(process_image, img_name)
            for img_name in detections.keys()
        }
        
        for img_name, future in tqdm(futures.items(), total=len(futures), desc="Visualizing images"):
            try:
                path = future.result()
                if path:
                    visualized_paths[img_name] = path
            except Exception as e:
                print(f"Error visualizing image {img_name}: {e}")
    
    return visualized_paths

def batch_display_angle_visualizations(
    image_paths: Dict[str, str],
    measurements: Dict[str, Dict[str, float]],
    num_cols: int = 2
) -> None:
    """Display multiple angle visualizations in a grid.
    
    Args:
        image_paths: Dictionary mapping image names to paths of visualized images
        measurements: Dictionary mapping image names to measurement dictionaries
        num_cols: Number of columns in the display grid
    """
    num_images = len(image_paths)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 5 * num_rows))
    
    for idx, (img_name, img_path) in enumerate(image_paths.items(), 1):
        plt.subplot(num_rows, num_cols, idx)
        
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            
        img_measurements = measurements.get(img_name, {})
        title = f"{img_name}\n"
        for key, value in img_measurements.items():
            title += f"{key}: {value:.2f}°\n"
        plt.title(title.strip())
        plt.axis('off')
    
    plt.tight_layout()
    plt.show() 