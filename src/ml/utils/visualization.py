import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

def draw_detections(image: np.ndarray,
                   boxes: np.ndarray,
                   classes: List[str],
                   scores: List[float],
                   show_scores: bool = True,
                   color_map: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
    """Draw detection boxes and labels on image.
    
    Args:
        image: Input BGR image
        boxes: Array of bounding boxes
        classes: List of class names
        scores: List of confidence scores
        show_scores: Whether to show confidence scores
        color_map: Optional mapping of class names to BGR colors
        
    Returns:
        Image with drawings
    """
    vis_img = image.copy()
    
    # Default color map if none provided
    if color_map is None:
        color_map = {
            'suture': (0, 255, 0),    # Green
            'knot': (255, 0, 0),      # Blue 
            'incision': (0, 0, 255),  # Red
            'tail': (255, 255, 0)     # Cyan
        }
    
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for class
        color = color_map.get(cls.lower(), (255, 255, 255))
        
        # Draw box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{cls} {score:.2f}" if show_scores else cls
        y = y1 - 10 if y1 > 20 else y1 + 10
        cv2.putText(vis_img, label, (x1, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_img

def visualize_measurements(image: np.ndarray,
                         points: List[Tuple[float, float]],
                         measurements: Dict[str, float],
                         pixels_per_mm: float,
                         incision_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> np.ndarray:
    """Visualize measurement results on image.
    
    Args:
        image: Input BGR image
        points: List of measurement points
        measurements: Dictionary of measurements
        pixels_per_mm: Conversion factor for mm
        incision_line: Optional incision line endpoints
        
    Returns:
        Image with measurements visualization
    """
    vis_img = image.copy()
    
    # Draw points
    for i, point in enumerate(points):
        x, y = map(int, point)
        cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(vis_img, f"P{i}", (x+5, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw incision line if provided
    if incision_line:
        p1, p2 = incision_line
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)
        cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw measurements
    y_offset = 30
    for key, value in measurements.items():
        text = f"{key}: {value:.2f}"
        cv2.putText(vis_img, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    return vis_img

def plot_analysis_results(measurements: Dict[str, float],
                         show_plot: bool = True,
                         save_path: Optional[str] = None):
    """Plot analysis results as charts.
    
    Args:
        measurements: Dictionary of measurements
        show_plot: Whether to display the plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    plt.subplot(121)
    plt.bar(measurements.keys(), measurements.values())
    plt.xticks(rotation=45)
    plt.title("Suture Analysis Measurements")
    
    # Create box plot for distributions
    plt.subplot(122)
    if 'spacing_std' in measurements and 'angle_std' in measurements:
        plt.boxplot([measurements['spacing_std'], measurements['angle_std']],
                   labels=['Spacing Variation', 'Angle Variation'])
        plt.title("Measurement Distributions")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()