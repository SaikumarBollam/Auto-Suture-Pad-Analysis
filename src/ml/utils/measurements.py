"""Measurement calculation utilities for suture analysis.

This module provides functionality for calculating various measurements
from suture analysis results, including lengths, distances, and angles.
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from datetime import datetime

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        p1: First point coordinates (x, y)
        p2: Second point coordinates (x, y)
        
    Returns:
        Euclidean distance between points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_suture_length(
    box: np.ndarray,
    pixels_per_mm: float
) -> Tuple[float, float]:
    """Calculate suture length in pixels and millimeters.
    
    Args:
        box: Bounding box coordinates
        pixels_per_mm: Conversion factor from pixels to millimeters
        
    Returns:
        Tuple of (length in pixels, length in millimeters)
    """
    p1, p2 = box[0], box[2]
    px_length = euclidean_distance(p1, p2)
    mm_length = px_length / pixels_per_mm
    return px_length, mm_length

def calculate_knot_to_incision_distance(
    knot_center: Tuple[float, float],
    incision_points: List[Tuple[float, float]],
    pixels_per_mm: float
) -> Tuple[float, float]:
    """Calculate distance from knot center to incision line.
    
    Args:
        knot_center: Center coordinates of the knot
        incision_points: List of incision point coordinates
        pixels_per_mm: Conversion factor from pixels to millimeters
        
    Returns:
        Tuple of (distance in pixels, distance in millimeters)
    """
    if not incision_points:
        return 0.0, 0.0
        
    # Calculate average x-coordinate of incision points
    avg_x = np.mean([p[0] for p in incision_points])
    
    # Calculate horizontal distance
    px_distance = abs(knot_center[0] - avg_x)
    mm_distance = px_distance / pixels_per_mm
    
    return px_distance, mm_distance

def calculate_tail_length(
    box: np.ndarray,
    pixels_per_mm: float
) -> Tuple[float, float]:
    """Calculate tail length in pixels and millimeters.
    
    Args:
        box: Bounding box coordinates
        pixels_per_mm: Conversion factor from pixels to millimeters
        
    Returns:
        Tuple of (length in pixels, length in millimeters)
    """
    return calculate_suture_length(box, pixels_per_mm)

def process_measurements(
    detections: List[Tuple[int, float, np.ndarray]],
    pixels_per_mm: float
) -> Dict[str, List[Dict]]:
    """Process all measurements from detections.
    
    Args:
        detections: List of (class_id, confidence, box) tuples
        pixels_per_mm: Conversion factor from pixels to millimeters
        
    Returns:
        Dictionary containing measurement results for each type
    """
    results = {
        "suture": [],
        "knot": [],
        "tail_top": [],
        "tail_end": []
    }
    
    incision_points = []
    knot_boxes = []
    
    for class_id, confidence, box in detections:
        if class_id in {7, 8, 9, 10}:  # Suture classes
            px_length, mm_length = calculate_suture_length(box, pixels_per_mm)
            results["suture"].append({
                "class_id": class_id,
                "confidence": confidence,
                "length_px": px_length,
                "length_mm": mm_length,
                "box": box
            })
        elif class_id in {3, 4, 5, 6}:  # Knot classes
            knot_boxes.append((class_id, confidence, box))
        elif class_id in {0, 1, 2}:  # Incision classes
            incision_points.extend(box)
        elif class_id == 12:  # Tail top
            px_length, mm_length = calculate_tail_length(box, pixels_per_mm)
            results["tail_top"].append({
                "class_id": class_id,
                "confidence": confidence,
                "length_px": px_length,
                "length_mm": mm_length,
                "box": box
            })
        elif class_id == 11:  # Tail end
            px_length, mm_length = calculate_tail_length(box, pixels_per_mm)
            results["tail_end"].append({
                "class_id": class_id,
                "confidence": confidence,
                "length_px": px_length,
                "length_mm": mm_length,
                "box": box
            })
    
    # Process knot-to-incision distances
    for class_id, confidence, box in knot_boxes:
        center = ((box[0][0] + box[2][0]) / 2, (box[0][1] + box[2][1]) / 2)
        px_distance, mm_distance = calculate_knot_to_incision_distance(
            center, incision_points, pixels_per_mm
        )
        results["knot"].append({
            "class_id": class_id,
            "confidence": confidence,
            "distance_px": px_distance,
            "distance_mm": mm_distance,
            "box": box
        })
    
    return results

def batch_process_measurements(
    detections: Dict[str, List[Tuple[int, float, np.ndarray]]],
    class_names: Dict[int, str],
    pixel_to_mm: float = 0.1,
    num_workers: int = 4
) -> Dict[str, Dict[str, float]]:
    """Process measurements for multiple images.
    
    Args:
        detections: Dictionary mapping image names to detection lists
        class_names: Dictionary mapping class IDs to names
        pixel_to_mm: Conversion factor from pixels to millimeters
        num_workers: Number of worker threads
        
    Returns:
        Dictionary mapping image names to measurement dictionaries
    """
    measurements = {}
    
    def process_image(img_name: str) -> Optional[Dict[str, float]]:
        img_detections = detections.get(img_name, [])
        if not img_detections:
            return None
            
        return process_measurements(img_detections, pixel_to_mm)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            img_name: executor.submit(process_image, img_name)
            for img_name in detections.keys()
        }
        
        for img_name, future in tqdm(futures.items(), total=len(futures), desc="Processing measurements"):
            try:
                result = future.result()
                if result:
                    measurements[img_name] = result
            except Exception as e:
                print(f"Error processing measurements for image {img_name}: {e}")
    
    return measurements

def save_batch_measurements(
    measurements: Dict[str, Dict[str, float]],
    output_path: str
) -> None:
    """Save batch measurement results to a JSON file.
    
    Args:
        measurements: Dictionary mapping image names to measurement dictionaries
        output_path: Path to save the JSON file
    """
    data = {
        'timestamp': datetime.now().isoformat(),
        'measurements': measurements
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_batch_measurements(
    input_path: str
) -> Dict[str, Dict[str, float]]:
    """Load batch measurement results from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        Dictionary mapping image names to measurement dictionaries
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
        return data.get('measurements', {})

def calculate_batch_statistics(
    measurements: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Calculate statistics for batch measurements.
    
    Args:
        measurements: Dictionary mapping image names to measurement dictionaries
        
    Returns:
        Dictionary mapping measurement types to statistics
    """
    stats = {}
    
    # Collect all values for each measurement type
    measurement_values = {}
    for img_measurements in measurements.values():
        for key, value in img_measurements.items():
            if key not in measurement_values:
                measurement_values[key] = []
            measurement_values[key].append(value)
    
    # Calculate statistics for each measurement type
    for key, values in measurement_values.items():
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    return stats 