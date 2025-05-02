"""Measurement utilities for suture analysis."""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional

def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], 
                   image_size: Optional[Tuple[int, int]] = None) -> float:
    """Calculate angle between a line segment and vertical.
    
    Args:
        p1: First point (x1, y1)
        p2: Second point (x2, y2)
        image_size: Optional image dimensions for normalized coordinates
        
    Returns:
        Angle in degrees from vertical (90 degrees)
    """
    if image_size:
        # Convert normalized coordinates to pixel coordinates
        x1, y1 = p1[0] * image_size[0], p1[1] * image_size[1]
        x2, y2 = p2[0] * image_size[0], p2[1] * image_size[1]
    else:
        x1, y1 = p1
        x2, y2 = p2
        
    dx = x2 - x1
    dy = y2 - y1
    angle_radians = math.atan2(dy, dx)
    angle_degrees = abs(math.degrees(angle_radians) - 90)
    return angle_degrees

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float], 
                      pixels_per_mm: Optional[float] = None) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x1, y1)
        p2: Second point (x2, y2)
        pixels_per_mm: Optional conversion factor for mm
        
    Returns:
        Distance in pixels or mm if conversion factor provided
    """
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    if pixels_per_mm:
        return dist / pixels_per_mm
    return dist

def calculate_perpendicular_distance(point: Tuple[float, float], 
                                  line_p1: Tuple[float, float], 
                                  line_p2: Tuple[float, float],
                                  pixels_per_mm: Optional[float] = None) -> float:
    """Calculate perpendicular distance from point to line.
    
    Args:
        point: Point to measure from (x, y)
        line_p1: Line start point (x1, y1)
        line_p2: Line end point (x2, y2)
        pixels_per_mm: Optional conversion factor for mm
        
    Returns:
        Perpendicular distance in pixels or mm
    """
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    
    dist = numerator/denominator if denominator != 0 else 0
    
    if pixels_per_mm:
        return dist / pixels_per_mm
    return dist

def analyze_stitch_pattern(points: List[Tuple[float, float]], 
                         incision_line: Tuple[Tuple[float, float], Tuple[float, float]],
                         pixels_per_mm: float) -> Dict[str, float]:
    """Analyze stitch pattern metrics.
    
    Args:
        points: List of stitch points [(x,y),...]
        incision_line: Tuple of line endpoints ((x1,y1), (x2,y2))
        pixels_per_mm: Conversion factor for mm
        
    Returns:
        Dict with metrics (average spacing, std deviation, etc)
    """
    distances = []
    angles = []
    
    for i in range(len(points)-1):
        # Calculate distance between consecutive stitches
        dist = euclidean_distance(points[i], points[i+1], pixels_per_mm)
        distances.append(dist)
        
        # Calculate angle relative to incision
        angle = calculate_angle(points[i], points[i+1])
        angles.append(angle)
        
    return {
        'avg_spacing': np.mean(distances),
        'spacing_std': np.std(distances),
        'avg_angle': np.mean(angles),
        'angle_std': np.std(angles),
        'min_spacing': min(distances),
        'max_spacing': max(distances)
    }