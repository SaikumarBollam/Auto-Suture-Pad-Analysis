import cv2
import numpy as np
from typing import Tuple, List, Optional

def process_image_variants(image: np.ndarray, draw_contours: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process image into grayscale, edges, and contours variants.
    
    Args:
        image: Input BGR image
        draw_contours: Whether to draw contours on the output image
        
    Returns:
        Tuple of (grayscale, edges, contours) images
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contour_img = image.copy()
    if draw_contours:
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
    
    return gray, edges, contour_img

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better detection.
    
    Args:
        image: Input BGR image
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced

def get_pixels_per_mm(ref_coords: List[Tuple[float, float]], image_size: Tuple[int, int]) -> float:
    """Calculate pixels per mm using reference coordinates.
    
    Args:
        ref_coords: List of reference coordinate pairs [(x1,y1), (x2,y2)]
        image_size: Image dimensions (width, height)
        
    Returns:
        Pixels per mm conversion factor
    """
    # Convert normalized coordinates to pixel coordinates
    p1 = [x * image_size[0] for x in ref_coords[0]]
    p2 = [x * image_size[0] for x in ref_coords[1]]
    
    # Calculate pixel distance
    px_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # Assuming reference distance is 10mm
    return px_dist / 10

def detect_scale_markers(image: np.ndarray) -> Optional[Tuple[float, List[Tuple[float, float]]]]:
    """Detect scale markers (ruler/onecm) in the image for automated calibration.
    
    Args:
        image: Input BGR image
        
    Returns:
        Tuple of (pixels_per_mm, reference_points) if scale detected, None otherwise
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for ruler markings (equally spaced vertical lines)
    ruler_lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > w * 3:  # Vertical lines have height > 3x width
            ruler_lines.append((x + w/2, y, y + h))
    
    if len(ruler_lines) >= 2:
        # Sort by x-coordinate
        ruler_lines.sort(key=lambda x: x[0])
        
        # Find most common spacing
        spacings = []
        for i in range(len(ruler_lines)-1):
            spacing = ruler_lines[i+1][0] - ruler_lines[i][0]
            spacings.append(spacing)
        
        if spacings:
            median_spacing = np.median(spacings)
            # Assuming 1mm spacing between ruler lines
            pixels_per_mm = median_spacing
            
            # Get reference points from first two consistent lines
            ref_points = []
            for i in range(len(ruler_lines)-1):
                if abs(ruler_lines[i+1][0] - ruler_lines[i][0] - median_spacing) < 2:
                    p1 = (ruler_lines[i][0], ruler_lines[i][1])
                    p2 = (ruler_lines[i+1][0], ruler_lines[i+1][1])
                    ref_points = [p1, p2]
                    break
            
            if ref_points:
                return pixels_per_mm, ref_points
    
    return None