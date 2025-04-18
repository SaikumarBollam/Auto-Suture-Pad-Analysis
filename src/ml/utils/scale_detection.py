import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import math

class ScaleDetector:
    """Detects scale in medical images and calculates pixel to mm ratio."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the scale detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {
            'min_scale_length': 10,  # Minimum scale length in pixels
            'max_scale_length': 500,  # Maximum scale length in pixels
            'scale_thickness': 2,  # Expected scale line thickness
            'scale_color': (0, 0, 0),  # Expected scale color (black)
            'known_length_mm': 10,  # Known length of scale in mm
            'hough_threshold': 100,  # Hough transform threshold
            'min_line_length': 50,  # Minimum line length for detection
            'max_line_gap': 20,  # Maximum gap between line segments
            'template_matching_threshold': 0.8,  # Template matching threshold
        }
        
    def detect_scale(self, image: Image.Image) -> float:
        """Detect scale in the image and calculate pixel to mm ratio.
        
        Args:
            image: Input image
            
        Returns:
            float: Pixel to mm conversion ratio
        """
        # Convert PIL image to OpenCV format
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
        # Try different detection methods
        scale_length = None
        
        # Method 1: Hough Line Transform
        scale_length = self._detect_scale_hough(img)
        
        # Method 2: Template Matching (if Hough fails)
        if scale_length is None:
            scale_length = self._detect_scale_template(img)
            
        # Method 3: Edge Detection (if both above fail)
        if scale_length is None:
            scale_length = self._detect_scale_edges(img)
            
        if scale_length is None:
            raise ValueError("Could not detect scale in image")
            
        # Calculate pixel to mm ratio
        return self.config['known_length_mm'] / scale_length
        
    def _detect_scale_hough(self, img: np.ndarray) -> Optional[float]:
        """Detect scale using Hough Line Transform.
        
        Args:
            img: Grayscale image
            
        Returns:
            Optional[float]: Detected scale length in pixels
        """
        # Apply edge detection
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.config['hough_threshold'],
            minLineLength=self.config['min_line_length'],
            maxLineGap=self.config['max_line_gap']
        )
        
        if lines is None:
            return None
            
        # Find the longest horizontal line
        max_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Check if line is approximately horizontal
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            if abs(angle) < 10 and length > max_length:
                max_length = length
                
        return max_length if max_length > 0 else None
        
    def _detect_scale_template(self, img: np.ndarray) -> Optional[float]:
        """Detect scale using template matching.
        
        Args:
            img: Grayscale image
            
        Returns:
            Optional[float]: Detected scale length in pixels
        """
        # Create a template for scale line
        template = np.zeros((self.config['scale_thickness'], 100), dtype=np.uint8)
        template.fill(255)
        
        # Apply template matching
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < self.config['template_matching_threshold']:
            return None
            
        # Return template width as scale length
        return template.shape[1]
        
    def _detect_scale_edges(self, img: np.ndarray) -> Optional[float]:
        """Detect scale using edge detection and contour analysis.
        
        Args:
            img: Grayscale image
            
        Returns:
            Optional[float]: Detected scale length in pixels
        """
        # Apply edge detection
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the longest horizontal contour
        max_length = 0
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour is approximately horizontal and within size limits
            if (self.config['min_scale_length'] <= w <= self.config['max_scale_length'] and
                h <= self.config['scale_thickness'] * 2):
                if w > max_length:
                    max_length = w
                    
        return max_length if max_length > 0 else None
        
    def _validate_scale_length(self, length: float) -> bool:
        """Validate detected scale length.
        
        Args:
            length: Detected scale length in pixels
            
        Returns:
            bool: True if length is valid
        """
        return (self.config['min_scale_length'] <= length <= self.config['max_scale_length'])
        
    def _draw_scale_detection(self, 
                            img: np.ndarray, 
                            scale_length: float,
                            output_path: Optional[str] = None) -> np.ndarray:
        """Draw detected scale on the image.
        
        Args:
            img: Input image
            scale_length: Detected scale length
            output_path: Optional path to save visualization
            
        Returns:
            np.ndarray: Image with scale visualization
        """
        # Convert to color if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        # Draw scale line
        height, width = img.shape[:2]
        start_point = (width - int(scale_length) - 10, height - 20)
        end_point = (width - 10, height - 20)
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)
        
        # Draw length label
        label = f"{self.config['known_length_mm']}mm"
        cv2.putText(img, label, (start_point[0], start_point[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                   
        if output_path:
            cv2.imwrite(output_path, img)
            
        return img 