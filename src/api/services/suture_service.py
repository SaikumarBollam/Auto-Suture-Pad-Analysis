from typing import Dict, Any, List, Tuple
from PIL import Image
import io
import cv2
import numpy as np
import base64
from fastapi import HTTPException

from ml.core.inference import InferencePipeline
from ..config import settings

class SutureService:
    """Service for handling suture analysis operations."""
    
    def __init__(self, pipeline: InferencePipeline):
        """Initialize the service with an ML pipeline.
        
        Args:
            pipeline: Initialized InferencePipeline instance
        """
        self.pipeline = pipeline
    
    async def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze a suture image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dict containing analysis results
            
        Raises:
            HTTPException: If analysis fails
        """
        try:
            results = self.pipeline.analyze_suture(image)
            return self._format_analysis_results(results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def detect_scale(self, image: Image.Image) -> float:
        """Detect scale in an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Float representing pixels to mm ratio
            
        Raises:
            HTTPException: If scale detection fails
        """
        try:
            return self.pipeline.detect_scale(image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scale detection failed: {str(e)}")
    
    async def visualize_scale(self, image: Image.Image) -> str:
        """Generate scale visualization.
        
        Args:
            image: PIL Image to visualize
            
        Returns:
            Base64 encoded visualization image
            
        Raises:
            HTTPException: If visualization fails
        """
        try:
            visualization = self.pipeline.visualize_scale_detection(image)
            return self._encode_image(visualization)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")
    
    async def predict_quality(self, image: Image.Image) -> Tuple[str, float]:
        """Predict suture quality (good, tight, loose).
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Tuple of (prediction, confidence)
            
        Raises:
            HTTPException: If prediction fails
        """
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(image)
            
            # Get detections
            detections = self.pipeline._get_detections(preprocessed)
            
            # Analyze quality based on measurements
            measurements = self.pipeline.analyze_suture(preprocessed)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(measurements)
            
            # Determine quality category
            if quality_score > 0.8:
                return "good", quality_score
            elif quality_score > 0.5:
                return "tight", quality_score
            else:
                return "loose", quality_score
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quality prediction failed: {str(e)}")
    
    async def measure_stitch_lengths(self, image: Image.Image, scale_px_to_mm: float) -> List[float]:
        """Measure stitch lengths.
        
        Args:
            image: PIL Image to analyze
            scale_px_to_mm: Scale factor for pixel to mm conversion
            
        Returns:
            List of stitch lengths in mm
            
        Raises:
            HTTPException: If measurement fails
        """
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(image)
            
            # Get detections
            detections = self.pipeline._get_detections(preprocessed)
            
            # Get stitches
            stitches = [d for d in detections if d['class'] == 'stitch']
            
            # Measure lengths
            lengths = []
            for stitch in stitches:
                length = self.pipeline._calculate_suture_distance(
                    stitch['bbox'][0], stitch['bbox'][1]
                ) * scale_px_to_mm
                lengths.append(length)
                
            return lengths
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stitch length measurement failed: {str(e)}")
    
    async def analyze_stitch_angles(self, image: Image.Image) -> List[float]:
        """Analyze angles of stitches.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            List of stitch angles in degrees
            
        Raises:
            HTTPException: If analysis fails
        """
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(image)
            
            # Get detections
            detections = self.pipeline._get_detections(preprocessed)
            
            # Get stitches and incision line
            stitches = [d for d in detections if d['class'] == 'stitch']
            incision = [d for d in detections if d['class'] == 'incision'][0]
            incision_line = self.pipeline._get_incision_line(incision)
            
            # Calculate angles
            angles = []
            for stitch in stitches:
                angle = self.pipeline._calculate_stitch_angle(stitch, incision_line)
                angles.append(angle)
                
            return angles
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stitch angle analysis failed: {str(e)}")
    
    async def measure_tail_lengths(self, image: Image.Image) -> Dict[str, Any]:
        """Measure tail lengths.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dict containing tail lengths and missing count
            
        Raises:
            HTTPException: If measurement fails
        """
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(image)
            
            # Get detections
            detections = self.pipeline._get_detections(preprocessed)
            
            # Get tails
            tails = [d for d in detections if d['class'] == 'tail']
            
            # Measure lengths
            lengths = []
            for tail in tails:
                length = self.pipeline._calculate_tail_length(tail)
                lengths.append(length)
                
            return {
                "lengths": lengths,
                "missing_count": max(0, 2 - len(tails))  # Expect 2 tails
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Tail length measurement failed: {str(e)}")
    
    async def measure_suture_distances(self, image: Image.Image, scale_px_to_mm: float) -> List[float]:
        """Calculate distances between sutures.
        
        Args:
            image: PIL Image to analyze
            scale_px_to_mm: Scale factor for pixel to mm conversion
            
        Returns:
            List of distances between sutures in mm
            
        Raises:
            HTTPException: If measurement fails
        """
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(image)
            
            # Get detections
            detections = self.pipeline._get_detections(preprocessed)
            
            # Get stitches
            stitches = [d for d in detections if d['class'] == 'stitch']
            
            # Calculate distances
            distances = []
            for i in range(len(stitches) - 1):
                distance = self.pipeline._measure_stitch_distance(
                    stitches[i], stitches[i + 1], 'left'
                ) * scale_px_to_mm
                distances.append(distance)
                
            return distances
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Suture distance measurement failed: {str(e)}")
    
    async def measure_knot_incision_distances(self, image: Image.Image) -> List[float]:
        """Measure distances from knots to incision.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            List of distances from knots to incision in mm
            
        Raises:
            HTTPException: If measurement fails
        """
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(image)
            
            # Get detections
            detections = self.pipeline._get_detections(preprocessed)
            
            # Get knots and incision line
            knots = [d for d in detections if d['class'] == 'knot']
            incision = [d for d in detections if d['class'] == 'incision'][0]
            incision_line = self.pipeline._get_incision_line(incision)
            
            # Calculate distances
            distances = []
            for knot in knots:
                distance = self.pipeline._calculate_knot_incision_distance(knot, incision_line)
                distances.append(distance)
                
            return distances
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Knot incision distance measurement failed: {str(e)}")
    
    def _format_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format analysis results for API response.
        
        Args:
            results: Raw analysis results from pipeline
            
        Returns:
            Formatted results dictionary
        """
        return {
            "measurements": [
                {
                    "type": m.type.value,
                    "value": round(m.value, 2),
                    "unit": m.unit,
                    "is_within_standard": m.is_within_standard,
                    "standard": {
                        "mean": m.standard_mean,
                        "std": m.standard_deviation
                    }
                }
                for m in results.get("measurements", [])
            ],
            "statistics": results.get("statistics", {}),
            "scale_px_to_mm": results.get("scale_px_to_mm")
        }
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode an image as base64 string.
        
        Args:
            image: OpenCV/Numpy image array
            
        Returns:
            Base64 encoded image string
        """
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode()
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image according to settings.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to numpy array
        img = np.array(image)
        
        # Apply preprocessing options
        if settings.PREPROCESSING_OPTIONS["grayscale"]:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
        if settings.PREPROCESSING_OPTIONS["gaussian_blur"]:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
        if settings.PREPROCESSING_OPTIONS["contour_extraction"]:
            edges = cv2.Canny(img, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            
        if settings.PREPROCESSING_OPTIONS["edge_detection"]:
            img = cv2.Canny(img, 100, 200)
            
        # Resize to target size
        img = cv2.resize(img, settings.IMAGE_SIZE)
        
        return Image.fromarray(img)
    
    def _calculate_quality_score(self, measurements: Dict[str, Any]) -> float:
        """Calculate overall quality score from measurements.
        
        Args:
            measurements: Dictionary of measurements
            
        Returns:
            Quality score between 0 and 1
        """
        # Initialize score components
        length_score = 0
        angle_score = 0
        distance_score = 0
        
        # Calculate length score
        lengths = [m.value for m in measurements["measurements"] 
                  if m.type.value.startswith("incision_line_to")]
        if lengths:
            length_score = sum(1 for l in lengths if 3 <= l <= 5) / len(lengths)
            
        # Calculate angle score
        angles = [m.value for m in measurements["measurements"] 
                 if m.type.value == "stitch_incision_line_angle"]
        if angles:
            angle_score = sum(1 for a in angles if 80 <= a <= 100) / len(angles)
            
        # Calculate distance score
        distances = [m.value for m in measurements["measurements"] 
                    if m.type.value.startswith("distance_between_stitches")]
        if distances:
            distance_score = sum(1 for d in distances if 3 <= d <= 5) / len(distances)
            
        # Calculate overall score
        if not any([length_score, angle_score, distance_score]):
            return 0
            
        return (length_score + angle_score + distance_score) / 3 