import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
import cv2
from scipy.spatial import distance
from scipy import stats
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
import mlflow
from mlflow.tracking import MlflowClient

from ..utils.processing import preprocess_image
from ..utils.validation import validate_image
from ..utils.visualization import draw_annotations
from ..utils.scale_detection import ScaleDetector

class MeasurementType(Enum):
    L1 = "incision_line_to_end_of_stitch"
    R1 = "incision_line_to_beginning_of_knot"
    T1A = "tail_1"
    T1B = "tail_2"
    K1 = "incision_line_to_middle_of_knot"
    ALPHA = "stitch_incision_line_angle"
    DL1_2 = "distance_between_stitches_left"
    DR1_2 = "distance_between_stitches_right"

@dataclass
class Measurement:
    type: MeasurementType
    value: float
    unit: str
    standard_mean: float
    standard_deviation: float
    is_within_standard: bool

class InferencePipeline:
    """Pipeline for performing suture analysis inference with caching and batch processing."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """Initialize the inference pipeline.
        
        Args:
            model: Trained model
            config: Model configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize scale detector
        self.scale_detector = ScaleDetector(config.get('scale_detection', None))
        
        # Initialize MLflow
        self.mlflow_client = MlflowClient()
        
        # Initialize thread pool for batch processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 4)
        )
        
        # Performance metrics
        self.inference_times = []
        self.batch_sizes = []
        
        # Standard measurements from medical guidelines
        self.standards = {
            MeasurementType.L1: (4.0, 1.0),  # mean, std
            MeasurementType.R1: (4.0, 1.0),
            MeasurementType.T1A: (6.0, 3.0),
            MeasurementType.T1B: (6.0, 3.0),
            MeasurementType.K1: (4.0, 1.0),
            MeasurementType.ALPHA: (90.0, 10.0),
            MeasurementType.DL1_2: (4.0, 1.0),
            MeasurementType.DR1_2: (4.0, 1.0)
        }
        
    @lru_cache(maxsize=100)
    def detect_scale(self, image: Image.Image) -> float:
        """Detect scale in the image and calculate pixel to mm ratio.
        
        Args:
            image: Input image
            
        Returns:
            float: Pixel to mm conversion ratio
        """
        return self.scale_detector.detect_scale(image)
        
    def batch_process(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process a batch of images in parallel.
        
        Args:
            images: List of input images
            
        Returns:
            List[Dict[str, Any]]: List of analysis results
        """
        start_time = time.time()
        
        # Process images in parallel
        futures = [
            self.thread_pool.submit(self.analyze_suture, img)
            for img in images
        ]
        
        # Collect results
        results = [f.result() for f in futures]
        
        # Record performance metrics
        duration = time.time() - start_time
        self.inference_times.append(duration)
        self.batch_sizes.append(len(images))
        
        # Log metrics to MLflow
        self._log_performance_metrics()
        
        return results
        
    def _log_performance_metrics(self):
        """Log performance metrics to MLflow."""
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            avg_batch_size = np.mean(self.batch_sizes)
            
            try:
                mlflow.log_metrics({
                    'avg_inference_time': avg_time,
                    'avg_batch_size': avg_batch_size,
                    'images_per_second': avg_batch_size / avg_time
                })
            except Exception as e:
                print(f"Failed to log metrics: {e}")
                
    def analyze_suture(self, image: Image.Image) -> Dict[str, Any]:
        """Perform comprehensive suture analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        start_time = time.time()
        
        # Detect scale
        scale_px_to_mm = self.detect_scale(image)
        
        # Get all measurements
        measurements = self._get_all_measurements(image, scale_px_to_mm)
        
        # Validate against standards
        validated_measurements = self._validate_measurements(measurements)
        
        # Calculate statistics
        stats = self._calculate_statistics(validated_measurements)
        
        # Record inference time
        duration = time.time() - start_time
        self.inference_times.append(duration)
        self.batch_sizes.append(1)
        
        return {
            'measurements': validated_measurements,
            'statistics': stats,
            'scale_px_to_mm': scale_px_to_mm,
            'inference_time': duration
        }
        
    def get_model_version(self) -> str:
        """Get the current model version from MLflow.
        
        Returns:
            str: Model version
        """
        try:
            model_name = "suture_detection"
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            if versions:
                return versions[0].version
            return "1.0"
        except Exception as e:
            print(f"Failed to get model version: {e}")
            return "1.0"
        
    def visualize_scale_detection(self, 
                                image: Image.Image,
                                output_path: Optional[str] = None) -> np.ndarray:
        """Visualize scale detection on the image.
        
        Args:
            image: Input image
            output_path: Optional path to save visualization
            
        Returns:
            np.ndarray: Image with scale visualization
        """
        # Convert PIL image to OpenCV format
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        # Detect scale
        scale_length = self.scale_detector._detect_scale_hough(img)
        if scale_length is None:
            scale_length = self.scale_detector._detect_scale_template(img)
        if scale_length is None:
            scale_length = self.scale_detector._detect_scale_edges(img)
            
        if scale_length is None:
            raise ValueError("Could not detect scale in image")
            
        # Draw scale detection
        return self.scale_detector._draw_scale_detection(img, scale_length, output_path)
        
    def _get_all_measurements(self, image: Image.Image, scale_px_to_mm: float) -> List[Measurement]:
        """Get all required measurements.
        
        Args:
            image: Input image
            scale_px_to_mm: Scale factor for pixel to mm conversion
            
        Returns:
            List[Measurement]: List of all measurements
        """
        measurements = []
        
        # Get detections
        detections = self._get_detections(image)
        
        # Get incision line
        incision_line = self._get_incision_line([d for d in detections if d['class'] == 'incision'][0])
        
        # Get stitches
        stitches = [d for d in detections if d['class'] == 'stitch']
        
        # Measure each stitch
        for i, stitch in enumerate(stitches):
            # L1 and R1 measurements
            l1 = self._measure_stitch_to_incision(stitch, incision_line, 'left') * scale_px_to_mm
            r1 = self._measure_stitch_to_incision(stitch, incision_line, 'right') * scale_px_to_mm
            
            # Angle measurement
            angle = self._calculate_stitch_angle(stitch, incision_line)
            
            # Add measurements
            measurements.extend([
                Measurement(MeasurementType.L1, l1, 'mm', *self.standards[MeasurementType.L1], True),
                Measurement(MeasurementType.R1, r1, 'mm', *self.standards[MeasurementType.R1], True),
                Measurement(MeasurementType.ALPHA, angle, 'degrees', *self.standards[MeasurementType.ALPHA], True)
            ])
            
            # Measure distances between stitches if not the last one
            if i < len(stitches) - 1:
                dl = self._measure_stitch_distance(stitch, stitches[i + 1], 'left') * scale_px_to_mm
                dr = self._measure_stitch_distance(stitch, stitches[i + 1], 'right') * scale_px_to_mm
                
                measurements.extend([
                    Measurement(MeasurementType.DL1_2, dl, 'mm', *self.standards[MeasurementType.DL1_2], True),
                    Measurement(MeasurementType.DR1_2, dr, 'mm', *self.standards[MeasurementType.DR1_2], True)
                ])
                
        # Measure tails and knots
        tails = [d for d in detections if d['class'] == 'tail']
        knots = [d for d in detections if d['class'] == 'knot']
        
        for tail in tails[:2]:  # First two tails
            length = self._calculate_tail_length(tail) * scale_px_to_mm
            measurements.append(
                Measurement(MeasurementType.T1A if len(measurements) % 2 == 0 else MeasurementType.T1B,
                          length, 'mm', *self.standards[MeasurementType.T1A], True)
            )
            
        for knot in knots:
            k1 = self._calculate_knot_incision_distance(knot, incision_line) * scale_px_to_mm
            measurements.append(
                Measurement(MeasurementType.K1, k1, 'mm', *self.standards[MeasurementType.K1], True)
            )
            
        return measurements
        
    def _validate_measurements(self, measurements: List[Measurement]) -> List[Measurement]:
        """Validate measurements against standards.
        
        Args:
            measurements: List of measurements
            
        Returns:
            List[Measurement]: Validated measurements
        """
        validated = []
        for m in measurements:
            mean, std = self.standards[m.type]
            is_within = abs(m.value - mean) <= 2 * std  # Within 2 standard deviations
            validated.append(Measurement(m.type, m.value, m.unit, mean, std, is_within))
        return validated
        
    def _calculate_statistics(self, measurements: List[Measurement]) -> Dict[str, Any]:
        """Calculate statistics for measurements.
        
        Args:
            measurements: List of measurements
            
        Returns:
            Dict[str, Any]: Statistics
        """
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
                    'count': len(values)
                }
        return stats
        
    def _measure_stitch_to_incision(self, stitch: Dict[str, Any], 
                                  incision_line: Tuple[np.ndarray, np.ndarray],
                                  side: str) -> float:
        """Measure distance from stitch to incision line.
        
        Args:
            stitch: Stitch detection
            incision_line: Incision line points
            side: 'left' or 'right'
            
        Returns:
            float: Distance in pixels
        """
        # Get stitch endpoint
        bbox = stitch['bbox']
        endpoint = np.array([bbox[0], (bbox[1] + bbox[3]) / 2]) if side == 'left' else \
                  np.array([bbox[2], (bbox[1] + bbox[3]) / 2])
                  
        # Calculate distance to line
        return distance.point_to_line_distance(endpoint, incision_line[0], incision_line[1])
        
    def _calculate_stitch_angle(self, stitch: Dict[str, Any], 
                              incision_line: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate angle between stitch and incision line.
        
        Args:
            stitch: Stitch detection
            incision_line: Incision line points
            
        Returns:
            float: Angle in degrees
        """
        # Get stitch orientation
        obb = stitch['obb']
        stitch_vector = obb[1] - obb[0]
        
        # Get incision line vector
        incision_vector = incision_line[1] - incision_line[0]
        
        # Calculate angle between vectors
        angle = np.arccos(np.dot(stitch_vector, incision_vector) / 
                         (np.linalg.norm(stitch_vector) * np.linalg.norm(incision_vector)))
        
        return np.degrees(angle)
        
    def _measure_stitch_distance(self, stitch1: Dict[str, Any], 
                               stitch2: Dict[str, Any],
                               side: str) -> float:
        """Measure distance between two stitches.
        
        Args:
            stitch1: First stitch detection
            stitch2: Second stitch detection
            side: 'left' or 'right'
            
        Returns:
            float: Distance in pixels
        """
        # Get endpoints
        bbox1 = stitch1['bbox']
        bbox2 = stitch2['bbox']
        
        point1 = np.array([bbox1[0], (bbox1[1] + bbox1[3]) / 2]) if side == 'left' else \
                 np.array([bbox1[2], (bbox1[1] + bbox1[3]) / 2])
                 
        point2 = np.array([bbox2[0], (bbox2[1] + bbox2[3]) / 2]) if side == 'left' else \
                 np.array([bbox2[2], (bbox2[1] + bbox2[3]) / 2])
                 
        return distance.euclidean(point1, point2)
        
    def predict_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Predict suture quality (good, tight, loose).
        
        Args:
            image: Input image
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Preprocess image
        img_tensor = preprocess_image(image, self.config)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Process outputs
        quality_scores = torch.softmax(outputs['quality'], dim=1)
        quality_class = torch.argmax(quality_scores, dim=1).item()
        confidence = quality_scores[0, quality_class].item()
        
        quality_map = {0: 'good', 1: 'tight', 2: 'loose'}
        return {
            'class': quality_map[quality_class],
            'confidence': confidence
        }
        
    def measure_stitch_lengths(self, image: Image.Image, scale_px_to_mm: float) -> List[float]:
        """Measure stitch lengths.
        
        Args:
            image: Input image
            scale_px_to_mm: Scale factor for pixel to mm conversion
            
        Returns:
            List[float]: Stitch lengths in mm
        """
        # Get stitch detections
        detections = self._get_detections(image)
        
        # Calculate lengths
        lengths = []
        for det in detections:
            if det['class'] == 'stitch':
                # Get bounding box dimensions
                x1, y1, x2, y2 = det['bbox']
                width = x2 - x1
                height = y2 - y1
                
                # Use the longer dimension as length
                length_px = max(width, height)
                length_mm = length_px * scale_px_to_mm
                lengths.append(length_mm)
                
        return lengths
        
    def analyze_stitch_angles(self, image: Image.Image) -> List[float]:
        """Analyze angles of stitches.
        
        Args:
            image: Input image
            
        Returns:
            List[float]: Stitch angles in degrees
        """
        # Get stitch detections
        detections = self._get_detections(image)
        
        # Calculate angles
        angles = []
        for det in detections:
            if det['class'] == 'stitch':
                # Get oriented bounding box
                obb = det['obb']
                
                # Calculate angle from oriented bounding box
                angle = self._calculate_obb_angle(obb)
                angles.append(angle)
                
        return angles
        
    def measure_tail_lengths(self, image: Image.Image) -> Dict[str, Any]:
        """Measure tail lengths.
        
        Args:
            image: Input image
            
        Returns:
            Dict[str, Any]: Tail length results
        """
        # Get tail detections
        detections = self._get_detections(image)
        
        # Calculate lengths
        lengths = []
        missing_count = 0
        for det in detections:
            if det['class'] == 'tail':
                # Get tail length
                length = self._calculate_tail_length(det)
                if length > 0:
                    lengths.append(length)
                else:
                    missing_count += 1
                    
        return {
            'lengths': lengths,
            'missing_count': missing_count
        }
        
    def measure_suture_distances(self, image: Image.Image, scale_px_to_mm: float) -> List[float]:
        """Calculate distances between sutures.
        
        Args:
            image: Input image
            scale_px_to_mm: Scale factor for pixel to mm conversion
            
        Returns:
            List[float]: Suture distances in mm
        """
        # Get suture detections
        detections = self._get_detections(image)
        suture_dets = [d for d in detections if d['class'] == 'suture']
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(suture_dets) - 1):
            for j in range(i + 1, len(suture_dets)):
                dist = self._calculate_suture_distance(suture_dets[i], suture_dets[j])
                dist_mm = dist * scale_px_to_mm
                distances.append(dist_mm)
                
        return distances
        
    def measure_knot_incision_distances(self, image: Image.Image) -> List[float]:
        """Measure distances from knots to incision.
        
        Args:
            image: Input image
            
        Returns:
            List[float]: Knot-to-incision distances
        """
        # Get detections
        detections = self._get_detections(image)
        knot_dets = [d for d in detections if d['class'] == 'knot']
        incision_dets = [d for d in detections if d['class'] == 'incision']
        
        if not incision_dets:
            raise ValueError("No incision detected")
            
        # Get incision line
        incision_line = self._get_incision_line(incision_dets[0])
        
        # Calculate distances
        distances = []
        for knot in knot_dets:
            dist = self._calculate_knot_incision_distance(knot, incision_line)
            distances.append(dist)
            
        return distances
        
    def _get_detections(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Get object detections from image.
        
        Args:
            image: Input image
            
        Returns:
            List[Dict[str, Any]]: List of detections
        """
        # Preprocess image
        img_tensor = preprocess_image(image, self.config)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Process detections
        detections = []
        for output in outputs['detections']:
            det = {
                'class': output['class'],
                'confidence': output['confidence'],
                'bbox': output['bbox'],
                'obb': output.get('obb', None)
            }
            detections.append(det)
            
        return detections
        
    def _calculate_obb_angle(self, obb: np.ndarray) -> float:
        """Calculate angle from oriented bounding box.
        
        Args:
            obb: Oriented bounding box coordinates
            
        Returns:
            float: Angle in degrees
        """
        # Get the longer side of the OBB
        side1 = obb[1] - obb[0]
        side2 = obb[2] - obb[1]
        
        # Use the longer side to calculate angle
        if np.linalg.norm(side1) > np.linalg.norm(side2):
            angle = np.arctan2(side1[1], side1[0])
        else:
            angle = np.arctan2(side2[1], side2[0])
            
        return np.degrees(angle)
        
    def _calculate_tail_length(self, detection: Dict[str, Any]) -> float:
        """Calculate tail length from detection.
        
        Args:
            detection: Tail detection
            
        Returns:
            float: Tail length
        """
        # Get tail points
        points = detection.get('points', None)
        if points is None or len(points) < 2:
            return 0.0
            
        # Calculate length
        length = 0.0
        for i in range(len(points) - 1):
            length += distance.euclidean(points[i], points[i + 1])
            
        return length
        
    def _calculate_suture_distance(self, det1: Dict[str, Any], det2: Dict[str, Any]) -> float:
        """Calculate distance between two sutures.
        
        Args:
            det1: First suture detection
            det2: Second suture detection
            
        Returns:
            float: Distance in pixels
        """
        # Get center points
        bbox1 = det1['bbox']
        bbox2 = det2['bbox']
        
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return distance.euclidean(center1, center2)
        
    def _get_incision_line(self, detection: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Get incision line from detection.
        
        Args:
            detection: Incision detection
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Line start and end points
        """
        # Get oriented bounding box
        obb = detection['obb']
        
        # Use the longer side as the incision line
        side1 = obb[1] - obb[0]
        side2 = obb[2] - obb[1]
        
        if np.linalg.norm(side1) > np.linalg.norm(side2):
            return obb[0], obb[1]
        else:
            return obb[1], obb[2]
            
    def _calculate_knot_incision_distance(self, knot: Dict[str, Any], 
                                        incision_line: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate distance from knot to incision line.
        
        Args:
            knot: Knot detection
            incision_line: Incision line start and end points
            
        Returns:
            float: Distance in pixels
        """
        # Get knot center
        bbox = knot['bbox']
        knot_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        # Calculate distance to line
        line_start, line_end = incision_line
        return distance.point_to_line_distance(knot_center, line_start, line_end) 