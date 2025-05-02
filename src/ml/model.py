"""ML model configuration and training utilities."""

import os
from pathlib import Path
from typing import Tuple, Union, List, Dict, Optional
import math
import yaml
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .utils.measurements import (
    calculate_angle, euclidean_distance, 
    calculate_perpendicular_distance, analyze_stitch_pattern
)
from .utils.image_processing import detect_scale_markers

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class DetectionError(ModelError):
    """Exception raised when detection fails."""
    pass

class MeasurementError(ModelError):
    """Exception raised when measurement fails."""
    pass

class SutureDetector:
    """YOLOv8-based suture detection model with enhanced preprocessing."""
    
    def __init__(self, model_path: Union[str, Path], conf_threshold: float = 0.5):
        """Initialize the detector with model weights and config."""
        try:
            with open('config/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)['ml']
        except Exception as e:
            raise ModelError(f"Failed to load config: {str(e)}")
            
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.device = torch.device(self.config['model']['device'])
            self.model.to(self.device)
        except Exception as e:
            raise ModelError(f"Failed to initialize model: {str(e)}")
        
        self.scaler = StandardScaler()
        self.score_model = None
        self.pixels_per_mm = None
        
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement."""
        if not self.config['preprocessing']['clahe']['enabled']:
            return image
            
        clahe = cv2.createCLAHE(
            clipLimit=self.config['preprocessing']['clahe']['clip_limit'],
            tileGridSize=tuple(self.config['preprocessing']['clahe']['tile_grid_size'])
        )
        
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return clahe.apply(image)
        
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering for noise reduction."""
        if not self.config['preprocessing']['bilateral_filter']['enabled']:
            return image
            
        return cv2.bilateralFilter(
            image,
            d=self.config['preprocessing']['bilateral_filter']['d'],
            sigmaColor=self.config['preprocessing']['bilateral_filter']['sigma_color'],
            sigmaSpace=self.config['preprocessing']['bilateral_filter']['sigma_space']
        )
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing pipeline."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Apply preprocessing steps
        image = self.apply_clahe(image)
        image = self.apply_bilateral_filter(image)
        
        # Normalize if enabled
        if self.config['preprocessing']['normalize']:
            image = image.astype(np.float32) / 255.0
            
        return image
        
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect sutures in an image with test-time augmentation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (boxes, scores, class_ids) as numpy arrays
            
        Raises:
            DetectionError: If detection fails
        """
        try:
            image = self.preprocess_image(image)
            
            # Configure test-time augmentation
            if self.config['model']['test_time_augmentation']:
                results = self.model(image, augment=True, conf=self.conf_threshold)[0]
            else:
                results = self.model(image, conf=self.conf_threshold)[0]
            
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Apply NMS
            keep_indices = self.non_max_suppression(
                boxes, scores, iou_threshold=self.config['model']['nms_iou_threshold']
            )
            
            return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]
            
        except Exception as e:
            raise DetectionError(f"Detection failed: {str(e)}")
        
    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, 
                          iou_threshold: float) -> np.ndarray:
        """Improved Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        keep = []
        order = scores.argsort()[::-1]
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = np.where(ovr <= iou_threshold)[0]
            order = order[ids + 1]
            
        return np.array(keep)
        
    def train(self, data_yaml: Union[str, Path], epochs: int = None):
        """Train with enhanced configuration."""
        # Setup training parameters
        train_config = self.config['training']
        epochs = epochs or train_config['epochs']
        
        # Configure optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate']['initial'],
            weight_decay=train_config['optimizer']['weight_decay']
        )
        
        # Configure learning rate scheduler with warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - train_config['learning_rate']['warmup_epochs']
        )
        
        # Train the model
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.config['preprocessing']['image_size'],
            batch=self.config['preprocessing']['batch_size'],
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            patience=train_config['patience'],
            save=True,
            save_period=10
        )

    def get_calibration(self, image: np.ndarray) -> float:
        """Get pixel to mm calibration factor using automated detection or config."""
        # Try automated detection first
        scale_result = detect_scale_markers(image)
        if scale_result is not None:
            self.pixels_per_mm, _ = scale_result
            return self.pixels_per_mm
            
        # Fall back to configured reference points
        ref_points = self.config.get('measurements', {}).get('pixel_mm_calibration', {}).get('reference_points')
        if ref_points and len(ref_points) == 2:
            h, w = image.shape[:2]
            p1 = [x * w for x in ref_points[0]]
            p2 = [x * w for x in ref_points[1]]
            px_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            self.pixels_per_mm = px_dist / 10  # Assuming 10mm reference distance
            return self.pixels_per_mm
            
        raise ValueError("No calibration available - neither automated detection nor reference points found")

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for quality scoring."""
        boxes, _, _ = self.detect(image)
        if len(boxes) == 0:
            return np.zeros(8)  # Return zero features if no detections
            
        # Get calibration
        try:
            pixels_per_mm = self.get_calibration(image)
        except ValueError:
            pixels_per_mm = 1.0  # Use pixel measurements if calibration fails
            
        # Calculate features
        lengths = self.measure_stitch_lengths(image, pixels_per_mm)
        angles = self.analyze_angles(image)
        pattern = self.analyze_suture_pattern_symmetry(image)
        spacing = self.analyze_suture_spacing_uniformity(image)
        
        features = [
            np.mean(lengths) if len(lengths) > 0 else 0,
            np.std(lengths) if len(lengths) > 0 else 0,
            np.mean(angles) if len(angles) > 0 else 0,
            np.std(angles) if len(angles) > 0 else 0,
            pattern.get('symmetry_score', 0),
            spacing.get('spacing_cv', 0),
            pattern.get('max_deviation', 0),
            spacing.get('mean_spacing', 0)
        ]
        return np.array(features)

    def train_quality_model(self, train_images: List[np.ndarray], train_labels: List[str]):
        """Train XGBoost model for quality prediction."""
        # Extract features from all training images
        features = []
        for img in train_images:
            feat = self.extract_features(img)
            features.append(feat)
        features = np.array(features)
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Convert labels to integers
        label_map = {'good': 0, 'tight': 1, 'loose': 2}
        y = np.array([label_map[label] for label in train_labels])
        
        # Train XGBoost model
        self.score_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            learning_rate=0.1,
            max_depth=4,
            n_estimators=100
        )
        self.score_model.fit(features, y)

    def predict_quality(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Predict suture quality with confidence scores and ablation metrics."""
        features = self.extract_features(image)
        
        if self.score_model is None:
            # Fall back to rule-based if no model trained
            return self._rule_based_quality(image, boxes), {}
            
        # Scale features
        features = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        pred_proba = self.score_model.predict_proba(features)[0]
        pred_class = self.score_model.predict(features)[0]
        
        # Map back to labels
        labels = ['good', 'tight', 'loose']
        prediction = labels[pred_class]
        
        # Calculate feature importance for ablation study
        importance = self.score_model.feature_importances_
        feature_names = [
            'mean_length', 'std_length', 'mean_angle', 'std_angle',
            'symmetry', 'spacing_cv', 'max_deviation', 'mean_spacing'
        ]
        
        ablation_metrics = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        ablation_metrics.update({
            f'{label}_probability': float(prob)
            for label, prob in zip(labels, pred_proba)
        })
        
        return prediction, ablation_metrics
        
    def _rule_based_quality(self, image: np.ndarray, boxes: np.ndarray) -> str:
        """Fallback rule-based quality assessment."""
        if len(boxes) == 0:
            return "loose"  # No sutures detected
            
        # Analyze suture characteristics
        stitch_lengths = self.measure_stitch_lengths(image, 1.0)  # Use pixel measurements
        angles = self.analyze_angles(image)
        
        # Decision logic based on measurements
        avg_length = np.mean(stitch_lengths)
        avg_angle = np.mean(angles) if len(angles) > 0 else 0
        
        if avg_length < 20:  # Too tight
            return "tight"
        elif avg_length > 50 or abs(avg_angle - 90) > 30:  # Too loose or inconsistent
            return "loose"
        else:
            return "good"

    def measure_stitch_lengths(self, image: np.ndarray, scale_px_to_mm: float) -> np.ndarray:
        """
        Measure lengths of detected stitches.
        
        Args:
            image: Input image
            scale_px_to_mm: Scale factor to convert pixels to millimeters
            
        Returns:
            Array of stitch lengths in millimeters
            
        Raises:
            MeasurementError: If measurement fails
        """
        try:
            boxes, _, _ = self.detect(image)
            if len(boxes) == 0:
                return np.array([], dtype=np.float32)
                
            lengths = []
            for box in boxes:
                x1, y1, x2, y2 = box
                length_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                lengths.append(length_px * scale_px_to_mm)
                
            return np.array(lengths, dtype=np.float32)
            
        except DetectionError as e:
            raise MeasurementError(f"Stitch length measurement failed: {str(e)}")
        except Exception as e:
            raise MeasurementError(f"Unexpected error in stitch length measurement: {str(e)}")

    def analyze_angles(self, image: np.ndarray) -> np.ndarray:
        """Calculate angles of stitches relative to horizontal."""
        boxes, _, _ = self.detect(image)
        angles = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(abs(angle))
            
        return np.array(angles)

    def measure_tail_lengths(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Measure tail lengths and count missing tails."""
        # Use edge detection to find tails
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        boxes, _, _ = self.detect(image)
        tail_lengths = []
        missing_count = 0
        
        for box in boxes:
            # Look for line segments extending from suture endpoints
            x1, y1, x2, y2 = map(int, box)
            roi = edges[max(0, y1-20):min(y1+20, edges.shape[0]),
                       max(0, x1-20):min(x1+20, edges.shape[1])]
            
            lines = cv2.HoughLinesP(roi, 1, np.pi/180, 10, 
                                  minLineLength=5, maxLineGap=3)
            
            if lines is not None:
                max_length = max(np.sqrt((line[0][2] - line[0][0])**2 + 
                                       (line[0][3] - line[0][1])**2) 
                               for line in lines)
                tail_lengths.append(max_length)
            else:
                missing_count += 1
        
        return np.array(tail_lengths), missing_count

    def measure_suture_distances(self, image: np.ndarray, scale_px_to_mm: float) -> np.ndarray:
        """Calculate distances between consecutive sutures."""
        boxes, _, _ = self.detect(image)
        distances = []
        
        # Sort boxes by x-coordinate
        boxes = sorted(boxes, key=lambda box: box[0])
        
        for i in range(len(boxes) - 1):
            x1 = boxes[i][2]  # End of current suture
            x2 = boxes[i+1][0]  # Start of next suture
            y1 = (boxes[i][1] + boxes[i][3]) / 2  # Middle y of current suture
            y2 = (boxes[i+1][1] + boxes[i+1][3]) / 2  # Middle y of next suture
            
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance * scale_px_to_mm)
            
        return np.array(distances)

    def measure_knot_incision_distances(self, image: np.ndarray) -> np.ndarray:
        """Measure distances from knots to incision line."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect incision line using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return np.array([])
            
        # Find the main incision line (longest line)
        main_line = max(lines, key=lambda line: 
                       np.sqrt((line[0][2] - line[0][0])**2 + 
                              (line[0][3] - line[0][1])**2))
        
        # Get knot positions (assuming they're at suture endpoints)
        boxes, _, _ = self.detect(image)
        distances = []
        
        for box in boxes:
            # Use both endpoints of each suture
            points = [(box[0], box[1]), (box[2], box[3])]
            
            for px, py in points:
                # Calculate perpendicular distance to incision line
                x1, y1, x2, y2 = main_line[0]
                distance = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / \
                          np.sqrt((y2-y1)**2 + (x2-x1)**2)
                distances.append(distance)
        
        return np.array(distances)

    def analyze_suture_pattern_symmetry(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze symmetry of suture pattern relative to incision line.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing symmetry metrics as float values
            
        Raises:
            MeasurementError: If analysis fails
        """
        try:
            boxes, classes, _ = self.detect(image)
            if len(boxes) == 0:
                raise MeasurementError("No sutures detected in image")
                
            # Find incision line
            incision_points = []
            suture_points = []
            
            for box, cls in zip(boxes, classes):
                if 'incision' in cls.lower():
                    incision_points.extend([(box[0], box[1]), (box[2], box[3])])
                elif 'suture' in cls.lower():
                    mid_x = (box[0] + box[2]) / 2
                    mid_y = (box[1] + box[3]) / 2
                    suture_points.append((mid_x, mid_y))
            
            if not incision_points or len(incision_points) < 2:
                raise MeasurementError("Incision line not detected")
            
            # Calculate distances and metrics
            distances = np.array([
                calculate_perpendicular_distance(
                    point, incision_points[0], incision_points[1]
                )
                for point in suture_points
            ], dtype=np.float32)
            
            return {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'max_deviation': float(np.max(np.abs(distances - np.mean(distances)))),
                'symmetry_score': float(1 / (1 + np.std(distances)))
            }
            
        except Exception as e:
            raise MeasurementError(f"Pattern symmetry analysis failed: {str(e)}")

    def analyze_suture_spacing_uniformity(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze uniformity of spacing between sutures.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary containing spacing uniformity metrics
        """
        boxes, classes, _ = self.detect(image)
        
        # Extract suture points in order along incision
        suture_points = []
        for box, cls in zip(boxes, classes):
            if 'suture' in cls.lower():
                mid_x = (box[0] + box[2]) / 2
                mid_y = (box[1] + box[3]) / 2
                suture_points.append((mid_x, mid_y))
        
        if len(suture_points) < 2:
            raise ValueError("Not enough sutures detected")
        
        # Sort points by x-coordinate (assuming horizontal incision)
        suture_points.sort(key=lambda p: p[0])
        
        # Calculate inter-suture distances
        distances = []
        for i in range(len(suture_points)-1):
            dist = euclidean_distance(suture_points[i], suture_points[i+1])
            distances.append(dist)
        
        distances = np.array(distances)
        return {
            'mean_spacing': float(np.mean(distances)),
            'spacing_std': float(np.std(distances)),
            'spacing_cv': float(np.std(distances) / np.mean(distances)),  # Coefficient of variation
            'min_spacing': float(np.min(distances)),
            'max_spacing': float(np.max(distances))
        }

    def evaluate_suture_depth_consistency(self, image: np.ndarray) -> Dict[str, float]:
        """Evaluate consistency of suture depths from the incision line.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary containing depth consistency metrics
        """
        boxes, classes, _ = self.detect(image)
        
        # Find incision and sutures
        incision_box = None
        suture_boxes = []
        
        for box, cls in zip(boxes, classes):
            if 'incision' in cls.lower():
                incision_box = box
            elif 'suture' in cls.lower():
                suture_boxes.append(box)
        
        if incision_box is None:
            raise ValueError("Incision not detected")
        
        # Calculate suture depths
        depths = []
        for box in suture_boxes:
            # Use the point of the suture furthest from the incision
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            
            dist1 = calculate_perpendicular_distance(
                p1, (incision_box[0], incision_box[1]), 
                (incision_box[2], incision_box[3])
            )
            dist2 = calculate_perpendicular_distance(
                p2, (incision_box[0], incision_box[1]), 
                (incision_box[2], incision_box[3])
            )
            depths.append(max(dist1, dist2))
        
        depths = np.array(depths)
        return {
            'mean_depth': float(np.mean(depths)),
            'depth_std': float(np.std(depths)),
            'depth_uniformity': float(1 / (1 + np.std(depths))),  # Normalized 0-1
            'min_depth': float(np.min(depths)),
            'max_depth': float(np.max(depths))
        }