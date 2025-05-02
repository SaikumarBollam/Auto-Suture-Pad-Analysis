"""Tests for ML components."""

import numpy as np
import pytest
import cv2
from src.ml.model import SutureDetector
from src.ml.utils.measurements import (
    calculate_angle, euclidean_distance, 
    calculate_perpendicular_distance, analyze_stitch_pattern
)
from src.ml.utils.image_processing import process_image_variants, enhance_image

@pytest.fixture
def test_image():
    """Create a test image with known suture pattern."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw some test sutures and incision line
    cv2.line(img, (320, 100), (320, 500), (0, 0, 255), 2)  # Incision
    cv2.line(img, (300, 200), (340, 200), (0, 255, 0), 2)  # Suture 1
    cv2.line(img, (280, 300), (360, 300), (0, 255, 0), 2)  # Suture 2
    return img

@pytest.fixture
def detector():
    """Initialize detector with test weights."""
    return SutureDetector("weights/yolov8n.pt")

def test_suture_detector_initialization():
    """Test SutureDetector initialization."""
    with pytest.raises(Exception):
        # Should raise error for invalid model path
        SutureDetector("invalid_path.pt")

def test_preprocess_image(test_image):
    """Test image preprocessing."""
    detector = SutureDetector("weights/yolov8n.pt")
    
    # Test RGB image
    processed = detector.preprocess_image(test_image)
    assert processed.shape == test_image.shape
    
    # Test grayscale image
    gray_image = np.zeros((640, 640), dtype=np.uint8)
    processed = detector.preprocess_image(gray_image)
    assert processed.shape == (640, 640, 3)
    
    # Test RGBA image
    rgba_image = np.zeros((640, 640, 4), dtype=np.uint8)
    processed = detector.preprocess_image(rgba_image)
    assert processed.shape == (640, 640, 3)

@pytest.mark.skipif(not SutureDetector("weights/yolov8n.pt").model, 
                    reason="Model weights not available")
def test_detect(test_image):
    """Test suture detection."""
    detector = SutureDetector("weights/yolov8n.pt")
    boxes, scores, class_ids = detector.detect(test_image)
    
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(class_ids, np.ndarray)
    assert len(boxes) == len(scores) == len(class_ids)

def test_predict_quality(test_image):
    """Test suture quality prediction."""
    detector = SutureDetector("weights/yolov8n.pt")
    boxes = np.array([[100, 100, 120, 120]])  # Mock detection
    quality = detector.predict_quality(test_image, boxes)
    
    assert quality in ["good", "tight", "loose"]

def test_measure_stitch_lengths(test_image):
    """Test stitch length measurement."""
    detector = SutureDetector("weights/yolov8n.pt")
    lengths = detector.measure_stitch_lengths(test_image, scale_px_to_mm=0.1)
    
    assert isinstance(lengths, np.ndarray)
    assert all(length >= 0 for length in lengths)

def test_analyze_angles(test_image):
    """Test angle analysis."""
    detector = SutureDetector("weights/yolov8n.pt")
    angles = detector.analyze_angles(test_image)
    
    assert isinstance(angles, np.ndarray)
    assert all(0 <= angle <= 180 for angle in angles)

def test_measure_tail_lengths(test_image):
    """Test tail length measurement."""
    detector = SutureDetector("weights/yolov8n.pt")
    lengths, missing = detector.measure_tail_lengths(test_image)
    
    assert isinstance(lengths, np.ndarray)
    assert isinstance(missing, int)
    assert missing >= 0
    assert all(length >= 0 for length in lengths)

def test_measure_suture_distances(test_image):
    """Test suture distance measurement."""
    detector = SutureDetector("weights/yolov8n.pt")
    distances = detector.measure_suture_distances(test_image, scale_px_to_mm=0.1)
    
    assert isinstance(distances, np.ndarray)
    assert all(distance >= 0 for distance in distances)

def test_measure_knot_incision_distances(test_image):
    """Test knot-to-incision distance measurement."""
    detector = SutureDetector("weights/yolov8n.pt")
    distances = detector.measure_knot_incision_distances(test_image)
    
    assert isinstance(distances, np.ndarray)
    assert all(distance >= 0 for distance in distances)

def test_image_processing():
    """Test image preprocessing functions."""
    # Create test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (60, 60), (255, 255, 255), -1)
    
    # Test processing variants
    gray, edges, contours = process_image_variants(img, draw_contours=True)
    assert gray.shape[:2] == img.shape[:2]
    assert edges.shape[:2] == img.shape[:2]
    assert contours.shape == img.shape
    
    # Test enhancement
    enhanced = enhance_image(img)
    assert enhanced.shape == img.shape

def test_measurement_utils():
    """Test measurement utility functions."""
    p1 = (0, 0)
    p2 = (3, 4)
    
    # Test Euclidean distance
    dist = euclidean_distance(p1, p2)
    assert dist == 5.0
    
    # Test with mm conversion
    dist_mm = euclidean_distance(p1, p2, pixels_per_mm=2)
    assert dist_mm == 2.5
    
    # Test angle calculation
    angle = calculate_angle(p1, p2)
    assert 0 <= angle <= 180
    
    # Test perpendicular distance
    point = (0, 1)
    line_p1 = (0, 0)
    line_p2 = (1, 0)
    perp_dist = calculate_perpendicular_distance(point, line_p1, line_p2)
    assert abs(perp_dist - 1.0) < 1e-6

def test_stitch_pattern_analysis(test_image, detector):
    """Test stitch pattern analysis."""
    # Set reference points for mm conversion
    detector.set_reference_points([(0.2, 0.5), (0.2, 0.3)], (640, 640))
    
    # Test pattern analysis
    pattern = detector.analyze_pattern(test_image)
    assert isinstance(pattern, dict)
    assert 'avg_spacing' in pattern
    assert 'avg_angle' in pattern
    assert pattern['avg_spacing'] > 0
    
def test_angle_analysis(test_image, detector):
    """Test angle analysis."""
    angles = detector.analyze_angles(test_image)
    assert isinstance(angles, np.ndarray)
    assert all(0 <= angle <= 180 for angle in angles)

def test_stitch_length_measurement(test_image, detector):
    """Test stitch length measurement."""
    # Test with manual scale
    lengths = detector.measure_stitch_lengths(test_image, scale_px_to_mm=0.1)
    assert isinstance(lengths, np.ndarray)
    assert all(length > 0 for length in lengths)
    
    # Test with reference points
    detector.set_reference_points([(0.2, 0.5), (0.2, 0.3)], (640, 640))
    lengths = detector.measure_stitch_lengths(test_image)
    assert isinstance(lengths, np.ndarray)
    assert all(length > 0 for length in lengths)

def test_pattern_symmetry_analysis(test_image, detector):
    """Test suture pattern symmetry analysis."""
    metrics = detector.analyze_suture_pattern_symmetry(test_image)
    assert isinstance(metrics, dict)
    assert "mean_distance" in metrics
    assert "std_distance" in metrics
    assert "max_deviation" in metrics
    assert "symmetry_score" in metrics
    assert isinstance(metrics["symmetry_score"], float)
    assert 0 <= metrics["symmetry_score"] <= 1

def test_spacing_uniformity_analysis(test_image, detector):
    """Test suture spacing uniformity analysis."""
    metrics = detector.analyze_suture_spacing_uniformity(test_image)
    assert isinstance(metrics, dict)
    assert "mean_spacing" in metrics
    assert "spacing_std" in metrics
    assert "spacing_cv" in metrics
    assert "min_spacing" in metrics
    assert "max_spacing" in metrics
    assert metrics["spacing_cv"] >= 0

def test_depth_consistency_analysis(test_image, detector):
    """Test suture depth consistency analysis."""
    metrics = detector.evaluate_suture_depth_consistency(test_image)
    assert isinstance(metrics, dict)
    assert "mean_depth" in metrics
    assert "depth_std" in metrics
    assert "depth_uniformity" in metrics
    assert "min_depth" in metrics
    assert "max_depth" in metrics
    assert 0 <= metrics["depth_uniformity"] <= 1

def test_invalid_pattern_analysis(test_image, detector):
    """Test pattern analysis with invalid inputs."""
    # Create blank image with no detections
    blank_image = np.zeros_like(test_image)
    
    # Test pattern symmetry with no incision
    with pytest.raises(ValueError, match="Incision line not detected"):
        detector.analyze_suture_pattern_symmetry(blank_image)
    
    # Test spacing uniformity with insufficient sutures
    with pytest.raises(ValueError, match="Not enough sutures detected"):
        detector.analyze_suture_spacing_uniformity(blank_image)
    
    # Test depth consistency with no incision
    with pytest.raises(ValueError, match="Incision not detected"):
        detector.evaluate_suture_depth_consistency(blank_image)