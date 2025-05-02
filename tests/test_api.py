"""Tests for API endpoints."""

import io
import base64
import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from typing import Dict

from src.api.routes import router

client = TestClient(router)

@pytest.fixture
def test_image() -> np.ndarray:
    """Create a test image with sutures."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw test patterns
    cv2.line(img, (320, 100), (320, 500), (0, 0, 255), 2)  # Incision
    cv2.line(img, (300, 200), (340, 200), (0, 255, 0), 2)  # Suture 1
    cv2.line(img, (280, 300), (360, 300), (0, 255, 0), 2)  # Suture 2
    return img

@pytest.fixture
def image_payload(test_image) -> Dict[str, str]:
    """Create base64 encoded image payload."""
    _, buffer = cv2.imencode('.png', test_image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return {"image": img_str}

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_analyze_endpoint(client, test_image):
    """Test the image analysis endpoint."""
    # Convert numpy array to bytes
    img = Image.fromarray(test_image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Send request
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test.png", img_byte_arr, "image/png")}
    )
    
    assert response.status_code == 200
    assert "results" in response.json()

def test_analyze_image(image_payload):
    """Test comprehensive image analysis endpoint."""
    response = client.post(
        "/analyze",
        json={
            **image_payload,
            "scale_px_to_mm": 0.1,
            "ref_points": [[0.2, 0.5], [0.2, 0.3]]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "measurements" in data
    assert "visualization" in data

def test_stitch_pattern(image_payload):
    """Test stitch pattern analysis endpoint."""
    response = client.post(
        "/stitch_pattern",
        json={
            **image_payload,
            "ref_points": [[0.2, 0.5], [0.2, 0.3]]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "avg_spacing" in data
    assert "avg_angle" in data

def test_stitch_lengths(image_payload):
    """Test stitch length measurement endpoint."""
    response = client.post(
        "/stitch_lengths",
        json={
            **image_payload,
            "scale_px_to_mm": 0.1
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "lengths_mm" in data
    assert "average_length" in data
    assert "std_deviation" in data
    assert isinstance(data["lengths_mm"], list)

def test_stitch_angles(image_payload):
    """Test stitch angle analysis endpoint."""
    response = client.post(
        "/stitch_angles",
        json=image_payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "angles_degrees" in data
    assert "average_angle" in data
    assert "std_deviation" in data

def test_predict_quality(image_payload):
    """Test suture quality prediction endpoint."""
    response = client.post(
        "/predict_suture_quality",
        json=image_payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["good", "tight", "loose"]

def test_tail_length_measurement(image_payload):
    """Test tail length measurement endpoint."""
    response = client.post(
        "/tail_length_measurement",
        json=image_payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "tail_lengths_mm" in data
    assert "missing_tail_count" in data
    assert isinstance(data["tail_lengths_mm"], list)

def test_suture_distance_measurement(image_payload):
    """Test suture distance measurement endpoint."""
    response = client.post(
        "/suture_distance_measurement",
        json={
            **image_payload,
            "scale_px_to_mm": 0.1
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "suture_distances_mm" in data
    assert "average_distance_mm" in data

def test_knot_incision_distance_measurement(image_payload):
    """Test knot-to-incision distance measurement endpoint."""
    response = client.post(
        "/knot_incision_distance_measurement",
        json=image_payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "distances_mm" in data
    assert "average_distance_mm" in data

def test_visualize_analysis(image_payload):
    """Test analysis visualization endpoint."""
    response = client.post(
        "/visualize_analysis",
        json={
            **image_payload,
            "measurements": {
                "avg_spacing": 10.5,
                "avg_angle": 45.0
            },
            "points": [[100, 100], [200, 200]],
            "incision_line": [[320, 100], [320, 500]]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "visualization" in data
    assert isinstance(data["visualization"], str)

def test_analyze_invalid_image(client):
    """Test the analysis endpoint with invalid image."""
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test.txt", b"invalid image data", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "error" in response.json()

def test_pattern_symmetry_endpoint(image_payload):
    """Test pattern symmetry analysis endpoint."""
    response = client.post("/analyze_pattern_symmetry", json=image_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "mean_distance" in data
    assert "std_distance" in data
    assert "symmetry_score" in data
    assert 0 <= data["symmetry_score"] <= 1

def test_spacing_uniformity_endpoint(image_payload):
    """Test spacing uniformity analysis endpoint."""
    response = client.post("/analyze_spacing_uniformity", json=image_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "mean_spacing" in data
    assert "spacing_std" in data
    assert "spacing_cv" in data
    assert data["spacing_cv"] >= 0

def test_depth_consistency_endpoint(image_payload):
    """Test depth consistency analysis endpoint."""
    response = client.post("/evaluate_depth_consistency", json=image_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "mean_depth" in data
    assert "depth_std" in data
    assert "depth_uniformity" in data
    assert 0 <= data["depth_uniformity"] <= 1

def test_comprehensive_pattern_analysis_endpoint(image_payload):
    """Test comprehensive pattern analysis endpoint."""
    response = client.post("/comprehensive_pattern_analysis", json=image_payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "symmetry_analysis" in data
    assert "spacing_analysis" in data
    assert "depth_analysis" in data
    assert "overall_pattern_score" in data
    assert "assessment" in data
    
    assert 0 <= data["overall_pattern_score"] <= 100
    assert data["assessment"] in ["good", "fair", "needs_improvement"]

@pytest.mark.parametrize("endpoint", [
    "/analyze_pattern_symmetry",
    "/analyze_spacing_uniformity",
    "/evaluate_depth_consistency",
    "/comprehensive_pattern_analysis"
])
def test_pattern_analysis_error_handling(endpoint, client):
    """Test error handling for pattern analysis endpoints."""
    # Test with invalid image data
    response = client.post(
        endpoint,
        json={"image": "invalid_base64_data"}
    )
    assert response.status_code == 400
    
    # Test with missing image data
    response = client.post(endpoint, json={})
    assert response.status_code == 422

def test_comprehensive_analysis_with_partial_failure(image_payload):
    """Test comprehensive analysis with some metrics failing."""
    # Create an image that will fail depth analysis but pass symmetry
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Only draw sutures, no incision
    cv2.line(img, (300, 200), (340, 200), (0, 255, 0), 2)
    cv2.line(img, (280, 300), (360, 300), (0, 255, 0), 2)
    
    _, buffer = cv2.imencode('.png', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    payload = {"image": img_str}
    
    response = client.post("/comprehensive_pattern_analysis", json=payload)
    assert response.status_code == 500
    assert "error" in response.json()