"""API routes for suture analysis."""

import os
from typing import List, Dict
import base64
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
import cv2

from ..ml.model import SutureDetector

# Input validation models
class ImageInput(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    scale_px_to_mm: float = Field(1.0, description="Scale factor to convert pixels to millimeters")

    @validator('image')
    def validate_image(cls, v):
        try:
            img_data = base64.b64decode(v)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError
            return v
        except:
            raise ValueError("Invalid image encoding")

    @validator('scale_px_to_mm')
    def validate_scale(cls, v):
        if v <= 0:
            raise ValueError("Scale must be positive")
        return v

router = APIRouter()
model = SutureDetector(
    model_path=os.getenv('MODEL_WEIGHTS', 'weights/yolov8n.pt'),
    conf_threshold=float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.5'))
)

def decode_base64_image(image_str: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    try:
        img_data = base64.b64decode(image_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image encoding")

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze sutures in uploaded image."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
            
        # Perform detection
        boxes, scores, class_ids = model.detect(image)
        
        # Format response
        results = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            results.append({
                "box": box.tolist(),
                "score": float(score),
                "class_id": int(class_id)
            })
            
        return {"results": results}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.post("/predict_suture_quality")
async def predict_quality(image_input: ImageInput):
    """Predict suture quality (good, tight, loose)."""
    try:
        img = decode_base64_image(image_input.image)
        boxes, scores, _ = model.detect(img)
        
        # Get quality prediction
        quality = model.predict_quality(img, boxes)
        confidence = float(np.mean(scores)) if len(scores) > 0 else 0.0
        
        return {
            "prediction": quality,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stitch_length_measurement")
async def measure_stitch_length(image_input: ImageInput):
    """Measure stitch lengths in millimeters."""
    try:
        img = decode_base64_image(image_input.image)
        
        # Get measurements
        lengths = model.measure_stitch_lengths(img, image_input.scale_px_to_mm)
        
        if len(lengths) == 0:
            return JSONResponse(
                status_code=404,
                content={"error": "No stitches detected in image"}
            )
        
        return {
            "stitch_lengths_mm": lengths.tolist(),
            "average_length_mm": float(np.mean(lengths)),
            "std_dev_mm": float(np.std(lengths))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stitch_angle_analysis")
async def analyze_angles(image: dict):
    """Analyze angles of stitches."""
    try:
        img = decode_base64_image(image["image"])
        
        # Calculate angles
        angles = model.analyze_angles(img)
        
        return {
            "angles_degrees": angles.tolist(),
            "average_angle": float(np.mean(angles)) if len(angles) > 0 else 0.0
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/tail_length_measurement")
async def measure_tail_length(image: dict):
    """Measure tail lengths."""
    try:
        img = decode_base64_image(image["image"])
        
        # Measure tails
        lengths, missing = model.measure_tail_lengths(img)
        
        return {
            "tail_lengths_mm": lengths.tolist(),
            "missing_tail_count": int(missing)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/suture_distance_measurement")
async def measure_distances(image: dict):
    """Calculate distances between sutures."""
    try:
        img = decode_base64_image(image["image"])
        scale_px_to_mm = image.get("scale_px_to_mm", 1.0)
        
        # Calculate distances
        distances = model.measure_suture_distances(img, scale_px_to_mm)
        
        return {
            "suture_distances_mm": distances.tolist(),
            "average_distance_mm": float(np.mean(distances)) if len(distances) > 0 else 0.0
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/knot_incision_distance_measurement")
async def measure_knot_distances(image: dict):
    """Measure distances from knots to incision."""
    try:
        img = decode_base64_image(image["image"])
        
        # Measure knot-to-incision distances
        distances = model.measure_knot_incision_distances(img)
        
        return {
            "distances_mm": distances.tolist(),
            "average_distance_mm": float(np.mean(distances)) if len(distances) > 0 else 0.0
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/analyze_pattern_symmetry")
async def analyze_pattern_symmetry(image: dict):
    """Analyze symmetry of suture pattern."""
    try:
        img = decode_base64_image(image["image"])
        symmetry_metrics = model.analyze_suture_pattern_symmetry(img)
        return symmetry_metrics
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/analyze_spacing_uniformity")
async def analyze_spacing_uniformity(image: dict):
    """Analyze uniformity of spacing between sutures."""
    try:
        img = decode_base64_image(image["image"])
        uniformity_metrics = model.analyze_suture_spacing_uniformity(img)
        return uniformity_metrics
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/evaluate_depth_consistency")
async def evaluate_depth_consistency(image: dict):
    """Evaluate consistency of suture depths."""
    try:
        img = decode_base64_image(image["image"])
        depth_metrics = model.evaluate_suture_depth_consistency(img)
        return depth_metrics
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/comprehensive_pattern_analysis")
async def analyze_pattern_comprehensive(image: dict):
    """Perform comprehensive analysis of suture pattern."""
    try:
        img = decode_base64_image(image["image"])
        
        # Collect all pattern metrics
        symmetry_metrics = model.analyze_suture_pattern_symmetry(img)
        spacing_metrics = model.analyze_suture_spacing_uniformity(img)
        depth_metrics = model.evaluate_suture_depth_consistency(img)
        
        # Calculate overall pattern score (0-100)
        symmetry_score = symmetry_metrics['symmetry_score']
        spacing_uniformity = 1 - spacing_metrics['spacing_cv']  # Lower CV means better uniformity
        depth_uniformity = depth_metrics['depth_uniformity']
        
        overall_score = int(
            (symmetry_score + spacing_uniformity + depth_uniformity) / 3 * 100
        )
        
        return {
            "symmetry_analysis": symmetry_metrics,
            "spacing_analysis": spacing_metrics,
            "depth_analysis": depth_metrics,
            "overall_pattern_score": overall_score,
            "assessment": "good" if overall_score >= 80 else 
                        "fair" if overall_score >= 60 else "needs_improvement"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/calibrate")
async def get_calibration(image_input: ImageInput):
    """Get pixel-to-mm calibration factor using automated detection."""
    try:
        img = decode_base64_image(image_input.image)
        pixels_per_mm = model.get_calibration(img)
        return {
            "pixels_per_mm": float(pixels_per_mm),
            "method": "automated" if model.pixels_per_mm is not None else "manual"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train_quality_model")
async def train_quality_model(train_data: List[Dict[str, str]]):
    """Train XGBoost model for quality prediction."""
    try:
        images = []
        labels = []
        for item in train_data:
            img = decode_base64_image(item["image"])
            images.append(img)
            labels.append(item["label"])
            
        model.train_quality_model(images, labels)
        return {"status": "success", "message": "Quality model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_quality_detailed")
async def predict_quality_detailed(image_input: ImageInput):
    """Get detailed quality prediction with ablation study metrics."""
    try:
        img = decode_base64_image(image_input.image)
        boxes, scores, _ = model.detect(img)
        
        prediction, ablation_metrics = model.predict_quality(img, boxes)
        confidence = float(np.mean(scores)) if len(scores) > 0 else 0.0
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "ablation_metrics": ablation_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))