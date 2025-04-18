from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import base64
import io
from PIL import Image
import numpy as np
import logging

from ..dependencies import get_inference_pipeline, get_device
from ..utils.errors import (
    ModelError,
    ValidationError,
    handle_model_error,
    handle_validation_error,
    handle_generic_error
)
from ml_models.utils.validation import validate_image

router = APIRouter()
logger = logging.getLogger(__name__)

def decode_base64_image(image_str: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    try:
        image_data = base64.b64decode(image_str)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValidationError("Invalid image format", field="image")

@router.post("/quality")
async def predict_suture_quality(
    image: str,
    inference: InferencePipeline = Depends(get_inference_pipeline)
) -> Dict[str, Any]:
    """Predict suture quality (good, tight, loose)."""
    try:
        img = decode_base64_image(image)
        if not validate_image(img):
            raise ValidationError("Invalid image", field="image")
            
        prediction = inference.predict_quality(img)
        return {
            "prediction": prediction["class"],
            "confidence": float(prediction["confidence"])
        }
    except ValidationError as e:
        raise handle_validation_error(e)
    except ModelError as e:
        raise handle_model_error(e)
    except Exception as e:
        raise handle_generic_error(e)

@router.post("/lengths")
async def measure_stitch_lengths(
    image: str,
    scale_px_to_mm: float,
    inference: InferencePipeline = Depends(get_inference_pipeline)
) -> Dict[str, Any]:
    """Measure stitch lengths."""
    try:
        img = decode_base64_image(image)
        if not validate_image(img):
            raise ValidationError("Invalid image", field="image")
            
        lengths = inference.measure_stitch_lengths(img, scale_px_to_mm)
        return {
            "stitch_lengths_mm": [float(l) for l in lengths],
            "average_length_mm": float(np.mean(lengths))
        }
    except ValidationError as e:
        raise handle_validation_error(e)
    except ModelError as e:
        raise handle_model_error(e)
    except Exception as e:
        raise handle_generic_error(e)

@router.post("/angles")
async def analyze_stitch_angles(
    image: str,
    inference: InferencePipeline = Depends(get_inference_pipeline)
) -> Dict[str, Any]:
    """Analyze angles of stitches."""
    try:
        img = decode_base64_image(image)
        if not validate_image(img):
            raise ValidationError("Invalid image", field="image")
            
        angles = inference.analyze_stitch_angles(img)
        return {
            "angles_degrees": [float(a) for a in angles],
            "average_angle": float(np.mean(angles))
        }
    except ValidationError as e:
        raise handle_validation_error(e)
    except ModelError as e:
        raise handle_model_error(e)
    except Exception as e:
        raise handle_generic_error(e)

@router.post("/tails")
async def measure_tail_lengths(
    image: str,
    inference: InferencePipeline = Depends(get_inference_pipeline)
) -> Dict[str, Any]:
    """Measure tail lengths."""
    try:
        img = decode_base64_image(image)
        if not validate_image(img):
            raise ValidationError("Invalid image", field="image")
            
        results = inference.measure_tail_lengths(img)
        return {
            "tail_lengths_mm": [float(l) for l in results["lengths"]],
            "missing_tail_count": results["missing_count"]
        }
    except ValidationError as e:
        raise handle_validation_error(e)
    except ModelError as e:
        raise handle_model_error(e)
    except Exception as e:
        raise handle_generic_error(e)

@router.post("/distances")
async def measure_suture_distances(
    image: str,
    scale_px_to_mm: float,
    inference: InferencePipeline = Depends(get_inference_pipeline)
) -> Dict[str, Any]:
    """Calculate distances between sutures."""
    try:
        img = decode_base64_image(image)
        if not validate_image(img):
            raise ValidationError("Invalid image", field="image")
            
        distances = inference.measure_suture_distances(img, scale_px_to_mm)
        return {
            "suture_distances_mm": [float(d) for d in distances],
            "average_distance_mm": float(np.mean(distances))
        }
    except ValidationError as e:
        raise handle_validation_error(e)
    except ModelError as e:
        raise handle_model_error(e)
    except Exception as e:
        raise handle_generic_error(e)

@router.post("/knots")
async def measure_knot_incision_distances(
    image: str,
    inference: InferencePipeline = Depends(get_inference_pipeline)
) -> Dict[str, Any]:
    """Measure distances from knots to incision."""
    try:
        img = decode_base64_image(image)
        if not validate_image(img):
            raise ValidationError("Invalid image", field="image")
            
        distances = inference.measure_knot_incision_distances(img)
        return {
            "distances_mm": [float(d) for d in distances],
            "average_distance_mm": float(np.mean(distances))
        }
    except ValidationError as e:
        raise handle_validation_error(e)
    except ModelError as e:
        raise handle_model_error(e)
    except Exception as e:
        raise handle_generic_error(e) 