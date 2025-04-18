from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from PIL import Image
import io
import torch
import base64
import time
from typing import Dict, Any, List
import sys
import os
from pydantic import BaseModel
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

try:
    from ml.core.inference import InferencePipeline
    from ml.utils.config import load_config
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

from .config import settings
from .services.suture_service import SutureService
from .logging_config import logger
from .monitoring import performance_monitor

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="REST API for YOLO-based suture analysis including detailed stitch measurements, suture classifications, advanced image preprocessing, robust detection methods, pixel-to-mm conversions, and complete error handling.",
    version=settings.MODEL_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize security
security = HTTPBearer()

# Initialize the ML pipeline and service
config = load_config(settings.CONFIG_PATH)
model = torch.load(settings.MODEL_PATH, map_location="cpu")
pipeline = InferencePipeline(model, config)
suture_service = SutureService(pipeline)

# Request/Response Models
class ImageRequest(BaseModel):
    """Base model for image requests."""
    image: str  # Base64 encoded image
    model_version: str = settings.MODEL_VERSION

class ScaleRequest(ImageRequest):
    """Model for requests requiring scale information."""
    scale_px_to_mm: float

class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    model_loaded: bool
    service_ready: bool
    model_version: str
    uptime: str
    performance_metrics: Dict[str, Any]

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    performance_monitor.record_request(process_time)
    return response

async def get_service() -> SutureService:
    """Dependency to get the suture service instance."""
    return suture_service

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the bearer token."""
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"status": "ok", "message": f"{settings.PROJECT_NAME} is running"}

@app.post("/predict_suture_quality", dependencies=[Security(get_current_user)])
async def predict_suture_quality(
    request: ImageRequest,
    service: SutureService = Depends(get_service)
) -> Dict[str, Any]:
    """Predict suture quality (good, tight, loose)."""
    start_time = time.time()
    try:
        logger.info("Received suture quality prediction request")
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
        prediction, confidence = await service.predict_quality(image)
        
        inference_time = time.time() - start_time
        performance_monitor.record_model_inference(inference_time)
        
        logger.info(f"Quality prediction completed in {inference_time:.2f}s")
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_version": request.model_version
        }
    except Exception as e:
        logger.error(f"Error in quality prediction: {str(e)}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stitch_length_measurement", dependencies=[Security(get_current_user)])
async def measure_stitch_lengths(
    request: ScaleRequest,
    service: SutureService = Depends(get_service)
) -> Dict[str, Any]:
    """Measure stitch lengths."""
    start_time = time.time()
    try:
        logger.info("Received stitch length measurement request")
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
        lengths = await service.measure_stitch_lengths(image, request.scale_px_to_mm)
        
        inference_time = time.time() - start_time
        performance_monitor.record_model_inference(inference_time)
        
        logger.info(f"Stitch length measurement completed in {inference_time:.2f}s")
        return {
            "stitch_lengths_mm": lengths,
            "average_length_mm": sum(lengths) / len(lengths) if lengths else 0,
            "model_version": request.model_version
        }
    except Exception as e:
        logger.error(f"Error in stitch length measurement: {str(e)}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stitch_angle_analysis", dependencies=[Security(get_current_user)])
async def analyze_stitch_angles(
    request: ImageRequest,
    service: SutureService = Depends(get_service)
) -> Dict[str, Any]:
    """Analyze angles of stitches."""
    start_time = time.time()
    try:
        logger.info("Received stitch angle analysis request")
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
        angles = await service.analyze_stitch_angles(image)
        
        inference_time = time.time() - start_time
        performance_monitor.record_model_inference(inference_time)
        
        logger.info(f"Stitch angle analysis completed in {inference_time:.2f}s")
        return {
            "angles_degrees": angles,
            "average_angle": sum(angles) / len(angles) if angles else 0,
            "model_version": request.model_version
        }
    except Exception as e:
        logger.error(f"Error in stitch angle analysis: {str(e)}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tail_length_measurement", dependencies=[Security(get_current_user)])
async def measure_tail_lengths(
    request: ImageRequest,
    service: SutureService = Depends(get_service)
) -> Dict[str, Any]:
    """Measure tail lengths."""
    start_time = time.time()
    try:
        logger.info("Received tail length measurement request")
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
        results = await service.measure_tail_lengths(image)
        
        inference_time = time.time() - start_time
        performance_monitor.record_model_inference(inference_time)
        
        logger.info(f"Tail length measurement completed in {inference_time:.2f}s")
        return {
            "tail_lengths_mm": results["lengths"],
            "missing_tail_count": results["missing_count"],
            "model_version": request.model_version
        }
    except Exception as e:
        logger.error(f"Error in tail length measurement: {str(e)}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suture_distance_measurement", dependencies=[Security(get_current_user)])
async def measure_suture_distances(
    request: ScaleRequest,
    service: SutureService = Depends(get_service)
) -> Dict[str, Any]:
    """Calculate distances between sutures."""
    start_time = time.time()
    try:
        logger.info("Received suture distance measurement request")
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
        distances = await service.measure_suture_distances(image, request.scale_px_to_mm)
        
        inference_time = time.time() - start_time
        performance_monitor.record_model_inference(inference_time)
        
        logger.info(f"Suture distance measurement completed in {inference_time:.2f}s")
        return {
            "suture_distances_mm": distances,
            "average_distance_mm": sum(distances) / len(distances) if distances else 0,
            "model_version": request.model_version
        }
    except Exception as e:
        logger.error(f"Error in suture distance measurement: {str(e)}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knot_incision_distance_measurement", dependencies=[Security(get_current_user)])
async def measure_knot_incision_distances(
    request: ImageRequest,
    service: SutureService = Depends(get_service)
) -> Dict[str, Any]:
    """Measure distances from knots to incision."""
    start_time = time.time()
    try:
        logger.info("Received knot incision distance measurement request")
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
        distances = await service.measure_knot_incision_distances(image)
        
        inference_time = time.time() - start_time
        performance_monitor.record_model_inference(inference_time)
        
        logger.info(f"Knot incision distance measurement completed in {inference_time:.2f}s")
        return {
            "distances_mm": distances,
            "average_distance_mm": sum(distances) / len(distances) if distances else 0,
            "model_version": request.model_version
        }
    except Exception as e:
        logger.error(f"Error in knot incision distance measurement: {str(e)}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    await FastAPILimiter.init(redis)

@app.get("/health", response_model=HealthResponse)
@RateLimiter(times=100, minutes=1)
async def health_check():
    """Health check endpoint."""
    try:
        # Test model with sample input
        sample_image = Image.new('RGB', settings.IMAGE_SIZE)
        pipeline.analyze_suture(sample_image)
        model_healthy = True
    except Exception as e:
        logger.error(f"Model health check failed: {str(e)}")
        model_healthy = False
    
    return {
        "status": "healthy" if model_healthy else "degraded",
        "model_loaded": pipeline is not None,
        "service_ready": suture_service is not None,
        "model_version": settings.MODEL_VERSION,
        "uptime": str(datetime.now() - app.startup_time),
        "performance_metrics": performance_monitor.get_metrics()
    }

# Store startup time
app.startup_time = datetime.now() 