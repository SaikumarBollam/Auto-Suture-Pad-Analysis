from fastapi import APIRouter
from .suture import router as suture_router
from .health import router as health_router

router = APIRouter()

# Include all route modules
router.include_router(suture_router, prefix="/suture", tags=["suture"])
router.include_router(health_router, prefix="/health", tags=["health"]) 