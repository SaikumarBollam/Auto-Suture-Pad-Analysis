import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

# Load environment variables
env_path = Path(__file__).parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="Auto-Suture-Pad Analysis API",
    description="API for analyzing surgical sutures using computer vision",
    version="0.1.0"
)

# Configure CORS with environment variables
allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:8080').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["analysis"])

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}