"""Test configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture
def test_image():
    """Create a simple test image."""
    import numpy as np
    return np.zeros((640, 640, 3), dtype=np.uint8)