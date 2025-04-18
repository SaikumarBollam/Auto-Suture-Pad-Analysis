"""Tests for model architectures."""

import pytest
import torch
from pathlib import Path
from ...core.config.config import Config
from ...models.model import get_model
from ...models import get_model as old_get_model, YOLOv12SutureModel

def test_get_model():
    config = Config()
    model = get_model(config)
    
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'forward')

def test_model_creation():
    """Test model creation through factory function."""
    model = old_get_model(
        model_type='yolo',
        model_size='n',
        num_classes=2,
        pretrained=False
    )
    assert isinstance(model, YOLOv12SutureModel)

def test_model_forward():
    """Test model forward pass."""
    model = YOLOv12SutureModel(
        model_size='n',
        num_classes=2,
        pretrained=False
    )
    
    # Create dummy input
    batch_size = 2
    channels = 3
    height = 640
    width = 640
    x = torch.randn(batch_size, channels, height, width)
    
    # Run forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output
    assert isinstance(output, list)
    assert len(output) == batch_size

def test_model_invalid_size():
    """Test model creation with invalid size."""
    with pytest.raises(ValueError):
        YOLOv12SutureModel(
            model_size='invalid',
            num_classes=2,
            pretrained=False
        )

def test_model_invalid_classes():
    """Test model creation with invalid number of classes."""
    with pytest.raises(ValueError):
        YOLOv12SutureModel(
            model_size='n',
            num_classes=0,
            pretrained=False
        ) 