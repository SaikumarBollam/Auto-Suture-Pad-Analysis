"""Model definitions for suture detection."""

from .model import (
    SutureModel,
    YOLOv12SutureModel,
    get_model
)

__all__ = [
    'SutureModel',
    'YOLOv12SutureModel',
    'get_model'
] 