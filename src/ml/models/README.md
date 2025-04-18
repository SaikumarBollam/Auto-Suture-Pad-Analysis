# Model Architectures

This directory contains the model architectures for suture detection.

## Components

### Base Model

`SutureModel` (base.py):
- Abstract base class for all suture detection models
- Defines common interface and utilities
- Implements basic model operations

### YOLOv12 Implementation

`YOLOv12SutureModel` (model.py):
- YOLOv12-based implementation
- Features:
  - Multiple model sizes (n, s, m, l, x)
  - Pretrained weights support
  - Custom number of classes
  - Batch processing
  - GPU acceleration

### Model Factory

`get_model()` (model.py):
- Factory function for model creation
- Supports different model types and configurations
- Handles model initialization and setup

## Usage

```python
from ml_models.models import get_model, YOLOv12SutureModel

# Using factory function
model = get_model(
    model_type='yolo',
    model_size='l',
    num_classes=2,
    pretrained=True
)

# Direct instantiation
model = YOLOv12SutureModel(
    model_size='l',
    num_classes=2,
    pretrained=True
)
```

## Model Sizes

| Size | Parameters | Input Resolution | Speed | Memory |
|------|------------|-----------------|-------|---------|
| n    | 1.8M      | 640x640         | Fast  | Low     |
| s    | 7.2M      | 640x640         | Fast  | Medium  |
| m    | 21.2M     | 640x640         | Med   | Medium  |
| l    | 46.5M     | 640x640         | Slow  | High    |
| x    | 86.7M     | 640x640         | Slow  | High    |

## Extending

To add a new model:

1. Inherit from `SutureModel`
2. Implement required methods
3. Add to model factory
4. Update documentation

Example:
```python
class CustomSutureModel(SutureModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Implementation

    def forward(self, x):
        # Implementation

    def predict(self, x):
        # Implementation
``` 