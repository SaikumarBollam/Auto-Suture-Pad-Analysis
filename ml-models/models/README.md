# Models Layer

This layer contains the machine learning models for suture analysis.

## Available Models

### YOLOSutureModel
YOLO-based model for suture detection and classification.

#### Features
- Object detection
- Real-time inference
- Transfer learning support

### CNNSutureModel
CNN-based model for suture analysis.

#### Features
- Feature extraction
- Classification
- Custom architecture

## Usage

```python
from ml_models.models import get_model
from ml_models.config import Config

# Initialize configuration
config = Config()

# Get YOLO model
yolo_model = get_model("yolo", **config.get_model_config())

# Get CNN model
cnn_model = get_model("cnn", **config.get_model_config())

# Make predictions
predictions = model.predict(image_tensor)
```

## Model Architecture

### YOLO Model
- Uses Ultralytics YOLO implementation
- Custom head for suture-specific tasks
- Optimized for real-time inference

### CNN Model
- Feature extraction layers
- Classification head
- Support for additional metrics input

## Configuration

Key model parameters can be configured through the Config class:
- Model type
- Input size
- Number of classes
- Weights path
- Architecture parameters

See `config.py` for all available parameters. 