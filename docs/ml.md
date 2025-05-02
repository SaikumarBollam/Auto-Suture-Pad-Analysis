# Machine Learning Documentation

## Model Architecture

The suture detection system uses YOLOv8, a state-of-the-art object detection model. The system is designed to detect and analyze surgical sutures in medical images.

## Model Configuration

The model configuration is stored in `config/config.yaml` and includes:
- Input image size: 640x640
- Confidence threshold: 0.5
- Device selection (CPU/CUDA)
- Data augmentation parameters

## Training

### Dataset Preparation
1. Organize your dataset in YOLO format:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### Training Process
1. Prepare a data.yaml file:
```yaml
path: dataset/
train: train/images
val: val/images
names:
  0: suture
```

2. Run training:
```python
from src.ml.model import SutureDetector

model = SutureDetector("weights/yolov8n.pt")
model.train("path/to/data.yaml", epochs=100)
```

## Inference

```python
import cv2
from src.ml.model import SutureDetector

detector = SutureDetector("weights/yolov8n.pt")
image = cv2.imread("image.jpg")
boxes, scores, class_ids = detector.detect(image)
```

## Model Performance

The model is designed to detect sutures with:
- High precision and recall
- Real-time inference capabilities
- Support for various image formats and sizes

## Best Practices

1. Use GPU acceleration when available
2. Preprocess images for optimal results
3. Regularly validate model performance
4. Keep model weights updated