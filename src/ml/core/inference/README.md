# Inference Layer

This layer handles model inference and prediction visualization.

## Components

### InferencePipeline
Main class for running inference and visualizing predictions.

#### Features
- Model loading
- Batch processing
- Prediction visualization
- Results saving

## Usage

```python
from ml_models.inference import InferencePipeline
from ml_models.config import Config

# Initialize configuration
config = Config()

# Initialize pipeline
pipeline = InferencePipeline(
    model_type="yolo",
    model_path=config.get_model_config()['weights_path']
)

# Process a single image
predictions = pipeline.predict(image_tensor)

# Process a directory of images
pipeline.process_directory(
    input_dir="path/to/images",
    output_dir="path/to/output",
    class_names=["suture", "knot"]
)
```

## Inference Process

1. **Model Loading**
   - Load trained model
   - Set to evaluation mode
   - Configure device

2. **Prediction**
   - Image preprocessing
   - Forward pass
   - Post-processing
   - Results formatting

3. **Visualization**
   - Bounding box drawing
   - Label overlay
   - Confidence scores
   - Save results

## Output Format

### YOLO Model
```python
{
    'boxes': np.array,  # Bounding boxes [x1, y1, x2, y2]
    'scores': np.array,  # Confidence scores
    'labels': np.array   # Class labels
}
```

### CNN Model
```python
{
    'class': np.array,        # Predicted class
    'probabilities': np.array # Class probabilities
}
```

## Configuration

Key inference parameters can be configured through the Config class:
- Model type
- Weights path
- Device selection
- Visualization parameters
- Output format

See `config.py` for all available parameters. 