# Utilities Layer

This layer contains utility functions and classes used across the project.

## Components

### PerformanceOptimizer
Class for optimizing model performance.

#### Features
- Model quantization
- ONNX conversion
- Batch processing
- Performance benchmarking

### Visualization
Functions for visualizing results and metrics.

#### Features
- Training metrics plotting
- Prediction visualization
- Confusion matrix plotting
- Performance curves

## Usage

### Performance Optimization
```python
from ml_models.utils import PerformanceOptimizer

# Initialize optimizer
optimizer = PerformanceOptimizer()

# Optimize model
optimized_model = optimizer.optimize_model(
    model=model,
    dummy_input=dummy_input,
    save_path="optimized_model.pt"
)

# Benchmark performance
results = optimizer.optimize_inference(
    model=model,
    input_shape=(1, 3, 224, 224)
)
```

### Visualization
```python
from ml_models.utils import plot_training_metrics

# Plot training metrics
plot_training_metrics(
    metrics={
        'train_loss': [...],
        'val_loss': [...],
        'train_acc': [...],
        'val_acc': [...]
    },
    save_path="training_metrics.png"
)
```

## Features

### Performance Optimization
- Model quantization for faster inference
- ONNX conversion for deployment
- Batch processing for memory efficiency
- Performance benchmarking

### Visualization
- Training curves
- Validation metrics
- Confusion matrices
- Prediction overlays
- Performance plots

## Configuration

Utility parameters can be configured through the Config class:
- Optimization parameters
- Visualization settings
- Output formats
- Save paths

See `config.py` for all available parameters. 