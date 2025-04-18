# Utility Functions

This directory contains utility functions and helper classes used throughout the project.

## Components

### Logging (`logging.py`)
- Configurable logging setup
- Consistent log formatting
- Stream and file handlers

```python
from ml_models.utils import setup_logging

logger = setup_logging(__name__)
logger.info("Message")
```

### Transforms (`transforms.py`)
- Data augmentation pipelines
- Training and validation transforms
- Custom transform compositions

```python
from ml_models.utils import get_transforms

transforms = get_transforms(config)
train_transform = transforms['train']
val_transform = transforms['val']
```

### Validation (`validation.py`)
- Data validation utilities
- Format checking
- Quality assurance

```python
from ml_models.utils import DataValidation

validator = DataValidation(config)
is_valid = validator.validate_dataset("data/")
```

### Processing (`processing.py`)
- Common data processing utilities
- Image preprocessing
- Label processing

```python
from ml_models.utils import DataProcessing

processor = DataProcessing(config)
processed_data = processor.process_train(data)
```

### Performance Optimization (`performance_optimizer.py`)
- Model optimization utilities
- Quantization
- Batch processing
- ONNX conversion

```python
from ml_models.utils import PerformanceOptimizer

optimizer = PerformanceOptimizer()
optimized_model = optimizer.optimize_model(model, dummy_input)
```

### Visualization (`visualization.py`)
- Result visualization
- Training metrics plotting
- Detection visualization

```python
from ml_models.utils import Visualizer

visualizer = Visualizer(output_dir="output/")
visualizer.visualize_predictions(image_path, predictions)
```

## Common Usage Patterns

### Data Processing Pipeline
```python
from ml_models.utils import (
    setup_logging,
    DataProcessing,
    DataValidation
)

logger = setup_logging(__name__)
processor = DataProcessing(config)
validator = DataValidation(config)

# Process and validate data
if validator.validate_dataset(data_dir):
    processed_data = processor.process_train(data)
    logger.info("Data processed successfully")
```

### Model Optimization
```python
from ml_models.utils import PerformanceOptimizer

optimizer = PerformanceOptimizer()
results = optimizer.optimize_inference(model)
print(f"Speedup: {results['speedup']}x") 