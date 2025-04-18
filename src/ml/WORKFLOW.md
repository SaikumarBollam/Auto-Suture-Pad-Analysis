# ML Models Workflow Guide

This document provides a comprehensive guide to understanding, setting up, and running the ML Models codebase.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Configuration](#configuration)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [Inference](#inference)
7. [Testing](#testing)
8. [Common Workflows](#common-workflows)

## Project Structure

```
ml-models/
├── config/                 # Configuration files
│   ├── model_config.yaml   # Model architecture and training settings
│   ├── training_config.yaml# Training pipeline settings
│   ├── data_config.yaml    # Dataset configuration
│   └── read_config.yaml    # File reading settings
├── core/                   # Core functionality
│   ├── models/            # Model implementations
│   ├── training/          # Training pipeline
│   └── inference/         # Inference pipeline
├── data/                  # Data handling
│   ├── reader.py         # Data reading
│   ├── processor.py      # Data processing
│   ├── manager.py        # Data management
│   └── validator.py      # Data validation
├── utils/                # Utility functions
├── tests/               # Test suite
└── notebooks/           # Example notebooks
```

## Setup and Installation

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

2. **Docker Setup** (Alternative)
```bash
# Option 1: Using Docker directly
docker build -t ml-models .
docker run -it ml-models

# Option 2: Using Docker Compose (Recommended)
# This will set up the complete environment with all services
docker-compose up -d

# Access services:
# - Jupyter Notebook: http://localhost:8888
# - TensorBoard: http://localhost:6006
# - MLflow: http://localhost:5000
# - MinIO Console: http://localhost:9001

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

The Docker Compose setup includes:
- ML Models service with GPU support
- TensorBoard for visualization
- Redis for caching
- MLflow for experiment tracking
- MinIO for model storage

## Configuration

1. **Understanding Configuration Files**
   - `model_config.yaml`: Model architecture and training parameters
   - `training_config.yaml`: Training pipeline settings
   - `data_config.yaml`: Dataset and preprocessing settings
   - `read_config.yaml`: File reading and validation settings

2. **Modifying Configurations**
```python
import yaml
from pathlib import Path

def load_config(config_name):
    config_path = Path('config') / f'{config_name}.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

## Data Pipeline

1. **Data Preparation**
```python
from ml_models.data.manager import DataManager
from ml_models.data.reader import DataReader

# Load configurations
data_config = load_config('data_config')
read_config = load_config('read_config')

# Initialize data components
data_reader = DataReader(read_config)
data_manager = DataManager(data_config)

# Read and process dataset
samples = data_reader.read_dataset(
    image_dir='data/raw/images',
    annotation_dir='data/raw/annotations'
)
```

2. **Data Validation**
```python
from ml_models.data.validator import DataValidator

validator = DataValidator(read_config)
valid_samples = [s for s in samples if validator.validate_sample(s)]
```

## Model Training

1. **Training Setup**
```python
from ml_models.core.training import TrainingPipeline

# Load configurations
model_config = load_config('model_config')
training_config = load_config('training_config')

# Initialize training pipeline
pipeline = TrainingPipeline(model_config)

# Start training
results = pipeline.train(num_epochs=training_config['training']['num_epochs'])
```

2. **Monitoring Training**
```python
# Training progress is automatically logged
# Check logs/training.log for details

# TensorBoard visualization (if enabled)
tensorboard --logdir=logs
```

## Inference

1. **Model Loading**
```python
from ml_models.core.models import get_model

model = get_model(model_config)
model.load_weights('weights/best_model.pt')
```

2. **Making Predictions**
```python
from ml_models.core.inference import InferencePipeline

inference = InferencePipeline(model, model_config)
predictions = inference.predict(image_path)
```

## Testing

1. **Running Tests**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_model.py

# Run with coverage
pytest --cov=ml_models
```

2. **Test Structure**
   - Unit tests for each component
   - Integration tests for pipelines
   - End-to-end tests for complete workflows

## Common Workflows

1. **Complete Training Pipeline**
```python
# 1. Load configurations
model_config = load_config('model_config')
training_config = load_config('training_config')
data_config = load_config('data_config')

# 2. Prepare data
data_manager = DataManager(data_config)
train_loader = data_manager.get_train_loader()
val_loader = data_manager.get_val_loader()

# 3. Initialize and train model
pipeline = TrainingPipeline(model_config)
results = pipeline.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=training_config['training']['num_epochs']
)

# 4. Save results
pipeline.save_results(results)
```

2. **Inference Pipeline**
```python
# 1. Load model
model = get_model(model_config)
model.load_weights('weights/best_model.pt')

# 2. Initialize inference
inference = InferencePipeline(model, model_config)

# 3. Process images
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
for path in image_paths:
    predictions = inference.predict(path)
    print(f"Predictions for {path}:", predictions)
```

## Troubleshooting

1. **Common Issues**
   - Configuration file not found: Check file paths and names
   - Data loading errors: Verify data directory structure
   - CUDA out of memory: Reduce batch size or image size
   - Training instability: Adjust learning rate or optimizer settings

2. **Debugging Tips**
   - Enable debug logging in config
   - Use smaller dataset for testing
   - Check data validation results
   - Monitor GPU memory usage

## Best Practices

1. **Code Organization**
   - Keep configurations separate from code
   - Use type hints and docstrings
   - Follow PEP 8 style guide
   - Write unit tests for new features

2. **Performance Optimization**
   - Use data caching when possible
   - Enable parallel processing for data loading
   - Monitor memory usage
   - Use appropriate batch sizes

3. **Version Control**
   - Keep configurations in version control
   - Document changes in commit messages
   - Use feature branches for development
   - Maintain clean git history

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [YOLO Implementation Guide](https://github.com/ultralytics/yolov5)
- [Data Loading Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Model Training Tips](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 