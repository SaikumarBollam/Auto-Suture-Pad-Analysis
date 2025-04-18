# Core Components

This directory contains the core functionality for suture detection and analysis.

## Structure

- `config/`: Configuration management
  - `config.py`: Main configuration class and utilities
  - Handles model, training, and data parameters

- `inference/`: Inference pipeline
  - `infer.py`: SutureDetector class for model inference
  - Handles image preprocessing, detection, and visualization

- `training/`: Training pipeline
  - `train.py`: TrainingPipeline class for model training
  - Manages training loop, validation, and checkpointing

## Usage

### Configuration

```python
from ml_models.core import Config

config = Config()
config.load('path/to/config.yaml')
```

### Training

```python
from ml_models.core import TrainingPipeline

pipeline = TrainingPipeline(config)
results = pipeline.train()
```

### Inference

```python
from ml_models.core import SutureDetector

detector = SutureDetector(weights_path='weights/model.pt')
results = detector.detect('image.jpg')
``` 