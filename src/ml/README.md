# Suture Analysis ML Models

This repository contains machine learning models for suture analysis using YOLOv12.

## Project Structure

```
ml-models/
├── core/                   # Core functionality
│   ├── config/            # Configuration management
│   ├── inference/         # Inference code
│   └── training/          # Training pipeline
├── data/                  # Data management
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── annotations/      # Data annotations
├── models/               # Model definitions
├── utils/                # Utility functions
├── tests/               # Test files
├── weights/             # Model weights
└── notebooks/          # Jupyter notebooks
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from ml_models import TrainingPipeline, Config

# Load configuration
config = Config()

# Create training pipeline
pipeline = TrainingPipeline(config)

# Train model
results = pipeline.train()
```

### Inference

```python
from ml_models import SutureDetector

# Initialize detector
detector = SutureDetector(weights_path='weights/yolov12l.pt')

# Run inference
results = detector.detect('path/to/image.jpg')
```

## Components

### Core

- `config/`: Configuration management for model parameters and settings
- `inference/`: Code for running inference with trained models
- `training/`: Training pipeline implementation

### Data

- `manager.py`: Data management and coordination
- `loader.py`: Data loading and preprocessing
- `processor.py`: Data processing utilities
- `validator.py`: Data validation tools

### Models

- `model.py`: Model architecture definitions
- Base model class and YOLOv12 implementation

### Utils

- `logging.py`: Logging configuration
- `transforms.py`: Data transformations
- `validation.py`: Validation utilities
- `processing.py`: Processing utilities
- `performance_optimizer.py`: Model optimization tools
- `visualization.py`: Visualization utilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
