# Tests

This directory contains test files for the suture detection project.

## Structure

```
tests/
├── core/                   # Core functionality tests
│   ├── test_config.py     # Configuration tests
│   ├── test_inference.py  # Inference tests
│   └── test_training.py   # Training tests
├── data/                  # Data management tests
│   ├── test_loader.py    # Data loading tests
│   ├── test_processor.py # Processing tests
│   └── test_validator.py # Validation tests
├── models/               # Model tests
│   └── test_model.py    # Model architecture tests
└── utils/               # Utility tests
    ├── test_logging.py  # Logging tests
    └── test_transforms.py # Transform tests
```

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Module
```bash
pytest tests/core/
pytest tests/data/
pytest tests/models/
pytest tests/utils/
```

### Single Test File
```bash
pytest tests/core/test_config.py
```

### With Coverage
```bash
pytest --cov=ml_models tests/
```

## Writing Tests

### Example Test
```python
import pytest
from ml_models.core import Config

def test_config_loading():
    config = Config()
    config.load('tests/fixtures/test_config.yaml')
    assert config.model_type == 'yolo'
    assert config.num_classes == 2
```

### Test Categories

1. **Unit Tests**
   - Individual component testing
   - Mocked dependencies
   - Fast execution

2. **Integration Tests**
   - Component interaction testing
   - Real dependencies
   - End-to-end workflows

3. **Performance Tests**
   - Speed benchmarks
   - Memory usage
   - Resource utilization

## Test Data

Test data and fixtures are stored in `tests/fixtures/`:
- Sample images
- Configuration files
- Model weights
- Expected outputs 