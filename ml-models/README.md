# Suture Analysis ML Models

A comprehensive machine learning pipeline for suture analysis, following CRISP-DM, MLOps, and ISO/IEC 23053 standards.

## Standards Compliance

### CRISP-DM
- Business Understanding: `docs/business/`
- Data Understanding: `docs/data/`
- Data Preparation: `preprocessing/`
- Modeling: `models/` and `training/`
- Evaluation: `evaluation/`
- Deployment: `deployment/`

### MLOps
- CI/CD: `.github/workflows/`
- Model Registry: `registry/`
- Monitoring: `monitoring/`
- Experiment Tracking: `experiments/`
- Data Versioning: `data/`

### ISO/IEC 23053
- Requirements: `docs/requirements/`
- Risk Assessment: `docs/risk/`
- Quality Assurance: `qa/`
- Validation: `validation/`
- Maintenance: `maintenance/`

## Project Structure

```
ml-models/
├── .github/
│   └── workflows/           # CI/CD pipelines
├── data/
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   └── versioned/          # Versioned datasets
├── docs/
│   ├── business/           # Business understanding
│   ├── data/              # Data documentation
│   ├── requirements/       # Requirements specification
│   └── risk/              # Risk assessment
├── preprocessing/          # Data preprocessing
├── models/                # Model definitions
├── training/              # Training pipeline
├── evaluation/            # Model evaluation
├── inference/             # Inference pipeline
├── deployment/            # Deployment configuration
├── monitoring/            # Model monitoring
├── registry/              # Model registry
├── experiments/           # Experiment tracking
├── qa/                    # Quality assurance
├── validation/            # Validation procedures
├── maintenance/           # Maintenance procedures
├── utils/                 # Utility functions
├── config.py              # Configuration management
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Layer Documentation

- [Preprocessing Layer](preprocessing/README.md)
- [Models Layer](models/README.md)
- [Training Layer](training/README.md)
- [Evaluation Layer](evaluation/README.md)
- [Inference Layer](inference/README.md)
- [Deployment Layer](deployment/README.md)
- [Monitoring Layer](monitoring/README.md)
- [Registry Layer](registry/README.md)
- [Experiments Layer](experiments/README.md)
- [QA Layer](qa/README.md)
- [Validation Layer](validation/README.md)
- [Maintenance Layer](maintenance/README.md)
- [Utilities Layer](utils/README.md)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from ml_models import (
    SutureProcessor,
    get_model,
    SutureTrainer,
    ModelEvaluator,
    InferencePipeline,
    Config
)

# Initialize configuration
config = Config()

# 1. Preprocessing
processor = SutureProcessor(**config.get_preprocessing_config())
features = processor.process_directory("raw_images", "processed_data")

# 2. Training
model = get_model("yolo", **config.get_model_config())
trainer = SutureTrainer(model_type="yolo")
trainer.train(train_loader, val_loader, **config.get_training_config())

# 3. Evaluation
evaluator = ModelEvaluator(model, **config.get_evaluation_config())
metrics = evaluator.cross_validate(train_loader, val_loader)

# 4. Inference
pipeline = InferencePipeline(
    model_type="yolo",
    model_path=config.get_model_config()['weights_path']
)
pipeline.process_directory("test_images", "predictions", ["suture", "knot"])
```

## Configuration

The project uses a centralized configuration system. See [config.py](config.py) for details.

Example configuration file (`config.yaml`):
```yaml
model:
  type: yolo
  weights_path: weights/yolo12m_saved.pt
  num_classes: 2
  input_size: [640, 640]

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

preprocessing:
  image_size: [640, 640]
  blur_kernel: [5, 5]
  canny_thresholds: [50, 150]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
