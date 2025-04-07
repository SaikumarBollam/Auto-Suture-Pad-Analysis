# Evaluation Layer

This layer handles model evaluation and performance analysis.

## Components

### ModelEvaluator
Main class for evaluating model performance.

#### Features
- Cross-validation
- Metrics calculation
- Confusion matrix visualization
- Performance analysis

## Usage

```python
from ml_models.evaluation import ModelEvaluator
from ml_models.config import Config

# Initialize configuration
config = Config()

# Initialize evaluator
evaluator = ModelEvaluator(
    model=model,
    n_splits=config.get_evaluation_config()['n_splits']
)

# Perform cross-validation
metrics = evaluator.cross_validate(train_loader, val_loader)

# Plot confusion matrix
evaluator.plot_confusion_matrix(
    cm=metrics['confusion_matrix'],
    class_names=["suture", "knot"],
    save_path="confusion_matrix.png"
)
```

## Evaluation Metrics

### Classification Metrics
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Cross-Validation
- K-fold cross-validation
- Stratified sampling
- Metrics aggregation

### Visualization
- Confusion matrix heatmap
- Performance curves
- Metric distributions

## Configuration

Key evaluation parameters can be configured through the Config class:
- Number of cross-validation splits
- Metrics to calculate
- Visualization parameters
- Performance thresholds

See `config.py` for all available parameters. 