# Registry Layer

This layer handles model versioning, storage, and management following MLOps best practices.

## Components

### Model Registry
- Model versioning
- Model metadata
- Model artifacts
- Model lineage

### Experiment Registry
- Experiment tracking
- Hyperparameter logging
- Metric tracking
- Artifact storage

### Data Registry
- Dataset versioning
- Data lineage
- Data metadata
- Data artifacts

## Usage

### Model Registration
```python
from ml_models.registry import ModelRegistry
from ml_models.config import Config

# Initialize configuration
config = Config()

# Initialize registry
registry = ModelRegistry(
    **config.get_registry_config()
)

# Register model
model_info = registry.register_model(
    model=model,
    name="suture_detector",
    version="1.0.0",
    metadata={
        "description": "YOLO-based suture detector",
        "metrics": metrics,
        "hyperparameters": hyperparams
    }
)

# Get model
model = registry.get_model("suture_detector", "1.0.0")
```

### Experiment Tracking
```python
from ml_models.registry import ExperimentRegistry

# Initialize registry
experiment_registry = ExperimentRegistry(
    **config.get_experiment_registry_config()
)

# Start experiment
experiment = experiment_registry.start_experiment(
    name="suture_detection_experiment",
    description="Testing different YOLO architectures"
)

# Log metrics
experiment.log_metrics(metrics)

# Log hyperparameters
experiment.log_params(hyperparams)

# Log artifacts
experiment.log_artifact("model.pt")
```

## Registry Process

1. **Model Registration**
   - Version control
   - Metadata management
   - Artifact storage
   - Lineage tracking

2. **Experiment Tracking**
   - Parameter logging
   - Metric tracking
   - Artifact storage
   - Result analysis

3. **Data Management**
   - Version control
   - Metadata management
   - Lineage tracking
   - Access control

4. **Registry Maintenance**
   - Cleanup procedures
   - Backup strategies
   - Access management
   - Audit logging

## Registry Features

### Model Features
- Version control
- Metadata management
- Artifact storage
- Lineage tracking
- Access control
- Tagging system

### Experiment Features
- Parameter tracking
- Metric logging
- Artifact storage
- Result visualization
- Comparison tools
- Collaboration features

### Data Features
- Version control
- Metadata management
- Lineage tracking
- Access control
- Quality metrics
- Usage tracking

## Configuration

Key registry parameters can be configured through the Config class:
- Storage settings
- Access control
- Versioning rules
- Cleanup policies

Example configuration:
```yaml
registry:
  model:
    storage_path: "models/"
    version_format: "major.minor.patch"
    cleanup_policy:
      max_versions: 10
      retention_days: 90
  experiment:
    storage_path: "experiments/"
    tracking_interval: 60
    artifact_storage: "artifacts/"
  data:
    storage_path: "data/"
    version_format: "YYYY.MM.DD"
    access_control: true
```

## Best Practices

1. **Version Control**
   - Semantic versioning
   - Clear changelogs
   - Version tagging
   - Release notes

2. **Metadata Management**
   - Comprehensive metadata
   - Standardized formats
   - Regular updates
   - Quality checks

3. **Access Control**
   - Role-based access
   - Audit logging
   - Security policies
   - Compliance checks

4. **Maintenance**
   - Regular cleanup
   - Backup procedures
   - Performance optimization
   - Security updates

## Integration

The registry integrates with:
1. Training pipeline
   - Model versioning
   - Experiment tracking
   - Metric logging

2. Deployment pipeline
   - Model serving
   - Version management
   - Rollback procedures

3. Monitoring system
   - Performance tracking
   - Usage statistics
   - Health monitoring

## Troubleshooting

Common registry issues and solutions:
1. Storage issues
   - Cleanup old versions
   - Optimize storage
   - Implement compression
   - Scale storage

2. Access issues
   - Check permissions
   - Verify credentials
   - Update policies
   - Audit access

3. Version issues
   - Check version format
   - Verify metadata
   - Update changelog
   - Fix conflicts

See `config.py` for all available parameters. 