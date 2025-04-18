# Configuration Files Documentation

This directory contains configuration files for different aspects of the machine learning pipeline. Each file serves a specific purpose and can be customized according to your needs.

## Configuration Files Overview

### 1. model_config.yaml
This file contains the core model and training configuration settings.

**Key Sections:**
- `model`: Model architecture settings (input size, number of classes, anchors)
- `training`: Training hyperparameters (batch size, learning rate, epochs)
- `data`: Data path configurations
- `output`: Directory settings for weights and outputs

**Usage Example:**
```python
from ml_models.core.training import TrainingPipeline
import yaml

with open('config/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
pipeline = TrainingPipeline(config)
```

### 2. training_config.yaml
Contains training pipeline specific settings and augmentation configurations.

**Key Sections:**
- `pipeline`: Training pipeline settings (checkpointing, early stopping)
- `augmentation`: Data augmentation settings for training and validation
- `logging`: Logging configuration

**Usage Example:**
```python
from ml_models.core.training import TrainingPipeline
import yaml

with open('config/training_config.yaml', 'r') as f:
    training_config = yaml.safe_load(f)
# Use with model_config.yaml for complete setup
```

### 3. data_config.yaml
Configuration for dataset handling and preprocessing.

**Key Sections:**
- `dataset`: Dataset specifications (format, classes)
- `loader`: Data loading parameters
- `preprocessing`: Image preprocessing settings
- `paths`: Dataset directory structure

**Usage Example:**
```python
from ml_models.data.manager import DataManager
import yaml

with open('config/data_config.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
data_manager = DataManager(data_config)
```

### 4. read_config.yaml
Settings for file reading and data validation.

**Key Sections:**
- `file_reading`: Supported formats and reading modes
- `validation`: Data validation rules
- `error_handling`: Error management settings
- `caching`: Caching configuration
- `parallel`: Parallel processing settings

**Usage Example:**
```python
from ml_models.data.reader import DataReader
import yaml

with open('config/read_config.yaml', 'r') as f:
    read_config = yaml.safe_load(f)
reader = DataReader(read_config)
```

## Configuration Management

### Loading Configurations
To load and use these configurations in your code:

```python
import yaml
from pathlib import Path

def load_config(config_name):
    config_path = Path('config') / f'{config_name}.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load all configurations
model_config = load_config('model_config')
training_config = load_config('training_config')
data_config = load_config('data_config')
read_config = load_config('read_config')
```

### Modifying Configurations
1. Open the desired configuration file
2. Modify the values according to your needs
3. Save the file
4. The changes will be automatically picked up by the pipeline

### Best Practices
1. Always make a backup of your configuration files before making changes
2. Document any custom changes you make
3. Test the configuration changes with a small subset of data first
4. Keep the configuration files in version control

## Directory Structure
```
ml-models/
└── config/
    ├── model_config.yaml    # Model and training settings
    ├── training_config.yaml # Training pipeline settings
    ├── data_config.yaml     # Dataset configuration
    ├── read_config.yaml     # File reading settings
    └── README.md           # This file
```

For more detailed information about each configuration parameter, refer to the comments in the respective YAML files. 