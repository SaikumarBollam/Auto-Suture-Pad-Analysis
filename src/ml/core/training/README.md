# Training Layer

This layer handles model training and hyperparameter optimization.

## Components

### SutureTrainer
Main class for training suture analysis models.

#### Features
- Training loop management
- Validation
- Checkpointing
- Metrics tracking
- Hyperparameter tuning

## Usage

```python
from ml_models.training import SutureTrainer
from ml_models.config import Config

# Initialize configuration
config = Config()

# Initialize trainer
trainer = SutureTrainer(
    model_type="yolo",
    model_kwargs=config.get_model_config()
)

# Train model
metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    **config.get_training_config()
)

# Tune hyperparameters
best_params = trainer.tune_hyperparameters(
    train_loader=train_loader,
    val_loader=val_loader,
    n_trials=100
)
```

## Training Process

1. **Initialization**
   - Model setup
   - Optimizer configuration
   - Loss function selection

2. **Training Loop**
   - Forward pass
   - Loss computation
   - Backward pass
   - Parameter update
   - Validation
   - Metrics tracking

3. **Checkpointing**
   - Model state saving
   - Optimizer state saving
   - Metrics logging

4. **Hyperparameter Tuning**
   - Learning rate optimization
   - Weight decay optimization
   - Batch size optimization

## Configuration

Key training parameters can be configured through the Config class:
- Number of epochs
- Batch size
- Learning rate
- Weight decay
- Device selection
- Checkpoint frequency

See `config.py` for all available parameters. 