import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from datetime import datetime
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from ..utils.logging import setup_logging
from ..data.manager import DataManager
from ..models.model import get_model

logger = setup_logging(__name__)

class TrainingPipeline:
    """Training pipeline for suture detection model with MLflow integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config['device'])
        self.data_manager = DataManager(config)
        self.model = get_model(config).to(self.device)
        self.optimizer = self._create_optimizer()
        self.criterion = nn.MSELoss()
        
        # Initialize MLflow
        self.mlflow_client = MlflowClient()
        self.experiment_id = self._setup_mlflow()
        
    def _setup_mlflow(self) -> str:
        """Setup MLflow experiment.
        
        Returns:
            str: Experiment ID
        """
        try:
            experiment_name = self.config.get('experiment_name', 'suture_training')
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.mlflow_client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            return experiment_id
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer.
        
        Returns:
            torch.optim.Optimizer: Created optimizer
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
    def train(self, num_epochs: int = 100) -> Dict[str, Any]:
        """Train the model with MLflow tracking.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Dict[str, Any]: Training results
        """
        train_loader = self.data_manager.get_train_loader()
        val_loader = self.data_manager.get_val_loader()
        
        results = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log parameters
            mlflow.log_params(self.config)
            
            for epoch in range(num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                    batch = self.data_manager.process_train_batch(batch)
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    self.optimizer.zero_grad()
                    loss = self.model.train_step(batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    
                train_loss /= len(train_loader)
                results['train_loss'].append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = self.data_manager.process_val_batch(batch)
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        
                        loss = self.model.val_step(batch)
                        val_loss += loss.item()
                        
                val_loss /= len(val_loader)
                results['val_loss'].append(val_loss)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, step=epoch)
                
                # Save best model
                if val_loss < results['best_val_loss']:
                    results['best_val_loss'] = val_loss
                    results['best_epoch'] = epoch + 1
                    self.save_model(f'best_model.pt')
                    # Log model
                    mlflow.pytorch.log_model(
                        self.model,
                        "model",
                        registered_model_name="suture_detection"
                    )
                    
                logger.info(
                    f'Epoch {epoch + 1}/{num_epochs}: '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}'
                )
                
            # Log final results
            mlflow.log_metrics({
                'final_train_loss': results['train_loss'][-1],
                'final_val_loss': results['val_loss'][-1],
                'best_val_loss': results['best_val_loss']
            })
            
        return results
        
    def save_model(self, filename: str) -> None:
        """Save model weights and register with MLflow.
        
        Args:
            filename: Output filename
        """
        weights_dir = Path(self.config['weights_dir'])
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = weights_dir / filename
        torch.save(self.model.state_dict(), model_path)
        logger.info(f'Saved model weights to {model_path}')
        
    def load_model(self, filename: str) -> None:
        """Load model weights.
        
        Args:
            filename: Input filename
        """
        model_path = Path(self.config['weights_dir']) / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path))
        logger.info(f'Loaded model weights from {model_path}')
        
    def save_config(self) -> None:
        """Save training configuration."""
        config_path = Path(self.config['output_dir']) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f'Saved configuration to {config_path}')
        
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save training results.
        
        Args:
            results: Training results
        """
        results_path = Path(self.config['output_dir']) / 'results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
        logger.info(f'Saved results to {results_path}')
        
    def get_model_version(self) -> str:
        """Get the current model version from MLflow.
        
        Returns:
            str: Model version
        """
        try:
            model_name = "suture_detection"
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            if versions:
                return versions[0].version
            return "1.0"
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return "1.0" 