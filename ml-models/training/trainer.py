import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import optuna
from pathlib import Path
import mlflow
from datetime import datetime

from ..models.model import get_model
from ..utils.visualization import plot_training_metrics

class SutureTrainer:
    def __init__(self,
                 model_type: str = "yolo",
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the suture trainer.
        
        Args:
            model_type (str): Type of model to use ("yolo" or "cnn")
            model_kwargs (Dict[str, Any], optional): Additional model parameters
            device (str): Device to use for training
        """
        self.device = device
        self.model = get_model(model_type, **(model_kwargs or {}))
        self.model.to(device)
        
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("suture_analysis")
        
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int = 100,
             learning_rate: float = 0.001,
             weight_decay: float = 0.0001,
             save_dir: str = "checkpoints") -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            save_dir (str): Directory to save checkpoints
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_type": type(self.model).__name__,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": train_loader.batch_size
            })
            
            # Training loop
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                
                train_loss = train_loss / len(train_loader)
                train_acc = 100. * train_correct / train_total
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100. * val_correct / val_total
                
                # Update metrics
                metrics["train_loss"].append(train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["train_acc"].append(train_acc)
                metrics["val_acc"].append(val_acc)
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc
                }, step=epoch)
                
                # Print progress
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    checkpoint = {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc
                    }
                    torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch+1}.pt")
            
            # Save final model
            torch.save(self.model.state_dict(), save_dir / "final_model.pt")
            
            # Plot and save training metrics
            plot_training_metrics(metrics, save_dir / "training_metrics.png")
            
            return metrics
    
    def tune_hyperparameters(self,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           n_trials: int = 100) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            n_trials (int): Number of trials for hyperparameter optimization
            
        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            
            # Create new model with suggested hyperparameters
            model = get_model(type(self.model).__name__.lower())
            model.to(self.device)
            
            # Train model
            metrics = self.train(
                train_loader,
                val_loader,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size
            )
            
            # Return validation accuracy as objective
            return max(metrics["val_acc"])
        
        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Return best hyperparameters
        return study.best_params