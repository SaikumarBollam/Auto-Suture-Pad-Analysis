import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation class combining validation and evaluation functionality."""
    
    def __init__(self, model, n_splits: int = 5):
        """Initialize the model evaluator.
        
        Args:
            model: Model to evaluate
            n_splits (int): Number of splits for cross-validation
        """
        self.model = model
        self.kfold = KFold(n_splits=n_splits, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def cross_validate(self, train_loader, val_loader) -> Dict[str, float]:
        """Perform cross-validation on the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dict[str, float]: Aggregated metrics across folds
        """
        metrics = []
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(train_loader.dataset)):
            # Train model
            self.model.train()
            self._train_epoch(train_loader)
            
            # Evaluate on validation set
            fold_metrics = self.evaluate(val_loader)
            metrics.append(fold_metrics)
            
            print(f"Fold {fold + 1} metrics: {fold_metrics}")
        
        return self._aggregate_metrics(metrics)
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                predictions.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_path: Optional[str] = None) -> None:
        """Plot and optionally save confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            class_names (List[str]): Names of classes
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def _train_epoch(self, train_loader) -> None:
        """Train model for one epoch.
        
        Args:
            train_loader: Training data loader
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for batch in train_loader:
            inputs = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    def _aggregate_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across folds.
        
        Args:
            metrics (List[Dict[str, float]]): List of metrics from each fold
            
        Returns:
            Dict[str, float]: Aggregated metrics
        """
        aggregated = {}
        for metric in metrics[0].keys():
            if metric != 'confusion_matrix':
                values = [m[metric] for m in metrics]
                aggregated[metric] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated