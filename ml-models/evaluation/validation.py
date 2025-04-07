from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

class ValidationPipeline:
    def __init__(self, model, n_splits=5):
        self.model = model
        self.kfold = KFold(n_splits=n_splits, shuffle=True)
        
    def cross_validate(self, X, y):
        metrics = []
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model.train(X_train, y_train)
            predictions = self.model.predict(X_val)
            
            fold_metrics = self._calculate_metrics(y_val, predictions)
            metrics.append(fold_metrics)
            
        return self._aggregate_metrics(metrics)
    
    def _calculate_metrics(self, y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        return {'precision': precision, 'recall': recall, 'f1': f1}