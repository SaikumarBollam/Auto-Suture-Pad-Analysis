"""Model management for ML pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam, SGD
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, OneCycleLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from ..utils.logging import setup_logging
from ..config.config import ModelConfig, TrainingConfig
import numpy as np

logger = setup_logging(__name__)

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and LeakyReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SutureModel(nn.Module):
    """YOLO-based model for suture detection."""
    
    def __init__(self, config: ModelConfig):
        """Initialize model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Define backbone
        self.backbone = self._create_backbone()
        
        # Define detection head
        self.detection_head = self._create_detection_head()
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self) -> nn.Module:
        """Create backbone network."""
        layers = []
        in_channels = 3  # RGB input
        
        # Define backbone architecture based on config
        if self.config.backbone == "darknet53":
            # Darknet53 architecture
            channels = [32, 64, 128, 256, 512, 1024]
            for out_channels in channels:
                layers.append(ConvBlock(in_channels, out_channels, stride=2))
                in_channels = out_channels
                layers.append(ConvBlock(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _create_detection_head(self) -> nn.Module:
        """Create detection head."""
        # Define detection head architecture
        layers = []
        in_channels = 1024  # Output channels from backbone
        
        # Add detection layers
        for _ in range(3):  # 3 detection layers
            layers.append(ConvBlock(in_channels, in_channels // 2))
            layers.append(ConvBlock(in_channels // 2, in_channels))
            layers.append(nn.Conv2d(in_channels, (5 + self.config.num_classes) * 3, 1))  # 3 anchors per scale
            in_channels = in_channels // 2
        
        return nn.ModuleList(layers)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Dictionary containing model outputs:
            - "detections": List of detection tensors at different scales
            - "features": List of feature maps at different scales
        """
        # Backbone forward pass
        features = self.backbone(x)
        
        # Detection head forward pass
        detections = []
        current_features = features
        
        for layer in self.detection_head:
            if isinstance(layer, nn.Conv2d):
                # Detection layer
                detections.append(layer(current_features))
            else:
                # Feature processing layer
                current_features = layer(current_features)
        
        return {
            "detections": detections,
            "features": [features]  # Add more feature maps if needed
        }

class ModelManager:
    """Manages model training and evaluation."""
    
    def __init__(self, 
                 model_config: ModelConfig,
                 training_config: TrainingConfig):
        """Initialize model manager.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        
        # Initialize model
        self.model = SutureModel(model_config)
        self.model.to(model_config.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize early stopping
        self.early_stopping = self._create_early_stopping()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.training_config.mixed_precision else None
        
        # Initialize best metrics
        self.best_metrics = {
            "val_loss": float('inf'),
            "mAP": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on config."""
        if self.training_config.optimizer == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                eps=1e-7  # Add epsilon for numerical stability
            )
        elif self.training_config.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                momentum=self.training_config.momentum,
                weight_decay=self.training_config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> _LRScheduler:
        """Create learning rate scheduler based on config."""
        # Create warmup scheduler
        warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(1.0, epoch / self.training_config.warmup_epochs)
        )
        
        # Create main scheduler
        if self.training_config.scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.epochs - self.training_config.warmup_epochs,
                eta_min=self.training_config.min_learning_rate
            )
        elif self.training_config.scheduler == "onecycle":
            main_scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.training_config.learning_rate,
                epochs=self.training_config.epochs - self.training_config.warmup_epochs,
                steps_per_epoch=self.training_config.steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.training_config.scheduler}")
        
        # Combine schedulers
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.training_config.warmup_epochs]
        )
    
    def _create_early_stopping(self) -> Any:
        """Create early stopping based on config."""
        class EarlyStopping:
            def __init__(self, patience: int, min_delta: float):
                self.patience = patience
                self.min_delta = min_delta
                self.counter = 0
                self.best_loss = float('inf')
                self.early_stop = False
            
            def __call__(self, val_loss: float) -> bool:
                if val_loss < self.best_loss - self.min_delta:
                    self.best_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                return self.early_stop
        
        return EarlyStopping(
            patience=self.training_config.early_stopping["patience"],
            min_delta=self.training_config.early_stopping["min_delta"]
        )
    
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        metrics = {}
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.model_config.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.training_config.mixed_precision):
                outputs = self.model(batch["images"])
                loss = self._compute_loss(outputs, batch["labels"])
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            if self.training_config.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                self.optimizer.step()
            
            # Update metrics
            metrics = self._update_metrics(metrics, outputs, batch["labels"])
        
        return metrics
    
    def validate(self, 
                val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        metrics = {}
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.model_config.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch["images"])
                
                # Update metrics
                metrics = self._update_metrics(metrics, outputs, batch["labels"])
        
        return metrics
    
    def _compute_loss(self, 
                     outputs: Dict[str, torch.Tensor],
                     labels: torch.Tensor) -> torch.Tensor:
        """Compute YOLO loss.
        
        Args:
            outputs: Model outputs containing detections at different scales
            labels: Ground truth labels of shape (B, N, 5) where N is max objects
                   and 5 represents (x, y, w, h, class)
            
        Returns:
            Total loss tensor
        """
        # Initialize loss components
        obj_loss = 0.0
        noobj_loss = 0.0
        box_loss = 0.0
        cls_loss = 0.0
        
        # Process each scale
        for scale_idx, detections in enumerate(outputs["detections"]):
            # Reshape detections to (B, H, W, A, 5 + C)
            B, _, H, W = detections.shape
            A = 3  # Number of anchors
            C = self.config.num_classes
            detections = detections.view(B, A, 5 + C, H, W).permute(0, 3, 4, 1, 2)
            
            # Get predictions
            pred_xy = torch.sigmoid(detections[..., :2])
            pred_wh = torch.exp(detections[..., 2:4])
            pred_obj = torch.sigmoid(detections[..., 4])
            pred_cls = torch.sigmoid(detections[..., 5:])
            
            # Compute IoU with anchors
            anchors = torch.tensor(self.config.anchors[scale_idx]).to(detections.device)
            iou = self._compute_iou(pred_xy, pred_wh, anchors)
            
            # Get best anchor for each ground truth
            best_anchor = torch.argmax(iou, dim=-1)
            obj_mask = torch.zeros_like(pred_obj)
            obj_mask[best_anchor] = 1
            
            # Compute losses
            obj_loss += self._compute_obj_loss(pred_obj, obj_mask)
            noobj_loss += self._compute_noobj_loss(pred_obj, obj_mask)
            box_loss += self._compute_box_loss(pred_xy, pred_wh, labels, obj_mask)
            cls_loss += self._compute_cls_loss(pred_cls, labels[..., 4:], obj_mask)
        
        # Combine losses
        total_loss = (
            self.training_config.loss_weights["obj"] * obj_loss +
            self.training_config.loss_weights["noobj"] * noobj_loss +
            self.training_config.loss_weights["box"] * box_loss +
            self.training_config.loss_weights["cls"] * cls_loss
        )
        
        return total_loss
    
    def _compute_iou(self, 
                    pred_xy: torch.Tensor,
                    pred_wh: torch.Tensor,
                    anchors: torch.Tensor) -> torch.Tensor:
        """Compute IoU between predictions and anchors."""
        # Convert to box format (x1, y1, x2, y2)
        pred_boxes = torch.cat([
            pred_xy - pred_wh / 2,
            pred_xy + pred_wh / 2
        ], dim=-1)
        
        anchor_boxes = torch.cat([
            torch.zeros_like(anchors),
            anchors
        ], dim=-1)
        
        # Compute IoU
        inter_area = self._compute_intersection(pred_boxes, anchor_boxes)
        union_area = self._compute_union(pred_boxes, anchor_boxes)
        
        return inter_area / (union_area + 1e-6)
    
    def _compute_intersection(self, 
                            box1: torch.Tensor,
                            box2: torch.Tensor) -> torch.Tensor:
        """Compute intersection area between boxes."""
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        
        return torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    def _compute_union(self, 
                      box1: torch.Tensor,
                      box2: torch.Tensor) -> torch.Tensor:
        """Compute union area between boxes."""
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        
        return area1 + area2 - self._compute_intersection(box1, box2)
    
    def _compute_obj_loss(self,
                         pred_obj: torch.Tensor,
                         obj_mask: torch.Tensor) -> torch.Tensor:
        """Compute objectness loss."""
        return nn.BCELoss()(pred_obj * obj_mask, obj_mask)
    
    def _compute_noobj_loss(self,
                           pred_obj: torch.Tensor,
                           obj_mask: torch.Tensor) -> torch.Tensor:
        """Compute no-objectness loss."""
        noobj_mask = 1 - obj_mask
        return nn.BCELoss()(pred_obj * noobj_mask, torch.zeros_like(pred_obj) * noobj_mask)
    
    def _compute_box_loss(self,
                         pred_xy: torch.Tensor,
                         pred_wh: torch.Tensor,
                         labels: torch.Tensor,
                         obj_mask: torch.Tensor) -> torch.Tensor:
        """Compute box regression loss."""
        # Convert labels to grid coordinates
        label_xy = labels[..., :2]
        label_wh = labels[..., 2:4]
        
        # Compute MSE loss
        xy_loss = nn.MSELoss()(pred_xy * obj_mask, label_xy * obj_mask)
        wh_loss = nn.MSELoss()(pred_wh * obj_mask, label_wh * obj_mask)
        
        return xy_loss + wh_loss
    
    def _compute_cls_loss(self,
                         pred_cls: torch.Tensor,
                         labels: torch.Tensor,
                         obj_mask: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        return nn.BCELoss()(pred_cls * obj_mask, labels * obj_mask)
    
    def _update_metrics(self,
                       metrics: Dict[str, float],
                       outputs: Dict[str, torch.Tensor],
                       labels: torch.Tensor) -> Dict[str, float]:
        """Update metrics dictionary.
        
        Args:
            metrics: Current metrics dictionary
            outputs: Model outputs
            labels: Ground truth labels
            
        Returns:
            Updated metrics dictionary
        """
        # Initialize metrics if empty
        if not metrics:
            metrics = {
                "obj_loss": 0.0,
                "noobj_loss": 0.0,
                "box_loss": 0.0,
                "cls_loss": 0.0,
                "total_loss": 0.0,
                "mAP": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Compute losses
        obj_loss = 0.0
        noobj_loss = 0.0
        box_loss = 0.0
        cls_loss = 0.0
        
        for detections in outputs["detections"]:
            # Convert detections to predictions
            pred_boxes, pred_scores, pred_classes = self._convert_detections(detections)
            
            # Convert labels to ground truth boxes
            gt_boxes = labels[..., :4]
            gt_classes = labels[..., 4]
            
            # Compute metrics
            metrics["mAP"] += self._compute_map(pred_boxes, pred_scores, pred_classes,
                                              gt_boxes, gt_classes)
            metrics["precision"] += self._compute_precision(pred_boxes, pred_scores, pred_classes,
                                                         gt_boxes, gt_classes)
            metrics["recall"] += self._compute_recall(pred_boxes, pred_scores, pred_classes,
                                                    gt_boxes, gt_classes)
        
        # Average metrics over scales
        num_scales = len(outputs["detections"])
        for key in ["mAP", "precision", "recall"]:
            metrics[key] /= num_scales
        
        return metrics
    
    def _convert_detections(self, detections: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw detections to boxes, scores, and classes."""
        B, _, H, W = detections.shape
        A = 3  # Number of anchors
        C = self.config.num_classes
        
        # Reshape detections
        detections = detections.view(B, A, 5 + C, H, W).permute(0, 3, 4, 1, 2)
        
        # Get predictions
        pred_xy = torch.sigmoid(detections[..., :2])
        pred_wh = torch.exp(detections[..., 2:4])
        pred_obj = torch.sigmoid(detections[..., 4])
        pred_cls = torch.sigmoid(detections[..., 5:])
        
        # Convert to boxes
        boxes = torch.cat([pred_xy, pred_wh], dim=-1)
        
        # Get scores and classes
        scores = pred_obj * torch.max(pred_cls, dim=-1)[0]
        classes = torch.argmax(pred_cls, dim=-1)
        
        return boxes, scores, classes
    
    def _compute_map(self,
                    pred_boxes: torch.Tensor,
                    pred_scores: torch.Tensor,
                    pred_classes: torch.Tensor,
                    gt_boxes: torch.Tensor,
                    gt_classes: torch.Tensor) -> float:
        """Compute mean Average Precision."""
        # Convert tensors to numpy
        pred_boxes = pred_boxes.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()
        gt_boxes = gt_boxes.cpu().numpy()
        gt_classes = gt_classes.cpu().numpy()
        
        # Initialize AP for each class
        aps = []
        
        # Compute AP for each class
        for class_id in range(self.config.num_classes):
            # Get predictions and ground truth for this class
            class_pred_boxes = pred_boxes[pred_classes == class_id]
            class_pred_scores = pred_scores[pred_classes == class_id]
            class_gt_boxes = gt_boxes[gt_classes == class_id]
            
            if len(class_pred_boxes) == 0 or len(class_gt_boxes) == 0:
                aps.append(0.0)
                continue
            
            # Sort predictions by score
            sorted_indices = np.argsort(class_pred_scores)[::-1]
            class_pred_boxes = class_pred_boxes[sorted_indices]
            class_pred_scores = class_pred_scores[sorted_indices]
            
            # Compute precision-recall curve
            precision, recall = self._compute_precision_recall(
                class_pred_boxes, class_pred_scores, class_gt_boxes
            )
            
            # Compute AP using 11-point interpolation
            ap = self._compute_ap(precision, recall)
            aps.append(ap)
        
        # Return mean AP
        return np.mean(aps)
    
    def _compute_precision_recall(self,
                                pred_boxes: np.ndarray,
                                pred_scores: np.ndarray,
                                gt_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute precision-recall curve."""
        # Initialize arrays
        precision = np.zeros(len(pred_boxes))
        recall = np.zeros(len(pred_boxes))
        
        # Compute IoU with ground truth
        ious = self._compute_iou_numpy(pred_boxes, gt_boxes)
        
        # Sort by IoU
        max_ious = np.max(ious, axis=1)
        sorted_indices = np.argsort(max_ious)[::-1]
        
        # Compute precision and recall
        tp = 0
        fp = 0
        total_gt = len(gt_boxes)
        
        for i, idx in enumerate(sorted_indices):
            if max_ious[idx] >= self.training_config.iou_threshold:
                tp += 1
            else:
                fp += 1
            
            precision[i] = tp / (tp + fp)
            recall[i] = tp / total_gt
        
        return precision, recall
    
    def _compute_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Compute AP using 11-point interpolation."""
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        return ap
    
    def _compute_iou_numpy(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes in numpy."""
        # Convert to (x1, y1, x2, y2) format
        boxes1 = np.concatenate([
            boxes1[..., :2] - boxes1[..., 2:] / 2,
            boxes1[..., :2] + boxes1[..., 2:] / 2
        ], axis=-1)
        
        boxes2 = np.concatenate([
            boxes2[..., :2] - boxes2[..., 2:] / 2,
            boxes2[..., :2] + boxes2[..., 2:] / 2
        ], axis=-1)
        
        # Compute intersection
        x1 = np.maximum(boxes1[..., 0:1], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1:2], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2:3], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3:4], boxes2[..., 3])
        
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        
        # Compute union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1[..., np.newaxis] + area2 - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def _compute_precision(self,
                          pred_boxes: torch.Tensor,
                          pred_scores: torch.Tensor,
                          pred_classes: torch.Tensor,
                          gt_boxes: torch.Tensor,
                          gt_classes: torch.Tensor) -> float:
        """Compute precision."""
        # Convert tensors to numpy
        pred_boxes = pred_boxes.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()
        gt_boxes = gt_boxes.cpu().numpy()
        gt_classes = gt_classes.cpu().numpy()
        
        # Initialize counters
        tp = 0
        fp = 0
        
        # Compute IoU with ground truth
        ious = self._compute_iou_numpy(pred_boxes, gt_boxes)
        
        # Count true positives and false positives
        for i in range(len(pred_boxes)):
            max_iou = np.max(ious[i])
            if max_iou >= self.training_config.iou_threshold:
                tp += 1
            else:
                fp += 1
        
        # Compute precision
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    def _compute_recall(self,
                       pred_boxes: torch.Tensor,
                       pred_scores: torch.Tensor,
                       pred_classes: torch.Tensor,
                       gt_boxes: torch.Tensor,
                       gt_classes: torch.Tensor) -> float:
        """Compute recall."""
        # Convert tensors to numpy
        pred_boxes = pred_boxes.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()
        gt_boxes = gt_boxes.cpu().numpy()
        gt_classes = gt_classes.cpu().numpy()
        
        # Initialize counters
        tp = 0
        total_gt = len(gt_boxes)
        
        if total_gt == 0:
            return 0.0
        
        # Compute IoU with ground truth
        ious = self._compute_iou_numpy(pred_boxes, gt_boxes)
        
        # Count true positives
        for i in range(len(gt_boxes)):
            max_iou = np.max(ious[:, i])
            if max_iou >= self.training_config.iou_threshold:
                tp += 1
        
        # Compute recall
        return tp / total_gt
    
    def save_checkpoint(self, 
                       path: Path,
                       is_best: bool = False,
                       metrics: Optional[Dict[str, float]] = None) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
            metrics: Current validation metrics
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": {
                "model": self.model_config.__dict__,
                "training": self.training_config.__dict__
            },
            "metrics": metrics
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        # Save best model
        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def export_model(self, path: Path) -> None:
        """Export model for inference.
        
        Args:
            path: Path to save exported model
        """
        # Create export directory
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "model": self.model_config.__dict__,
                "training": self.training_config.__dict__
            }
        }, path)
        
        logger.info(f"Exported model to {path}")
    
    def update_best_metrics(self, metrics: Dict[str, float]) -> bool:
        """Update best metrics and return whether current metrics are best.
        
        Args:
            metrics: Current validation metrics
            
        Returns:
            Whether current metrics are best
        """
        is_best = False
        
        # Update best metrics
        if metrics["val_loss"] < self.best_metrics["val_loss"]:
            self.best_metrics["val_loss"] = metrics["val_loss"]
            is_best = True
        
        if metrics["mAP"] > self.best_metrics["mAP"]:
            self.best_metrics["mAP"] = metrics["mAP"]
            is_best = True
        
        if metrics["precision"] > self.best_metrics["precision"]:
            self.best_metrics["precision"] = metrics["precision"]
            is_best = True
        
        if metrics["recall"] > self.best_metrics["recall"]:
            self.best_metrics["recall"] = metrics["recall"]
            is_best = True
        
        return is_best 