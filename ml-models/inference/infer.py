import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from ml_models.models.model import get_model
from ml_models.config import Config

class SutureDetector:
    """Suture detection and classification using YOLOv12."""
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize suture detector.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to model weights
            device: Device to use for inference
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Override config with arguments
        if model_path is not None:
            self.config.config['model']['weights_path'] = model_path
        if device is not None:
            self.config.config['deployment']['device'] = device
            
        # Get model configuration
        model_config = self.config.get_model_config()
        
        # Initialize model
        try:
            self.model = get_model(
                model_type=model_config['type'],
                model_size=model_config['model_size'],
                num_classes=model_config['num_classes'],
                pretrained=model_config['pretrained']
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        
        # Load weights if provided
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights from {model_path}: {str(e)}")
            
        # Set device
        self.device = torch.device(self.config.get_deployment_config()['device'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get validation configuration
        self.val_config = self.config.get_validation_config()
        
        # Get input size from config
        self.input_size = model_config['input_size']
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Validate image size
        if image.shape[0] < 32 or image.shape[1] < 32:
            raise ValueError(f"Image size {image.shape[:2]} is too small. Minimum size is 32x32")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to input size
        pad_w = self.input_size[0] - new_w
        pad_h = self.input_size[1] - new_h
        image = cv2.copyMakeBorder(
            image,
            0, pad_h,
            0, pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
        
    def postprocess(
        self,
        results: List[torch.Tensor],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Postprocess model predictions.
        
        Args:
            results: Raw model predictions
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List[Dict[str, Any]]: Processed predictions
        """
        conf_threshold = conf_threshold or self.val_config['conf_threshold']
        iou_threshold = iou_threshold or self.val_config['iou_threshold']
        
        processed_results = []
        for result in results:
            # YOLOv12 outputs are in format [batch, num_detections, 6]
            # where 6 is [x1, y1, x2, y2, conf, class]
            if len(result) == 0:
                processed_results.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0)
                })
                continue
                
            # Filter by confidence
            mask = result[:, 4] > conf_threshold
            result = result[mask]
            
            if len(result) == 0:
                processed_results.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0)
                })
                continue
                
            # Perform NMS
            boxes = result[:, :4]
            scores = result[:, 4]
            labels = result[:, 5]
            
            keep = torchvision.ops.nms(boxes, scores, iou_threshold)
            
            processed_results.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep]
            })
            
        return processed_results
        
    def detect(
        self,
        image: Union[np.ndarray, str, List[str]],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """Detect sutures in image(s).
        
        Args:
            image: Input image(s)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            batch_size: Batch size for processing multiple images
            
        Returns:
            List[Dict[str, Any]]: Detection results
        """
        # Handle different input types
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image: {image}")
            image = self.preprocess(image)
        elif isinstance(image, np.ndarray):
            image = self.preprocess(image)
        elif isinstance(image, list):
            # Process images in batches
            results = []
            for i in range(0, len(image), batch_size):
                batch = image[i:i + batch_size]
                batch_tensors = []
                
                for img_path in batch:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    img = self.preprocess(img)
                    batch_tensors.append(img)
                    
                # Stack batch tensors
                batch_tensor = torch.cat(batch_tensors, dim=0)
                
                # Run inference
                with torch.no_grad():
                    batch_results = self.model(batch_tensor)
                    
                # Postprocess results
                batch_processed = self.postprocess(batch_results, conf_threshold, iou_threshold)
                results.extend(batch_processed)
                
            return results
            
        # Run inference
        with torch.no_grad():
            results = self.model(image)
            
        # Postprocess results
        return self.postprocess(results, conf_threshold, iou_threshold)
        
    def visualize(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """Visualize detection results.
        
        Args:
            image: Input image
            results: Detection results
            class_names: List of class names
            colors: Dictionary mapping class indices to colors
            
        Returns:
            np.ndarray: Image with visualizations
        """
        if class_names is None:
            class_names = ['suture', 'knot']
            
        if colors is None:
            colors = {
                0: (0, 255, 0),  # Green for sutures
                1: (0, 0, 255)   # Red for knots
            }
            
        # Create copy of image
        vis_image = image.copy()
        
        # Draw boxes and labels
        for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
            # Convert box coordinates to integers
            box = box.int().tolist()
            
            # Draw box
            cv2.rectangle(
                vis_image,
                (box[0], box[1]),
                (box[2], box[3]),
                colors[label.item()],
                2
            )
            
            # Draw label
            label_text = f"{class_names[label.item()]}: {score:.2f}"
            cv2.putText(
                vis_image,
                label_text,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[label.item()],
                2
            )
            
        return vis_image

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect sutures in images')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to model weights')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save output image')
    parser.add_argument('--conf', type=float, help='Confidence threshold')
    parser.add_argument('--iou', type=float, help='IoU threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing multiple images')
    args = parser.parse_args()
    
    # Initialize detector
    detector = SutureDetector(
        config_path=args.config,
        model_path=args.model
    )
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")
        
    # Run detection
    results = detector.detect(
        image,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        batch_size=args.batch_size
    )
    
    # Visualize results
    vis_image = detector.visualize(image, results[0])
    
    # Save output
    cv2.imwrite(args.output, vis_image) 