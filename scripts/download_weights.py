"""Script to download YOLOv8 model weights."""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def download_weights(model_name: str = "yolov8n.pt"):
    """Download YOLOv8 weights."""
    try:
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        
        model = YOLO(model_name)
        model_path = weights_dir / model_name
        
        if not model_path.exists():
            print(f"Downloading {model_name}...")
            model.export(format="onnx")  # This also saves the PyTorch model
            print(f"Model saved to {model_path}")
        else:
            print(f"Model already exists at {model_path}")
            
    except Exception as e:
        print(f"Error downloading weights: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_weights()