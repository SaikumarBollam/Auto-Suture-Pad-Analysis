import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from tqdm import tqdm
import numpy as np
import logging
import shutil
from datetime import datetime
from ..models.model import get_model
from ..config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train(
    config_path: Optional[str] = None,
    data_yaml: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train suture detection model.
    
    Args:
        config_path: Path to configuration file
        data_yaml: Path to data configuration file
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        device: Device to use for training
        output_dir: Directory to save outputs
        **kwargs: Additional training arguments
        
    Returns:
        Dict[str, Any]: Training results
        
    Raises:
        FileNotFoundError: If required paths don't exist
        RuntimeError: If training fails
    """
    try:
        # Load configuration
        config = Config(config_path)
        logger.info("Loaded configuration")
        
        # Override config with arguments
        if epochs is not None:
            config.config['training']['epochs'] = epochs
        if batch_size is not None:
            config.config['training']['batch_size'] = batch_size
        if imgsz is not None:
            config.config['model']['input_size'] = (imgsz, imgsz)
        if device is not None:
            config.config['deployment']['device'] = device
            
        # Get training configuration
        train_config = config.get_training_config()
        logger.info(f"Training configuration: {train_config}")
        
        # Get model configuration
        model_config = config.get_model_config()
        logger.info(f"Model configuration: {model_config}")
        
        # Create output directory
        if output_dir is None:
            output_dir = Path('outputs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Save configuration
        config.save_config(output_dir / 'config.yaml')
        logger.info("Saved configuration")
        
        # Initialize model
        model = get_model(
            model_type=model_config['type'],
            model_size=model_config['model_size'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained']
        )
        logger.info("Initialized model")
        
        # Set device
        device = torch.device(config.get_deployment_config()['device'])
        model.to(device)
        logger.info(f"Using device: {device}")
        
        # Create data configuration
        if data_yaml is None:
            train_path = Path(config.get_data_config()['train_path'])
            val_path = Path(config.get_data_config()['val_path'])
            
            if not train_path.exists():
                raise FileNotFoundError(f"Training path does not exist: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"Validation path does not exist: {val_path}")
                
            data_yaml = {
                'train': str(train_path),
                'val': str(val_path),
                'nc': model_config['num_classes'],
                'names': ['suture', 'knot']
            }
            
            # Save data configuration
            data_yaml_path = output_dir / 'data.yaml'
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            data_yaml = str(data_yaml_path)
            logger.info(f"Created data configuration: {data_yaml}")
            
        # Train model
        logger.info("Starting training")
        with tqdm(total=train_config['epochs'], desc="Training") as pbar:
            results = model.model.train(
                data=data_yaml,
                epochs=train_config['epochs'],
                batch_size=train_config['batch_size'],
                imgsz=model_config['input_size'][0],
                device=device,
                project=str(output_dir),
                name='train',
                **kwargs
            )
            pbar.update(1)
            
        # Save best model
        best_model_path = output_dir / 'weights' / f'best_{model_config["model_size"]}.pt'
        best_model_path.parent.mkdir(exist_ok=True)
        model.save(best_model_path)
        logger.info(f"Saved best model to {best_model_path}")
        
        # Save final model
        final_model_path = output_dir / 'weights' / f'final_{model_config["model_size"]}.pt'
        model.save(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise RuntimeError(f"Training failed: {str(e)}")

# %%
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train suture detection model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data configuration file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--imgsz', type=int, help='Image size')
    parser.add_argument('--device', type=str, help='Device to use for training')
    parser.add_argument('--output-dir', type=str, help='Directory to save outputs')
    args = parser.parse_args()
    
    # Train model
    results = train(
        config_path=args.config,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir
    )
    
    print(f"Training completed. Results: {results}") 
# %%
