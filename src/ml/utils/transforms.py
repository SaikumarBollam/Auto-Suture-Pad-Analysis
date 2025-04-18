import torchvision.transforms as T
from typing import Tuple, Dict, Any

def create_train_transform(input_size: Tuple[int, int]) -> T.Compose:
    """Create training transform.
    
    Args:
        input_size: Input image size
        
    Returns:
        T.Compose: Training transform
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        T.RandomResizedCrop(
            size=input_size,
            scale=(0.8, 1.0)
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_val_transform(input_size: Tuple[int, int]) -> T.Compose:
    """Create validation transform.
    
    Args:
        input_size: Input image size
        
    Returns:
        T.Compose: Validation transform
    """
    return T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_transforms(config: Dict[str, Any]) -> Dict[str, T.Compose]:
    """Get transforms for training and validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, T.Compose]: Dictionary of transforms
    """
    input_size = config.get('input_size', (640, 640))
    return {
        'train': create_train_transform(input_size),
        'val': create_val_transform(input_size)
    } 