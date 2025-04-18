"""Image processing utilities for suture analysis.

This module provides functionality for image processing operations
including cropping, resizing, and basic image manipulations.
"""

import os
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def crop_image(
    image_path: str,
    box: np.ndarray,
    output_path: Optional[str] = None,
    padding: int = 0
) -> Optional[np.ndarray]:
    """Crop an image using bounding box coordinates.
    
    Args:
        image_path: Path to the input image
        box: Bounding box coordinates
        output_path: Optional path to save the cropped image
        padding: Additional padding around the crop
        
    Returns:
        Cropped image array or None if cropping fails
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width = image.shape[:2]
    
    # Get coordinates from box
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    
    # Calculate crop boundaries with padding
    x_min = max(0, min(x_coords) - padding)
    y_min = max(0, min(y_coords) - padding)
    x_max = min(width, max(x_coords) + padding)
    y_max = min(height, max(y_coords) + padding)
    
    if x_max <= x_min or y_max <= y_min:
        return None

    # Crop the image
    cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    if cropped.size == 0:
        return None

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cropped)

    return cropped

def crop_and_save_detections(
    image_path: str,
    detections: list,
    output_dir: str,
    padding: int = 0
) -> list:
    """Crop and save multiple detections from an image.
    
    Args:
        image_path: Path to the input image
        detections: List of (class_id, confidence, box) tuples
        output_dir: Directory to save cropped images
        padding: Additional padding around crops
        
    Returns:
        List of paths to saved cropped images
    """
    saved_paths = []
    image_name = os.path.basename(image_path)
    
    for idx, (class_id, confidence, box) in enumerate(detections):
        crop_name = f"{os.path.splitext(image_name)[0]}_crop_{idx}.jpg"
        output_path = os.path.join(output_dir, crop_name)
        
        cropped = crop_image(image_path, box, output_path, padding)
        if cropped is not None:
            saved_paths.append(output_path)
    
    return saved_paths

def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """Resize an image to target dimensions.
    
    Args:
        image: Input image array
        target_size: Target (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image array
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size)
    else:
        return cv2.resize(image, target_size)

def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Normalize image pixel values.
    
    Args:
        image: Input image array
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image array
    """
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image

def draw_detections(
    image: np.ndarray,
    detections: list,
    class_names: dict,
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """Draw detection boxes and labels on image.
    
    Args:
        image: Input image array
        detections: List of (class_id, confidence, box) tuples
        class_names: Dictionary mapping class IDs to names
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Image with drawn detections
    """
    image = image.copy()
    
    for class_id, confidence, box in detections:
        if confidence < confidence_threshold:
            continue
            
        # Convert box coordinates to integers
        box = box.astype(int)
        
        # Draw box
        cv2.polylines(image, [box], True, (0, 255, 0), 2)
        
        # Prepare label
        class_name = class_names.get(class_id, f"Class {class_id}")
        label = f"{class_name}: {confidence:.2f}"
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_pos = (box[0][0], box[0][1] - 10 if box[0][1] > 20 else box[0][1] + 20)
        cv2.rectangle(
            image,
            (label_pos[0], label_pos[1] - label_size[1]),
            (label_pos[0] + label_size[0], label_pos[1]),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            image,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return image 

def process_directory(
    input_dir: str,
    output_dir: str,
    process_func: callable,
    file_pattern: str = "*.jpg",
    num_workers: int = 4
) -> List[str]:
    """Process all images in a directory using multiple workers.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        process_func: Function to process each image
        file_pattern: Pattern to match image files
        num_workers: Number of worker threads
        
    Returns:
        List of paths to processed images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob(file_pattern))
    processed_paths = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for img_path in image_files:
            output_path = output_dir / img_path.name
            futures.append(executor.submit(process_func, str(img_path), str(output_path)))
        
        for future in tqdm(futures, total=len(image_files), desc="Processing images"):
            try:
                result = future.result()
                if result:
                    processed_paths.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
    
    return processed_paths

def batch_crop_images(
    input_dir: str,
    output_dir: str,
    detections: Dict[str, List[Tuple[int, float, np.ndarray]]],
    padding: int = 0,
    num_workers: int = 4
) -> Dict[str, List[str]]:
    """Crop multiple images based on detections.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped images
        detections: Dictionary mapping image names to detection lists
        padding: Additional padding around crops
        num_workers: Number of worker threads
        
    Returns:
        Dictionary mapping image names to lists of cropped image paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cropped_paths = {}
    
    def process_image(img_name: str) -> List[str]:
        img_path = input_dir / img_name
        if not img_path.exists():
            return []
            
        img_detections = detections.get(img_name, [])
        if not img_detections:
            return []
            
        return crop_and_save_detections(
            str(img_path),
            img_detections,
            str(output_dir),
            padding
        )
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            img_name: executor.submit(process_image, img_name)
            for img_name in detections.keys()
        }
        
        for img_name, future in tqdm(futures.items(), total=len(futures), desc="Cropping images"):
            try:
                paths = future.result()
                if paths:
                    cropped_paths[img_name] = paths
            except Exception as e:
                print(f"Error cropping image {img_name}: {e}")
    
    return cropped_paths

def batch_resize_images(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
    num_workers: int = 4
) -> List[str]:
    """Resize all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save resized images
        target_size: Target (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        num_workers: Number of worker threads
        
    Returns:
        List of paths to resized images
    """
    def process_image(input_path: str, output_path: str) -> Optional[str]:
        img = cv2.imread(input_path)
        if img is None:
            return None
            
        resized = resize_image(img, target_size, keep_aspect_ratio)
        cv2.imwrite(output_path, resized)
        return output_path
    
    return process_directory(
        input_dir,
        output_dir,
        process_image,
        num_workers=num_workers
    ) 