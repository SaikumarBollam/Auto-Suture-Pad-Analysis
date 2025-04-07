import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from pathlib import Path
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats, signal, ndimage
import sys
from skimage import measure, morphology, filters, feature, segmentation
from sklearn.cluster import KMeans

@dataclass
class SutureMetrics:
    """Class to store suture analysis metrics."""
    # Preprocessing metrics
    processing_time: float
    memory_usage: float
    contour_count: int
    contour_quality: float
    noise_level: float
    texture_score: float
    sharpness_score: float
    contrast_score: float
    entropy_score: float
    homogeneity_score: float
    uniformity_score: float
    segmentation_score: float
    feature_score: float
    
    # Quality metrics
    tension_score: float
    knot_tightness: float
    symmetry_score: float
    
    # Analysis metrics
    suture_length: float
    knot_to_incision_distance: float
    tail_length: float
    angle_count: int
    mean_angle: float
    angle_variance: float

class SutureProcessor:
    def __init__(self, 
                 image_size: Tuple[int, int] = (640, 640),
                 blur_kernel: Tuple[int, int] = (5, 5),
                 canny_thresholds: Tuple[int, int] = (50, 150),
                 min_contour_area: float = 100.0,
                 adaptive_threshold_block: int = 11,
                 adaptive_threshold_c: int = 2,
                 morph_kernel_size: int = 3,
                 texture_window: int = 5,
                 sharpness_threshold: float = 0.1,
                 contrast_clip_limit: float = 2.0,
                 contrast_grid_size: int = 8,
                 num_clusters: int = 3,
                 feature_scale: float = 1.0,
                 segmentation_scale: float = 100.0,
                 pixel_to_mm: float = 0.1,
                 min_angle: float = 30.0):
        """Initialize the suture processor with enhanced parameters."""
        self.image_size = image_size
        self.blur_kernel = blur_kernel
        self.canny_thresholds = canny_thresholds
        self.min_contour_area = min_contour_area
        self.adaptive_threshold_block = adaptive_threshold_block
        self.adaptive_threshold_c = adaptive_threshold_c
        self.morph_kernel_size = morph_kernel_size
        self.texture_window = texture_window
        self.sharpness_threshold = sharpness_threshold
        self.contrast_clip_limit = contrast_clip_limit
        self.contrast_grid_size = contrast_grid_size
        self.num_clusters = num_clusters
        self.feature_scale = feature_scale
        self.segmentation_scale = segmentation_scale
        self.pixel_to_mm = pixel_to_mm
        self.min_angle = min_angle
        
    def preprocess_image(self, 
                        image: np.ndarray,
                        resize: bool = True) -> Dict[str, Union[np.ndarray, SutureMetrics]]:
        """Preprocess an image with multiple enhanced techniques."""
        start_time = time.time()
        
        # Resize if needed
        if resize:
            image = cv2.resize(image, self.image_size)
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.contrast_clip_limit,
            tileGridSize=(self.contrast_grid_size, self.contrast_grid_size)
        )
        enhanced = clahe.apply(blurred)
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_threshold_block,
            self.adaptive_threshold_c
        )
        
        # Morphological operations
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection
        edges = cv2.Canny(morph, *self.canny_thresholds)
        
        # Texture analysis
        texture = self.analyze_texture(enhanced)
        
        # Sharpness analysis
        sharpness = self.analyze_sharpness(enhanced)
        
        # Feature detection
        features = self.detect_features(enhanced)
        
        # Image segmentation
        segments = self.segment_image(enhanced)
        
        # Find contours
        contours, hierarchy = self.detect_contours(edges)
        
        # Draw contours with hierarchy
        contour_img = image.copy()
        self.draw_contours_with_hierarchy(contour_img, contours, hierarchy)
        
        # Calculate metrics
        metrics = self.calculate_metrics(contours, edges, enhanced, segments)
        
        return {
            "original": image,
            "grayscale": gray,
            "blurred": blurred,
            "enhanced": enhanced,
            "adaptive": adaptive,
            "morphological": morph,
            "edges": edges,
            "texture": texture,
            "sharpness": sharpness,
            "features": features,
            "segments": segments,
            "contours": contour_img,
            "metrics": metrics
        }
    
    def analyze_suture(self, image: np.ndarray) -> Dict[str, Union[np.ndarray, SutureMetrics]]:
        """Analyze a suture image with comprehensive metrics."""
        # Preprocess image
        results = self.preprocess_image(image)
        
        # Get contours
        contours = self.detect_contours(results["edges"])[0]
        
        if not contours:
            return results
            
        # Find main suture contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate suture length
        suture_length = cv2.arcLength(main_contour, True) * self.pixel_to_mm
        
        # Find knot and incision points
        knot_point, incision_point = self.find_knot_and_incision(main_contour)
        
        # Calculate knot-to-incision distance
        knot_to_incision = cv2.norm(knot_point - incision_point) * self.pixel_to_mm
        
        # Calculate tail length
        tail_length = self.calculate_tail_length(main_contour, knot_point) * self.pixel_to_mm
        
        # Analyze angles
        angles = self.detect_angles(main_contour)
        angle_count = len(angles)
        mean_angle = np.mean(angles) if angles else 0.0
        angle_variance = np.var(angles) if angles else 0.0
        
        # Assess quality
        tension_score = self.analyze_tension(main_contour)
        knot_tightness = self.assess_knot_tightness(main_contour, knot_point)
        symmetry_score = self.evaluate_symmetry(main_contour)
        
        # Update metrics
        metrics = results["metrics"]
        metrics.suture_length = suture_length
        metrics.knot_to_incision_distance = knot_to_incision
        metrics.tail_length = tail_length
        metrics.angle_count = angle_count
        metrics.mean_angle = mean_angle
        metrics.angle_variance = angle_variance
        metrics.tension_score = tension_score
        metrics.knot_tightness = knot_tightness
        metrics.symmetry_score = symmetry_score
        
        return results
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str) -> None:
        """Process all images in a directory with enhanced features."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metrics summary file
        metrics_file = output_dir / 'suture_metrics.csv'
        with open(metrics_file, 'w') as f:
            f.write('image,processing_time,memory_usage,contour_count,contour_quality,'
                   'noise_level,texture_score,sharpness_score,contrast_score,entropy_score,'
                   'homogeneity_score,uniformity_score,segmentation_score,feature_score,'
                   'tension_score,knot_tightness,symmetry_score,suture_length,'
                   'knot_to_incision_distance,tail_length,angle_count,mean_angle,angle_variance\n')
        
        for img_path in input_dir.glob("*.jpg"):
            try:
                # Process image
                results = self.analyze_suture(
                    cv2.imread(str(img_path))
                )
                
                # Save results
                image_output_dir = output_dir / img_path.stem
                image_output_dir.mkdir(exist_ok=True)
                
                for name, img in results.items():
                    if name != 'metrics':
                        output_path = image_output_dir / f"{name}.jpg"
                        cv2.imwrite(str(output_path), img)
                
                # Save visualization
                self.visualize_analysis(
                    results,
                    output_path=str(image_output_dir / 'analysis_results.jpg')
                )
                
                # Save metrics
                metrics = results['metrics']
                with open(metrics_file, 'a') as f:
                    f.write(f"{img_path.stem},{metrics.processing_time:.2f},"
                           f"{metrics.memory_usage:.2f},{metrics.contour_count},"
                           f"{metrics.contour_quality:.2f},{metrics.noise_level:.2f},"
                           f"{metrics.texture_score:.2f},{metrics.sharpness_score:.2f},"
                           f"{metrics.contrast_score:.2f},{metrics.entropy_score:.2f},"
                           f"{metrics.homogeneity_score:.2f},{metrics.uniformity_score:.2f},"
                           f"{metrics.segmentation_score:.2f},{metrics.feature_score:.2f},"
                           f"{metrics.tension_score:.2f},{metrics.knot_tightness:.2f},"
                           f"{metrics.symmetry_score:.2f},{metrics.suture_length:.2f},"
                           f"{metrics.knot_to_incision_distance:.2f},{metrics.tail_length:.2f},"
                           f"{metrics.angle_count},{metrics.mean_angle:.2f},{metrics.angle_variance:.2f}\n")
                
                print(f"Processed {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
                
        print(f"\nProcessing complete. Results saved to: {output_dir}")
    
    def visualize_analysis(self, 
                         results: Dict[str, Union[np.ndarray, SutureMetrics]],
                         output_path: Optional[str] = None) -> None:
        """Visualize analysis results with enhanced display."""
        # Create a figure with subplots
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle('Suture Analysis Results', fontsize=16)
        
        # Plot each preprocessing step
        steps = [
            ('Original', 'original'),
            ('Grayscale', 'grayscale'),
            ('Blurred', 'blurred'),
            ('Enhanced', 'enhanced'),
            ('Adaptive', 'adaptive'),
            ('Morphological', 'morphological'),
            ('Edges', 'edges'),
            ('Texture', 'texture'),
            ('Sharpness', 'sharpness'),
            ('Features', 'features'),
            ('Segments', 'segments'),
            ('Contours', 'contours')
        ]
        
        for i, (title, key) in enumerate(steps):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            if key == 'original':
                ax.imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(results[key], cmap='gray')
            
            ax.set_title(title)
            ax.axis('off')
        
        # Add metrics to the last subplot
        metrics = results['metrics']
        ax = axes[3, 2]
        ax.text(0.1, 0.5, 
                f"Processing Time: {metrics.processing_time:.2f}s\n"
                f"Memory Usage: {metrics.memory_usage:.2f}MB\n"
                f"Contour Count: {metrics.contour_count}\n"
                f"Contour Quality: {metrics.contour_quality:.2f}\n"
                f"Noise Level: {metrics.noise_level:.2f}\n"
                f"Texture Score: {metrics.texture_score:.2f}\n"
                f"Sharpness Score: {metrics.sharpness_score:.2f}\n"
                f"Contrast Score: {metrics.contrast_score:.2f}\n"
                f"Entropy Score: {metrics.entropy_score:.2f}\n"
                f"Homogeneity: {metrics.homogeneity_score:.2f}\n"
                f"Uniformity: {metrics.uniformity_score:.2f}\n"
                f"Segmentation: {metrics.segmentation_score:.2f}\n"
                f"Feature Score: {metrics.feature_score:.2f}\n"
                f"Tension Score: {metrics.tension_score:.2f}\n"
                f"Knot Tightness: {metrics.knot_tightness:.2f}\n"
                f"Symmetry Score: {metrics.symmetry_score:.2f}\n"
                f"Suture Length: {metrics.suture_length:.2f}mm\n"
                f"Knot-Incision Distance: {metrics.knot_to_incision_distance:.2f}mm\n"
                f"Tail Length: {metrics.tail_length:.2f}mm\n"
                f"Angle Count: {metrics.angle_count}\n"
                f"Mean Angle: {metrics.mean_angle:.2f}°\n"
                f"Angle Variance: {metrics.angle_variance:.2f}",
                fontsize=10)
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show() 