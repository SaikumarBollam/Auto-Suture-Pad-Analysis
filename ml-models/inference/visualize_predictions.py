import argparse
from pathlib import Path
from ml_models.utils.visualization import Visualizer
from ml_models.config import VISUALIZATION_CONFIG, DATASET_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO predictions on images")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--predictions_dir", type=str, required=True,
                       help="Directory containing YOLO prediction files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualized images (default: config output dir)")
    args = parser.parse_args()
    
    # Use provided output directory or default from config
    output_dir = args.output_dir or str(VISUALIZATION_CONFIG["output_dir"])
    
    # Initialize visualizer
    visualizer = Visualizer(output_dir)
    
    # Process all images in the directory
    visualizer.process_directory(
        image_dir=args.image_dir,
        predictions_dir=args.predictions_dir,
        class_names=DATASET_CONFIG["class_names"]
    )
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main() 