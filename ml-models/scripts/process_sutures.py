import argparse
from pathlib import Path
from suture_processor import SutureProcessor

def main():
    parser = argparse.ArgumentParser(description='Process suture images with comprehensive analysis.')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('output_dir', type=str, help='Directory to save processed results')
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 640],
                       help='Image size for processing (width height)')
    parser.add_argument('--blur_kernel', type=int, nargs=2, default=[5, 5],
                       help='Gaussian blur kernel size (width height)')
    parser.add_argument('--canny_thresholds', type=int, nargs=2, default=[50, 150],
                       help='Canny edge detection thresholds (low high)')
    parser.add_argument('--min_contour_area', type=float, default=100.0,
                       help='Minimum contour area for detection')
    parser.add_argument('--adaptive_threshold_block', type=int, default=11,
                       help='Block size for adaptive thresholding')
    parser.add_argument('--adaptive_threshold_c', type=int, default=2,
                       help='Constant for adaptive thresholding')
    parser.add_argument('--morph_kernel_size', type=int, default=3,
                       help='Size of morphological operation kernel')
    parser.add_argument('--texture_window', type=int, default=5,
                       help='Window size for texture analysis')
    parser.add_argument('--sharpness_threshold', type=float, default=0.1,
                       help='Threshold for sharpness analysis')
    parser.add_argument('--contrast_clip_limit', type=float, default=2.0,
                       help='Clip limit for contrast enhancement')
    parser.add_argument('--contrast_grid_size', type=int, default=8,
                       help='Grid size for contrast enhancement')
    parser.add_argument('--num_clusters', type=int, default=3,
                       help='Number of clusters for segmentation')
    parser.add_argument('--feature_scale', type=float, default=1.0,
                       help='Scale factor for feature detection')
    parser.add_argument('--segmentation_scale', type=float, default=100.0,
                       help='Scale factor for image segmentation')
    parser.add_argument('--pixel_to_mm', type=float, default=0.1,
                       help='Pixel to millimeter conversion factor')
    parser.add_argument('--min_angle', type=float, default=30.0,
                       help='Minimum angle for angle detection')
    
    args = parser.parse_args()
    
    # Initialize suture processor with command line arguments
    processor = SutureProcessor(
        image_size=tuple(args.image_size),
        blur_kernel=tuple(args.blur_kernel),
        canny_thresholds=tuple(args.canny_thresholds),
        min_contour_area=args.min_contour_area,
        adaptive_threshold_block=args.adaptive_threshold_block,
        adaptive_threshold_c=args.adaptive_threshold_c,
        morph_kernel_size=args.morph_kernel_size,
        texture_window=args.texture_window,
        sharpness_threshold=args.sharpness_threshold,
        contrast_clip_limit=args.contrast_clip_limit,
        contrast_grid_size=args.contrast_grid_size,
        num_clusters=args.num_clusters,
        feature_scale=args.feature_scale,
        segmentation_scale=args.segmentation_scale,
        pixel_to_mm=args.pixel_to_mm,
        min_angle=args.min_angle
    )
    
    # Process the directory
    processor.process_directory(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main() 