# Preprocessing Layer

This layer handles all image preprocessing and feature extraction for suture analysis.

## Components

### SutureProcessor
The main class for processing suture images and extracting features.

#### Features
- Image preprocessing (resizing, normalization)
- Edge detection
- Contour analysis
- Texture analysis
- Feature extraction

## Usage

```python
from ml_models.preprocessing import SutureProcessor
from ml_models.config import Config

# Initialize with configuration
config = Config()
processor = SutureProcessor(**config.get_preprocessing_config())

# Process a single image
features = processor.process_image("path/to/image.jpg")

# Process a directory of images
processor.process_directory("input_dir", "output_dir")
```

## Output Format

The processor returns a dictionary containing:
- `image`: Preprocessed image tensor
- `contours`: Detected contours
- `edges`: Edge detection results
- `metrics`: Various measurements (area, perimeter, etc.)
- `features`: Extracted features for model input

## Configuration

Key preprocessing parameters can be configured through the Config class:
- Image size
- Blur kernel size
- Edge detection thresholds
- Contour analysis parameters
- Texture analysis parameters

See `config.py` for all available parameters. 