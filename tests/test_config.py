"""Tests for configuration management."""

import pytest
import os
import yaml
from src.ml.utils.config import ConfigManager

@pytest.fixture
def test_config_path(tmp_path):
    """Create a temporary test config file."""
    config = {
        'ml': {
            'model': {
                'weights': 'test_weights.pt',
                'confidence_threshold': 0.25
            },
            'preprocessing': {
                'image_size': [640, 640],
                'augmentation': {'enabled': True}
            }
        },
        'measurements': {
            'reference_distance_mm': 10.0,
            'pixel_mm_calibration': {
                'enabled': True,
                'reference_points': [
                    [0.22995, 0.5814],
                    [0.22428, 0.30907]
                ]
            }
        },
        'analysis': {
            'suture_classes': ['suture_good', 'suture_loose'],
            'knot_classes': ['knot_good', 'knot_loose'],
            'incision_classes': ['incision'],
            'quality_thresholds': {
                'angle_deviation_max': 15.0
            }
        },
        'visualization': {
            'colors': {
                'suture': [0, 255, 0],
                'knot': [255, 0, 0]
            }
        }
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return str(config_path)

def test_config_loading(test_config_path):
    """Test basic config loading."""
    config_manager = ConfigManager(test_config_path)
    assert config_manager.config is not None
    
    # Test non-existent config
    with pytest.raises(FileNotFoundError):
        ConfigManager("non_existent.yaml")

def test_get_model_config(test_config_path):
    """Test model config retrieval."""
    config_manager = ConfigManager(test_config_path)
    model_config = config_manager.get_model_config()
    
    assert model_config['weights'] == 'test_weights.pt'
    assert model_config['confidence_threshold'] == 0.25

def test_get_preprocessing_config(test_config_path):
    """Test preprocessing config retrieval."""
    config_manager = ConfigManager(test_config_path)
    preproc_config = config_manager.get_preprocessing_config()
    
    assert preproc_config['image_size'] == [640, 640]
    assert preproc_config['augmentation']['enabled'] is True

def test_get_measurement_config(test_config_path):
    """Test measurement config retrieval."""
    config_manager = ConfigManager(test_config_path)
    meas_config = config_manager.get_measurement_config()
    
    assert meas_config['reference_distance_mm'] == 10.0
    assert meas_config['pixel_mm_calibration']['enabled'] is True

def test_get_reference_points(test_config_path):
    """Test reference points retrieval."""
    config_manager = ConfigManager(test_config_path)
    ref_points = config_manager.get_reference_points()
    
    assert ref_points is not None
    assert len(ref_points) == 2
    assert ref_points[0] == (0.22995, 0.5814)
    assert ref_points[1] == (0.22428, 0.30907)

def test_get_class_names(test_config_path):
    """Test class names retrieval."""
    config_manager = ConfigManager(test_config_path)
    class_names = config_manager.get_class_names()
    
    assert 'suture' in class_names
    assert 'knot' in class_names
    assert 'incision' in class_names
    assert 'suture_good' in class_names['suture']
    assert 'knot_good' in class_names['knot']

def test_get_quality_thresholds(test_config_path):
    """Test quality thresholds retrieval."""
    config_manager = ConfigManager(test_config_path)
    thresholds = config_manager.get_quality_thresholds()
    
    assert 'angle_deviation_max' in thresholds
    assert thresholds['angle_deviation_max'] == 15.0

def test_get_colors(test_config_path):
    """Test visualization colors retrieval."""
    config_manager = ConfigManager(test_config_path)
    colors = config_manager.get_colors()
    
    assert 'suture' in colors
    assert 'knot' in colors
    assert colors['suture'] == [0, 255, 0]
    assert colors['knot'] == [255, 0, 0]

def test_save_config(test_config_path):
    """Test config saving."""
    config_manager = ConfigManager(test_config_path)
    
    # Modify config
    config_manager.config['new_key'] = 'test_value'
    config_manager.save_config(config_manager.config)
    
    # Load modified config
    new_config_manager = ConfigManager(test_config_path)
    assert 'new_key' in new_config_manager.config
    assert new_config_manager.config['new_key'] == 'test_value'