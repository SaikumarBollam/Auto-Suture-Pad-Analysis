"""Tests for logging utilities."""

import pytest
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from src.ml.utils.logging import SutureLogger

@pytest.fixture
def test_log_config(tmp_path):
    """Create test logging configuration."""
    return {
        'save_path': str(tmp_path),
        'level': 'DEBUG',
        'save_images': True,
        'save_measurements': True
    }

@pytest.fixture
def test_logger(test_log_config):
    """Create test logger instance."""
    logger = SutureLogger(test_log_config)
    yield logger
    logger.close()

@pytest.fixture
def test_detections():
    """Create test detection results."""
    return {
        'boxes': np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        'classes': ['suture_good', 'knot_good'],
        'scores': [0.95, 0.87]
    }

@pytest.fixture
def test_measurements():
    """Create test measurement results."""
    return {
        'stitch_lengths': [10.5, 11.2, 9.8],
        'angles': [85.2, 88.5, 87.1],
        'average_spacing': 12.3
    }

def test_logger_initialization(test_log_config):
    """Test logger initialization."""
    logger = SutureLogger(test_log_config)
    log_dir = Path(test_log_config['save_path'])
    
    assert logger.log_dir == log_dir
    assert logger.measurement_file == log_dir / 'measurements.jsonl'
    logger.close()

def test_log_detection(test_logger, test_detections):
    """Test detection logging."""
    test_logger.log_detection(
        'test_image.jpg',
        test_detections['boxes'],
        test_detections['classes'],
        test_detections['scores']
    )
    
    # Verify log file exists and contains detection info
    log_files = list(test_logger.log_dir.glob('*.log'))
    assert len(log_files) == 1
    
    with open(log_files[0], 'r') as f:
        content = f.read()
        assert 'test_image.jpg' in content
        assert 'Found 2 detections' in content
        assert 'suture_good' in content
        assert 'knot_good' in content

def test_log_measurements(test_logger, test_measurements):
    """Test measurement logging."""
    test_logger.log_measurements('test_image.jpg', test_measurements)
    
    # Verify measurements are saved to JSONL file
    assert test_logger.measurement_file.exists()
    
    with open(test_logger.measurement_file, 'r') as f:
        record = json.loads(f.readline())
        assert record['image'] == 'test_image.jpg'
        assert 'stitch_lengths' in record['measurements']
        assert 'angles' in record['measurements']
        assert 'average_spacing' in record['measurements']

def test_save_visualization(test_logger):
    """Test visualization saving."""
    # Create test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[40:60, 40:60] = [255, 255, 255]
    
    test_logger.save_visualization('test_image.jpg', test_image)
    
    # Verify visualization is saved
    vis_dir = test_logger.log_dir / 'visualizations'
    assert vis_dir.exists()
    assert len(list(vis_dir.glob('test_image_*.png'))) == 1

def test_log_error(test_logger):
    """Test error logging."""
    test_error = ValueError("Test error")
    test_context = {'image': 'test_image.jpg', 'stage': 'detection'}
    
    test_logger.log_error(test_error, test_context)
    
    # Verify error is logged
    log_files = list(test_logger.log_dir.glob('*.log'))
    assert len(log_files) == 1
    
    with open(log_files[0], 'r') as f:
        content = f.read()
        assert 'ERROR' in content
        assert 'Test error' in content
        assert 'test_image.jpg' in content

def test_logger_cleanup(test_log_config):
    """Test logger cleanup."""
    logger = SutureLogger(test_log_config)
    
    # Log something
    logger.log_detection('test.jpg', np.array([[0, 0, 1, 1]]), ['test'], [0.9])
    
    # Close logger
    logger.close()
    
    # Verify handlers are removed
    assert len(logger.logger.handlers) == 0