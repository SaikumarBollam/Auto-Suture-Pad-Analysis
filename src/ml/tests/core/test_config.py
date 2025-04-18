"""Tests for configuration management."""

import pytest
from ml_models.core import Config

def test_config_initialization():
    """Test basic configuration initialization."""
    config = Config()
    assert hasattr(config, 'model_type')
    assert hasattr(config, 'num_classes')
    assert hasattr(config, 'input_size')

def test_config_validation():
    """Test configuration validation."""
    config = Config()
    
    # Test valid model type
    config.model_type = 'yolo'
    assert config.model_type == 'yolo'
    
    # Test invalid model type
    with pytest.raises(ValueError):
        config.model_type = 'invalid_model'
        
def test_config_saving_loading(tmp_path):
    """Test configuration saving and loading."""
    config = Config()
    config.model_type = 'yolo'
    config.num_classes = 2
    
    # Save config
    config_path = tmp_path / 'config.yaml'
    config.save(config_path)
    
    # Load config
    new_config = Config()
    new_config.load(config_path)
    
    assert new_config.model_type == config.model_type
    assert new_config.num_classes == config.num_classes 