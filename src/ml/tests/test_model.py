import unittest
import torch
from pathlib import Path
import tempfile

from ..models.model import get_model
from ..config import Config

class TestModel(unittest.TestCase):
    """Test the model class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.model = get_model(self.config.get_model_config())
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertTrue(isinstance(self.model, torch.nn.Module))
        
    def test_model_forward(self):
        """Test model forward pass."""
        # Create dummy input
        batch_size = 2
        input_size = self.config.get_model_config()['input_size']
        x = torch.randn(batch_size, 3, *input_size)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(x)
            
        # Check output shape
        self.assertEqual(output.shape[0], batch_size)
        
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = Path(temp_dir) / "model.pt"
            self.model.save_weights(model_path)
            
            # Load model
            new_model = get_model(self.config.get_model_config())
            new_model.load_weights(model_path)
            
            # Check if models are equal
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
                
    def test_model_training_step(self):
        """Test model training step."""
        # Create dummy input and target
        batch_size = 2
        input_size = self.config.get_model_config()['input_size']
        x = torch.randn(batch_size, 3, *input_size)
        y = torch.randint(0, 2, (batch_size,))
        
        # Training step
        loss = self.model.train_step(x, y)
        
        # Check loss
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(loss.requires_grad)
        
if __name__ == '__main__':
    unittest.main() 