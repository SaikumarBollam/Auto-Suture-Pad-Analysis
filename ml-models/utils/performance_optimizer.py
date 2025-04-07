import torch
import onnx
from typing import Dict, Any, Optional
from pathlib import Path

class PerformanceOptimizer:
    """Class for optimizing model performance through various techniques."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the performance optimizer.
        
        Args:
            device (str, optional): Device to use for optimization
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def optimize_model(self, model: torch.nn.Module, dummy_input: torch.Tensor, 
                      save_path: Optional[str] = None) -> torch.nn.Module:
        """Optimize a model using various techniques.
        
        Args:
            model (torch.nn.Module): Model to optimize
            dummy_input (torch.Tensor): Example input for model tracing
            save_path (str, optional): Path to save optimized model
            
        Returns:
            torch.nn.Module: Optimized model
        """
        # Move model to device
        model = model.to(self.device)
        
        # Convert to ONNX for better inference
        if save_path:
            onnx_path = Path(save_path).with_suffix('.onnx')
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        
        # Quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save quantized model
        if save_path:
            torch.save(quantized_model.state_dict(), save_path)
        
        return quantized_model
    
    def batch_process(self, model: torch.nn.Module, data: torch.Tensor, 
                     batch_size: int = 32) -> torch.Tensor:
        """Process data in batches for memory efficiency.
        
        Args:
            model (torch.nn.Module): Model to use for processing
            data (torch.Tensor): Input data
            batch_size (int): Size of each batch
            
        Returns:
            torch.Tensor: Processed results
        """
        model.eval()
        with torch.no_grad():
            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(self.device)
                output = model(batch)
                results.append(output.cpu())
            return torch.cat(results)
    
    def optimize_inference(self, model: torch.nn.Module, 
                         input_shape: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
        """Optimize model for inference.
        
        Args:
            model (torch.nn.Module): Model to optimize
            input_shape (tuple): Shape of input tensor
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Benchmark original model
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            
            original_time = start.elapsed_time(end)
        
        # Optimize model
        optimized_model = self.optimize_model(model, dummy_input)
        
        # Benchmark optimized model
        with torch.no_grad():
            start.record()
            _ = optimized_model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            
            optimized_time = start.elapsed_time(end)
        
        return {
            'original_time_ms': original_time,
            'optimized_time_ms': optimized_time,
            'speedup': original_time / optimized_time
        }