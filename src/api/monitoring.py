import time
import psutil
import threading
from typing import Dict, Any
from datetime import datetime
from ..config import settings
from .logging_config import logger

class PerformanceMonitor:
    """Monitor and log system and model performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "request_count": 0,
            "average_response_time": 0,
            "error_count": 0,
            "model_inference_times": [],
            "memory_usage": [],
            "cpu_usage": []
        }
        self.lock = threading.Lock()
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start the background monitoring thread."""
        if settings.ENABLE_PERFORMANCE_MONITORING:
            self.monitor_thread = threading.Thread(
                target=self._monitor_system_metrics,
                daemon=True
            )
            self.monitor_thread.start()
    
    def _monitor_system_metrics(self):
        """Monitor system metrics in the background."""
        while True:
            try:
                # Collect system metrics
                memory = psutil.virtual_memory().percent
                cpu = psutil.cpu_percent()
                
                with self.lock:
                    self.metrics["memory_usage"].append(memory)
                    self.metrics["cpu_usage"].append(cpu)
                    
                    # Keep only last hour of metrics
                    if len(self.metrics["memory_usage"]) > 3600:
                        self.metrics["memory_usage"] = self.metrics["memory_usage"][-3600:]
                        self.metrics["cpu_usage"] = self.metrics["cpu_usage"][-3600:]
                
                # Log metrics at specified interval
                if len(self.metrics["memory_usage"]) % settings.PERFORMANCE_LOG_INTERVAL == 0:
                    self._log_metrics()
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request and its response time."""
        with self.lock:
            self.metrics["request_count"] += 1
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["request_count"] - 1) + response_time)
                / self.metrics["request_count"]
            )
            if not success:
                self.metrics["error_count"] += 1
    
    def record_model_inference(self, inference_time: float):
        """Record model inference time."""
        with self.lock:
            self.metrics["model_inference_times"].append(inference_time)
            if len(self.metrics["model_inference_times"]) > 1000:  # Keep last 1000 inferences
                self.metrics["model_inference_times"] = self.metrics["model_inference_times"][-1000:]
    
    def _log_metrics(self):
        """Log current performance metrics."""
        with self.lock:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "request_count": self.metrics["request_count"],
                "average_response_time": self.metrics["average_response_time"],
                "error_count": self.metrics["error_count"],
                "average_inference_time": (
                    sum(self.metrics["model_inference_times"]) / len(self.metrics["model_inference_times"])
                    if self.metrics["model_inference_times"]
                    else 0
                ),
                "average_memory_usage": sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"]),
                "average_cpu_usage": sum(self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"])
            }
            
            logger.info("Performance metrics: %s", metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            return {
                "request_count": self.metrics["request_count"],
                "average_response_time": self.metrics["average_response_time"],
                "error_count": self.metrics["error_count"],
                "average_inference_time": (
                    sum(self.metrics["model_inference_times"]) / len(self.metrics["model_inference_times"])
                    if self.metrics["model_inference_times"]
                    else 0
                ),
                "current_memory_usage": self.metrics["memory_usage"][-1] if self.metrics["memory_usage"] else 0,
                "current_cpu_usage": self.metrics["cpu_usage"][-1] if self.metrics["cpu_usage"] else 0
            }

# Initialize global performance monitor
performance_monitor = PerformanceMonitor() 