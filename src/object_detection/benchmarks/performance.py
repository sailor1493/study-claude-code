"""Performance benchmark for measuring inference speed."""

from typing import List, Dict, Any
import numpy as np
from PIL import Image
from pathlib import Path

from object_detection.benchmarks.base import BaseBenchmark
from object_detection.models.base import DetectionResult


class PerformanceBenchmark(BaseBenchmark):
    """Benchmark for measuring inference performance (speed, throughput)."""
    
    def __init__(self, detector, num_images: int = 100, image_size: tuple = (640, 480)):
        """
        Initialize performance benchmark.
        
        Args:
            detector: Object detector to benchmark
            num_images: Number of synthetic images to test
            image_size: Size of synthetic images (width, height)
        """
        super().__init__("Performance", detector)
        self.num_images = num_images
        self.image_size = image_size
    
    def load_data(self) -> List[Image.Image]:
        """Generate synthetic images for performance testing."""
        images = []
        for _ in range(self.num_images):
            # Create random RGB image
            img_array = np.random.randint(0, 255, 
                                         (self.image_size[1], self.image_size[0], 3),
                                         dtype=np.uint8)
            images.append(Image.fromarray(img_array))
        return images
    
    def evaluate(self, predictions: List[DetectionResult]) -> Dict[str, float]:
        """
        Evaluate performance metrics.
        
        Args:
            predictions: List of detection results
            
        Returns:
            Dictionary of performance metrics
        """
        total_detections = sum(len(pred.boxes) for pred in predictions)
        avg_detections = total_detections / len(predictions)
        
        return {
            "total_detections": float(total_detections),
            "avg_detections_per_image": avg_detections,
        }
