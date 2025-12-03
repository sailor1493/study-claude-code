"""Accuracy benchmark using COCO-style evaluation."""

from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from pathlib import Path

from object_detection.benchmarks.base import BaseBenchmark, compute_map
from object_detection.models.base import DetectionResult


class AccuracyBenchmark(BaseBenchmark):
    """Benchmark for measuring detection accuracy."""
    
    def __init__(
        self, 
        detector, 
        data_path: Optional[Path] = None,
        ground_truths: Optional[List[DetectionResult]] = None,
    ):
        """
        Initialize accuracy benchmark.
        
        Args:
            detector: Object detector to benchmark
            data_path: Path to test images
            ground_truths: Ground truth detection results
        """
        super().__init__("Accuracy", detector, data_path)
        self.ground_truths = ground_truths or []
        self.images = []
    
    def load_data(self) -> List[Image.Image]:
        """Load test images."""
        if self.data_path and self.data_path.exists():
            # Load images from directory
            image_files = list(self.data_path.glob("*.jpg")) + \
                         list(self.data_path.glob("*.png"))
            self.images = [Image.open(f) for f in sorted(image_files)]
        else:
            # Generate synthetic test data if no real data available
            for _ in range(10):
                img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                self.images.append(Image.fromarray(img_array))
        
        return self.images
    
    def evaluate(self, predictions: List[DetectionResult]) -> Dict[str, float]:
        """
        Evaluate accuracy metrics.
        
        Args:
            predictions: List of detection results
            
        Returns:
            Dictionary of accuracy metrics
        """
        if not self.ground_truths:
            # If no ground truth, compute basic statistics
            total_detections = sum(len(pred.boxes) for pred in predictions)
            avg_confidence = np.mean([
                pred.scores.mean() if len(pred.scores) > 0 else 0.0
                for pred in predictions
            ])
            
            return {
                "total_detections": float(total_detections),
                "avg_confidence": float(avg_confidence),
            }
        
        # Compute mAP with ground truth
        metrics = compute_map(predictions, self.ground_truths)
        return metrics
