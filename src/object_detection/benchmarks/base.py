"""Base classes for object detection benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import time
from pathlib import Path

from object_detection.models.base import BaseDetector, DetectionResult


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""
    
    benchmark_name: str
    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    num_images: int = 0
    total_time: float = 0.0
    avg_time_per_image: float = 0.0
    device: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.num_images > 0 and self.total_time > 0:
            self.avg_time_per_image = self.total_time / self.num_images
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "num_images": self.num_images,
            "total_time": self.total_time,
            "avg_time_per_image": self.avg_time_per_image,
            "device": self.device,
            "metadata": self.metadata,
        }
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"Benchmark: {self.benchmark_name}",
            f"Model: {self.model_name}",
            f"Device: {self.device}",
            f"{'='*60}",
            f"Images processed: {self.num_images}",
            f"Total time: {self.total_time:.2f}s",
            f"Avg time per image: {self.avg_time_per_image:.4f}s",
            f"\nMetrics:",
        ]
        
        for metric_name, value in self.metrics.items():
            lines.append(f"  {metric_name}: {value:.4f}")
        
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class BaseBenchmark(ABC):
    """Base class for object detection benchmarks."""
    
    def __init__(
        self,
        name: str,
        detector: BaseDetector,
        data_path: Optional[Path] = None,
    ):
        """
        Initialize benchmark.
        
        Args:
            name: Benchmark name
            detector: Object detector to benchmark
            data_path: Path to benchmark data
        """
        self.name = name
        self.detector = detector
        self.data_path = data_path
    
    @abstractmethod
    def load_data(self) -> List[Any]:
        """Load benchmark data."""
        pass
    
    @abstractmethod
    def evaluate(self, predictions: List[DetectionResult]) -> Dict[str, float]:
        """
        Evaluate predictions and compute metrics.
        
        Args:
            predictions: List of detection results
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def run(self) -> BenchmarkResult:
        """
        Run the benchmark.
        
        Returns:
            BenchmarkResult containing metrics and timing info
        """
        # Load data
        data = self.load_data()
        
        # Run inference
        predictions = []
        start_time = time.time()
        
        for item in data:
            pred = self.detector.predict(item)
            predictions.append(pred)
        
        total_time = time.time() - start_time
        
        # Evaluate
        metrics = self.evaluate(predictions)
        
        # Create result
        return BenchmarkResult(
            benchmark_name=self.name,
            model_name=self.detector.model_name,
            metrics=metrics,
            num_images=len(data),
            total_time=total_time,
            device=str(self.detector.device),
        )


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_map(
    predictions: List[DetectionResult],
    ground_truths: List[DetectionResult],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) metrics.
    
    Args:
        predictions: List of predicted detection results
        ground_truths: List of ground truth detection results
        iou_threshold: IoU threshold for positive matches
        
    Returns:
        Dictionary with mAP and other metrics
    """
    # Simplified mAP computation
    # In production, would use proper COCO evaluation toolkit
    
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for pred, gt in zip(predictions, ground_truths):
        total_gt += len(gt.boxes)
        
        matched_gt = set()
        
        for pred_box in pred.boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt.boxes):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP@0.5": precision,  # Simplified
    }
