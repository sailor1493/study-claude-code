"""Object Detection Framework - A modular framework for object detection models and benchmarks."""

__version__ = "0.1.0"

from object_detection.models.base import BaseDetector, DetectionResult
from object_detection.benchmarks.base import BaseBenchmark, BenchmarkResult

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "BaseBenchmark",
    "BenchmarkResult",
]
