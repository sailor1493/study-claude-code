"""Test benchmark functionality."""

import pytest
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from object_detection.benchmarks.base import (
    BaseBenchmark, 
    BenchmarkResult,
    compute_iou,
    compute_map
)
from object_detection.models.base import DetectionResult


def test_benchmark_result():
    """Test BenchmarkResult dataclass."""
    result = BenchmarkResult(
        benchmark_name="test",
        model_name="model",
        metrics={"accuracy": 0.95},
        num_images=100,
        total_time=10.0,
        device="cpu"
    )
    
    assert result.benchmark_name == "test"
    assert result.num_images == 100
    assert result.avg_time_per_image == 0.1


def test_benchmark_result_to_dict():
    """Test converting BenchmarkResult to dictionary."""
    result = BenchmarkResult(
        benchmark_name="test",
        model_name="model",
        metrics={"accuracy": 0.95},
        num_images=100,
        total_time=10.0,
    )
    
    result_dict = result.to_dict()
    
    assert "benchmark_name" in result_dict
    assert "metrics" in result_dict
    assert result_dict["num_images"] == 100


def test_compute_iou():
    """Test IoU computation."""
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 5, 15, 15])
    
    iou = compute_iou(box1, box2)
    
    # Boxes overlap in 5x5 region
    # Intersection = 25, Union = 100 + 100 - 25 = 175
    expected_iou = 25 / 175
    
    assert abs(iou - expected_iou) < 1e-6


def test_compute_iou_no_overlap():
    """Test IoU with non-overlapping boxes."""
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([20, 20, 30, 30])
    
    iou = compute_iou(box1, box2)
    
    assert iou == 0.0


def test_compute_iou_perfect_overlap():
    """Test IoU with identical boxes."""
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([0, 0, 10, 10])
    
    iou = compute_iou(box1, box2)
    
    assert iou == 1.0


def test_compute_map():
    """Test mAP computation."""
    # Create predictions
    pred = DetectionResult(
        boxes=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
        scores=np.array([0.9, 0.8]),
        labels=np.array([0, 1])
    )
    
    # Create ground truth
    gt = DetectionResult(
        boxes=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
        scores=np.array([1.0, 1.0]),
        labels=np.array([0, 1])
    )
    
    metrics = compute_map([pred], [gt], iou_threshold=0.5)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1


def test_benchmark_result_summary():
    """Test BenchmarkResult summary formatting."""
    result = BenchmarkResult(
        benchmark_name="test",
        model_name="model",
        metrics={"accuracy": 0.95, "precision": 0.90},
        num_images=100,
        total_time=10.0,
        device="cpu"
    )
    
    summary = result.summary()
    
    assert "test" in summary
    assert "model" in summary
    assert "accuracy" in summary
    assert "precision" in summary
