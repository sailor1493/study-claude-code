"""Test utility functions."""

import pytest
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from object_detection.utils.device import get_device, get_device_info
from object_detection.utils.visualization import visualize_detections
from object_detection.models.base import DetectionResult


def test_get_device():
    """Test device selection utility."""
    device = get_device()
    assert device is not None
    
    device = get_device("cpu")
    assert str(device) == "cpu"


def test_get_device_info():
    """Test device info utility."""
    info = get_device_info()
    
    assert "cuda_available" in info
    assert "num_gpus" in info
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["num_gpus"], int)


def test_visualize_detections():
    """Test detection visualization."""
    # Create test image
    img = Image.new('RGB', (100, 100), color='white')
    
    # Create test detections
    result = DetectionResult(
        boxes=np.array([[10, 10, 30, 30], [50, 50, 80, 80]]),
        scores=np.array([0.9, 0.8]),
        labels=np.array([0, 1]),
        class_names=["cat", "dog"]
    )
    
    # Visualize
    viz_img = visualize_detections(img, result)
    
    assert isinstance(viz_img, Image.Image)
    assert viz_img.size == img.size


def test_visualize_detections_numpy():
    """Test visualization with numpy array input."""
    # Create test image as numpy array
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # Create test detections
    result = DetectionResult(
        boxes=np.array([[10, 10, 30, 30]]),
        scores=np.array([0.9]),
        labels=np.array([0]),
        class_names=["cat"]
    )
    
    # Visualize
    viz_img = visualize_detections(img, result)
    
    assert isinstance(viz_img, Image.Image)


def test_visualize_empty_detections():
    """Test visualization with no detections."""
    img = Image.new('RGB', (100, 100), color='white')
    
    result = DetectionResult(
        boxes=np.empty((0, 4)),
        scores=np.empty((0,)),
        labels=np.empty((0,), dtype=int)
    )
    
    viz_img = visualize_detections(img, result)
    
    assert isinstance(viz_img, Image.Image)
    assert viz_img.size == img.size
