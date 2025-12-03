"""Test base detector functionality."""

import pytest
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from object_detection.models.base import BaseDetector, DetectionResult


def test_detection_result():
    """Test DetectionResult dataclass."""
    boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    scores = np.array([0.9, 0.8])
    labels = np.array([0, 1])
    
    result = DetectionResult(
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=["cat", "dog"]
    )
    
    assert len(result.boxes) == 2
    assert len(result.scores) == 2
    assert len(result.labels) == 2
    assert result.class_names == ["cat", "dog"]


def test_detection_result_validation():
    """Test DetectionResult validation."""
    with pytest.raises(AssertionError):
        # Mismatched lengths
        DetectionResult(
            boxes=np.array([[10, 20, 30, 40]]),
            scores=np.array([0.9, 0.8]),
            labels=np.array([0])
        )
    
    with pytest.raises(AssertionError):
        # Wrong box shape
        DetectionResult(
            boxes=np.array([[10, 20, 30]]),  # Only 3 values
            scores=np.array([0.9]),
            labels=np.array([0])
        )


def test_detection_result_filter():
    """Test filtering detections by score."""
    boxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])
    scores = np.array([0.9, 0.4, 0.7])
    labels = np.array([0, 1, 2])
    
    result = DetectionResult(boxes=boxes, scores=scores, labels=labels)
    filtered = result.filter_by_score(0.5)
    
    assert len(filtered.boxes) == 2
    assert len(filtered.scores) == 2
    assert filtered.scores[0] >= 0.5
    assert filtered.scores[1] >= 0.5


def test_detection_result_to_dict():
    """Test converting DetectionResult to dictionary."""
    boxes = np.array([[10, 20, 30, 40]])
    scores = np.array([0.9])
    labels = np.array([0])
    
    result = DetectionResult(
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=["cat"]
    )
    
    result_dict = result.to_dict()
    
    assert "boxes" in result_dict
    assert "scores" in result_dict
    assert "labels" in result_dict
    assert "class_names" in result_dict
    assert isinstance(result_dict["boxes"], list)


def test_base_detector_device():
    """Test device selection in BaseDetector."""
    
    class MockDetector(BaseDetector):
        def load_model(self):
            pass
        
        def preprocess(self, image):
            pass
        
        def predict(self, image):
            pass
        
        def postprocess(self, outputs, original_shape):
            pass
    
    # Test auto device selection
    detector = MockDetector("test_model")
    assert detector.device is not None
    
    # Test explicit device
    detector = MockDetector("test_model", device="cpu")
    assert str(detector.device) == "cpu"


def test_base_detector_info():
    """Test getting detector information."""
    
    class MockDetector(BaseDetector):
        def load_model(self):
            self.class_names = ["class1", "class2"]
        
        def preprocess(self, image):
            pass
        
        def predict(self, image):
            pass
        
        def postprocess(self, outputs, original_shape):
            pass
    
    detector = MockDetector("test_model", confidence_threshold=0.7)
    detector.load_model()
    
    info = detector.get_info()
    
    assert info["model_name"] == "test_model"
    assert info["confidence_threshold"] == 0.7
    assert info["num_classes"] == 2
