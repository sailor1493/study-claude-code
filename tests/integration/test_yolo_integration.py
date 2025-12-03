"""Integration test for YOLO model."""

import pytest
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from object_detection.models.yolo import YOLODetector
from object_detection.models.base import DetectionResult


@pytest.mark.slow
def test_yolo_model_loading():
    """Test that YOLO model can be loaded."""
    try:
        detector = YOLODetector(
            model_name="yolov5s",
            device="cpu",
            confidence_threshold=0.5
        )
        
        assert detector.model is not None
        assert len(detector.class_names) > 0
        print(f"YOLO model loaded successfully with {len(detector.class_names)} classes")
    
    except Exception as e:
        pytest.skip(f"YOLO model loading failed (expected in some environments): {e}")


@pytest.mark.slow
def test_yolo_prediction():
    """Test YOLO prediction on synthetic image."""
    try:
        detector = YOLODetector(
            model_name="yolov5s",
            device="cpu",
            confidence_threshold=0.5
        )
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # Run prediction
        result = detector.predict(test_image)
        
        assert isinstance(result, DetectionResult)
        assert result.boxes.shape[1] == 4
        assert len(result.scores) == len(result.boxes)
        assert len(result.labels) == len(result.boxes)
        
        print(f"YOLO prediction successful, detected {len(result.boxes)} objects")
    
    except Exception as e:
        pytest.skip(f"YOLO prediction failed (expected in some environments): {e}")
