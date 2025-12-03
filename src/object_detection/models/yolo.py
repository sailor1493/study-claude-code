"""YOLO model implementation using torchvision."""

from typing import Union, List, Optional
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from object_detection.models.base import BaseDetector, DetectionResult


class YOLODetector(BaseDetector):
    """YOLO detector using pretrained models from torchvision or ultralytics."""
    
    def __init__(
        self,
        model_name: str = "yolov5s",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLO model variant (e.g., 'yolov5s', 'yolov5m')
            device: Device to run on
            confidence_threshold: Minimum confidence score
        """
        super().__init__(model_name, device, confidence_threshold, **kwargs)
        self.load_model()
    
    def load_model(self) -> None:
        """Load YOLO model."""
        try:
            # Try loading from torch hub (ultralytics)
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                self.model_name,
                pretrained=True,
                trust_repo=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.class_names = self.model.names if hasattr(self.model, 'names') else []
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model {self.model_name}: {e}")
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for YOLO."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # YOLO models from torch hub handle preprocessing internally
        return image
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> DetectionResult:
        """Run YOLO detection."""
        if isinstance(image, np.ndarray):
            original_shape = image.shape[:2]
            image = Image.fromarray(image)
        else:
            original_shape = (image.height, image.width)
        
        with torch.no_grad():
            results = self.model(image)
        
        return self.postprocess(results, original_shape)
    
    def postprocess(self, outputs, original_shape: tuple) -> DetectionResult:
        """Convert YOLO outputs to DetectionResult."""
        # YOLOv5 results object
        predictions = outputs.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        
        if len(predictions) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4)),
                scores=np.empty((0,)),
                labels=np.empty((0,), dtype=int),
                class_names=self.class_names,
            )
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        labels = predictions[:, 5].astype(int)
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        
        return DetectionResult(
            boxes=boxes[mask],
            scores=scores[mask],
            labels=labels[mask],
            class_names=self.class_names,
            metadata={"original_shape": original_shape}
        )
