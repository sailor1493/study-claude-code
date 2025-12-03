"""Base classes for object detection models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    """Result of object detection."""
    
    boxes: np.ndarray  # Shape: [N, 4] - (x1, y1, x2, y2)
    scores: np.ndarray  # Shape: [N]
    labels: np.ndarray  # Shape: [N]
    class_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate detection result."""
        assert len(self.boxes) == len(self.scores) == len(self.labels), \
            "Boxes, scores, and labels must have the same length"
        assert self.boxes.shape[1] == 4, "Boxes must have shape [N, 4]"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "labels": self.labels.tolist(),
            "class_names": self.class_names,
            "metadata": self.metadata,
        }
    
    def filter_by_score(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence score."""
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=self.labels[mask],
            class_names=self.class_names,
            metadata=self.metadata,
        )


class BaseDetector(ABC):
    """Base class for all object detection models."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize detector.
        
        Args:
            model_name: Name/path of the model
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
            confidence_threshold: Minimum confidence score for detections
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Get appropriate device."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass
    
    @abstractmethod
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input."""
        pass
    
    @abstractmethod
    def predict(self, image: Union[Image.Image, np.ndarray]) -> DetectionResult:
        """
        Run object detection on an image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            DetectionResult containing boxes, scores, and labels
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs: Any, original_shape: tuple) -> DetectionResult:
        """Postprocess model outputs to DetectionResult."""
        pass
    
    def batch_predict(
        self, 
        images: List[Union[Image.Image, np.ndarray]]
    ) -> List[DetectionResult]:
        """
        Run object detection on multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of DetectionResult objects
        """
        return [self.predict(img) for img in images]
    
    def to(self, device: str) -> "BaseDetector":
        """Move model to specified device."""
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "confidence_threshold": self.confidence_threshold,
            "num_classes": len(self.class_names),
        }
