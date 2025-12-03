"""DETR (Detection Transformer) model implementation."""

from typing import Union, Optional
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from object_detection.models.base import BaseDetector, DetectionResult


class DETRDetector(BaseDetector):
    """DETR detector using pretrained models from torchvision."""
    
    COCO_CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(
        self,
        model_name: str = "detr_resnet50",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize DETR detector.
        
        Args:
            model_name: DETR model variant
            device: Device to run on
            confidence_threshold: Minimum confidence score
        """
        super().__init__(model_name, device, confidence_threshold, **kwargs)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = self.COCO_CLASSES
        self.load_model()
    
    def load_model(self) -> None:
        """Load DETR model from torchvision."""
        try:
            if self.model_name == "detr_resnet50":
                self.model = torchvision.models.detection.detr_resnet50(
                    weights="DEFAULT"
                )
            elif self.model_name == "detr_resnet101":
                self.model = torchvision.models.detection.detr_resnet101(
                    weights="DEFAULT"
                )
            else:
                raise ValueError(f"Unknown DETR model: {self.model_name}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DETR model {self.model_name}: {e}")
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for DETR."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> DetectionResult:
        """Run DETR detection."""
        if isinstance(image, np.ndarray):
            original_shape = image.shape[:2]
        else:
            original_shape = (image.height, image.width)
        
        preprocessed = self.preprocess(image)
        
        with torch.no_grad():
            outputs = self.model(preprocessed)
        
        return self.postprocess(outputs, original_shape)
    
    def postprocess(self, outputs, original_shape: tuple) -> DetectionResult:
        """Convert DETR outputs to DetectionResult."""
        # Get predictions
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        # Get probabilities and labels
        probas = pred_logits.softmax(-1)[:, :-1]  # Exclude no-object class
        scores, labels = probas.max(-1)
        
        # Convert to numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        boxes = pred_boxes.cpu().numpy()
        
        # Filter by confidence
        mask = scores >= self.confidence_threshold
        scores = scores[mask]
        labels = labels[mask]
        boxes = boxes[mask]
        
        # Convert from [cx, cy, w, h] (normalized) to [x1, y1, x2, y2] (absolute)
        h, w = original_shape
        boxes = boxes * np.array([w, h, w, h])
        boxes[:, 0] -= boxes[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] -= boxes[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] += boxes[:, 0]      # x2 = x1 + w
        boxes[:, 3] += boxes[:, 1]      # y2 = y1 + h
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=self.class_names,
            metadata={"original_shape": original_shape}
        )
