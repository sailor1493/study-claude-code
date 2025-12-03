"""VLM-based object detection using VLLM servers."""

import asyncio
import re
from typing import Union, List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

from object_detection.models.base import BaseDetector, DetectionResult
from object_detection.api.vllm_client import VLLMClient


class VLMServerDetector(BaseDetector):
    """
    VLM-based object detector that calls VLLM servers.
    
    Supports models like Qwen3VL, GPT-4V, etc. that can return
    bounding box coordinates in text format.
    """
    
    def __init__(
        self,
        model_name: str,
        vllm_server_url: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize VLM server detector.
        
        Args:
            model_name: Name of VLM model on VLLM server
            vllm_server_url: URL of VLLM server
            device: Device (not used for server-based inference, kept for compatibility)
            confidence_threshold: Minimum confidence score
            prompt_template: Custom prompt template for detection
            **kwargs: Additional parameters
        """
        super().__init__(model_name, device, confidence_threshold, **kwargs)
        self.vllm_server_url = vllm_server_url
        self.client: Optional[VLLMClient] = None
        
        # Default prompt template for object detection
        self.prompt_template = prompt_template or (
            "Detect all objects in this image. For each object, provide: "
            "object name, confidence score (0-1), and bounding box coordinates "
            "in format [x1, y1, x2, y2] where coordinates are normalized (0-1). "
            "Output format: <object_name>|<confidence>|<x1>,<y1>,<x2>,<y2>"
        )
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load VLM client (no actual model loading needed)."""
        # Create client instance (will be initialized in async context)
        self.client = VLLMClient(
            base_url=self.vllm_server_url,
            model_name=self.model_name,
        )
        self.class_names = []  # Will be populated from detections
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Preprocess image for VLM."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image
    
    def _parse_detection_response(
        self, 
        response: str,
        original_shape: Tuple[int, int]
    ) -> DetectionResult:
        """
        Parse VLM text response to extract bounding boxes.
        
        Expected format: <object_name>|<confidence>|<x1>,<y1>,<x2>,<y2>
        
        Args:
            response: Text response from VLM
            original_shape: (height, width) of original image
            
        Returns:
            DetectionResult with parsed detections
        """
        boxes = []
        scores = []
        labels = []
        class_names_set = set()
        class_name_to_id = {}
        
        # Parse each line of the response
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Try to parse the detection format
                # Format: object_name|confidence|x1,y1,x2,y2
                # Supports negative coords and scientific notation
                match = re.match(
                    r'([^|]+)\|([-0-9.e+]+)\|([-0-9.e+,\s]+)',
                    line
                )
                
                if match:
                    obj_name = match.group(1).strip()
                    confidence = float(match.group(2))
                    coords_str = match.group(3)
                    
                    # Parse coordinates
                    coords = [float(x.strip()) for x in coords_str.split(',')]
                    if len(coords) != 4:
                        continue
                    
                    x1, y1, x2, y2 = coords
                    
                    # Skip low confidence detections
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Convert normalized coordinates to absolute
                    h, w = original_shape
                    abs_x1 = x1 * w
                    abs_y1 = y1 * h
                    abs_x2 = x2 * w
                    abs_y2 = y2 * h
                    
                    # Add to class mapping
                    if obj_name not in class_name_to_id:
                        class_name_to_id[obj_name] = len(class_name_to_id)
                        class_names_set.add(obj_name)
                    
                    boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
                    scores.append(confidence)
                    labels.append(class_name_to_id[obj_name])
                
                else:
                    # Try alternative formats (JSON-like, etc.)
                    # Look for patterns like: {"class": "person", "bbox": [x1,y1,x2,y2], "score": 0.9}
                    json_match = re.search(
                        r'"class"\s*:\s*"([^"]+)".*?"bbox"\s*:\s*\[([0-9.,\s]+)\].*?"score"\s*:\s*([0-9.]+)',
                        line
                    )
                    if json_match:
                        obj_name = json_match.group(1)
                        bbox_str = json_match.group(2)
                        confidence = float(json_match.group(3))
                        
                        if confidence < self.confidence_threshold:
                            continue
                        
                        coords = [float(x.strip()) for x in bbox_str.split(',')]
                        if len(coords) == 4:
                            h, w = original_shape
                            abs_x1, abs_y1, abs_x2, abs_y2 = [
                                coords[0] * w, coords[1] * h,
                                coords[2] * w, coords[3] * h
                            ]
                            
                            if obj_name not in class_name_to_id:
                                class_name_to_id[obj_name] = len(class_name_to_id)
                                class_names_set.add(obj_name)
                            
                            boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
                            scores.append(confidence)
                            labels.append(class_name_to_id[obj_name])
            
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                # Could add logging here if needed:
                # import logging
                # logging.warning(f"Failed to parse line: {line}, error: {e}")
                continue
        
        # Convert to numpy arrays
        if boxes:
            boxes = np.array(boxes)
            scores = np.array(scores)
            labels = np.array(labels)
            class_names_list = sorted(class_names_set, key=lambda x: class_name_to_id[x])
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))
            labels = np.empty((0,), dtype=int)
            class_names_list = []
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=class_names_list,
            metadata={
                "original_shape": original_shape,
                "vllm_response": response,
                "server_url": self.vllm_server_url,
            }
        )
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> DetectionResult:
        """
        Run VLM detection via VLLM server.
        
        Args:
            image: Input image
            
        Returns:
            DetectionResult with detected objects
        """
        if isinstance(image, np.ndarray):
            original_shape = image.shape[:2]
        else:
            original_shape = (image.height, image.width)
        
        preprocessed = self.preprocess(image)
        
        # Run async prediction synchronously
        return asyncio.run(self._async_predict(preprocessed, original_shape))
    
    async def _async_predict(
        self,
        image: Image.Image,
        original_shape: Tuple[int, int]
    ) -> DetectionResult:
        """Async prediction helper."""
        async with self.client:
            # Check server health
            is_healthy = await self.client.health_check()
            if not is_healthy:
                raise RuntimeError(f"VLLM server at {self.vllm_server_url} is not healthy")
            
            # Generate detection response
            response = await self.client.generate(
                prompt=self.prompt_template,
                image=image,
                max_tokens=2048,
                temperature=0.0,
            )
            
            # Parse response
            return self._parse_detection_response(response, original_shape)
    
    async def batch_predict_async(
        self,
        images: List[Union[Image.Image, np.ndarray]]
    ) -> List[DetectionResult]:
        """
        Run detection on multiple images asynchronously.
        
        This enables efficient batch processing with VLLM's multi-GPU support.
        
        Args:
            images: List of input images
            
        Returns:
            List of DetectionResult objects
        """
        preprocessed_images = []
        original_shapes = []
        
        for image in images:
            if isinstance(image, np.ndarray):
                original_shapes.append(image.shape[:2])
                image = Image.fromarray(image)
            else:
                original_shapes.append((image.height, image.width))
            preprocessed_images.append(image)
        
        async with self.client:
            # Check server health
            is_healthy = await self.client.health_check()
            if not is_healthy:
                raise RuntimeError(f"VLLM server at {self.vllm_server_url} is not healthy")
            
            # Generate responses in parallel
            prompts = [self.prompt_template] * len(images)
            responses = await self.client.batch_generate(
                prompts=prompts,
                images=preprocessed_images,
                max_tokens=2048,
                temperature=0.0,
            )
            
            # Parse all responses
            results = []
            for response, original_shape in zip(responses, original_shapes):
                result = self._parse_detection_response(response, original_shape)
                results.append(result)
            
            return results
    
    def batch_predict(
        self,
        images: List[Union[Image.Image, np.ndarray]]
    ) -> List[DetectionResult]:
        """
        Run detection on multiple images (synchronous wrapper).
        
        Args:
            images: List of input images
            
        Returns:
            List of DetectionResult objects
        """
        return asyncio.run(self.batch_predict_async(images))
    
    def postprocess(
        self,
        outputs: Union[str, Dict[str, Any]],
        original_shape: tuple
    ) -> DetectionResult:
        """
        Postprocess VLM outputs.
        
        Args:
            outputs: Text response or dict from VLM
            original_shape: Original image shape
            
        Returns:
            DetectionResult
        """
        if isinstance(outputs, str):
            return self._parse_detection_response(outputs, original_shape)
        else:
            # Handle dict format if needed
            return self._parse_detection_response(str(outputs), original_shape)
