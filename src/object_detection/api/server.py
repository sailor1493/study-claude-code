"""FastAPI server for object detection with VLLM compatibility."""

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import torch

from object_detection.models.base import BaseDetector
from object_detection.models.yolo import YOLODetector
from object_detection.models.detr import DETRDetector
from object_detection.models.vlm import VLMDetector


class DetectionRequest(BaseModel):
    """Request model for object detection."""
    
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    confidence_threshold: float = Field(0.5, description="Confidence threshold", ge=0.0, le=1.0)
    model_name: Optional[str] = Field(None, description="Specific model to use")


class DetectionResponse(BaseModel):
    """Response model for object detection."""
    
    boxes: List[List[float]] = Field(description="Bounding boxes [x1, y1, x2, y2]")
    scores: List[float] = Field(description="Confidence scores")
    labels: List[int] = Field(description="Class labels")
    class_names: Optional[List[str]] = Field(None, description="Class names")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    device: str
    confidence_threshold: float
    num_classes: int
    available: bool = True


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    models_loaded: int
    device: str
    gpu_available: bool


class DetectionAPI:
    """FastAPI application for object detection."""
    
    def __init__(
        self,
        default_model: str = "yolov5s",
        device: Optional[str] = None,
        enable_gpu: bool = True,
    ):
        """
        Initialize Detection API.
        
        Args:
            default_model: Default model to load
            device: Device to run on
            enable_gpu: Whether to enable GPU
        """
        self.app = FastAPI(
            title="Object Detection API",
            description="API for object detection with support for multiple models",
            version="0.1.0",
        )
        
        # Determine device
        if device is None:
            if enable_gpu and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model registry
        self.models: Dict[str, BaseDetector] = {}
        self.default_model_name = default_model
        
        # Load default model
        self._load_model(default_model)
        
        # Setup routes
        self._setup_routes()
    
    def _load_model(self, model_name: str) -> BaseDetector:
        """Load a model by name."""
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            if model_name.startswith("yolo"):
                detector = YOLODetector(
                    model_name=model_name,
                    device=self.device,
                )
            elif model_name.startswith("detr"):
                detector = DETRDetector(
                    model_name=model_name,
                    device=self.device,
                )
            elif model_name.startswith("vlm") or model_name.startswith("owl"):
                detector = VLMDetector(
                    model_name=model_name,
                    device=self.device,
                )
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            self.models[model_name] = detector
            return detector
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                models_loaded=len(self.models),
                device=self.device,
                gpu_available=torch.cuda.is_available(),
            )
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List available models."""
            model_infos = []
            for name, detector in self.models.items():
                info = detector.get_info()
                model_infos.append(ModelInfo(
                    model_name=info["model_name"],
                    device=info["device"],
                    confidence_threshold=info["confidence_threshold"],
                    num_classes=info["num_classes"],
                ))
            return model_infos
        
        @self.app.post("/detect", response_model=DetectionResponse)
        async def detect_objects(
            file: UploadFile = File(...),
            confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
            model_name: Optional[str] = Query(None),
        ):
            """
            Detect objects in an image.
            
            Args:
                file: Image file
                confidence_threshold: Confidence threshold for detections
                model_name: Model to use (default: loaded model)
                
            Returns:
                DetectionResponse with bounding boxes, scores, and labels
            """
            try:
                # Load image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                
                # Get detector
                model_name = model_name or self.default_model_name
                if model_name not in self.models:
                    self._load_model(model_name)
                detector = self.models[model_name]
                
                # Run detection - filter results after prediction to avoid thread safety issues
                result = detector.predict(image)
                
                # Apply confidence threshold filtering
                if confidence_threshold != detector.confidence_threshold:
                    result = result.filter_by_score(confidence_threshold)
                
                return DetectionResponse(
                    boxes=result.boxes.tolist(),
                    scores=result.scores.tolist(),
                    labels=result.labels.tolist(),
                    class_names=result.class_names,
                    metadata=result.metadata,
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/detect", response_model=DetectionResponse)
        async def detect_objects_v1(
            file: UploadFile = File(...),
            confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
            model_name: Optional[str] = Query(None),
        ):
            """
            VLLM-compatible detection endpoint (v1 API).
            
            This endpoint follows VLLM API conventions for compatibility.
            """
            return await detect_objects(file, confidence_threshold, model_name)
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Object Detection API",
                "version": "0.1.0",
                "models_loaded": len(self.models),
                "device": self.device,
            }


def create_app(
    model_name: str = "yolov5s",
    device: Optional[str] = None,
    enable_gpu: bool = True,
) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_name: Default model to load
        device: Device to run on
        enable_gpu: Whether to enable GPU
        
    Returns:
        FastAPI application
    """
    api = DetectionAPI(
        default_model=model_name,
        device=device,
        enable_gpu=enable_gpu,
    )
    return api.app
