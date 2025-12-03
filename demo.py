"""Minimal demo showing framework capabilities without large model downloads."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from object_detection.models.base import BaseDetector, DetectionResult
from object_detection.utils.device import print_device_info, get_device_info
from object_detection.utils.visualization import visualize_detections
from object_detection.benchmarks.base import BenchmarkResult


class MockDetector(BaseDetector):
    """Mock detector for demonstration purposes."""
    
    def load_model(self):
        """Load mock model."""
        self.class_names = ["person", "car", "dog", "cat", "bicycle"]
        print(f"Mock model loaded with {len(self.class_names)} classes")
    
    def preprocess(self, image):
        """Mock preprocessing."""
        return image
    
    def predict(self, image):
        """Mock prediction - generates random detections."""
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.height, image.width
        
        # Generate 2-5 random detections
        num_detections = np.random.randint(2, 6)
        
        boxes = []
        scores = []
        labels = []
        
        for _ in range(num_detections):
            # Random box
            x1 = np.random.randint(0, w - 50)
            y1 = np.random.randint(0, h - 50)
            x2 = x1 + np.random.randint(30, min(100, w - x1))
            y2 = y1 + np.random.randint(30, min(100, h - y1))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(np.random.uniform(0.5, 0.95))
            labels.append(np.random.randint(0, len(self.class_names)))
        
        return DetectionResult(
            boxes=np.array(boxes),
            scores=np.array(scores),
            labels=np.array(labels),
            class_names=self.class_names,
            metadata={"original_shape": (h, w)}
        )
    
    def postprocess(self, outputs, original_shape):
        """Mock postprocessing."""
        pass


def main():
    """Run minimal demo."""
    print("="*70)
    print("Object Detection Framework - Minimal Demo")
    print("="*70)
    
    # Show device info
    print("\n1. Device Information")
    print("-" * 70)
    device_info = get_device_info()
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"Number of GPUs: {device_info['num_gpus']}")
    print(f"Device: {device_info.get('current_device', 'CPU')}")
    
    # Create mock detector
    print("\n2. Creating Detector")
    print("-" * 70)
    detector = MockDetector(
        model_name="mock_detector",
        device="cpu",
        confidence_threshold=0.5
    )
    detector.load_model()
    
    info = detector.get_info()
    print(f"Model: {info['model_name']}")
    print(f"Device: {info['device']}")
    print(f"Confidence Threshold: {info['confidence_threshold']}")
    print(f"Number of Classes: {info['num_classes']}")
    
    # Create test images
    print("\n3. Running Detection on Test Images")
    print("-" * 70)
    
    test_images = []
    for i in range(3):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(img))
    
    results = []
    for i, img in enumerate(test_images):
        result = detector.predict(img)
        results.append(result)
        print(f"Image {i+1}: Detected {len(result.boxes)} objects")
        print(f"  Classes: {[result.class_names[l] for l in result.labels]}")
        print(f"  Scores: {[f'{s:.2f}' for s in result.scores]}")
    
    # Test filtering
    print("\n4. Testing Score Filtering")
    print("-" * 70)
    result = results[0]
    print(f"Original detections: {len(result.boxes)}")
    filtered = result.filter_by_score(0.8)
    print(f"After filtering (threshold=0.8): {len(filtered.boxes)}")
    
    # Test visualization
    print("\n5. Testing Visualization")
    print("-" * 70)
    viz_image = visualize_detections(test_images[0], results[0])
    output_path = Path("/tmp/demo_detection.jpg")
    viz_image.save(output_path)
    print(f"Saved visualization to: {output_path}")
    print(f"Visualization size: {viz_image.size}")
    
    # Test to_dict conversion
    print("\n6. Testing Result Serialization")
    print("-" * 70)
    result_dict = results[0].to_dict()
    print(f"Result converted to dict with keys: {list(result_dict.keys())}")
    
    # Demonstrate benchmark result
    print("\n7. Mock Benchmark Result")
    print("-" * 70)
    benchmark_result = BenchmarkResult(
        benchmark_name="Demo Benchmark",
        model_name="mock_detector",
        metrics={
            "mAP@0.5": 0.87,
            "precision": 0.92,
            "recall": 0.85,
            "f1_score": 0.88,
        },
        num_images=len(test_images),
        total_time=0.5,
        device="cpu"
    )
    print(benchmark_result.summary())
    
    # Framework features summary
    print("\n8. Framework Features Summary")
    print("-" * 70)
    print("✓ Modular architecture with base classes")
    print("✓ Support for multiple model types (YOLO, DETR, VLM)")
    print("✓ GPU/CPU device management")
    print("✓ Comprehensive benchmarking tools")
    print("✓ REST API with FastAPI (VLLM-compatible)")
    print("✓ Visualization utilities")
    print("✓ Configuration management")
    print("✓ Extensive test coverage")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  - Run 'python examples/basic_detection.py' for YOLO example")
    print("  - Run 'python examples/run_benchmark.py' for benchmarking")
    print("  - Run 'python examples/run_server.py' to start API server")
    print("  - Run 'pytest tests/' to execute test suite")
    print("="*70)


if __name__ == "__main__":
    main()
