"""Example: Basic object detection with YOLO."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from object_detection.models.yolo import YOLODetector
from object_detection.utils.visualization import visualize_detections
from object_detection.utils.device import print_device_info


def main():
    """Run basic detection example."""
    print("="*60)
    print("Object Detection Example - YOLO")
    print("="*60)
    
    # Print device info
    print_device_info()
    
    # Create detector
    print("Loading YOLO model...")
    detector = YOLODetector(
        model_name="yolov5s",
        confidence_threshold=0.5,
    )
    print(f"Model loaded on {detector.device}")
    print(f"Number of classes: {len(detector.class_names)}")
    
    # Create a test image
    print("\nCreating test image...")
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_image)
    
    # Run detection
    print("Running detection...")
    result = detector.predict(test_image)
    
    print(f"\nDetection Results:")
    print(f"  Number of detections: {len(result.boxes)}")
    if len(result.boxes) > 0:
        print(f"  Confidence scores: {result.scores}")
        print(f"  Labels: {result.labels}")
        if result.class_names:
            print(f"  Classes detected: {[result.class_names[l] for l in result.labels]}")
    
    # Visualize results
    print("\nVisualizing results...")
    viz_image = visualize_detections(test_image, result)
    
    # Save visualization
    output_path = Path("/tmp/detection_example.jpg")
    viz_image.save(output_path)
    print(f"Saved visualization to {output_path}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
