"""Example: VLM-based object detection using VLLM server.

Note: For proper usage, install the package first:
  pip install -e .

This example uses sys.path for demonstration purposes only.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import asyncio

# Add src to path (for demo purposes only - install package for production)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from object_detection.models.vlm_server import VLMServerDetector
from object_detection.utils.visualization import visualize_detections


def main():
    """Run VLM server detection example."""
    print("="*70)
    print("VLM Server Detection Example")
    print("="*70)
    
    # Configuration
    VLLM_SERVER_URL = "http://localhost:8000"  # Your VLLM server URL
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # Or GPT-4V, etc.
    
    print(f"\nConnecting to VLLM server: {VLLM_SERVER_URL}")
    print(f"Model: {MODEL_NAME}")
    
    try:
        # Create detector
        detector = VLMServerDetector(
            model_name=MODEL_NAME,
            vllm_server_url=VLLM_SERVER_URL,
            confidence_threshold=0.5,
        )
        
        print(f"\n✓ Detector initialized")
        
        # Create test image
        print("\nCreating test image...")
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # Run detection
        print("Running detection (this will call VLLM server)...")
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
            output_path = Path("/tmp/vlm_detection_example.jpg")
            viz_image.save(output_path)
            print(f"Saved visualization to {output_path}")
        else:
            print("  No objects detected")
        
        # Example: Batch processing with async
        print("\n" + "="*70)
        print("Batch Processing Example")
        print("="*70)
        
        # Create multiple test images
        test_images = [
            Image.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8))
            for _ in range(3)
        ]
        
        print(f"\nProcessing {len(test_images)} images in parallel...")
        results = detector.batch_predict(test_images)
        
        for i, result in enumerate(results):
            print(f"  Image {i+1}: {len(result.boxes)} detections")
        
        print("\n" + "="*70)
        print("Example completed successfully!")
        print("="*70)
    
    except RuntimeError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("1. VLLM server is running at the specified URL")
        print("2. The model supports vision inputs")
        print("3. Server is healthy and responding")
        print("\nTo start a VLLM server, use:")
        print(f"  python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model {MODEL_NAME} \\")
        print(f"    --port 8000")


async def async_example():
    """Async example with explicit async context."""
    print("\n" + "="*70)
    print("Async Example (Advanced)")
    print("="*70)
    
    VLLM_SERVER_URL = "http://localhost:8000"
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    
    detector = VLMServerDetector(
        model_name=MODEL_NAME,
        vllm_server_url=VLLM_SERVER_URL,
        confidence_threshold=0.5,
    )
    
    # Create test images
    test_images = [
        Image.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    
    print(f"Processing {len(test_images)} images with async API...")
    
    # Use async batch predict
    results = await detector.batch_predict_async(test_images)
    
    total_detections = sum(len(r.boxes) for r in results)
    print(f"Total detections across all images: {total_detections}")
    
    print("Async example completed!")


if __name__ == "__main__":
    main()
    
    # Uncomment to run async example
    # asyncio.run(async_example())
