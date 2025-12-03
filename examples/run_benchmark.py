"""Example: Running benchmarks."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from object_detection.models.yolo import YOLODetector
from object_detection.benchmarks.performance import PerformanceBenchmark
from object_detection.benchmarks.accuracy import AccuracyBenchmark


def main():
    """Run benchmark example."""
    print("="*60)
    print("Object Detection Benchmark Example")
    print("="*60)
    
    # Create detector
    print("\nLoading YOLO model...")
    detector = YOLODetector(
        model_name="yolov5s",
        confidence_threshold=0.5,
    )
    print(f"Model loaded on {detector.device}")
    
    # Run performance benchmark
    print("\n" + "="*60)
    print("Running Performance Benchmark")
    print("="*60)
    
    perf_benchmark = PerformanceBenchmark(
        detector=detector,
        num_images=50,
        image_size=(640, 480),
    )
    
    print("Running benchmark (this may take a moment)...")
    perf_result = perf_benchmark.run()
    
    print(perf_result.summary())
    
    # Run accuracy benchmark
    print("\n" + "="*60)
    print("Running Accuracy Benchmark")
    print("="*60)
    
    acc_benchmark = AccuracyBenchmark(detector=detector)
    
    print("Running benchmark (this may take a moment)...")
    acc_result = acc_benchmark.run()
    
    print(acc_result.summary())
    
    print("\n" + "="*60)
    print("Benchmark completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
