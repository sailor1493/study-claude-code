"""Verify basic framework structure and imports."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_imports():
    """Verify all modules can be imported."""
    print("Verifying imports...")
    
    try:
        from object_detection.models.base import BaseDetector, DetectionResult
        print("✓ Base models imported")
        
        from object_detection.models.yolo import YOLODetector
        print("✓ YOLO model imported")
        
        from object_detection.models.detr import DETRDetector
        print("✓ DETR model imported")
        
        from object_detection.models.vlm import VLMDetector
        print("✓ VLM model imported")
        
        from object_detection.benchmarks.base import BaseBenchmark, BenchmarkResult
        print("✓ Base benchmarks imported")
        
        from object_detection.benchmarks.performance import PerformanceBenchmark
        print("✓ Performance benchmark imported")
        
        from object_detection.benchmarks.accuracy import AccuracyBenchmark
        print("✓ Accuracy benchmark imported")
        
        from object_detection.api.server import DetectionAPI, create_app
        print("✓ API server imported")
        
        from object_detection.utils.device import get_device, get_device_info
        print("✓ Device utilities imported")
        
        from object_detection.utils.visualization import visualize_detections
        print("✓ Visualization utilities imported")
        
        from object_detection.configs.config import Config, load_config
        print("✓ Configuration imported")
        
        return True
    
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_structure():
    """Verify project structure."""
    print("\nVerifying project structure...")
    
    base_path = Path(__file__).parent
    
    required_paths = [
        "src/object_detection",
        "src/object_detection/models",
        "src/object_detection/benchmarks",
        "src/object_detection/api",
        "src/object_detection/utils",
        "src/object_detection/configs",
        "examples",
        "tests/unit",
        "tests/integration",
        "requirements.txt",
        "setup.py",
        "README.md",
    ]
    
    all_present = True
    for path_str in required_paths:
        path = base_path / path_str
        if path.exists():
            print(f"✓ {path_str}")
        else:
            print(f"✗ {path_str} missing")
            all_present = False
    
    return all_present


def main():
    """Main verification function."""
    print("="*60)
    print("Object Detection Framework Verification")
    print("="*60)
    
    imports_ok = verify_imports()
    structure_ok = verify_structure()
    
    print("\n" + "="*60)
    if imports_ok and structure_ok:
        print("✓ All verifications passed!")
        print("="*60)
        return 0
    else:
        print("✗ Some verifications failed")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
