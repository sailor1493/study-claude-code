# Object Detection Framework

A modular, extensible framework for object detection supporting multiple models, benchmarks, and deployment options including GPU acceleration and VLLM-compatible API servers.

## Features

- **Multiple Model Support**: YOLO, DETR, Vision-Language Models (VLM)
- **GPU Acceleration**: Automatic device detection and GPU support
- **Modular Architecture**: Easy to extend with new models and benchmarks
- **Comprehensive Benchmarking**: Performance and accuracy evaluation
- **REST API**: FastAPI-based server with VLLM compatibility
- **Visualization Tools**: Built-in detection visualization utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/sailor1493/study-claude-code.git
cd study-claude-code

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

For development with testing tools:
```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Detection

```python
from object_detection.models.yolo import YOLODetector
from PIL import Image

# Create detector
detector = YOLODetector(model_name="yolov5s", confidence_threshold=0.5)

# Load and process image
image = Image.open("test.jpg")
result = detector.predict(image)

# Access results
print(f"Found {len(result.boxes)} objects")
print(f"Boxes: {result.boxes}")
print(f"Scores: {result.scores}")
print(f"Labels: {result.labels}")
```

### Running Benchmarks

```python
from object_detection.models.yolo import YOLODetector
from object_detection.benchmarks.performance import PerformanceBenchmark

# Create detector
detector = YOLODetector(model_name="yolov5s")

# Run performance benchmark
benchmark = PerformanceBenchmark(detector, num_images=100)
result = benchmark.run()

# Display results
print(result.summary())
```

### API Server

Start the API server:

```bash
python examples/run_server.py --model yolov5s --port 8000
```

Or with GPU:
```bash
python examples/run_server.py --model yolov5s --device cuda --port 8000
```

Access API documentation at `http://localhost:8000/docs`

## Supported Models

### YOLO (You Only Look Once)
```python
from object_detection.models.yolo import YOLODetector

detector = YOLODetector(
    model_name="yolov5s",  # or yolov5m, yolov5l, yolov5x
    device="cuda",
    confidence_threshold=0.5
)
```

### DETR (DEtection TRansformer)
```python
from object_detection.models.detr import DETRDetector

detector = DETRDetector(
    model_name="detr_resnet50",  # or detr_resnet101
    device="cuda",
    confidence_threshold=0.7
)
```

### Vision Language Models (VLM)
```python
from object_detection.models.vlm import VLMDetector

detector = VLMDetector(
    model_name="owlvit-base",
    text_queries=["person", "car", "dog"],
    device="cuda"
)
```

## Benchmarking

### Performance Benchmark
Measures inference speed and throughput:

```python
from object_detection.benchmarks.performance import PerformanceBenchmark

benchmark = PerformanceBenchmark(
    detector=detector,
    num_images=100,
    image_size=(640, 480)
)
result = benchmark.run()
```

### Accuracy Benchmark
Evaluates detection accuracy with metrics like mAP:

```python
from object_detection.benchmarks.accuracy import AccuracyBenchmark

benchmark = AccuracyBenchmark(
    detector=detector,
    data_path="path/to/test/images",
    ground_truths=ground_truth_detections
)
result = benchmark.run()
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### List Models
```bash
curl http://localhost:8000/models
```

### Detect Objects
```bash
curl -X POST "http://localhost:8000/detect?confidence_threshold=0.5" \
  -F "file=@image.jpg"
```

### VLLM-Compatible Endpoint
```bash
curl -X POST "http://localhost:8000/v1/detect?confidence_threshold=0.5" \
  -F "file=@image.jpg"
```

## GPU Support

The framework automatically detects and uses GPU if available. You can also explicitly specify the device:

```python
detector = YOLODetector(model_name="yolov5s", device="cuda:0")
```

Check available devices:
```python
from object_detection.utils.device import print_device_info

print_device_info()
```

## Project Structure

```
study-claude-code/
├── src/
│   └── object_detection/
│       ├── models/          # Detection models
│       │   ├── base.py      # Base detector class
│       │   ├── yolo.py      # YOLO implementation
│       │   ├── detr.py      # DETR implementation
│       │   └── vlm.py       # VLM implementation
│       ├── benchmarks/      # Benchmarking tools
│       │   ├── base.py      # Base benchmark class
│       │   ├── performance.py
│       │   └── accuracy.py
│       ├── api/             # REST API
│       │   └── server.py    # FastAPI server
│       ├── utils/           # Utilities
│       │   ├── device.py    # Device management
│       │   └── visualization.py
│       └── configs/         # Configuration management
├── examples/                # Example scripts
│   ├── basic_detection.py
│   ├── run_benchmark.py
│   └── run_server.py
├── tests/                   # Test suite
│   ├── unit/
│   └── integration/
├── requirements.txt
├── requirements-dev.txt
└── setup.py
```

## Examples

Run the provided examples:

```bash
# Basic detection example
python examples/basic_detection.py

# Run benchmarks
python examples/run_benchmark.py

# Start API server
python examples/run_server.py --model yolov5s --port 8000
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

## Extending the Framework

### Adding a New Model

1. Create a new model class inheriting from `BaseDetector`:

```python
from object_detection.models.base import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    def load_model(self):
        # Load your model
        pass
    
    def preprocess(self, image):
        # Preprocess image
        pass
    
    def predict(self, image):
        # Run inference
        pass
    
    def postprocess(self, outputs, original_shape):
        # Convert to DetectionResult
        pass
```

### Adding a New Benchmark

1. Create a new benchmark class inheriting from `BaseBenchmark`:

```python
from object_detection.benchmarks.base import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    def load_data(self):
        # Load benchmark data
        pass
    
    def evaluate(self, predictions):
        # Compute metrics
        pass
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- YOLOv5 by Ultralytics
- DETR by Facebook AI Research
- PyTorch and torchvision teams
