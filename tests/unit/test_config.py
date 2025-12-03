"""Test configuration management."""

import pytest
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from object_detection.configs.config import (
    Config,
    ModelConfig,
    BenchmarkConfig,
    APIConfig,
    load_config
)


def test_model_config():
    """Test ModelConfig dataclass."""
    config = ModelConfig(
        name="yolo",
        model_type="yolo",
        device="cuda",
        confidence_threshold=0.5
    )
    
    assert config.name == "yolo"
    assert config.model_type == "yolo"
    assert config.device == "cuda"


def test_benchmark_config():
    """Test BenchmarkConfig dataclass."""
    config = BenchmarkConfig(
        name="perf",
        benchmark_type="performance",
        params={"num_images": 100}
    )
    
    assert config.name == "perf"
    assert config.benchmark_type == "performance"
    assert config.params["num_images"] == 100


def test_api_config():
    """Test APIConfig dataclass."""
    config = APIConfig(
        host="0.0.0.0",
        port=8000,
        default_model="yolov5s"
    )
    
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.default_model == "yolov5s"


def test_config_creation():
    """Test Config object creation."""
    config = Config()
    
    assert isinstance(config.models, dict)
    assert isinstance(config.benchmarks, dict)
    assert isinstance(config.api, APIConfig)


def test_config_yaml_roundtrip():
    """Test saving and loading config to/from YAML."""
    # Create config
    config = Config()
    config.models["yolo"] = ModelConfig(
        name="yolo",
        model_type="yolo",
        device="cuda"
    )
    config.api.port = 9000
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        config.to_yaml(temp_path)
        
        # Load back
        loaded_config = Config.from_yaml(temp_path)
        
        assert "yolo" in loaded_config.models
        assert loaded_config.models["yolo"].model_type == "yolo"
        assert loaded_config.api.port == 9000
    
    finally:
        temp_path.unlink()


def test_load_config_nonexistent():
    """Test loading config from nonexistent file."""
    config = load_config(Path("/nonexistent/path.yaml"))
    
    # Should return default config
    assert isinstance(config, Config)
    assert len(config.models) == 0
