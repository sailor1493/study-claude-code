"""Configuration management for object detection framework."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a detection model."""
    
    name: str
    model_type: str  # 'yolo', 'detr', 'vlm'
    device: str = "cpu"
    confidence_threshold: float = 0.5
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    
    name: str
    benchmark_type: str  # 'performance', 'accuracy'
    data_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIConfig:
    """Configuration for API server."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    default_model: str = "yolov5s"
    enable_gpu: bool = True
    device: Optional[str] = None


@dataclass
class Config:
    """Main configuration object."""
    
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    benchmarks: Dict[str, BenchmarkConfig] = field(default_factory=dict)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        models = {}
        for name, model_data in data.get('models', {}).items():
            models[name] = ModelConfig(name=name, **model_data)
        
        benchmarks = {}
        for name, bench_data in data.get('benchmarks', {}).items():
            benchmarks[name] = BenchmarkConfig(name=name, **bench_data)
        
        api = APIConfig(**data.get('api', {}))
        
        return cls(models=models, benchmarks=benchmarks, api=api)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            'models': {
                name: {
                    'model_type': model.model_type,
                    'device': model.device,
                    'confidence_threshold': model.confidence_threshold,
                    'params': model.params,
                }
                for name, model in self.models.items()
            },
            'benchmarks': {
                name: {
                    'benchmark_type': bench.benchmark_type,
                    'data_path': bench.data_path,
                    'params': bench.params,
                }
                for name, bench in self.benchmarks.items()
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'default_model': self.api.default_model,
                'enable_gpu': self.api.enable_gpu,
                'device': self.api.device,
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


def load_config(path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Config object
    """
    if path is None or not path.exists():
        return Config()
    return Config.from_yaml(path)
