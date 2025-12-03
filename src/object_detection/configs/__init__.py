"""Configs module initialization."""

from object_detection.configs.config import (
    Config,
    ModelConfig,
    BenchmarkConfig,
    APIConfig,
    load_config,
)

__all__ = [
    "Config",
    "ModelConfig", 
    "BenchmarkConfig",
    "APIConfig",
    "load_config",
]
