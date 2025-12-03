"""Device management utilities."""

import torch
from typing import Optional, Dict, Any


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get appropriate device for model execution.
    
    Args:
        device: Requested device ('cpu', 'cuda', 'cuda:0', etc.)
                If None, automatically selects best available device
    
    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "id": i,
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "multi_processor_count": device_props.multi_processor_count,
            })
    
    return info


def print_device_info():
    """Print device information in a human-readable format."""
    info = get_device_info()
    
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"Number of GPUs: {info['num_gpus']}")
        print(f"Current Device: {info['current_device']}")
        print(f"\nGPU Details:")
        for device in info['devices']:
            print(f"  GPU {device['id']}: {device['name']}")
            print(f"    Total Memory: {device['total_memory'] / (1024**3):.2f} GB")
            print(f"    Multiprocessors: {device['multi_processor_count']}")
    else:
        print("Running on CPU")
    
    print("="*60 + "\n")
