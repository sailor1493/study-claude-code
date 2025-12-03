"""CLI tool for running the API server."""

import sys
from pathlib import Path
import argparse
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from object_detection.api.server import create_app


def main():
    """Run API server."""
    parser = argparse.ArgumentParser(description="Object Detection API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5s",
        help="Model to load (default: yolov5s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cpu/cuda/cuda:0, default: auto)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU even if available",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Object Detection API Server")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"GPU Enabled: {not args.no_gpu}")
    print("="*60)
    
    # Create app
    app = create_app(
        model_name=args.model,
        device=args.device,
        enable_gpu=not args.no_gpu,
    )
    
    # Run server
    print("\nStarting server...")
    print(f"API docs available at http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
