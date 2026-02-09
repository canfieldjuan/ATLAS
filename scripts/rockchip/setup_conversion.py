#!/usr/bin/env python3
"""
Setup script for Rockchip model conversion.

Installs required dependencies on x86_64 Linux for model conversion.
The converted models can then be deployed to Orange Pi 5 Plus.

Usage:
    ./setup_rockchip_conversion.sh
"""

import os
import platform
import subprocess
import sys


def check_platform():
    """Check if running on x86_64 Linux (required for RKNN-Toolkit2)."""
    machine = platform.machine()
    system = platform.system()
    
    print(f"Platform: {system} {machine}")
    
    if system != 'Linux':
        print("WARNING: RKNN-Toolkit2 requires Linux")
        return False
    
    if machine not in ['x86_64', 'AMD64', 'amd64']:
        print("WARNING: RKNN-Toolkit2 requires x86_64 architecture")
        print("Current architecture:", machine)
        return False
    
    return True


def install_dependencies():
    """Install required Python packages."""
    packages = [
        'rknn-toolkit2',  # Rockchip conversion toolkit
        'ultralytics',     # YOLOv8
        'onnx',           # ONNX format
        'onnxruntime',    # ONNX runtime
        'opencv-python',  # Image processing
        'openai-whisper', # Whisper (optional)
        'torch',          # PyTorch
        'torchaudio',     # Audio processing
    ]
    
    print("\nInstalling dependencies...")
    print("=" * 60)
    
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package
            ])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}")
            if package == 'rknn-toolkit2':
                print("\nRKNN-Toolkit2 installation failed.")
                print("Try manual installation:")
                print("1. Clone: git clone https://github.com/rockchip-linux/rknn-toolkit2")
                print("2. Install wheel from: rknn-toolkit2/packages/")
                print("   pip install rknn_toolkit2-*-py3-none-any.whl")


def verify_installation():
    """Verify that required packages are installed."""
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("=" * 60)
    
    checks = {
        'rknn-toolkit2': 'from rknn.api import RKNN',
        'ultralytics': 'from ultralytics import YOLO',
        'onnx': 'import onnx',
        'torch': 'import torch',
    }
    
    all_ok = True
    for name, import_stmt in checks.items():
        try:
            exec(import_stmt)
            print(f"✓ {name}: OK")
        except ImportError:
            print(f"✗ {name}: NOT FOUND")
            all_ok = False
    
    return all_ok


def download_models():
    """Download example models for conversion."""
    print("\n" + "=" * 60)
    print("Download models? (y/n)")
    print("=" * 60)
    
    if input().lower() != 'y':
        print("Skipping model download")
        return
    
    # Download YOLOv8n if not present
    yolo_models = ['yolov8n.pt', 'yolov8s.pt']
    
    for model in yolo_models:
        if os.path.exists(model):
            print(f"✓ {model} already exists")
        else:
            print(f"Downloading {model}...")
            try:
                from ultralytics import YOLO
                YOLO(model)  # Auto-downloads
                print(f"✓ {model} downloaded")
            except Exception as e:
                print(f"✗ Failed to download {model}: {e}")


def create_conversion_script():
    """Create a quick conversion script."""
    script = """#!/bin/bash
# Quick YOLO to RKNN conversion script

set -e

echo "Converting YOLOv8n to RKNN format..."
python scripts/rockchip/convert_yolo_to_rknn.py \\
    --model yolov8n.pt \\
    --imgsz 640 \\
    --target rk3588

echo ""
echo "Conversion complete!"
echo "Copy yolov8n.rknn to your Orange Pi 5 Plus"
"""
    
    script_path = "convert_yolo_quick.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    print(f"\n✓ Quick conversion script created: {script_path}")


def main():
    print("=" * 60)
    print("Rockchip Model Conversion Setup")
    print("=" * 60)
    
    # Check platform
    if not check_platform():
        print("\nWARNING: Platform may not be compatible")
        print("Continue anyway? (y/n)")
        if input().lower() != 'y':
            sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Some packages failed to install")
        print("Please install missing packages manually")
        sys.exit(1)
    
    print("\n✓ All required packages installed!")
    
    # Download models
    download_models()
    
    # Create helper script
    create_conversion_script()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Convert YOLO model:")
    print("   python scripts/rockchip/convert_yolo_to_rknn.py --model yolov8n.pt")
    print("\n2. Or use quick script:")
    print("   ./convert_yolo_quick.sh")
    print("\n3. Copy .rknn file to Orange Pi 5 Plus")
    print("\nSee ROCKCHIP_MODEL_CONVERSION.md for full documentation")


if __name__ == '__main__':
    main()
