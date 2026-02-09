#!/usr/bin/env python3
"""
Convert YOLOv8 model to RKNN format for Rockchip NPU (RK3588).

Usage:
    python convert_yolo_to_rknn.py --model yolov8n.pt --imgsz 640

Requirements:
    pip install rknn-toolkit2 ultralytics onnx
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def export_yolo_to_onnx(model_path: str, imgsz: int = 640):
    """Export YOLOv8 PyTorch model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)

    onnx_path = model_path.replace('.pt', '.onnx')
    
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Input size: {imgsz}x{imgsz}")
    print(f"  Opset: 12 (RKNN compatible)")
    
    model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=True,
        opset=12,  # RKNN supports opset 11-13
        dynamic=False  # Static shapes required for NPU
    )
    
    print(f"✓ ONNX model saved: {onnx_path}")
    return onnx_path


def create_calibration_dataset(num_samples: int = 100, imgsz: int = 640):
    """Create random calibration dataset for INT8 quantization."""
    print(f"Creating calibration dataset: {num_samples} samples of {imgsz}x{imgsz}")
    
    # Generate random images (RGB, 0-255)
    dataset = []
    for i in range(num_samples):
        img = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)
        dataset.append(img)
    
    return dataset


def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    imgsz: int = 640,
    quantize: bool = True,
    target_platform: str = 'rk3588'
):
    """Convert ONNX model to RKNN format."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 not installed.")
        print("Install from: https://github.com/rockchip-linux/rknn-toolkit2")
        print("Note: Only works on x86_64 Linux for conversion")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Converting ONNX to RKNN")
    print(f"{'='*60}")
    print(f"Input:  {onnx_path}")
    print(f"Output: {output_path}")
    print(f"Target: {target_platform}")
    print(f"Quantization: {'INT8' if quantize else 'FP16'}")
    
    # Create RKNN object
    rknn = RKNN(verbose=True)
    
    # Pre-processing configuration for YOLOv8
    # YOLOv8 expects normalized input [0, 1]
    print("\nConfiguring pre-processing...")
    rknn.config(
        mean_values=[[0, 0, 0]],  # No mean subtraction
        std_values=[[255, 255, 255]],  # Normalize to [0, 1]
        target_platform=target_platform,
        quantized_algorithm='normal',  # 'normal' or 'mmse'
        quantized_method='channel' if quantize else None,
        optimization_level=3,  # 0-3, higher is more optimized
    )
    
    # Load ONNX model
    print("\nLoading ONNX model...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('ERROR: Load ONNX model failed!')
        return False
    
    # Build RKNN model
    print("\nBuilding RKNN model...")
    
    if quantize:
        # Create calibration dataset for INT8 quantization
        print("Creating calibration dataset for INT8 quantization...")
        dataset = create_calibration_dataset(num_samples=100, imgsz=imgsz)
        
        ret = rknn.build(
            do_quantization=True,
            dataset=dataset,
            rknn_batch_size=1
        )
    else:
        ret = rknn.build(
            do_quantization=False,
            rknn_batch_size=1
        )
    
    if ret != 0:
        print('ERROR: Build RKNN model failed!')
        return False
    
    # Export RKNN model
    print(f"\nExporting RKNN model to: {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('ERROR: Export RKNN model failed!')
        return False
    
    # Accuracy analysis (optional, takes time)
    print("\n[Optional] Run accuracy analysis? (y/n)")
    # Uncomment to enable interactive prompt
    # if input().lower() == 'y':
    #     print("\nRunning accuracy analysis...")
    #     ret = rknn.accuracy_analysis(inputs=[dataset[0]], target=target_platform)
    
    # Release RKNN instance
    rknn.release()
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion successful!")
    print(f"{'='*60}")
    print(f"RKNN model: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return True


def test_rknn_model(rknn_path: str, imgsz: int = 640):
    """Test RKNN model inference (requires RK3588 device or simulator)."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("RKNN toolkit not available for testing")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing RKNN model: {rknn_path}")
    print(f"{'='*60}")
    
    rknn = RKNN()
    
    # Load RKNN model
    print("Loading RKNN model...")
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print('ERROR: Load RKNN model failed!')
        return
    
    # Initialize runtime
    # Use 'rk3588' for actual device, 'None' for simulator
    print("Initializing runtime on simulator...")
    ret = rknn.init_runtime()  # Use target='rk3588' on device
    if ret != 0:
        print('ERROR: Init runtime failed!')
        print('Note: This requires RK3588 device or RKNN simulator')
        rknn.release()
        return
    
    # Create dummy input
    print(f"Creating test input: {imgsz}x{imgsz}x3")
    test_input = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Inference
    print("Running inference...")
    outputs = rknn.inference(inputs=[test_input])
    
    print(f"\nInference successful!")
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")
    
    # Release
    rknn.release()
    print("✓ Test complete")


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLOv8 to RKNN format for Rockchip NPU'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to YOLOv8 .pt model file'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        choices=[320, 416, 640],
        help='Input image size (320, 416, or 640)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output RKNN model path (default: same as input with .rknn)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='rk3588',
        choices=['rk3588', 'rk3588s'],
        help='Target Rockchip platform'
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable INT8 quantization (use FP16 instead)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test RKNN model after conversion (requires RK3588 or simulator)'
    )
    parser.add_argument(
        '--skip-onnx',
        action='store_true',
        help='Skip ONNX export (use existing .onnx file)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Step 1: Export to ONNX
    if args.skip_onnx:
        onnx_path = args.model.replace('.pt', '.onnx')
        if not os.path.exists(onnx_path):
            print(f"ERROR: ONNX file not found: {onnx_path}")
            sys.exit(1)
        print(f"Using existing ONNX file: {onnx_path}")
    else:
        onnx_path = export_yolo_to_onnx(args.model, args.imgsz)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.model.replace('.pt', '.rknn')
    
    # Step 2: Convert ONNX to RKNN
    success = convert_onnx_to_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        imgsz=args.imgsz,
        quantize=not args.no_quantize,
        target_platform=args.target
    )
    
    if not success:
        sys.exit(1)
    
    # Step 3: Test (optional)
    if args.test:
        test_rknn_model(output_path, args.imgsz)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Copy {output_path} to your Orange Pi 5 Plus")
    print(f"2. Install rknn-toolkit2-lite on the device")
    print(f"3. Use the model with RKNN runtime")
    print("\nSee ROCKCHIP_MODEL_CONVERSION.md for deployment instructions")


if __name__ == '__main__':
    main()
