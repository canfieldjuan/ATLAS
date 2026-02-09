#!/usr/bin/env python3
"""
Convert Whisper model to RKNN format for Rockchip NPU (RK3588).

Note: Whisper is complex for NPU conversion. Consider alternatives:
- Sherpa-ONNX (pre-optimized, recommended)
- Vosk (lightweight, CPU-based)
- Faster-Whisper with CTranslate2 (CPU, often faster than NPU)

This script converts the Whisper encoder only.
Decoder runs on CPU (less critical for latency).

Usage:
    python convert_whisper_to_rknn.py --model tiny --language en

Requirements:
    pip install rknn-toolkit2 openai-whisper onnx torch
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def export_whisper_encoder_to_onnx(
    model_size: str = 'tiny',
    output_dir: str = 'models/whisper_onnx'
):
    """Export Whisper encoder to ONNX format."""
    try:
        import whisper
    except ImportError:
        print("ERROR: openai-whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)

    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    encoder = model.encoder
    encoder.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'whisper_{model_size}_encoder.onnx')

    # Whisper encoder input: mel spectrogram
    # Shape: (batch_size, n_mels, n_frames)
    # n_mels = 80 (for all Whisper models)
    # n_frames = 3000 (30 seconds at 16kHz with 10ms hop)
    batch_size = 1
    n_mels = 80
    n_frames = 3000

    dummy_input = torch.randn(batch_size, n_mels, n_frames)

    print(f"\nExporting Whisper encoder to ONNX...")
    print(f"  Model: {model_size}")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output: {output_path}")

    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        input_names=['mel_spectrogram'],
        output_names=['encoder_output'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size', 2: 'n_frames'},
            'encoder_output': {0: 'batch_size', 1: 'n_frames'}
        },
        opset_version=12,  # RKNN compatible
        do_constant_folding=True,
    )

    print(f"✓ ONNX encoder saved: {output_path}")
    print(f"  Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path


def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    quantize: bool = True,
    target_platform: str = 'rk3588'
):
    """Convert ONNX encoder to RKNN format."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 not installed.")
        print("Install from: https://github.com/rockchip-linux/rknn-toolkit2")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Converting Whisper Encoder ONNX to RKNN")
    print(f"{'='*60}")
    print(f"Input:  {onnx_path}")
    print(f"Output: {output_path}")
    print(f"Target: {target_platform}")
    print(f"Quantization: {'INT8' if quantize else 'FP16'}")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Configuration for Whisper encoder
    print("\nConfiguring...")
    rknn.config(
        target_platform=target_platform,
        quantized_algorithm='normal',
        quantized_method='channel' if quantize else None,
        optimization_level=3,
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
        print("Creating calibration dataset for INT8 quantization...")
        # Generate random mel spectrograms for calibration
        dataset = []
        for _ in range(50):  # Fewer samples for transformer models
            mel = np.random.randn(1, 80, 3000).astype(np.float32)
            dataset.append(mel)

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

    # Release
    rknn.release()

    print(f"\n{'='*60}")
    print(f"✓ Conversion successful!")
    print(f"{'='*60}")
    print(f"RKNN model: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert Whisper encoder to RKNN format for Rockchip NPU'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='tiny',
        choices=['tiny', 'base', 'small'],
        help='Whisper model size (tiny, base, small). Larger models may not fit/be too slow.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/whisper_rknn',
        help='Output directory for RKNN model'
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
        '--skip-onnx',
        action='store_true',
        help='Skip ONNX export (use existing .onnx file)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Export to ONNX
    onnx_dir = 'models/whisper_onnx'
    onnx_path = os.path.join(onnx_dir, f'whisper_{args.model}_encoder.onnx')

    if args.skip_onnx:
        if not os.path.exists(onnx_path):
            print(f"ERROR: ONNX file not found: {onnx_path}")
            sys.exit(1)
        print(f"Using existing ONNX file: {onnx_path}")
    else:
        onnx_path = export_whisper_encoder_to_onnx(args.model, onnx_dir)

    # Step 2: Convert ONNX to RKNN
    output_path = os.path.join(args.output_dir, f'whisper_{args.model}_encoder.rknn')

    success = convert_onnx_to_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        quantize=not args.no_quantize,
        target_platform=args.target
    )

    if not success:
        sys.exit(1)

    print("\n" + "="*60)
    print("⚠️  IMPORTANT NOTES")
    print("="*60)
    print("1. This converts ENCODER ONLY (feature extraction)")
    print("2. Decoder still runs on CPU (text generation)")
    print("3. Whisper on NPU may not be faster than CPU alternatives")
    print("\nRECOMMENDED ALTERNATIVES:")
    print("- Sherpa-ONNX: Pre-optimized for ARM, streaming support")
    print("- Faster-Whisper: CTranslate2, often faster than NPU")
    print("- Vosk: Lightweight, real-time on CPU")
    print("\nSee ROCKCHIP_MODEL_CONVERSION.md for more details")


if __name__ == '__main__':
    main()
