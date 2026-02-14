# Rockchip NPU Model Conversion Guide

This guide covers converting YOLO, STT (Whisper), and TTS models for Rockchip NPU (RK3588/RK3588S on Orange Pi 5 Plus).

## Hardware Info

- Device: Orange Pi 5 Plus
- SoC: Rockchip RK3588/RK3588S
- NPU: 6 TOPS (INT8) NPU with 3 cores
- Supported: INT8, FP16 quantization

## Overview

Rockchip NPU uses RKNN (Rockchip Neural Network) format. The conversion process:

1. Export model to ONNX format
2. Use RKNN-Toolkit2 to convert ONNX → RKNN
3. Deploy with RKNN-Toolkit2 Lite runtime on device

## Prerequisites

### On Development Machine (x86_64/AMD64)

```bash
# Install RKNN-Toolkit2 for conversion (runs on x86_64 only)
pip install rknn-toolkit2

# Required dependencies
pip install onnx onnxruntime opencv-python
```

### On Orange Pi 5 Plus (aarch64)

```bash
# Install RKNN-Toolkit2-Lite for inference
# Download from: https://github.com/rockchip-linux/rknn-toolkit2
# Or use pre-built wheels for RK3588
```

## 1. YOLO Model Conversion

### Current Models

You have: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l-world.pt`

### Recommended: YOLOv8n (Nano) for Edge Device

- Smallest, fastest variant
- Good balance of speed/accuracy for edge devices
- ~3-6 MB model size

### Conversion Steps

#### Step 1: Export YOLO to ONNX

```python
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Export to ONNX with specific settings for RKNN
model.export(
    format='onnx',
    imgsz=640,  # or 416 for faster inference
    simplify=True,
    opset=12,  # RKNN supports opset 11-13
    dynamic=False  # Static shapes required for NPU
)
# Outputs: yolov8n.onnx
```

#### Step 2: Convert ONNX to RKNN

See `scripts/convert_yolo_to_rknn.py`

### Optimization Tips

- Use 416x416 or 320x320 input for faster inference
- INT8 quantization provides ~4x speedup
- Use calibration dataset for better INT8 accuracy

## 2. STT Model Conversion (Whisper/Faster-Whisper)

### Recommended Models for Edge

- **Whisper Tiny**: 39M params, ~150 MB, fastest
- **Whisper Base**: 74M params, ~290 MB, better accuracy
- **Faster-Whisper**: Optimized CTranslate2 version (recommended)

### Current Setup

You're using NeMo/Nemotron ASR which is NVIDIA-specific.

### Alternative: RKNN-Compatible STT

#### Option A: Whisper Tiny/Base RKNN

```python
# Convert Whisper encoder to RKNN
# Decoder runs on CPU (less critical for latency)
```

#### Option B: Sherpa-ONNX (Recommended)

- Already optimized for ARM/NPU
- Supports streaming ASR
- Pre-converted models available
- Better than full Whisper conversion

#### Option C: Vosk (Lightweight)

- Small models (50-100 MB)
- Real-time on CPU
- No NPU needed, good fallback

### Conversion Script

See `scripts/convert_whisper_to_rknn.py`

**Note**: Transformer-based models (Whisper) are complex for NPU conversion. Consider:

1. CPU-based Faster-Whisper (CTranslate2) - often faster than NPU conversion
2. Sherpa-ONNX with pre-optimized models
3. Lightweight alternatives like Vosk

## 3. TTS Model Conversion

### Current Options

Based on your setup, you might be using:

- Piper TTS (mentioned in pipeline.py)
- MagPie TTS (found in models/)

### Recommended for Rockchip

#### Option A: Piper TTS (Recommended)

- Already optimized for edge devices
- ONNX-based, runs efficiently on CPU
- Small models (5-20 MB per voice)
- No NPU conversion needed - already fast enough

```bash
# Piper is pre-optimized and fast on ARM
# Download voices from: https://github.com/rhasspy/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx
```

#### Option B: Lightweight Tacotron2 + WaveGlow/WaveRNN

Could be converted to RKNN, but complex and often slower than CPU inference.

#### Option C: Coqui TTS Lite Models

- Fast CPU inference on ARM
- Good quality
- May not benefit from NPU acceleration

### Conversion (if needed)

See `scripts/convert_tts_to_rknn.py`

**Recommendation**: Don't convert TTS to RKNN. Piper TTS on CPU is already extremely fast and efficient for edge devices.

## Performance Expectations

### RK3588 NPU (6 TOPS INT8)

- **YOLOv8n 640x640**: ~30-60 FPS (INT8)
- **YOLOv8n 416x416**: ~60-120 FPS (INT8)
- **Whisper Tiny Encoder**: ~2-5x faster than CPU
- **TTS**: CPU inference often better (sequential processing)

## Calibration Dataset

For INT8 quantization, you need a calibration dataset:

```python
# For YOLO: Random images similar to target domain
import numpy as np
calibration_images = [
    np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    for _ in range(100)
]

# For STT: Short audio segments
# For TTS: Sample mel-spectrograms
```

## Testing Converted Models

```python
from rknn.api import RKNN

# Initialize RKNN
rknn = RKNN()

# Load RKNN model
rknn.load_rknn('yolov8n.rknn')

# Initialize runtime (use NPU)
rknn.init_runtime(target='rk3588')

# Run inference
outputs = rknn.inference(inputs=[input_data])

# Release
rknn.release()
```

## Next Steps

1. Start with YOLOv8n conversion (highest ROI)
2. Use Sherpa-ONNX or Vosk for STT (skip Whisper RKNN conversion)
3. Keep Piper TTS on CPU (don't convert)

## Resources

- RKNN-Toolkit2: <https://github.com/rockchip-linux/rknn-toolkit2>
- Model Zoo: <https://github.com/airockchip/rknn_model_zoo>
- YOLOv8 RKNN: <https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8>
- Sherpa-ONNX: <https://github.com/k2-fsa/sherpa-onnx>
- Piper TTS: <https://github.com/rhasspy/piper>
