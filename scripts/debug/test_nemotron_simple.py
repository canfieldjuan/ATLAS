#!/usr/bin/env python3
"""Simple Nemotron test - record, transcribe, show output."""

import time
import numpy as np
import sounddevice as sd
from scipy import signal

# Audio settings
MIC_DEVICE = None  # Use default (HyperX SoloCast via PipeWire)
NATIVE_RATE = 44100
TARGET_RATE = 16000
DURATION = 5  # seconds


def main():
    print("=" * 50)
    print("NEMOTRON SIMPLE TEST")
    print("=" * 50)

    # Load model
    print("\n1. Loading model...")
    start = time.time()

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model.eval()

    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Loaded on CUDA in {time.time() - start:.1f}s")
    else:
        print(f"   Loaded on CPU in {time.time() - start:.1f}s")

    # Record audio
    print(f"\n2. Recording {DURATION}s from HyperX mic...")
    print("   Get ready to say: 'Atlas turn on the TV'")
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...")
        time.sleep(1)
    print("   >>> SPEAK NOW <<<")

    audio = sd.rec(
        int(DURATION * NATIVE_RATE),
        samplerate=NATIVE_RATE,
        channels=1,
        dtype=np.float32,
        device=MIC_DEVICE
    )
    sd.wait()
    audio = audio.flatten()

    # Resample to 16kHz for model
    audio_16k = signal.resample(audio, int(len(audio) * TARGET_RATE / NATIVE_RATE))
    audio_16k = audio_16k.astype(np.float32)

    # Check raw audio level - no normalization
    peak = np.max(np.abs(audio_16k))
    print(f"   Raw peak: {peak:.4f} (no normalization)")

    # Check audio levels
    audio_max = np.max(np.abs(audio_16k))
    audio_rms = np.sqrt(np.mean(audio_16k ** 2))
    print(f"   Recorded {len(audio_16k)} samples @ 16kHz")
    print(f"   Audio max: {audio_max:.4f}")
    print(f"   Audio RMS: {audio_rms:.4f}")

    if audio_max < 0.01:
        print("   WARNING: Audio level very low!")
    elif audio_max < 0.05:
        print("   Note: Audio level is low but may work.")

    # Transcribe
    print("\n3. Transcribing...")
    start = time.time()

    with torch.no_grad():
        results = model.transcribe([audio_16k], batch_size=1)

    latency = (time.time() - start) * 1000

    # Get text
    result = results[0]
    if hasattr(result, 'text'):
        text = result.text
    else:
        text = str(result)

    print(f"\n" + "=" * 50)
    print(f"RESULT ({latency:.0f}ms):")
    print(f"=" * 50)
    print(f"\n   '{text}'")
    print()

    if not text.strip():
        print("   (empty transcript - no speech detected)")
    else:
        # Check punctuation
        has_punct = any(p in text for p in ".?!")
        print(f"   Punctuation detected: {has_punct}")
        if "." in text:
            print("   - Has period (.)")
        if "?" in text:
            print("   - Has question mark (?)")
        if "!" in text:
            print("   - Has exclamation (!)")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
