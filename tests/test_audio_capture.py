#!/usr/bin/env python3
"""
Test script to verify audio capture is working.

This tests PyAudio directly without Pipecat to isolate audio issues.
"""

import pyaudio
import time
import numpy as np


def list_audio_devices():
    """List all audio devices."""
    p = pyaudio.PyAudio()
    print("=" * 60)
    print("Available Audio Devices:")
    print("=" * 60)

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        in_ch = info['maxInputChannels']
        out_ch = info['maxOutputChannels']
        name = info['name']
        rate = int(info['defaultSampleRate'])

        if in_ch > 0 or out_ch > 0:
            device_type = []
            if in_ch > 0:
                device_type.append(f"IN:{in_ch}ch")
            if out_ch > 0:
                device_type.append(f"OUT:{out_ch}ch")
            print(f"  [{i}] {name} ({', '.join(device_type)}) @ {rate}Hz")

    p.terminate()
    print()


def find_device_by_name(name_substring: str, input_device: bool = True) -> int | None:
    """Find audio device index by name substring."""
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        has_channels = info['maxInputChannels'] > 0 if input_device else info['maxOutputChannels'] > 0
        if has_channels and name_substring.lower() in info['name'].lower():
            p.terminate()
            return i

    p.terminate()
    return None


def test_audio_capture(device_index: int | None = None, sample_rate: int = 44100, duration: float = 3.0):
    """Test audio capture from microphone."""
    p = pyaudio.PyAudio()

    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
        print(f"Using device [{device_index}]: {info['name']}")
    else:
        print("Using default input device")

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print()

    frames = []
    chunk_size = int(sample_rate / 10)  # 100ms chunks

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
        )

        print("Recording... speak into the microphone!")
        print("-" * 40)

        start_time = time.time()
        chunk_count = 0

        while time.time() - start_time < duration:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
            chunk_count += 1

            # Calculate audio level
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_array ** 2))
            db = 20 * np.log10(rms + 1e-10)

            # Visual meter
            bars = int(max(0, min(50, (db + 60) / 2)))
            meter = "|" + "#" * bars + " " * (50 - bars) + "|"
            print(f"\rChunk {chunk_count:3d}: {meter} {db:5.1f} dB", end="", flush=True)

        print("\n" + "-" * 40)

        stream.stop_stream()
        stream.close()

        # Analyze recording
        all_audio = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
        max_val = np.max(np.abs(all_audio))
        avg_rms = np.sqrt(np.mean(all_audio ** 2))

        print(f"Recording complete!")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total samples: {len(all_audio)}")
        print(f"  Max amplitude: {max_val:.0f} ({20 * np.log10(max_val / 32768 + 1e-10):.1f} dBFS)")
        print(f"  Average RMS: {avg_rms:.0f} ({20 * np.log10(avg_rms + 1e-10):.1f} dB)")

        if max_val < 100:
            print("\n[WARNING] Very low audio levels detected - check microphone!")
        elif max_val < 1000:
            print("\n[INFO] Low audio levels - might need to speak louder or adjust mic gain")
        else:
            print("\n[OK] Audio capture working correctly!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()


if __name__ == "__main__":
    import sys

    list_audio_devices()

    # Try to find SoloCast
    device_index = find_device_by_name("SoloCast")
    if device_index is not None:
        print(f"Found HyperX SoloCast at index {device_index}")
    else:
        print("SoloCast not found, using default device")

    # Test at different sample rates
    print("\n" + "=" * 60)
    print("Testing at 44100 Hz (SoloCast native rate)")
    print("=" * 60)
    test_audio_capture(device_index, sample_rate=44100, duration=3.0)

    print("\n" + "=" * 60)
    print("Testing at 16000 Hz (Pipecat/VAD rate)")
    print("=" * 60)
    try:
        test_audio_capture(device_index, sample_rate=16000, duration=3.0)
    except Exception as e:
        print(f"16000 Hz not supported: {e}")
        print("This is expected - SoloCast needs resampling from 44100 Hz")
