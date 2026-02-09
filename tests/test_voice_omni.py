#!/usr/bin/env python3
"""
Test voice pipeline with Omni mode via WebSocket.
"""

import asyncio
import json
import wave
import io
import sys

import websockets
import sounddevice as sd
import soundfile as sf
import numpy as np


ATLAS_WS_URL = "ws://localhost:8001/api/v1/ws/orchestrated"
SAMPLE_RATE = 16000
CHANNELS = 1


async def record_and_send():
    """Record audio and send through the orchestrated voice pipeline."""
    print("=" * 60)
    print("Testing Atlas Voice Pipeline with Omni Mode")
    print("=" * 60)

    # Connect to WebSocket
    print(f"\nConnecting to {ATLAS_WS_URL}...")

    try:
        async with websockets.connect(ATLAS_WS_URL, ping_interval=60, ping_timeout=120) as ws:
            print("Connected!")

            # Wait for initial state
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            state = json.loads(msg)
            print(f"Initial state: {state.get('state', 'unknown')}")

            # Record audio
            print("\n" + "-" * 40)
            duration = 4
            print(f"Recording for {duration} seconds... (say 'Atlas, what time is it?')")

            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
            )
            sd.wait()
            print("Recording complete!")

            # Convert to bytes (raw PCM, not WAV)
            audio_bytes = audio.tobytes()

            # Send audio in chunks (simulating real-time streaming)
            print("\nSending audio to Atlas...")
            chunk_size = 3200  # 100ms chunks at 16kHz

            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)  # Small delay between chunks

            print("Audio sent, waiting for response...")

            # Wait for response (with timeout)
            response_audio = None
            response_text = None

            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)

                    if isinstance(msg, bytes):
                        # Audio response
                        print(f"Received audio: {len(msg)} bytes")
                        response_audio = msg
                    else:
                        data = json.loads(msg)
                        event_type = data.get("type", data.get("event", "unknown"))

                        print(f"Event: {event_type}")

                        if "transcript" in data:
                            print(f"  Transcript: {data['transcript']}")
                            response_text = data.get("transcript")

                        if "response_text" in data:
                            print(f"  Response: {data['response_text']}")
                            response_text = data.get("response_text")

                        if "state" in data:
                            print(f"  State: {data['state']}")

                        if data.get("state") == "idle" or event_type == "response_complete":
                            break

            except asyncio.TimeoutError:
                print("Timeout waiting for response")

            # Play audio response if received
            if response_audio:
                print("\n" + "-" * 40)
                print("Playing response audio...")
                try:
                    audio_data, sr = sf.read(io.BytesIO(response_audio))
                    sd.play(audio_data, sr)
                    sd.wait()
                    print("Audio playback complete!")
                except Exception as e:
                    print(f"Could not play audio: {e}")
            else:
                print("\nNo audio response received")

            print("\n" + "=" * 60)
            print("Test complete!")
            print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    asyncio.run(record_and_send())


if __name__ == "__main__":
    main()
