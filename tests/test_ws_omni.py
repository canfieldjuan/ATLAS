#!/usr/bin/env python3
"""
Test the WebSocket voice pipeline with Omni mode.
"""

import asyncio
import json
import websockets


async def test_websocket():
    """Test WebSocket orchestrator with test audio."""
    print("=" * 60)
    print("Testing WebSocket Voice Pipeline with Omni")
    print("=" * 60)

    # Read test audio
    with open("test_atlas_final.wav", "rb") as f:
        audio_bytes = f.read()

    print(f"Loaded test audio: {len(audio_bytes)} bytes")

    # Connect to WebSocket
    url = "ws://localhost:8005/api/v1/ws/orchestrated"
    print(f"\nConnecting to {url}...")

    async with websockets.connect(url, ping_interval=60, ping_timeout=120) as ws:
        print("Connected!")

        # Wait for initial state
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(msg)
        print(f"Initial: {data}")

        # Send audio in chunks (simulating streaming)
        print("\nSending audio...")
        chunk_size = 3200  # 100ms at 16kHz

        # Skip WAV header (44 bytes) for raw PCM
        raw_audio = audio_bytes[44:]

        for i in range(0, len(raw_audio), chunk_size):
            chunk = raw_audio[i:i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.05)

        # Send end signal
        print("Audio sent, waiting for response...")

        # Wait for responses
        response_audio = None
        try:
            for _ in range(50):  # Max 50 messages
                msg = await asyncio.wait_for(ws.recv(), timeout=30)

                if isinstance(msg, bytes):
                    print(f"  Audio response: {len(msg)} bytes")
                    response_audio = msg
                else:
                    data = json.loads(msg)
                    event_type = data.get("type", data.get("event", data.get("state", "?")))
                    print(f"  Event: {event_type}")

                    if "transcript" in data:
                        print(f"    Transcript: {data['transcript']}")
                    if "response_text" in data:
                        print(f"    Response: {data['response_text'][:100]}...")
                    if "error" in data:
                        print(f"    Error: {data['error']}")

                    # Check if complete
                    if data.get("state") == "idle":
                        break
                    if data.get("type") == "complete":
                        break

        except asyncio.TimeoutError:
            print("Timeout waiting for response")

        # Save audio if received
        if response_audio:
            with open("test_ws_response.wav", "wb") as f:
                f.write(response_audio)
            print(f"\nSaved response audio: test_ws_response.wav ({len(response_audio)} bytes)")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_websocket())
