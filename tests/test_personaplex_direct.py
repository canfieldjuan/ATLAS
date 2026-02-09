#!/usr/bin/env python3
"""
Direct PersonaPlex connection test - bypasses SignalWire.
Plays a test tone and measures response latency.
"""
import asyncio
import time
import numpy as np

async def main():
    from atlas_brain.services.personaplex.service import PersonaPlexService
    from atlas_brain.services.personaplex.config import get_personaplex_config

    config = get_personaplex_config()
    print(f"PersonaPlex config: {config.host}:{config.port}")

    service = PersonaPlexService(config)

    audio_received = []
    text_received = []

    def on_audio(opus_data: bytes):
        t = time.time()
        audio_received.append((t, len(opus_data)))
        print(f"[{t:.3f}] RX audio: {len(opus_data)} bytes")

    def on_text(text: str):
        t = time.time()
        text_received.append((t, text))
        print(f"[{t:.3f}] RX text: {text}")

    service.set_audio_callback(on_audio)
    service.set_text_callback(on_text)

    print("Connecting to PersonaPlex...")
    t0 = time.time()
    connected = await service.connect(
        text_prompt="Be helpful and concise.",
        voice_prompt="NATF0",
    )
    t1 = time.time()
    print(f"Connection took {t1-t0:.2f}s, connected={connected}")

    if not connected:
        print("Failed to connect!")
        return

    # Generate test audio: 3 seconds of 440Hz sine wave (simulate speech)
    print("\nSending 3 seconds of test audio...")
    sample_rate = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32768).astype(np.int16)

    # Encode and send in 80ms chunks
    import sphn
    encoder = sphn.OpusStreamWriter(sample_rate)

    chunk_size = 1920  # 80ms at 24kHz
    t_start = time.time()

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            # Pad last chunk
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        samples = chunk.astype(np.float32) / 32768.0
        encoder.append_pcm(samples)
        opus_data = encoder.read_bytes()

        if opus_data:
            sent = await service.send_audio(opus_data)
            print(f"TX {len(opus_data)} bytes opus")

        # Send at real-time rate
        await asyncio.sleep(0.08)

    t_send_done = time.time()
    print(f"\nDone sending audio in {t_send_done - t_start:.2f}s")

    # Wait for responses
    print("Waiting 10s for responses...")
    await asyncio.sleep(10)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Audio chunks received: {len(audio_received)}")
    print(f"Text chunks received: {len(text_received)}")

    if audio_received:
        first_audio = audio_received[0][0]
        latency = first_audio - t_send_done
        print(f"First audio latency after send: {latency:.2f}s")

    if text_received:
        print("Text received:")
        for t, text in text_received:
            print(f"  [{t:.3f}] {text}")

    await service.disconnect()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
