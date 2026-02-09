#!/usr/bin/env python3
"""
Test the voice pipeline with Omni mode using a test audio file.
"""

import asyncio
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


async def test_pipeline():
    """Test the orchestrator pipeline with omni mode."""
    print("=" * 60)
    print("Testing Atlas Voice Pipeline with Omni Mode")
    print("=" * 60)

    # Read test audio file
    test_file = "test_atlas_final.wav"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    with open(test_file, "rb") as f:
        audio_bytes = f.read()

    print(f"Loaded test audio: {len(audio_bytes)} bytes")

    # Check omni status
    from atlas_brain.services import omni_registry, stt_registry

    omni = omni_registry.get_active()
    stt = stt_registry.get_active()

    print(f"\nServices:")
    print(f"  STT: {stt.model_info.name if stt else 'Not loaded'}")
    print(f"  Omni: {omni.model_info.name if omni else 'Not loaded'}")

    if not stt:
        print("STT not loaded, activating...")
        stt = stt_registry.activate("nemotron")

    # Step 1: Transcribe with STT
    print("\n" + "-" * 40)
    print("Step 1: STT Transcription")
    print("-" * 40)

    result = await stt.transcribe(audio_bytes)
    transcript = result.get("transcript", "")
    print(f"Transcript: '{transcript}'")

    if not transcript:
        print("No transcript, cannot continue")
        return

    # Step 2: Process with Omni (chat mode with transcript)
    print("\n" + "-" * 40)
    print("Step 2: Omni Response Generation")
    print("-" * 40)

    if not omni:
        print("Omni not loaded!")
        return

    from atlas_brain.services.protocols import Message

    messages = [Message(role="user", content=transcript)]
    response = await omni.chat(messages, include_audio=True)

    print(f"Text response: {response.text}")
    print(f"Audio duration: {response.audio_duration_sec:.1f}s")

    if response.audio_bytes:
        output_file = "test_pipeline_response.wav"
        with open(output_file, "wb") as f:
            f.write(response.audio_bytes)
        print(f"Audio saved: {output_file}")

    # Step 3: Also test speech-to-speech directly
    print("\n" + "-" * 40)
    print("Step 3: Direct Speech-to-Speech")
    print("-" * 40)

    s2s_response = await omni.speech_to_speech(audio_bytes)
    print(f"S2S Text: {s2s_response.text}")
    print(f"S2S Audio duration: {s2s_response.audio_duration_sec:.1f}s")

    if s2s_response.audio_bytes:
        output_file = "test_pipeline_s2s.wav"
        with open(output_file, "wb") as f:
            f.write(s2s_response.audio_bytes)
        print(f"Audio saved: {output_file}")

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
