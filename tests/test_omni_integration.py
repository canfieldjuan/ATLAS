#!/usr/bin/env python3
"""
Test Qwen2.5-Omni integration with Atlas voice pipeline.
"""

import asyncio
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


async def test_omni_service():
    """Test the OmniService directly."""
    print("=" * 60)
    print("Testing Atlas Qwen-Omni Service Integration")
    print("=" * 60)

    from atlas_brain.services import omni_registry

    # Check available services
    print(f"\nAvailable omni services: {omni_registry.list_available()}")

    # Activate the service
    print("\nActivating qwen-omni service...")
    omni = omni_registry.activate("qwen-omni")
    print(f"Service activated: {omni.model_info.name}")
    print(f"Device: {omni.model_info.device}")
    print(f"Capabilities: {omni.model_info.capabilities}")

    # Test 1: Text-to-speech (synthesize)
    print("\n" + "-" * 40)
    print("Test 1: Text-to-Speech")
    print("-" * 40)

    audio_bytes = await omni.synthesize("Hello, I am Atlas, your home assistant.")
    print(f"Generated audio: {len(audio_bytes)} bytes")

    # Save the audio
    with open("test_omni_tts.wav", "wb") as f:
        f.write(audio_bytes)
    print("Saved to: test_omni_tts.wav")

    # Test 2: Chat with audio output
    print("\n" + "-" * 40)
    print("Test 2: Chat with Audio Output")
    print("-" * 40)

    from atlas_brain.services.protocols import Message

    messages = [
        Message(role="user", content="What time is it?"),
    ]

    response = await omni.chat(messages, include_audio=True)
    print(f"Text response: {response.text}")
    print(f"Audio duration: {response.audio_duration_sec:.1f}s")

    if response.audio_bytes:
        with open("test_omni_chat.wav", "wb") as f:
            f.write(response.audio_bytes)
        print("Saved audio to: test_omni_chat.wav")

    # Test 3: Speech-to-speech (using previous audio as input)
    print("\n" + "-" * 40)
    print("Test 3: Speech-to-Speech")
    print("-" * 40)

    # Use the TTS output as input
    s2s_response = await omni.speech_to_speech(audio_bytes)
    print(f"Text response: {s2s_response.text}")
    print(f"Audio duration: {s2s_response.audio_duration_sec:.1f}s")

    if s2s_response.audio_bytes:
        with open("test_omni_s2s.wav", "wb") as f:
            f.write(s2s_response.audio_bytes)
        print("Saved audio to: test_omni_s2s.wav")

    # Cleanup
    print("\n" + "-" * 40)
    print("Cleanup")
    print("-" * 40)
    omni_registry.deactivate()
    print("Service deactivated")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_omni_service())
