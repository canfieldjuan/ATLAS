#!/usr/bin/env python3
"""
Record real audio and test Nemotron STT.
Say "ATLAS" into your microphone!
"""
import asyncio
import sys
from pathlib import Path
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("❌ Missing dependencies. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice", "soundfile"])
    import sounddevice as sd
    import soundfile as sf

# Add Atlas to path
sys.path.insert(0, str(Path(__file__).parent))

from atlas_brain.services.stt.nemotron import NemotronSTT


async def record_and_test():
    """Record audio and test transcription."""
    print("=" * 70)
    print("🎤 Real Audio Test: Nemotron STT for 'ATLAS' Keyword Detection")
    print("=" * 70)
    
    # Initialize STT
    print("\n1️⃣  Loading Nemotron model...")
    stt = NemotronSTT()
    stt.load()
    
    print(f"✅ Model loaded on {stt.device}")
    
    # Record audio
    duration = 3  # seconds
    sample_rate = 16000
    
    print(f"\n2️⃣  Recording for {duration} seconds...")
    print("   🔴 Say 'ATLAS' clearly into your microphone now!")
    print("   3... 2... 1... GO!\n")
    
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished
    
    print("   ✅ Recording complete!\n")
    
    # Save to temp file
    temp_file = "test_atlas_recording.wav"
    sf.write(temp_file, audio, sample_rate)
    print(f"   📁 Saved to: {temp_file}")
    
    # Transcribe
    print("\n3️⃣  Transcribing with Nemotron...")
    
    with open(temp_file, "rb") as f:
        audio_bytes = f.read()
    
    transcription = await stt.transcribe(audio_bytes)
    
    # Results
    print("\n" + "=" * 70)
    print("📝 TRANSCRIPTION RESULT:")
    print("=" * 70)
    print(f"\n   {transcription!r}\n")
    
    # Check for keyword
    keyword = "atlas"
    transcription_lower = transcription.lower()
    
    if keyword in transcription_lower:
        print("✅ ✅ ✅  KEYWORD 'ATLAS' DETECTED! ✅ ✅ ✅")
    else:
        print(f"❌ Keyword '{keyword}' NOT detected")
        print("\n   Possible matches:")
        similar_words = ['at last', 'at las', 'at loss', 'at us', 'atlas']
        for word in similar_words:
            if word in transcription_lower:
                print(f"   - Found: '{word}'")
    
    print("\n" + "=" * 70)
    
    # Cleanup
    stt.unload()
    
    print("\n🔄 Want to test again? Run this script again!")
    print(f"   Audio saved to: {temp_file} (you can replay it)")


if __name__ == "__main__":
    try:
        asyncio.run(record_and_test())
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
