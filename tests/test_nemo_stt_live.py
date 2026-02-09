#!/usr/bin/env python3
"""
Test Nemotron STT with a real audio recording.
Record yourself saying "Atlas" and test keyword detection.
"""
import asyncio
import sys
from pathlib import Path

# Add Atlas to path
sys.path.insert(0, str(Path(__file__).parent))

from atlas_brain.services.stt.nemotron import NemotronSTT


async def test_nemo_stt():
    """Test NeMo STT service."""
    print("=" * 60)
    print("Testing NeMo ASR STT for Atlas")
    print("=" * 60)
    
    # Initialize service
    print("\n1️⃣  Initializing NeMo ASR service...")
    stt = NemotronSTT()
    
    # Load model
    print("2️⃣  Loading model (this may take a moment)...")
    stt.load()
    
    print(f"\n✅ Model loaded!")
    print(f"   Device: {stt.device}")
    print(f"   Model: {stt.model_info.model_id}")
    
    # Test transcription with a sample file if available
    print("\n3️⃣  To test keyword detection:")
    print("   Option A: Provide a test audio file:")
    print("   >>> python -c \"import soundfile as sf; import sounddevice as sd; import numpy as np; audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1); sd.wait(); sf.write('test_atlas.wav', audio, 16000)\"")
    print("   Then say 'ATLAS' clearly into your microphone")
    print()
    print("   Option B: Use existing audio file:")
    
    test_files = list(Path(".").glob("*.wav")) + list(Path(".").glob("*.mp3"))
    if test_files:
        print(f"\n   Found {len(test_files)} audio files:")
        for f in test_files[:5]:
            print(f"   - {f}")
        
        # Test with first file
        test_file = test_files[0]
        print(f"\n4️⃣  Testing with: {test_file}")
        
        with open(test_file, "rb") as f:
            audio_bytes = f.read()
        
        print("   Transcribing...")
        transcription = await stt.transcribe(audio_bytes)
        
        print(f"\n📝 Transcription: {transcription!r}")
        
        # Check for keyword
        keyword = "atlas"
        if keyword in transcription.lower():
            print(f"✅ Keyword '{keyword}' DETECTED!")
        else:
            print(f"❌ Keyword '{keyword}' NOT detected")
            print(f"   Got: {transcription}")
    else:
        print("   No audio files found in current directory")
    
    # Cleanup
    print("\n5️⃣  Unloading model...")
    stt.unload()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)
    print("\nTo use in Atlas:")
    print("1. Set ATLAS_STT__DEFAULT_MODEL=nemo-asr in .env")
    print("2. Restart Atlas")
    print("3. Say 'Atlas' to trigger keyword detection")


if __name__ == "__main__":
    asyncio.run(test_nemo_stt())
