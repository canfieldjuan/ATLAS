#!/usr/bin/env python3
"""
Test Nemotron Speech Streaming model for Atlas STT.
NVIDIA Nemotron 0.6B - Fast, streaming ASR model.
"""
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
from pathlib import Path

def test_nemotron_stt():
    """Test Nemotron speech model for keyword detection."""
    
    # Model path
    model_path = Path.home() / ".cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots"
    
    # Find the actual snapshot directory
    snapshot_dirs = list(model_path.glob("*/"))
    if not snapshot_dirs:
        print("❌ No snapshot found in model directory")
        return
    
    nemo_file = snapshot_dirs[0] / "nemotron-speech-streaming-en-0.6b.nemo"
    
    if not nemo_file.exists():
        print(f"❌ Model file not found: {nemo_file}")
        return
    
    print(f"✅ Found model: {nemo_file}")
    print(f"📦 Model size: {nemo_file.stat().st_size / (1024**3):.2f} GB")
    
    # Load model
    print("\n🔄 Loading Nemotron model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
        
        # Load the ASR model
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(nemo_file), map_location=device)
        asr_model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Print model info
        print(f"\n📊 Model Info:")
        print(f"   Encoder: {asr_model.encoder.__class__.__name__}")
        print(f"   Vocab size: {asr_model.decoder.vocab_size}")
        print(f"   Sample rate: {asr_model.cfg.sample_rate if hasattr(asr_model.cfg, 'sample_rate') else 16000} Hz")
        
        # Test transcription with a sample keyword phrase
        print("\n🎤 Testing keyword transcription:")
        test_phrases = [
            "Hey Atlas, what's the weather?",
            "Atlas, turn on the lights",
            "Computer Atlas, play some music"
        ]
        
        print("\n   To fully test, you'd need audio files. This model expects:")
        print(f"   - Sample rate: 16000 Hz")
        print(f"   - Format: WAV, mono")
        print(f"   - Input: audio file path or numpy array")
        print("\n   Example usage:")
        print("   >>> transcription = asr_model.transcribe(['path/to/audio.wav'])")
        print("   >>> print(transcription)")
        
        # Check VRAM usage
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"\n💾 GPU Memory:")
            print(f"   Allocated: {allocated:.2f} GB")
        
        return asr_model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Testing NVIDIA Nemotron Speech Streaming for Atlas")
    print("=" * 60)
    
    model = test_nemotron_stt()
    
    if model:
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Model is compatible!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Create a NeMo STT service for Atlas")
        print("2. Test with real audio (say 'Atlas' into mic)")
        print("3. Compare accuracy vs faster-whisper medium.en")
        print("4. Benchmark speed and VRAM usage")
