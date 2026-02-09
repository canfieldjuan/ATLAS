
import asyncio
import os
import sys
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VoiceDiagnostic")

async def test_pipeline():
    print("="*50)
    print("DIAGNOSTIC: ATLAS VOICE PIPELINE")
    print("="*50)

    # 1. Check Environment & Imports
    print("\n[1/4] Checking Imports...")
    try:
        import torch
        print(f"✅ Torch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"❌ Torch Import Failed: {e}")
        return

    try:
        import nemo.collections.asr as nemo_asr
        print(f"✅ NeMo ASR: Imported successfully")
    except ImportError as e:
        print(f"❌ NeMo ASR Import Failed: {e}")
        print("   -> Did you run 'pip install -r requirements.txt'?")
        return

    try:
        from atlas_brain.services.tts.kokoro import KokoroTTS
        print(f"✅ KokoroTTS Class: Found")
    except ImportError as e:
        print(f"❌ KokoroTTS Import Failed: {e}")

    try:
        from atlas_brain.services.stt.nemotron import NemotronSTT
        print(f"✅ NemotronSTT Class: Found")
    except ImportError as e:
        print(f"❌ NemotronSTT Import Failed: {e}")
        return

    # 2. Test STT Loading & Inference
    print("\n[2/4] Testing STT (Nemotron)...")
    stt = NemotronSTT()
    try:
        start_time = time.time()
        print("   -> Loading Nemotron (this may take a moment)...")
        stt.load()
        load_time = time.time() - start_time
        print(f"✅ Nemotron Loaded in {load_time:.2f}s")
        
        # Test file inference
        test_file = Path("test_atlas.wav")
        if test_file.exists():
            print(f"   -> Testing inference on {test_file}...")
            with open(test_file, "rb") as f:
                audio_data = f.read()
            
            # Simple transcribe call
            try:
                text = await stt.transcribe(audio_data)
                print(f"✅ STT Result: '{text}'")
            except Exception as e:
                print(f"❌ STT Inference Failed: {e}")
        else:
            print("⚠️ Skipping file test (test_atlas.wav not found)")
            
    except Exception as e:
        print(f"❌ Nemotron Load/Run Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stt.unload()

    # 3. Test TTS Loading & Inference
    print("\n[3/4] Testing TTS (Kokoro)...")
    try:
        tts = KokoroTTS()
        start_time = time.time()
        print("   -> Loading Kokoro...")
        await tts.load() # Expecting async load based on pattern, check implementation if fails
        print(f"✅ Kokoro Loaded")
        
        print("   -> Generating test audio...")
        output_path = Path("diagnostic_tts_output.wav")
        try:
            # Assume synthesize method exists and returns path or bytes
            # Checking signature might be needed, but let's try standard 'synthesize'
            await tts.synthesize("System diagnostic complete.", output_file=output_path)
            
            if output_path.exists() and output_path.stat().st_size > 0:
                 print(f"✅ TTS Generation Successful: {output_path} ({output_path.stat().st_size} bytes)")
            else:
                 print(f"❌ TTS Generation Failed: Output file missing or empty")
                 
        except Exception as e:
             print(f"❌ TTS Generation Failed: {e}")
             
    except Exception as e:
        print(f"❌ Kokoro Load/Run Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("DIAGNOSTIC COMPLETE")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(test_pipeline())
