#!/usr/bin/env python3
"""
Test script for Nemotron Speech Streaming ASR.

Tests:
1. Model loading
2. Streaming transcription from microphone
3. Punctuation output verification
4. Latency measurement

Press Ctrl+C to stop.
"""

import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd

# Audio settings
SAMPLE_RATE = 16000
CHUNK_MS = 560  # 560ms chunks (good balance of latency/accuracy)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


def load_model():
    """Load Nemotron ASR model."""
    print("Loading Nemotron model...")
    start = time.time()

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model.eval()

    # Move to GPU if available
    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on CUDA in {time.time() - start:.2f}s")
    else:
        print(f"Model loaded on CPU in {time.time() - start:.2f}s")

    return model


def create_streaming_config(model):
    """Create cache-aware streaming config."""
    from omegaconf import OmegaConf

    # Streaming config for 560ms chunks
    # att_context_size = [left_context, right_context] in 80ms frames
    # [70, 6] = 70 left + 6 right = 560ms chunk
    streaming_cfg = OmegaConf.create({
        "att_context_size": [70, 6],  # 560ms chunks
        "shift_size": 7,  # Process in 7-frame shifts
    })

    return streaming_cfg


class AudioBuffer:
    """Thread-safe audio buffer for streaming."""

    def __init__(self, chunk_samples):
        self.chunk_samples = chunk_samples
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        self.chunk_queue = queue.Queue()

    def add_audio(self, audio_data):
        """Add audio data to buffer."""
        with self.lock:
            self.buffer = np.concatenate([self.buffer, audio_data.flatten()])

            # Extract complete chunks
            while len(self.buffer) >= self.chunk_samples:
                chunk = self.buffer[:self.chunk_samples]
                self.buffer = self.buffer[self.chunk_samples:]
                self.chunk_queue.put(chunk)

    def get_chunk(self, timeout=0.1):
        """Get next chunk if available."""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def run_streaming_test(model):
    """Run streaming transcription test."""
    print("\n" + "=" * 60)
    print("STREAMING TEST")
    print("=" * 60)
    print(f"Chunk size: {CHUNK_MS}ms")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print("Watch for punctuation (. ? !) in the output.")
    print("=" * 60 + "\n")

    audio_buffer = AudioBuffer(CHUNK_SAMPLES)
    running = True

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        audio_buffer.add_audio(indata)

    # For simpler testing, use batch transcription on chunks
    # Full streaming requires cache management which is complex
    print("Starting microphone capture...\n")

    accumulated_audio = []
    last_transcript = ""

    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype=np.float32,
            blocksize=int(SAMPLE_RATE * 0.05),  # 50ms blocks
        ):
            while running:
                chunk = audio_buffer.get_chunk(timeout=0.1)
                if chunk is not None:
                    accumulated_audio.append(chunk)

                    # Transcribe accumulated audio every ~1 second
                    if len(accumulated_audio) >= 2:  # ~1.12 seconds
                        audio_array = np.concatenate(accumulated_audio)

                        # Transcribe
                        start_time = time.time()

                        import torch
                        with torch.no_grad():
                            # Convert to proper format
                            audio_tensor = torch.tensor(audio_array).unsqueeze(0)
                            audio_len = torch.tensor([len(audio_array)])

                            if torch.cuda.is_available():
                                audio_tensor = audio_tensor.cuda()
                                audio_len = audio_len.cuda()

                            # Transcribe
                            transcripts = model.transcribe(
                                [audio_array],
                                batch_size=1,
                            )

                        latency = (time.time() - start_time) * 1000

                        if transcripts and transcripts[0]:
                            # Handle both string and Hypothesis object
                            result = transcripts[0]
                            if hasattr(result, 'text'):
                                transcript = result.text
                            elif hasattr(result, 'y_sequence'):
                                transcript = str(result)
                            else:
                                transcript = str(result)
                            if transcript != last_transcript:
                                # Check for punctuation
                                has_period = "." in transcript
                                has_question = "?" in transcript
                                has_exclaim = "!" in transcript
                                punct_status = ""
                                if has_period:
                                    punct_status += " [.]"
                                if has_question:
                                    punct_status += " [?]"
                                if has_exclaim:
                                    punct_status += " [!]"

                                print(f"[{latency:5.0f}ms] {transcript}{punct_status}")
                                last_transcript = transcript

                                # If we see end punctuation, clear for next utterance
                                if transcript.rstrip().endswith((".", "?", "!")):
                                    print("         ^ END OF UTTERANCE DETECTED")
                                    accumulated_audio = []
                                    last_transcript = ""

                        # Keep last chunk for context overlap
                        if len(accumulated_audio) > 4:
                            accumulated_audio = accumulated_audio[-2:]

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        running = False


def test_punctuation_samples(model):
    """Test punctuation on sample phrases."""
    print("\n" + "=" * 60)
    print("PUNCTUATION TEST (from text prompts)")
    print("=" * 60)
    print("Testing if model outputs punctuation correctly...")
    print("(This simulates what you'd say)\n")

    # We can't generate audio from text, but we can note
    # what to look for when testing live
    test_phrases = [
        "Atlas turn on the TV",
        "What time is it",
        "Dim the lights to fifty percent",
        "Turn off the lights",
        "Hey Atlas",
    ]

    print("When you say these phrases, watch for punctuation:")
    for phrase in test_phrases:
        print(f"  '{phrase}' -> should have period or question mark")

    print("\nNow speak these phrases into the mic to test...")


def main():
    print("=" * 60)
    print("NEMOTRON SPEECH STREAMING TEST")
    print("Model: nvidia/nemotron-speech-streaming-en-0.6b")
    print("=" * 60)

    # Load model
    model = load_model()

    # Show test phrases
    test_punctuation_samples(model)

    # Run streaming test
    run_streaming_test(model)


if __name__ == "__main__":
    main()
