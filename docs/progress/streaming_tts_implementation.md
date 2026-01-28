# Streaming TTS Implementation - Progress Log

**Created:** 2026-01-28
**Last Updated:** 2026-01-28
**Status:** Planning

---

## Overview

Implement true streaming TTS with Piper to reduce time-to-first-audio. Currently, TTS waits for full synthesis before playback begins. With streaming, audio plays as it's generated.

### Current Flow (Batch)
```
Text → Piper (full synthesis to file) → Read file → Play chunks
       [~200ms for short text]         [blocking]   [chunked]

Time to first audio: synthesis_time + file_read_time
```

### Target Flow (Streaming)
```
Text → Piper --output-raw → stdout pipe → Play chunks as received
       [generates incrementally]         [immediate playback]

Time to first audio: ~50-100ms (first chunk ready)
```

### Expected Improvement
- Short responses (1-2 sentences): 100-200ms faster
- Long responses (paragraphs): 500ms+ faster
- Better perceived responsiveness

---

## Technical Analysis

### Piper Capabilities Verified
```bash
piper --output-raw  # Streams raw int16 PCM to stdout
```
- Output format: int16 PCM, little-endian
- Sample rate: 16000 Hz (from model config)
- Channels: mono

### Files to Modify

| File | Change | Risk |
|------|--------|------|
| `atlas_brain/voice/pipeline.py` | Modify `PiperTTS.speak()` to use streaming | Medium - core TTS |
| `atlas_brain/voice/playback.py` | No change needed - protocol compatible | None |
| `atlas_brain/voice/launcher.py` | Pass sample_rate to PiperTTS | Low |
| `atlas_brain/config.py` | Add optional `piper_sample_rate` field | Low |

### Dependencies
- `sounddevice` - already used for playback
- `subprocess.Popen` - for pipe-based streaming (stdlib)

### Backward Compatibility
- Keep file-based synthesis as fallback if streaming fails
- Same `SpeechEngine` protocol interface maintained
- No changes to `PlaybackController` needed

---

## Implementation Plan

### Phase 1: Streaming PiperTTS (Core Change)

**Goal:** Modify `PiperTTS.speak()` to stream audio from Piper stdout

**Changes to `atlas_brain/voice/pipeline.py`:**

1. Add `sample_rate` parameter to `PiperTTS.__init__`
2. Replace `subprocess.run()` with `subprocess.Popen()`
3. Read stdout in chunks (e.g., 4096 bytes = 2048 samples = 128ms)
4. Play each chunk immediately via sounddevice
5. Support stop/interrupt mid-stream

**Pseudocode:**
```python
def speak(self, text: str):
    cmd = [self.binary_path, "--model", self.model_path, "--output-raw", ...]
    process = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    process.stdin.write(text.encode())
    process.stdin.close()

    with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
        while not self.stop_event.is_set():
            chunk = process.stdout.read(4096)  # 2048 samples
            if not chunk:
                break
            audio = np.frombuffer(chunk, dtype=np.int16)
            stream.write(audio)
```

### Phase 2: Config & Launcher Wiring

**Goal:** Wire sample rate from config to TTS

**Changes:**
1. `config.py`: Add `piper_sample_rate: int = 16000`
2. `launcher.py`: Pass `sample_rate=cfg.piper_sample_rate` to PiperTTS

### Phase 3: Error Handling & Fallback

**Goal:** Robust error handling with fallback to file-based synthesis

**Changes:**
1. Catch streaming errors (broken pipe, process crash)
2. Log warnings and fall back to current file-based method
3. Add `streaming_enabled` config flag for easy toggle

### Phase 4: Testing & Validation

**Tests:**
1. Short text (~5 words): Verify faster first-audio
2. Long text (~100 words): Verify streaming works throughout
3. Interrupt mid-speech: Verify clean stop
4. Error conditions: Verify fallback works

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Piper process hangs | Timeout on stdout read, kill process |
| Audio buffer underrun | Larger initial buffer, handle gracefully |
| Different model sample rates | Read from model config JSON |
| Breaking existing functionality | Keep fallback, comprehensive testing |

---

## Session Log

### 2026-01-28 Session 1
- Analyzed current PiperTTS implementation (batch synthesis)
- Verified Piper `--output-raw` outputs int16 PCM at 16kHz
- Identified files to modify
- Created implementation plan
- **Status:** Awaiting approval to proceed

### 2026-01-28 Session 2 - Phase 1 Implementation
- Added `piper_sample_rate` config field (default 16000)
- Updated `PiperTTS.__init__` to accept `sample_rate` parameter
- Implemented `_speak_streaming()` using Popen + stdout pipe
- Kept `_speak_batch()` as fallback on streaming failure
- Updated `stop()` to terminate Popen process
- Updated `launcher.py` to pass `sample_rate` to PiperTTS
- Verified compilation of all components
- Tested streaming TTS in isolation - works correctly
- **Status:** Ready for full pipeline testing

---

## Next Steps (After Approval)
1. Implement Phase 1: Streaming PiperTTS.speak()
2. Test with short/long text
3. Implement Phase 2: Config wiring
4. Implement Phase 3: Error handling
5. Final validation
